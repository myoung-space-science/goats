from pathlib import Path
import collections.abc
import numbers
import typing

import numpy as np

from goats.common import aliased
from goats.common import datasets
from goats.common import quantities
from goats.common import indexing
from goats.common import iterables
from goats.common import physical
from goats.common import numerical


class Time(indexing.IndexComputer):
    """EPREM time indexer."""


class Shell(indexing.Indexer):
    """EPREM shell indexer."""


class Species(indexing.IndexMapper):
    """EPREM species indexer."""


class Energy(indexing.IndexComputer):
    """EPREM energy indexer."""

    def __init__(
        self,
        reference: quantities.Measured,
        species_map: Species,
        size: int=None,
    ) -> None:
        super().__init__(reference, size=size)
        self.map = species_map

    def __call__(self, *user, species: typing.Union[str, int]=0):
        targets = self._normalize(*user)
        if all(isinstance(value, numbers.Integral) for value in targets):
            return indexing.Indices(targets)
        s = self.map(species)[0]
        if targets == self.reference:
            targets = targets[s, :]
        measured = quantities.measure(*targets)
        vector = quantities.Vector(measured.values, measured.unit)
        values = (
            vector.unit(self.unit)
            if vector.unit().dimension == self.unit.dimension
            else vector
        )
        reference = self.reference[s, :]
        indices = [
            numerical.find_nearest(reference, float(value)).index
            for value in values
        ]
        return indexing.Coordinates(indices, values, self.unit)


class Mu(indexing.IndexComputer):
    """EPREM pitch-angle cosine indexer."""


class VariablesView(aliased.Mapping):
    """A view of the variables in a dataset."""

    def __init__(
        self,
        dataset: datasets.DatasetView,
        system: str=None,
    ) -> None:
        meta = metadata['variables'].copy()
        self._attrs = {
            name: {
                'data': data,
                'axes': self._get_axes_from_data(data),
                'unit': self._get_unit_from_data(data),
                'name': meta.alias(name, include=True),
            } for name, data in dataset.variables.items()
        }
        create = quantities.Variable
        variables = {
            current['name']: create(**current)
            for name, current in self._attrs.items()
            if name in _VARIABLES
        }
        super().__init__(variables)
        self._system = quantities.MetricSystem(system) if system else None

    def _get_axes_from_data(self, data):
        """Compute appropriate variable axes from a dataset object."""
        return tuple(getattr(data, 'dimensions', ()))

    _ukeys = ('unit', 'units')

    def _get_unit_from_data(self, data):
        """Compute appropriate variable units from a dataset object."""
        available = (
            standardize(getattr(data, attr))
            for attr in self._ukeys if hasattr(data, attr)
        )
        return next(available, None)

    def __getitem__(self, key: str):
        """Get a variable by name or alias."""
        if key in self._flat_keys():
            variable = super().__getitem__(key)
            if not self._system:
                return variable
            unit = self._get_unit(key)
            return variable.unit(unit)
        raise KeyError(f"No variable corresponding to {key!r}") from None

    def _get_unit(self, key: str):
        """Get a variable's unit in the current metric system."""
        if variable := metadata['variables'].get(key):
            return self._system.get_unit(quantity=variable['quantity'])


class AxesView(aliased.Mapping):
    """A view of the axis-indexing objects for a dataset."""

    def __init__(
        self,
        dataset: datasets.DatasetView,
        system: quantities.MetricSystem=None,
    ) -> None:
        variables = VariablesView(dataset, system)
        mass = variables['mass'].unit('nuc')
        charge = variables['charge'].unit('e')
        sizes = dataset.sizes
        arrays = {
            'time': variables['time'],
            'shell': np.array(variables['shell'], dtype=int),
            'species': physical.elements(mass, charge),
            'energy': variables['energy'],
            'mu': variables['mu'],
        }
        args = {
            name: {
                'reference': array,
                'size': sizes[name],
            } for name, array in arrays.items()
        }
        args['energy']['species_map'] = Species(args['species'])
        base = {
            name: {
                'indexer': _AXES[name]['indexer'](**args[name]),
            }
            for name in arrays
            # NOTE: Using `arrays.keys()` instead of `_AXES.items()` ensures
            # that the items in the resultant dictionary are in the correct
            # order. This is necessary to allow downstream objects to assume
            # correctly ordered axes but it's a somewhat fragile solution.
        }
        meta = metadata['axes'].copy()
        axis = indexing.Axis
        axes = {
            meta.alias(name, include=True): axis(**attrs)
            for name, attrs in base.items()
            if name in _AXES
        }
        super().__init__(axes)


class SubsetKeys(typing.NamedTuple):
    """A subset of names of attributes."""

    full: typing.Tuple[str]
    aliased: typing.Tuple[aliased.MappingKey]
    canonical: typing.Tuple[str]


class Dataset(collections.abc.Container):
    """Interface to an EPREM dataset."""

    def __init__(
        self,
        path: typing.Union[str, Path],
        system: typing.Union[str, quantities.MetricSystem]=None,
    ) -> None:
        self._dataset = datasets.DatasetView(path)
        self._system = system
        self._variables = None
        self._axes = None
        self._canonical = {
            'variables': tuple(
                variable for variable in self.variables
                if variable in self._dataset.variables
            ),
            'axes': tuple(
                axis for axis in self.axes
                if axis in self._dataset.axes
            ),
        }

    def __contains__(self, key: str) -> bool:
        """True if `key` corresponds to an available variable or axis."""
        return key in self.variables or key in self.axes

    @property
    def variables(self):
        """The variables in this dataset."""
        if self._variables is None:
            self._variables = VariablesView(self._dataset, self._system)
        return self._variables

    @property
    def axes(self):
        """The axis-indexing objects for this dataset."""
        if self._axes is None:
            self._axes = AxesView(self._dataset, self._system)
        return self._axes

    def system(self, new: str=None):
        """Get or set the metric system for this dataset."""
        if not new:
            return self._system
        self._system = new
        self._variables = None
        self._axes = None
        return self

    def available(self, key: str):
        """Provide the names of available attributes."""
        if key in {'variable', 'variables'}:
            return SubsetKeys(
                full=tuple(self.variables),
                aliased=tuple(self.variables.keys(aliased=True)),
                canonical=self._canonical['variables'],
            )
        if key in {'axis', 'axes'}:
            return SubsetKeys(
                full=tuple(self.axes),
                aliased=tuple(self.axes.keys(aliased=True)),
                canonical=self._canonical['axes'],
            )

    def iter_axes(self, name: str):
        """Iterate over the axes for the named variable."""
        this = self.variables[name].axes if name in self.variables else ()
        return iter(this)

    def resolve_axes(self, names: typing.Iterable[str]):
        """Compute and order the available axes in `names`."""
        axes = self.available('axes').canonical
        return tuple(name for name in axes if name in names)


_VARIABLES = {
    'time': {
        'aliases': ['t', 'times'],
        'quantity': 'time',
    },
    'shell': {
        'aliases': ['shells'],
        'quantity': 'number',
    },
    'mu': {
        'aliases': [
            'mus',
            'pitch angle', 'pitch-angle cosine',
            'pitch angles', 'pitch-angle cosines',
        ],
        'quantity': 'ratio',
    },
    'mass': {
        'aliases': ['m'],
        'quantity': 'mass',
    },
    'charge': {
        'aliases': ['q'],
        'quantity': 'charge',
    },
    'egrid': {
        'aliases': ['energy', 'energies', 'E'],
        'quantity': 'energy',
    },
    'vgrid': {
        'aliases': ['speed', 'v', 'vparticle'],
        'quantity': 'velocity',
    },
    'R': {
        'aliases': ['r', 'radius'],
        'quantity': 'length',
    },
    'T': {
        'aliases': ['theta'],
        'quantity': 'plane angle',
    },
    'P': {
        'aliases': ['phi'],
        'quantity': 'plane angle',
    },
    'Br': {
        'aliases': ['br'],
        'quantity': 'magnetic field',
    },
    'Bt': {
        'aliases': ['bt', 'Btheta', 'btheta'],
        'quantity': 'magnetic field',
    },
    'Bp': {
        'aliases': ['bp', 'Bphi', 'bphi'],
        'quantity': 'magnetic field',
    },
    'Vr': {
        'aliases': ['vr'],
        'quantity': 'velocity',
    },
    'Vt': {
        'aliases': ['vt', 'Vtheta', 'vtheta'],
        'quantity': 'velocity',
    },
    'Vp': {
        'aliases': ['vp', 'Vphi', 'vphi'],
        'quantity': 'velocity',
    },
    'Rho': {
        'aliases': ['rho'],
        'quantity': 'number density',
    },
    'Dist': {
        'aliases': ['dist', 'f'],
        'quantity': 'particle distribution',
    },
    'flux': {
        'aliases': ['J', 'J(E)', 'Flux', 'j', 'j(E)'],
        'quantity': 'number / (area * solid_angle * time * energy / mass)',
    },
}

_AXES = {
    name: {**_VARIABLES[name], 'indexer': indexer}
    for name, indexer in {'time': Time, 'shell': Shell, 'mu': Mu}.items()
}
_AXES.update(
    energy={**_VARIABLES['egrid'], 'indexer': Energy},
    species={
        'aliases': [],
        'quantity': 'identity',
        'indexer': Species,
    }
)

_metadata = {
    'variables': _VARIABLES,
    'axes': _AXES,
}

metadata = {k: aliased.Mapping(v) for k, v in _metadata.items()}
"""Metadata for axes and variables."""


unit_conversions = {
    'julian date': 'day',
    'shell': '1',
    'cos(mu)': '1',
    'e-': 'e',
    '# / cm^2 s sr MeV': '# / cm^2 s sr MeV/nuc',
}
"""Conversions from non-standard EPREM units."""

T = typing.TypeVar('T')

def standardize(unit: T):
    """Replace this unit string with a standard unit string, if possible.

    This function looks for `unit` in the known conversions and returns the
    standard unit string if it exists. If this doesn't find a standard unit
    string, it just returns the input.
    """
    unit = unit_conversions.get(str(unit), unit)
    if '/' in unit:
        num, den = str(unit).split('/', maxsplit=1)
        unit = ' / '.join((num.strip(), f"({den.strip()})"))
    return unit


