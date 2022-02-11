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
        vector = quantities.measure(*targets).asvector
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
        dataset = datasets.DatasetView(path)
        self._system = quantities.MetricSystem(system) if system else None
        self._quantities = None
        self._units = None
        self._variables = None
        self._axes = None
        self._init_v = self._build_variables(dataset)
        self._init_a = self._build_axes(dataset)
        self._canonical = {
            'variables': tuple(
                variable for variable in self._init_v
                if variable in dataset.variables
            ),
            'axes': tuple(
                axis for axis in self._init_a
                if axis in dataset.axes
            ),
        }

    def __contains__(self, key: str) -> bool:
        """True if `key` corresponds to an available variable or axis."""
        return key in self.variables or key in self.axes

    @property
    def variables(self):
        """The variables in this dataset."""
        if self._variables is None:
            base = {
                key: variable.unit(self.units[key])
                for key, variable in self._init_v.items(aliased=True)
            } if self._system else self._init_v
            self._variables = aliased.Mapping(base)
        return self._variables

    @property
    def units(self):
        """The unit of each dataset attribute in the current system."""
        if self._units is None:
            base = {
                key: self._system.get_unit(quantity=quantity)
                for key, quantity in self.quantities.items(aliased=True)
            } if self._system else {
                key: variable.unit()
                for key, variable in self._init_v
            }
            self._units = aliased.Mapping(base)
        return self._units

    @property
    def quantities(self):
        """The physical quantities of each dataset attribute."""
        if self._quantities is None:
            base = {
                key: variable['quantity']
                for key, variable in metadata['variables'].items(aliased=True)
                if key in self._init_v.keys(aliased=True)
            }
            self._quantities = aliased.Mapping(base)
        return self._quantities

    @property
    def axes(self):
        """The axes in this dataset."""
        if self._axes is None:
            axes = self._init_a
            self._axes = aliased.Mapping(axes)
        return self._axes

    def system(self, new: str=None):
        """Get or set the metric system for this dataset."""
        if not new:
            return self._system
        self._system = quantities.MetricSystem(new)
        self._variables = None
        self._units = None
        return self

    def available(self, key: str):
        """Provide the names of available attributes."""
        if key in {'variable', 'variables'}:
            return SubsetKeys(
                full=tuple(self._init_v),
                aliased=tuple(self._init_v.keys(aliased=True)),
                canonical=self._canonical['variables'],
            )
        if key in {'axis', 'axes'}:
            return SubsetKeys(
                full=tuple(self._init_a),
                aliased=tuple(self._init_a.keys(aliased=True)),
                canonical=self._canonical['axes'],
            )

    def iter_axes(self, name: str):
        """Iterate over the axes for the named variable."""
        if name in self._init_v:
            variable = self._init_v[name]
            return iter(variable.axes)
        return iter(())

    def resolve_axes(self, names: typing.Iterable[str]):
        """Compute and order the available axes in `names`."""
        axes = self.available('axes').canonical
        return tuple(name for name in axes if name in names)

    def _build_variables(self, dataset: datasets.DatasetView):
        """Create a collection of variables from the dataset."""
        meta = metadata['variables'].copy()
        base = {
            name: {
                'data': data,
                'axes': self._get_axes_from_data(data),
                'unit': self._get_unit_from_data(data),
                'name': meta.alias(name, include=True),
            } for name, data in dataset.variables.items()
        }
        variable = quantities.Variable
        variables = {
            current['name']: variable(**current)
            for name, current in base.items()
            if name in _VARIABLES
        }
        return aliased.Mapping(variables)

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

    def _build_axes(self, dataset: datasets.DatasetView):
        """Create a collection of axes from the dataset."""
        mass = self.variables['mass'].unit('nuc')
        charge = self.variables['charge'].unit('e')
        sizes = dataset.sizes
        arrays = {
            'time': self.variables['time'],
            'shell': np.array(self.variables['shell'], dtype=int),
            'species': physical.elements(mass, charge),
            'energy': self.variables['energy'],
            'mu': self.variables['mu'],
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
        return aliased.Mapping(axes)


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
    '# / cm^2 s sr MeV': '# / (cm^2 s sr MeV)',
}
"""Conversions from non-standard EPREM units."""

T = typing.TypeVar('T')

def standardize(unit: T):
    """Replace this unit string with a standard unit string, if possible.

    This function looks for `unit` in the known conversions and returns the
    standard unit string if it exists. If this doesn't find a standard unit
    string, it just returns the input.
    """
    return unit_conversions.get(str(unit), unit)


