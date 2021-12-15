from pathlib import Path
from typing import *

import numpy as np

from goats.common import datasets
from goats.common import quantities
from goats.common import indexing
from goats.common import iterables
from goats.common import iotools
from goats.common import elements
from goats.eprem import parameters


_variables = {
    'time': {
        'aliases': ('t', 'times'),
        'unit': 's',
    },
    'shell': {
        'aliases': ('shells',),
        'unit': '1',
    },
    'mu': {
        'aliases': (
            'mus',
            'pitch angle', 'pitch-angle cosine',
            'pitch angles', 'pitch-angle cosines',
        ),
        'unit': '1',
    },
    'mass': {
        'aliases': ('m',),
        'unit': 'g',
    },
    'charge': {
        'aliases': ('q',),
        'unit': 'statcoul',
    },
    'egrid': {
        'aliases': ('energy', 'energies', 'E'),
        'unit': 'erg',
    },
    'vgrid': {
        'aliases': ('speed', 'v', 'vparticle'),
        'unit': 'cm/s',
    },
    'R': {
        'aliases': ('r', 'radius'),
        'unit': 'cm',
    },
    'T': {
        'aliases': ('theta',),
        'unit': 'radian',
    },
    'P': {
        'aliases': ('phi',),
        'unit': 'radian',
    },
    'Br': {
        'aliases': ('br',),
        'unit': 'G',
    },
    'Bt': {
        'aliases': ('bt', 'Btheta', 'btheta'),
        'unit': 'G',
    },
    'Bp': {
        'aliases': ('bp', 'Bphi', 'bphi'),
        'unit': 'G',
    },
    'Vr': {
        'aliases': ('vr',),
        'unit': 'cm/s',
    },
    'Vt': {
        'aliases': ('vt', 'Vtheta', 'vtheta'),
        'unit': 'cm/s',
    },
    'Vp': {
        'aliases': ('vp', 'Vphi', 'vphi'),
        'unit': 'cm/s',
    },
    'Rho': {
        'aliases': ('rho',),
        'unit': '1/cm^3',
    },
    'Dist': {
        'aliases': ('dist', 'f'),
        'unit': 's^3/cm^6',
    },
    'flux': {
        'aliases': ('J', 'J(E)', 'Flux', 'j', 'j(E)'),
        'quantity': 'particle flux',
        'unit': '# / (m^2 * sr * s * (MeV/nuc))',
    },
}

_axes = {
    'time': _variables['time'].copy(),
    'shell': _variables['shell'].copy(),
    'species': {},
    'energy': _variables['egrid'].copy(),
    'mu': _variables['mu'].copy(),
}

_metadata = {
    'variables': _variables,
    'axes': _axes,
}


metadata = {k: iterables.AliasedMapping.of(v) for k, v in _metadata.items()}
"""Metadata for axes and variables."""


class DatasetView(datasets.DatasetView, iterables.ReprStrMixin):
    """An EPREM dataset."""

    _config_names = (
        'eprem_input_file',
    )

    def __init__(
        self,
        path: Union[str, Path],
        system: str,
    ) -> None:
        super().__init__(path)
        self.system = quantities.MetricSystem(system)
        self._configpath = None
        self._config = None

    @property
    def configpath(self):
        """The current configuration file path."""
        if self._configpath is None:
            for name in self._config_names:
                current = self.path.parent / name
                if current.exists():
                    self._configpath = iotools.ReadOnlyPath(current)
        return self._configpath

    @property
    def config(self):
        """An object representing the configuration file."""
        if self._config is None:
            self._config = parameters.ConfigManager(self.configpath)
        return self._config

    def with_config(
        self,
        new: Union[str, Path],
        local: bool=True,
        **kwargs
    ) -> 'DatasetView':
        """Associate a new configuration file with this dataset.
        
        Parameters
        ----------
        new : string or Path
            The path of the new configuration file. The path may be relative and
            contain wildcards, but it must exist.

        local : boolean, default=True
            If true (default), assume that `new` is relative to the directory
            containing the dataset. Otherwise, treat `new` as an independent
            absolute or relative path.

        **kwargs
            Any keywords to be passed to `parameters.ConfigManager`
        """
        path = self.path.parent / new if local else new
        self._configpath = iotools.ReadOnlyPath(path)
        self._config = parameters.ConfigManager(self._configpath, **kwargs)
        return self

    def __str__(self) -> str:
        """A simplified representation of this object."""
        return f"path={self.path}, system={self.system}"


unit_conversions = {
    'julian date': 'day',
    'shell': '1',
    'cos(mu)': '1',
    'e-': 'e',
}
"""Conversions from non-standard EPREM units."""


T = TypeVar('T')

def standardize(unit: T):
    """Replace this unit string with a standard unit string, if possible.

    This function looks for `unit` in the known conversions and returns the
    standard unit string if it exists. If this doesn't find a standard unit
    string, it just returns the input.
    """
    return unit_conversions.get(str(unit), unit)


class Variables(iterables.AliasedMapping):
    """An aliased mapping from variable name to `Variable` object."""

    def __init__(self, dataset: DatasetView) -> None:
        ukeys = ('unit', 'units')
        attrmap = {
            k: {
                'data': v,
                'axes': tuple(getattr(v, 'dimensions', ())),
                'unit': next(
                    (
                        standardize(getattr(v, a))
                        for a in ukeys if hasattr(v, a)
                    ), None
                )
            } for k, v in dataset.variables.items()
        }
        mapping = {
            tuple([k, *v.get('aliases', ())]): attrmap[k]
            for k, v in _variables.items()
            if k in dataset.variables
        }
        super().__init__(mapping=mapping)
        self.canonical = tuple(dataset.variables.keys())
        """The names of variables within the dataset."""
        self.system = dataset.system
        """The metric system of the underlying dataset."""

    def __getitem__(self, key: str):
        """Aliased access to dataset variables."""
        if key in self:
            stored = super().__getitem__(key)
            data = stored['data']
            unit = stored['unit']
            axes = stored['axes']
            variable = quantities.Variable(data, unit, axes)
            return variable.to(self.system.get_unit(unit))
        raise KeyError(f"No variable corresponding to '{key!r}'") from None

    def __str__(self) -> str:
        """A simplified representation of this object."""
        return ', '.join(
            f"{k}: unit={v['unit']!r}, axes={v['axes']}"
            for k, v in self.items().aliased
        )


class Time(indexing.Measured):
    """EPREM time indexer."""


class Shell(indexing.Trivial):
    """EPREM shell indexer."""


class Species(indexing.Mapped):
    """EPREM species indexer."""


class Energy(indexing.Measured):
    """EPREM energy indexer."""

    # FIXME: This only works for a single species!
    def index(self, *targets):
        if isinstance(targets[0], np.ndarray):
            targets = targets[0]
        return super().index(*targets)


class Mu(indexing.Measured):
    """EPREM pitch-angle cosine indexer."""


class IndexerFactory(iterables.MappingBase):
    """A mapping of EPREM axis names to indexing objects."""

    def __init__(self, dataset: datasets.DatasetView) -> None:
        self.variables = Variables(dataset)
        self._indexers = None
        self._arrays = None
        self.names = tuple(self.indexers.keys())
        """The names of available indexing objects."""
        super().__init__(self.names)
        self._sizes = dataset.sizes

    def __getitem__(self, name: str):
        """Create the indexer corresponding to the named axis."""
        if name in self:
            indexer = self.indexers[name]
            reference = self.arrays[name]
            size = self._sizes[name]
            return indexer(reference, size=size)
        raise KeyError(f"No indexer available for '{name}'") from None

    @property
    def indexers(self):
        """The indexer (class) of each index."""
        if self._indexers is None:
            self._indexers = {
                'time': Time,
                'shell': Shell,
                'species': Species,
                'energy': Energy,
                'mu': Mu,
            }
        return self._indexers

    @property
    def arrays(self) -> Dict[str, quantities.Variable]:
        """The array of dataset values for each index."""
        if self._arrays is None:
            mass = self.variables['mass'].to('nuc')
            charge = self.variables['charge'].to('e')
            self._arrays = {
                'time': self.variables['time'],
                'shell': np.array(self.variables['shell'], dtype=int),
                'species': elements.symbols(mass, charge),
                'energy': self.variables['egrid'],
                'mu': self.variables['mu'],
            }
        return self._arrays


class Axes(iterables.AliasedMapping):
    """An aliased mapping from axis name to `Axis` object."""

    def __init__(self, dataset: DatasetView) -> None:
        self.indexers = IndexerFactory(dataset)
        mapping = {
            tuple([k, *v.get('aliases', ())]): self.indexers[k]
            for k, v in _axes.items()
        }
        super().__init__(mapping=mapping)
        self.canonical = tuple(dataset.axes)
        """The names of axes within the dataset."""

    def __getitem__(self, key: str):
        """Aliased access to dataset axes."""
        if key in self:
            indexer = super().__getitem__(key)
            return indexing.Axis(indexer)
        raise KeyError(f"No axis corresponding to '{key!r}'") from None

