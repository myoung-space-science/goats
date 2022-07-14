"""Tools for managing datasets."""

import typing

from goats.core import aliased
from goats.core import datafile
from goats.core import datatypes
from goats.core import iotools
from goats.core import observables
from goats.core import metric


S = typing.TypeVar('S', bound='Variables')


class Variables(aliased.Mapping):
    """An interface to dataset variables.
    
    This class provides aliased key-based access to all variables in a dataset.
    It converts each requested dataset variable into a `~datatypes.Variable`
    instance with the appropriate MKS unit.
    """

    def __init__(self, dataset: datafile.Interface) -> None:
        known = {
            k: v for k, v in dataset.variables.items(aliased=True)
            if k in observables.METADATA
        }
        super().__init__(known)
        self._system = metric.System('mks')
        self._units = None
        self._cache = {}

    @property
    def units(self):
        """The MKS unit of each variable."""
        if self._units is None:
            units = {
                name: self._get_unit(name)
                for name in self.keys(aliased=True)
            }
            self._units = aliased.Mapping(units)
        return self._units

    def _get_unit(self, name: str):
        """Get a standard unit for the named variable."""
        variable = self._system[observables.METADATA[name]['quantity']]
        return metric.Unit(variable.unit)

    def __getitem__(self, key: str):
        """Create the named variable, if possible."""
        if key in self._cache:
            return self._cache[key]
        datavar = super().__getitem__(key)
        variable = datatypes.Variable(
            datavar.data,
            unit=standardize(datavar.unit),
            axes=datavar.axes,
            name=observables.ALIASES[key],
        )
        result = variable.convert(self.units[key])
        self._cache[key] = result
        return result


class Indexers(aliased.Mapping):
    """The default collection of axis indexers."""

    def __init__(self, dataset: datafile.Interface) -> None:
        self.dataset = dataset
        self.references = {}
        super().__init__(dataset.axes)

    def __getitem__(self, key: str) -> datatypes.Indexer:
        """Get the default indexer for `key`."""
        axis = super().__getitem__(key)
        reference = range(axis.size)
        method = lambda _: datatypes.Indices(reference)
        return datatypes.Indexer(method, reference)


class Axes(aliased.Mapping):
    """An interface to dataset axes."""

    def __init__(
        self,
        dataset: datafile.Interface,
        factory: typing.Type[Indexers],
    ) -> None:
        indexers = factory(dataset)
        super().__init__(indexers)
        self.dataset = dataset

    def __getitem__(self, key: str) -> datatypes.Axis:
        indexer = super().__getitem__(key)
        size = self.dataset.axes[key].size
        names = observables.ALIASES.get(key, [key])
        return datatypes.Axis(size, indexer, *names)


class Interface:
    """The user interface to a dataset."""

    def __init__(
        self,
        path: iotools.PathLike,
        indexers: typing.Type[Indexers]=None,
    ) -> None:
        self.path = path
        self.view = datafile.Interface(path)
        self.indexers = indexers or Indexers
        self._variables = None
        self._axes = None

    @property
    def variables(self):
        """Objects representing the variables in this dataset."""
        if self._variables is None:
            self._variables = Variables(self.view)
        return self._variables

    @property
    def axes(self):
        """Objects representing the axes in this dataset."""
        if self._axes is None:
            self._axes = Axes(self.view, self.indexers)
        return self._axes

    Default = typing.TypeVar('Default')

    def iter_axes(self, name: str, default: Default=None):
        """Iterate over the axes for the named variable."""
        if name in self.variables:
            return iter(self.variables[name].axes)
        if default is not None:
            return default
        raise ValueError(
            f"Can't iterate over axes of missing variable {name!r}"
        ) from None

    def resolve_axes(
        self,
        names: typing.Iterable[str],
        mode: str='strict',
    ) -> typing.Tuple[str]:
        """Compute and order the available axes in `names`."""
        axes = self.view.available('axes').canonical
        ordered = tuple(name for name in axes if name in names)
        if mode == 'strict':
            return ordered
        extra = tuple(name for name in names if name not in ordered)
        if not extra:
            return ordered
        if mode == 'append':
            return ordered + extra
        raise ValueError(f"Unrecognized mode {mode!r}")

    def get_indices(self, name: str, **user):
        """Extract indexing objects for the named variable."""
        variable = self.variables.get(name)
        if not variable:
            return ()
        return tuple(
            self.axes[axis](*user.get(axis, ()))
            for axis in variable.axes
        )


substitutions = {
    'julian date': 'day',
    'shell': '1',
    'cos(mu)': '1',
    'e-': 'e',
    '# / cm^2 s sr MeV': '# / cm^2 s sr MeV/nuc',
}
"""Conversions from non-standard units."""

T = typing.TypeVar('T')

def standardize(unit: T):
    """Replace this unit string with a standard unit string, if possible.

    This function looks for `unit` in the known conversions and returns the
    standard unit string if it exists. If this doesn't find a standard unit
    string, it just returns the input.
    """
    unit = substitutions.get(str(unit), unit)
    if '/' in unit:
        num, den = str(unit).split('/', maxsplit=1)
        unit = ' / '.join((num.strip(), f"({den.strip()})"))
    return unit


