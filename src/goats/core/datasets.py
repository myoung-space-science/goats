"""Tools for managing datasets."""

import abc
import collections.abc
import numbers
import typing

import netCDF4
import numpy
import numpy.typing

from goats.core import datatypes
from goats.core import aliased
from goats.core import iotools
from goats.core import iterables
from goats.core import observables
from goats.core import quantities


class DataViewer(collections.abc.Mapping):
    """An abstract base class for data-viewing objects."""

    def __init__(self, path: iotools.ReadOnlyPath) -> None:
        self.members = self.get_members(path)

    def __iter__(self) -> typing.Iterator:
        return iter(self.members)

    def __len__(self) -> int:
        return len(self.members)

    @abc.abstractmethod
    def get_members(self, path: iotools.ReadOnlyPath) -> typing.Mapping:
        """Get the appropriate members for this viewer."""
        pass

    def __str__(self) -> str:
        """A simplified representation of this object."""
        return '\n\n'.join(f"{k}:\n{v!r}" for k, v in self.items())

    def __repr__(self) -> str:
        """An unambiguous representation of this object."""
        return f"\n::{self.__class__.__qualname__}::\n\n{self}"


class DatasetVariable(typing.NamedTuple):
    """A dataset variable."""

    data: numpy.typing.ArrayLike
    unit: str
    axes: typing.Tuple[str]
    name: str

    def __str__(self) -> str:
        """A simplified representation of this object."""
        return self._display(sep='\n')

    def __repr__(self) -> str:
        """An unambiguous representation of this object."""
        display = self._display(sep='\n', tab=4)
        return f"{self.__class__.__qualname__}(\n{display}\n)"

    def _display(
        self,
        sep: str=', ',
        tab: int=0,
    ) -> str:
        """Helper for `__str__` and `__repr__`."""
        attrs = [
            f"data={type(self.data)}",
            f"unit={self.unit!r}",
            f"axes={self.axes}",
            f"name={self.name!r}",
        ]
        indent = ' ' * tab
        return sep.join(f"{indent}{attr}" for attr in attrs)


class NetCDFVariables(DataViewer):
    """An object for viewing variables in a NetCDF dataset."""

    def get_members(self, path: iotools.ReadOnlyPath) -> typing.Mapping:
        dataset = netCDF4.Dataset(path, 'r')
        return dataset.variables

    def __getitem__(self, name: str) -> DatasetVariable:
        if name in self.members:
            data = self.members[name]
            unit = self._get_unit_from_data(data)
            axes = self._get_axes_from_data(data)
            return DatasetVariable(data, unit, axes, name)
        raise KeyError(f"No variable called '{name}'")

    def _get_axes_from_data(self, data):
        """Compute appropriate variable axes from a dataset object."""
        return tuple(getattr(data, 'dimensions', ()))

    def _get_unit_from_data(self, data):
        """Compute appropriate variable units from a dataset object."""
        available = (
            getattr(data, attr) for attr in ('unit', 'units')
            if hasattr(data, attr)
        )
        return next(available, None)


class DatasetAxis(typing.NamedTuple):
    """A dataset axis."""

    size: int
    name: str

    def __str__(self) -> str:
        """A simplified representation of this object."""
        return f"{self.name!r}, size={self.size}"

    def __repr__(self) -> str:
        """An unambiguous representation of this object."""
        module = f"{self.__module__.replace('goats.', '')}."
        name = self.__class__.__qualname__
        return f"{module}{name}({self})"


class NetCDFAxes(DataViewer):
    """An object for viewing axes in a NetCDF dataset."""

    def get_members(self, path: iotools.ReadOnlyPath) -> typing.Mapping:
        dataset = netCDF4.Dataset(path, 'r')
        return dataset.dimensions

    def __getitem__(self, name: str) -> DatasetAxis:
        if name in self.members:
            data = self.members[name]
            size = getattr(data, 'size', None)
            return DatasetAxis(size, name)
        raise KeyError(f"No axis corresponding to {name!r}")


class ViewerFactory(collections.abc.MutableMapping):
    """A class that creates appropriate viewers for a dataset."""

    _viewer_map = {
        '.nc': {
            'variables': NetCDFVariables,
            'axes': NetCDFAxes,
        }
    }

    def __init__(self, path: iotools.ReadOnlyPath) -> None:
        self._viewers = self._get_viewers(path)
        self.path = path

    def __iter__(self) -> typing.Iterator:
        return iter(self._viewers)

    def __len__(self) -> int:
        return len(self._viewers)

    def _get_viewers(
        self,
        path: iotools.ReadOnlyPath,
    ) -> typing.Dict[str, typing.Type[DataViewer]]:
        """Get the viewers for this file format.

        This may expand to accommodate additional file formats or alternate
        methods.
        """
        try:
            viewers = self._viewer_map[path.suffix]
        except KeyError:
            TypeError(f"Unrecognized file type for {path}")
        else:
            return viewers

    def __getitem__(self, group: str):
        """Get the appropriate viewer for this dataset group."""
        if group in self._viewers:
            viewer = self._viewers[group]
            return viewer(self.path)
        raise KeyError(f"No viewer for group '{group}' in {self.path}")

    def __setitem__(self, group: str, viewer: typing.Type[DataViewer]):
        """Associate a new viewer with this dataset group."""
        self._viewers[group] = viewer

    def __delitem__(self, group: str):
        """Delete the viewer associated with this dataset group."""
        del self._viewers[group]


class SubsetKeys(typing.NamedTuple):
    """A subset of names of attributes."""

    full: typing.Tuple[str]
    aliased: typing.Tuple[aliased.MappingKey]
    canonical: typing.Tuple[str]


class DatasetView(iterables.ReprStrMixin, metaclass=iotools.PathSet):
    """A format-agnostic view of a dataset.
    
    An instance of this class provides aliased access to variables and axes
    defined in a specific dataset, given a path to that dataset. It is designed
    to provide a single interface, regardless of file type, with as little
    overhead as possible. Therefore, it does not attempt to modify attributes
    (e.g., converting variable units), since doing so could result in reading a
    potentially large array from disk.
    """

    def __init__(self, path: iotools.PathLike) -> None:
        self.path = iotools.ReadOnlyPath(path)
        self.viewers = ViewerFactory(self.path)
        self._variables = None
        self._axes = None
        self._units = None

    @property
    def variables(self):
        """The variables in this dataset."""
        if self._variables is None:
            variables = {
                observables.ALIASES.get(name, name): variable
                for name, variable in self.viewers['variables'].items()
            }
            self._variables = aliased.Mapping(variables)
        return self._variables

    @property
    def units(self):
        """The unit of each variable, if available."""
        if self._units is None:
            units = {
                name: variable.unit
                for name, variable in self.variables.items(aliased=True)
            }
            self._units = aliased.Mapping(units)
        return self._units

    @property
    def axes(self):
        """The axes in this dataset."""
        if self._axes is None:
            axes = {
                observables.ALIASES.get(name, name): axis
                for name, axis in self.viewers['axes'].items()
            }
            self._axes = aliased.Mapping(axes)
        return self._axes

    def available(self, key: str):
        """Provide the names of available attributes."""
        if key in {'variable', 'variables'}:
            return SubsetKeys(
                full=tuple(self.variables),
                aliased=tuple(self.variables.keys(aliased=True)),
                canonical=tuple(self.viewers['variables'].keys()),
            )
        if key in {'axis', 'axes'}:
            return SubsetKeys(
                full=tuple(self.axes),
                aliased=tuple(self.axes.keys(aliased=True)),
                canonical=tuple(self.viewers['axes'].keys()),
            )

    def use(self, **viewers) -> 'DatasetView':
        """Update the viewers for this instance."""
        self.viewers.update(viewers)
        return self

    def __str__(self) -> str:
        return str(self.path)


S = typing.TypeVar('S', bound='Variables')


class Variables(aliased.Mapping):
    """An interface to dataset variables.
    
    This class provides aliased key-based access to all variables in a dataset.
    It converts each requested dataset variable into a `~quantities.Variable`
    instance with the appropriate MKS unit.
    """

    def __init__(self, dataset: DatasetView) -> None:
        known = {
            k: v for k, v in dataset.variables.items(aliased=True)
            if k in observables.METADATA
        }
        super().__init__(known)
        self._system = quantities.MetricSystem('mks')
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
        metric = self._system[observables.METADATA[name]['quantity']]
        return quantities.Unit(metric.unit)

    def __getitem__(self, key: str):
        """Create the named variable, if possible."""
        if key in self._cache:
            return self._cache[key]
        datavar = super().__getitem__(key)
        variable = datatypes.Variable(
            datavar.data,
            standardize(datavar.unit),
            datavar.axes,
            name=observables.ALIASES[key],
        )
        result = variable.convert_to(self.units[key])
        self._cache[key] = result
        return result


class Indexers(aliased.Mapping):
    """The default collection of axis indexers."""

    def __init__(self, dataset: DatasetView) -> None:
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
        dataset: DatasetView,
        factory: typing.Type[Indexers],
    ) -> None:
        indexers = factory(dataset)
        super().__init__(indexers)
        self.dataset = dataset

    def __getitem__(self, key: str) -> datatypes.Axis:
        indexer = super().__getitem__(key)
        size = self.dataset.axes[key].size
        name = f"'{observables.ALIASES.get(key, key)}'"
        return datatypes.Axis(size, indexer, name=name)


class Dataset:
    """The user interface to a dataset."""

    def __init__(
        self,
        path: iotools.PathLike,
        indexers: typing.Type[Indexers]=None,
    ) -> None:
        self.path = path
        self.view = DatasetView(path)
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

    def resolve_axes(self, names: typing.Iterable[str]):
        """Compute and order the available axes in `names`."""
        axes = self.view.available('axes').canonical
        return tuple(name for name in axes if name in names)


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


