"""
Tools for reading data and metadata from file.
"""

import abc
import collections.abc
import typing

import netCDF4
import numpy
import numpy.typing

from goats.core import aliased
from goats.core import iotools
from goats.core import iterables
from goats.core import reference


class Viewer(collections.abc.Mapping):
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


class Variable(typing.NamedTuple):
    """A dataset variable."""

    data: numpy.typing.ArrayLike
    unit: str
    dimensions: typing.Tuple[str]
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
            f"dimensions={self.dimensions}",
            f"name={self.name!r}",
        ]
        indent = ' ' * tab
        return sep.join(f"{indent}{attr}" for attr in attrs)


class NetCDFVariables(Viewer):
    """An object for viewing variables in a NetCDF dataset."""

    def get_members(self, path: iotools.ReadOnlyPath) -> typing.Mapping:
        dataset = netCDF4.Dataset(path, 'r')
        return dataset.variables

    def __getitem__(self, name: str) -> Variable:
        if name in self.members:
            data = self.members[name]
            unit = self._get_unit_from_data(data)
            dimensions = self._get_dimensions_from_data(data)
            return Variable(data, unit, dimensions, name)
        raise KeyError(f"No variable called '{name}'")

    def _get_dimensions_from_data(self, data):
        """Compute appropriate variable dimensions from a dataset object."""
        return tuple(getattr(data, 'dimensions', ()))

    def _get_unit_from_data(self, data):
        """Compute appropriate variable units from a dataset object."""
        available = (
            getattr(data, attr) for attr in ('unit', 'units')
            if hasattr(data, attr)
        )
        return next(available, None)


class Axis(typing.NamedTuple):
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


class NetCDFAxes(Viewer):
    """An object for viewing axes in a NetCDF dataset."""

    def get_members(self, path: iotools.ReadOnlyPath) -> typing.Mapping:
        dataset = netCDF4.Dataset(path, 'r')
        return dataset.dimensions

    def __getitem__(self, name: str) -> Axis:
        if name in self.members:
            data = self.members[name]
            size = getattr(data, 'size', None)
            return Axis(size, name)
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
    ) -> typing.Dict[str, typing.Type[Viewer]]:
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

    def __setitem__(self, group: str, viewer: typing.Type[Viewer]):
        """Associate a new viewer with this dataset group."""
        self._viewers[group] = viewer

    def __delitem__(self, group: str):
        """Delete the viewer associated with this dataset group."""
        del self._viewers[group]


class SubsetKeys(typing.NamedTuple):
    """A subset of names of attributes."""

    full: typing.Tuple[str]
    """Every attribute's names and aliases in a flat sequence."""
    grouped: typing.Tuple[aliased.Group]
    """Groups of aliases for each attribute."""
    canonical: typing.Tuple[str]
    """The (unaliased) name of each attribute as defined on disk."""


class Interface(iterables.ReprStrMixin, metaclass=iotools.PathSet):
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
    def variables(self) -> aliased.Mapping[str, Variable]:
        """The variables in this dataset."""
        if self._variables is None:
            self._variables = aliased.Mapping(
                self.viewers['variables'],
                aliases=reference.ALIASES,
            )
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
    def axes(self) -> aliased.Mapping[str, Axis]:
        """The axes in this dataset."""
        if self._axes is None:
            self._axes = aliased.Mapping(
                self.viewers['axes'],
                aliases=reference.ALIASES,
            )
        return self._axes

    def available(self, key: str):
        """Provide the names of available attributes."""
        if key in {'variable', 'variables'}:
            return SubsetKeys(
                full=tuple(self.variables),
                grouped=tuple(self.variables.keys(aliased=True)),
                canonical=tuple(self.viewers['variables'].keys()),
            )
        if key in {'axis', 'axes', 'dimension', 'dimensions'}:
            return SubsetKeys(
                full=tuple(self.axes),
                grouped=tuple(self.axes.keys(aliased=True)),
                canonical=tuple(self.viewers['axes'].keys()),
            )

    def use(self, **viewers) -> 'Interface':
        """Update the viewers for this instance."""
        self.viewers.update(viewers)
        return self

    def __str__(self) -> str:
        return str(self.path)


