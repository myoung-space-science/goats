"""Tools for managing datasets."""

import abc
import collections.abc
from pathlib import Path
from typing import *

import netCDF4

from goats.common.iotools import ReadOnlyPath, SingleInstance
from goats.common.iterables import CollectionMixin


class DataViewer(CollectionMixin, collections.abc.Mapping):
    """An abstract base class for data-viewing objects."""

    def __init__(self, path: ReadOnlyPath) -> None:
        self.members = self.get_members(path)
        self.collect('members')

    @abc.abstractmethod
    def get_members(self, path: ReadOnlyPath) -> Mapping:
        """Get the appropriate members for this viewer."""
        pass


class NetCDFVariables(DataViewer):
    """An object for viewing variables in a NetCDF dataset."""

    def get_members(self, path: ReadOnlyPath) -> Mapping:
        dataset = netCDF4.Dataset(path, 'r')
        return dataset.variables

    def __getitem__(self, name: str):
        if name in self.members:
            return self.members[name]
        raise KeyError(f"No variable called '{name}'")


class NetCDFSizes(DataViewer):
    """An object for viewing sizes in a NetCDF dataset."""

    def get_members(self, path: ReadOnlyPath) -> Mapping:
        dataset = netCDF4.Dataset(path, 'r')
        return dataset.dimensions

    def __getitem__(self, name: str) -> Optional[int]:
        if name in self.members:
            data = self.members[name]
            return getattr(data, 'size', None)
        raise KeyError(f"No dimension called '{name}'")


class ViewerFactory(CollectionMixin, collections.abc.MutableMapping):
    """A class that creates appropriate viewers for a dataset."""

    _viewer_map = {
        '.nc': {
            'variables': NetCDFVariables,
            'sizes': NetCDFSizes,
        }
    }

    def __init__(self, path: ReadOnlyPath) -> None:
        self._viewers = self._get_viewers(path)
        self.collect('_viewers')
        self.path = path

    def _get_viewers(self, path: ReadOnlyPath) -> Dict[str, Type[DataViewer]]:
        """Get the viewers for this file format.

        This may expand to accomodate additional file formats or alternate
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

    def __setitem__(self, group: str, viewer: Type[DataViewer]):
        """Associate a new viewer with this dataset group."""
        self._viewers[group] = viewer

    def __delitem__(self, group: str):
        """Delete the viewer associated with this dataset group."""
        del self._viewers[group]


class DatasetView(metaclass=SingleInstance):
    """A format-agnostic view of a dataset."""

    def __init__(self, path: Union[str, Path]) -> None:
        self.path = ReadOnlyPath(path)
        self.viewers = ViewerFactory(self.path)

    @property
    def variables(self) -> DataViewer:
        """The variables in this dataset."""
        return self.viewers['variables']

    @property
    def axes(self) -> Tuple[str]:
        """The names of axes in this dataset."""
        return tuple(self.sizes)

    @property
    def sizes(self) -> DataViewer:
        """The sizes of axes in this dataset."""
        return self.viewers['sizes']

    def use(self, **viewers) -> 'DatasetView':
        """Update the viewers for this instance."""
        self.viewers.update(viewers)
        return self


