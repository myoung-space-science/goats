import pathlib
import typing

from goats import common
from goats.common import iterables
from goats.common import iotools
from goats.common import indexing
from goats.eprem import parameters
from goats.eprem import datasets
from goats.eprem import observables


def find_file_by_template(
    templates: typing.List[typing.Callable],
    name: str,
    datadir: typing.Union[str, pathlib.Path]=pathlib.Path().cwd(),
) -> pathlib.Path:
    """Find a valid path that conforms to a given template."""
    datadir = iotools.ReadOnlyPath(datadir)
    for template in templates:
        test = datadir / str(template(name))
        if test.exists():
            return test


class Observer(common.Observer):
    """Base class for EPREM observers."""

    def __init__(
        self,
        templates: typing.Iterable[typing.Callable],
        name: int=None,
        directory: typing.Union[str, pathlib.Path]=None,
        path: typing.Union[str, pathlib.Path]=None,
        system: str='mks',
    ) -> None:
        self._templates = templates
        self._name = name
        self._directory = directory
        self._path = path
        self.system = system
        super().__init__(observables.Observables(self.dataset))
        self._axes = datasets.Axes(self.dataset)

    @property
    def path(self):
        """The path to this observer's dataset."""
        if not iterables.missing(self._name):
            directory = self._directory or pathlib.Path().cwd()
            return self._get_datapath(self._name, directory)
        if self._path:
            return pathlib.Path(self._path).expanduser().resolve()
        message = (
            "You must provide either a name (and optional directory)"
            " or the path to a dataset."
        )
        raise TypeError(message) from None

    @property
    def config(self):
        """The configuration parameters for this observer's dataset."""
        return parameters.ConfigManager(self.path)

    @property
    def dataset(self):
        """This observer's dataset."""
        return datasets.DatasetView(self.path, self.system)

    def _get_datapath(
        self,
        name: typing.Union[int, str],
        directory: typing.Union[str, pathlib.Path],
    ) -> iotools.ReadOnlyPath:
        """Retrieve the full path for a given observer."""
        return find_file_by_template(self._templates, name, datadir=directory)

    def time(self, unit: str=None):
        """This observer's times."""
        return self._get_indices('time', unit=unit)

    def shell(self):
        """This observer's shells."""
        return self._get_indices('shell')

    def species(self):
        """This observer's species."""
        return self._get_indices('species')

    def energy(self, unit: str=None):
        """This observer's energies."""
        return self._get_indices('energy', unit=unit)

    def mu(self, unit: str=None):
        """This observer's pitch-angle cosines."""
        return self._get_indices('mu', unit=unit)

    def _get_indices(self, name: str, unit: str=None, **kwargs):
        """Get the index-like object for this axis."""
        axis = self._axes[name]
        values = axis(**kwargs)
        if unit and isinstance(values, indexing.Coordinates):
            return values.to(unit)
        return values


class Stream(Observer):
    """An EPREM stream observer."""

    def __init__(
        self,
        name: int=None,
        directory: typing.Union[str, pathlib.Path]=None,
        path: typing.Union[str, pathlib.Path]=None,
        system: str='mks'
    ) -> None:
        templates = [
            lambda n: f'obs{n:06}.nc',
            lambda n: f'flux{n:06}.nc',
        ]
        super().__init__(
            templates,
            name=name,
            directory=directory,
            path=path,
            system=system
        )


class Point(Observer):
    """An EPREM point observer."""

    def __init__(
        self,
        name: int=None,
        directory: typing.Union[str, pathlib.Path]=None,
        path: typing.Union[str, pathlib.Path]=None,
        system: str='mks'
    ) -> None:
        templates = [
            lambda n: f'p_obs{n:06}.nc',
        ]
        super().__init__(
            templates,
            name=name,
            directory=directory,
            path=path,
            system=system
        )

