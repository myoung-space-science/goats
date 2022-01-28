import pathlib
import typing

from goats import Environment
from goats.common import base
from goats.common import iterables
from goats.common import iotools
from goats.common import indexing
from goats.common import quantities
from goats.eprem import parameters
from goats.eprem import datasets
from goats.eprem import observables


_pkg = Environment('eprem')


def find_file_by_template(
    templates: typing.List[typing.Callable],
    name: str,
    datadir: iotools.PathLike=pathlib.Path.cwd(),
) -> pathlib.Path:
    """Find a valid path that conforms to a given template."""
    datadir = iotools.ReadOnlyPath(datadir)
    for template in templates:
        test = datadir / str(template(name))
        if test.exists():
            return test


class Observer(base.Observer):
    """Base class for EPREM observers."""

    def __init__(
        self,
        templates: typing.Iterable[typing.Callable],
        name: int=None,
        path: iotools.PathLike=None,
        config: iotools.PathLike=None,
        system: str='mks',
    ) -> None:
        self._templates = templates
        self._name = name
        self._path = path
        self._config = self._build_confpath(config or _pkg['config'])
        self.system = quantities.MetricSystem(system)
        self._dataset = None
        self._arguments = None
        super().__init__(
            observables.Observables(
                self.dataset,
                self.system,
                self.arguments,
            )
        )

    @property
    def path(self) -> iotools.ReadOnlyPath:
        """The path to this observer's dataset."""
        if not iterables.missing(self._name):
            path = self._build_datapath(self._name, self._path)
            return iotools.ReadOnlyPath(path)
        if self._path and self._path.is_file():
            return iotools.ReadOnlyPath(self._path)
        message = (
            "You must provide either a name (and optional directory)"
            " or the path to a dataset."
        )
        raise TypeError(message) from None

    @property
    def dataset(self):
        """This observer's dataset."""
        if self._dataset is None:
            return datasets.Dataset(self.path, self.system)
        return self._dataset

    @property
    def arguments(self):
        """The parameter arguments available to this observer."""
        if self._arguments is None:
            source_path = _pkg['src']
            config_path = self._config
            self._arguments = parameters.Arguments(
                source_path=source_path,
                config_path=config_path,
            )
        return self._arguments

    def _build_datapath(
        self,
        name: typing.Union[int, str],
        directory: iotools.PathLike=None,
    ) -> pathlib.Path:
        """Create the full path for a given observer from components."""
        if directory is None:
            return find_file_by_template(
                self._templates,
                name,
                datadir=pathlib.Path.cwd(),
            )
        dpath = pathlib.Path(directory).expanduser().resolve()
        if dpath.is_dir():
            return find_file_by_template(
                self._templates,
                name,
                datadir=dpath,
            )
        raise TypeError(
            "Can't create path to dataset"
            f" from directory {directory!r}"
        )

    def _build_confpath(self, arg: iotools.PathLike):
        """Create the full path to the named configuration file."""
        path = pathlib.Path(arg)
        if path.name == arg: # just the file name
            return iotools.ReadOnlyPath(self.path.parent / arg)
        return iotools.ReadOnlyPath(arg)

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
        axis = self.dataset.axes[name]
        values = axis(**kwargs)
        if unit and isinstance(values, indexing.Coordinates):
            return values.to(unit)
        return values


class Stream(Observer):
    """An EPREM stream observer."""

    def __init__(
        self,
        name: int=None,
        path: iotools.PathLike=None,
        config: iotools.PathLike=None,
        system: str='mks'
    ) -> None:
        templates = [
            lambda n: f'obs{n:06}.nc',
            lambda n: f'flux{n:06}.nc',
        ]
        super().__init__(
            templates,
            name=name,
            path=path,
            config=config,
            system=system
        )


class Point(Observer):
    """An EPREM point observer."""

    def __init__(
        self,
        name: int=None,
        path: iotools.PathLike=None,
        config: iotools.PathLike=None,
        system: str='mks'
    ) -> None:
        templates = [
            lambda n: f'p_obs{n:06}.nc',
        ]
        super().__init__(
            templates,
            name=name,
            path=path,
            config=config,
            system=system
        )
