import abc
from pathlib import Path
from typing import *

from goats.common import spelling
from goats.common import base
from goats.common import quantities
from goats.common.iotools import ReadOnlyPath
from goats.common.iterables import missing
from goats.eprem import datasets
from goats.eprem import observables
from goats.eprem import parameters


class ObservableKeyError(Exception):
    """The given key does not correspond to a known observable."""
    def __init__(self, key: str) -> None:
        self.key = key

    def __str__(self) -> str:
        return f"No known observable named {self.key}"


class Observer(base.Observer):
    """The base class for all EPREM observers."""
    def __init__(
        self,
        name: int=None,
        directory: Union[str, Path]=None,
        path: Union[str, Path]=None,
        system: str='mks',
    ) -> None:
        if not missing(name):
            directory = directory or Path().cwd()
            path = self._get_datapath(name, directory)
        elif path:
            path = Path(path).expanduser().resolve()
        else:
            message = (
                "You must provide either a name (and optional directory)"
                " or the path to a dataset."
            )
            raise TypeError(message) from None
        super().__init__(path=path)
        self.system = system
        self._dataset = datasets.DatasetView(path, self.system)
        self._config = parameters.ConfigManager(path)
        self._units = quantities.MetricSystem(self.system)
        self._observables = observables.Observables(self._dataset)
        self._time = None
        self._energy = None
        self._mu = None
        self._coordinates = datasets.Axes(self._dataset)
        self._spellcheck = spelling.SpellChecker(self._observables.names)

    @property
    def time(self):
        """The times available to this observer."""
        if self._time is None:
            Time = self._coordinates['time']
            self._time = Time()
        return self._time

    @property
    def energy(self):
        """The energies available to this observer."""
        if self._energy is None:
            Energy = self._coordinates['energy']
            self._energy = Energy()
        return self._energy

    @property
    def mu(self):
        """The pitch-angle cosines available to this observer."""
        if self._mu is None:
            Mu = self._coordinates['mu']
            self._mu = Mu()
        return self._mu

    def _get_datapath(
        self,
        name: Union[int, str],
        directory: Union[str, Path],
    ) -> ReadOnlyPath:
        """Retrieve the full path for a given observer."""
        return find_file_by_template(self._templates, name, datadir=directory)

    @property
    @abc.abstractmethod
    def _templates(self) -> List[Callable]:
        """Filename templates to use when searching for datasets."""
        return []

    def _can_observe(self, name: str) -> bool:
        return name in self._observables

    def _observe(self, name: str) -> base.Observable:
        return self._observables[name]

    def _cannot_observe(self, name: str):
        if self._spellcheck.misspelled(name):
            raise spelling.SpellingError(name, self._spellcheck.suggestions)
        message = f"{self.__class__.__qualname__} cannot observe {name}"
        raise TypeError(message) from None


class Stream(Observer):
    """The class representing an EPREM stream observer."""
    @property
    def _templates(self) -> List[Callable]:
        return [
            lambda n: f'obs{n:06}.nc',
            lambda n: f'flux{n:06}.nc',
        ]


class Point(Observer):
    """The class representing an EPREM point observer."""
    @property
    def _templates(self) -> List[Callable]:
        return [
            lambda n: f'p_obs{n:06}.nc',
        ]


def find_file_by_template(
    templates: List[Callable],
    name: str,
    datadir: Union[str, Path]=Path().cwd(),
) -> Path:
    """Find a valid path that conforms to a given template."""
    datadir = ReadOnlyPath(datadir)
    for template in templates:
        test = datadir / str(template(name))
        if test.exists():
            return test


