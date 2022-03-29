import pathlib
import typing
import numbers

import numpy
import numpy.typing

from goats import Environment
from ..core import (
    aliased,
    base,
    datasets,
    datatypes,
    iterables,
    iotools,
    numerical,
    physical,
    quantities,
)
from . import observables
from .parameters import BaseTypesH


ENV = Environment('eprem')


basetypes = BaseTypesH(source=ENV['src'])


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


class IndexerFactory(iterables.ReprStrMixin, aliased.Mapping):
    """A factory for EPREM array-indexing objects."""

    def __init__(self, dataset: datasets.DatasetView) -> None:
        self.variables = datasets.Variables(dataset)
        mass = self.variables['mass'].convert_to('nuc')
        charge = self.variables['charge'].convert_to('e')
        self.symbols = physical.elements(mass, charge)
        # TODO: Consider using reference arrays in methods, with the possible
        # exception of `_build_shell`.
        indexers = {
            'time': {
                'method': self._build_time,
                'reference': self.variables['time'],
            },
            'shell': {
                'method': self._build_shell,
                'reference': numpy.array(self.variables['shell'], dtype=int),
            },
            'species': {
                'method': self._build_species,
                'reference': self.symbols,
            },
            'energy': {
                'method': self._build_energy,
                'reference': self.variables['energy'],
            },
            'mu': {
                'method': self._build_mu,
                'reference': self.variables['mu'],
            },
        }
        mapping = {
            dataset.axes.alias(name, include=True): indexer
            for name, indexer in indexers.items()
        }
        super().__init__(mapping)

    def __getitem__(self, key: str) -> datatypes.Indexer:
        this = super().__getitem__(key)
        return datatypes.Indexer(this['method'], this['reference'])

    def _build_time(self, targets):
        """Build the time-axis indexer."""
        return self._build_coordinates(targets, self.variables['time'])

    def _build_shell(self, targets):
        """Build the shell-axis indexer."""
        return datatypes.Indices(targets)

    def _build_species(self, targets):
        """Build the species-axis indexer."""
        indices = []
        symbols = []
        for target in targets:
            if isinstance(target, str):
                indices.append(self.symbols.index(target))
                symbols.append(target)
            elif isinstance(target, numbers.Integral):
                indices.append(target)
                symbols.append(self.symbols[target])
        return datatypes.IndexMap(indices, targets)

    def _build_energy(self, targets, species: typing.Union[str, int]=0):
        """Build the energy-axis indexer."""
        s = self._build_species([species])
        _targets = (
            numpy.squeeze(targets[s, :]) if getattr(targets, 'ndim', None) == 2
            else targets
        )
        _reference = numpy.squeeze(self.variables['energy'][s, :])
        return self._build_coordinates(_targets, _reference)

    def _build_mu(self, targets):
        """Build the mu-axis indexer."""
        return self._build_coordinates(targets, self.variables['mu'])

    def _build_coordinates(
        self,
        targets: numpy.typing.ArrayLike,
        reference: datatypes.Variable,
    ) -> datatypes.Coordinates:
        """Build an arbitrary coordinate object."""
        measured = quantities.measure(targets)
        vector = quantities.Vector(measured.values, measured.unit)
        values = (
            vector.unit(reference.unit)
            if vector.unit().dimension == reference.unit.dimension
            else vector
        )
        indices = [
            numerical.find_nearest(reference, float(value)).index
            for value in values
        ]
        return datatypes.Coordinates(indices, values, reference.unit)


    def __str__(self) -> str:
        return ', '.join(str(key) for key in self.keys(aliased=True))


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
        self._config = self._build_confpath(config or ENV['config'])
        self.system = quantities.MetricSystem(system)
        self._dataset = None
        self._arguments = None
        interface = observables.Observables(
            self.dataset,
            self.arguments,
        )
        super().__init__(interface, self.arguments)

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
            self._dataset = datasets.Dataset(self.path, IndexerFactory)
        return self._dataset

    @property
    def arguments(self):
        """The parameter arguments available to this observer."""
        if self._arguments is None:
            source_path = ENV['src']
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
            f" ({dpath})"
        )

    def _build_confpath(self, arg: iotools.PathLike):
        """Create the full path to the named configuration file."""
        path = pathlib.Path(arg)
        if path.name == arg: # just the file name
            return iotools.ReadOnlyPath(self.path.parent / arg)
        return iotools.ReadOnlyPath(arg)

    def time(self, unit: str=None):
        """This observer's times."""
        return self._get_indices('time', unit=unit).values

    def shell(self):
        """This observer's shells."""
        return self._get_indices('shell')

    def species(self):
        """This observer's species."""
        return self._get_indices('species')

    def energy(self, unit: str=None):
        """This observer's energies."""
        return self._get_indices('energy', unit=unit).values

    def mu(self, unit: str=None):
        """This observer's pitch-angle cosines."""
        return self._get_indices('mu', unit=unit).values

    def _get_indices(self, name: str, unit: str=None, **kwargs):
        """Get the index-like object for this axis."""
        axis = self.dataset.axes[name]
        values = axis(**kwargs)
        if unit and isinstance(values, datatypes.Coordinates):
            return values.with_unit(unit)
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

