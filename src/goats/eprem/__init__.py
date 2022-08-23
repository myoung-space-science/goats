import pathlib
import typing
import numbers

import numpy
import numpy.typing

from goats import Environment
from ..core import (
    axis,
    aliased,
    datafile,
    fundamental,
    iterables,
    iotools,
    index,
    measurable,
    numerical,
    observer,
    observing,
    physical,
    reference,
    variable,
)
from . import interpolation
from .runtime import BaseTypesH


ENV = Environment('eprem')


basetypes = BaseTypesH(source=ENV['src'])


class Indexers(aliased.Mapping, iterables.ReprStrMixin):
    """A factory for EPREM array-indexing objects."""

    def __init__(self, data: datafile.Interface) -> None:
        self.variables = variable.Interface(data)
        mass = self.variables['mass']['nuc']
        charge = self.variables['charge']['e']
        self.symbols = fundamental.elements(mass, charge)
        # TODO: Consider using reference arrays in methods, with the possible
        # exception of `_build_shell`.
        indexers = {
            'time': {
                'method': self._build_time,
                'size': data.axes['time'].size,
                'reference': self.variables['time'],
            },
            'shell': {
                'method': self._build_shell,
                'size': data.axes['shell'].size,
                'reference': numpy.array(self.variables['shell'], dtype=int),
            },
            'species': {
                'method': self._build_species,
                'size': data.axes['species'].size,
                'reference': self.symbols,
            },
            'energy': {
                'method': self._build_energy,
                'size': data.axes['energy'].size,
                'reference': self.variables['energy'],
            },
            'mu': {
                'method': self._build_mu,
                'size': data.axes['mu'].size,
                'reference': self.variables['mu'],
            },
        }
        mapping = {
            data.axes.alias(name, include=True): indexer
            for name, indexer in indexers.items()
        }
        super().__init__(mapping)

    def __getitem__(self, key: str) -> index.Factory:
        this = super().__getitem__(key)
        return index.Factory(this['method'], this['size'], this['reference'])

    def _build_time(self, targets):
        """Build the time-axis indexer."""
        return self._build_coordinates(targets, self.variables['time'])

    def _build_shell(self, targets):
        """Build the shell-axis indexer."""
        return index.create(targets)

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
        return index.create(indices, values=targets)

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
        reference: variable.Quantity,
    ) -> index.Quantity:
        """Build an arbitrary coordinate object."""
        result = measurable.measure(targets)
        array = physical.Array(result.values, unit=result.unit)
        values = numpy.array(
            array[reference.unit]
            if array.unit.dimension == reference.unit.dimension
            else array
        )
        indices = [
            numerical.find_nearest(reference, float(value)).index
            for value in values
        ]
        return index.create(indices, values=values, unit=reference.unit)

    def __str__(self) -> str:
        return ', '.join(str(key) for key in self.keys(aliased=True))


Instance = typing.TypeVar('Instance', bound='Observer')


class Observer(observer.Interface, iterables.ReprStrMixin):
    """Base class for EPREM observers."""

    _templates: typing.Iterable[typing.Callable] = None

    def __init__(
        self,
        __id: int,
        source: iotools.PathLike=ENV['source'],
        config: iotools.PathLike=ENV['config'],
        system: str='mks',
    ) -> None:
        self._id = __id
        super().__init__('preEruption', 'phiOffset', system=system)
        self._confpath = None
        self.readfrom(source, config=config)

    def readfrom(
        self: Instance,
        source: iotools.PathLike,
        config: iotools.PathLike=None
    ) -> Instance:
        # Use arguments to update paths.
        datapath = self._build_datapath(source)
        confpath = self._build_confpath(config, directory=datapath.parent)
        # Update internal dataset attribute.
        dataset = datafile.Interface(datapath)
        system = self.system()
        axes = axis.Interface(Indexers(dataset), dataset, system)
        variables = variable.Interface(dataset, system)
        constants = runtime.Arguments(
            source_path=ENV['src'],
            config_path=confpath,
        )
        self._data = observing.Interface(axes, variables, constants)
        # Update path attributes if everything succeeded.
        self._confpath = confpath
        return super().readfrom(datapath)

    def _build_datapath(self, directory: iotools.PathLike):
        """Create the path to the dataset from `directory`."""
        default = ENV['source'] or pathlib.Path.cwd()
        this = iotools.ReadOnlyPath(directory or default)
        if this.is_dir():
            path = iotools.find_file_by_template(
                self._templates,
                self._id,
                directory=this,
            )
            return iotools.ReadOnlyPath(path)
        raise TypeError(
            f"Can't create path to dataset from {directory!r}"
        ) from None

    def _build_confpath(
        self,
        config: iotools.PathLike,
        directory: iotools.PathLike=pathlib.Path.cwd(),
    ) -> iotools.ReadOnlyPath:
        """Create the path to the configuration file."""
        if not config: # use default directory and name
            return iotools.ReadOnlyPath(directory / ENV['config'])
        this = pathlib.Path(config)
        if this.is_dir(): # use default directory
            return iotools.ReadOnlyPath(this / ENV['config'])
        if this.name == config: # use default name
            return iotools.ReadOnlyPath(directory / this)
        raise ValueError(
            f"Can't create path to configuration file from {config!r}"
        ) from None

    @property
    def confpath(self):
        """The full path to this observer's runtime parameter file."""
        return self._confpath

    @property
    def datapath(self) -> iotools.ReadOnlyPath:
        """The path to this observer's dataset."""
        return self._source

    def process(self, old: variable.Quantity) -> variable.Quantity:
        if any(self._get_reference(alias) for alias in old.name):
            # This is an axis-reference quantity.
            return self._subscript(old)
        needed = self._compute_coordinates(old)
        if not needed:
            # There are no axes over which to interpolate.
            return self._subscript(old)
        new = self._interpolate(old, needed)
        # We only want to subscript the uninterpolated axes.
        interpolated = [
            'shell' if d == 'radius' else d
            for d in needed
        ]
        axes = list(set(old.axes) - set(interpolated))
        return self._subscript(new, *axes)

    def _subscript(self, q: variable.Quantity, *axes: str):
        """Extract a subset of this quantity."""
        if axes:
            indices = [
                self.context.get_index(a) if a in axes else slice(None)
                for a in q.axes
            ]
        else:
            indices = [self.context.get_index(a) for a in q.axes]
        return q[tuple(indices)]

    def _compute_coordinates(self, q: variable.Quantity):
        """Determine the measurable observing indices."""
        dimensions = self._compute_dimensions(q)
        coordinates = {}
        if not dimensions:
            return coordinates
        for a in q.axes:
            idx = self.context.get_index(a)
            if a in dimensions and idx.unit is not None:
                coordinates[a] = {
                    'targets': numpy.array(idx.data),
                    'reference': self._get_reference(a),
                }
        for key in reference.ALIASES['radius']:
            if values := self.context.get_value(key):
                try:
                    iter(values)
                except TypeError:
                    floats = [float(values)]
                else:
                    floats = [float(value) for value in values]
                coordinates['radius'] = {
                    'targets': numpy.array(floats),
                    'reference': self._get_reference('radius'),
                }
        return coordinates

    def _compute_dimensions(self, q: variable.Quantity):
        """Determine over which axes to interpolate, if any."""
        coordinates = {}
        references = []
        for a in q.axes:
            idx = self.context.get_index(a)
            if idx.unit is not None:
                coordinates[a] = idx.data
                references.append(self._get_reference(a))
        axes = [
            a for (a, c), r in zip(coordinates.items(), references)
            if not numpy.all([r.array_contains(target) for target in c])
        ]
        if any(r in self.context for r in reference.ALIASES['radius']):
            axes.append('radius')
        return axes

    _references = None

    def _get_reference(self, name: str) -> typing.Optional[variable.Quantity]:
        """Get a reference quantity for indexing."""
        if self._references is None:
            axes = {
                k: v.reference
                for k, v in self.data.axes.items(aliased=True)
            }
            rtp = {
                (k, *self.data.variables.alias(k, include=True)):
                self.data.variables[k]
                for k in {'radius', 'theta', 'phi'}
            }
            self._references = aliased.Mapping({**axes, **rtp})
        return self._references.get(name)

    def _interpolate(
        self,
        q: variable.Quantity,
        coordinates: dict,
    ) -> variable.Quantity:
        """Internal interpolation logic."""
        array = None
        for coordinate, current in coordinates.items():
            array = self._interpolate_coordinate(
                q,
                current['targets'],
                current['reference'],
                coordinate=coordinate,
                workspace=array,
            )
        meta = {k: getattr(q, k, None) for k in {'unit', 'name', 'axes'}}
        return variable.Quantity(array, **meta)

    def _interpolate_coordinate(
        self,
        q: variable.Quantity,
        targets: numpy.ndarray,
        reference: variable.Quantity,
        coordinate: str=None,
        workspace: numpy.ndarray=None,
    ) -> numpy.ndarray:
        """Interpolate a variable array based on a known coordinate."""
        array = numpy.array(q) if workspace is None else workspace
        indices = (q.axes.index(d) for d in reference.axes)
        dst, src = zip(*enumerate(indices))
        reordered = numpy.moveaxis(array, src, dst)
        interpolated = interpolation.apply(
            reordered,
            numpy.array(reference),
            targets,
            coordinate=coordinate,
        )
        return numpy.moveaxis(interpolated, dst, src)

    def __str__(self) -> str:
        return str(self.datapath)


class Stream(Observer):
    """An EPREM stream observer."""

    _templates = [
        lambda n: f'obs{n:06}.nc',
        lambda n: f'flux{n:06}.nc',
    ]


class Point(Observer):
    """An EPREM point observer."""

    _templates = [
        lambda n: f'p_obs{n:06}.nc',
    ]

