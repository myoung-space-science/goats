import pathlib
import typing
import numbers

import numpy
import numpy.typing

from goats import Environment
from ..core import (
    axis,
    aliased,
    computable,
    datafile,
    fundamental,
    iterables,
    iotools,
    index,
    measurable,
    metric,
    numerical,
    observable,
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


class Observer(observer.Interface, iterables.ReprStrMixin):
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
        self.system = metric.System(system)
        self._dataset = None
        self._arguments = None
        variables = variable.Interface(self.dataset, self.system)
        interface = observing.Interface(
            axis.Interface(Indexers(self.dataset), self.dataset, self.system),
            variables,
            self.arguments,
        )
        super().__init__(
            interface,
            [*variables, *computable.REGISTRY],
            self.arguments,
        )

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

    # Convert to property?
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
        """The interface to this observer's data."""
        return datafile.Interface(self.path)

    @property
    def arguments(self):
        """The parameter arguments available to this observer."""
        if self._arguments is None:
            source_path = ENV['src']
            config_path = self._config
            self._arguments = runtime.Arguments(
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
            default = ENV['datadir'] or pathlib.Path.cwd()
            return iotools.find_file_by_template(
                self._templates,
                name,
                datadir=default,
            )
        dpath = pathlib.Path(directory).expanduser().resolve()
        if dpath.is_dir():
            return iotools.find_file_by_template(
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
        if unit and values.unit is not None:
            return values.with_unit(unit)
        return values

    def __str__(self) -> str:
        return str(self.path)


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

