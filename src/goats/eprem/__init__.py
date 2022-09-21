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
    metadata,
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


class Axes(axis.Interface):
    """Interface to EPREM axis-indexing objects."""

    def __init__(self, data: datafile.Interface) -> None:
        super().__init__(data)
        self.variables = variable.Interface(data)
        self._time = None
        self._shell = None
        self._species = None
        self._energy = None
        self._mu = None
        defined = {
            'time': self.time,
            'shell': self.shell,
            'species': self.species,
            'energy': self.energy,
            'mu': self.mu,
        }
        self.update(defined)

    @property
    def shell(self):
        """Indexer for the EPREM shell dimension."""
        if self._shell is None:
            def method(targets, unit):
                # NOTE: The presence of `unit` is a hack because 'shell'
                # currently gets a unit of '1' even though it should probably be
                # None. This hack is due to the design of `axis.Quantity`.
                return axis.Data(targets)
            self._shell = axis.Indexer(method, len(self.variables['shell']))
        return self._shell

    @property
    def time(self):
        """Indexer for the EPREM time dimension."""
        if self._time is None:
            this = self.variables['time']
            self._time = axis.Indexer(self._build_coordinate(this), len(this))
        return self._time

    @property
    def mu(self):
        """Indexer for the EPREM pitch-angle dimension."""
        if self._mu is None:
            this = self.variables['mu']
            self._mu = axis.Indexer(self._build_coordinate(this), len(this))
        return self._mu

    @property
    def energy(self):
        """Indexer for the EPREM energy dimension."""
        if self._energy is None:
            this = self.variables['energy']
            def method(
                targets,
                unit: metadata.UnitLike,
                species: typing.Union[str, int]=0,
            ) -> axis.Data:
                s = self.species.compute([species]).points
                t = (
                    numpy.squeeze(targets[s, :])
                    if getattr(targets, 'ndim', None) == 2
                    else targets
                )
                compute = self._build_coordinate(numpy.squeeze(this[s, :]))
                return compute(t, unit)
            self._energy = axis.Indexer(method, this.shape[1])
        return self._energy

    @property
    def species(self):
        """Indexer for the EPREM species dimension."""
        if self._species is None:
            mass = self.variables['mass']['nuc']
            charge = self.variables['charge']['e']
            symbols = fundamental.elements(mass, charge)
            def method(targets):
                indices = [
                    symbols.index(target)
                    if isinstance(target, str) else int(target)
                    for target in targets
                ]
                return axis.Data(indices, values=symbols)
            self._species = axis.Indexer(method, len(symbols))
        return self._species

    def _build_coordinate(self, this: variable.Quantity):
        """Create coordinate-like axis data from the given variable."""
        def method(targets, unit: metadata.UnitLike):
            if not targets:
                # If there are no target values, we assume the user wants the
                # entire axis.
                return axis.Data(range(len(this)), values=numpy.array(this))
            if all(isinstance(t, typing.SupportsIndex) for t in targets):
                # All the target values are already indices.
                return axis.Data(targets, values=numpy.array(this))
            measured = measurable.measure(targets)
            if measured.unit != '1':
                # If the targets include a dimensioned unit, we want to
                # initialize the array with that unit.
                array = physical.Array(measured.values, unit=measured.unit)
            else:
                # If the measured unit is dimensionless, it could be because the
                # targets truly are dimensionless or because the user wants to
                # use the default unit. Since we have no choice but to assume
                # that the calling object (probably an instance of
                # `core.axis.Quantity`) passed an appropriate default unit,
                # which may be dimensionless, the default unit is the
                # appropriate unit for both cases.
                array = physical.Array(measured.values, unit=unit)
            # Convert the reference variable quantity to the default unit.
            this[unit]
            if array.unit | this.unit: # Could also use try/except
                array[this.unit]
            values = numpy.array(array)
            indices = [
                numerical.find_nearest(this, float(value)).index
                for value in values
            ]
            return axis.Data(indices, values=values)
        return method


class Context(observing.Context):
    """The EPREM-specific observing context."""

    # TODO:
    # - generalize interpolation from this subpackage
    # - implement generalized interpolation in core.observing.Context
    # - add EPREM-specific interpolation here
    def process(self, q: variable.Quantity) -> variable.Quantity:
        """Compute observer-specific updates to a variable quantity."""
        if any(alias in self.coordinates for alias in q.name):
            # This is an axis-reference quantity.
            return self._subscript(q)
        needed = self._compute_coordinates(q)
        if not needed:
            # There are no axes over which to interpolate.
            return self._subscript(q)
        new = self._interpolate(q, needed)
        # We only want to subscript the uninterpolated axes.
        interpolated = [
            'shell' if d == 'radius' else d
            for d in needed
        ]
        axes = list(set(q.axes) - set(interpolated))
        return self._subscript(new, *axes)

    def _build_coordinates(self):
        base = super()._build_coordinates()
        grid = {
            (k, *self.interface.variables.alias(k, include=True)):
            self.interface.variables[k]
            for k in {'radius', 'theta', 'phi'}
        }
        return {**base, **grid}

    def _compute_coordinates(self, q: variable.Quantity):
        base = super()._compute_coordinates(q)
        for key in reference.ALIASES['radius']:
            if values := self.get_value(key):
                try:
                    iter(values)
                except TypeError:
                    floats = [float(values)]
                else:
                    floats = [float(value) for value in values]
                base['radius'] = {
                    'targets': numpy.array(floats),
                    'reference': self.coordinates['radius'],
                }
        return base

    def _compute_dimensions(self, q: variable.Quantity):
        base = super()._compute_dimensions(q)
        if any(r in self for r in reference.ALIASES['radius']):
            # See note at `observing.Context.__contains__`.
            base.append('radius')
        return base

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
        self._dataset = None
        self.readfrom(source, config=config)

    def readfrom(
        self: Instance,
        source: iotools.PathLike,
        config: iotools.PathLike=None
    ) -> Instance:
        # Use arguments to update paths.
        datapath = self._build_datapath(source)
        confpath = self._build_confpath(config, directory=datapath.parent)
        # Update the internal data interface.
        dataset = datafile.Interface(datapath)
        axes = Axes(dataset)
        variables = variable.Interface(dataset)
        constants = runtime.Arguments(
            source_path=ENV['src'],
            config_path=confpath,
        )
        self._data = observing.Interface(
            axes,
            variables,
            constants,
            system=self.system(),
        )
        # Update other attributes if everything succeeded.
        self._dataset = dataset
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
    def confpath(self) -> iotools.ReadOnlyPath:
        """The full path to this observer's runtime parameter file."""
        return self._confpath

    @property
    def datapath(self) -> iotools.ReadOnlyPath:
        """The path to this observer's dataset."""
        return self._source

    @property
    def dataset(self) -> datafile.Interface:
        """This observer's original dataset."""
        return self._dataset

    def process(self, old: variable.Quantity) -> variable.Quantity:
        if any(self._get_reference(alias) is not None for alias in old.name):
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

