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
    measurable,
    metadata,
    metric,
    numerical,
    observer,
    observing,
    physical,
    reference,
    variable,
)
from .runtime import BaseTypesH


ENV = Environment('eprem')


basetypes = BaseTypesH(source=ENV['src'])


class Axes(axis.Interface):
    """Interface to EPREM axis-indexing objects."""

    def __init__(
        self,
        data: datafile.Interface,
        system: typing.Union[str, metric.System]=None,
    ) -> None:
        super().__init__(data, system)
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
        def method(targets, unit: typing.Union[str, metadata.Unit]):
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
            converted = this[unit]
            if array.unit | converted.unit: # Could also use try/except
                array = array[converted.unit]
            values = numpy.array(array)
            indices = [
                numerical.find_nearest(this, float(value)).index
                for value in values
            ]
            return axis.Data(indices, values=values)
        return method


class Application(observing.Application):
    """The EPREM-specific observing context."""

    def interpolate(
        self,
        q: variable.Quantity,
        coordinates: typing.Dict[str, typing.Dict[str, typing.Any]],
    ) -> variable.Quantity:
        base = super().interpolate(q, coordinates)
        # We only want to subscript the uninterpolated axes.
        interpolated = [
            'shell' if d == 'radius' else d
            for d in coordinates
        ]
        axes = list(set(q.axes) - set(interpolated))
        return self._subscript(base, *axes)

    def _build_coordinates(self):
        base = super()._build_coordinates()
        grid = {
            (k, *self.interface.variables.alias(k, include=True)):
            self.interface.variables[k]
            for k in {'radius', 'theta', 'phi'}
        }
        return {**base, **grid}

    _axes = {
        'time': 0,
        'energy': 1,
        'mu': 1,
    }

    def _compute_interpolants(self, q: variable.Quantity):
        base = {
            k: {**c, 'axis': self._axes.get(k)}
            for k, c in super()._compute_interpolants(q).items()
        }
        if 'shell' not in q.axes:
            # The rest of this method deals with radial interpolation, which
            # only applies when 'shell' is one of the target quantity's axes.
            return base
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
                    'axis': 1,
                }
        return base


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
        super().__init__(
            'preEruption', 'phiOffset',
            system=system,
            apply=Application,
        )
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
        axes = Axes(dataset, system=self.system())
        variables = variable.Interface(dataset, system=self.system())
        constants = runtime.Arguments(
            source_path=ENV['src'],
            config_path=confpath,
        )
        self._data = observing.Interface(axes, variables, constants)
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

