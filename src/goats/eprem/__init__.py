import contextlib
import pathlib
import typing

import numpy
import numpy.typing

from ..core import (
    axis,
    constant,
    datafile,
    fundamental,
    iotools,
    index,
    iterables,
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


basetypes = BaseTypesH()


T = typing.TypeVar('T')


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
                return index.Data(targets)
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
            ) -> index.Data:
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
                return index.Data(indices, values=symbols)
            self._species = axis.Indexer(method, len(symbols))
        return self._species

    def _build_coordinate(self, this: variable.Quantity):
        """Create coordinate-like axis data from the given variable."""
        def method(targets, unit: typing.Union[str, metadata.Unit]):
            # Convert the reference variable quantity to the default unit.
            converted = this[unit]
            if not targets:
                # If there are no target values, we assume the user wants the
                # entire axis.
                return index.Data(
                    range(len(converted)),
                    values=numpy.array(converted),
                )
            if all(isinstance(t, typing.SupportsIndex) for t in targets):
                # All the target values are already indices.
                return index.Data(targets, values=numpy.array(converted))
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
            if array.unit | converted.unit: # Could also use try/except
                array = array[converted.unit]
            values = numpy.array(array)
            indices = [
                numerical.find_nearest(converted, float(value)).index
                for value in values
            ]
            return index.Data(indices, values=values)
        return method


Instance = typing.TypeVar('Instance', bound='Dataset')


class Dataset(observing.Dataset):
    """Interface to an EPREM dataset."""

    def __init__(
        self,
        datapath: iotools.ReadOnlyPath,
        confpath: iotools.ReadOnlyPath,
    ) -> None:
        super().__init__(datafile.Interface, datapath)
        self._datapath = datapath
        self._confpath = confpath

    def get_axes(self, system: str=None) -> axis.Interface:
        return Axes(self.data, system=system)

    def get_variables(self, system: str=None) -> variable.Interface:
        return variable.Interface(self.data, system=system)

    def get_constants(self) -> constant.Interface:
        return runtime.Arguments(config_path=self.confpath)

    def readfrom(
        self: Instance,
        datapath: iotools.ReadOnlyPath,
        confpath: iotools.ReadOnlyPath=None,
    ) -> Instance:
        self._datapath = datapath
        if confpath:
            self._confpath = confpath
        return super().readfrom(datapath)

    @property
    def confpath(self) -> iotools.ReadOnlyPath:
        """The full path to this dataset's runtime parameter file."""
        return self._confpath

    @property
    def datapath(self) -> iotools.ReadOnlyPath:
        """The path to this dataset."""
        return self._datapath


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


Instance = typing.TypeVar('Instance', bound='Application')


class Application(observing.Application):
    """The EPREM observing application."""

    def __init__(
        self,
        __data: datafile.Interface,
        constraints: typing.Mapping=None
    ) -> None:
        quantities = observing.Interface()
        super().__init__(quantities, constraints)
        self._data = __data

    def apply(self, constraints: typing.Mapping):
        return type(self)(self._data, constraints)


Instance = typing.TypeVar('Instance', bound='Observer')


class Observer(observer.Interface, iterables.ReprStrMixin):
    """Base class for EPREM observers."""

    _templates: typing.Iterable[typing.Callable] = None

    _unobservable = [
        'preEruption',
        'phiOffset',
    ]

    def __init__(
        self,
        __id: int,
        source: iotools.PathLike=None,
        config: iotools.PathLike=None,
        system: str='mks',
    ) -> None:
        super().__init__(
            *self._unobservable,
            system=system,
        )
        self._id = __id
        self._source = source or pathlib.Path.cwd()
        self._config = config
        self._datapath = None
        self._confpath = None
        self._axes = None
        

    def update(
        self: Instance,
        source: iotools.PathLike=None,
        config: iotools.PathLike=None
    ) -> Instance:
        if config:
            self._config = config
            self._confpath = None
        if source:
            self._source = source
            self._datapath = None
        quantities = observing.Interface()
        application = Application(quantities)
        return super().update(application)

    def _build_application(self, datapath, confpath):
        """"""

    @property
    def confpath(self) -> iotools.ReadOnlyPath:
        """The full path to this dataset's runtime parameter file."""
        if self._confpath is None:
            # Compare the current config-file string to its filename.
            if pathlib.Path(self._config).name == self._config:
                # The current config-file string is just a filename.
                full = self.datapath.parent / self._config
                if full.exists():
                    # The config file exists in the data directory.
                    return full
            # Expand and resolve the current config-file path.
            this = iotools.ReadOnlyPath(self._config)
            if this.exists():
                # The absolute path exists.
                return this
            raise ValueError(
                "Can't create path to configuration file"
                f" from {self._config!r}"
            ) from None
        return self._confpath

    @property
    def datapath(self) -> iotools.ReadOnlyPath:
        """The path to this dataset."""
        if self._datapath is None:
            # Expand and resolve the current data source.
            this = iotools.ReadOnlyPath(self._source or '.')
            if this.is_dir():
                # The current data source is a directory.
                path = iotools.find_file_by_template(
                    self._templates,
                    self._id,
                    directory=this,
                )
                with contextlib.suppress(TypeError):
                    return iotools.ReadOnlyPath(path)
            # We couldn't create a valid data path for some reason.
            raise TypeError(
                f"Can't create path to dataset from {self._source!r}"
            ) from None
        return self._datapath

    def _build_datapath(self, directory: iotools.PathLike):
        """Create the path to the dataset from `directory`."""
        this = iotools.ReadOnlyPath(directory or pathlib.Path.cwd())
        if this.is_dir():
            path = iotools.find_file_by_template(
                self._templates,
                self._id,
                directory=this,
            )
            with contextlib.suppress(TypeError):
                # Couldn't initialize with `path` for some reason.
                return iotools.ReadOnlyPath(path)
        raise TypeError(
            f"Can't create path to dataset from {directory!r}"
        ) from None

    def _build_confpath(
        self,
        config: iotools.PathLike,
        directory: iotools.PathLike=None,
    ) -> iotools.ReadOnlyPath:
        """Create the path to the configuration file."""
        # Create a `Path` but don't expand or resolve it.
        path = pathlib.Path(config)
        if directory and path.name == config:
            # There is a directory to check and `config` is just a filename.
            full = iotools.ReadOnlyPath(directory) / config
            if full.exists():
                return full
        # Expand and resolve path.
        this = iotools.ReadOnlyPath(config)
        if this.exists():
            # The absolute path exists, so return it.
            return this
        raise ValueError(
            f"Can't create path to configuration file from {config!r}"
        ) from None

    @property
    def radius(self):
        """The time-dependent radius values in this observer's dataset."""
        return self.quantities.variables['radius']

    @property
    def theta(self):
        """The time-dependent theta values in this observer's dataset."""
        return self.quantities.variables['theta']

    @property
    def phi(self):
        """The time-dependent phi values in this observer's dataset."""
        return self.quantities.variables['phi']

    @property
    def time(self):
        """The time values in this observer's dataset."""
        return self.quantities.axes['time'].reference

    @property
    def shell(self):
        """The shell values in this observer's dataset."""
        return self.quantities.axes['shell'].reference

    @property
    def species(self):
        """The species values in this observer's dataset."""
        return self.quantities.axes['species'].reference

    @property
    def energy(self):
        """The energy values in this observer's dataset."""
        return self.quantities.axes['energy'].reference

    @property
    def mu(self):
        """The pitch-angle cosine values in this observer's dataset."""
        return self.quantities.axes['mu'].reference

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

