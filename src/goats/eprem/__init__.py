import contextlib
import pathlib
import typing

import numpy
import numpy.typing

from ..core import (
    aliased,
    axis,
    computed,
    constant,
    datafile,
    fundamental,
    iotools,
    index,
    interpolation,
    iterables,
    measurable,
    metadata,
    metric,
    numerical,
    observed,
    observer,
    observing,
    physical,
    reference,
    symbolic,
    variable,
)
from .runtime import BaseTypesH


basetypes = BaseTypesH()


T = typing.TypeVar('T')


class Variables(variable.Interface):
    """An interface to EPREM variables."""

    def __init__(
        self,
        __data: datafile.Interface,
        system: typing.Union[str, metric.System]=None,
    ) -> None:
        super().__init__(__data, system=system)

    def build(self, __v: datafile.Variable):
        name = __v.name
        v = variable.Quantity(
            __v.data,
            dimensions=__v.dimensions,
            unit=standardize(__v.unit),
        )
        return (
            v[self.system['mass'].unit]
            if name in {'mass', 'm'}
            else v[str(self.system)]
        )


_UNIT_MAP = {
    'julian date': 'day',
    'shell': '1',
    'cos(mu)': '1',
    'e-': 'e',
    '# / cm^2 s sr MeV': '# / cm^2 s sr MeV/nuc',
}
"""Substitutions for non-standard EPREM units.

This dictionary maps each non-standard unit associated with an EPREM output
quantity to a standard unit, as defined in `~core.metric.py`.
"""


def standardize(unit: T):
    """Replace this unit string with a standard unit string, if possible.

    This function looks for `unit` in the known conversions and returns the
    standard unit string if it exists. If this doesn't find a standard unit
    string, it just returns the input.
    """
    unit = _UNIT_MAP.get(str(unit), unit)
    if '/' in unit:
        num, den = str(unit).split('/', maxsplit=1)
        unit = ' / '.join((num.strip(), f"({den.strip()})"))
    return unit


class Axes(aliased.Mapping):
    """Interface to EPREM axis-indexing objects."""

    def __init__(
        self,
        __data: datafile.Interface,
        system: typing.Union[str, metric.System]=None,
    ) -> None:
        """Initialize this instance."""
        self._system = metric.System(system or 'mks')
        self._variables = Variables(__data, system=system)
        self._names = __data.available('axes').canonical
        self._time = None
        self._shell = None
        self._species = None
        self._energy = None
        self._mu = None
        indexers = {
            'time': self.time,
            'shell': self.shell,
            'species': self.species,
            'energy': self.energy,
            'mu': self.mu,
        }
        super().__init__(indexers, aliases=reference.ALIASES)

    def __getitem__(self, __k: str):
        """Get the named axis object, if possible."""
        if __k in self.keys():
            method = super().__getitem__(__k)
            try:
                unit = self.variables[__k].unit
            except KeyError:
                unit = None
            return axis.Quantity(method, unit=unit)
        raise KeyError(
            f"No known indexing method for {__k}"
        ) from None

    def resolve(
        self,
        names: typing.Iterable[str],
        mode: str='strict',
    ) -> typing.Tuple[str]:
        """Compute and order the available dimensions in `names`."""
        ordered = tuple(name for name in self._names if name in names)
        if mode == 'strict':
            return ordered
        extra = tuple(name for name in names if name not in ordered)
        if not extra:
            return ordered
        if mode == 'append':
            return ordered + extra
        raise ValueError(f"Unrecognized mode {mode!r}")

    @property
    def variables(self):
        """The variable quantities that these axes support."""
        return self._variables

    @property
    def system(self):
        """The metric system of these axes."""
        return self._system

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
            try:
                # Versions of EPREM with logically 2D egrid.
                size = this.shape[1]
            except IndexError:
                # Versions of EPREM with truly 1D egrid.
                size = this.shape[0]
            self._energy = axis.Indexer(method, size)
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
                array = physical.Array(measured.data, unit=measured.unit)
            else:
                # If the measured unit is dimensionless, it could be because the
                # targets truly are dimensionless or because the user wants to
                # use the default unit. Since we have no choice but to assume
                # that the calling object (probably an instance of
                # `core.axis.Quantity`) passed an appropriate default unit,
                # which may be dimensionless, the default unit is the
                # appropriate unit for both cases.
                array = physical.Array(measured.data, unit=unit)
            if array.unit | converted.unit: # Could also use try/except
                array = array[converted.unit]
            values = numpy.array(array)
            indices = [
                numerical.find_nearest(converted, float(value)).index
                for value in values
            ]
            return index.Data(indices, values=values)
        return method


class Functions(aliased.Mapping):
    """An interface to computable quantities."""

    def __init__(
        self,
        axes: Axes,
        variables: Variables,
    ) -> None:
        super().__init__(computed.registry, aliases=reference.ALIASES)
        self.axes = axes
        self.variables = variables
        self._cache = {}

    def __getitem__(self, __k: str) -> computed.Quantity:
        if 'quantity' not in self._cache:
            self._cache['quantity'] = {}
        if __k in self._cache['quantity']:
            return self._cache['quantity'][__k]
        quantity = computed.Quantity(
            self.get_method(__k),
            dimensions=self.get_dimensions(__k),
            unit=self.get_unit(__k),
        )
        self._cache['quantity'][__k] = quantity
        return quantity

    def get_method(self, key: str):
        """Get a `~computed.Method` for `key`."""
        try:
            this = super().__getitem__(key).copy()
        except KeyError:
            return None
        else:
            method = this.pop('method')
            return computed.Method(method, **this)

    def get_name(self, key: str):
        """Get the set of aliases for `key`."""
        return self.alias(key, include=True)

    def get_unit(self, key: str):
        """Determine the unit of `key` based on its metric quantity."""
        this = reference.METADATA.get(key, {}).get('quantity')
        return self.variables.system.get_unit(quantity=this)

    def get_dimensions(self, key: str):
        """Compute appropriate axis names for `key`."""
        if 'dimensions' not in self._cache:
            self._cache['dimensions'] = {}
        if key in self._cache['dimensions']:
            return self._cache['dimensions'][key]
        method = self.get_method(key)
        self._removed = self._get_metadata(method, 'removed')
        self._added = self._get_metadata(method, 'added')
        self._accumulated = []
        dimensions = self._gather_dimensions(method)
        self._cache['dimensions'][key] = dimensions
        return dimensions

    def _gather_dimensions(self, target: computed.Method):
        """Recursively gather appropriate axes."""
        for parameter in target.parameters:
            if parameter in self.variables:
                self._accumulated.extend(self.variables[parameter].dimensions)
            elif method := self.get_method(parameter):
                self._removed.extend(self._get_metadata(method, 'removed'))
                self._added.extend(self._get_metadata(method, 'added'))
                self._accumulated.extend(self._gather_dimensions(method))
        unique = set(self._accumulated) - set(self._removed) | set(self._added)
        return self.axes.resolve(unique, mode='append')

    def _get_metadata(self, method: computed.Method, key: str) -> list:
        """Helper for accessing a method's metadata dictionary."""
        value = method.get(key)
        return list(iterables.whole(value)) if value else []


class ContextKeyError(Exception):
    """Can't find a parameter value within this context."""


Instance = typing.TypeVar('Instance', bound='Context')


class Context(observing.Context):
    """The EPREM observing context."""

    def __init__(
        self,
        variables: Variables,
        axes: Axes,
        constants: runtime.Interface,
    ) -> None:
        """Initialize this instance.
        
        Parameters
        ----------
        system : string or `~metric.System`
            The metric system to use.
        """
        functions = Functions(axes, variables)
        super().__init__(variables, functions, constraints=constants)
        self._constants = constants
        self._variables = variables
        self._axes = axes
        self._functions = functions
        self._system = variables.system
        self._parameters = None
        self._coordinates = None

    def observe(self, name: str) -> observing.Observation:
        result = self._observe(name)
        return observing.Observation(
            result,
            indices={k: self.get_index(k) for k in result.dimensions},
            assumptions={k: self.get_value(k) for k in result.parameters},
        )

    def _observe(self, key: str) -> observing.Quantity:
        """Create an observation within the context of this application."""
        result = self._build(key)
        if any(k in self.coordinates for k in reference.ALIASES.find(key, [])):
            # This is an axis-reference quantity.
            return self._subscript(result)
        needed = self._compute_interpolants(result)
        if not needed:
            # There are no axes over which to interpolate.
            return self._subscript(result)
        return self.interpolate(result, needed)

    def _build(self, key: str) -> observing.Quantity:
        """Internal observing logic."""
        expression = symbolic.Expression(reference.NAMES.get(key, key))
        term = expression[0]
        result = self.get_observable(term.base)
        if len(expression) == 1:
            # We don't need to multiply or divide quantities.
            if term.exponent == 1:
                # We don't even need to raise this quantity to a power.
                return result
            return result ** term.exponent
        q0 = result ** term.exponent
        if len(expression) > 1:
            for term in expression[1:]:
                result = self.get_observable(term.base)
                q0 *= result ** term.exponent
        return q0

    def compute(self, q: computed.Quantity) -> variable.Quantity:
        """Determine dependencies and compute the result of this function."""
        dependencies = {p: self.get_dependency(p) for p in q.parameters}
        return q(**dependencies)

    def get_dependency(self, key: str):
        """Get the named constant or variable quantity."""
        if this := self.get_observable(key):
            return this
        return self.get_value(key)

    def get_observable(self, key: str):
        """Retrieve and evaluate an observable quantity."""
        quantity = self.get_quantity(key)
        if isinstance(quantity, computed.Quantity):
            data = self.compute(quantity)
            parameters = [
                parameter for parameter in quantity.parameters
                if parameter in self.parameters
            ]
            return observing.Quantity(data, parameters=parameters)
        if isinstance(quantity, variable.Quantity):
            return observing.Quantity(quantity)

    def _subscript(self, q: variable.Quantity, *dimensions: str):
        """Extract a subset of this quantity."""
        indices = [
            self.get_index(dimension, slice(None))
            if dimension in dimensions else slice(None)
            for dimension in q.dimensions
        ] if dimensions else [
            self.get_index(dimensions, slice(None))
            for dimensions in q.dimensions
        ]
        return q[tuple(indices)]

    _axis_indices = {
        'time': 0,
        'energy': 1,
        'mu': 1,
    }

    def _compute_interpolants(self, q: variable.Quantity):
        """Determine the coordinate axes over which to interpolate."""
        coordinates = {}
        for dimension in q.dimensions:
            idx = self.get_index(dimension)
            if idx and idx.unit is not None:
                contained = [
                    self.coordinates[dimension].array_contains(target)
                    for target in idx.values
                ]
                if not numpy.all(contained):
                    coordinates[dimension] = {
                        'targets': numpy.array(idx.values),
                        'reference': self.coordinates[dimension],
                    }
        interpolants = {
            k: {**c, 'axis': self._axis_indices.get(k)}
            for k, c in coordinates.items()
        }
        if 'shell' not in q.dimensions:
            # The rest of this method deals with radial interpolation, which
            # only applies when 'shell' is one of the target quantity's
            # dimensions.
            return interpolants
        for key in reference.ALIASES.find('radius'):
            if values := self.get_value(key):
                try:
                    iter(values)
                except TypeError:
                    floats = [float(values)]
                else:
                    floats = [float(value) for value in values]
                interpolants['radius'] = {
                    'targets': numpy.array(floats),
                    'reference': self.coordinates['radius'],
                    'axis': 1,
                }
        return interpolants

    def interpolate(
        self,
        q: variable.Quantity,
        coordinates: typing.Dict[str, typing.Dict[str, typing.Any]],
    ) -> variable.Quantity:
        """Internal interpolation logic."""
        array = None
        for coordinate in coordinates.values():
            array = self._interpolate_coordinate(
                q,
                coordinate['targets'],
                coordinate['reference'],
                axis=coordinate.get('axis'),
                workspace=array,
            )
        meta = {k: getattr(q, k, None) for k in q.meta.parameters}
        base = type(q)(array, **meta)
        interpolated = [
            # We only want to subscript the uninterpolated axes.
            'shell' if d == 'radius' else d
            for d in coordinates
        ]
        dimensions = list(set(q.dimensions) - set(interpolated))
        return self._subscript(base, *dimensions)

    def _interpolate_coordinate(
        self,
        q: variable.Quantity,
        targets: numpy.ndarray,
        reference: variable.Quantity,
        axis: int=None,
        workspace: numpy.ndarray=None,
    ) -> numpy.ndarray:
        """Interpolate a variable array based on a known coordinate."""
        array = numpy.array(q) if workspace is None else workspace
        indices = (q.dimensions.index(d) for d in reference.dimensions)
        dst, src = zip(*enumerate(indices))
        reordered = numpy.moveaxis(array, src, dst)
        interpolated = interpolation.apply(
            reordered,
            numpy.array(reference),
            targets,
            axis=axis,
        )
        return numpy.moveaxis(interpolated, dst, src)

    @property
    def coordinates(self) -> typing.Mapping[str, variable.Quantity]:
        """The reference quantities for dataset coordinate axes."""
        if self._coordinates is None:
            base = {
                k: self.variables.get(k)
                for k in self.axes.keys(aliased=True)
            }
            grid = {
                (k, *self.variables.alias(k, include=True)):
                self.variables[k]
                for k in {'radius', 'theta', 'phi'}
            }
            self._coordinates = aliased.Mapping({**base, **grid})
        return self._coordinates

    def get_index(self, key: str, default: T=None) -> index.Quantity:
        """Get the axis-indexing object for `key`."""
        if 'indices' not in self._cache:
            self._cache['indices'] = {}
        if key in self._cache['indices']:
            return self._cache['indices'][key]
        with contextlib.suppress(ContextKeyError):
            idx = self.compute_index(key)
            self._cache['indices'][key] = idx
            return idx
        return default

    def compute_index(self, key: str) -> index.Quantity:
        """Compute the axis-indexing object for `key`."""
        if key not in self.axes:
            raise ContextKeyError(f"No axis corresponding to {key!r}") from None
        if key not in self.constraints:
            return self.axes[key].index()
        this = self.constraints[key]
        if isinstance(this, index.Quantity):
            return this
        return self.axes[key].index(*iterables.whole(this))

    def get_value(self, key: str, default: T=None) -> physical.Scalar:
        """Get the parameter value correpsonding to `key`."""
        if 'values' not in self._cache:
            self._cache['values'] = {}
        if key in self._cache['values']:
            return self._cache['values'][key]
        with contextlib.suppress(ContextKeyError):
            val = self.compute_value(key)
            self._cache['values'][key] = val
            return val
        return default

    def compute_value(self, key: str) -> physical.Scalar:
        """Create a parameter value for `key`."""
        if key not in self.constants and key not in self.constraints:
            raise ContextKeyError(
                f"No parameter corresponding to {key!r}"
            ) from None
        if key in self.constraints:
            return self._compute_value(self.constraints[key])
        return self._compute_value(self.constants[key])

    def _compute_value(self, this):
        """Compute a parameter value."""
        # TODO: Generalize beyond scalar parameters.
        scalar = physical.scalar(this)
        unit = self._system.get_unit(unit=scalar.unit)
        return scalar[unit]

    @property
    def parameters(self):
        """The names of available physical constants."""
        if self._parameters is None:
            self._parameters = tuple(self.constants)
        return self._parameters

    @property
    def variables(self):
        """The available array-like quantities."""
        return self._variables

    @property
    def axes(self):
        """The available axis-indexing quantities."""
        return self._axes

    @property
    def functions(self):
        """The available callable quantities."""
        return self._functions

    @property
    def constants(self):
        """The available constant quantities."""
        return self._constants


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
        config: iotools.PathLike=None,
        source: iotools.PathLike=None,
        system: str=None,
    ) -> None:
        self._id = __id
        self._source = source or pathlib.Path.cwd()
        self._config = config
        self._system = system or 'mks'
        self._datapath = None
        self._confpath = None
        context = self._build_context(system=system)
        if context:
            self._variables = context.variables
            self._axes = context.axes
        super().__init__(*self._unobservable, context=context)

    def update(
        self: Instance,
        source: iotools.PathLike=None,
        config: iotools.PathLike=None,
    ) -> Instance:
        """Change the data source or config-file path.
        
        Parameters
        ----------
        source : string or `~pathlib.Path`, optional
            A new directory in which to search for this observer's data.

        config : string or `~pathlib.Path`, optional
            The path to a new EPREM configuration file from which to extract
            simulation runtime parameter arguments.
        """
        context = self._build_context(source=source, config=config)
        if context:
            self._variables = context.variables
            self._axes = context.axes
            return super().update(context)
        return self

    def _build_context(self, source=None, config=None, system=None):
        """Create an instance of the EPREM observing context."""
        if config:
            self._config = config
            self._confpath = None
        if source:
            self._source = source
            self._datapath = None
        if self.datapath and self.confpath:
            dataset = datafile.Interface(self.datapath)
            variables = Variables(dataset, system=system or 'mks')
            axes = Axes(dataset, system=system or 'mks')
            constants = runtime.Interface(self.confpath)
            return Context(variables, axes, constants)

    @property
    def confpath(self) -> iotools.ReadOnlyPath:
        """The full path to this dataset's runtime parameter file."""
        if self._confpath is None:
            self._confpath = self._build_confpath()
        return self._confpath

    def _build_confpath(self):
        """Build the config-file path."""
        # If the user hasn't supplied a path, there's nothing to do.
        if self._config is None:
            return
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
        # We're out of options.
        raise ValueError(
            "Can't create path to configuration file"
            f" from {self._config!r}"
        ) from None

    @property
    def datapath(self) -> iotools.ReadOnlyPath:
        """The path to this dataset."""
        if self._datapath is None:
            self._datapath = self._build_datapath()
        return self._datapath

    def _build_datapath(self):
        """Build the path to this observer's dataset, if possible."""
        # If the user hasn't supplied a path, there's nothing to do.
        if self._source is None:
            return
        # Expand and resolve the current data source.
        this = iotools.ReadOnlyPath(self._source or '.')
        if not this.is_dir():
            # We can't create a valid path.
            raise TypeError(
                f"Can't create path to dataset from {this!r}"
            ) from None
        # The current data source is a directory.
        path = iotools.find_file_by_template(
            self._templates,
            self._id,
            directory=this,
        )
        with contextlib.suppress(TypeError):
            return iotools.ReadOnlyPath(path)

    @property
    def radius(self):
        """The time-dependent radius values in this observer's dataset."""
        return self._variables['radius']

    @property
    def theta(self):
        """The time-dependent theta values in this observer's dataset."""
        return self._variables['theta']

    @property
    def phi(self):
        """The time-dependent phi values in this observer's dataset."""
        return self._variables['phi']

    @property
    def time(self):
        """The time values in this observer's dataset."""
        return self._axes['time'].reference

    @property
    def shell(self):
        """The shell values in this observer's dataset."""
        return self._axes['shell'].reference

    @property
    def species(self):
        """The species values in this observer's dataset."""
        return self._axes['species'].reference

    @property
    def energy(self):
        """The energy values in this observer's dataset."""
        return self._axes['energy'].reference

    @property
    def mu(self):
        """The pitch-angle cosine values in this observer's dataset."""
        return self._axes['mu'].reference

    def __str__(self) -> str:
        return str(self._id)


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

