import abc
import collections
import collections.abc
import contextlib
import numbers
import operator as standard
import typing

import numpy

from goats.core import aliased
from goats.core import axis
from goats.core import computed
from goats.core import constant
from goats.core import datafile
from goats.core import interpolation
from goats.core import iterables
from goats.core import metadata
from goats.core import metric
from goats.core import observed
from goats.core import physical
from goats.core import reference
from goats.core import symbolic
from goats.core import variable


T = typing.TypeVar('T')


class Parameters(collections.abc.Sequence, iterables.ReprStrMixin):
    """The parameters of an `~observing.Quantity`."""

    def __init__(self, *names: str) -> None:
        self._names = self._init(*names)

    def _init(self, *args):
        names = iterables.unique(*iterables.unwrap(args, wrap=tuple))
        if not names or all(isinstance(name, str) for name in names):
            return names
        raise TypeError(
            f"Can't initialize instance of {type(self)}"
            f" with {names!r}"
        )

    __abs__ = metadata.identity(abs)
    """Called for abs(self)."""
    __pos__ = metadata.identity(standard.pos)
    """Called for +self."""
    __neg__ = metadata.identity(standard.neg)
    """Called for -self."""

    __add__ = metadata.identity(standard.add)
    """Called for self + other."""
    __sub__ = metadata.identity(standard.sub)
    """Called for self - other."""

    def merge(a, *others):
        """Return the unique axis names in order."""
        names = list(a._names)
        for b in others:
            if isinstance(b, Parameters):
                names.extend(b._names)
        return Parameters(*set(names))

    __mul__ = merge
    """Called for self * other."""
    __rmul__ = merge
    """Called for other * self."""
    __truediv__ = merge
    """Called for self / other."""

    def __pow__(self, other):
        """Called for self ** other."""
        if isinstance(other, numbers.Real):
            return self
        return NotImplemented

    def __eq__(self, other):
        """True if self and other represent the same axes."""
        return (
            isinstance(other, Parameters) and other._names == self._names
            or (
                isinstance(other, str)
                and len(self) == 1
                and other == self._names[0]
            )
            or (
                isinstance(other, typing.Iterable)
                and len(other) == len(self)
                and all(i in self for i in other)
            )
        )

    def __hash__(self):
        """Support use as a mapping key."""
        return hash(self._names)

    def __len__(self) -> int:
        """Called for len(self)."""
        return len(self._names)

    def __getitem__(self, __i: typing.SupportsIndex):
        """Called for index-based access."""
        return self._names[__i]

    def __str__(self) -> str:
        return f"[{', '.join(repr(name) for name in self._names)}]"


class Quantity(variable.Quantity):
    """A quantity with one or more name(s), a unit, and axes."""

    _parameters: Parameters=None

    def __init__(self, __data, **meta) -> None:
        super().__init__(__data, **meta)
        parsed = self.parse_attrs(__data, meta, parameters=())
        self._parameters = Parameters(parsed['parameters'])
        self.meta.register('parameters')
        self.display.register('parameters')
        self.display['__str__'].append("parameters={parameters}")
        self.display['__repr__'].append("parameters={parameters}")

    def parse_attrs(self, this, meta: dict, **targets):
        if (
            isinstance(this, variable.Quantity)
            and not isinstance(this, Quantity)
        ): # barf
            meta.update({k: getattr(this, k) for k in ('unit', 'name', 'axes')})
            this = this.data
        return super().parse_attrs(this, meta, **targets)

    @property
    def parameters(self):
        """The optional parameters that define this observing quantity."""
        return Parameters(self._parameters)


class Dataset(abc.ABC, typing.Generic[T]):
    """Abstract base class for observing-related data objects."""

    def __init__(self, __type: typing.Type[T], source) -> None:
        self._type = __type
        self._data = self._type(source)

    @abc.abstractmethod
    def get_axes(self, system: str=None) -> axis.Interface:
        """Get the available axis-managing objects."""
        return axis.Interface(self.data, system=system)

    @abc.abstractmethod
    def get_variables(self, system: str=None) -> variable.Interface:
        """Get the available variable quantities."""
        return variable.Interface(self.data, system=system)

    @abc.abstractmethod
    def get_constants(self) -> constant.Interface:
        """Get the available constant quantities."""
        return constant.Interface()

    @property
    def data(self):
        """The object that manages access to this dataset's data."""
        return self._data

    def readfrom(self, source):
        """Update the data source."""
        self._data = self._type(source)
        return self


class Quantities(collections.abc.Collection):
    """An interface to quantities required to make observations."""

    def __init__(self, dataset: Dataset, system: str=None) -> None:
        self._dataset = dataset
        self._system = metric.System(system or 'mks')
        self._axes = None
        self._variables = None
        self._constants = None
        self._names = None
        self._primary = None
        self._derived = None
        self._functions = None
        self._assumptions = None

    def __contains__(self, __x) -> bool:
        return __x in self.names or any(__x in name for name in self.names)

    def __iter__(self) -> typing.Iterator:
        return iter(self.names)

    def __len__(self) -> int:
        return len(self.names)

    @property
    def axes(self):
        """The available axis-managing objects."""
        if self._axes is None:
            self._axes = self._dataset.get_axes(self.system)
        return self._axes

    @property
    def variables(self):
        """The available variable quantities."""
        if self._variables is None:
            self._variables = self._dataset.get_variables(self.system)
        return self._variables

    @property
    def assumptions(self):
        """The default physical assumptions."""
        if self._assumptions is None:
            assumptions = {
                k: v
                for k, v in self.constants.items(aliased=True)
                if isinstance(v, constant.Assumption)
            } if self.constants else {}
            self._assumptions = aliased.Mapping(assumptions)
        return self._assumptions

    @property
    def constants(self):
        """The available constant quantities."""
        if self._constants is None:
            self._constants = self._dataset.get_constants()
        return self._constants

    @property
    def system(self):
        """The metric system to use for observations."""
        return self._system

    @property
    def names(self):
        """The names of all available quantities."""
        if self._names is None:
            self._names = self.primary + self.derived
        return self._names

    @property
    def primary(self):
        """The names of all available primary quantities.
        
        A primary quantity is a quantity that comes directly from the dataset.
        """
        if self._primary is None:
            self._primary = tuple(self.variables.keys(aliased=True))
        return self._primary

    @property
    def derived(self):
        """The names of all available quantities.
        
        A derived quantity is a quantity that is the result of a defined
        function of one or more primary or derived quantities, and zero or more
        physical assumptions.
        """
        if self._derived is None:
            self._derived = tuple(self.functions.keys(aliased=True))
        return self._derived

    @property
    def functions(self):
        """The computable quantities available to this observer."""
        if self._functions is None:
            self._functions = computed.Interface(self.axes, self.variables)
        return self._functions

    def get_observable(self, key: str):
        """Get the observable quantity corresponding to `key`."""
        for this in (self.variables, self.functions):
            if key in this:
                return this[key]

    # TODO: Refactor `get_unit` and `get_axes` to reduce overlap.

    def get_unit(self, key: str):
        """Determine the unit corresponding to `key`."""
        if key in self.variables:
            return self.variables[key].unit
        if key in self.functions:
            return self.functions[key].unit
        s = str(key)
        expression = symbolic.Expression(reference.NAMES.get(s, s))
        term = expression[0]
        this = self.get_unit(term.base) ** term.exponent
        if len(expression) > 1:
            for term in expression[1:]:
                this *= self.get_unit(term.base) ** term.exponent
        return metadata.Unit(this)

    def get_axes(self, key: str):
        """Determine the axes corresponding to `key`."""
        if key in self.variables:
            return self.variables[key].axes
        if key in self.functions:
            return self.functions[key].axes
        s = str(key)
        expression = symbolic.Expression(reference.NAMES.get(s, s))
        term = expression[0]
        this = self.get_axes(term.base)
        if len(expression) > 1:
            for term in expression[1:]:
                this *= self.get_axes(term.base)
        return metadata.Axes(this)

    def compute_index(self, key: str, **constraints) -> axis.Quantity:
        """Compute the axis-indexing object for `key`."""
        if key not in self.axes:
            raise ValueError(f"No axis corresponding to {key!r}") from None
        if key not in constraints:
            return self.axes[key].index()
        this = constraints[key]
        if isinstance(this, axis.Quantity):
            return this
        return self.axes[key].index(*iterables.whole(this))

    def compute_value(self, key: str, **constraints) -> physical.Scalar:
        """Create a parameter value for `key`."""
        if key not in self.constants and key not in constraints:
            raise ValueError(
                f"No parameter corresponding to {key!r}"
            ) from None
        if key in constraints:
            return self._compute_value(constraints[key])
        return self._compute_value(self.constants[key])

    def _compute_value(self, this):
        """Compute a parameter value."""
        # TODO: Generalize beyond scalar parameters.
        scalar = physical.scalar(this)
        unit = self.system.get_unit(unit=scalar.unit)
        return scalar[unit]


class Application:
    """A general observing application."""

    def __init__(self, interface: Quantities) -> None:
        self.interface = interface
        """The interface to an observer's dataset."""
        self.user = None
        """The current set of user constraints."""
        self._coordinates = None
        self._cache = {}

    def __contains__(self, __k: str):
        """True if the named constraint affects this application."""
        # Should this check all dimensions and parameters or only user
        # constraints?
        return __k in self.user

    def apply(self, **constraints):
        """Update the observing constraints.
        
        This method will apply the given constraints when computing axis indices
        and parameter values within the context of this application. It will
        also clear the corresponding items from the instance cache.
        """
        if self.user is None:
            self.user = {}
        for key in constraints:
            for subset in self._cache.values():
                if key in subset:
                    del subset[key]
        self.user.update(constraints)
        return self

    def reset(self, *keys: str):
        """Reset constraints to their default values.
        
        Parameters
        ----------
        *keys : string
            Zero or more names of constraints to reset. This method will also
            remove all named items from the instance cache By default, this
            method will reset all constraints and clear the cache.
        """
        if not keys:
            self.user = {}
            self._cache = {}
        else:
            subsets = self._cache.values()
            for key in keys:
                if key in self.user:
                    del self.user[key]
                for subset in subsets:
                    if key in subset:
                        del subset[key]
        return self

    def observe(self, key: str) -> Quantity:
        """Create an observation within the context of this application."""
        result = self._observe(key)
        if any(alias in self.coordinates for alias in result.name):
            # This is an axis-reference quantity.
            return self._subscript(result)
        needed = self._compute_interpolants(result)
        if not needed:
            # There are no axes over which to interpolate.
            return self._subscript(result)
        return self.interpolate(result, needed)

    def _observe(self, key: str) -> Quantity:
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

    def evaluate(self, q) -> Quantity:
        """Create an observing quantity based on the given quantity."""
        if isinstance(q, computed.Quantity):
            parameters = [
                parameter for parameter in q.parameters
                if parameter in self.interface.constants
            ]
            return Quantity(self.compute(q), parameters=parameters)
        if isinstance(q, variable.Quantity):
            return Quantity(self.process(q))
        raise ValueError(f"Unknown quantity: {q!r}") from None

    def process(self, q: variable.Quantity) -> variable.Quantity:
        """Compute observer-specific updates to a variable quantity.
        
        The default implementation immediately returns `q`.
        """
        return q

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
        if quantity := self.interface.get_observable(key):
            return self.evaluate(quantity)

    def _subscript(self, q: variable.Quantity, *axes: str):
        """Extract a subset of this quantity."""
        if not axes:
            return q[tuple(self.get_index(a, slice(None)) for a in q.axes)]
        indices = [
            self.get_index(a, slice(None))
            if a in axes else slice(None)
            for a in q.axes
        ]
        return q[tuple(indices)]

    def _compute_interpolants(self, q: variable.Quantity):
        """Determine the coordinate axes over which to interpolate."""
        coordinates = {}
        for a in q.axes:
            idx = self.get_index(a)
            if idx and idx.unit is not None:
                contained = [
                    self.coordinates[a].array_contains(target)
                    for target in idx.values
                ]
                if not numpy.all(contained):
                    coordinates[a] = {
                        'targets': numpy.array(idx.values),
                        'reference': self.coordinates[a],
                    }
        return coordinates

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
        return type(q)(array, **meta)

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
        indices = (q.axes.index(d) for d in reference.axes)
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
    def coordinates(self):
        """The reference quantities for dataset coordinate axes."""
        if self._coordinates is None:
            coordinates = self._build_coordinates()
            self._coordinates = aliased.Mapping(coordinates)
        return self._coordinates

    def _build_coordinates(self):
        """Helper for `~coordinates` property. Extracted for overloading."""
        return {
            k: self.interface.variables.get(k)
            for k in self.interface.axes.keys(aliased=True)
        }

    def get_index(self, key: str, default: T=None) -> axis.Quantity:
        """Get the axis-indexing object for `key`."""
        if 'indices' not in self._cache:
            self._cache['indices'] = {}
        if key in self._cache['indices']:
            return self._cache['indices'][key]
        with contextlib.suppress(ValueError):
            idx = self.interface.compute_index(key, **self.user)
            self._cache['indices'][key] = idx
            return idx
        return default

    def get_value(self, key: str, default: T=None) -> physical.Scalar:
        """Get the parameter value correpsonding to `key`."""
        if 'values' not in self._cache:
            self._cache['values'] = {}
        if key in self._cache['values']:
            return self._cache['values'][key]
        with contextlib.suppress(ValueError):
            val = self.interface.compute_value(key, **self.user)
            self._cache['values'][key] = val
            return val
        return default


class Implementation:
    """The base observing implementation."""

    def __init__(
        self,
        name: str,
        interface: Quantities,
    ) -> None:
        """
        Initialize this instance.

        Parameters
        ----------
        name : string
            The name of the quantity to observe.

        interface : `~Interface`
            The interface to all observing-related quantities.
        """
        self.name = name
        self.interface = interface
        self._cache = {}

    # TODO:
    # - refactor `get_unit` and `get_axes` to reduce overlap.
    # - redefine `_cache` as an aliased mapping.

    def get_unit(
        self,
        name: typing.Union[str, typing.Iterable[str], metadata.Name],
    ) -> metadata.Unit:
        """Return the unit corresponding to `name`."""
        if 'unit' not in self._cache:
            self._cache['unit'] = {}
        if name in self._cache['unit']:
            return self._cache['unit'][name]
        unit = self._lookup_unit(name)
        self._cache['unit'][name] = unit
        return unit

    def _lookup_unit(
        self,
        name: typing.Union[str, typing.Iterable[str], metadata.Name],
    ) -> metadata.Unit:
        """Determine the unit corresponding to `name`."""
        if isinstance(name, str):
            return self.interface.get_unit(name)
        unit = (self.interface.get_unit(key) for key in name)
        return next(unit, None)

    def get_axes(
        self,
        name: typing.Union[str, typing.Iterable[str], metadata.Name],
    ) -> metadata.Axes:
        """Return the axes corresponding to `name`."""
        if 'axes' not in self._cache:
            self._cache['axes'] = {}
        if name in self._cache['axes']:
            return self._cache['axes'][name]
        axes = self._lookup_axes(name)
        self._cache['axes'][name] = axes
        return axes

    def _lookup_axes(
        self,
        name: typing.Union[str, typing.Iterable[str], metadata.Name],
    ) -> metadata.Unit:
        """Determine the axes corresponding to `name`."""
        if isinstance(name, str):
            return self.interface.get_axes(name)
        axes = (self.interface.get_axes(key) for key in name)
        return next(axes, None)

    def apply(
        self,
        application: Application,
        unit: metadata.UnitLike=None,
    ) -> observed.Quantity:
        """Apply an observing context to the target quantity."""
        result = application.observe(self.name)
        indices = {k: application.get_index(k) for k in result.axes}
        scalars = {k: application.get_value(k) for k in result.parameters}
        context = observed.Context(indices, scalars=scalars)
        if unit:
            return observed.Quantity(result[unit], context)
        return observed.Quantity(result, context)
