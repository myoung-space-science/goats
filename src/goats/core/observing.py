import abc
import collections
import collections.abc
import numbers
import operator as standard
import typing

import numpy

from goats.core import aliased
from goats.core import constant
from goats.core import index
from goats.core import iterables
from goats.core import metadata
from goats.core import metric
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


class Result:
    """A general observing result.
    
    An instance of this class represents the observation of a named observable
    quantity within a well-defined context.
    """

    def __init__(
        self,
        __data: variable.Quantity,
        indices: typing.Mapping[str, index.Quantity],
        assumptions: typing.Mapping[str, constant.Assumption]=None,
    ) -> None:
        self._data = __data
        self._indices = indices
        self._assumptions = assumptions
        self._array = None
        self._parameters = None

    @property
    def array(self):
        """The observed data array, with singular dimensions removed.
        
        This property primarily provides a shortcut for cases in which the
        result of observing an N-dimensional quantity is an effectively
        M-dimensional quantity, with M<N, but singular dimensions cause it to
        appear to have higher dimensionality.
        """
        if self._array is None:
            self._array = numpy.array(self.data).squeeze()
        return self._array

    @property
    def data(self):
        """The observed variable quantity.
        
        This property provides direct access to the variable-quantity interface,
        as well as to metadata properties of the observed quantity.
        """
        return self._data

    @property
    def parameters(self):
        """The physical parameters relevant to this observation."""
        if self._parameters is None:
            self._parameters = list(self._assumptions)
        return self._parameters

    def __getitem__(self, __x):
        """Get context items or update the unit.
        
        Parameters
        ----------
        __x : string
            If `__x` names a known array axis or physical assumption, return
            that quantity. If `__x` is a valid unit for this observed quantity,
            return a new instance with updated unit.
        """
        if not isinstance(__x, (str, aliased.MappingKey, metric.Unit)):
            raise TypeError(
                f"{__x!r} must name a context item or a unit."
                "Use the array property to access data values."
            ) from None
        if __x in self._indices:
            return self._indices[__x]
        if __x in self._assumptions:
            return self._assumptions[__x]
        return type(self)(self.data, self._indices, self._assumptions)

    def __eq__(self, __o) -> bool:
        """True if two instances have equivalent attributes."""
        if isinstance(__o, Quantity):
            return all(
                getattr(self, attr) == getattr(__o, attr)
                for attr in ('data', 'indices', 'assumptions')
            )
        return NotImplemented

    def __str__(self) -> str:
        """A simplified representation of this object."""
        attrs = [
            f"unit='{self.data.unit}'",
            f"dimensions={self.data.axes}",
            f"parameters={self.parameters}",
        ]
        return f"'{self.data.name}': {', '.join(attrs)}"


class Context(collections.abc.Collection, typing.Generic[T]):
    """ABC for observer-specific observing contexts.
    
    Concrete subclasses must overload the abstract method `observe`, which
    should take the string name of an observable quantity and return an instance
    of `~observing.Result`.
    """

    def __init__(
        self,
        *mappings: typing.Mapping[str, T],
        constraints: typing.Mapping=None,
    ) -> None:
        """Create a new instance.
        
        Parameters
        ----------
        *mappings : iterable of mappings
            Zero or more mappings from string to physical quantity

        constraints : mapping, optional
            User-provided observing constraints.
        """
        self._mappings = mappings
        self._constraints = dict(constraints or {})
        self._available = None

    @abc.abstractmethod
    def observe(self, name: str) -> Result:
        """Observe the named quantity."""

    def constrain(self, new: typing.Mapping, update: bool=False):
        """Update the observing constraints.
        
        Parameters
        ----------
        new : mapping
            A mapping from string name to constraint value.

        update : bool, default=false
            If true, update the current constraints from `new`.
            Otherwise, overwrite the current constraints with `new`.
        """
        if update:
            self._constraints.update(new)
        else:
            self._constraints = dict(new)
        return self

    @property
    def constraints(self):
        """The current set of observing constraints."""
        if self._constraints is None:
            self._constraints = {}
        if isinstance(self._constraints, dict):
            return self._constraints
        return dict(self._constraints)

    def __contains__(self, __x: str) -> bool:
        """True if `__x` is an available physical quantity."""
        return __x in self.available

    def __len__(self) -> int:
        """Compute the number of available physical quantities."""
        return len(self.available)

    def __iter__(self) -> typing.Iterator[str]:
        """Iterate over available physical quantities."""
        return iter(self.available)

    @property
    def available(self):
        """The names of available physical quantities."""
        if self._available is None:
            self._available = tuple({k for m in self._mappings for k in m})
        return self._available

    def get_unit(self, name: str):
        """Compute or retrieve the metric unit of a physical quantity."""
        if found := self.get_attribute('unit', name):
            return found
        s = str(name)
        expression = symbolic.Expression(reference.NAMES.get(s, s))
        term = expression[0]
        this = self.get_unit(term.base) ** term.exponent
        if len(expression) > 1:
            for term in expression[1:]:
                this *= self.get_unit(term.base) ** term.exponent
        return metadata.Unit(this)

    def get_dimensions(self, name: str):
        """Compute or retrieve the array dimensions of a physical quantity."""
        if found := self.get_attribute('axes', name):
            return found
        s = str(name)
        expression = symbolic.Expression(reference.NAMES.get(s, s))
        term = expression[0]
        this = self.get_dimensions(term.base)
        if len(expression) > 1:
            for term in expression[1:]:
                this *= self.get_dimensions(term.base)
        return metadata.Axes(this)

    def get_parameters(self, name: str):
        """Compute or retrieve the parameters of a physical quantity."""
        if found := self.get_attribute('parameters', name):
            return found
        s = str(name)
        expression = symbolic.Expression(reference.NAMES.get(s, s))
        term = expression[0]
        this = self.get_parameters(term.base)
        if len(expression) > 1:
            for term in expression[1:]:
                this *= self.get_parameters(term.base)
        return Parameters(this)

    def get_quantity(self, name: str):
        """Retrieve a physical quantity by name, if available."""
        for mapping in self._mappings:
            if name in mapping:
                return mapping[name]

    def get_attribute(self, __name: str, target: str):
        """Get the named attribute for the target quantity, if possible."""
        for mapping in self._mappings:
            if target in mapping:
                return getattr(mapping[target], __name, None)


C = typing.TypeVar('C', bound=Context)


class Target:
    """An arbitrary observing target."""

    def __init__(self, name: str) -> None:
        self._name = name

    def apply(self, context: Context):
        """Observe this quantity within the given context."""
        return context.observe(self.name)

    @property
    def name(self):
        """The name of the target quantity."""
        return self._name


class Implementation:
    """An arbitrary observable quantity."""

    def __init__(
        self,
        name: str,
        context: Context,
        unit: metadata.UnitLike=None,
    ) -> None:
        """Initialize this instance.

        Parameters
        ----------
        name
            The name of the quantity to observe.

        context
            An observer-specific observing context.

        unit : unit-like, optional
            The metric unit to which to convert observations of this quantity.

        Notes
        -----
        * It is uncommon to initialize an instance of this class with an
          explicit unit. Most users will create an instance of this class via an
          observer, then change the instance unit as necessary via bracket
          syntax. Doing so actually creates a new instance, which the `unit`
          argument makes possible. For example, suppose `MyObserver` supports a
          'velocity' quantity and that its default unit is 'm / s':

        >>> observer = MyObserver(...)
        >>> v0 = observer['velocity']
        >>> v0.unit
        'm / s'
        >>> v1 = v0['km / h']
        >>> v1.unit
        'km / h'
        >>> v0.unit
        'm / s'

        """
        self._name = name
        self._context = context
        self._target = None
        self._unit = metadata.Unit(unit) if unit else None
        self._dimensions = None
        self._parameters = None

    def observe(self, **constraints):
        """Create an observation within the given user constraints."""
        return self.target.apply(self._context.constrain(constraints))

    @property
    def target(self):
        """The quantity to observe."""
        if self._target is None:
            self._target = Target(self.name)
        return self._target

    def __getitem__(self, __x: metadata.UnitLike):
        """Create a quantity with the new unit."""
        unit = (
            self.unit.norm[__x]
            if str(__x).lower() in metric.SYSTEMS else __x
        )
        if unit == self._unit:
            return self
        return Quantity(self.name, self._context, unit=unit)

    @property
    def unit(self):
        """The metric unit of this observable quantity."""
        if self._unit is None:
            self._unit = self._context.get_unit(self.name)
        return self._unit

    @property
    def dimensions(self):
        """The array dimensions of this observable quantity."""
        if self._dimensions is None:
            self._dimensions = self._context.get_dimensions(self.name)
        return self._dimensions

    @property
    def parameters(self):
        """The physical parameters of this observable quantity."""
        if self._parameters is None:
            self._parameters = self._context.get_parameters(self.name)
        return self._parameters

    @property
    def name(self):
        """The name of the target observable quantity."""
        return self._name

    _checkable = (
        'name',
        'unit',
        'dimensions',
        'parameters',
    )

    def __eq__(self, __o) -> bool:
        """True if two instances have equivalent attributes."""
        if isinstance(__o, Quantity):
            return all(
                getattr(self, attr) == getattr(__o, attr)
                for attr in self._checkable
            )
        return NotImplemented

    def __str__(self) -> str:
        display = [
            f"{str(self.name)!r}",
            f"unit={str(self.unit)!r}",
            f"dimensions={str(self.dimensions)}",
            f"parameters={str(self.parameters)}",
        ]
        return ', '.join(display)

