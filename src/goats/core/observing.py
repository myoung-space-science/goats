import abc
import collections
import collections.abc
import contextlib
import numbers
import operator as standard
import typing

from goats.core import iterables
from goats.core import metadata
from goats.core import metric
from goats.core import observed
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


class Interface(collections.abc.Mapping):
    """An interface to observing-related physical quantities."""

    def __init__(
        self,
        *observables: typing.Mapping[str],
        **others: typing.Mapping[str],
    ) -> None:
        self._observables = observables
        self._others = others

    def __len__(self) -> int:
        """Compute the number of available physical quantities."""
        return len(self.available)

    def __iter__(self) -> typing.Iterator[str]:
        """Iterate over available physical quantities."""
        return iter(self.available)

    @property
    def available(self):
        """The names of all available physical quantities."""
        others = tuple(k for m in self._others.values() for k in m)
        return self.observable + others

    @property
    def observable(self):
        """The names of observable physical quantities."""
        return tuple(k for m in self._observables for k in m)

    def __getitem__(self, __k: str):
        """Access physical quantities by key."""
        # Is it an observable quantity?
        for mapping in self._observables:
            if __k in mapping:
                return mapping[__k]
        # Is it an unobservable quantity?
        for mapping in self._others.values():
            if __k in mapping:
                return mapping[__k]
        # Is it a group of unobservable quantities?
        if __k in self._others:
            return self._others[__k]
        # We're out of options.
        raise KeyError(f"No known quantity for {__k!r}") from None

    def get_unit(self, key: str):
        """Compute or retrieve the metric unit of a physical quantity."""
        if found := self._lookup('unit', key):
            return found
        s = str(key)
        expression = symbolic.Expression(reference.NAMES.get(s, s))
        term = expression[0]
        this = self.get_unit(term.base) ** term.exponent
        if len(expression) > 1:
            for term in expression[1:]:
                this *= self.get_unit(term.base) ** term.exponent
        return metadata.Unit(this)

    def get_dimensions(self, key: str):
        """Compute or retrieve the array dimensions of a physical quantity."""
        if found := self._lookup('dimensions', key):
            return found
        s = str(key)
        expression = symbolic.Expression(reference.NAMES.get(s, s))
        term = expression[0]
        this = self.get_dimensions(term.base)
        if len(expression) > 1:
            for term in expression[1:]:
                this *= self.get_dimensions(term.base)
        return metadata.Axes(this)

    def get_parameters(self, key: str):
        """Compute or retrieve the parameters of a physical quantity."""
        if found := self._lookup('parameters', key):
            return found
        s = str(key)
        expression = symbolic.Expression(reference.NAMES.get(s, s))
        term = expression[0]
        this = self.get_parameters(term.base)
        if len(expression) > 1:
            for term in expression[1:]:
                this *= self.get_parameters(term.base)
        return Parameters(this)

    def _lookup(self, __name: str, target: str):
        """Search for an attribute among available quantities."""
        for mapping in self._observables:
            if target in mapping:
                return getattr(mapping[target], __name, None)


class Application(collections.abc.Collection):
    """Base class for observing applications."""

    def __init__(
        self,
        __quantities: Interface,
        constraints: typing.Mapping=None,
    ) -> None:
        """Create a new instance.
        
        Parameters
        ----------
        __quantities
            An instance of `~observing.Interface` or a subclass.

        constraints : mapping, optional
            User-provided observing constraints.
        """
        self._quantities = __quantities
        self._constraints = dict(constraints or {})
        self._cache = {}
        self._observables = None

    def __contains__(self, __x: str) -> bool:
        """True if `__x` is an available quantity."""
        return __x in self.quantities

    def __len__(self) -> int:
        """Compute the number of available quantities."""
        return len(self.quantities)

    def __iter__(self) -> typing.Iterator[str]:
        """Iterate over names of available quantities."""
        return iter(self.quantities)

    @property
    def quantities(self):
        """The names of available physical quantities."""
        return self._quantities

    @property
    def constraints(self):
        """The set of observing constraints."""
        return self._constraints

    @property
    def observables(self):
        """The names of observable physical quantities."""
        return self._observables

    def get_quantity(self, key: str):
        """Compute or retrieve a physical quantity."""
        return self._quantities.get(key)

    def constrain(self, constraints: typing.Mapping, update: bool=False):
        """Create a new instance with the given constraints.
        
        Parameters
        ----------
        constraints : mapping
            A mapping from string name to constraint value.

        update : bool, default=false
            If true, update the current constraints from `constraints`.
            Otherwise, overwrite the current constraints with `constraints`.
        """
        if not update:
            return self.apply(constraints)
        user = self.constraints.copy()
        user.update(constraints)
        return self.apply(user)

    def apply(self, constraints: typing.Mapping):
        """Unconditionally create a new instance with the given constraints.

        This method acts like a hook for `constrain`. Concrete subclasses that
        overload `__init__` may want to overload this method to ensure
        consistency.
        """
        return type(self)(self._quantities, constraints)

    @abc.abstractmethod
    def get_result(self, key: str) -> Quantity:
        """Compute an observed quantity."""

    @abc.abstractmethod
    def get_context(self, key: str) -> typing.Mapping:
        """Define the observing context."""


class Implementation:
    """The implementation of an observable quantity."""

    def __init__(
        self,
        __application: Application,
        name: str,
        unit: metadata.UnitLike=None,
    ) -> None:
        """Initialize this instance.

        Parameters
        ----------
        __application : subclass of `~observing.Application`
            The concrete observing application with which to create observations
            of the named quantity.

        name : string
            The name of the quantity to observe.

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
        self._application = __application
        self._name = name
        self._unit = metadata.Unit(unit) if unit else None
        self._dimensions = None
        self._parameters = None

    def observe(self, **constraints):
        """Create an observation within the given user constraints."""
        application = self._application.constrain(constraints)
        context = application.get_context(self.name)
        return observed.Quantity(
            application.get_result(self.name),
            context['axes'],
            constants=context.get('constants'),
        )

    def __getitem__(self, __x: metadata.UnitLike):
        """Create a quantity with the new unit."""
        unit = (
            self.unit.norm[__x]
            if str(__x).lower() in metric.SYSTEMS else __x
        )
        if unit == self._unit:
            return self
        return Implementation(self._application, self.name, self.unit)

    @property
    def quantities(self):
        """The observing-related physical quantities."""
        return self._application.quantities

    @property
    def unit(self):
        """The metric unit of this observable quantity."""
        if self._unit is None:
            self._unit = self.quantities.get_unit(self.name)
        return self._unit

    @property
    def dimensions(self):
        """The array dimensions of this observable quantity."""
        if self._dimensions is None:
            self._dimensions = self.quantities.get_dimensions(self.name)
        return self._dimensions

    @property
    def parameters(self):
        """The physical parameters of this observable quantity."""
        if self._parameters is None:
            self._parameters = self.quantities.get_parameters(self.name)
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
