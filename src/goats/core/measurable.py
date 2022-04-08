import abc
import collections.abc
import functools
import math
import numbers
import operator
import typing

import numpy
import numpy.typing

from goats.core import iterables
from goats.core import metric


@typing.runtime_checkable
class Orderable(typing.Protocol):
    """Protocol for objects that support ordering.
    
    Instance checks against this ABC will return `True` iff the instance
    implements the following methods: `__lt__`, `__gt__`, `__le__`, `__ge__`,
    `__eq__`, and `__ne__`. It exists to support type-checking orderable objects
    outside the `~quantified.Algebraic` framework (e.g., pure numbers).
    """

    __slots__ = ()

    @abc.abstractmethod
    def __lt__(self, other):
        pass

    @abc.abstractmethod
    def __eq__(self, other):
        pass

    @abc.abstractmethod
    def __le__(self, other):
        pass

    @abc.abstractmethod
    def __gt__(self, other):
        pass

    @abc.abstractmethod
    def __ge__(self, other):
        pass

    @abc.abstractmethod
    def __ne__(self, other):
        pass


class Ordered:
    """Abstract base class for all objects that support relative ordering.

    Concrete implementations of this class must define the six binary comparison
    operators (a.k.a "rich comparison" operators): `__lt__`, `__gt__`, `__le__`,
    `__ge__`, `__eq__`, and `__ne__`.

    The following default implementations are available by calling their
    equivalents on `super()`:

    - `__ne__`: defined as not equal.
    - `__le__`: defined as less than or equal.
    - `__gt__`: defined as not less than and not equal.
    - `__ge__`: defined as not less than.
    """

    __slots__ = ()

    __hash__ = None

    @abc.abstractmethod
    def __lt__(self, other) -> bool:
        """True if self < other."""
        pass

    @abc.abstractmethod
    def __eq__(self, other) -> bool:
        """True if self == other."""
        pass

    @abc.abstractmethod
    def __le__(self, other) -> bool:
        """True if self <= other."""
        return self.__lt__(other) or self.__eq__(other)

    @abc.abstractmethod
    def __gt__(self, other) -> bool:
        """True if self > other."""
        return not self.__le__(other)

    @abc.abstractmethod
    def __ge__(self, other) -> bool:
        """True if self >= other."""
        return not self.__lt__(other)

    @abc.abstractmethod
    def __ne__(self, other) -> bool:
        """True if self != other."""
        return not self.__eq__(other)


Self = typing.TypeVar('Self', bound='Additive')


class Additive(abc.ABC):
    """Abstract base class for additive objects."""

    __slots__ = ()

    @abc.abstractmethod
    def __add__(self: Self, other) -> Self:
        pass

    @abc.abstractmethod
    def __radd__(self: Self, other) -> Self:
        pass

    @abc.abstractmethod
    def __sub__(self: Self, other) -> Self:
        pass

    @abc.abstractmethod
    def __rsub__(self: Self, other) -> Self:
        pass


Self = typing.TypeVar('Self', bound='Multiplicative')


class Multiplicative(abc.ABC):
    """Abstract base class for multiplicative objects."""

    __slots__ = ()

    @abc.abstractmethod
    def __mul__(self: Self, other) -> Self:
        pass

    @abc.abstractmethod
    def __rmul__(self: Self, other) -> Self:
        pass

    @abc.abstractmethod
    def __truediv__(self: Self, other) -> Self:
        pass

    @abc.abstractmethod
    def __rtruediv__(self: Self, other) -> Self:
        pass

    @abc.abstractmethod
    def __pow__(self: Self, other) -> Self:
        pass

    @abc.abstractmethod
    def __rpow__(self: Self, other) -> Self:
        pass


class Algebraic(Ordered, Additive, Multiplicative):
    """Base class for algebraic objects.

    Concrete subclasses of this class must implement the six comparison
    operators,
        - `__lt__` (less than; called for `self < other`)
        - `__gt__` (greater than; called for `self > other`)
        - `__le__` (less than or equal to; called for `self <= other`)
        - `__ge__` (greater than or equal to; called for `self >= other`)
        - `__eq__` (equal to; called for `self == other`)
        - `__ne__` (not equal to; called for `self != other`)
    
    the following unary arithmetic operators,
        - `__abs__` (absolute value; called for `abs(self)`)
        - `__neg__` (negative value; called for `-self`)
        - `__pos__` (positive value; called for `+self`)

    and the following binary arithmetic operators,
        - `__add__` (addition; called for `self + other`)
        - `__radd__` (reflected addition; called for `other + self`)
        - `__sub__` (subtraction; called for `self - other`)
        - `__rsub__` (reflected subtraction; called for `other - self`)
        - `__mul__` (multiplication; called for `self * other`)
        - `__rmul__` (reflected multiplication; called for `other * self`)
        - `__truediv__` (division; called for `self / other`)
        - `__rtruediv__` (reflected division; called for `other / self`)
        - `__pow__` (exponentiation; called for `self ** other`)
        - `__rpow__` (reflected exponentiation; called for `other ** self`)

    Any required method may return `NotImplemented`.
    """

    __slots__ = ()

    @abc.abstractmethod
    def __abs__(self):
        """Implements abs(self)."""
        pass

    @abc.abstractmethod
    def __neg__(self):
        """Called for -self."""
        pass

    @abc.abstractmethod
    def __pos__(self):
        """Called for +self."""
        pass


Self = typing.TypeVar('Self', bound='SupportsNeg')


@typing.runtime_checkable
class SupportsNeg(typing.Protocol):
    """Protocol for objects that support negation (``-self``)."""

    __slots__ = ()

    @abc.abstractmethod
    def __neg__(self: Self) -> Self:
        pass


class RealValued(Algebraic):
    """Abstract base class for all real-valued objects.
    
    This class is similar to ``numbers.Real``, but it does not presume to
    represent a single value.
    
    Concrete subclasses of this object must implement all the
    `~quantified.Algebraic` operators except for `__sub__` and `__rsub__`
    (defined here with respect to `__neg__`). Subclasses may, of course,
    override these base implementations.
    """

    def __sub__(self, other: SupportsNeg):
        """Called for self - other."""
        return self + -other

    def __rsub__(self, other: SupportsNeg):
        """Called for other - self."""
        return -self + other


RealValued.register(numbers.Real)
RealValued.register(numpy.ndarray) # close enough for now...


class Quantifiable(Algebraic, iterables.ReprStrMixin):
    """A real-valued amount and the associated metric.
    
    This abstract base class represents the basis for quantifiable objects.
    Concrete subclasses must implement all the `~quantified.Algebraic`
    operators, and should do so in a way that self-consistently handles the
    instance metric.
    """

    def __init__(
        self,
        __amount: RealValued,
        __metric: Multiplicative,
    ) -> None:
        self._amount = __amount
        self._metric = __metric
        self.display['__str__'].update(
            strings=["{_amount} {_metric}"],
            separator=' ',
        )

    def __bool__(self) -> bool:
        """Always true for a valid instance."""
        return True


class ComparisonError(TypeError):
    """Incomparable instances of the same type."""

    def __init__(self, __this: typing.Any, __that: typing.Any, name: str):
        self.this = getattr(__this, name, None)
        self.that = getattr(__that, name, None)

    def __str__(self) -> str:
        return f"Can't compare '{self.this}' to '{self.that}'"


class same:
    """A callable class that enforces object consistency.

    When used to decorate a method that takes two arguments, this class will
    ensure that the arguments have equal values of a named attribute. This may
    be useful when writing binary comparison methods that are only valid for two
    objects of the same kind (e.g., physical objects with the same dimension).
    """

    def __init__(
        self,
        *names: str,
        allowed: typing.Iterable[typing.Type]=None,
    ) -> None:
        self.names = names
        self.allowed = iterables.whole(allowed)

    def __call__(self, func: typing.Callable) -> typing.Callable:
        """Ensure argument consistency before calling `func`."""
        if not self.names:
            return func
        @functools.wraps(func)
        def wrapper(this, that):
            allowed = (type(this), *self.allowed)
            if not isinstance(that, allowed):
                return NotImplemented
            if isinstance(that, type(this)):
                for name in self.names:
                    if not self._comparable(this, that, name):
                        raise ComparisonError(this, that, name) from None
            return func(this, that)
        return wrapper

    def _comparable(
        self,
        this: typing.Any,
        that: typing.Any,
        name: str,
    ) -> bool:
        """Check whether the instances are comparable."""
        return getattr(this, name) == getattr(that, name)


RT = typing.TypeVar('RT')
Operator = typing.TypeVar('Operator', bound=typing.Callable)
Operator = typing.Callable[..., RT]


def _comparison(opr: Operator):
    """Implement a comparison operator."""
    @same('_metric', allowed=numbers.Real)
    def func(a: Quantifiable, b):
        if isinstance(b, Quantifiable):
            return opr(a._amount, b._amount)
        if isinstance(b, Orderable):
            return opr(a._amount, b)
        return NotImplemented
    func.__name__ = f"__{opr.__name__}__"
    func.__doc__ = opr.__doc__
    return func


def _unary(opr: Operator):
    """Implement a unary arithmetic operator."""
    def func(a: Quantifiable):
        return type(a)(opr(a._amount), a._metric)
    func.__name__ = f"__{opr.__name__}__"
    func.__doc__ = opr.__doc__
    return func


def _preserve_metric(opr: Operator):
    """Implement a forward operator that preserves the instance metric."""
    @same('_metric')
    def func(a: Quantifiable, b: Quantifiable):
        return type(a)(opr(a._amount, b._amount), a._metric)
    return func


def _combine_metric(opr: Operator):
    """Implement a forward operator that combines instance metrics."""
    def func(a: Quantifiable, b: Quantifiable):
        return type(a)(opr(a._amount, b._amount), opr(a._metric, b._metric))
    return func


def _update_metric(opr: Operator):
    """Implement a forward operator that updates the instance metric."""
    def func(a: Quantifiable, b: RealValued):
        return type(a)(opr(a._amount, b), opr(a._metric, b))
    return func


def _standard(opr: Operator):
    """Implement a standard forward operator."""
    def func(a: Quantifiable, b: RealValued):
        return type(a)(opr(a._amount, b), a._metric)
    return func


Wrapper = typing.TypeVar('Wrapper', bound=typing.Callable)
Wrapper = typing.Callable[..., typing.Callable[..., Quantifiable]]


Rules = typing.TypeVar('Rules', bound=dict)
Rules = typing.Dict[
    typing.Tuple[type],
    Wrapper,
]


_operators: typing.Dict[Operator, Rules] = {
    operator.add: {
        (Quantifiable, Quantifiable): _preserve_metric,
        (Quantifiable, RealValued): _standard,
    },
    operator.sub: {
        (Quantifiable, Quantifiable): _preserve_metric,
        (Quantifiable, RealValued): _standard,
    },
    operator.mul: {
        (Quantifiable, Quantifiable): _combine_metric,
        (Quantifiable, RealValued): _standard,
    },
    operator.truediv: {
        (Quantifiable, Quantifiable): _combine_metric,
        (Quantifiable, RealValued): _standard,
    },
    operator.pow: {
        (Quantifiable, RealValued): _update_metric,
    },
}


def _get_rule(rules: dict, *args) -> Wrapper:
    types = tuple(type(arg) for arg in args)
    if types in rules:
        return rules[types]
    for key, rule in rules.items():
        if all(issubclass(t, k) for t, k in zip(types, key)):
            return rule


def _forward(opr: Operator):
    """Implement a forward operator."""
    rules = _operators.get(opr)
    def func(a: Quantifiable, b):
        rule = _get_rule(rules, a, b)
        return rule(opr)(a, b) if rule else NotImplemented
    func.__name__ = f"__{opr.__name__}__"
    func.__doc__ = opr.__doc__
    return func


def _inplace(opr: Operator):
    """Implement an in-place operator."""
    rules = _operators.get(opr)
    def func(a: Quantifiable, b):
        if rule := _get_rule(rules, a, b):
            result = rule(opr)(a, b)
            a._amount = result._amount
            a._metric = result._metric
            return a
        return NotImplemented
    func.__name__ = f"__i{opr.__name__}__"
    func.__doc__ = opr.__doc__
    return func


def _reverse(opr: Operator):
    """Implement a standard reverse operator."""
    def func(b: Quantifiable, a):
        return (
            type(b)(opr(a, b._amount), b._metric) if isinstance(a, RealValued)
            else NotImplemented
        )
    func.__name__ = f"__r{opr.__name__}__"
    func.__doc__ = opr.__doc__
    return func


def _suppress(name: str):
    """Suppress an operator."""
    def func(*args, **kwargs):
        return NotImplemented
    func.__name__ = name
    func.__doc__ = """Not implemented."""
    return func


class OperatorMixin:
    """A mixin class that defines operators for quantifiable objects.
    
    This class implements the `~quantified.Algebraic` operators with the
    following rules:
        - unary `-`, `+`, and `abs` on an instance
        - binary `+` and `-` between two instances with an identical metric
        - binary `*` and `/` between two instances
        - symmetric binary `*` between an instance and a number
        - right-sided `/` and `**` between an instance and a number

    Notes on allowed binary arithmetic operations:
        - This class does not support floor division (`//`) in any form because
          of the ambiguity it would create with `~metric.Unit` floor division.
        - This class does not support floating-point division (`/`) in which the
          left operand is not the same type or a subtype. The reason for this
          choice is that the result may be ambiguous. For example, suppose we
          have an instance called ``d`` with values ``[10.0, 20.0]`` and unit
          ``cm``. Whereas the result of ``d / 2.0`` should clearly be a new
          instance with values ``[5.0, 10.0]`` and the same unit, it is unclear
          whether the values of ``2.0 / d`` should be element-wise ratios (i.e.,
          ``[0.2, 0.1]``) or a single value (e.g., ``2.0 / ||d||``) and it is
          not at all obvious what the unit or dimensions should be.
    """

    __lt__ = _comparison(operator.lt)
    __le__ = _comparison(operator.le)
    __gt__ = _comparison(operator.gt)
    __ge__ = _comparison(operator.ge)
    __eq__ = _comparison(operator.eq)
    __ne__ = _comparison(operator.ne)

    __abs__ = _unary(operator.abs)
    __pos__ = _unary(operator.pos)
    __neg__ = _unary(operator.neg)

    __add__ = _forward(operator.add)
    __iadd__ = _inplace(operator.add)
    __radd__ = _reverse(operator.add)

    __sub__ = _forward(operator.sub)
    __isub__ = _inplace(operator.sub)
    __rsub__ = _reverse(operator.sub)

    __mul__ = _forward(operator.mul)
    __imul__ = _inplace(operator.mul)
    __rmul__ = _reverse(operator.mul)

    __truediv__ = _forward(operator.truediv)
    __itruediv__ = _inplace(operator.truediv)
    __rtruediv__ = _suppress('__rtruediv__')

    __pow__ = _forward(operator.pow)
    __rpow__ = _suppress('__rpow__')
    __ipow__ = _inplace(operator.pow)


T = typing.TypeVar('T')


def _cast(__type: typing.Type[T]) -> typing.Callable[[Quantifiable], T]:
    """Implement a type-casting operator."""
    def func(a):
        return (
            __type(a._amount) if isinstance(a, Quantifiable)
            else NotImplemented
        )
    name = __type.__class__.__qualname__
    func.__name__ = f"__{name}__"
    func.__doc__ = f"""Called for {name}(self)."""
    return func


class Measurement(collections.abc.Sequence, iterables.ReprStrMixin):
    """A sequence of values and their associated unit."""

    # Should this use `__new__` since it's immutable?
    def __init__(
        self,
        values: typing.Iterable[numbers.Real],
        unit: metric.UnitLike,
    ) -> None:
        self._values = values
        self._unit = unit

    @property
    def values(self):
        """This measurement's values."""
        return tuple(self._values)

    @property
    def unit(self):
        """This measurement's unit."""
        return metric.Unit(self._unit)

    def __len__(self) -> int:
        return len(self._values)

    def __getitem__(self, index):
        """Called for index-based value access."""
        if isinstance(index, typing.SupportsIndex) and index < 0:
            index += len(self)
        values = iterables.whole(self.values[index])
        unit = str(self.unit)
        return [(value, unit) for value in values]

    def __str__(self) -> str:
        values = ', '.join(str(value) for value in self.values)
        return f"{values} [{self._unit}]"


class Quantity(Quantifiable):
    """A real-valued amount and the associated unit.
    
    This abstract base class represents the basis for all measurable quantities.
    It builds on `~measurable.Quantifiable` by specifying an instance of
    `~metric.Unit` as the metric.

    This class is intended as the primary ABC on which to build more specific
    abstract or concrete measurable objects. Concrete subclasses must implement
    all the abstract methods of `~measurable.Algebraic`, but may choose to do so
    in a way that leverages properties of the `~metric.Unit` class. Implementors
    may use `~measurable.OperatorMixin` as a simple way to provide default
    versions of all required operators.
    """

    def __init__(
        self,
        amount: RealValued,
        unit: metric.UnitLike=None,
    ) -> None:
        super().__init__(amount, metric.Unit(unit or '1'))
        display = {
            '__str__': {
                'strings': ["{_amount}", "[{_metric}]"],
                'separator': ' ',
            },
            '__repr__': {
                'strings': ["{_amount}", "unit='{_metric}'"],
                'separator': ', ',
            },
        }
        self.display.update(display)

    def unit(self, unit: metric.UnitLike=None):
        """Get or set the unit of this object's values."""
        if not unit:
            return self._metric
        new = metric.Unit(unit)
        self._amount *= new // self._metric
        self._metric = new
        return self


class Measurable(Quantity):
    """A quantifiable object that supports direct measurement.
    
    Concrete subclasses of this ABC must implement all the abstract methods of
    `~measurable.Algebraic`, as well as a new method, `__measure__`, which the
    `~measurable.measure` function will call to produce an instance of
    `~measurable.Measurement` from the given instance.
    """

    @abc.abstractmethod
    def __measure__(self) -> Measurement:
        """Measure this object."""
        pass


class SingleValued(Measurable):
    """Abstract definition of a single-valued measurable quantity."""

    def __init__(
        self,
        amount: numbers.Real,
        unit: metric.UnitLike=None,
    ) -> None:
        super().__init__(float(amount), unit)

    @abc.abstractmethod
    def __float__(self) -> float:
        """Called for float(self)."""
        pass

    @abc.abstractmethod
    def __int__(self) -> int:
        """Called for int(self)."""
        pass

    @abc.abstractmethod
    def __round__(self, **kwargs) -> int:
        """Called for round(self)."""
        pass

    @abc.abstractmethod
    def __ceil__(self) -> int:
        """Called for ceil(self)."""
        pass

    @abc.abstractmethod
    def __floor__(self) -> int:
        """Called for floor(self)."""
        pass

    @abc.abstractmethod
    def __trunc__(self) -> int:
        """Called for trunc(self)."""
        pass

    def __measure__(self) -> Measurement:
        values = iterables.whole(self._amount)
        return Measurement(values, self.unit())


class MultiValued(Measurable):
    """Abstract definition of a multi-valued measurable quantity."""

    def __init__(
        self,
        amount: typing.Union[RealValued, numpy.typing.ArrayLike],
        unit: metric.UnitLike=None,
    ) -> None:
        amount = numpy.asfarray(list(iterables.whole(amount)))
        super().__init__(amount, unit)

    @abc.abstractmethod
    def __len__(self) -> int:
        """Called for len(self)."""
        pass

    Item = typing.TypeVar('Item', bound=typing.Iterable)
    Item = typing.Union[SingleValued, typing.List[SingleValued]]

    @abc.abstractmethod
    def __getitem__(self, index) -> Item:
        """Called for index-based value access."""
        pass

    def __measure__(self) -> Measurement:
        return Measurement(self._amount, self.unit())


class Scalar(OperatorMixin, SingleValued):
    """A measured object with a single value."""

    __float__ = _cast(float)
    __int__ = _cast(int)

    __round__ = _unary(round)
    __ceil__ = _unary(math.ceil)
    __floor__ = _unary(math.floor)
    __trunc__ = _unary(math.trunc)


class Vector(OperatorMixin, MultiValued):
    """A measured object with multiple values."""

    def __len__(self) -> int:
        """Called for len(self)."""
        return len(self._amount)

    def __getitem__(self, index):
        """Called for index-based value access."""
        if isinstance(index, typing.SupportsIndex) and index < 0:
            index += len(self)
        values = self._amount[index]
        iter_values = isinstance(values, typing.Iterable)
        unit = self.unit()
        return (
            [Scalar(value, unit) for value in values] if iter_values
            else Scalar(values, unit)
        )


def ismeasurable(this):
    """True if we can measure `this`.
    
    A measurable object may be:
    
    - an object that defines `__measure__`
    - a number
    - an iterable of numbers
    - an iterable of numbers followed by a unit-like object
    - an two-element iterable whose first element is an iterable of numbers and
      whose second element is a unit-like object
    - an iterable of measurable objects.

    Parameters
    ----------
    this
        The candidate measurable object.

    Returns
    -------
    bool
        True if `this` is measurable; false otherwise.
    """
    args = iterables.unwrap(this)
    if hasattr(args, '__measure__'):
        return True
    if isinstance(args, numbers.Number):
        return True
    if not isinstance(args, iterables.whole):
        return False
    if iterables.allinstance(args, numbers.Number):
        return True
    if isinstance(args[-1], metric.UnitLike):
        arg0 = args[0]
        values = arg0 if isinstance(arg0, typing.Iterable) else args[:-1]
        if iterables.allinstance(values, numbers.Number):
            return True
    if all(ismeasurable(i) for i in args):
        return True
    return False


class Unmeasurable(Exception):
    """Cannot measure this type of object."""

    def __init__(self, arg: object) -> None:
        self.arg = arg

    def __str__(self) -> str:
        return f"Cannot measure {self.arg!r}"


class MeasuringTypeError(TypeError):
    """A type-related error occurred while trying to measure this object."""


def measure(*args):
    """Create a measurement from a measurable object.

    This function will first check whether `args` is a single object that
    conforms to the `Measurable` protocol, and call a special method if so.
    Otherwise, it will attempt to parse `args` into one or more values and a
    corresponding unit.
    """
    if len(args) == 1 and isinstance(args[0], Measurable):
        return args[0].__measure__()
    parsed = parse_measurable(args, distribute=False)
    return Measurement(parsed[:-1], parsed[-1])


def parse_measurable(args, distribute: bool=False):
    """Extract one or more values and an optional unit from `args`.
    
    See Also
    --------
    measure : returns the parsed object as a `Measurement`.
    """

    # Strip redundant lists and tuples.
    unwrapped = iterables.unwrap(args)

    # Raise an error for null input.
    if iterables.missing(unwrapped):
        raise Unmeasurable(unwrapped) from None

    # Handle a single numerical value:
    if isinstance(unwrapped, numbers.Number):
        result = (unwrapped, '1')
        return [result] if distribute else result

    # Count the number of distinct unit-like objects.
    types = [type(arg) for arg in unwrapped]
    n_units = sum(types.count(t) for t in (str, metric.Unit))

    # Raise an error for multiple units.
    if n_units > 1:
        errmsg = "You may only specify one unit."
        raise MeasuringTypeError(errmsg) from None

    # TODO: The structure below suggests that there may be available
    # refactorings, though they may require first redefining or dismantling
    # `_callback_parse`.

    # Handle flat numerical iterables, like (1.1,) or (1.1, 2.3).
    if all(isinstance(arg, numbers.Number) for arg in unwrapped):
        return _wrap_measurable(unwrapped, '1', distribute)

    # Recursively handle an iterable of whole (distinct) items.
    if all(isinstance(arg, iterables.whole) for arg in unwrapped):
        return _callback_parse(unwrapped, distribute)

    # Ensure an explicit unit-like object
    unit = ensure_unit(unwrapped)

    # Handle flat iterables with a unit, like (1.1, 'm') or (1.1, 2.3, 'm').
    if all(isinstance(arg, numbers.Number) for arg in unwrapped[:-1]):
        return _wrap_measurable(unwrapped[:-1], unit, distribute)

    # Handle iterable values with a unit, like [(1.1, 2.3), 'm'].
    if isinstance(unwrapped[0], (list, tuple, range)):
        return _wrap_measurable(unwrapped[0], unit, distribute)


def _wrap_measurable(values, unit, distribute: bool):
    """Wrap a parsed measurable and return to caller."""
    if distribute:
        return list(iterables.distribute(values, unit))
    return (*values, unit)


def _callback_parse(unwrapped, distribute: bool):
    """Parse the measurable by calling back to `parse_measurable`."""
    if distribute:
        return [
            item
            for arg in unwrapped
            for item in parse_measurable(arg, distribute=True)
        ]
    parsed = [
        parse_measurable(arg, distribute=False) for arg in unwrapped
    ]
    units = [item[-1] for item in parsed]
    if any(unit != units[0] for unit in units):
        errmsg = "Can't combine measurements with different units."
        raise MeasuringTypeError(errmsg)
    values = [i for item in parsed for i in item[:-1]]
    unit = units[0]
    return (*values, unit)


def ensure_unit(args):
    """Extract the given unit or assume the quantity is unitless."""
    last = args[-1]
    implicit = not any(isinstance(arg, (str, metric.Unit)) for arg in args)
    explicit = last in ['1', metric.Unit('1')]
    if implicit or explicit:
        return '1'
    return str(last)


