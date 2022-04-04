import abc
import functools
import math
import numbers
import typing

import numpy as np

from goats.core import iterables
from goats.core import metric


class ComparisonError(TypeError):
    """Incomparable instances of the same type."""

    def __init__(self, __this: typing.Any, __that: typing.Any, name: str):
        self.this = getattr(__this, name, None)
        self.that = getattr(__that, name, None)

    def __str__(self) -> str:
        return f"Can't compare '{self.this}' to '{self.that}'"


class same:
    """A decorator class that enforces object consistency.

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
        """Ensure attribute consistency before calling `func`."""
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


class Comparable(metaclass=abc.ABCMeta):
    """The base class for all comparable objects.

    Comparable objects support relative ordering. Concrete implementations of
    this class must define the six binary comparison operators (a.k.a "rich
    comparison" operators): `__lt__`, `__gt__`, `__le__`, `__ge__`, `__eq__`,
    and `__ne__`.
    """

    __slots__ = ()

    __hash__ = None

    @abc.abstractmethod
    def __lt__(self, other) -> bool:
        """True if self < other."""
        pass

    @abc.abstractmethod
    def __le__(self, other) -> bool:
        """True if self <= other."""
        pass

    @abc.abstractmethod
    def __gt__(self, other) -> bool:
        """True if self > other."""
        pass

    @abc.abstractmethod
    def __ge__(self, other) -> bool:
        """True if self >= other."""
        pass

    @abc.abstractmethod
    def __eq__(self, other) -> bool:
        """True if self == other."""
        pass

    @abc.abstractmethod
    def __ne__(self, other) -> bool:
        """True if self != other."""
        pass


class RealValued(Comparable):
    """A comparable object with one or more numerical values.

    This class borrows from base classes in the ``numbers`` module but it isn't
    in the numerical hierarchy because it doesn't require conversion to a single
    numerical value. However, it does register ``numbers.Real`` as a virtual
    subclass.
    """

    @abc.abstractmethod
    def __bool__(self) -> bool:
        """The truth value of this object as returned by bool(self)."""
        pass

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

    @abc.abstractmethod
    def __add__(self, other):
        """Called for self + other."""
        pass

    @abc.abstractmethod
    def __radd__(self, other):
        """Called for other + self."""
        pass

    def __iadd__(self, other):
        """Called for self += other."""
        return NotImplemented

    def __sub__(self, other):
        """Called for self - other."""
        return self + -other

    def __rsub__(self, other):
        """Called for other - self."""
        return -self + other

    def __isub__(self, other):
        """Called for self -= other."""
        return NotImplemented

    @abc.abstractmethod
    def __mul__(self, other):
        """Called for self * other."""
        pass

    @abc.abstractmethod
    def __rmul__(self, other):
        """Called for other * self."""
        pass

    def __imul__(self, other):
        """Called for self *= other."""
        return NotImplemented

    @abc.abstractmethod
    def __truediv__(self, other):
        """Called for self / other."""
        pass

    @abc.abstractmethod
    def __rtruediv__(self, other):
        """Called for other / self."""
        pass

    def __itruediv__(self, other):
        """Called for self /= other."""
        return NotImplemented

    @abc.abstractmethod
    def __pow__(self, other):
        """Called for self ** other or pow(self, other)."""
        pass

    @abc.abstractmethod
    def __rpow__(self, other):
        """Called for other ** self or pow(other, self)."""
        pass

    def __ipow__(self, other):
        """Called for self **= other."""
        return NotImplemented

RealValued.register(numbers.Real)
RealValued.register(np.ndarray)


Instance = typing.TypeVar('Instance', bound='Quantified')


class Quantified(RealValued, iterables.ReprStrMixin):
    """The base class for all quantified objects.

    A quantified object has an amount and a quantity. This class does not place
    any restrictions on the type of either attribute.

    This class declares which operators require that their operands have
    consistent quantities, and enforces those requirements on subclasses.
    """

    _quantified = (
        '__lt__',
        '__le__',
        '__gt__',
        '__ge__',
        '__eq__',
        '__ne__',
        '__add__',
        '__sub__',
        '__iadd__',
        '__isub__',
    )

    def __init_subclass__(cls, **kwargs) -> None:
        """Enforce quantifiability on method implementations.

        This implementation uses an instance of the `same` decorator class to
        ensure that the operands to each method named in the `_quantified` class
        attribute have the same quantity (i.e., the same value of their
        `quantity` attribute). Subclasses may individually customize this
        behavior via the keyword arguments described below

        Parameters
        ----------
        allowed : mapping from string to type
            A mapping (e.g., `dict`) from method name to a type or iterable of
            types in addition to instances of the decorated class that the named
            method should accept. The `same` class will not check objects of
            these additional types for sameness. For example, to indicate that a
            subclass accepts integers to its addition and subtraction methods,
            pass `allowed={'__add__': int, '__sub__': int}` to the class
            constructor.
        """
        allowed = kwargs.get('allowed', {})
        for method in cls._quantified:
            if current:= getattr(cls, method, None):
                update = same('quantity', allowed=allowed.get(method))
                updated = update(current)
                setattr(cls, method, updated)

    @typing.overload
    def __new__(
        cls: typing.Type[Instance],
        amount: typing.Any,
        quantity: typing.Any,
    ) -> Instance: ...

    @typing.overload
    def __new__(
        cls: typing.Type[Instance],
        instance: Instance,
    ) -> Instance: ...

    _amount: typing.Any=None
    _quantity: typing.Any=None

    def __new__(cls, *args):
        """Create a new instance of `cls`."""
        self = super().__new__(cls)
        if len(args) == 1 and isinstance(args[0], cls):
            instance = args[0]
            self._amount = instance._amount
            self._quantity = instance._quantity
            return self
        if len(args) == 2:
            self._amount, self._quantity = args
            return self
        raise TypeError(
            f"Can't instantiate {cls} from {args}"
        ) from None

    @property
    def quantity(self):
        """The type of thing that this object represents."""
        return self._quantity

    def __str__(self) -> str:
        """A simplified representation of this object."""
        return f"{self._amount} {self._quantity}"


Instance = typing.TypeVar('Instance', bound='Ordered')


class Ordered(Quantified):
    """A quantified object that supports comparisons

    An ordered object has an amount and a quantity. The amount must be formally
    comparable -- that is, a comparison to another amount using one of the six
    binary relations (i.e., <, >, <=, >=, ==, !=) must produce well-defined
    results. The quantity may be anything that supports equality comparison,
    which should be true unless the object explicitly disables `__eq__`.
    """

    @typing.overload
    def __new__(
        cls: typing.Type[Instance],
        amount: Comparable,
        quantity: typing.Any,
    ) -> Instance: ...

    @typing.overload
    def __new__(
        cls: typing.Type[Instance],
        instance: Instance,
    ) -> Instance: ...

    def __new__(cls, *args):
        return super().__new__(cls, *args)

    _amount: Comparable=None

    def __lt__(self, other: 'Ordered') -> bool:
        return self._amount < other._amount

    def __le__(self, other: 'Ordered') -> bool:
        return self._amount <= other._amount

    def __gt__(self, other: 'Ordered') -> bool:
        return self._amount > other._amount

    def __ge__(self, other: 'Ordered') -> bool:
        return self._amount >= other._amount

    def __eq__(self, other: 'Ordered') -> bool:
        return self._amount == other._amount

    def __ne__(self, other) -> bool:
        """True if self != other.

        Explicitly defined with respect to `self.__eq__` to promote consistency
        in subclasses that overload `__eq__`.
        """
        return not self.__eq__(other)

    def __str__(self) -> str:
        """A simplified representation of this object."""
        return f"{self._amount} {self._quantity}"


Instance = typing.TypeVar('Instance', bound='Measured')


class Measured(Ordered):
    """An ordered object with a unit.

    Building on the `Ordered` class, a measured object must have an amount and a
    unit. The amount must be formally real-valued in the sense that the
    following arithmetic operations must produce well-defined results:
    - unary `-`, `+`, and `abs`
    - binary `+` and `-` between two instances with an identical unit
    - binary `*` and `/` between two instances
    - symmetric binary `*` between an instance and a number
    - right-sided `/` and `**` between an instance and a number

    Notes on allowed binary arithmetic operations:
    - This class does not support floor division (`//`) in any form because of
      the ambiguity it would create with `~metric.Unit` floor division.
    - This class does not support floating-point division (`/`) in which the
      left operand is not the same type or a subtype. The reason for this choice
      is that the result may be ambiguous. For example, suppose we have an
      instance called ``d`` with values ``[10.0, 20.0]`` and unit ``cm``.
      Whereas the result of ``d / 2.0`` should clearly be a new instance with
      values ``[5.0, 10.0]`` and the same unit, it is unclear whether the values
      of ``2.0 / d`` should be element-wise ratios (i.e., ``[0.2, 0.1]``) or a
      single value (e.g., ``2.0 / ||d||``) and it is not at all obvious what the
      unit or dimensions should be.
    """

    @typing.overload
    def __new__(
        cls: typing.Type[Instance],
        amount: RealValued,
    ) -> Instance:
        """Create a new measured object.
        
        Parameters
        ----------
        amount : real-valued
            The measured amount.
        """

    @typing.overload
    def __new__(
        cls: typing.Type[Instance],
        amount: RealValued,
        unit: typing.Union[str, metric.Unit],
    ) -> Instance:
        """Create a new measured object.
        
        Parameters
        ----------
        amount : real-valued
            The measured amount.

        unit : string or `~metric.Unit`
            The unit in which `amount` is measured.
        """

    @typing.overload
    def __new__(
        cls: typing.Type[Instance],
        instance: Instance,
    ) -> Instance:
        """Create a new measured object.
        
        Parameters
        ----------
        instance : `~measurables.Measured`
            An existing instance of this class.
        """

    _amount: RealValued=None
    _unit: metric.Unit=None

    def __new__(cls, *args, **kwargs):
        """The concrete implementation of `~measurables.Measured.__new__`.
        
        Notes
        -----
        This method first extracts a local `unit` in order to pass it as a `str`
        to its parent class's `_quantity` attribute while returning it as a
        `~metric.Unit` for initializing the `_unit` attribute of the current
        instance.
        """
        if not kwargs and len(args) == 1 and isinstance(args[0], cls):
            instance = args[0]
            amount = instance._amount
            unit = instance.unit()
        else:
            attrs = list(args)
            attr_dict = {
                k: attrs.pop(0) if attrs
                else kwargs.pop(k, None)
                for k in ('amount', 'unit')
            }
            amount = attr_dict['amount']
            unit = metric.Unit(attr_dict['unit'] or '1')
        self = super().__new__(cls, amount, str(unit))
        self._unit = unit
        return self

    @classmethod
    def _new(cls: typing.Type[Instance], *args, **kwargs) -> Instance:
        """Create a new instance from updated attributes."""
        return cls(*args, **kwargs)

    @typing.overload
    def unit(self: Instance) -> metric.Unit:
        """Get this object's unit of measurement.
        
        Parameters
        ----------
        None

        Returns
        -------
        `~metric.Unit`
            The current unit of `amount`.
        """

    @typing.overload
    def unit(
        self: Instance,
        new: typing.Union[str, metric.Unit],
    ) -> Instance:
        """Update this object's unit of measurement.

        Parameters
        ----------
        new : string or `~metric.Unit`
            The new unit in which to measure `amount`.

        Returns
        -------
        Subclass of `~measurables.Measured`
            A new instance of this class.
        """

    def unit(self, new=None):
        """Concrete implementation."""
        if not new:
            return self._unit
        scale = metric.Unit(new) // self._unit
        amount = (scale * self)._amount
        return self._new(amount=amount, unit=new)

    def __bool__(self) -> bool:
        """Called for bool(self).
        
        A measured object is always `True` because every instance has a valid
        amount and unit, even though the amount may be 0.
        """
        return True

    def __abs__(self):
        return self._new(
            amount=abs(self._amount),
            unit=self._unit,
        )

    def __neg__(self):
        return self._new(
            amount=-self._amount,
            unit=self._unit,
        )

    def __pos__(self):
        return self._new(
            amount=+self._amount,
            unit=self._unit,
        )

    def __add__(self, other: 'Measured'):
        return self._new(
            amount=self._amount + other._amount,
            unit=self._unit,
        )

    def __radd__(self, other: typing.Any):
        return NotImplemented

    def __sub__(self, other: 'Measured'):
        return self._new(
            amount=self._amount - other._amount,
            unit=self._unit,
        )

    def __rsub__(self, other: typing.Any):
        return NotImplemented

    def __mul__(self, other: typing.Any):
        if isinstance(other, Measured):
            amount = self._amount * other._amount
            unit = self._unit * other._unit
            return self._new(
                amount=amount,
                unit=unit,
            )
        if isinstance(other, numbers.Number):
            return self._new(
                amount=self._amount * other,
                unit=self._unit,
            )
        return NotImplemented

    def __rmul__(self, other: typing.Any):
        if isinstance(other, numbers.Number):
            return self._new(
                amount=other * self._amount,
                unit=self._unit,
            )
        return NotImplemented

    def __truediv__(self, other: typing.Any):
        if isinstance(other, Measured):
            amount = self._amount / other._amount
            unit = self._unit / other._unit
            return self._new(
                amount=amount,
                unit=unit,
            )
        if isinstance(other, numbers.Number):
            return self._new(
                amount=self._amount / other,
                unit=self._unit,
            )
        return NotImplemented

    def __rtruediv__(self, other: typing.Any):
        return NotImplemented

    def __pow__(self, other: typing.Any):
        if isinstance(other, numbers.Number):
            return self._new(
                amount=self._amount ** other,
                unit=self._unit ** other
            )
        return NotImplemented

    def __rpow__(self, other: typing.Any):
        return NotImplemented

    def __str__(self) -> str:
        return f"{self._amount} [{self._unit}]"


Instance = typing.TypeVar('Instance', bound='Measured')


VT = typing.TypeVar('VT', bound=RealValued)
VT = RealValued


allowed = {m: numbers.Real for m in ['__lt__', '__le__', '__gt__', '__ge__']}
class Scalar(Measured, allowed=allowed):
    """A single numerical value and associated unit."""

    @typing.overload
    def __new__(
        cls: typing.Type[Instance],
        value: VT,
    ) -> Instance:
        """Create a new scalar object.
        
        Parameters
        ----------
        value : real number
            The numerical value of this scalar. The argument must implement the
            `~numbers.Real` interface.
        """

    @typing.overload
    def __new__(
        cls: typing.Type[Instance],
        value: VT,
        unit: typing.Union[str, metric.Unit],
    ) -> Instance:
        """Create a new scalar object.
        
        Parameters
        ----------
        value : real number
            The numerical value of this scalar. The argument must implement the
            `~numbers.Real` interface.

        unit : string or  `~metric.Unit`
            The metric unit of `value`.
        """

    @typing.overload
    def __new__(
        cls: typing.Type[Instance],
        instance: Instance,
    ) -> Instance:
        """Create a new scalar object.
        
        Parameters
        ----------
        instance : `~measurables.Scalar`
            An existing instance of this class.
        """

    _value: VT

    def __new__(cls, *args, **kwargs):
        """The concrete implementation of `~measurables.Scalar.__new__`."""
        if not kwargs and len(args) == 1 and isinstance(args[0], cls):
            instance = args[0]
            value = instance._value
            unit = instance.unit()
        else:
            attrs = list(args)
            if 'amount' in kwargs:
                kwargs['value'] = kwargs.pop('amount')
            attr_dict = {
                k: attrs.pop(0) if attrs
                else kwargs.pop(k, None)
                for k in ('value', 'unit')
            }
            value = attr_dict['value']
            unit = attr_dict['unit']
        self = super().__new__(cls, value, unit=unit)
        self._value = value
        return self

    def __lt__(self, other: Ordered) -> bool:
        if isinstance(other, Comparable):
            return self._amount < other
        return super().__lt__(other)

    def __le__(self, other: Ordered) -> bool:
        if isinstance(other, Comparable):
            return self._amount <= other
        return super().__le__(other)

    def __gt__(self, other: Ordered) -> bool:
        if isinstance(other, Comparable):
            return self._amount > other
        return super().__gt__(other)

    def __ge__(self, other: Ordered) -> bool:
        if isinstance(other, Comparable):
            return self._amount >= other
        return super().__ge__(other)

    def __float__(self):
        """Called for float(self)."""
        return float(self._value)

    def __int__(self):
        """Called for int(self)."""
        return int(self._value)

    def __round__(self, ndigits: int=None):
        """Called for round(self)."""
        return round(self._value, ndigits=ndigits)

    def __floor__(self):
        """Called for math.floor(self)."""
        return math.floor(self._value)

    def __ceil__(self):
        """Called for math.ceil(self)."""
        return math.ceil(self._value)

    def __trunc__(self):
        """Called for math.trunc(self)."""
        return math.trunc(self._value)

    # NOTE: This class is immutable, so in-place operations defer to forward
    # operations. That automatically happens for `__iadd__`, `__isub__`,
    # `__imul__`, and `__itruediv__`, but not for `__ipow__`, so we need to
    # explicitly define a trivial implementation here.
    def __ipow__(self, other: typing.Any):
        return super().__pow__(other)

    def __hash__(self):
        """Called for hash(self)."""
        return hash((self._value, str(self._unit)))


Instance = typing.TypeVar('Instance', bound='Measured')


class Vector(Measured):
    """Multiple numerical values and their associated unit."""

    _values: typing.Iterable[VT]

    @typing.overload
    def __new__(
        cls: typing.Type[Instance],
        values: typing.Iterable[VT],
    ) -> Instance:
        """Create a new vector object.
        
        Parameters
        ----------
        values : iterable of real numbers
            The numerical values of this vector. Each member must implement the
            `~numbers.Real` interface.
        """

    @typing.overload
    def __new__(
        cls: typing.Type[Instance],
        values: typing.Iterable[VT],
        unit: typing.Union[str, metric.Unit],
    ) -> Instance:
        """Create a new vector object.
        
        Parameters
        ----------
        values : iterable of real numbers
            The numerical values of this vector. Each member must implement the
            `~numbers.Real` interface.

        unit : string or `~metric.Unit`
            The metric unit of `values`.
        """

    @typing.overload
    def __new__(
        cls: typing.Type[Instance],
        instance: Instance,
    ) -> Instance:
        """Create a new vector object.

        Parameters
        ----------
        instance : `~measurables.Vector`
            An existing instance of this class.
        """

    def __new__(cls, *args, **kwargs):
        """The concrete implementation of `~measurables.Vector.__new__`."""
        if not kwargs and len(args) == 1 and isinstance(args[0], cls):
            instance = args[0]
            values = instance._values
            unit = instance.unit()
        else:
            attrs = list(args)
            if 'amount' in kwargs:
                kwargs['values'] = kwargs.pop('amount')
            attr_dict = {
                k: attrs.pop(0) if attrs
                else kwargs.pop(k, None)
                for k in ('values', 'unit')
            }
            values = attr_dict['values']
            unit = attr_dict['unit']
        self = super().__new__(cls, values, unit=unit)
        self._values = list(iterables.whole(self._amount))
        return self

    def __len__(self):
        """Called for len(self)."""
        return len(self._values)

    def __iter__(self):
        """Called for iter(self)."""
        return iter(self._values)

    def __contains__(self, item):
        """Called for item in self."""
        return item in self._values

    def __add__(self, other: typing.Any):
        if isinstance(other, Vector):
            values = [s + o for s, o in zip(self._values, other._values)]
            return self._new(amount=values, unit=self._unit)
        if isinstance(other, Measured):
            values = [s + other for s in self._values]
            return self._new(amount=values, unit=self._unit)
        return NotImplemented

    def __sub__(self, other: typing.Any):
        if isinstance(other, Vector):
            values = [s - o for s, o in zip(self._values, other._values)]
            return self._new(amount=values, unit=self._unit)
        if isinstance(other, Measured):
            values = [s - other for s in self._values]
            return self._new(amount=values, unit=self._unit)
        return NotImplemented

    def __mul__(self, other: typing.Any):
        if isinstance(other, Vector):
            values = [s * o for s, o in zip(self._values, other._values)]
            unit = self._unit * other._unit
            return self._new(amount=values, unit=unit)
        if isinstance(other, Scalar):
            other = float(other)
        if isinstance(other, RealValued):
            values = [s * other for s in self._values]
            return self._new(amount=values, unit=self._unit)
        return NotImplemented

    def __rmul__(self, other: typing.Any):
        if isinstance(other, Scalar):
            other = float(other)
        if isinstance(other, numbers.Number):
            values = [other * s for s in self._values]
            return self._new(amount=values, unit=self._unit)
        return NotImplemented

    def __truediv__(self, other: typing.Any):
        if isinstance(other, Vector):
            values = [s / o for s, o in zip(self._values, other._values)]
            unit = self._unit / other._unit
            return self._new(amount=values, unit=unit)
        if isinstance(other, Scalar):
            other = float(other)
        if isinstance(other, RealValued):
            values = [s / other for s in self._values]
            return self._new(amount=values, unit=self._unit)
        return NotImplemented

    def __pow__(self, other: typing.Any):
        if isinstance(other, numbers.Number):
            values = [s ** other for s in self._values]
            unit = self._unit ** other
            return self._new(amount=values, unit=unit)
        return NotImplemented

    def __bool__(self) -> bool:
        return super().__bool__()


class Measurement(Vector):
    """The result of measuring an object.

    While it is possible to directly instantiate this class, it serves primarily
    as the return type of `measurables.measure`, which accepts a much wider
    domain of arguments.
    """

    def __getitem__(self, index):
        """Called for index-based value access."""
        if isinstance(index, typing.SupportsIndex) and index < 0:
            index += len(self)
        values = self._values[index]
        iter_values = isinstance(values, typing.Iterable)
        return (
            [Scalar(value, self._unit) for value in values] if iter_values
            else Scalar(values, self.unit)
        )

    @property
    def values(self):
        """The numerical values of this measurement."""
        return self._values

    @property
    def unit(self):
        """The metric unit of this measurement.
        
        Unlike instances of `~measurables.Vector`, this object does not support
        updates to `unit`.
        """
        return super().unit()

    def __float__(self) -> float:
        """Represent a single-valued measurement as a `float`."""
        return self._cast_to(float)

    def __int__(self) -> int:
        """Represent a single-valued measurement as a `int`."""
        return self._cast_to(int)

    _T = typing.TypeVar('_T', int, float)
    def _cast_to(self, __type: _T) -> _T:
        """Internal method for casting to numeric type."""
        nv = len(self.values)
        if nv == 1:
            return __type(self.values[0])
        errmsg = f"Can't convert measurement with {nv!r} values to {__type}"
        raise TypeError(errmsg) from None

    def __str__(self) -> str:
        """A simplified representation of this object."""
        return f"{self.values} [{self.unit}]"


@typing.runtime_checkable
class Measurable(typing.Protocol):
    """Protocol defining a formally measurable object."""

    __slots__ = ()

    @abc.abstractmethod
    def __measure__(self) -> Measurement:
        """Create a measured object from input."""
        pass


def measurable(this):
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
    if all(measurable(i) for i in args):
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


