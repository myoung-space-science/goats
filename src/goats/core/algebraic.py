import abc
import math
import numbers
import operator as standard
import typing

import numpy


@typing.runtime_checkable
class Orderable(typing.Protocol):
    """Protocol for objects that support ordering.
    
    Instance checks against this ABC will return `True` iff the instance
    implements the following methods: `__lt__`, `__gt__`, `__le__`, `__ge__`,
    `__eq__`, and `__ne__`. It exists to support type-checking orderable objects
    outside the `~algebraic.Quantity` framework (e.g., pure numbers).
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


class Ordered(abc.ABC):
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


class Base(Ordered, Additive, Multiplicative):
    """Base class for algebraic quantities.

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


class Real(Base):
    """Abstract base class for all real-valued objects.
    
    This class is similar to ``numbers.Real``, but it does not presume to
    represent a single value.
    
    Concrete subclasses of this object must implement all the `~algebraic.Base`
    operators except for `__sub__` and `__rsub__` (defined here with respect to
    `__neg__`). Subclasses may, of course, override these base implementations.
    """

    def __sub__(self, other: SupportsNeg):
        """Called for self - other."""
        return self + -other

    def __rsub__(self, other: SupportsNeg):
        """Called for other - self."""
        return -self + other


Real.register(numbers.Real)
Real.register(numpy.ndarray) # close enough for now...


class Quantity(abc.ABC):
    """ABC for algebraic quantities.
    
    Concrete subclasses must define the built-in `__eq__` method and an
    `implement` method that computes the result of a given operation on specific
    operands. Operator mixin classes can leverage this class to programmatically
    implement operator methods required by `~algebraic.Base` or similar ABCs.
    Concrete subclasses of this class are always true in a boolean sense.
    """

    def __bool__(self) -> bool:
        """Always true for a valid instance."""
        return True

    def __abs__(self):
        """Called for abs(self)."""
        return self.implement(abs, 'arithmetic')

    def __pos__(self):
        """Called for +self."""
        return self.implement(standard.pos, 'arithmetic')

    def __neg__(self):
        """Called for -self."""
        return self.implement(standard.neg, 'arithmetic')

    def __ne__(self, other) -> bool:
        """Called for self != other."""
        return not self == other

    def __lt__(self, other) -> bool:
        """Called for self < other."""
        return self.implement(standard.lt, 'comparison', other)

    def __le__(self, other) -> bool:
        """Called for self <= other."""
        return self.implement(standard.le, 'comparison', other)

    def __gt__(self, other) -> bool:
        """Called for self > other."""
        return self.implement(standard.gt, 'comparison', other)

    def __ge__(self, other) -> bool:
        """Called for self >= other."""
        return self.implement(standard.ge, 'comparison', other)

    def __add__(self, other):
        """Called for self + other."""
        return self.implement(standard.add, 'forward', other)

    def __radd__(self, other):
        """Called for other + self."""
        return self.implement(standard.add, 'reverse', other)

    def __iadd__(self, other):
        """Called for self += other."""
        return self.implement(standard.add, 'inplace', other)

    def __sub__(self, other):
        """Called for self - other."""
        return self.implement(standard.sub, 'forward', other)

    def __rsub__(self, other):
        """Called for other - self."""
        return self.implement(standard.sub, 'reverse', other)

    def __isub__(self, other):
        """Called for self -= other."""
        return self.implement(standard.sub, 'inplace', other)

    def __mul__(self, other):
        """Called for self * other."""
        return self.implement(standard.mul, 'forward', other)

    def __rmul__(self, other):
        """Called for other * self."""
        return self.implement(standard.mul, 'reverse', other)

    def __imul__(self, other):
        """Called for self *= other."""
        return self.implement(standard.mul, 'inplace', other)

    def __truediv__(self, other):
        """Called for self / other."""
        return self.implement(standard.truediv, 'forward', other)

    def __rtruediv__(self, other):
        """Called for other / self."""
        return self.implement(standard.truediv, 'reverse', other)

    def __itruediv__(self, other):
        """Called for self /= other."""
        return self.implement(standard.truediv, 'inplace', other)

    def __pow__(self, other):
        """Called for self ** other."""
        return self.implement(standard.pow, 'forward', other)

    def __rpow__(self, other):
        """Called for other ** self."""
        return self.implement(standard.pow, 'reverse', other)

    def __ipow__(self, other):
        """Called for self **= other."""
        return self.implement(standard.pow, 'inplace', other)

    @abc.abstractmethod
    def implement(self, func: typing.Callable, mode: str, *others, **kwargs):
        """Implement a standard operator."""
        pass


class Scalar(Quantity):
    """ABC for single-valued algebraic quantities.
    
    This class `~algebraic.Quantity` to define two numeric cast operators
    (`_int__` and `__float__`) and four unary arithmetic operators (`__round__`,
    `__floor__`, `__ceil__`, and `__trunc__`) in terms of the abstract method
    `implement`.
    """

    def __int__(self):
        """Called for int(self)."""
        return self.implement(int, 'cast')

    def __float__(self):
        """Called for float(self)."""
        return self.implement(float, 'cast')

    def __round__(self):
        """Called for round(self)."""
        return self.implement(round, 'arithmetic')

    def __floor__(self):
        """Called for math.floor(self)."""
        return self.implement(math.floor, 'arithmetic')

    def __ceil__(self):
        """Called for math.ceil(self)."""
        return self.implement(math.ceil, 'arithmetic')

    def __trunc__(self):
        """Called for math.trunc(self)."""
        return self.implement(math.trunc, 'arithmetic')


