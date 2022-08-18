import abc
import collections.abc
import math
import numbers
import operator as standard
import typing

import numpy
import numpy.typing

from goats.core import algebraic
from goats.core import iterables
from goats.core import metadata


Self = typing.TypeVar('Self', bound='SupportsNeg')


@typing.runtime_checkable
class SupportsNeg(typing.Protocol):
    """Protocol for objects that support negation (``-self``)."""

    __slots__ = ()

    @abc.abstractmethod
    def __neg__(self: Self) -> Self:
        pass


class Real(algebraic.Quantity):
    """Abstract base class for all real-valued objects.
    
    This class is similar to ``numbers.Real``, but it does not presume to
    represent a single value.
    
    Concrete subclasses of this object must implement all the
    `~algebraic.Quantity` operators except for `__sub__` and `__rsub__` (defined
    here with respect to `__neg__`). Subclasses may, of course, override these
    base implementations.
    """

    def __sub__(self, other: SupportsNeg):
        """Called for self - other."""
        return self + -other

    def __rsub__(self, other: SupportsNeg):
        """Called for other - self."""
        return -self + other


Real.register(numbers.Real)
Real.register(numpy.ndarray) # close enough for now...


class Measurement(collections.abc.Sequence, iterables.ReprStrMixin):
    """A sequence of values and their associated unit.
    
    Notes
    -----
    * The function `~measurable.measure` returns an instance of this class.
    * A `~measurable.Measurement` is not a subclass of `~measurable.Quantity`,
      even though it has real values and a unit, because it does not require or
      define any of the `~algebraic.Quantity` operators. Its existence outside
      of the `~measurable.Quantity` hierarchy is consistent with the notion that
      it is the result of measuring a measurable quantity, rather than an object
      that exists to be measured.
    """

    def __init__(
        self,
        values: typing.Iterable[numbers.Real],
        unit: metadata.UnitLike,
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
        return metadata.Unit(self._unit)

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


class Quantifiable(abc.ABC):
    """ABC for measurable quantities.
    
    Concrete subclasses must define the built-in `__eq__` method and an
    `implement` method that computes the result of a given operation on specific
    operands. Operator mixin classes can leverage this class to programmatically
    implement operator methods required by other ABCs (e.g.,
    `~algebraic.Quantity`). Concrete subclasses of this class are always true in
    a boolean sense.
    """

    def __bool__(self) -> bool:
        """Always true for a valid instance."""
        return True

    def __ne__(self, other) -> bool:
        """Called for self != other."""
        return not self == other

    @abc.abstractmethod
    def implement(self, func: typing.Callable, mode: str, *others, **kwargs):
        """Implement a standard operator."""
        pass


class AlgebraicOperators(Quantifiable, algebraic.Quantity):
    """Algebraic operators for measurable quantities.
    
    This class extends `~algebraic.Quantity` in order to simplify the creation
    of measurable quantities.
    """

    def __abs__(self):
        """Called for abs(self)."""
        return self.implement(abs, 'arithmetic')

    def __pos__(self):
        """Called for +self."""
        return self.implement(standard.pos, 'arithmetic')

    def __neg__(self):
        """Called for -self."""
        return self.implement(standard.neg, 'arithmetic')

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


class ScalarOperators(Quantifiable):
    """Operators for single-valued measurable quantities."""

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


class Quantified(Quantifiable, iterables.ReprStrMixin):
    """A concrete realization of a quantifiable object.

    This class implements the `~algebraic.Quantity` operators (via
    `~measurable.Quantifiable`) with the following rules:
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

    def __init__(self, __data: Real) -> None:
        self._data = __data
        self._meta = None

    @property
    def data(self):
        """This quantity's data."""
        return self._data

    @property
    def meta(self):
        """This quantity's metadata attributes."""
        if self._meta is None:
            self._meta = metadata.OperatorFactory(type(self))
        return self._meta

    def __eq__(self, other) -> bool:
        """Called for self == other."""
        if not isinstance(other, Quantified):
            return other == self.data
        if other.data != self.data:
            return False
        for name in self.meta.parameters:
            v = getattr(self, name)
            if hasattr(other, name) and getattr(other, name) != v:
                return False
        return True

    def implement(self, func: typing.Callable, mode: str, *others, **kwargs):
        """Implement a standard operation."""
        name = func.__name__
        if mode == 'cast':
            return func(self.data)
        if mode == 'arithmetic':
            data = func(self.data, **kwargs)
            meta = self.meta[name].evaluate(self, **kwargs)
            return type(self)(data, **meta)
        operands = [self] + list(others)
        args = [
            i.data if isinstance(i, Quantifiable) else i
            for i in operands
        ]
        if mode == 'comparison':
            self.meta.check(self, *others)
            return func(*args)
        if mode == 'forward':
            data = func(*args, **kwargs)
            meta = self.meta[name].evaluate(*operands, **kwargs)
            return type(self)(data, **meta)
        if mode == 'reverse':
            data = func(*reversed(args), **kwargs)
            meta = self.meta[name].evaluate(*reversed(operands), **kwargs)
            return type(self)(data, **meta)
        if mode == 'inplace':
            data = func(*args, **kwargs)
            meta = self.meta[name].evaluate(*operands, **kwargs)
            self._data = data
            for k, v in meta:
                setattr(self, k, v)
            return self
        raise ValueError(f"Unknown operator mode {mode!r}")


Instance = typing.TypeVar('Instance', bound='Quantity')


class Quantity(Quantified, AlgebraicOperators):
    """A measurable quantity."""

    @typing.overload
    def __init__(
        self: Instance,
        __data: Real,
        *,
        unit: metadata.UnitLike=None,
    ) -> None: ...

    @typing.overload
    def __init__(
        self: Instance,
        instance: Instance,
    ) -> None: ...

    def __init__(self, __a, **meta) -> None:
        """Initialize this instance from arguments or an existing instance."""
        super().__init__(__a.data if isinstance(__a, Quantified) else __a)
        self.meta['true divide'].suppress(Real, Quantifiable)
        self.meta['power'].suppress(Quantifiable, Quantifiable)
        self.meta['power'].suppress(Real, Quantifiable)
        self.meta['power'].suppress(
            Quantifiable,
            typing.Iterable,
            symmetric=True
        )
        parsed = self.parse_attrs(__a, meta, unit='1')
        self._unit = parsed['unit']
        """This quantity's metric unit."""
        self.meta.register('unit')
        self.display.register('data', 'unit')
        self.display['__str__'] = "{data} [{unit}]"
        self.display['__repr__'] = "{data}, unit='{unit}'"
        self.display['__repr__'].separator = ', '

    @property
    def unit(self):
        """This quantity's metric unit."""
        return metadata.Unit(self._unit)

    def parse_attrs(self, this, meta: dict, **targets):
        """Get instance attributes from initialization arguments."""
        if isinstance(this, Quantified):
            return {k: getattr(this, k) for k in targets}
        return {k: meta.get(k, v) for k, v in targets.items()}

    def __getitem__(self, unit: metadata.UnitLike):
        """Set the unit of this object's values.
        
        Notes
        -----
        Using this special method to update the instance unit causes it to act
        like an in-place version of ``__setitem__``. It supports a simple and
        relatively intuitive syntax but is arguably an abuse of notation.
        """
        if unit != self._unit:
            new = metadata.Unit(unit)
            self.apply_unit(new)
            self._unit = new
        return self

    def apply_unit(self, new: metadata.Unit):
        """Update data values based on the new unit.
        
        Extracted for overloading, to allow subclasses to customize how to
        update the instance unit and to apply the corresponding conversion
        factor. For example, some subclasses may wish to simply store an updated
        scale factor, and to defer application of the scale factor to the data
        object if doing so here would be inefficient.
        """
        self._data *= new // self._unit

    def __measure__(self):
        """Create a measurement from this quantity's data and unit."""
        value = iterables.whole(self.data)
        return Measurement(value, self.unit)


class Scalar(Quantity, ScalarOperators):
    """A single-valued measurable quantity"""

    @typing.overload
    def __init__(
        self: Instance,
        __data: numbers.Real,
        *,
        unit: metadata.UnitLike=None,
    ) -> None: ...

    @typing.overload
    def __init__(
        self: Instance,
        instance: Instance,
    ) -> None: ...

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


@typing.runtime_checkable
class _Formal(typing.Protocol):
    """Definition of an object that supports direct measurement.
    
    If an object defines the `__measure__` method, `~measurable.measure` will
    call it instead of attempting to parse the object. Concrete implementations
    of `__measure__` should return an instance of `~measurable.Measurement`
    """

    @abc.abstractmethod
    def __measure__(self) -> Measurement:
        """Measure this object."""
        pass


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
    if isinstance(args[-1], metadata.UnitLike):
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
    conforms to the measurable protocol, and call a special method if so.
    Otherwise, it will attempt to parse `args` into one or more values and a
    corresponding unit.
    """
    first = args[0]
    if len(args) == 1 and isinstance(first, _Formal):
        return first.__measure__()
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
    n_units = sum(types.count(t) for t in (str, metadata.Unit))

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
    implicit = not any(isinstance(arg, (str, metadata.Unit)) for arg in args)
    explicit = last in ['1', metadata.Unit('1')]
    if implicit or explicit:
        return '1'
    return str(last)

