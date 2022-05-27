import abc
import collections.abc
import functools
import numbers
import operator as standard
import typing

import numpy
import numpy.typing

from goats.core import algebraic
from goats.core import iterables
from goats.core import metric
from goats.core import operations
from goats.core import utilities


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


class Quantifiable(algebraic.Quantity, iterables.ReprStrMixin):
    """A real-valued amount and the associated metric.
    
    This abstract base class represents the basis for quantifiable objects.
    Concrete subclasses must implement all the `~algebraic.Quantity` operators,
    and should do so in a way that self-consistently handles the instance
    metric.
    """

    def __init__(
        self,
        __amount: Real,
        __metric: algebraic.Multiplicative,
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


Instance = typing.TypeVar('Instance', bound='Quantity')


class Quantity(Quantifiable):
    """Real-valued data and the associated unit.
    
    This abstract base class represents the basis for all measurable quantities.
    It builds on `~measurable.Quantifiable` by specifying an instance of
    `~metric.Unit` as the metric.

    This class is intended as the primary ABC on which to build more specific
    abstract or concrete measurable objects. Concrete subclasses must implement
    all the abstract methods of `~algebraic.Quantity`. Implementors may use
    `~measurable.OperatorMixin` as a simple way to provide default versions of
    all required operators. Those default implementations inter-operate with all
    concrete types that define the `~measurable.Multiplicative` operators, and
    therefore automatically handle `~metric.Unit`. However, implementators may
    choose to overload certain operators in order to leverage properties of the
    `~metric.Unit` class.
    """

    @typing.overload
    def __init__(
        self: Instance,
        data: Real,
        unit: metric.UnitLike=None,
    ) -> None:
        """Initialize this instance from arguments."""

    @typing.overload
    def __init__(
        self: Instance,
        instance: Instance,
    ) -> None:
        """Initialize this instance from an existing one."""

    def __init__(self, *args, **kwargs) -> None:
        init = self._parse(*args, **kwargs)
        super().__init__(*init)
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

    @property
    def data(self):
        """This quantity's data."""
        return self._amount

    Attrs = typing.TypeVar('Attrs', bound=tuple)
    Attrs = typing.Tuple[Real, metric.UnitLike]

    def _parse(self, *args, **kwargs) -> Attrs:
        """Parse input arguments to initialize this instance."""
        if not kwargs and len(args) == 1 and isinstance(args[0], type(self)):
            instance = args[0]
            return tuple(
                getattr(instance, name)
                for name in ('_amount', '_metric')
            )
        pos = list(args)
        data = kwargs.get('data') or pos.pop(0)
        unit = metric.Unit(kwargs.get('unit') or pos.pop(0) or '1')
        return data, unit

    def unit(self, unit: metric.UnitLike=None):
        """Get or set the unit of this object's values."""
        if not unit:
            return self._metric
        if unit == self._metric:
            return self
        return self._update_unit(unit)

    def _update_unit(self, unit: metric.UnitLike=None):
        """Update the instance unit. Extracted for overloading."""
        new = metric.Unit(unit)
        self._amount *= new // self._metric
        self._metric = new
        return self


class OperatorMixin:
    """A mixin class that defines operators for quantifiable objects.
    
    This class implements the `~algebraic.Quantity` operators with the following
    rules:
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

    factory = operations.Interface('data', 'unit')
    # factory.guarantee(Quantity, 'data')

    unary = factory.unary
    unary.rules.register(Quantity, 'data')
    __abs__ = unary.apply(standard.abs)
    __pos__ = unary.apply(standard.pos)
    __neg__ = unary.apply(standard.neg)

    comparison = factory.comparison
    comparison.rules.register([Quantity, Quantity], 'data')
    comparison.rules.register([Quantity, Real], 'data')
    comparison.rules.register([Real, Quantity], 'data')
    __lt__ = comparison.apply(standard.lt)
    __le__ = comparison.apply(standard.le)
    __gt__ = comparison.apply(standard.gt)
    __ge__ = comparison.apply(standard.ge)
    __eq__ = comparison.apply(standard.eq)
    __ne__ = comparison.apply(standard.ne)

    numeric = factory.numeric
    numeric.rules.register([Quantity, Quantity], 'data', 'unit')
    numeric.rules.register([Quantity, Real], 'data')
    numeric.rules.register([Real, Quantity], 'data')
    __add__ = numeric.apply(standard.add)
    __radd__ = numeric.apply(standard.add, 'reverse')
    __sub__ = numeric.apply(standard.sub)
    __rsub__ = numeric.apply(standard.sub, 'reverse')
    __mul__ = numeric.apply(standard.mul)
    __rmul__ = numeric.apply(standard.mul, 'reverse')
    _truediv = numeric.child
    _truediv.rules.suppress([Real, Quantity])
    __truediv__ = _truediv.apply(standard.truediv)
    __rtruediv__ = _truediv.apply(standard.truediv, 'reverse')
    _pow = numeric.child
    _pow.rules.modify([Quantity, Real], 'data', 'unit')
    _pow.rules.suppress([Real, Quantity])
    _pow.rules.suppress([Quantity, Quantity])
    __pow__ = _pow.apply(pow)
    __rpow__ = _pow.apply(pow, 'reverse')


class Scalar(Quantity):
    """ABC for a single-valued measurable quantity."""

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


class Vector(Quantity):
    """ABC for a multi-valued measurable quantity.
    
    Notes
    -----
    This class converts `data` into a `numpy.ndarray` via a `list` during
    instantiation. It may therefore become a bottleneck for large 1-D objects
    and may produce unexpected results for higher-dimension objects. In those
    cases, an array-like class derived from `~measurable.Quantity` that
    incorporates native `numpy` operators may be more appropriate.
    """

    @abc.abstractmethod
    def __len__(self) -> int:
        """Called for len(self)."""
        pass

    Item = typing.TypeVar('Item', bound=typing.Iterable)
    Item = typing.Union[Scalar, typing.List[Scalar]]

    @abc.abstractmethod
    def __getitem__(self, index) -> Item:
        """Called for index-based value access."""
        pass


@typing.runtime_checkable
class Measurable(typing.Protocol):
    """Protocol for a quantity that supports direct measurement.
    
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

