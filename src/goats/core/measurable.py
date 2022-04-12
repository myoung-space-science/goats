import abc
import collections.abc
import functools
import numbers
import operator
import typing

import numpy
import numpy.typing

from goats.core import algebraic
from goats.core import iterables
from goats.core import metric


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
OprType = typing.TypeVar('OprType', bound=typing.Callable)
OprType = typing.Callable[..., RT]


def _comparison(opr: OprType):
    """Implement a comparison operator."""
    @same('_metric', allowed=numbers.Real)
    def func(a: Quantifiable, b):
        if isinstance(b, Quantifiable):
            return opr(a._amount, b._amount)
        if isinstance(b, algebraic.Orderable):
            return opr(a._amount, b)
        return NotImplemented
    func.__name__ = f"__{opr.__name__}__"
    func.__doc__ = opr.__doc__
    return func


def _unary(opr: OprType):
    """Implement a unary arithmetic operator."""
    def func(a: Quantifiable):
        return type(a)(opr(a._amount), a._metric)
    func.__name__ = f"__{opr.__name__}__"
    func.__doc__ = opr.__doc__
    return func


class Method:
    """An operator wrapper that simplifies evaluation."""

    def __init__(self, method: OprType) -> None:
        self._method = method

    def evaluate(self, attr: str, *args):
        """Apply the operator to arguments or their attributes."""
        operands = [getattr(arg, attr, arg) for arg in args]
        return self._method(*operands)

    def preserves(self, *args, **kwargs):
        """Require that this operation preserve certain attributes."""
        def wrapper(func):
            decorated = same(*args, **kwargs)
            return decorated(func)
        return wrapper


def _preserve_metric(opr: OprType):
    """Implement a forward operator that preserves the instance metric."""
    method = Method(opr)
    @method.preserves('_metric', allowed=numbers.Real)
    def func(a: Quantifiable, b):
        return type(a)(
            method.evaluate('_amount', a, b),
            a._metric
        )
    return func


def _update_metric(opr: OprType):
    """Implement a forward operator that updates the instance metric."""
    method = Method(opr)
    def func(a: Quantifiable, b):
        return type(a)(
            method.evaluate('_amount', a, b),
            method.evaluate('_metric', a, b),
        )
    return func


def _name_tbd(*preserved: str):
    """"""
    def _inside(opr: OprType):
        method = Method(opr)
        @method.preserves(*preserved, allowed=Real)
        def func(a: Quantifiable, b):
            if not isinstance(b, (Quantifiable, Real)):
                return NotImplemented
            args = [
                getattr(a, attr) if attr in preserved
                else method.evaluate(attr, a, b)
                for attr in ('_amount', '_metric')
            ]
            return type(a)(*args)
        return func
    return _inside


Wrapper = typing.TypeVar('Wrapper', bound=typing.Callable)
Wrapper = typing.Callable[..., typing.Callable[..., Quantifiable]]


Rules = typing.TypeVar('Rules', bound=dict)
Rules = typing.Dict[
    typing.Tuple[type],
    Wrapper,
]


_operators: typing.Dict[OprType, Rules] = {
    operator.add: {
        (Quantifiable, Quantifiable): '_metric',
        (Quantifiable, Real): '_metric',
    },
    operator.sub: {
        (Quantifiable, Quantifiable): '_metric',
        (Quantifiable, Real): '_metric',
    },
    operator.mul: {
        (Quantifiable, Quantifiable): None,
        (Quantifiable, Real): '_metric',
    },
    operator.truediv: {
        (Quantifiable, Quantifiable): None,
        (Quantifiable, Real): '_metric',
    },
    operator.pow: {
        (Quantifiable, Real): None,
    },
}


def _get_rule(rules: dict, *args) -> Wrapper:
    types = tuple(type(arg) for arg in args)
    if types in rules:
        return rules[types]
    for key, rule in rules.items():
        if all(issubclass(t, k) for t, k in zip(types, key)):
            return rule


def _forward(opr: OprType):
    """Implement a forward operator."""
    rules = _operators.get(opr)
    def func(a: Quantifiable, b):
        rule = _get_rule(rules, a, b)
        return rule(opr)(a, b) if rule else NotImplemented
    func.__name__ = f"__{opr.__name__}__"
    func.__doc__ = opr.__doc__
    return func


def _forward(opr: OprType):
    """Implement a forward operator."""
    rules = _operators.get(opr)
    def func(a: Quantifiable, b):
        preserved = _get_rule(rules, a, b)
        method = _name_tbd(*iterables.whole(preserved))
        return method(opr)(a, b) if preserved is not None else NotImplemented
    func.__name__ = f"__{opr.__name__}__"
    func.__doc__ = opr.__doc__
    return func


# This could eventually just be `Operator` and it could take `_attrs` as an
# argument to `__init__`.
class QuantifiableOperator:
    """An operator that knows how to handle quantifiable objects."""

    def __init__(self, opr: OprType) -> None:
        self._opr = opr

    _attrs = ('_amount', '_metric')

    def __call__(self, *args):
        """Apply the operator to arguments or their attributes."""
        updated = [self._evaluate(attr, *args) for attr in self._attrs]
        return type(args[0])(*updated)

    def _evaluate(self, attr: str, *args):
        operands = [getattr(arg, attr, arg) for arg in args]
        return self._opr(*operands)


def _forward(opr: OprType):
    """Implement a forward operator."""
    def func(a: Quantifiable, b):
        if isinstance(b, Quantifiable):
            return type(a)(
                opr(a._amount, b._amount),
                opr(a._metric, b._metric),
            )
        if isinstance(b, Real):
            return type(a)(
                opr(a._amount, b),
                a._metric,
            )
        return NotImplemented
    func.__name__ = f"__{opr.__name__}__"
    func.__doc__ = opr.__doc__
    return func


def _inplace(opr: OprType):
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


def _reverse(opr: OprType):
    """Implement a standard reverse operator."""
    def func(b: Quantifiable, a):
        return (
            type(b)(opr(a, b._amount), b._metric) if isinstance(a, Real)
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
        data, *args = args
        unit = metric.Unit(
            args[0] if len(args) == 1
            else kwargs.get('unit') or '1'
        )
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


class Comparison:
    """A comparison operation between two quantifiable objects."""

    def __init__(self, opr: OprType) -> None:
        self._opr = opr

    def __call__(self, a: Quantifiable, b):
        """Apply the operator to arguments or their attributes."""
        if isinstance(b, Quantifiable):
            return self._opr(a._amount, b._amount)
        if isinstance(b, algebraic.Orderable):
            return self._opr(a._amount, b)
        return NotImplemented


class Unary:
    """An arithmetic operation one a single quantifiable object."""

    def __init__(self, opr: OprType) -> None:
        self._opr = opr

    def __call__(self, a: Quantifiable):
        """Apply the operator to the argument."""
        return type(a)(self._opr(a._amount), a._metric)


class Binary:
    """An arithmetic operation between two quantifiable objects."""

    def __init__(self, opr: OprType) -> None:
        self._opr = opr

    # _attrs = ('_amount', '_metric')

    def __call__(self, a: Quantifiable, b):
        """Apply the operator to arguments or their attributes."""
        # updated = [self._evaluate(attr, *args) for attr in self._attrs]
        # return type(args[0])(*updated)
        if isinstance(b, Quantifiable):
            return type(a)(
                self._opr(a._amount, b._amount),
                self._opr(a._metric, b._metric),
            )
        if isinstance(b, Real):
            return type(a)(
                self._opr(a._amount, b),
                self._opr(a._metric, b),
            )
        return NotImplemented

    # def _evaluate(self, attr: str, a, b):
    #     operands = [getattr(arg, attr, arg) for arg in (a, b)]
    #     try:
    #         return self._opr(*operands)
    #     except TypeError:
    #         return getattr(a, attr)


class Operator: # Really only binary for now.
    """"""

    def __init__(
        self,
        rules: typing.Dict[typing.Tuple[type], typing.Tuple[str]],
    ) -> None:
        self._rules = rules

    _attrs = ('_amount', '_metric')

    def implement(self, opr: OprType, *modes: str):
        """Implement `opr` for the requested modes."""
        forward = self._build_forward(opr)
        forward.__name__ = f"__{opr.__name__}__"
        forward.__doc__ = opr.__doc__
        reverse = (
            self._build_reverse(opr) if 'reverse' in modes
            else self._suppress(opr)
        )
        reverse.__name__ = f"__r{opr.__name__}__"
        reverse.__doc__ = opr.__doc__
        inplace = (
            self._build_inplace(opr) if 'inplace' in modes
            else self._suppress(opr)
        )
        inplace.__name__ = f"__i{opr.__name__}__"
        inplace.__doc__ = opr.__doc__
        return forward, reverse, inplace

    def _get_updatable(self, *args):
        """Get the names of updatable attributes, based on arg types."""
        types = tuple(type(arg) for arg in args)
        if types in self._rules:
            return self._rules[types]
        for key, names in self._rules.items():
            if all(issubclass(t, k) for t, k in zip(types, key)):
                return names
        raise TypeError(f"No operator rule for {types}") from None

    def _build_forward(self, opr: OprType):
        """Build the forward version of `opr`."""
        def func(a: Quantifiable, b):
            updatable = self._get_updatable(a, b)
            values = []
            for attr in self._attrs:
                if attr in updatable:
                    operands = [getattr(arg, attr, arg) for arg in (a, b)]
                    values.append(opr(*operands))
                else:
                    values.append(getattr(a, attr))
            return type(a)(*values)
        return func

    def _build_reverse(self, opr: OprType):
        """Build the reverse version of `opr`."""
        def func(b: Quantifiable, a):
            return (
                type(b)(opr(a, b._amount), b._metric) if isinstance(a, Real)
                else NotImplemented
            )
        return func

    def _build_inplace(self, opr: OprType):
        """Build the in-place version of `opr`."""
        def func(a: Quantifiable, b):
            result = self._build_forward(opr)(a, b)
            a._amount = result._amount
            a._metric = result._metric
            return a
        return func

    def _suppress(self, opr: OprType):
        """Explicitly suppress use of `opr`."""
        def func(*args, **kwargs):
            return NotImplemented
        return func


binary = Operator(
    {
        (Quantifiable, Quantifiable): ('_amount', '_metric'),
        (Quantifiable, Real): ('_amount',),
    },
)
pow_tmp = Operator(
    {
        (Quantifiable, Real): ('_amount', '_metric'),
    },
)
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

    __add__, __radd__, __iadd__ = binary.implement(
        operator.add, 'reverse', 'inplace',
    )

    __sub__, __rsub__, __isub__ = binary.implement(
        operator.sub, 'reverse', 'inplace',
    )

    __mul__, __rmul__, __imul__ = binary.implement(
        operator.mul, 'reverse', 'inplace',
    )

    __truediv__, __rtruediv__, __itruediv__ = binary.implement(
        operator.truediv, 'inplace',
    )

    __pow__, __rpow__, __ipow__ = pow_tmp.implement(
        operator.pow, 'inplace',
    )


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


