import abc
import collections.abc
import contextlib
import functools
import numbers
import operator as standard
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


class Operand:
    """A class representing a measurable operand."""

    def __init__(self, operand, attributes: typing.Container[str]) -> None:
        self.operand = operand
        self.attributes = attributes
        self._type = type(operand)

    def validate(self, other, ignored: typing.Iterable[typing.Type]=None):
        """Make sure `other` is a valid co-operand."""
        allowed = self._type if ignored is None else (self._type, *ignored)
        if not isinstance(other, allowed):
            return False
        if isinstance(other, self._type):
            for name in self.attributes:
                if not self._comparable(other, name):
                    raise ComparisonError(
                        self.operand, other, name
                    ) from None
        return True

    def _comparable(self, that, name: str) -> bool:
        """Determine whether the instances are comparable."""
        return getattrval(self.operand, name) == getattrval(that, name)


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
        def wrapper(*args):
            if len(args) == 1:
                return func(*args)
            this, *those = args
            target = Operand(this, self.names)
            valid = [
                target.validate(that, ignored=self.allowed)
                for that in those
            ]
            return func(this, *those) if all(valid) else NotImplemented
        return wrapper


RT = typing.TypeVar('RT')
Method = typing.TypeVar('Method', bound=typing.Callable)
Method = typing.Callable[..., RT]


T = typing.TypeVar('T')


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


def getattrval(
    __object: T,
    __name: str,
    *args,
    **kwargs
) -> typing.Union[typing.Any, T]:
    """Get an appropriate value based on the given object type.
    
    Parameters
    ----------
    __object : Any
        The object from which to retrieve the target attribute, if available.

    __name : string
        The name of the target attribute.

    *args
        Optional positional arguments to pass to the target attribute, if it is
        callable.

    **kwargs
        Optional keyword arguments to pass to the target attribute, if it is
        callable.

    Returns
    -------
    Any
        The value of the attribute on the given object, or the object itself.
        See Notes for further explanation.

    Notes
    -----
    This function will attempt to retrieve the named attribute from the given
    object. If the attribute exists and is callable (e.g., a class method), this
    function will call the attribute with `*args` and `**kwargs`, and return the
    result. If the attribute exists and is not callable, this function will
    return it as-is. If the attribute does not exist, this function will return
    the given object. This case supports programmatic use when the calling code
    does not know the type of object until runtime.

    Examples
    --------
    TODO
    """
    attr = getattr(__object, __name, __object)
    return attr(*args, **kwargs) if callable(attr) else attr


@typing.overload
def setattrval(__object: T, __name: str, __value) -> None: ...


@typing.overload
def setattrval(__object: T, __name: str, __value, *args, **kwargs) -> None: ...


@typing.overload
def setattrval(__object: T, __name: str, *args, **kwargs) -> None: ...


def setattrval(*args, **kwargs):
    """Set an appropriate value based on the given object type.
    
    Parameters
    ----------
    __object : Any
        The object on which to set the target attribute.

    __name : string
        The name of the target attribute.

    __value : Any
        The new value of the target attribute.

    *args
        Positional arguments to pass the target attribute, if it is callable.
        See Notes for further explanation.

    **kwargs
        Keyword arguments to pass to the target attribute, if it is callable.
        See Notes for further explanation.

    Returns
    -------
    None

    Notes
    -----
    This function will attempt to set the named attribute on the given object.
    If the attribute exists and is callable (e.g., a class method), this
    function will call the attribute with all positional arguments after
    `__object` and `__name`, as well as any given keyword arguments. The user
    may pass the new value as the first positional argument or as a keyword
    argument, in order to support as many forms of callable attributes as
    possible. If the attribute exists and is not callable, this function will
    set the new value from the first positional argument after `__object` and
    `__name`. If the attribute does not exist, this function will raise an
    ``AttributeError``.

    Examples
    --------
    TODO
    """
    obj, name, *args = args
    attr = getattr(obj, name)
    if callable(attr):
        attr(*args, **kwargs)
    else:
        setattr(obj, name, args[0])


Signature = typing.TypeVar('Signature', bound=tuple)
Signature = typing.Tuple[type, type]


Rules = typing.TypeVar('Rules', bound=typing.Mapping)
Rules = typing.Mapping[Signature, typing.Collection[str]]


class Updater(collections.abc.Mapping):
    """A mapping from type signature to updatable attributes."""

    def __init__(self, rules: Rules) -> None:
        self.rules = rules or {}
        self._names = None
        self._fixed = None

    @property
    def names(self):
        """The (unordered) names of all known attributes."""
        if self._names is None:
            names = {name for names in self.rules.values() for name in names}
            self._names = tuple(names) + self.fixed
        return self._names

    @property
    def fixed(self):
        """The names of immutable attributes."""
        if self._fixed is None:
            self._fixed = tuple(self.rules.get(None, ()))
        return self._fixed

    def __len__(self) -> int:
        return len(self.rules)

    def __iter__(self) -> typing.Iterator:
        return iter(self.rules)

    def __getitem__(self, types: Signature):
        """The updatable attributes for the given arguments types."""
        if types in self.rules:
            return self.rules[types]
        for rule, attributes in self.rules.items():
            if isinstance(rule, type) and issubclass(types[0], rule):
                return attributes
            if all(issubclass(t, r) for t, r in zip(types, rule)):
                return attributes
        raise KeyError(
            f"No updatable attributes for types {types!r}"
        ) from None


class OperandError(Exception):
    """Operands are incompatible with operator."""


class Implementation:
    """A generalized operator implementation."""

    def __init__(self, rules: Rules):
        self.attributes = Updater(rules)

    def apply(self, method: Method):
        @same(*self.attributes.fixed)
        def func(*args):
            try:
                return self._apply(method, *args)
            except metric.UnitError as err:
                raise OperandError(err) from err
        return func

    def _apply(self, method: Method, *args):
        """Internal method-application logic."""
        types = tuple(type(i) for i in args)
        if types not in self.attributes:
            return NotImplemented
        updatable = list(self.attributes[types])
        reference = self._get_reference(*args)
        # NOTE: This uses a `dict` because we can't guarantee that
        # `self.attribute.names` will be in the correct order. An alternate
        # approach could be to get parameter order from `inspect.signature`.
        return {
            name: method(*[getattrval(arg, name) for arg in args])
            if name in updatable
            else getattrval(reference, name)
            for name in self.attributes.names
        }

    def _get_reference(self, *args):
        """Get a reference quantity.
        
        This method relies on the assuption that at least one operand in any
        operation that this class handles is an instance of a
        `~measurable.Quantity` or a subclass.
        """
        instance = (arg for arg in args if isinstance(arg, Quantity))
        return next(instance)


class Operator(abc.ABC):
    """Abstract base class for operators."""

    def __init__(self, method: Method, rules: Rules=None) -> None:
        self.method = method
        self.implementation = Implementation(rules)

    def evaluate(self, *args, **kwargs):
        """Call the implemented method."""
        operator = self.implementation.apply(self.method)
        return operator(*args, **kwargs)

    @abc.abstractmethod
    def implement(self, *args, **kwargs):
        """Impement the instance method."""
        pass

    def suppress(self, *args, **kwargs):
        """Suppress the instance method"""
        implemented = self.implement(*args, **kwargs)
        def func(*a, **k):
            return NotImplemented
        func.__name__ = implemented.__name__
        func.__doc__ = implemented.__doc__
        return func


class Unary(Operator):
    """A concrete implementation of a unary arithmetic operator."""

    def implement(self) -> typing.Callable:
        def operator(a: Quantity):
            result = self.evaluate(a)
            if isinstance(result, typing.Mapping):
                return type(a)(**result)
            return result
        operator.__name__ = f"__{self.method.__name__}__"
        operator.__doc__ = self.method.__doc__
        return operator


class Cast(Operator):
    """A concrete implementation of a unary type-casting operator."""

    def implement(self):
        def operator(a: Quantity):
            result = self.evaluate(a)
            if isinstance(result, typing.Mapping):
                return result['data']
            return result
        operator.__name__ = f"__{self.method.__name__}__"
        operator.__doc__ = self.method.__doc__
        return operator


class Numeric(Operator):
    """A concrete implementation of a binary numeric operator."""

    def implement(self, mode: str='forward') -> typing.Callable:
        def forward(a: Quantity, b):
            result = self.evaluate(a, b)
            if isinstance(result, typing.Mapping):
                return type(a)(**result)
            return result
        def reverse(b: Quantity, a):
            result = self.evaluate(a, b)
            if isinstance(result, typing.Mapping):
                return type(b)(**result)
            return result
        def inplace(a: Quantity, b):
            result = self.evaluate(a, b)
            if isinstance(result, typing.Mapping):
                for name in self.implementation.attributes.names: # YIKES
                    setattrval(a, name, result.get(name))
                return a
            return result
        if mode == 'forward':
            operator = forward
            operator.__name__ = f"__{self.method.__name__}__"
        elif mode == 'reverse':
            operator = reverse
            operator.__name__ = f"__r{self.method.__name__}__"
        elif mode == 'inplace':
            operator = inplace
            operator.__name__ = f"__i{self.method.__name__}__"
        else:
            raise ValueError(
                f"Unknown implementation mode {mode!r}"
            ) from None
        operator.__doc__ = self.method.__doc__
        return operator


class Comparison(Operator):
    """A concrete implementation of a binary comparison operator."""

    def implement(self) -> typing.Callable:
        def operator(a: Quantity, b):
            result = self.evaluate(a, b)
            if isinstance(result, typing.Mapping):
                return result['data']
            return result
        operator.__name__ = f"__{self.method.__name__}__"
        operator.__doc__ = self.method.__doc__
        return operator


GT = typing.TypeVar('GT', bound=Operator)


class OperatorFactory(typing.Generic[GT]):
    """A factory for creating generic operators."""

    def __init__(self, __type: typing.Type[GT], rules: Rules=None) -> None:
        self._implement = __type
        init = dict(rules or {})
        self.rules = {
            signature: list(attributes)
            for signature, attributes in init.items()
        }

    def operator(self, method: Method):
        """Create an operator from the given method."""
        return self._implement(method, rules=self.rules)

    def add_rules(self, rules: Rules):
        """Add one or more implementation rule(s).
        
        This method will insert the given rules into the current dictionary of
        rules, overwriting existing rules if necessary.
        """
        self.rules.update(rules)
        return self

    def update_rules(self, rules: Rules):
        """Update one or more implementation rule(s).
        
        This method will append the attributes in each of the given rules to the
        current attributes in that rule. The rule must exist.
        """
        for signature, attributes in rules.items():
            self.rules[signature].extend(attributes)
        return self


RULES = {
    'unary': {
        Quantity: ['data'],
        None: ['unit'],
    },
    'cast': {
        Quantity: ['data'],
        None: ['unit'],
    },
    'comparison': {
        (Quantity, Quantity): ['data'],
        (Quantity, algebraic.Orderable): ['data'],
        None: ['unit'],
    },
    'additive': {
        (Quantity, Quantity): ['data', 'unit'],
        (Quantity, Real): ['data'],
        (Real, Quantity): ['data'],
    },
    'multiplicative': {
        (Quantity, Quantity): ['data', 'unit'],
        (Quantity, Real): ['data'],
        (Real, Quantity): ['data'],
    },
    'exponential': {
        (Quantity, Real): ['data', 'unit'],
        (Real, Quantity): ['data', 'unit'],
    },
}


# An operator requires three things:
# - a callable object
# - an algorithm for evaluating arguments via the callable object
# - rules that declare allowed types of operands and which attributes to update
#
# The final two define an operator category (e.g., binary comparison operators),
# so we could equivalently say that an operator requires a callable object and a
# category.

comparison = OperatorFactory(Comparison, RULES['comparison'])
cast = OperatorFactory(Cast, RULES['cast'])
unary = OperatorFactory(Unary, RULES['unary'])
additive = OperatorFactory(Numeric, RULES['additive'])
multiplicative = OperatorFactory(Numeric, RULES['multiplicative'])
exponential = OperatorFactory(Numeric, RULES['exponential'])


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

    __lt__ = comparison.operator(standard.lt).implement()
    __le__ = comparison.operator(standard.le).implement()
    __gt__ = comparison.operator(standard.gt).implement()
    __ge__ = comparison.operator(standard.ge).implement()
    __eq__ = comparison.operator(standard.eq).implement()
    __ne__ = comparison.operator(standard.ne).implement()

    __abs__ = unary.operator(standard.abs).implement()
    __neg__ = unary.operator(standard.neg).implement()
    __pos__ = unary.operator(standard.pos).implement()

    addition = additive.operator(standard.add)
    __add__ = addition.implement('forward')
    __radd__ = addition.implement('reverse')
    __iadd__ = addition.implement('inplace')

    subtraction = additive.operator(standard.sub)
    __sub__ = subtraction.implement('forward')
    __rsub__ = subtraction.implement('reverse')
    __isub__ = subtraction.implement('inplace')

    multiplication = multiplicative.operator(standard.mul)
    __mul__ = multiplication.implement('forward')
    __rmul__ = multiplication.implement('reverse')
    __imul__ = multiplication.implement('inplace')

    division = multiplicative.operator(standard.truediv)
    __truediv__ = division.implement('forward')
    __rtruediv__ = division.suppress('reverse')
    __itruediv__ = division.implement('inplace')

    exponentiation = exponential.operator(standard.pow)
    __pow__ = exponentiation.implement('forward')
    __rpow__ = exponentiation.suppress('reverse')
    __ipow__ = exponentiation.implement('inplace')


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

