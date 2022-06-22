import abc
import collections.abc
import inspect
import math
import numbers
import operator as standard
import typing

import numpy
import numpy.typing

from goats.core import algebraic
from goats.core import iterables
from goats.core import metadata
from goats.core import metric
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

    __metadata__: typing.ClassVar = '_metric'

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

    def __eq__(self, other) -> bool:
        """Determine if two quantifiable objects are equal."""
        if isinstance(other, Quantifiable):
            return (
                other._amount == self._amount
                and other._metric == self._metric
            )
        return other == self._amount

    def __ne__(self, other):
        """Determine if two quantifiable are not equal."""
        return not self == other


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

    __metadata__: typing.ClassVar = 'unit'

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

    @property
    def unit(self):
        """This quantity's metric unit."""
        return self._metric

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
        unit = self._init_unit(pos, **kwargs)
        return data, metric.Unit(unit)

    def _init_unit(self, pos: list, **kwargs):
        """Parse the unit attribute from arguments or use the default value."""
        if given := kwargs.get('unit'):
            return given
        try:
            return pos.pop(0)
        except IndexError:
            return '1'

    def convert(self, unit: metric.UnitLike):
        """Set the unit of this object's values."""
        if unit == self._metric:
            return self
        new = metric.Unit(unit)
        self._amount *= new // self._metric
        self._metric = new
        return self


T = typing.TypeVar('T')
A = typing.TypeVar('A')
B = typing.TypeVar('B')


Q = typing.TypeVar('Q', bound=Quantity)


def unary(method: typing.Callable):
    """Implement a unary operator from `method`."""
    def operator(q: Quantity, **kwargs):
        return method(q, **kwargs)
    operator.__name__ = f'__{method.__name__}__'
    operator.__doc__ = method.__doc__
    operator.__text_signature__ = str(inspect.signature(method))
    return operator


def binary(method: typing.Callable):
    """Implement a binary operator from `method`."""
    def operator(q: Quantity, b: B, **kwargs):
        return (
            method(q.data, b.data, **kwargs) if isinstance(b, Quantity)
            else method(q.data, b, **kwargs)
        )
    operator.__name__ = f'__{method.__name__}__'
    operator.__doc__ = method.__doc__
    operator.__text_signature__ = str(inspect.signature(method))
    return operator


class Operators:
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

    _callables = {
        'int': int,
        'float': float,
        'abs': abs,
        'pos': standard.pos,
        'neg': standard.neg,
        'ceil': math.ceil,
        'floor': math.floor,
        'trunc': math.trunc,
        'round': round,
        'lt': standard.lt,
        'le': standard.le,
        'gt': standard.gt,
        'ge': standard.ge,
        'add': standard.add,
        'sub': standard.sub,
        'mul': standard.mul,
        'truediv': standard.truediv,
        'pow': pow,
    }
    _metadata = {
        **_callables,
        **{k: None for k in ('int', 'float', 'lt', 'le', 'gt', 'ge')},
    }

    def __init_subclass__(cls, **kwargs) -> None:
        """Define operators on a subclass of `~measurable.Quantity`."""
        super().__init_subclass__(**kwargs)
        factory = metadata.OperatorFactory(cls, callables=cls._metadata)
        factory.check('lt', 'le', 'gt', 'ge', 'add', 'sub')
        factory['true divide'].suppress(Real, Quantity)
        factory['power'].suppress(Quantity, Quantity)
        factory['power'].suppress(Real, Quantity)
        factory['power'].suppress(Quantity, typing.Iterable, symmetric=True)
        parameters = cls._collect('__metadata__')
        factory.register(*iterables.unique(*parameters))
        cls.metadata = factory
        operators = cls._collect('__operators__')
        for operator in operators:
            if not isinstance(operator, str):
                raise TypeError(f"Invalid operator specification {operator!r}")
            name = operator
            operation, mode = cls._parse_input(name)
            datafunc = cls._callables[operation]
            metafunc = factory.implement(operation)
            setattr(cls, name, cls.implement(datafunc, metafunc, mode=mode))

    @classmethod
    def _collect(cls, __name: str):
        """Internal helper for collecting special class attributes.
        
        This method iteratively collects the values of a named class attribute,
        starting from a base class and progressing through successive subclasses
        so that the order of the resultant list reflects the order of
        inheritance.
        """
        ancestors = cls.mro()[::-1]
        return [
            name for c in ancestors
            for name in iterables.whole(getattr(c, __name, ()))
        ]

    @classmethod
    def _parse_input(cls, user: str):
        """Extract the operation name and mode from user input."""
        if user in cls._callables:
            return user, 'forward'
        if (stripped := user.strip('__')) in cls._callables:
            return stripped, 'forward'
        if stripped[1:] in cls._callables:
            token, name = stripped[0], stripped[1:]
            if token == 'r':
                return name, 'reverse'
            if token == 'i':
                return name, 'inplace'
            return name, 'forward'
        raise ValueError(f"Unknown operator {user!r}")

    @typing.overload
    @classmethod
    def implement(
        cls,
        datafunc: typing.Callable[[Quantity], T],
        metafunc: typing.Callable[[Quantity], None],
    ) -> T: ...

    @typing.overload
    @classmethod
    def implement(
        cls,
        datafunc: typing.Callable[[Quantity], T],
        metafunc: typing.Callable[[Quantity], dict],
    ) -> Quantity: ...

    @typing.overload
    @classmethod
    def implement(
        cls,
        datafunc: typing.Callable[[typing.Any, typing.Any], T],
        metafunc: typing.Callable[[Quantity, typing.Any], None],
        *,
        mode: str,
    ) -> T: ...

    @typing.overload
    @classmethod
    def implement(
        cls,
        datafunc: typing.Callable[[typing.Any, typing.Any], T],
        metafunc: typing.Callable[[Quantity, typing.Any], dict],
        *,
        mode: typing.Union[
            typing.Literal['forward'],
            typing.Literal['inplace'],
        ],
    ) -> Quantity: ...

    @typing.overload
    @classmethod
    def implement(
        cls,
        datafunc: typing.Callable[[typing.Any, typing.Any], T],
        metafunc: typing.Callable[[typing.Any, Quantity], dict],
        *,
        mode: typing.Literal['reverse'],
    ) -> Quantity: ...

    @classmethod
    def implement(cls, datafunc, metafunc, mode: str='forward'):
        """Create an operator from the named operation."""
        def operator(*args, **kwargs):
            operands = args[::-1] if mode == 'reverse' else args
            values = [
                i.data if isinstance(i, Quantity) else i
                for i in operands
            ]
            datavals = datafunc(*values, **kwargs)
            metavals = metafunc(*operands, **kwargs)
            if metavals is None:
                return datavals
            target = next(arg for arg in args if isinstance(arg, Quantity))
            if mode == 'inplace':
                utilities.setattrval(target, 'data', datavals)
                for k, v in metavals:
                    utilities.setattrval(target, k, v)
                return target
            return type(target)(datavals, **metavals)
        operator.__name__ = f'__{datafunc.__name__}__'
        operator.__doc__ = datafunc.__doc__
        if callable(datafunc) and not isinstance(datafunc, type):
            operator.__text_signature__ = str(inspect.signature(datafunc))
        return operator


# Dev idea (not in use): pass methods for computing data and metadata to these functions instead of `Operators.implement`.
def cast(data: typing.Type[T]):
    """Implement a unary cast operator."""
    def operator(q: Q) -> T:
        return data(q.data)
    return operator

def arithmetic(data: typing.Callable[[A], T], meta):
    """Implement a unary arithmetic operator."""
    def operator(q: Q, **kwargs) -> Q:
        return type(q)(data(q.data, **kwargs), **meta(q, **kwargs))
    return operator

def comparison(data: typing.Callable[[A, B], bool], check):
    """Implement a binary comparison operator."""
    def operator(q: Q, b: B) -> bool:
        operands = [
            i.data if isinstance(i, Quantity) else i
            for i in (q, b)
        ]
        if check(q, b):
            return data(*operands)
    return operator

def numeric(data: typing.Callable[[A, B], T], meta):
    """Implement a binary numeric operator."""
    def operator(q: Q, b: B, **kwargs) -> Q:
        operands = [
            i.data if isinstance(i, Quantity) else i
            for i in (q, b)
        ]
        return type(q)(data(*operands, **kwargs), **meta(q, b, **kwargs))
    return operator


class Subtype(typing.Generic[T]):
    """Generic type to indicate a subclass in type hints."""


Q = typing.TypeVar('Q', bound=Quantity)


# def operators(
#     __target: typing.Type[Q],
#     *bases: typing.Type[T],
#     include: typing.Iterable[str]=None,
#     exclude: typing.Iterable[str]=None,
# ) -> Subtype[Q]:
#     """Create a mixin class of measurable operators."""

#     # Notes:
#     # - Consider defining a method on `metadata.OperatorFactory` that just
#     #   checks attribute constency on the given operand(s). It wouldn't need to
#     #   know the operation name or method.
#     # - Consider defining a method on `metadata.Operation` that is equivalent to
#     #   the definition of `operator` within
#     #   `metadata.OperatorFactory.implement`. That would allow this function to
#     #   use `q.metadata[<name>].apply(<arguments>)`.
#     # - The functions below can be split into unary/binary for computing data
#     #   and check/compute for handling metadata:
#     #  - cast = unary + check
#     #  - arithmetic = unary + compute
#     #  - comparison = binary + check
#     #  - numeric = binary + compute

#     def unary(method, q: Q, **kwargs):
#         """Apply a unary operator to arguments."""
#         return method(q, **kwargs)

#     def binary(method, q: Q, b: B, **kwargs):
#         """Apply a binary operator to arguments."""
#         return (
#             method(q.data, b.data, **kwargs) if isinstance(b, Quantity)
#             else method(q.data, b, **kwargs)
#         )

#     def cast(name: str, method: typing.Type[T]):
#         """Implement a unary cast operator."""
#         def operator(q: Q) -> T:
#             data = method(q.data)
#             q.metadata.implement(name)(q) # consistency check
#             return data
#         return operator

#     def arithmetic(name: str, method: typing.Callable[[A], T]):
#         """Implement a unary arithmetic operator."""
#         def operator(q: Q, **kwargs) -> Q:
#             data = method(q.data, **kwargs)
#             meta = q.metadata.implement(name)(q, **kwargs)
#             return type(q)(data, **meta)
#         return operator

#     def comparison(name: str, method: typing.Callable[[A, B], bool]):
#         """Implement a binary comparison operator."""
#         def operator(q: Q, b: B) -> bool:
#             data = (
#                 method(q.data, b.data) if isinstance(b, Quantity)
#                 else method(q.data, b)
#             )
#             q.metadata.implement(name)(q, b) # consistency check
#             return data
#         return operator

#     def numeric(name: str, method: typing.Callable[[A, B], T]):
#         """Implement a binary numeric operator."""
#         def operator(q: Q, b: B, **kwargs) -> Q:
#             data = (
#                 method(q.data, b.data, **kwargs) if isinstance(b, Quantity)
#                 else method(q.data, b, **kwargs)
#             )
#             meta = q.metadata.implement(name)(q, b, **kwargs)
#             return type(q)(data, **meta)
#         return operator

#     definitions = {
#         '__int__': cast('int', int),
#         '__float__': cast('float', float),
#         '__abs__': arithmetic('abs', abs),
#     }


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

