import abc
import collections.abc
import contextlib
import math
import numbers
import operator as standard
import typing

import numpy
import numpy.typing

from goats.core import aliased
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


def sqrt(a):
    """Square-root implementation for metadata."""
    try:
        return pow(a, 0.5)
    except TypeError:
        return f'sqrt({a})'


OPERATIONS = {
    'abs': {
        'callable': abs,
        'aliases': ['absolute'],
        'category': 'unary',
    },
    'pos': {
        'callable': standard.pos,
        'aliases': ['positive'],
        'category': 'unary',
    },
    'neg': {
        'callable': standard.neg,
        'aliases': ['negative'],
        'category': 'unary',
    },
    'ceil': {
        'callable': math.ceil,
        'aliases': ['ceiling'],
        'category': 'unary',
    },
    'floor': {
        'callable': math.floor,
        'category': 'unary',
    },
    'trunc': {
        'callable': math.trunc,
        'aliases': ['truncate'],
        'category': 'unary',
    },
    'round': {
        'callable': round,
        'category': 'unary',
    },
    'lt': {
        'callable': standard.lt,
        'aliases': ['less'],
        'strict': True,
        'category': 'comparison',
    },
    'le': {
        'callable': standard.le,
        'aliases': ['less_equal', 'less equal'],
        'strict': True,
        'category': 'comparison',
    },
    'gt': {
        'callable': standard.gt,
        'aliases': ['greater'],
        'strict': True,
        'category': 'comparison',
    },
    'ge': {
        'callable': standard.ge,
        'aliases': ['greater_equal', 'greater equal'],
        'strict': True,
        'category': 'comparison',
    },
    'add': {
        'callable': standard.add,
        'strict': True,
        'category': 'binary',
    },
    'sub': {
        'callable': standard.sub,
        'aliases': ['subtract'],
        'strict': True,
        'category': 'binary',
    },
    'mul': {
        'callable': standard.mul,
        'aliases': ['multiply'],
        'category': 'binary',
    },
    'truediv': {
        'callable': standard.truediv,
        'aliases': ['true_divide'],
        'category': 'binary',
    },
    'pow': {
        'callable': pow,
        'aliases': ['power'],
        'category': 'binary',
    },
    'sqrt': {
        'callable': sqrt,
        'aliases': ['square_root', 'square root'],
    },
}


class MetadataError(Exception):
    """Error in computing metadata value."""


class NTypesError(Exception):
    """Inconsistent number of types in a new rule."""


class Types(collections.abc.MutableSet, iterables.ReprStrMixin):
    """A collection of allowed operand types."""

    def __init__(
        self,
        *init: typing.Union[type, typing.Sequence[type]],
        implied: type=None,
    ) -> None:
        """Initialize this instance.
        
        Parameters
        ----------
        *init : type or sequence of types
            Zero or more types or sequences of types with which to initialize
            this instance's collection.

        implied : type, optional
            A single type to treat as an implied type when checking for the
            existence of other types in this collection. This type will not show
            up in the explicit collection of types. See Notes for clarification
            of containment (i.e., queries of the form ``x in types``).

        Notes
        -----
        Checks for existence of the implied type will depend on whether this
        collection is empty or has explicit type groups: If there are no
        explicit type groups, checking for the implied type or a homogeneous
        tuple of the implied type of any length will evaluate as ``True``. If
        there are explicit type groups, the number of types must agree. For
        example::

        >>> types = operations.Types(implied=str)
        >>> str in types
        True
        >>> (str, str) in types
        True
        >>> (str, str, str) in types
        True
        >>> types.add(int, float)
        >>> str in types
        False
        >>> (str, str) in types
        True
        >>> (str, str, str) in types
        False
        """
        self.ntypes = None
        if init:
            ntypes = len(init[0])
            if all(len(i) == ntypes for i in init):
                self.ntypes = ntypes
            else:
                raise NTypesError(
                    f"Can't initialize {self.__class__.__qualname__}"
                    " with variable-length types."
                )
        self.implied = implied
        self._types = set(init)

    def add(self, *types: type, symmetric: bool=False):
        """Add these types to the collection, if possible.
        
        Parameters
        ----------
        *types
            One or more type specifications to register. A type specification
            consists of one or more type(s) of operand(s) that are
            inter-operable. Multiple type specifications must be grouped into
            lists or tuples, even if they represent a single type.

        Raises
        ------
        NTypesError
            The number of types in a given type specification is not equal to
            the number of types in existing type specifications.
        """
        if not types:
            raise ValueError("No types to add.") from None
        if isinstance(types[0], type):
            return self._add(types, symmetric=symmetric)
        for these in types:
            self._add(these)
        return self

    def _add(self, types, symmetric=False):
        """Internal helper for `~Types.add`."""
        ntypes = len(types)
        if self.ntypes is None:
            self.ntypes = ntypes
        if ntypes != self.ntypes:
            raise NTypesError(
                f"Can't add a length-{ntypes} rule to a collection"
                f" of length-{self.ntypes} rules."
            ) from None
        self._types |= set([types])
        if symmetric:
            self._types |= set([types[::-1]])
        return self

    def discard(self, *types: type):
        """Remove these types from the collection."""
        self._types.discard(types)
        if all(t == self.implied for t in types):
            self.implied = None

    def clear(self) -> None:
        """Remove all types from the collection."""
        self._types = set()

    def copy(self):
        """Copy types into a new instance."""
        return type(self)(*self._types, implied=self.implied)

    def supports(self, *types: type):
        """Determine if this collection contains `types` or subtypes."""
        if self.ntypes and len(types) != self.ntypes:
            return False
        if types in self:
            return True
        for t in self:
            if all(issubclass(i, j) for i, j in zip(types, t)):
                return True
        return (
            isinstance(self.implied, type)
            and all(issubclass(t, self.implied) for t in types)
        )

    def __contains__(self, __x) -> bool:
        """Called for x in self."""
        x = (__x,) if isinstance(__x, type) else __x
        if self.implied is None:
            return x in self._types
        truth = all(t == self.implied for t in x) or x in self._types
        if self.ntypes is None:
            return truth
        return len(x) == self.ntypes and truth

    def __iter__(self):
        """Called for iter(self)."""
        return iter(self._types)

    def __len__(self):
        """Called for len(self)."""
        return len(self._types)

    def __str__(self) -> str:
        return ', '.join(str(t) for t in self._types)


class OperandTypeError(Exception):
    """Operands are incompatible for a given operation."""


T = typing.TypeVar('T')
A = typing.TypeVar('A')
B = typing.TypeVar('B')


class Metadata:
    """Manages metadata attributes for measurable quantities."""

    operations = aliased.MutableMapping(OPERATIONS)

    def __init__(self, *names: str, types: Types=None) -> None:
        self._names = list(names)
        self.types = Types() if types is None else types.copy()

    @property
    def names(self):
        """The current collection of updatable metadata attributes."""
        return tuple(self._names)

    def register(self, *names: str):
        """Register additional names of metadata attributes."""
        self._names.extend(names)
        return self

    def implement(self, __key: str):
        """Implement an operation"""
        operation = self.operations[__key]
        method = operation['callable']
        strict = operation.get('strict', False)
        get = utilities.getattrval
        def cast(arg):
            """Type-cast operations require no metadata."""
            return None
        def unary(arg, **kwargs):
            """Compute values of metadata attribute from `arg`."""
            results = {}
            for name in self._names:
                value = get(arg, name)
                try:
                    results[name] = method(value, **kwargs)
                except TypeError:
                    results[name] = value
            return results
        def comparison(*args):
            """Check metadata consistency of `args`."""
            if not strict:
                return
            for name in self._names:
                if not self.consistent(name, *args):
                    raise TypeError(
                        f"Inconsistent metadata for {name!r}"
                    ) from None
        def binary(*args, **kwargs):
            """Compute values of metadata attribute from `args`."""
            types = [type(arg) for arg in args]
            if not self.types.supports(*types):
                raise OperandTypeError(
                    f"Can't apply {method.__qualname__!r} to metadata"
                    f" with types {', '.join(t.__qualname__ for t in types)}"
                ) from None
            results = {}
            for name in self._names:
                available = [hasattr(arg, name) for arg in args]
                if strict and not self.consistent(name, *args):
                    raise TypeError(
                        f"Inconsistent metadata for {name!r}"
                    ) from None
                values = [get(arg, name) for arg in args]
                try:
                    results[name] = method(*values, **kwargs)
                except TypeError as err:
                    if all(available):
                        # This is an error because the method failed despite the
                        # fact that all arguments have the attribute. That
                        # implies that one of the attributes said no thank you.
                        raise MetadataError(err) from err
                    # At least one of the arguments has the attribute, so let's
                    # find it and use its value.
                    #
                    # results[name] = next(values[i] for i, t in available if t)
            return results
        def default(*args, **kwargs):
            """Compute metadata attribute values if possible."""
            results = {}
            for name in self._names:
                values = [get(arg, name) for arg in args]
                with contextlib.suppress(TypeError):
                    results[name] = method(*values, **kwargs)
            return results
        category = operation.get('category')
        # TODO: Refactor this case block into something more exstensible.
        if category == 'cast':
            return cast
        if category == 'unary':
            return unary
        if category == 'comparison':
            return comparison
        if category == 'binary':
            return binary
        return default

    def consistent(self, name: str, *args):
        """Check consistency of a metadata attribute across `args`."""
        values = [utilities.getattrval(arg, name) for arg in args]
        v0 = values[0]
        return (
            all(hasattr(arg, name) for arg in args)
            and any([v != v0 for v in values])
        )


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
        self.metadata = Metadata('unit')
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
        unit = metric.Unit(kwargs.get('unit') or pos.pop(0) or '1')
        return data, unit

    def convert(self, unit: metric.UnitLike):
        """Set the unit of this object's values."""
        if unit == self._metric:
            return self
        new = metric.Unit(unit)
        self._amount *= new // self._metric
        self._metric = new
        return self

    def __eq__(self, other) -> bool:
        """Determine if two quantities are equal."""
        if isinstance(other, Quantity):
            return other.data == self.data and other.unit == self.unit
        return other == self.data


class OperatorMixin:
    """Mixin operators for measurable quantities."""

    def cast(__callable: typing.Type[T]):
        """"""
        def operator(a: A) -> T:
            if isinstance(a, Quantity):
                return __callable(a.data)
            return __callable(a)
        return operator

    def unary(__callable: typing.Callable[[A], T]):
        """"""
        def operator(a: A, **kwargs) -> A:
            if isinstance(a, Quantity):
                return __callable(a.data, **kwargs)
            return __callable(a, **kwargs)
        return operator

    def comparison(__callable: typing.Callable[[A, B], T]):
        """"""
        def operator(a: A, b: B) -> typing.Union[bool, T]:
            if isinstance(a, Quantity):
                if isinstance(b, Quantity):
                    return __callable(a.data, b.data)
                return __callable(a.data, b)
            return __callable(a, b)
        return operator

    def forward(__callable: typing.Callable[[A, B], T]):
        """"""
        def operator(a: A, b: B, **kwargs) -> A:
            if isinstance(a, Quantity):
                if isinstance(b, Quantity):
                    return __callable(a.data, b.data, **kwargs)
                return __callable(a.data, b, **kwargs)
            return __callable(a, b, **kwargs)
        return operator

    def reverse(__callable: typing.Callable[[A, B], T]):
        """"""
        def operator(a: A, b: B, **kwargs) -> B:
            return
        return operator

    def inplace(__callable: typing.Callable[[A, B], T]):
        """"""
        def operator(a: A, b: B, **kwargs) -> A:
            return
        return operator


def mixin(*parameters: str, name: str='QuantityMixin', bases=(), **kwargs):
    """Create a class that defines mixin operators for quantities."""
    interface = operations.Interface(Quantity, *parameters)
    rules = {
        'numeric': [
            (Quantity, Quantity),
            (Quantity, Real),
            (Real, Quantity),
        ]
    }
    interface['numeric'].types.add(*rules['numeric'])
    interface['__rtruediv__'].types.discard(Real, Quantity)
    interface['__rpow__'].types.discard(Real, Quantity)
    interface['__pow__'].types.discard(Quantity, Quantity)
    return interface.subclass(name, *bases, **kwargs)


interface = operations.Interface(Quantity, 'data', 'unit')
rules = {
    'numeric': [
        (Quantity, Quantity),
        (Quantity, Real),
        (Real, Quantity),
    ]
}
interface['numeric'].types.add(*rules['numeric'])
interface['__rtruediv__'].types.discard(Real, Quantity)
interface['__rpow__'].types.discard(Real, Quantity)
interface['__pow__'].types.discard(Quantity, Quantity)
QuantityMixin = interface.subclass(
    'QuantityMixin',
    exclude=['cast', '__round__', '__ceil__', '__floor__', '__trunc__']
)
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


def operators(*parameters: str, **kwargs):
    """"""
    names = ['data', 'unit', *parameters]
    interface = operations.Interface(Quantity, *names)
    rules = {
        'comparison': [
            (Quantity, Quantity),
            (Quantity, Real),
            (Real, Quantity),
        ],
        'numeric': [
            (Quantity, Quantity),
            (Quantity, Real),
            (Real, Quantity),
        ],
    }
    for key, types in rules.items():
        interface[key].types.add(*types)
    interface['__rtruediv__'].types.discard(Real, Quantity)
    interface['__rpow__'].types.discard(Real, Quantity)
    interface['__pow__'].types.discard(Quantity, Quantity)
    return operations.operators(interface, **kwargs)


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

