import abc
import collections
import collections.abc
import contextlib
import functools
import inspect
import math
import operator as standard
import typing

from goats.core import aliased
from goats.core import iterables
from goats.core import metric
from goats.core import utilities


# An operation is the result of applying an operator to operands.

# Categories:
# - cast
#  - (a: T) -> r: Any
#  - no reference
# - unary
#  - (a: T, **kwargs) -> r: T
#  - reference is a
# - comparison
#  - (a: T, b: T | Any) -> r: Any
#  - no reference
# - forward
#  - (a: T, b: T | Any, **kwargs) -> r: T
#  - reference is a
# - reverse
#  - (a: T | Any, b: T, **kwargs) -> r: T
#  - reference is b
# - inplace
#  - (a: T, b: T | Any, **kwargs) -> a: T
#  - reference is a


Parameters = typing.TypeVar('Parameters', str, typing.Collection)
Parameters = typing.Union[str, typing.Collection[str]]


T = typing.TypeVar('T')


def unique(*items: T) -> typing.List[T]:
    """Remove repeated items while preserving order."""
    collection = []
    for item in items:
        if item not in collection:
            collection.append(item)
    return collection


class ComparisonError(TypeError):
    """Incomparable instances of the same type."""

    def __init__(self, __this: typing.Any, __that: typing.Any, name: str):
        self.this = getattr(__this, name, None)
        self.that = getattr(__that, name, None)

    def __str__(self) -> str:
        return f"Can't compare {self.this!r} to {self.that!r}"


def arg_or_kwarg(args: list, kwargs: dict, name: str):
    """Get the value of a positional or keyword argument, if possible.
    
    This checks for the presence of `name` in `kwargs` rather than calling
    `kwargs.get(name)` in order to avoid prematurely returning `None` before
    trying to retrieve a value from `args`.
    """
    if name in kwargs:
        return kwargs[name]
    with contextlib.suppress(IndexError):
        return args.pop(0)


def isbuiltin(__object):
    """Convenience method for testing if an object is a built-in type."""
    return type(__object).__module__ == 'builtins'


def get_parameters(__object):
    """Determine the initialization parameters for an object, if possible."""
    return (
        {} if isbuiltin(__object)
        else inspect.signature(type(__object)).parameters
    )


class Operands(collections.abc.Sequence, iterables.ReprStrMixin):
    """A sequence of operands."""

    def __init__(self, *args: T, reference: T=None) -> None:
        self._args = list(args)
        self._reference = reference
        self._types = None

    @property
    def reference(self):
        """The reference object.
        
        Notes
        -----
        - The default reference object is the first argument used to initialize
          this instance.
        - The reference object is not part of the sequence of operands and
          therefore will not explicitly appear when iterating over an instance.
        """
        if self._reference is None:
            self._reference = self[0]
        return self._reference

    def agree(self, *names: str):
        """Compare values of named attribute(s) across operands.
        
        This method determines if all the named attributes have the same value
        when present in an operand. The result is trivially ``True`` for a
        single operand
        """
        if len(self) == 1:
            return True
        if not all(hasattr(self.reference, name) for name in names):
            return False
        others = self.find(*names)
        value = utilities.getattrval
        return all(
            value(target, name) == value(self.reference, name)
            for name in names for target in others
        )

    def find(self, *names: str):
        """Get all operands with the named attributes."""
        return [
            obj for obj in self
            if all(hasattr(obj, name) for name in names)
        ]

    def consistent(self, *names: str):
        """Determine if all operands have the named attributes."""
        return all(hasattr(obj, name) for obj in self for name in names)

    def get(self, name: str):
        """Get operand values for the named attribute."""
        return [self._get(name, operand) for operand in self]

    def _get(self, name: str, operand):
        """Helper for `~Operands.get`."""
        if hasattr(operand, name):
            return utilities.getattrval(operand, name)
        return utilities.getattrval(self.reference, name)

    @property
    def types(self):
        """The type of each object."""
        if self._types is None:
            self._types = tuple(type(i) for i in self)
        return self._types

    def __getitem__(self, __i):
        """Access operands by index."""
        if isinstance(__i, typing.SupportsIndex):
            return self._args[__i]
        return Operands(*self._args[__i])

    def __len__(self) -> int:
        """The number of operands. Called for len(self)."""
        return len(self._args)

    def __iter__(self):
        """Iterate over operands. Called for iter(self)."""
        yield from self._args

    def __eq__(self, other):
        """Called for self == other."""
        if not len(self) == len(other):
            return False
        return all(i in other for i in self)

    def __str__(self) -> str:
        return ', '.join(str(i) for i in self)


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
    ) -> None:
        self.names = names

    def __call__(self, func: typing.Callable) -> typing.Callable:
        """Ensure argument consistency before calling `func`."""
        if not self.names:
            return func
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if len(args) == 1:
                return func(*args, **kwargs)
            operands = Operands(*args)
            if not operands.consistent(*self.names):
                return NotImplemented
            if operands.agree(*self.names):
                return func(*args, **kwargs)
            raise OperandTypeError(*args)
        return wrapper


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
        """Add these types to the collection, if possible."""
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
        if types in self:
            return True
        for t in self:
            if all(issubclass(i, j) for i, j in zip(types, t)):
                return True
        return (
            self.implied
            and (
                self.implied in types
                or any(issubclass(t, self.implied) for t in types)
            )
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


class OperationError(Exception):
    """An error occurred during an operation."""


class OperandTypeError(Exception):
    """Operands are incompatible for a given operation."""


Caller = typing.TypeVar('Caller', bound=typing.Callable)
Caller = typing.Callable[..., T]


A = typing.TypeVar('A')
B = typing.TypeVar('B')


_operators = [
    ('add', '__add__', 'addition'),
    ('sub', '__sub__', 'subtraction'),
]


@typing.runtime_checkable
class Metadata(typing.Protocol):
    """Protocol definition for metadata attributes in operations."""

    @abc.abstractmethod
    def __abs__(self):
        pass

    @abc.abstractmethod
    def __pos__(self):
        pass

    @abc.abstractmethod
    def __neg__(self):
        pass

    @abc.abstractmethod
    def __round__(self):
        pass

    @abc.abstractmethod
    def __ceil__(self):
        pass

    @abc.abstractmethod
    def __floor__(self):
        pass

    @abc.abstractmethod
    def __trunc__(self):
        pass

    @abc.abstractmethod
    def __add__(self, other):
        pass

    @abc.abstractmethod
    def __radd__(self, other):
        pass

    @abc.abstractmethod
    def __iadd__(self, other):
        pass

    @abc.abstractmethod
    def __sub__(self, other):
        pass

    @abc.abstractmethod
    def __rsub__(self, other):
        pass

    @abc.abstractmethod
    def __isub__(self, other):
        pass

    @abc.abstractmethod
    def __mul__(self, other):
        pass

    @abc.abstractmethod
    def __rmul__(self, other):
        pass

    @abc.abstractmethod
    def __imul__(self, other):
        pass

    @abc.abstractmethod
    def __truediv__(self, other):
        pass

    @abc.abstractmethod
    def __rtruediv__(self, other):
        pass

    @abc.abstractmethod
    def __itruediv__(self, other):
        pass

    @abc.abstractmethod
    def __pow__(self, other):
        pass

    @abc.abstractmethod
    def __rpow__(self, other):
        pass

    @abc.abstractmethod
    def __ipow__(self, other):
        pass


# Idea: Instead of specifying active parameters for certain rules, require the
# operand objects to declare whether or not they implement the operation. This
# could preclude the need to pass names of updatable attributes to `Rules`, and
# possibly the need to even have a Rule object. The interface would still need
# to know parameters in order to avoid the `inspect.signature`
# variable-arguments bug. To implement this change, `Operation.compute` could
# try all active parameters (which it gets from the interface) and skip any for
# which either operand does not implement the given operation.


class Operation:
    """A general arithmetic operation."""

    def __init__(self, method, types: Types, *parameters: str) -> None:
        self.method = method
        self.types = types.copy()
        self.parameters = parameters

    # New implementation:
    # - if allowed types is None, use `rule is None` block
    # - convert argument(s) to operand(s)
    # - for each active attribute
    #  - try to apply operator to attribute on operand(s)
    #  - if that succeeds, keep the result
    #  - if it fails (NotImplemented? TypeError?), do nothing
    # - if no results, return `NotImplemented`
    # - otherwise, proceed from `target is None`
    def compute(self, *args, reference=None, target=None, **kwargs):
        """Evaluate arguments within this operational context."""
        if not self.parameters:
            # We don't know which arguments to operate on, so we hand execution
            # over to the given operands, in case they implement this operator
            # in their class definitions. Note that this will lead to recursion
            # if they define the operator via this class.
            try:
                return self.method(*args, **kwargs)
            except RecursionError as err:
                raise OperationError(
                    f"Caught {err!r} when attempting to implement"
                    f" {self.method!r}. This may be because one of the"
                    f" operands uses {self!r} to implement {self.method!r}"
                    " without explicit knowledge of the updatable attributes."
                ) from err
        if not self.types.supports(*(type(arg) for arg in args)):
            return NotImplemented
        operands = Operands(*args, reference=reference)
        computed = {
            parameter: self._compute(operands, parameter, **kwargs)
            for parameter in self.parameters
        }
        valid = [k for k, v in computed.items() if v != NotImplemented]
        if not valid:
            return NotImplemented
        fixed = [k for k, v in computed.items() if v == NotImplemented]
        if not operands.agree(*fixed):
            errmsg = self._operand_errmsg(operands, *fixed)
            raise OperandTypeError(errmsg) from None
        defaults = self._get_defaults(reference)
        values = [ # order matters
            computed[k] if k in valid else defaults[k]
            for k in self.parameters
        ] if defaults else [computed[k] for k in valid]
        if target is None:
            return (
                values[0] if len(values) == 1 or reference is None
                else values
            )
        if isinstance(target, type):
            return target(*values)
        zipped = zip(defaults, values)
        for name, value in zipped:
            utilities.setattrval(target, name, value)
        return target

    def _operand_errmsg(self, operands: Operands, *fixed: str):
        """Build an error message based on `rule` and `operands`."""
        method_string = repr(self.method.__qualname__)
        types = operands.types
        types_string = (
            types[0].__qualname__ if len(types) == 1
            else f"({', '.join(t.__qualname__ for t in types)})"
        )
        attrs_string = (
            repr(fixed[0]) if len(fixed) == 1
            else f"{fixed[0]!r} and {fixed[1]!r}" if len(fixed) == 2
            else f"{', '.join(fixed[:-1])} and {fixed[-1]}"
        )
        return (
            f"Can't apply operator {method_string} to {types_string}"
            f" with different values of {attrs_string}"
        )

    def _get_defaults(self, reference=None) -> typing.Dict[str, typing.Any]:
        """Get default values for initialization arguments."""
        if reference is None:
            return {}
        parameters = get_parameters(reference)
        if not parameters:
            # There's nothing to operate on.
            return {}
        kinds = {parameter.kind for parameter in parameters.values()}
        # Should this check for `VAR_POSITIONAL in kinds` instead?
        # variable-length keyword parameters aren't fatal -- they just mean
        # we'll initialize the new object with default values.
        if kinds == {
            # Inspection found only variable-length parameters, so we can't
            # determine the names of attributes to get; we need to fall back on
            # the user-provided attribute names.
            inspect.Parameter.VAR_KEYWORD,
            inspect.Parameter.VAR_POSITIONAL,
        }: return {
            name: utilities.getattrval(reference, name)
            for name in self.parameters
        }
        return {
            name: utilities.getattrval(reference, name)
            for name in parameters
        }

    def _compute(self, operands: Operands, name: str, **kwargs):
        """Compute a value if possible."""
        try:
            value = self.method(*operands.get(name), **kwargs)
        except (ValueError, TypeError):
            value = NotImplemented
        return value


class Context(abc.ABC, iterables.ReprStrMixin):
    """Abstract base class for operation-related contexts."""

    def __init__(self, *parameters: str, types: Types=None) -> None:
        self.parameters = parameters
        self.types = Types() if types is None else types.copy()

    def spawn(self):
        """Create a new instance of this context from the current state."""
        return type(self)(*self.parameters, types=self.types)

    @abc.abstractmethod
    def apply(self, __callable: typing.Callable):
        """Use the given callable object within this context."""
        pass

    def __str__(self) -> str:
        return f"{self.parameters}, {self.types}"


class Default(Context):
    """A factory for generalized operators."""

    def apply(self, __callable: typing.Callable[..., T]):
        """Implement this operation with the given callable object."""
        operation = Operation(__callable, self.types, *self.parameters)
        def operator(*args, **kwargs) -> T:
            return operation.compute(*args, **kwargs)
        return operator


class Cast(Context):
    """A factory for type-casting operators."""

    def apply(self, __callable: typing.Type[T]):
        operation = Operation(__callable, self.types, *self.parameters)
        def operator(a: A) -> T:
            return operation.compute(a)
        operator.__name__ = f'__{__callable.__name__}__'
        operator.__doc__ = __callable.__doc__
        return operator


class Unary(Context):
    """A factory for unary arithmetic operators."""

    CType = typing.TypeVar('CType', bound=typing.Callable)
    CType = typing.Callable[[A], A]

    def apply(self, __callable: CType):
        operation = Operation(__callable, self.types, *self.parameters)
        def operator(a: A, /, **kwargs) -> A:
            return operation.compute(a, reference=a, target=type(a), **kwargs)
        operator.__name__ = f'__{__callable.__name__}__'
        operator.__doc__ = __callable.__doc__
        return operator


class Comparison(Context):
    """A factory for binary comparison operators."""

    CType = typing.TypeVar('CType', bound=typing.Callable)
    CType = typing.Callable[[A, B], T]

    def apply(self, __callable: CType):
        operation = Operation(__callable, self.types, *self.parameters)
        def operator(a: A, b: B, /) -> T:
            return operation.compute(a, b)
        operator.__name__ = f'__{__callable.__name__}__'
        operator.__doc__ = __callable.__doc__
        return operator


class Forward(Context):
    """A factory for standard binary numeric operators."""

    CType = typing.TypeVar('CType', bound=typing.Callable)
    CType = typing.Callable[[A, B], T]

    def apply(self, __callable: CType):
        operation = Operation(__callable, self.types, *self.parameters)
        def operator(a: A, b: B, /, **kwargs) -> A:
            try:
                result = operation.compute(a, b, reference=a, target=type(a), **kwargs)
            except metric.UnitError as err:
                raise OperandTypeError(err) from err
            else:
                return result
        operator.__name__ = f'__{__callable.__name__}__'
        operator.__doc__ = __callable.__doc__
        return operator


class Reverse(Context):
    """A factory for binary numeric operators with reflected operands."""

    CType = typing.TypeVar('CType', bound=typing.Callable)
    CType = typing.Callable[[A, B], T]

    def apply(self, __callable: CType):
        operation = Operation(__callable, self.types, *self.parameters)
        def operator(b: B, a: A, /, **kwargs) -> B:
            try:
                result = operation.compute(a, b, reference=b, target=type(b), **kwargs)
            except metric.UnitError as err:
                raise OperandTypeError(err) from err
            else:
                return result
        operator.__name__ = f'__r{__callable.__name__}__'
        operator.__doc__ = __callable.__doc__
        return operator


class Inplace(Context):
    """A factory for binary numeric operators with in-place updates."""

    CType = typing.TypeVar('CType', bound=typing.Callable)
    CType = typing.Callable[[A, B], T]

    def apply(self, __callable: CType):
        operation = Operation(__callable, self.types, *self.parameters)
        def operator(a: A, b: B, /, **kwargs) -> A:
            try:
                result = operation.compute(a, b, reference=a, target=a, **kwargs)
            except metric.UnitError as err:
                raise OperandTypeError(err) from err
            else:
                return result
        operator.__name__ = f'__i{__callable.__name__}__'
        operator.__doc__ = __callable.__doc__
        return operator


class Numeric(Context):
    """A factory for binary numeric operators."""

    CType = typing.TypeVar('CType', bound=typing.Callable)
    CType = typing.Callable[[A, B], T]

    @typing.overload
    def apply(self, __callable: CType) -> typing.Callable[[A, B], A]: ...

    @typing.overload
    def apply(
        self,
        __callable: CType,
        mode: typing.Literal['forward'],
    ) -> typing.Callable[[A, B], A]: ...

    @typing.overload
    def apply(
        self,
        __callable: CType,
        mode: typing.Literal['reverse'],
    ) -> typing.Callable[[B, A], B]: ...

    @typing.overload
    def apply(
        self,
        __callable: CType,
        mode: typing.Literal['inplace'],
    ) -> typing.Callable[[A, B], A]: ...

    def apply(self, __callable, mode: str='forward'):
        operation = Operation(__callable, self.types, *self.parameters)
        def forward(a: A, b: B, /, **kwargs) -> A:
            """Apply this operation to `a` and `b`."""
            try:
                result = operation.compute(a, b, reference=a, target=type(a), **kwargs)
            except metric.UnitError as err:
                raise OperandTypeError(err) from err
            else:
                return result
        forward.__name__ = f'__{__callable.__name__}__'
        forward.__doc__ = __callable.__doc__
        def reverse(b: B, a: A, /, **kwargs) -> B:
            """Apply this operation to `a` and `b` with reflected operands."""
            try:
                result = operation.compute(a, b, reference=b, target=type(b), **kwargs)
            except metric.UnitError as err:
                raise OperandTypeError(err) from err
            else:
                return result
        reverse.__name__ = f'__r{__callable.__name__}__'
        reverse.__doc__ = __callable.__doc__
        def inplace(a: A, b: B, /, **kwargs) -> A:
            """Apply this operation to `a` and `b` in-place."""
            try:
                result = operation.compute(a, b, reference=a, target=a, **kwargs)
            except metric.UnitError as err:
                raise OperandTypeError(err) from err
            else:
                return result
        inplace.__name__ = f'__i{__callable.__name__}__'
        inplace.__doc__ = __callable.__doc__
        if mode == 'forward':
            return forward
        if mode == 'reverse':
            return reverse
        if mode == 'inplace':
            return inplace
        raise ValueError(f"Unknown implementation mode {mode!r}") from None


CATEGORIES: typing.Dict[str, typing.Type[Context]] = {
    'default': Default,
    'cast': Cast,
    'unary': Unary,
    'comparison': Comparison,
    'numeric': Numeric,
}
"""The canonical operation-category implementations."""


OPERATIONS = {
    'int': {
        'category': 'cast',
        'callable': int,
    },
    'float': {
        'category': 'cast',
        'callable': float,
    },
    'abs': {
        'category': 'unary',
        'callable': abs,
    },
    'neg': {
        'category': 'unary',
        'callable': standard.neg,
    },
    'pos': {
        'category': 'unary',
        'callable': standard.pos,
    },
    'ceil': {
        'category': 'unary',
        'callable': math.ceil,
    },
    'floor': {
        'category': 'unary',
        'callable': math.floor,
    },
    'trunc': {
        'category': 'unary',
        'callable': math.trunc,
    },
    'round': {
        'category': 'unary',
        'callable': round,
    },
    'lt': {
        'category': 'comparison',
        'callable': standard.lt,
    },
    'le': {
        'category': 'comparison',
        'callable': standard.le,
    },
    'gt': {
        'category': 'comparison',
        'callable': standard.gt,
    },
    'ge': {
        'category': 'comparison',
        'callable': standard.ge,
    },
    'eq': {
        'category': 'comparison',
        'callable': standard.eq,
    },
    'ne': {
        'category': 'comparison',
        'callable': standard.ne,
    },
    'add': {
        'category': 'numeric',
        'callable': standard.add,
    },
    'sub': {
        'category': 'numeric',
        'callable': standard.sub,
    },
    'mul': {
        'category': 'numeric',
        'callable': standard.mul,
    },
    'truediv': {
        'category': 'numeric',
        'callable': standard.truediv,
    },
    'pow': {
        'category': 'numeric',
        'callable': pow,
    },
}
"""A mapping of operation name to metadata."""


OPERATORS = {
    f'__{k}__': v.copy() for k, v in OPERATIONS.items()
    if v['category'] != 'numeric'
}
OPERATORS.update(
    {
        f'__{k}__': {**v, 'mode': 'forward'}
        for k, v in OPERATIONS.items() if v['category'] == 'numeric'
    }
)
OPERATORS.update(
    {
        f'__r{k}__': {**v, 'mode': 'reverse'}
        for k, v in OPERATIONS.items() if v['category'] == 'numeric'
    }
)
OPERATORS.update(
    {
        f'__i{k}__': {**v, 'mode': 'inplace'}
        for k, v in OPERATIONS.items() if v['category'] == 'numeric'
    }
)


NAMES = {
    c: [f'__{k}__' for k, v in OPERATIONS.items() if v['category'] == c]
    for c in CATEGORIES if c != 'numeric'
}
_numeric = [k for k, v in OPERATIONS.items() if v['category'] == 'numeric']
NAMES.update(
    {
        'forward': [f'__{i}__' for i in _numeric],
        'reverse': [f'__r{i}__' for i in _numeric],
        'inplace': [f'__i{i}__' for i in _numeric],
    }
)
copied = NAMES.copy()
NAMES['all'] = [
    name for category in copied.values() for name in category
]


def identity(__operator: typing.Callable):
    """Create an operator that immediately returns its argument."""
    def operator(*args: T):
        first = args[0]
        if (
            all(type(arg) == type(first) for arg in args)
            and all(arg == first for arg in args)
        ): return first
        return NotImplemented
    operator.__name__ = f'__{__operator.__name__}__'
    operator.__doc__ = __operator.__doc__
    return operator


def suppress(__operator: typing.Callable):
    """Unconditionally suppress an operation."""
    def operator(*args, **kwargs):
        return NotImplemented
    operator.__name__ = f'__{__operator.__name__}__'
    operator.__doc__ = __operator.__doc__
    return operator


class Interface(collections.abc.Mapping):
    """Top-level interface to arithmetic operations."""

    def __init__(self, __type: type, *parameters: str) -> None:
        """
        Initialize this instance.

        Parameters
        ----------
        *parameters : string
            Zero or more strings representing the updatable attributes in each
            operand to these operations.
        """
        self._type = __type
        self.parameters = parameters
        """The names of all updatable attributes"""
        self.types = Types(implied=__type)
        self.categories = {
            k: v(*parameters, types=self.types)
            for k, v in CATEGORIES.items()
        }
        self._operations = None
        self._contexts = None

    def implement(self, __k: str, method: typing.Callable=None) -> Context:
        """Implement the named operator."""
        context = self[__k] if __k in self else self.categories['default']
        operator = OPERATORS[__k]
        method = method or operator['callable']
        if isinstance(context, Numeric):
            return context.apply(method, mode=operator.get('mode'))
        return context.apply(method)

    @typing.overload
    def __getitem__(self, __k: typing.Literal['cast']) -> Cast: ...

    @typing.overload
    def __getitem__(self, __k: typing.Literal['unary']) -> Unary: ...

    @typing.overload
    def __getitem__(self, __k: typing.Literal['comparison']) -> Comparison: ...

    @typing.overload
    def __getitem__(self, __k: typing.Literal['numeric']) -> Numeric: ...

    @typing.overload
    def __getitem__(self, __k: str) -> Default: ...

    def __getitem__(self, __k):
        """Retrieve the appropriate operation context."""
        keys = (__k, f'__{__k}__')
        for key in keys:
            if key in self.categories:
                return self.categories[key]
            # This block ensures that we don't overwrite a context.
            if key in self.operations:
                if current := self.operations[key]:
                    return current
                new = self.categories[OPERATORS[key]['category']].spawn()
                self.operations[key] = new
                return new
        raise KeyError(f"Unknown context {__k!r}") from None

    def __len__(self) -> int:
        """The number of defined operations."""
        return len(self.operations)

    def __iter__(self):
        """Iterate over operation contexts."""
        return iter(self.operations)

    @property
    def operations(self) -> typing.Dict[str, Context]:
        """The standard operation contexts defined here."""
        if self._operations is None:
            self._operations = {k: None for k in OPERATORS}
        return self._operations

    def subclass(
        self,
        __name: str,
        *bases: type,
        include: typing.Iterable[str]=None,
        exclude: typing.Iterable[str]=None,
    ) -> typing.Type:
        """Generate a subclass with mixin operators from the current state.
        
        Parameters
        ----------
        __name : str
            The name of the new subclass.

        *bases : types or iterable of types
            Zero or more base classes from which the new subclass will inherit.
            This method will append `bases` to the default type if one was
            passed during initialization, so that the default type will appear
            with in the subclass's MRO.

        include : iterable of strings, optional
            Names of operators or operation categories to exlicitly implement in
            the new subclass.

        exclude : iterable of strings, optional
            Names of operators or operation categories to exlicitly not
            implement in the new subclass.
        """
        included = set(OPERATORS) if include is None else set()
        for name in include or []:
            if name in NAMES:
                included.update(set(NAMES[name]))
            else:
                included.update({name})
        for name in exclude or []:
            if name in NAMES:
                included.difference_update(set(NAMES[name]))
            else:
                included.difference_update({name})
        operators = {k: self.implement(k) for k in included}
        parents = (self._type,) + iterables.unwrap(bases, wrap=tuple)
        return type(__name, parents, operators)

