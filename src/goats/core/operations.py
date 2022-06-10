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

    def allhave(self, *names: str):
        """Determine if all operands have the named attributes."""
        return all(hasattr(obj, name) for obj in self for name in names)

    def get(self, name: str, force: bool=False):
        """Get operand values for the named attribute."""
        return [self._get(name, operand, force=force) for operand in self]

    def _get(self, name: str, operand, force):
        """Helper for `~Operands.get`."""
        if hasattr(operand, name) or force:
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
            if not operands.allhave(*self.names):
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


class OperationError(Exception):
    """An error occurred during an operation."""


class OperandTypeError(Exception):
    """Operands are incompatible for a given operation."""


Caller = typing.TypeVar('Caller', bound=typing.Callable)
Caller = typing.Callable[..., T]


A = typing.TypeVar('A')
B = typing.TypeVar('B')


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

    # What does each operation actually need?
    # - cast
    #   - operate on the data attribute of a single operand
    #   - directly return the result
    # - lt, le, gt, ge
    #   - check metadata attribute consistency if necessary
    #   - operate on the data attribute of mulitple operands
    #   - directly return the result
    # - eq, ne
    #   - operate on data and metadata attributes of multiple operands
    #   - return the union of boolean results
    # - unary
    #   - operate on the data and metadata attributes of a single operand
    #   - initialize a new instance of the operand with the results
    #   - return the instance
    # - numeric
    #   - operate on the data attribute of multiple operands
    #   - attempt to operate on the metadata attributes of the same operands
    #   - fall back on a reference value if necessary
    #   - initialize a new instance or update an existing instance
    #   - return the instance
    #   - default values are necessary when the co-operand(s) in a binary
    #     numeric operation define some but not all of the metadata attributes
    #   - needs to raise `OperandTypeError` if a metadata operator raises
    #     `TypeError` because we don't know à priori which metadata attributes
    #     will implement a given operation

    def compute(self, *args, reference=None, target=None, **kwargs):
        """Evaluate arguments within this operational context."""
        if not self.parameters or all(isbuiltin(arg) for arg in args):
            # Either we don't know which arguments to operate on or this
            # implementation is unnecessary.
            # - solution: hand execution over to the given operands, in case
            #   they implement this operator in their class definitions. 
            # - this will lead to recursion if they define the operator via this
            #   class
            # - see old implementation for try/catch recursion block
            # - consider reducing recursion limit via `sys.setrecursionlimit`
            #   inside this block
            try:
                return self.method(*args, **kwargs)
            except RecursionError as err:
                raise OperationError(
                    f"Caught {err!r} when attempting to implement"
                    f" {self.method!r}. This may be because one of the"
                    f" operands uses {self!r} to implement {self.method!r}"
                    " without explicit knowledge of the updatable attributes."
                ) from err
        types = [type(arg) for arg in args]
        if not self.types.supports(*types):
            raise OperandTypeError(self._operand_errmsg(types))
        operands = Operands(*args, reference=reference)
        primary, *secondary = self.parameters
        data = self.method(*operands.get(primary, force=True), **kwargs)
        if reference is None and target is None:
            if operands.agree(*secondary):
                return data
            errmsg = self._operand_errmsg(types, *secondary)
            raise OperandTypeError(errmsg)
        values = [data] + [
            self._compute(name, operands, **kwargs)
            for name in secondary
        ]
        if isinstance(target, type):
            return target(*values)
        zipped = zip(self.parameters, values)
        for name, value in zipped:
            utilities.setattrval(target, name, value)
        return target

    def _compute(self, name: str, operands: Operands, **kwargs):
        """Internal helper for `~Operation.compute`."""
        try:
            if operands.allhave(name):
                return self.method(*operands.get(name), **kwargs)
            with contextlib.suppress(TypeError):
                return self.method(*operands.get(name, force=True), **kwargs)
        except TypeError as err:
            if len(operands) > 1:
                errmsg = self._operand_errmsg(operands.types, name)
                raise OperandTypeError(errmsg) from err
        return utilities.getattrval(operands.reference, name)

    def _operand_errmsg(self, types: typing.Iterable[type], *fixed: str):
        """Build an error message based on `rule` and `operands`."""
        method_string = repr(self.method.__qualname__)
        types_string = (
            types[0].__qualname__ if len(types) == 1
            else f"({', '.join(t.__qualname__ for t in types)})"
        )
        message = f"Can't apply operator {method_string} to {types_string}"
        if fixed:
            attrs_string = (
                repr(fixed[0]) if len(fixed) == 1
                else f"{fixed[0]!r} and {fixed[1]!r}" if len(fixed) == 2
                else f"{', '.join(fixed[:-1])} and {fixed[-1]}"
            )
            message += f" with different values of {attrs_string}"
        return message


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
        # operator.__name__ = f'__{__callable.__name__}__'
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
        # operator.__name__ = f'__{__callable.__name__}__'
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
        # operator.__name__ = f'__{__callable.__name__}__'
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
        # operator.__name__ = f'__{__callable.__name__}__'
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
        # operator.__name__ = f'__r{__callable.__name__}__'
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
        # operator.__name__ = f'__i{__callable.__name__}__'
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
        # forward.__name__ = f'__{__callable.__name__}__'
        forward.__doc__ = __callable.__doc__
        def reverse(b: B, a: A, /, **kwargs) -> B:
            """Apply this operation to `a` and `b` with reflected operands."""
            try:
                result = operation.compute(a, b, reference=b, target=type(b), **kwargs)
            except metric.UnitError as err:
                raise OperandTypeError(err) from err
            else:
                return result
        # reverse.__name__ = f'__r{__callable.__name__}__'
        reverse.__doc__ = __callable.__doc__
        def inplace(a: A, b: B, /, **kwargs) -> A:
            """Apply this operation to `a` and `b` in-place."""
            try:
                result = operation.compute(a, b, reference=a, target=a, **kwargs)
            except metric.UnitError as err:
                raise OperandTypeError(err) from err
            else:
                return result
        # inplace.__name__ = f'__i{__callable.__name__}__'
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
        'aliases': ['less'],
    },
    'le': {
        'category': 'comparison',
        'callable': standard.le,
        'aliases': ['less_equal'],
    },
    'gt': {
        'category': 'comparison',
        'callable': standard.gt,
        'aliases': ['greater'],
    },
    'ge': {
        'category': 'comparison',
        'callable': standard.ge,
        'aliases': ['greater_equal'],
    },
    # We may want to exclude 'eq' and 'ne' because
    # 1) unlike the other comparison operators, they can be valid when metadata
    #    attributes are unequal
    # 2) `__eq__` is implemented for `object` (via `__hash__`?) whereas others
    #    are not, so `==` will work on most objects by default.
    #
    # 'eq': {
    #     'category': 'comparison',
    #     'callable': standard.eq,
    # },
    # 'ne': {
    #     'category': 'comparison',
    #     'callable': standard.ne,
    # },
    'add': {
        'category': 'numeric',
        'callable': standard.add,
    },
    'sub': {
        'category': 'numeric',
        'callable': standard.sub,
        'aliases': ['subtract'],
    },
    'mul': {
        'category': 'numeric',
        'callable': standard.mul,
        'aliases': ['multiply'],
    },
    'truediv': {
        'category': 'numeric',
        'callable': standard.truediv,
        'aliases': ['true_divide'],
    },
    'pow': {
        'category': 'numeric',
        'callable': pow,
    },
}
"""A mapping of operation name to metadata."""


class REFERENCE(typing.NamedTuple):
    """Reference objects for operations and operators."""

    _numeric = [k for k, v in OPERATIONS.items() if v['category'] == 'numeric']

    OPERATORS = aliased.MutableMapping(OPERATIONS)
    for k, v in OPERATIONS.items():
        OPERATORS.alias(k, f'__{k}__')
    for k in _numeric:
        OPERATORS[k].update({'mode': 'forward'})
        OPERATORS[f'__r{k}__'] = {**OPERATORS[k], 'mode': 'reverse'}
        OPERATORS[f'__i{k}__'] = {**OPERATORS[k], 'mode': 'inplace'}

    NAMES = {
        c: [f'__{k}__' for k, v in OPERATIONS.items() if v['category'] == c]
        for c in CATEGORIES if c != 'numeric'
    }
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
        for arg in args:
            if type(arg) == type(first) and arg != first:
                return NotImplemented
        return first
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


def restrict(__f: typing.Callable, *types: type, reverse: bool=False):
    """Restrict allowed operand types for an operation."""
    s = 'r' if reverse else ''
    name = f"__{s}{__f.__name__}__"
    def operator(a, b, **kwargs):
        method = getattr(super(type(a), a), name)
        if not isinstance(b, (type(a), *types)):
            return NotImplemented
        return method(b, **kwargs)
    operator.__name__ = name
    operator.__doc__ = __f.__doc__
    return operator


class Interface(collections.abc.Mapping):
    """Top-level interface to arithmetic operations."""

    def __init__(self, __type: T, *parameters: str) -> None:
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
        self._categories = None
        self._operations = None

    def implement(self, __k: str, method: typing.Callable=None) -> Context:
        """Implement the named operator."""
        context = self[__k] if __k in self else self.categories['default']
        operator = REFERENCE.OPERATORS[__k]
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
        if __k in self.categories:
            return self.categories[__k]
        # This block ensures that we don't overwrite a context.
        if __k in self.operations:
            if current := self.operations[__k]:
                return current
            new = self.categories[REFERENCE.OPERATORS[__k]['category']].spawn()
            self.operations[__k] = new
            return new
        raise KeyError(f"Unknown context {__k!r}") from None

    def __len__(self) -> int:
        """The number of defined operations."""
        return len(self.operations)

    def __iter__(self):
        """Iterate over operation contexts."""
        return iter(self.operations)

    @property
    def categories(self) -> typing.Dict[str, Context]:
        """The defined operation-category contexts."""
        if self._categories is None:
            self._categories = {
                k: v(*self.parameters, types=self.types)
                for k, v in CATEGORIES.items()
            }
        return self._categories

    @property
    def operations(self) -> typing.Dict[str, Context]:
        """The standard operation contexts defined here."""
        if self._operations is None:
            self._operations = aliased.MutableMapping.fromkeys(
                REFERENCE.OPERATORS
            )
        return self._operations

    def subclass(
        self,
        __name: str,
        *bases: type,
        include: typing.Iterable[str]=None,
        exclude: typing.Iterable[str]=None,
    ) -> T:
        """Generate a subclass with mixin operators from the current state.
        
        Parameters
        ----------
        __name : str
            The name of the new subclass.

        *bases : types or iterable of types
            Zero or more base classes from which the new subclass will inherit.
            This method will append `bases` to the default type passed during
            initialization, so that the default type will appear first in the
            subclass's MRO.

        include : iterable of strings, optional
            Names of operators or operation categories to exlicitly implement in
            the new subclass.

        exclude : iterable of strings, optional
            Names of operators or operation categories to exlicitly not
            implement in the new subclass.
        """
        included = set(REFERENCE.NAMES['all']) if include is None else set()
        for name in include or []:
            if name in REFERENCE.NAMES:
                included.update(set(REFERENCE.NAMES[name]))
            else:
                included.update({name})
        for name in exclude or []:
            if name in REFERENCE.NAMES:
                included.difference_update(set(REFERENCE.NAMES[name]))
            else:
                included.difference_update({name})
        operators = {k: self.implement(k) for k in included}
        parents = (self._type,) + iterables.unwrap(bases, wrap=tuple)
        return type(__name, parents, operators)


class Operators(abc.ABC):
    """"""

    @abc.abstractmethod
    def implement(self, __k: str, method: typing.Callable=None) -> Context:
        """Implement the named operator."""
        pass



def operators(interface: Interface, **kwargs):
    """"""
    include = kwargs.get('include')
    exclude = kwargs.get('exclude')
    included = set(REFERENCE.NAMES['all']) if include is None else set()
    for name in include or []:
        if name in REFERENCE.NAMES:
            included.update(set(REFERENCE.NAMES[name]))
        else:
            included.update({name})
    for name in exclude or []:
        if name in REFERENCE.NAMES:
            included.difference_update(set(REFERENCE.NAMES[name]))
        else:
            included.difference_update({name})
    operators = {k: interface.implement(k) for k in included}
    operators['implement'] = interface.implement
    return type('OperatorMixin', (Operators,), operators)
