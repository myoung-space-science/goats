"""
Objects and functions for operating on metadata.
"""

import collections.abc
import contextlib
import inspect
import math
import operator as standard
import typing

from goats.core import aliased
from goats.core import iterables
from goats.core import utilities


T = typing.TypeVar('T')


class NTypesError(Exception):
    """Inconsistent number of types in a new rule."""


class Types(collections.abc.MutableSet, iterables.ReprStrMixin):
    """A set-like collection of allowed operand types."""

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

    def agree(self, *names: str, strict: bool=False):
        """Compare values of named attribute(s) across operands.
        
        This method determines if all the named attributes have the same value
        when present in an operand. The result is trivially true for a single
        operand
        """
        if len(self) == 1:
            return True
        if strict:
            for name in names:
                values = self.getall(name, mode='strict')
                v0 = values[0]
                if any(value != v0 for value in values):
                    return False
            return True
        targets = self.find(*names)
        for name in names:
            values = [
                utilities.getattrval(target, name)
                for target in targets
            ]
            v0 = values[0]
            if any(value != v0 for value in values):
                return False
        return True

    def find(self, *names: str):
        """Get all operands with the named attributes."""
        return [
            obj for obj in self
            if all(hasattr(obj, name) for name in names)
        ]

    def allhave(self, *names: str):
        """Determine if all operands have the named attributes."""
        return all(hasattr(obj, name) for obj in self for name in names)

    def getall(self, name: str, mode: str=None):
        """Get operand values for the named attribute."""
        if mode:
            [self._get(name, operand, mode) for operand in self]
        return [self._get(name, operand) for operand in self]

    def _get(self, name: str, operand, mode):
        """Helper for `~Operands.getall`."""
        available = hasattr(operand, name)
        if not available and mode == 'strict':
            raise AttributeError(f"{operand!r} has no attribute {name!r}")
        if available or mode == 'force':
            return utilities.getattrval(operand, name)
        return utilities.getattrval(self.reference, name)

    @property
    def types(self):
        """The type of each object."""
        if self._types is None:
            self._types = tuple(type(i) for i in self)
        return self._types

    @typing.overload
    def __getitem__(self, __i: typing.SupportsIndex) -> T: ...

    @typing.overload
    def __getitem__(self, __s: slice) -> 'Operands': ...

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


class Context(iterables.ReprStrMixin):
    """A general operation context."""

    def __init__(
        self,
        *parameters: str,
        types: Types=None,
        **kwargs
    ) -> None:
        self.parameters = parameters
        self.types = Types() if types is None else types.copy()
        self._kwargs = kwargs

    @property
    def strict(self):
        """True if this context requires consistent attribute values."""
        return bool(self._kwargs.get('strict', False))

    def copy(self):
        """Create a deep copy of this instance."""
        return Context(
            *self.parameters,
            types=self.types.copy(),
            **self._kwargs
        )

    def suppress(self, *types: type):
        """Suppress operations on these types within this context."""
        self.types.discard(types)
        return self

    def supports(self, *types: type):
        """Determine if this collection contains `types` or subtypes."""
        if len(types) != len(self.types):
            return False
        if types in self.types:
            return True
        for t in self.types:
            if all(issubclass(i, j) for i, j in zip(types, t)):
                return True
        return (
            isinstance(self.types.implied, type)
            and all(issubclass(t, self.types.implied) for t in types)
        )


# Operation categories (nargs, returns metadata?, strict?):
# - cast (1, F, F)
# - unary (1, T, F)
# - comparison (2, F, T)
# - numeric (2, T, T)
class Operation(iterables.ReprStrMixin):
    """A general operation on metadata attributes."""

    def __init__(
        self,
        context: Context,
        method: typing.Callable=None,
    ) -> None:
        self.context = context.copy()
        self.method = method

    def apply(self, method: typing.Callable):
        """Apply `method` to this operation."""
        self.method = method
        return self

    @property
    def operator(self):
        """A function for computing metadata for this operation."""
        @self.documenting
        def operator(*args, **kwargs):
            if self.context.strict:
                for p in self.context.parameters:
                    if not self.consistent(p, *args):
                        raise TypeError(f"Inconsistent metadata for {p!r}")
            if not self.active:
                return None
            types = [type(arg) for arg in args]
            if not self.context.supports(*types):
                raise OperandTypeError(
                    f"Can't apply {self.method.__qualname__!r} to metadata"
                    f" with types {', '.join(t.__qualname__ for t in types)}"
                ) from None
            results = {}
            for p in self.context.parameters:
                values = [utilities.getattrval(arg, p) for arg in args]
                try:
                    results[p] = self.method(*values, **kwargs)
                except TypeError as err:
                    if all([hasattr(arg, p) for arg in args]):
                        raise MetadataError(err) from err
            return results
        return operator

    def documenting(self, func: typing.Callable):
        """Decorator for attaching documentation to an operator."""
        if self.active:
            # BUG: This will provide a misleading name for non-standard
            # operators (e.g., `sqrt`). Maybe this class should get have a
            # `name` attribute.
            func.__name__ = f'__{self.method.__name__}__'
            func.__doc__ = self.method.__doc__
            func.__text_signature__ = str(inspect.signature(self.method))
        return func

    @property
    def active(self):
        """True if this operation has a callable method."""
        return callable(self.method)

    def consistent(self, name: str, *args):
        """Check consistency of a metadata attribute across `args`."""
        values = [utilities.getattrval(arg, name) for arg in args]
        v0 = values[0]
        return (
            all(hasattr(arg, name) for arg in args)
            and any([v != v0 for v in values])
        )

    def __str__(self) -> str:
        return str(self.method)


class OperandTypeError(Exception):
    """Operands are incompatible for a given operation."""


class MetadataError(Exception):
    """Error in computing metadata value."""


class OperatorFactory(collections.abc.Mapping):
    """A factory for objects that operator on metadata."""

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
        self._parameters = list(iterables.unique(parameters))
        self.types = Types(implied=__type)
        self._operations = None

    @property
    def parameters(self):
        """The names of updatable metadata attributes."""
        return tuple(self._parameters)

    def register(self, *names: str):
        """Register additional names of metadata attributes."""
        self._parameters.extend(iterables.unique(names))
        return self

    # TODO: This needs to return a callable that takes category-appropriate
    # arguments, and returns a `dict` of appropriate metadata or `None`.
    def implement(self, name: str, method: typing.Callable=None):
        """Implement the named operator."""
        if name not in self and method is None:
            raise ValueError(
                f"Can't implement {name!r} with unknown method."
            ) from None
        # => name in self or method is not None
        if name not in self:
            return self._default(method)
        operation = self.operations[name]
        if method:
            operation.apply(method)
        return operation

    def consistent(self, name: str, *args):
        """Check consistency of a metadata attribute across `args`."""
        values = [utilities.getattrval(arg, name) for arg in args]
        v0 = values[0]
        return (
            all(hasattr(arg, name) for arg in args)
            and any([v != v0 for v in values])
        )

    def _default(self, method: typing.Callable):
        """Apply `method` to the default implementation."""
        def operator(*args, **kwargs):
            """Compute metadata attribute values if possible."""
            results = {}
            for p in self.parameters:
                values = [utilities.getattrval(arg, p) for arg in args]
                with contextlib.suppress(TypeError):
                    results[p] = method(*values, **kwargs)
            return results
        operator.__name__ = f'__{method.__name__}__'
        operator.__doc__ = method.__doc__
        return operator

    def __getitem__(self, __k: str):
        """Retrieve the appropriate operation context."""
        if __k in self.operations:
            return self.operations[__k].context
        raise KeyError(f"Unknown context {__k!r}") from None

    def __len__(self) -> int:
        """The number of defined operations."""
        return len(self.operations)

    def __iter__(self):
        """Iterate over operation contexts."""
        return iter(self.operations)

    @property
    def operations(self) -> typing.Mapping[str, Operation]:
        """The operations defined here."""
        if self._operations is None:
            operations = aliased.MutableMapping.fromkeys(OPERATIONS)
            for k, v in OPERATIONS.items():
                context = Context(*self.parameters, types=self.types)
                definition = Operation(context, v.get('callable'))
                operations[k] = definition
                operators = v.get('operators', ())
                if 'dunder' in operators:
                    operations.alias(k, f'__{k}__')
                    if 'numeric' in operators:
                        operations[f'__r{k}__'] = definition
                        operations[f'__i{k}__'] = definition
            self._operations = operations
        return self._operations


def sqrt(a):
    """Square-root implementation for metadata."""
    try:
        return pow(a, 0.5)
    except TypeError:
        return f'sqrt({a})'


OPERATIONS: typing.Dict[str, dict] = {
    'int': {
        # 'callable': int,
        'operators': ['dunder'],
        # 'category': 'cast',
    },
    'float': {
        # 'callable': float,
        'operators': ['dunder'],
        # 'category': 'cast',
    },
    'abs': {
        'callable': abs,
        'aliases': ['absolute'],
        'operators': ['dunder'],
        # 'category': 'unary',
    },
    'pos': {
        'callable': standard.pos,
        'aliases': ['positive'],
        'operators': ['dunder'],
        # 'category': 'unary',
    },
    'neg': {
        'callable': standard.neg,
        'aliases': ['negative'],
        'operators': ['dunder'],
        # 'category': 'unary',
    },
    'ceil': {
        'callable': math.ceil,
        'aliases': ['ceiling'],
        'operators': ['dunder'],
        # 'category': 'unary',
    },
    'floor': {
        'callable': math.floor,
        'operators': ['dunder'],
        # 'category': 'unary',
    },
    'trunc': {
        'callable': math.trunc,
        'aliases': ['truncate'],
        'operators': ['dunder'],
        # 'category': 'unary',
    },
    'round': {
        'callable': round,
        'operators': ['dunder'],
        # 'category': 'unary',
    },
    'lt': {
        # 'callable': standard.lt,
        'aliases': ['less'],
        'modes': ['strict'],
        'operators': ['dunder'],
        # 'category': 'comparison',
    },
    'le': {
        # 'callable': standard.le,
        'aliases': ['less_equal', 'less equal'],
        'modes': ['strict'],
        'operators': ['dunder'],
        # 'category': 'comparison',
    },
    'gt': {
        # 'callable': standard.gt,
        'aliases': ['greater'],
        'modes': ['strict'],
        'operators': ['dunder'],
        # 'category': 'comparison',
    },
    'ge': {
        # 'callable': standard.ge,
        'aliases': ['greater_equal', 'greater equal'],
        'modes': ['strict'],
        'operators': ['dunder'],
        # 'category': 'comparison',
    },
    'add': {
        'callable': standard.add,
        'modes': ['strict'],
        'operators': ['dunder', 'numeric'],
        # 'category': 'numeric',
    },
    'sub': {
        'callable': standard.sub,
        'aliases': ['subtract'],
        'modes': ['strict'],
        'operators': ['dunder', 'numeric'],
        # 'category': 'numeric',
    },
    'mul': {
        'callable': standard.mul,
        'aliases': ['multiply'],
        'operators': ['dunder', 'numeric'],
        # 'category': 'numeric',
    },
    'truediv': {
        'callable': standard.truediv,
        'aliases': ['true_divide', 'true divide'],
        'operators': ['dunder', 'numeric'],
        # 'category': 'numeric',
    },
    'pow': {
        'callable': pow,
        'aliases': ['power'],
        'operators': ['dunder', 'numeric'],
        # 'category': 'numeric',
    },
    'sqrt': {
        'callable': sqrt,
        'aliases': ['square_root', 'square root'],
    },
}
