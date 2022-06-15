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

    def add(self, *types: type):
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
            return self._add(types)
        for these in types:
            self._add(these)
        return self

    def _add(self, types):
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


class Operation(iterables.ReprStrMixin):
    """A general operation context."""

    def __init__(
        self,
        *constraints: str,
        types: Types=None,
        method: typing.Callable=None,
    ) -> None:
        self.constraints = constraints
        """Strings indicating operational constraints."""
        self.types = Types() if types is None else types.copy()
        """The operand types allowed in this operation."""
        self.method = method
        """The default callable for computing metadata."""
        self._suppressed = set()

    def copy(self):
        """Create a deep copy of this operation."""
        return Operation(
            *self.constraints,
            types=self.types.copy(),
            method=self.method,
        )

    def suppress(self, *types: type, symmetric: bool=False):
        """Suppress operations on these types.
        
        Parameters
        ----------
        *types
            One or more type specifications to suppress. A type specification
            consists of one or more type(s) of operand(s) that are
            inter-operable. Multiple type specifications must be grouped into
            lists or tuples, even if they represent a single type; this method
            will interpret ungrouped types as a single type specification.

        symmetric : bool, default=False
            If true, suppress the forward and reverse version of each type
            specification. Otherwise, suppress only the given type
            specifications. This option can simplify method calls for binary
            operations; it has no effect on unary operations.
        """
        self._suppressed.add(types)
        if symmetric:
            self._suppressed.add(types[::-1])
        return self

    def support(self, *types: type, symmetric: bool=False):
        """Support operations on these types.
        
        Parameters
        ----------
        *types
            One or more type specifications to support. A type specification
            consists of one or more type(s) of operand(s) that are
            inter-operable. Multiple type specifications must be grouped into
            lists or tuples, even if they represent a single type; this method
            will interpret ungrouped types as a single type specification.

        symmetric : bool, default=False
            If true, support the forward and reverse version of each type
            specification. Otherwise, support only the given type
            specifications. This option can simplify method calls for binary
            operations; it has no effect on unary operations.
        """
        self.types.add(*types)
        if symmetric:
            self.types.add(*types[::-1])
        return self

    def supports(self, *types: type):
        """Determine if this operation supports `types` or subtypes."""
        if self.types.ntypes and len(types) != self.types.ntypes:
            return False
        if types in self._suppressed:
            return False
        for t in self._suppressed:
            if all(issubclass(i, j) for i, j in zip(types, t)):
                return False
        if types in self.types:
            return True
        for t in self.types:
            if all(issubclass(i, j) for i, j in zip(types, t)):
                return True
        return (
            isinstance(self.types.implied, type)
            and self.types.implied in types
        )


class OperandTypeError(Exception):
    """Operands are incompatible for a given operation."""


class MetadataError(Exception):
    """Error in computing metadata value."""


def consistent(name: str, *args):
    """Check consistency of a metadata attribute across `args`."""
    values = [
        utilities.getattrval(arg, name) for arg in args
        if hasattr(arg, name)
    ]
    if not values:
        return True # trivial case
    v0 = values[0]
    return all([v == v0 for v in values])


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
        self._parameters = list(iterables.unique(*parameters))
        self.types = Types(implied=__type)
        self._operations = None

    @property
    def parameters(self):
        """The names of updatable metadata attributes."""
        return tuple(self._parameters)

    def register(self, *names: str):
        """Register additional names of metadata attributes."""
        self._parameters.extend(iterables.unique(*names))
        return self

    def implement(self, name: str, method: typing.Callable=None):
        """Implement the named operator."""
        # Notes:
        # - `parameters` are the same for all operations
        # - `types` are operation-specific
        # - `strict` is operation-specific
        # - if user doesn't pass `method`, we need to check for a default
        if name not in self:
            return self._default(method)
        operation = self[name]
        # I would like to put this in something like `Operation.apply`, which
        # would take `method` and return a callable equivalent to `operator`. We
        # would still be able to update `operator.__name__` and
        # `operator.__doc__` here.
        method = method or operation.method
        def operator(*args, **kwargs):
            # - Check operand consistency. This needs to happen before checking
            #   `method` because some operations (e.g., binary comparisons)
            #   require consistency even though they don't operate on metadata.
            if 'strict' in operation.constraints:
                for p in self.parameters:
                    if not consistent(p, *args):
                        raise TypeError(f"Inconsistent metadata for {p!r}")
            # - If a method really is missing, we'll take that to mean there is
            #   no metadata to compute. This is a common case (e.g., type casts
            #   and binary comparisons).
            if not method:
                return None
            # - At this point, we can proceed to process metadata.
            types = [type(arg) for arg in args]
            if not operation.supports(*types):
                raise OperandTypeError(
                    f"Can't apply {method.__qualname__!r} to metadata"
                    f" with types {', '.join(t.__qualname__ for t in types)}"
                ) from None
            results = {}
            for p in self.parameters:
                values = [utilities.getattrval(arg, p) for arg in args]
                try:
                    results[p] = method(*values, **kwargs)
                except TypeError as err:
                    if all([hasattr(arg, p) for arg in args]):
                        raise MetadataError(err) from err
            return results
        operator.__name__ = name
        operator.__doc__ = method.__doc__
        if callable(method):
            operator.__text_signature__ = str(inspect.signature(method))
        return operator

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
            return self.operations[__k]
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
                definition = Operation(
                    *v.get('constraints', ()),
                    types=self.types,
                    method=v.get('callable'),
                )
                # Use `copy` because definitions need to be independent.
                operations[k] = definition.copy()
                operators = v.get('operators', ())
                if 'dunder' in operators:
                    operations.alias(k, f'__{k}__')
                    if 'numeric' in operators:
                        operations[f'__r{k}__'] = definition.copy()
                        operations[f'__i{k}__'] = definition.copy()
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
        'callable': None,
        'operators': ['dunder'],
    },
    'float': {
        'callable': None,
        'operators': ['dunder'],
    },
    'abs': {
        'callable': abs,
        'aliases': ['absolute'],
        'operators': ['dunder'],
    },
    'pos': {
        'callable': standard.pos,
        'aliases': ['positive'],
        'operators': ['dunder'],
    },
    'neg': {
        'callable': standard.neg,
        'aliases': ['negative'],
        'operators': ['dunder'],
    },
    'ceil': {
        'callable': math.ceil,
        'aliases': ['ceiling'],
        'operators': ['dunder'],
    },
    'floor': {
        'callable': math.floor,
        'operators': ['dunder'],
    },
    'trunc': {
        'callable': math.trunc,
        'aliases': ['truncate'],
        'operators': ['dunder'],
    },
    'round': {
        'callable': round,
        'operators': ['dunder'],
    },
    'lt': {
        'callable': None,
        'aliases': ['less'],
        'constraints': ['strict'],
        'operators': ['dunder'],
    },
    'le': {
        'callable': None,
        'aliases': ['less_equal', 'less equal'],
        'constraints': ['strict'],
        'operators': ['dunder'],
    },
    'gt': {
        'callable': None,
        'aliases': ['greater'],
        'constraints': ['strict'],
        'operators': ['dunder'],
    },
    'ge': {
        'callable': None,
        'aliases': ['greater_equal', 'greater equal'],
        'constraints': ['strict'],
        'operators': ['dunder'],
    },
    'add': {
        'callable': standard.add,
        'constraints': ['strict'],
        'operators': ['dunder', 'numeric'],
    },
    'sub': {
        'callable': standard.sub,
        'aliases': ['subtract'],
        'constraints': ['strict'],
        'operators': ['dunder', 'numeric'],
    },
    'mul': {
        'callable': standard.mul,
        'aliases': ['multiply'],
        'operators': ['dunder', 'numeric'],
    },
    'truediv': {
        'callable': standard.truediv,
        'aliases': ['true_divide', 'true divide'],
        'operators': ['dunder', 'numeric'],
    },
    'pow': {
        'callable': pow,
        'aliases': ['power'],
        'operators': ['dunder', 'numeric'],
    },
    'sqrt': {
        'callable': sqrt,
        'aliases': ['square_root', 'square root'],
    },
}
