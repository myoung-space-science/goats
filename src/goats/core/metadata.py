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


class NTypesError(Exception):
    """Inconsistent number of types in a new rule."""


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
        """Determine if all operands have the named attribute(s)."""
        return all(hasattr(obj, name) for obj in self for name in names)

    def getany(self, name: str):
        """Get the a value for the named attribute, if possible."""
        valid = (v for v in self.getall(name, mode='next') if v)
        return next(valid, None)

    def getall(self, name: str, mode: str=None):
        """Get operand values for the named attribute."""
        return [self._get(name, operand, mode) for operand in self]

    def _get(self, name: str, operand, mode: str=None):
        """Helper for `~Operands.getall`."""
        available = hasattr(operand, name)
        if mode == 'next':
            return getattr(operand, name) if available else None
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


class Operator(iterables.ReprStrMixin):
    """A generalized metadata operator."""

    def __init__(self, __callable: typing.Callable) -> None:
        self.method = __callable

    def __call__(self, name: str, operands: Operands, **kwargs):
        """Apply this operator's method to the given operands."""
        values = operands.getall(name, mode='force')
        try:
            return self.method(*values, **kwargs)
        except TypeError as err:
            if len(operands) == 1:
                return values[0]
            if operands.allhave(name):
                raise MetadataError(err) from err
            return operands.getany(name)

    def __str__(self) -> str:
        return str(self.method)


class Operation(typing.Generic[T], iterables.ReprStrMixin):
    """A general operation context."""

    def __init__(
        self,
        __type: typing.Type[T],
        *parameters: str,
        name: str=None,
    ) -> None:
        self._type = __type
        self._parameters = list(parameters)
        self.name = name
        """The name of this operation, if any."""
        self.supported = set()
        """The type(s) of operand(s) for which this operation is valid."""
        self.suppressed = set()
        """The type(s) of operand(s) for which this operation is invalid."""
        self.nargs = None
        """The number of required arguments in this operation."""

    @property
    def parameters(self):
        """The names of metadata attributes."""
        return tuple(self._parameters)

    @property
    def implemented(self):
        """False if this operation supports no operand types."""
        return bool(self.supported)

    def copy(self):
        """Create a deep copy of this operation."""
        return Operation(
            self._type,
            *self.parameters,
            name=self.name,
        )

    def update(self, *parameters: str):
        """Update the names of metadata attributes."""
        if not parameters:
            raise ValueError(
                "Update requires at least one parameter."
            ) from None
        self._parameters.extend(parameters)
        return self

    Types = typing.Union[
        type,
        typing.Tuple[type],
        typing.List[type],
    ]

    def suppress(self, *types: Types, symmetric: bool=False):
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
        self.suppressed.add(types)
        if symmetric:
            self.suppressed.add(types[::-1])
        return self

    def support(self, *types: Types, symmetric: bool=False):
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

        Raises
        ------
        NTypesError
            The number of types in a given type specification is not equal to
            the number of types in existing type specifications.
        """
        if not types:
            raise ValueError("No types to add.") from None
        these = [types] if all(isinstance(t, type) for t in types) else types
        for this in these:
            ntypes = len(this)
            if self.nargs is None:
                self.nargs = ntypes
            if ntypes != self.nargs:
                raise NTypesError(
                    f"Can't add a length-{ntypes} rule to a collection"
                    f" of length-{self.nargs} rules."
                ) from None
            self.supported.add(this)
            if symmetric:
                self.supported.add(this[::-1])
        return self

    def supports(self, *types: Types):
        """Determine if this operation supports `types` or subtypes."""
        if self.nargs and len(types) != self.nargs:
            return False
        if types in self.suppressed:
            return False
        for t in self.suppressed:
            if all(issubclass(i, j) for i, j in zip(types, t)):
                return False
        if types in self.supported:
            return True
        for t in self.supported:
            if all(issubclass(i, j) for i, j in zip(types, t)):
                return True
        return self._type in types

    def evaluate(self, *args, **kwargs):
        """Compute attribute values using the default method."""
        return self.operator(*args, **kwargs)

    @property
    def operator(self):
        """The operator created from this operation's default callable."""
        return self.apply(REFERENCE[self.name])

    def apply(self, method: typing.Callable):
        """Create an operator by applying the given method.
        
        Parameters
        ----------
        method : callable
            The callable object that should operate on metadata attributes.

        Notes
        -----
        * If the method is not callable, this method assumes that there is no
          metadata to compute. This is the case, for example, with type casts
          and binary comparisons.
        """
        compute = Operator(method)
        def operator(*args, **kwargs):
            # NOTE: A more conservative approach would be to raise an error for
            # un-callable methods and force downstream code to define a trivial
            # function that returns `None` for any arguments.
            if not callable(method):
                return None
            operands = Operands(*args)
            types = operands.types
            if not self.supports(*types):
                raise OperandTypeError(
                    f"Can't apply {method.__qualname__!r} to metadata"
                    f" with types {', '.join(t.__qualname__ for t in types)}"
                ) from None
            return {p: compute(p, operands, **kwargs) for p in self.parameters}
        operator.__name__ = self.name or f'__{method.__name__}__'
        operator.__doc__ = method.__doc__
        if callable(method) and not isinstance(method, type):
            operator.__text_signature__ = str(inspect.signature(method))
        return operator

    def __eq__(self, __o):
        """Determine if two operational contexts are equal."""
        return (
            isinstance(__o, Operation)
            and __o._type == self._type
            and __o.supported == self.supported
            and __o.suppressed == self.suppressed
        )


class OperandTypeError(Exception):
    """Operands are incompatible for a given operation."""


class MetadataError(Exception):
    """Error in computing metadata value."""


class OperatorFactory(collections.abc.Mapping):
    """A factory for objects that operate on metadata."""

    def __init__(self, __type: T, *parameters: str) -> None:
        """
        Initialize this instance.

        Parameters
        ----------
        __type : type
            The type of object to which these operators will apply.

        *parameters : string
            Zero or more strings representing the updatable attributes in each
            operand to these operations.
        """
        self._type = __type
        self._parameters = list(iterables.unique(*parameters))
        self._operations = None

    @property
    def parameters(self):
        """The names of updatable metadata attributes."""
        return tuple(self._parameters)

    def register(self, *names: str):
        """Register additional names of metadata attributes."""
        self._parameters.extend(iterables.unique(*names))
        for operation in self.operations.values():
            operation.update(*names)
        return self

    def implement(self, method: typing.Callable):
        """Apply `method` to the default implementation."""
        return Operation(self._type, *self.parameters).apply(method)

    def check(self, *args):
        """Ensure that all arguments have consistent metadata values."""
        for p in self.parameters:
            if not consistent(p, *args):
                raise ValueError(f"Inconsistent metadata for {p!r}")

    def consistent(self, *args):
        """True if all arguments have consistent metadata values."""
        try:
            self.check(*args)
        except ValueError:
            return False
        else:
            return True

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
        """The operations defined here.
        
        Notes
        -----
        Operations are not the same as operators. For example, the special
        operator methods `__truediv__`, `__rtruediv__`, and `__itruediv__`, as
        well as the `numpy` ufunc `true_divide`, all correspond to the 'true
        divide' operation. This aliased mapping of operations does not currently
        support look-up by operator name in order to avoid misleading the user
        into thinking there are different objects for different operators
        corresponding to the same operation.
        """
        if self._operations is None:
            operations = aliased.MutableMapping.fromkeys(REFERENCE)
            for k in REFERENCE:
                operations[k] = Operation(
                    self._type,
                    *self.parameters,
                    name=k,
                )
            self._operations = operations
        return self._operations


_reference: typing.Dict[str, dict] = {
    'int': {
        'callable': None,
    },
    'float': {
        'callable': None,
    },
    'abs': {
        'callable': abs,
        'aliases': ['absolute'],
    },
    'pos': {
        'callable': standard.pos,
        'aliases': ['positive'],
    },
    'neg': {
        'callable': standard.neg,
        'aliases': ['negative'],
    },
    'ceil': {
        'callable': math.ceil,
        'aliases': ['ceiling'],
    },
    'floor': {
        'callable': math.floor,
    },
    'trunc': {
        'callable': math.trunc,
        'aliases': ['truncate'],
    },
    'round': {
        'callable': round,
    },
    'lt': {
        'callable': None,
        'aliases': ['less'],
    },
    'le': {
        'callable': None,
        'aliases': ['less_equal', 'less equal'],
    },
    'gt': {
        'callable': None,
        'aliases': ['greater'],
    },
    'ge': {
        'callable': None,
        'aliases': ['greater_equal', 'greater equal'],
    },
    'add': {
        'callable': standard.add,
    },
    'sub': {
        'callable': standard.sub,
        'aliases': ['subtract'],
    },
    'mul': {
        'callable': standard.mul,
        'aliases': ['multiply'],
    },
    'truediv': {
        'callable': standard.truediv,
        'aliases': ['true_divide', 'true divide'],
    },
    'pow': {
        'callable': pow,
        'aliases': ['power'],
    },
}
REFERENCE = aliased.Mapping(_reference).squeeze(strict=True)


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


