"""
Objects and functions for operating on metadata.
"""

import collections.abc
import math
import operator as standard
import typing

from goats.core import aliased
from goats.core import iterables


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


class Context:
    """An operation context."""

    def __init__(self, *parameters: str, types: Types=None) -> None:
        self.parameters = parameters
        self.types = Types() if types is None else types.copy()

    def suppress(self, *types: type):
        """Suppress operations on these types within this context."""
        self.types.discard(types)
        return self


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
        self._contexts = None

    @property
    def parameters(self):
        """The names of updatable metadata attributes."""
        return tuple(self._parameters)

    def register(self, *names: str):
        """Register additional names of metadata attributes."""
        self._parameters.extend(iterables.unique(names))
        return self

    def __getitem__(self, __k: str) -> Context:
        """Retrieve the appropriate operation context."""
        if __k in self.contexts:
            if current := self.contexts[__k]:
                return current
            new = Context(*self.parameters, types=self.types)
            self.contexts[__k] = new
            return new
        raise KeyError(f"Unknown context {__k!r}") from None

    def __len__(self) -> int:
        """The number of defined operations."""
        return len(self.contexts)

    def __iter__(self):
        """Iterate over operation contexts."""
        return iter(self.contexts)

    @property
    def contexts(self) -> typing.Dict[str, Context]:
        """The operation contexts defined here."""
        if self._contexts is None:
            contexts = aliased.MutableMapping.fromkeys(OPERATIONS)
            for k, v in OPERATIONS.items():
                category = v.get('category')
                if category:
                    contexts.alias(k, f'__{k}__')
                    if category == 'numeric':
                        contexts[f'__r{k}__'] = v.copy()
                        contexts[f'__i{k}__'] = v.copy()
            self._contexts = contexts
        return self._contexts


def sqrt(a):
    """Square-root implementation for metadata."""
    try:
        return pow(a, 0.5)
    except TypeError:
        return f'sqrt({a})'


OPERATIONS = {
    'int': {
        'callable': int,
        'category': 'cast',
    },
    'float': {
        'callable': float,
        'category': 'cast',
    },
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
        'aliases': ['true_divide', 'true divide'],
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
