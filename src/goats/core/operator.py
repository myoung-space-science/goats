import abc
import collections
import collections.abc
import operator as standard
import typing

from goats.core import aliased
from goats.core import iterables


Types = typing.TypeVar('Types', type, tuple)
Types = typing.Union[type, typing.Tuple[type, ...]]


Parameters = typing.TypeVar('Parameters', str, typing.Collection)
Parameters = typing.Union[str, typing.Collection[str]]


T = typing.TypeVar('T')


def prune(items: typing.Iterable[T]) -> typing.List[T]:
    """Remove repeated items while preserving order."""
    collection = []
    for item in items:
        if item not in collection:
            collection.append(item)
    return collection


class Rule(iterables.ReprStrMixin):
    """A correspondence between operand types and affected attributes."""

    def __init__(
        self,
        __types: Types,
        *parameters: str,
    ) -> None:
        self._types = list(iterables.whole(__types))
        self._ntypes = len(self._types)
        self.default = prune(parameters)
        """The default parameters."""
        self._parameters = self.default.copy()

    @property
    def updated(self):
        """The parameters affected by this rule."""
        if self._parameters is not None:
            return tuple(self._parameters)

    @property
    def ignored(self):
        """The parameters unaffected by this rule."""
        if self._parameters is None:
            return tuple(self.default)
        these = set(self.default) - set(self._parameters)
        return tuple(prune(these))

    @property
    def types(self):
        """The operand types that define this rule."""
        return tuple(self._types)

    def __len__(self) -> int:
        """The number of operands in this rule."""
        return self._ntypes

    def __contains__(self, t: type) -> bool:
        """True if `t` is one of the types in this rule."""
        return t in self.types

    def __eq__(self, other) -> bool:
        """Called for self == other.
        
        This method return ``True`` iff each type in `other` is strictly equal
        to the corresponding type in `self` under element-wise comparison.
        """
        return self._compare(other, standard.eq)

    def _subtypes(self, other) -> bool:
        """Helper for self > other and self >= other.
        
        This method return ``True`` iff each type in `other` is a subtype (i.e.,
        subclass) of the corresponding type in `self` under element-wise
        comparison.
        """
        return self._compare(other, issubclass)

    def _compare(self, other, method):
        """Compare `other` to `self` via `method`, if possible."""
        equivalent = iterables.allinstance(other, type)
        supported = isinstance(other, Rule) or equivalent
        if not supported:
            return NotImplemented
        types = other._types if isinstance(other, Rule) else other
        return all(method(i, j) for i, j in zip(types, self._types))

    __gt__ = _subtypes
    """Called for self > other."""

    __ge__ = _subtypes
    """Called for self >= other."""

    def replace(self, old: type, new: type):
        """Replace the first occurrence of `old` type with `new` type."""
        try:
            index = self._types.index(old)
        except ValueError as err:
            raise ValueError(
                f"There are no occurrences of {old!r} to replace"
            ) from err
        else:
            self._types[index] = new
            return self

    # Consider letting this return a new immutable instance that raises an
    # informative exception when calling code tries to append, etc.
    @property
    def suppress(self):
        """Suppress this operand rule.
        
        Invoking this property will set the internal set of parameters to
        ``None``, which will signal to operator implementations that they should
        not implement the operator for these operand types.
        """
        self._parameters = None
        return self

    def reset(self):
        """Reset the parameter set to the default set."""
        self._parameters = self.default.copy()
        return self

    def append(self, *parameters: str):
        """Append the given parameter(s) to the current set."""
        self._catch_suppressed()
        new = self._parameters + list(parameters)
        self._parameters = prune(new)
        return self

    def insert(self, index: typing.SupportsIndex, *parameters: str):
        """Insert the given parameter(s) at `index`."""
        self._catch_suppressed()
        new = self._parameters.copy()
        for parameter in parameters:
            new.insert(index, parameter)
            index += 1
        self._parameters = prune(new)
        return self

    def remove(self, *parameters: str):
        """Remove the named parameter(s) from the current set."""
        self._catch_suppressed()
        new = self._parameters.copy()
        for parameter in parameters:
            new.remove(parameter)
        self._parameters = prune(new)
        return self

    def _catch_suppressed(self):
        """Raise an exception if the user tries to update a suppressed rule."""
        if self._parameters is None:
            raise TypeError("Can't update suppressed rule") from None

    def __str__(self) -> str:
        names = [t.__qualname__ for t in self.types]
        types = names[0] if len(names) == 1 else tuple(names)
        parameters = (
            NotImplemented if self.updated is None
            else list(self.updated)
        )
        return f"{types}: {parameters}"


class Operands(collections.abc.Mapping):
    """A class for managing operand update rules."""

    def __init__(self, *parameters: str) -> None:
        self._default = list(parameters)
        self._rulemap = None
        self._ntypes = 0
        self._internal = []

    def register(self, key: Types, *parameters: str):
        """Add an update rule to the collection."""
        types = tuple(key) if isinstance(key, typing.Iterable) else (key,)
        ntypes = len(types)
        if self._rulemap is None:
            self._rulemap = {}
            self._ntypes = ntypes
        if types not in self._rulemap:
            if ntypes == self._ntypes:
                self._rulemap[types] = parameters or self._default.copy()
                return self
            raise ValueError(
                f"Can't add {ntypes} type(s) to collection"
                f" of length-{self._ntypes} items."
            ) from None
        raise KeyError(
            f"{key!r} is already in the collection."
        ) from None

    def __len__(self) -> int:
        """Returns the number of rules. Called for len(self)."""
        return len(self._rulemap)

    def __iter__(self) -> typing.Iterator:
        """Iterate over rules. Called for iter(self)."""
        return iter(self._rulemap)

    def __getitem__(self, key: Types) -> Rule:
        """Retrieve the operand-update rule for `types`."""
        types = tuple(key) if isinstance(key, typing.Iterable) else (key,)
        if types in self._rulemap:
            parameters = self._rulemap[types]
            return Rule(types, *parameters)
        for t, p in self._rulemap.items():
            if all(issubclass(i, j) for i, j in zip(types, t)):
                return Rule(t, *p)
        raise KeyError(
            f"No rule for operand type(s) {key!r}"
        ) from None


class Operator:
    """Base class for operator application schemes."""

    def __init__(
        self,
        __callable: typing.Callable,
        operands: Operands,
    ) -> None:
        self._callable = __callable
        self.operands = operands

    def evaluate(self, *args, **kwargs):
        """Evaluate the arguments via this implementation."""
        types = tuple(type(i) for i in args)
        rule = self.operands[types]


class Implementation(iterables.ReprStrMixin):
    """A generalized arithmetic operator implementation."""

    def __init__(self, operands: Operands) -> None:
        self._build = Operator
        self._rules = {}
        self.operands = operands
        self._operations = []

    def operations(self, *operations: str):
        """Update the operations that this implementation handles."""
        if not operations:
            return tuple(self._operations)
        new = self._operations.copy()
        new.extend(operations)
        self._operations = prune(new)
        return self

    def apply(self, new: typing.Type[Operator]):
        """Set the application class for this operator."""
        self._build = new
        return self

    def implement(self, __callable: typing.Callable):
        """Implement an operator with the given callable."""
        return self._build(__callable, self.operands)

    def __str__(self) -> str:
        name = self._build.__name__
        if self._operations:
            return f"{name}: {self._operations}"
        return name


class Implementations(aliased.MutableMapping):
    """An updatable interface to operator implementations."""

    _internal: typing.Dict[str, Implementation]

    def __init__(self, *parameters: str) -> None:
        super().__init__()
        self.operands = Operands(parameters)
        """The default operands for these implementations."""
        self._internal = {}

    def register(self, key: str):
        """Register a new implementation."""
        if key in self:
            raise KeyError(f"Implementation {key!r} already exists.")
        self[key] = Implementation(self.operands)
        return self

    def __getitem__(self, key: str) -> Implementation:
        """Retrieve an implementation by keyword. Called for self[key]."""
        try:
            return super().__getitem__(key)
        except KeyError:
            for implementation in self.values(aliased=True):
                if key in implementation.operations():
                    return implementation
        raise KeyError(f"No implementation for {key!r}")

    def __str__(self) -> str:
        return ', '.join(
            f"'{g}': {v}"
            for g, v in zip(self._aliased.keys(), self._aliased.values())
        )

