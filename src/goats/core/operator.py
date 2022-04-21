import abc
import collections
import collections.abc
import operator as standard
import typing

from goats.core import iterables


Types = typing.TypeVar('Types', type, tuple)
Types = typing.Union[type, typing.Tuple[type, ...]]


Parameters = typing.TypeVar('Parameters', str, typing.Collection)
Parameters = typing.Union[str, typing.Collection[str]]


class Implementation:
    """Base class for operator implementation schemes."""

    def __init__(self) -> None:
        pass


class Rule(iterables.ReprStrMixin):
    """A correspondence between operand types and affected attributes."""

    def __init__(
        self,
        __types: Types,
        *parameters: str,
    ) -> None:
        self._types = list(iterables.whole(__types))
        self._ntypes = len(self._types)
        self.default = self._prune(parameters)
        """The default parameters."""
        self._parameters = self.default.copy()

    @property
    def parameters(self):
        """The parameters affected by this rule."""
        if self._parameters is not None:
            return tuple(self._parameters)

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
        new = self._parameters + list(parameters)
        self._parameters = self._prune(new)
        return self

    def insert(self, index: typing.SupportsIndex, *parameters: str):
        """Insert the given parameter(s) at `index`."""
        new = self._parameters.copy()
        for parameter in parameters:
            new.insert(index, parameter)
            index += 1
        self._parameters = self._prune(new)
        return self

    def remove(self, *parameters: str):
        """Remove the named parameter(s) from the current set."""
        new = self._parameters.copy()
        for parameter in parameters:
            new.remove(parameter)
        self._parameters = self._prune(new)
        return self

    def _prune(self, parameters: typing.Iterable[str]) -> typing.List[str]:
        """Remove repeated parameters while preserving order."""
        p = []
        for parameter in parameters:
            if parameter not in p:
                p.append(parameter)
        return p

    def __str__(self) -> str:
        names = [t.__qualname__ for t in self.types]
        types = names[0] if len(names) == 1 else tuple(names)
        parameters = (
            NotImplemented if self.parameters is None
            else list(self.parameters)
        )
        return f"{types}: {parameters}"


class Operator:
    """A generalized arithmetic operator."""

    operands: typing.List[type]

    def __init__(self, *parameters: str) -> None:
        self._default = list(parameters)
        self._implement = Implementation
        self._rules = {}
        self.operands = []

    def implementation(self, new: typing.Type[Implementation]=None):
        """Get or set the implementation type of this operator."""
        if new:
            self._implement = new
            return self
        return self._implement

    @property
    def rules(self) -> typing.Dict[Types, Parameters]:
        """The operand rules for this operator."""
        for rule in self.operands:
            if rule not in self._rules:
                self._rules[rule] = self._default.copy()
        return self._rules

    def __getitem__(self, types: Types) -> Rule:
        """Retrieve the operand-update rule for `types`."""
        if types in self.rules:
            parameters = self._rules[types]
            return Rule(types, *parameters)
        raise KeyError(
            f"No rule for operand type(s) {types!r}"
        ) from None


class Interface(collections.abc.Mapping):
    """An updatable interface to generalized operators."""

    _operators: typing.Dict[str, Operator]

    def __init__(self, *parameters: str) -> None:
        super().__init__()
        self.parameters = list(parameters).copy()
        """The default parameters for these operators."""
        self._operators = {}

    def register(self, key: str):
        """Register a new operator."""
        if key in self._operators:
            raise KeyError(f"Operator {key!r} already exists.")
        self._operators[key] = Operator(self.parameters)

    def __len__(self) -> int:
        """Returns the number of operators. Called for len(self)."""
        return len(self._operators)

    def __iter__(self) -> typing.Iterator:
        """Iterate over registered operators. Called for iter(self)."""
        return iter(self._operators)

    def __getitem__(self, key: str):
        """Retrieve an operator by keyword. Called for self[key]."""
        if key in self._operators:
            return self._operators[key]
        raise KeyError(f"No operator for {key!r}") from None


