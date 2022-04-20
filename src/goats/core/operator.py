import abc
import collections.abc
import math
import operator as standard
import typing

from goats.core import aliased


Types = typing.TypeVar('Types', type, tuple)
Types = typing.Union[type, typing.Tuple[type, ...]]


Parameters = typing.TypeVar('Parameters', str, typing.Collection)
Parameters = typing.Union[str, typing.Collection[str]]


class Category(abc.ABC):
    """Abstract base class representing an operator category."""

    def __init__(self, reference: type, *operands: Types) -> None:
        self.reference = reference
        self._operands = self._normalize(operands)

    size: typing.ClassVar=None

    def __len__(self) -> int:
        """The number of operands per function call."""
        return len(self._operands)

    def __iter__(self) -> typing.Iterator:
        """Iterate over known operand groups."""
        return iter(self._operands)

    @abc.abstractmethod
    def identify(self, types: Types) -> Types:
        """Determine the supported operands corresponding to `types`."""
        pass

    @abc.abstractmethod
    def parse(self, *args) -> typing.Tuple[Types, Parameters]:
        """Parse the arguments into types and updatable parameters."""
        pass

    def operands(self, *operands: Types):
        """Get or set the collection of operands."""
        if operands:
            self._operands = self._normalize(operands)
            return self
        return self._operands

    def _normalize(self, operands: Types):
        """Convert a collection of operands to a standard form."""
        return [operand for operand in operands if operand]


class Unary(Category):
    """Concrete representation of a unary operator."""

    size = 1

    def identify(self, target: type) -> Types:
        for operand in self.operands:
            if issubclass(target, operand):
                return operand


class Binary(Category):
    """Concrete representation of a binary operator."""

    size = 2

    def identify(self, types: Types):
        for pair in self.operands:
            if all(issubclass(i, j) for i, j in zip(types, pair)):
                return pair


class Rules(collections.abc.MutableMapping):
    """A mapping from type(s) to updatable parameters(s)."""

    def __init__(
        self,
        category: Category,
        rules: typing.Mapping[Types, Parameters]=None,
    ) -> None:
        self.category = category
        self._rules = rules
        self.category.operands(self.rules)

    @property
    def rules(self) -> typing.Dict[Types, typing.List[str]]:
        if self._rules is None:
            self._rules = {}
        return self._rules

    @property
    def fixed(self):
        """The names of immutable attributes."""
        return tuple(self.rules.get(None, ()))

    def __len__(self) -> int:
        """Compute the number of rules. Called for len(self)."""
        return len(self.rules)

    def __iter__(self) -> typing.Iterator:
        """Iterate over rules. Called for iter(self)."""
        return iter(self.rules)

    def __getitem__(self, types: Types):
        """Get the updatable attributes for the given arguments type(s)."""
        if operands := self.category.identify(types):
            return self.rules[operands]
        raise KeyError(
            f"No updatable attributes for types {types!r}"
        ) from None

    def __setitem__(
        self,
        types: Types,
        attributes: typing.Iterable[str],
    ) -> None:
        """Set the updatable attributes for the given arguments type(s)."""
        self.rules[types] = list(attributes)
        self.category.operands(self.rules)
        return self

    def __delitem__(self, types: Types) -> None:
        """Delete the attribute-updating rule."""
        if types in self.rules:
            del self.rules[types]
            self.category.operands(self.rules)
            return self
        raise KeyError(types)


class Category:
    """A representation of an operator category."""

    def __init__(
        self,
        __type: type,
        rules: typing.Mapping[Types, Parameters]=None,
    ) -> None:
        self._type = __type
        self.rules = rules


class Implementation:
    """The default operator implementation."""

    def __init__(self, rules: Rules) -> None:
        self.rules = rules


class Operator:
    """"""

    def __init__(
        self,
        __callable: typing.Callable,
        rules: Rules,
        attributes: typing.Iterable[str],
        implementation: Implementation=None,
    ) -> None:
        self.callable = __callable
        self.rules = rules
        self.attributes = list(attributes)
        self.implementation = implementation

    @typing.overload
    def register(self, __type, *attributes): ...

    @typing.overload
    def register(self, *attributes): ...

    def register(self, *args):
        """Register an attribute-updating rule."""
        if not args:
            raise TypeError("Nothing to register.")
        # rule = list(args)
        # types = (
        #     (self._type, rule.pop(0)) if isinstance(rule[0], type)
        #     else self._type
        # )
        types, parameters = self.category.parse(list(args))
        self.rules[types] = parameters

    def apply(
        self,
        implementation: Implementation=None,
        rules: typing.Mapping[Types, Parameters]=None,
    ) -> 'Operator':
        """Apply the given argument(s) to this operator."""
        if implementation:
            self.implementation = implementation
        if rules:
            self.rules.update(rules)
        return self

    def implement(self, implementation: Implementation):
        """Set the implementation for this operator."""
        self.implementation = implementation
        return self

    def evaluate(self, *args, **kwargs):
        """Apply the current implementation to the given arguments."""
        # See measurable.OperatorABC.evaluate
        return self.implementation(*args, **kwargs)


class Group(collections.abc.MutableSequence):
    """A mutable sequence of operators.
    
    This class behaves like a standard ``list`` of `~operator.Operator`
    instances with an additional `register` method that applies the given rule
    to all its operators.
    """

    def __init__(self, *included: Operator) -> None:
        super().__init__()
        self.included = list(included)

    def apply(self, **attributes) -> None:
        """Apply the given attribute(s) to all operators in this group."""
        for operator in self.included:
            operator.apply(**attributes)

    def __getitem__(self, __i: typing.SupportsIndex):
        """Retrieve the operator at index `__i`."""
        return self.included[__i]

    def __setitem__(self, __i: typing.SupportsIndex, __o: Operator):
        """Replace the operator at index `__i` with `__o`."""
        self.included[__i] = __o

    def __delitem__(self, __i: typing.SupportsIndex):
        """Delete the operator at index `__i`."""
        del self.included[__i]

    def __len__(self):
        """Compute the length of this group. Called for len(self)."""
        return len(self.included)

    def insert(self, __i: typing.SupportsIndex, __o: Operator):
        """Insert operator `__o` at index `__i`."""
        return self.included.insert(__i, __o)


T = typing.TypeVar('T')


class Interface:
    """An interface to operator implementations."""

    def __init__(
        self,
        __object: T,
        *attributes,
    ) -> None:
        self._type = type(__object)
        self.attributes = list(attributes)
        """The default list of attributes that are valid operands."""
        self.methods = STANDARD.copy()
        """An aliased mapping of method names to callable objects."""
        self._groups = None

    def register(self, *attributes: str):
        """Add default operand attributes."""
        self.attributes.extend(attributes)
        return self

    def group(self, key: str, *included: str):
        """Create or retrieve an operator group."""
        if not included and key in self.groups:
            return self.groups[key]
        operators = [self._build(name) for name in included]
        group = Group(*operators)
        self._groups[key] = group
        return group

    @property
    def groups(self) -> typing.Dict[str, Group]:
        """All registered operator groups."""
        if self._groups is None:
            self._groups = {}
        return self._groups

    @property
    def add(self):
        """Addition: a + b"""
        return self._build('add', Binary)

    # This needs to coordinate with `self.groups`.
    def _build(self, name: str, category: typing.Type[Category]):
        """Build an operator by name."""
        if name in self.methods:
            return Operator(
                self.methods[name],
                Rules(category(self._type)),
                self.attributes,
            )
        raise KeyError(f"Unknown operator {name!r}")


_methods = {
    'abs': standard.abs,
    'pos': standard.pos,
    'neg': standard.neg,
    'round': round,
    'ceil': math.ceil,
    'floor': math.floor,
    'trunc': math.trunc,
    'float': float,
    'int': int,
    'lt': standard.lt,
    'le': standard.le,
    'gt': standard.gt,
    'ge': standard.ge,
    'eq': standard.eq,
    'ne': standard.ne,
    'add': standard.add,
    'sub': standard.sub,
    'mul': standard.mul,
    'truediv': standard.truediv,
    'pow': standard.pow,
}
STANDARD = aliased.Mapping(
    {
        (name, f'__{name}__'): method
        for name, method in _methods.items()
    }
)

# Operator taxonomy:
# - __abs__ == unary + arithmetic
# - __pos__ == unary + arithmetic
# - __neg__ == unary + arithmetic
# - __round__ == unary + arithmetic
# - __ceil__ == unary + arithmetic
# - __floor__ == unary + arithmetic
# - __trunc__ == unary + arithmetic
# - __float__ == unary + cast
# - __int__ == unary + cast
# - __lt__ == binary + comparison
# - __le__ == binary + comparison
# - __gt__ == binary + comparison
# - __ge__ == binary + comparison
# - __eq__ == binary + comparison
# - __ne__ == binary + comparison
# - __add__ == binary + additive + forward
# - __radd__ == binary + additive + reverse
# - __iadd__ == binary + additive + inplace
# - __sub__ == binary + additive + forward
# - __rsub__ == binary + additive + reverse
# - __isub__ == binary + additive + inplace
# - __mul__ == binary + multiplicative + forward
# - __rmul__ == binary + multiplicative + reverse
# - __imul__ == binary + multiplicative + inplace
# - __truediv__ == binary + multiplicative + forward
# - __rtruediv__ == binary + multiplicative + reverse/suppressed
# - __itruediv__ == binary + multiplicative + inplace
# - __pow__ == binary + exponential + forward
# - __rpow__ == binary + exponential + reverse/suppressed
# - __ipow__ == binary + exponential + inplace


