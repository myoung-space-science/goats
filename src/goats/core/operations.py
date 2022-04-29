import abc
import collections
import collections.abc
import functools
import operator as standard
import typing

from goats.core import aliased
from goats.core import iterables
from goats.core import metric
from goats.core import utilities


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


class Operand:
    """A class representing a single operand."""

    def __init__(self, operand, parameters: typing.Container[str]) -> None:
        self.operand = operand
        self.parameters = parameters
        self._type = type(operand)

    def validate(self, other):
        """Make sure `other` is a valid co-operand."""
        if not isinstance(other, self._type):
            return # Indeterminate
        for name in self.parameters:
            if not self._comparable(other, name):
                return False
        return True

    def _comparable(self, that, name: str) -> bool:
        """Determine whether the instances are comparable."""
        return utilities.getattrval(
            self.operand, name
        ) == utilities.getattrval(that, name)


class ComparisonError(TypeError):
    """Incomparable instances of the same type."""

    def __init__(self, __this: typing.Any, __that: typing.Any, name: str):
        self.this = getattr(__this, name, None)
        self.that = getattr(__that, name, None)

    def __str__(self) -> str:
        return f"Can't compare {self.this!r} to {self.that!r}"


class Result(collections.UserDict):
    """The result of applying a rule to an operation."""

    def __init__(self, __attributes, *parameters: str) -> None:
        super().__init__(__attributes)
        self.names = parameters
        """The names of all parameters in the target operands."""

    def format(self, form=None):
        """Convert this result into the appropriate object.
        
        Parameters
        ----------
        form : Any, default=`None`
            The final form that will contain this result's data. See Returns for
            details of how the type of `form` affects the result of this method.

        Returns
        -------
        Any
            If `form` is a ``type``, this method will return a new instance of
            that type, initialized with this result's data. If `form` is an
            instance of some type, this method will return the updated instance.
        """
        if isinstance(form, type):
            return form(**self.data)
        for name in self.names:
            value = self.data.get(name)
            utilities.setattrval(form, name, value)
        return form


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
        self.issuppressed = False

    @property
    def updated(self):
        """The parameters affected by this rule."""
        if self._parameters is not None:
            return tuple(self._parameters)
        return ()

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

    def __bool__(self) -> bool:
        """True unless this rule is suppressed."""
        return not self.issuppressed

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
        self.issuppressed = True
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
        if self.issuppressed:
            raise TypeError("Can't update suppressed rule") from None

    def validate(self, *args):
        """Ensure arguments are consistent with fixed parameters.
        
        This method does not check argument types; rather, it checks whether all
        arguments with an "ignored" attribute have the same value for that
        attribute. This is necessary, for example, when performing a comparison
        operation on two objects with a ``unit`` attribute. If those objects
        have different units, the comparison is meaningless.
        """
        if not self.ignored or len(args) == 1:
            return
        this, *those = args
        target = Operand(this, self.ignored)
        for that in those:
            if target.validate(that) is False:
                raise ComparisonError(this, that)

    def __str__(self) -> str:
        names = [t.__qualname__ for t in self.types]
        types = names[0] if len(names) == 1 else tuple(names)
        parameters = (
            NotImplemented if self.updated is None
            else list(self.updated)
        )
        return f"{types}: {parameters}"


class Rules(typing.Mapping[Types, Rule], collections.abc.Mapping):
    """A class for managing operand-update rules."""

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

    def __iter__(self) -> typing.Iterator[Rule]:
        """Iterate over rules. Called for iter(self)."""
        for types in self._rulemap:
            yield Rule(types, self._rulemap[types])

    def __getitem__(self, key: Types):
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


class OperandError(Exception):
    """Operands are incompatible with operator."""


AType = typing.TypeVar('AType', bound=type)
BType = typing.TypeVar('BType', bound=type)
RType = typing.TypeVar('RType')


class Operator:
    """Base class for operator application schemes."""

    def __init__(
        self,
        __callable: typing.Callable[..., RType],
        rules: Rules=None,
    ) -> None:
        self.method = __callable
        self.rules = rules
        self._reference = None

    def reference(self, new=None):
        """Get or set the reference operand."""
        if new is None:
            return self._reference
        self._reference = new
        return self

    def evaluate(self, *args, **kwargs):
        """Evaluate the arguments with the current method."""
        if self.rules is None:
            return self.method(*args, **kwargs)
        try:
            result = self._evaluate(*args, **kwargs)
        except metric.UnitError as err:
            raise OperandError(err) from err
        else:
            return result

    def _evaluate(self, *args, **kwargs):
        """Internal evaluation logic."""
        types = tuple(type(i) for i in args)
        rule = self.rules.get(types)
        if not rule:
            return NotImplemented
        rule.validate(args)
        updated = {
            name: self.method(
                *[utilities.getattrval(arg, name) for arg in args],
                **kwargs
            ) for name in rule.updated
        }
        if not self.reference():
            return updated
        ignored = {
            name: utilities.getattrval(self.reference(), name)
            for name in rule.ignored
        }
        return {**updated, **ignored}


class Application(typing.Generic[AType]):
    """Base class for operator applications."""

    def __init__(self, rules: Rules) -> None:
        self.rules = rules

    def apply(self, method: typing.Callable):
        """Create an operator by applying `method`."""
        def wrapper(*args, **kwargs):
            return method(*args, **kwargs)
        return wrapper


class Unary(Application):
    """Generalized application of a unary arithmetic operator."""

    def apply(self, method: typing.Callable[[AType], RType]):
        """Create a unary arithmetic operator by applying `method`."""
        operator = Operator(method, self.rules)
        def wrapper(a: AType, **kwargs):
            rule = self.rules.get(type(a))
            if not rule:
                return NotImplemented
            operator.reference(a)
            attrs = operator.evaluate(a, **kwargs)
            return type(a)(**attrs)
        return wrapper


class Cast(Application):
    """Generalized application of a unary cast operator."""

    def apply(self, method: typing.Callable[[AType], RType]):
        """Create a unary cast operator by applying `method`."""
        def wrapper(a: AType):
            rule = self.rules.get(type(a))
            if not rule:
                return NotImplemented
            # There should only be one updatable attribute for a cast.
            return method(utilities.getattrval(a, rule.updated))
        return wrapper


class Binary(Application):
    """Generalized application of a binary arithmetic operator."""

    # Add `mode` keyword for forward/reverse/inplace?
    def apply(self, method: typing.Callable[[AType, BType], RType]):
        """Create a binary arithmetic operator by applying `method`."""
        operator = Operator(method, self.rules)
        def wrapper(a: AType, b: BType, **kwargs):
            rule = self.rules.get(type(a), type(b))
            if not rule:
                return NotImplemented
            rule.validate(a, b)
            operator.reference(a)
            attrs = operator.evaluate(a, b, **kwargs)
            return type(a)(**attrs)
        return wrapper


class Comparison(Application):
    """Generalized application of a binary comparison operator."""

    def apply(self, method: typing.Callable[[AType, BType], RType]):
        """Create a binary comparison operator by applying `method`."""
        def wrapper(a: AType, b: BType):
            rule = self.rules.get(type(a), type(b))
            if not rule:
                return NotImplemented
            rule.validate(a, b)
            # There should only be one updatable attribute for a comparison.
            args = [
                utilities.getattrval(a, rule.updated),
                utilities.getattrval(b, rule.updated),
            ]
            return method(*args)
        return wrapper


IType = typing.TypeVar('IType', bound=Application)
IType = typing.Union[
    Unary,
    Cast,
    Binary,
    Comparison,
]


class Implementation(iterables.ReprStrMixin):
    """A generalized arithmetic operator implementation."""

    def __init__(self, rules: Rules) -> None:
        self._type = Application
        self._rules = {}
        self.rules = rules
        self._operations = []

    def operations(self, *operations: str):
        """Update the operations that this implementation handles."""
        if not operations:
            return tuple(self._operations)
        new = self._operations.copy()
        new.extend(operations)
        self._operations = prune(new)
        return self

    def apply(self, new: typing.Type[IType]):
        """Set the application class for this operator."""
        self._type = new
        return self

    # This should return an operation-specific operator -- namely, one whose
    # type signature is more specific than `(*args, **kwargs)`.
    def implement(self, __callable: typing.Callable) -> IType:
        """Implement an operator with the given callable."""
        application = self._type(self.rules)
        return application.apply(__callable)

    def __str__(self) -> str:
        name = self._type.__name__
        if self._operations:
            return f"{name}: {self._operations}"
        return name


class Interface(aliased.MutableMapping):
    """An updatable interface to operator implementations."""

    _internal: typing.Dict[str, Implementation]

    def __init__(self, __type, *parameters: str) -> None:
        super().__init__()
        self._type = __type
        self.rules = Rules(__type, parameters)
        """The default operand-update rules for these implementations."""
        self._internal = {}

    def register(self, key: str):
        """Register a new implementation."""
        if key in self:
            raise KeyError(f"Implementation {key!r} already exists.")
        self[key] = Implementation(self.rules)
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
        if len(self) == 0:
            return str(self._type)
        implementations = ', '.join(
            f"'{g}'={v}"
            for g, v in zip(self._aliased.keys(), self._aliased.values())
        )
        return f"{self._type}: {implementations}"

