import collections
import collections.abc
import inspect
import operator as standard
import typing

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


class Result(collections.OrderedDict):
    """The result of applying a rule to an operation."""

    def __init__(self, *parameters: str) -> None:
        super().__init__({p: None for p in parameters})

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
        parameters = inspect.signature(form).parameters
        args = []
        kwds = {}
        loop = zip(self.values(), parameters.items())
        for value, (name, parameter) in loop:
            kind = parameter.kind
            if kind is inspect.Parameter.POSITIONAL_ONLY:
                args.append(value)
            elif kind is inspect.Parameter.POSITIONAL_OR_KEYWORD:
                args.append(self[name])
            elif kind is inspect.Parameter.KEYWORD_ONLY:
                kwds[name] = self.get(name)
        if isinstance(form, type):
            return form(*args, **kwds)
        for name in self:
            value = self.get(name)
            utilities.setattrval(form, name, value)
        return form


class Rule(iterables.ReprStrMixin):
    """A correspondence between operand types and affected attributes."""

    def __init__(
        self,
        __types: Types,
        *default: str,
    ) -> None:
        self._types = list(iterables.whole(__types))
        self._ntypes = len(self._types)
        self.default = prune(default)
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
        return tuple(
            parameter for parameter in self.default
            if parameter not in self._parameters
        )

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

    def define(self, *parameters: str):
        """Declare the parameters that this rule affects."""
        self._parameters = prune(parameters)
        return self

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

    def apply(self, method, *args, reference=None, **kwargs):
        """Call `method` on arguments within the context of this rule."""
        result = Result(*self.default.copy())
        updated = {
            name: method(
                *[utilities.getattrval(arg, name) for arg in args],
                **kwargs
            ) for name in self.updated
        }
        result.update(updated)
        if reference:
            ignored = {
                name: utilities.getattrval(reference, name)
                for name in self.ignored
            }
            result.update(ignored)
        return result

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
        self.default = list(parameters)
        """The default parameters to update for each rule."""
        self.ntypes = None
        """The number of types in these rules."""
        self._rulemap = None

    def register(self, key: Types, *parameters: typing.Optional[str]):
        """Add an update rule to the collection."""
        types = tuple(key) if isinstance(key, typing.Iterable) else (key,)
        ntypes = len(types)
        self._check_ntypes(ntypes)
        if types not in self.mapping:
            self.mapping[types] = self._resolve(*parameters)
            return self
        raise KeyError(
            f"{key!r} is already in the collection."
        ) from None

    def _resolve(self, *parameters) -> typing.List[str]:
        """Determine the affected parameters based on input."""
        if len(parameters) == 1 and parameters[0] is None:
            return []
        if not parameters:
            return self.default.copy()
        return list(parameters)

    @property
    def mapping(self) -> typing.Dict[Types, Parameters]:
        """The current mapping from types to affected parameters."""
        if self._rulemap is None:
            self._rulemap = {}
        return self._rulemap

    def _check_ntypes(self, __ntypes: int):
        """Helper for enforcing consistency in number of types.
        
        The first time through, this will set the internal attribute that keeps
        track of how many types are allowed for these rules. After that, it will
        raise an exception if the user tries to register a rule with a different
        number of types.
        """
        if self.ntypes is None:
            self.ntypes = __ntypes
            return
        if __ntypes != self.ntypes:
            raise ValueError(
                f"Can't add {__ntypes} type(s) to collection"
                f" of length-{self.ntypes} items."
            ) from None

    def __len__(self) -> int:
        """Returns the number of rules. Called for len(self)."""
        return len(self.mapping)

    def __iter__(self) -> typing.Iterator[Rule]:
        """Iterate over rules. Called for iter(self)."""
        for types in self.mapping:
            yield Rule(types, self.mapping[types])

    def __getitem__(self, key: Types):
        """Retrieve the operand-update rule for `types`."""
        types = tuple(key) if isinstance(key, typing.Iterable) else (key,)
        if types in self.mapping:
            return self._from(types)
        for t in self.mapping:
            if all(issubclass(i, j) for i, j in zip(types, t)):
                return self._from(t)
        raise KeyError(
            f"No rule for operand type(s) {key!r}"
        ) from None

    def _from(self, __types: Types):
        """Build a rule from the given types."""
        rule = Rule(__types, *self.default.copy())
        parameters = self.mapping[__types]
        return rule.define(*parameters)


class OperandError(Exception):
    """Operands are incompatible with operator."""


AType = typing.TypeVar('AType', bound=type)
BType = typing.TypeVar('BType', bound=type)
RType = typing.TypeVar('RType')
OType = typing.TypeVar('OType')
OType = typing.Union[typing.Type[T], T]


class Operator:
    """Base class for operator application schemes."""

    def __init__(
        self,
        __callable: typing.Callable[..., RType],
        rules: Rules=None,
    ) -> None:
        self.method = __callable
        self.rules = rules

    def evaluate(self, *args, mode: str=None, **kwargs):
        """Evaluate the arguments with the current method."""
        if self.rules is None:
            return self.method(*args, **kwargs)
        types = tuple(type(i) for i in args)
        rule = self.rules.get(types)
        if not rule:
            return NotImplemented
        reference = self._get_reference(mode, *args)
        result = self._get_result(rule, *args, reference=reference, **kwargs)
        target = self._get_target(mode, *args)
        if target is None:
            if len(rule.updated) > 1:
                raise ValueError(
                    "Can't represent multiple attributes"
                    " with unknown output type."
                ) from None
            values = list(result.values())
            return values[0]
        return result.format(target)

    def _get_reference(self, mode, *args):
        """Get the reference object based on `mode`."""
        if mode in {'foward', 'inplace'}:
            return args[0]
        if mode == 'reverse':
            return args[-1]
        return mode

    def _get_target(self, mode, *args):
        """Get the output target based on `mode`."""
        if mode == 'forward':
            return type(args[0])
        if mode == 'reverse':
            return type(args[-1])
        if mode == 'inplace':
            return args[0]
        return mode

    def _get_result(
        self,
        rule: Rule,
        *args,
        reference=None,
        **kwargs
    ) -> Result:
        """Compute the result of applying this operator's method."""
        if len(args) == 1:
            arg = args[0]
            return rule.apply(self.method, arg, reference=arg, **kwargs)
        rule.validate(args)
        return rule.apply(self.method, *args, reference=reference, **kwargs)


def unary(operator: Operator) -> AType:
    """Create a unary arithmetic operation from `operator`."""
    def wrapper(a: AType, **kwargs):
        return operator.evaluate(a, mode='forward', **kwargs)
    wrapper.__name__ = f"__{operator.method.__name__}__"
    wrapper.__doc__ = operator.method.__doc__
    return wrapper


def cast(operator: Operator):
    """Create a unary cast operation from `operator`."""
    def wrapper(a: AType):
        return operator.evaluate(a)
    wrapper.__name__ = f"__{operator.method.__name__}__"
    wrapper.__doc__ = operator.method.__doc__
    return wrapper


def _binary(operator: Operator, mode: str='forward'):
    """Create a binary arithmetic operation from `operator`."""
    def evaluate(*args, **kwargs):
        try:
            result = operator.evaluate(*args, mode=mode, **kwargs)
        except metric.UnitError as err:
            raise OperandError(err) from err
        else:
            return result
    return evaluate


def forward(operator: Operator):
    """Create a forward binary arithmetic operation from `operator`."""
    operation = _binary(operator, 'forward')
    def wrapper(a: AType, b: BType, **kwargs):
        return operation(a, b, **kwargs)
    wrapper.__name__ = f"__{operator.method.__name__}__"
    wrapper.__doc__ = operator.method.__doc__
    return wrapper


def reverse(operator: Operator):
    """Create a reverse binary arithmetic operation from `operator`."""
    operation = _binary(operator, 'reverse')
    def wrapper(b: BType, a: AType, **kwargs):
        return operation(a, b, **kwargs)
    wrapper.__name__ = f"__r{operator.method.__name__}__"
    wrapper.__doc__ = operator.method.__doc__
    return wrapper


def inplace(operator: Operator):
    """Create a inplace binary arithmetic operation from `operator`."""
    operation = _binary(operator, 'inplace')
    def wrapper(a: AType, b: BType, **kwargs):
        return operation(a, b, **kwargs)
    wrapper.__name__ = f"__{operator.method.__name__}__"
    wrapper.__doc__ = operator.method.__doc__
    return wrapper


def comparison(operator: Operator):
    """Create a binary comparison operation from `operator`."""
    def wrapper(a: AType, b: BType):
        return operator.evaluate(a, b)
    wrapper.__name__ = f"__i{operator.method.__name__}__"
    wrapper.__doc__ = operator.method.__doc__
    return wrapper


DefinitionT = typing.TypeVar('DefinitionT')
DefinitionT = typing.Callable[[Operator], typing.Callable]


class Implementation(typing.Generic[T]):
    """A generalized operator implementation."""

    def __init__(
        self,
        __definition: DefinitionT,
        rules: Rules,
    ) -> None:
        self._implement = __definition
        self.rules = rules

    def rule(self, __types: Types, *parameters: str):
        """Register an operand-update rule for this implementation."""
        self.rules.register(__types, *parameters)
        return self

    def operator(self, method: typing.Callable):
        """Create an operator implementation from `method`."""
        operator = Operator(method, self.rules)
        return self._implement(operator)


UnaryT = typing.TypeVar('UnaryT', bound=typing.Callable)
UnaryT = typing.Callable[[AType], AType]


NT = typing.TypeVar('NT', int, float, complex)
NT = typing.Union[int, float, complex]


CastT = typing.TypeVar('CastT', bound=typing.Callable)
CastT = typing.Callable[[AType], NT]


BinaryT = typing.TypeVar('BinaryT', bound=typing.Callable)
BinaryT = typing.Callable[[AType, BType], AType]


ComparisonT = typing.TypeVar('ComparisonT', bound=typing.Callable)
ComparisonT = typing.Callable[[AType, BType], bool]


class Operators:
    """An interface to generalized operator implementations."""

    def __init__(self, *parameters: str) -> None:
        self.rules = Rules(*parameters)
        """The default operand-update rules for all operators."""

    def unary(self) -> Implementation[UnaryT]:
        """Unary arithmetic operator implementations."""
        return self.implement(unary)

    def cast(self) -> Implementation[CastT]:
        """Unary cast operator implementations."""
        return self.implement(cast)

    def binary(self, mode: str='forward') -> Implementation[BinaryT]:
        """Binary arithmetic operator implementations."""
        if mode == 'forward':
            return self.implement(forward)
        if mode == 'reverse':
            return self.implement(reverse)
        if mode == 'inplace':
            return self.implement(inplace)
        raise ValueError(f"Unknown mode {mode} for binary operator")

    def comparison(self) -> Implementation[ComparisonT]:
        """Create a comparison arithmetic operator from `method`."""
        return self.implement(comparison)

    def implement(self, definition: DefinitionT):
        """Implement `method` according to `definition`."""
        return Implementation(definition, self.rules)


# IType = typing.TypeVar('IType', bound=Application)
# IType = typing.Union[
#     Unary,
#     Cast,
#     Binary,
#     Comparison,
# ]


# class Implementation(iterables.ReprStrMixin):
#     """A generalized arithmetic operator implementation."""

#     def __init__(self, rules: Rules) -> None:
#         self._type = Application
#         self._rules = {}
#         self.rules = rules
#         self._operations = []

#     def operations(self, *operations: str):
#         """Update the operations that this implementation handles."""
#         if not operations:
#             return tuple(self._operations)
#         new = self._operations.copy()
#         new.extend(operations)
#         self._operations = prune(new)
#         return self

#     def apply(self, new: typing.Type[IType]):
#         """Set the application class for this operator."""
#         self._type = new
#         return self

#     @property
#     def application(self):
#         """"""
#         return self._type(self.rules)

#     # This should return an operation-specific operator -- namely, one whose
#     # type signature is more specific than `(*args, **kwargs)`.
#     def implement(self, __callable: typing.Callable):
#         """Implement an operator with the given callable."""
#         application = self._type(self.rules)
#         return application.apply(__callable)

#     def __str__(self) -> str:
#         name = self._type.__name__
#         if self._operations:
#             return f"{name}: {self._operations}"
#         return name


# class Interface(aliased.MutableMapping):
#     """An updatable interface to operator implementations."""

#     _internal: typing.Dict[str, Implementation]

#     def __init__(self, __type, *parameters: str) -> None:
#         super().__init__()
#         self._type = __type
#         self.rules = Rules(__type, parameters)
#         """The default operand-update rules for these implementations."""
#         self._internal = {}

#     def register(self, key: str):
#         """Register a new implementation."""
#         if key in self:
#             raise KeyError(f"Implementation {key!r} already exists.")
#         self[key] = Implementation(self.rules)
#         return self

#     def __getitem__(self, key: str) -> Implementation:
#         """Retrieve an implementation by keyword. Called for self[key]."""
#         try:
#             return super().__getitem__(key)
#         except KeyError:
#             for implementation in self.values(aliased=True):
#                 if key in implementation.operations():
#                     return implementation
#         raise KeyError(f"No implementation for {key!r}")

#     def __str__(self) -> str:
#         if len(self) == 0:
#             return str(self._type)
#         implementations = ', '.join(
#             f"'{g}'={v}"
#             for g, v in zip(self._aliased.keys(), self._aliased.values())
#         )
#         return f"{self._type}: {implementations}"


