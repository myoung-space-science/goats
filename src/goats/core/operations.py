import abc
import collections
import collections.abc
import contextlib
import inspect
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


class Operand(typing.Generic[T], iterables.ReprStrMixin):
    """A class representing a single operand."""

    def __init__(self, __object: T) -> None:
        self._object = __object
        self._type = type(__object)
        self._parameters = None
        self._positional = None
        self._keyword = None

    @property
    def parameters(self):
        """All parameters used to initialize this operand."""
        if self._parameters is None:
            self._parameters = (
                {} if self._type.__module__ == 'builtins'
                else inspect.signature(self._type).parameters
            )
        return self._parameters

    _postypes = {
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    }

    @property
    def positional(self):
        """The names of positional arguments to this operand."""
        if self._positional is None:
            names = [
                name for name, parameter in self.parameters.items()
                if parameter.kind in self._postypes
            ]
            self._positional = tuple(names)
        return self._positional

    @property
    def keyword(self):
        """The names of keyword arguments to this operand."""
        if self._keyword is None:
            names = [
                name for name, parameter in self.parameters.items()
                if parameter.kind == inspect.Parameter.KEYWORD_ONLY
            ]
            self._keyword = tuple(names)
        return self._keyword

    def validate(self, other, *ignored: str):
        """Make sure `other` is a valid co-operand."""
        if not isinstance(other, self._type):
            return # Indeterminate
        for name in ignored:
            if not self._comparable(other, name):
                return False
        return True

    def _comparable(self, that, name: str) -> bool:
        """Determine whether the instances are comparable."""
        v0 = utilities.getattrval(self, name)
        v1 = utilities.getattrval(that, name)
        return v0 == v1

    def __eq__(self, other):
        """Called for self == other."""
        return (
            self._object == other._object if isinstance(other, Operand)
            else self._object == other
        )

    def __getattr__(self, __name: str):
        """Retrieve an attribute from the underlying object."""
        return getattr(self._object, __name)

    def __str__(self) -> str:
        return str(self._object)


class ComparisonError(TypeError):
    """Incomparable instances of the same type."""

    def __init__(self, __this: typing.Any, __that: typing.Any, name: str):
        self.this = getattr(__this, name, None)
        self.that = getattr(__that, name, None)

    def __str__(self) -> str:
        return f"Can't compare {self.this!r} to {self.that!r}"


class Result(iterables.ReprStrMixin):
    """The result of applying a rule to an operation."""

    def __init__(self, *args, **kwargs) -> None:
        self.args = list(args)
        self.kwargs = kwargs

    def format(self, form):
        """Convert this result into the appropriate object.
        
        Parameters
        ----------
        form : Any
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
            parameters = inspect.signature(form).parameters
            args, kwargs = self._get_args_kwargs(parameters)
            return form(*args, **kwargs)
        parameters = inspect.signature(type(form)).parameters
        args, kwargs = self._get_args_kwargs(parameters)
        for name in parameters:
            value = self._arg_or_kwarg(args, kwargs, name)
            utilities.setattrval(form, name, value)
        return form

    def _get_args_kwargs(self, signature: inspect.Signature):
        """Extract appropriate argument values.
        
        This method will attempt to build appropriate positional and keyword
        arguments from this result, based on the parameters in `signature`.
        """
        args = []
        kwargs = {}
        for name, parameter in signature.parameters.items():
            kind = parameter.kind
            if kind is inspect.Parameter.POSITIONAL_ONLY:
                args.append(self.args.pop(0))
            elif kind is inspect.Parameter.POSITIONAL_OR_KEYWORD:
                args.append(self._arg_or_kwarg(args, kwargs, name))
            elif kind is inspect.Parameter.KEYWORD_ONLY:
                kwargs[name] = self.kwargs.get(name)
        return tuple(args), kwargs

    def _arg_or_kwarg(self, args: list, kwargs: dict, name: str):
        """Get the value of a positional or keyword argument, if possible.
        
        This checks for the presence of `name` in `kwargs` rather than calling
        `kwargs.get(name)` in order to avoid prematurely returning `None` before
        trying to retrieve a value from `args`.
        """
        if name in kwargs:
            return kwargs[name]
        with contextlib.suppress(IndexError):
            return args.pop(0)

    def __eq__(self, other) -> bool:
        """True iff two results have equal arguments."""
        return isinstance(other, Result) and (
            self.args == other.args and self.kwargs == other.kwargs
        )

    def __str__(self) -> str:
        args = ', '.join(str(arg) for arg in self.args)
        kwargs = ', '.join(f"{k}={v}" for k, v in self.kwargs.items())
        parts = (args, kwargs)
        return ', '.join(part for part in parts if part)


# Possible specialized return type for `Rule.suppress`. Another option is that a
# common ABC could require all the methods the `Rule` defines, and this class
# could implement versions that raise exceptions with informative messages.
class Suppressed(iterables.Singleton):
    """A suppressed operand rule."""

    def __bool__(self) -> bool:
        return False

    def __str__(self) -> str:
        names = [t.__qualname__ for t in self.types]
        types = names[0] if len(names) == 1 else tuple(names)
        return f"{types}: NotImplemented"


# TODO: New `Rule` class:
# - requires affected parameters
# - does not require default parameters
# - computes ignored parameters based on `reference` in `apply`
# - passes ignored parameters to some version of `validate`, which may be in a
#   different class (e.g., `measurable.same`)
# - will require updates to `Rules`


class Rule:
    """"""

    def __init__(
        self,
        __types: Types,
        *parameters: str,
    ) -> None:
        self._types = list(iterables.whole(__types))
        self._ntypes = len(self._types)
        self.parameters = prune(parameters)
        """The parameters that this rule affects."""
        self.issuppressed = False

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

    # Consider letting this return `Suppressed` (see above).
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

    def apply(self, method, *args, reference=None, **kwargs):
        """Call `method` on arguments within the context of this rule."""
        target = Operand(reference or args[0])
        if not target.parameters:
            return Result(method(*args, **kwargs))
        a = []
        k = {}
        for name in target.positional:
            a.append(
                method(
                    *[utilities.getattrval(arg, name) for arg in args],
                    **kwargs
                ) if name in self.parameters
                else utilities.getattrval(reference, name)
            )
        for name in target.keyword:
            k[name] = (
                method(
                    *[utilities.getattrval(arg, name) for arg in args],
                    **kwargs
                ) if name in self.parameters
                else utilities.getattrval(reference, name)
            )
        result = Result(*a, **k)
        return result

    def __str__(self) -> str:
        names = [t.__qualname__ for t in self.types]
        types = names[0] if len(names) == 1 else tuple(names)
        return f"{types}: {list(self.parameters)}"


class Rules(typing.Mapping[Types, Rule], collections.abc.Mapping):
    """A class for managing operand-update rules."""

    def __init__(self, *parameters: str, nargs: int=None) -> None:
        self.default = list(parameters)
        """The default parameters to update for each rule."""
        self.nargs = nargs
        """The number of arguments in these rules."""
        self._rulemap = None

    def register(self, key: Types, *parameters: typing.Optional[str]):
        """Add an update rule to the collection."""
        types = tuple(key) if isinstance(key, typing.Iterable) else (key,)
        nargs = len(types)
        self._check_nargs(nargs)
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

    def _check_nargs(self, __nargs: int):
        """Helper for enforcing consistency in number of types.
        
        If the internal attribute that keeps track of how many arguments are
        allowed for all rules is `None`, this method will set it on the first
        time through. After that, it will raise an exception if the user tries
        to register a rule with a different number of types.
        """
        if self.nargs is None:
            self.nargs = __nargs
            return
        if __nargs != self.nargs:
            raise ValueError(
                f"Can't add {__nargs} type(s) to collection"
                f" of length-{self.nargs} items."
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
        parameters = self.mapping.get(__types, self.default.copy())
        return Rule(__types, *parameters)


# An operation is the result of applying an operator to operands.
class Operation:
    """The application of an operator to one or more operand(s)."""

    def __init__(
        self,
        rule: Rule,
        reference: typing.Any=None,
        form: typing.Any=None,
    ) -> None:
        self.rule = rule
        self.reference = reference
        self.form = form

    def __bool__(self) -> bool:
        """True if this operation has a defined rule."""
        return bool(self.rule)

    def apply(self, method: typing.Callable, *args, **kwargs):
        """Apply `method` to the given arguments."""
        result = self.rule.apply(
            method,
            *args,
            reference=self.reference,
            **kwargs
        )
        if self.form:
            return result.format(self.form)
        return result


class Context:
    """The context of an operation."""

    def __init__(
        self,
        *parameters: str,
        nargs: int=None,
    ) -> None:
        self.rules = Rules(*parameters, nargs=nargs)

    def rule(self, __types: Types, *parameters: str):
        """Register an operand-update rule for this context."""
        self.rules.register(__types, *parameters)
        return self

    def interpret(self, *args):
        """Create an operation based on argument types.
        
        This method will determine the appropriate operand-update rule, if any,
        for `args`. If there is no rule appropriate to `args`, the corresponding
        operation will not be implemented.
        """
        rule = self.get_rule(*args)
        return Operation(rule)

    def supports(self, *args):
        """True if this context has an operand-update rule for `args`."""
        return bool(self.get_rule(*args))

    def get_rule(self, *args):
        """Retrieve an update-update rule for `args`, if possible."""
        types = tuple(type(i) for i in args)
        return self.rules.get(types)


# Contexts
# - default: return result of `method(*args, **kwargs)`
# - cast: default
# - comparison: default
# - numeric: return result as type of first argument
# - reverse: return result as type of second argument
# - inplace: return result as updated first argument


RT = typing.TypeVar('RT')


class Operator:
    """A generalized numerical operator."""

    def __init__(
        self,
        method: typing.Callable,
        context: Context,
    ) -> None:
        self.method = method
        self.context = context
        # Context will handle number of operands, return type, etc.

    def support(self, __types: Types):
        """Support operations on the given type(s)."""
        self.context.rule(__types)
        return self

    def __call__(self, *args, **kwargs):
        """Evaluate the arguments within the given context."""
        if not self.context.supports(*args):
            return NotImplemented
        operation = self.context.interpret(*args)
        return operation.apply(self.method, *args, **kwargs)


# Functional implementations:

class OperandError(Exception):
    """Operands are incompatible with operator."""


AType = typing.TypeVar('AType', bound=type)
BType = typing.TypeVar('BType', bound=type)


def default(method, rule: Rule=None):
    """Create a default operation."""
    def wrapper(*args, **kwargs):
        if not rule:
            return NotImplemented
        return method(*args, **kwargs)
    wrapper.__name__ = f"__{method.__name__}__"
    wrapper.__doc__ = method.__doc__
    return wrapper


def unary(method, rule: Rule=None) -> AType:
    """Create a unary arithmetic operation."""
    def wrapper(a: AType, **kwargs):
        if not rule:
            return NotImplemented
        return rule.apply(method, a, reference=a, **kwargs).format(type(a))
    wrapper.__name__ = f"__{method.__name__}__"
    wrapper.__doc__ = method.__doc__
    return wrapper


def cast(method, rule: Rule=None):
    """Create a unary cast operation."""
    def wrapper(a: AType):
        if not rule:
            return NotImplemented
        return rule.apply(method, a)
    wrapper.__name__ = f"__{method.__name__}__"
    wrapper.__doc__ = method.__doc__
    return wrapper


def forward(method, rule: Rule=None):
    """Create a forward binary arithmetic operation."""
    def wrapper(a: AType, b: BType, **kwargs):
        if not rule:
            return NotImplemented
        try:
            result = rule.apply(method, a, b, reference=a, **kwargs)
        except metric.UnitError as err:
            raise OperandError(err) from err
        else:
            return result.format(type(a))
    wrapper.__name__ = f"__{method.__name__}__"
    wrapper.__doc__ = method.__doc__
    return wrapper


def reverse(method, rule: Rule=None):
    """Create a reverse binary arithmetic operation."""
    def wrapper(b: BType, a: AType, **kwargs):
        if not rule:
            return NotImplemented
        try:
            result = rule.apply(method, a, b, reference=b, **kwargs)
        except metric.UnitError as err:
            raise OperandError(err) from err
        else:
            return result.format(type(b))
    wrapper.__name__ = f"__r{method.__name__}__"
    wrapper.__doc__ = method.__doc__
    return wrapper


def inplace(method, rule: Rule=None):
    """Create a inplace binary arithmetic operation."""
    def wrapper(a: AType, b: BType, **kwargs):
        if not rule:
            return NotImplemented
        try:
            result = rule.apply(method, a, b, reference=a, **kwargs)
        except metric.UnitError as err:
            raise OperandError(err) from err
        else:
            return result.format(a)
    wrapper.__name__ = f"__i{method.__name__}__"
    wrapper.__doc__ = method.__doc__
    return wrapper


def comparison(method, rule: Rule=None):
    """Create a binary comparison operation."""
    def wrapper(a: AType, b: BType, **kwargs):
        if not rule:
            return NotImplemented
        return rule.apply(method, a, b, **kwargs)
    wrapper.__name__ = f"__{method.__name__}__"
    wrapper.__doc__ = method.__doc__
    return wrapper


class Implementation:
    """"""

    def __init__(self, method, definition=None) -> None:
        self.method = method
        self._definition = definition
        self._operator = None

    @property
    def definition(self):
        """"""
        if self._definition is None:
            self._definition = default
        return self._definition

    def apply(self, rule: Rule):
        """Evaluate the arguments according to this operand rule."""
        func = self.definition(self.method, rule)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        wrapper.__name__ = f"__{func.__name__}__"
        wrapper.__doc__ = func.__doc__
        return wrapper


class Operator:
    """"""

    def __init__(
        self,
        __implementation: Implementation,
        rules: Rules,
    ) -> None:
        self._implementation = __implementation
        self.rules = rules

    def __call__(self, *args, **kwargs):
        """"""
        types = (type(arg) for arg in args)
        rule = self.rules.get(types)
        if not rule:
            return NotImplemented
        operation = self._implementation.apply(rule)
        return operation(*args, **kwargs)


class Context:
    """A general operation context."""

    def __init__(
        self,
        rules: Rules,
        definition=None,
    ) -> None:
        self.rules = rules
        """The operand-update rules for this context."""
        self._definition = definition

    @property
    def definition(self):
        """"""
        if self._definition is None:
            self._definition = default
        return self._definition

    def implement(self, method):
        """"""
        def operator(*args, **kwargs):
            types = (type(arg) for arg in args)
            rule = self.rules.get(types)
            if not rule:
                return NotImplemented
            func = self._definition(method, rule)
            return func(*args, **kwargs)
        operator.__name__ = f"__{method.__name__}__"
        operator.__doc__ = method.__doc__
        return operator


class Interface:
    """"""

    # The user provides the name of the attribute that contains numerical data.
    # If absent, operations will treat (the) entire input argument(s) as the
    # numerical data. This behavior will support for subclasses of
    # `numbers.Number`. The user can add affected attributes for specific
    # operators. The `Rules` class may need to change in order to accommodate.
    def __init__(self, __type: type, target: str=None) -> None:
        self._type = __type
        self.target = target
        """The name of the data-like attribute."""
        # This should probably be an aliased mapping that associates
        # operators with categories by default.
        self._implementations = None
        self._unary = None
        self._cast = None
        self._comparison = None
        self._forward = None
        self._reverse = None
        self._inplace = None

    @property
    def unary(self):
        """The implementation context for unary arithmetic operations."""
        if self._unary is None:
            rules = Rules(self.target, nargs=1)
            rules.register(self._type)
            self._unary = Context(unary, rules)
        return self._unary

    @property
    def cast(self):
        """The implementation context for unary casting operations."""
        if self._cast is None:
            rules = Rules(self.target, nargs=1)
            rules.register(self._type)
            self._cast = Context(cast, rules)
        return self._cast

    @property
    def comparison(self):
        """The implementation context for binary comparison operations."""
        if self._comparison is None:
            rules = Rules(self.target, nargs=2)
            rules.register([self._type, self._type])
            self._comparison = Context(comparison, rules)
        return self._comparison

    @property
    def forward(self):
        """The implementation context for standard numeric operations."""
        if self._forward is None:
            rules = Rules(self.target, nargs=2)
            rules.register([self._type, self._type])
            self._forward = Context(forward, rules)
        return self._forward

    @property
    def reverse(self):
        """The implementation context for reversed numeric operations."""
        if self._reverse is None:
            rules = Rules(self.target, nargs=2)
            rules.register([self._type, self._type])
            self._reverse = Context(reverse, rules)
        return self._reverse

    @property
    def inplace(self):
        """The implementation context for in-place numeric operations."""
        if self._inplace is None:
            rules = Rules(self.target, nargs=2)
            rules.register([self._type, self._type])
            self._inplace = Context(inplace, rules)
        return self._inplace

    _names = [
        'unary',
        'cast',
        'comparison',
        'forward',
        'reverse',
        'inplace',
    ]

    @property
    def implementations(self):
        """All available implementation contexts."""
        if self._implementations is None:
            self._implementations = aliased.MutableMapping()
        contexts = {name: getattr(self, name) for name in self._names}
        self._implementations.update(contexts)
        return self._implementations

    # The user must be able to request an operation by name. If this instance
    # has an implementation for it, it should return that implementation; if
    # not, it should return a default implementation. The default implementation
    # should operate on the numerical data and return the raw result.
    def find(self, __name: str) -> Context:
        """Get an appropriate implementation context for the named operator."""
        if context := self._implementations.get(__name):
            return context
        rules = Rules(self.target)
        return Context(default, rules)


_operators = [
    ('add', '__add__', 'addition'),
    ('sub', '__sub__', 'subtraction'),
]


_categories = {
    'unary': ['abs', 'pos', 'neg', 'trunc', 'round', 'ceil', 'floor'],
    'cast': ['int', 'float', 'complex'],
    'comparison': ['lt', 'le', 'gt', 'ge', 'eq', 'ne'],
    'numeric': ['add', 'sub', 'mul', 'truediv', 'floordiv', 'pow'],
}
