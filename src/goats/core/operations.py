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


# An operation is the result of applying an operator to operands.

# Categories:
# - cast
#  - (a: T) -> r: Any
#  - no reference
# - unary
#  - (a: T, **kwargs) -> r: T
#  - reference is a
# - comparison
#  - (a: T, b: T | Any) -> r: Any
#  - no reference
# - forward
#  - (a: T, b: T | Any, **kwargs) -> r: T
#  - reference is a
# - reverse
#  - (a: T | Any, b: T, **kwargs) -> r: T
#  - reference is b
# - inplace
#  - (a: T, b: T | Any, **kwargs) -> a: T
#  - reference is a


Types = typing.TypeVar('Types', type, tuple)
Types = typing.Union[type, typing.Tuple[type, ...]]


Parameters = typing.TypeVar('Parameters', str, typing.Collection)
Parameters = typing.Union[str, typing.Collection[str]]


T = typing.TypeVar('T')


def unique(items: typing.Iterable[T]) -> typing.List[T]:
    """Remove repeated items while preserving order."""
    collection = []
    for item in items:
        if item not in collection:
            collection.append(item)
    return collection


class Object(typing.Generic[T], iterables.ReprStrMixin):
    """A wrapper around a single object."""

    def __init__(self, __object: typing.Union[T, 'Object']) -> None:
        self._object = self._init_object(__object)
        self._type = type(self._object)
        self.isbuiltin = self._type.__module__ == 'builtins'
        self._parameters = None
        self._positional = None
        self._keyword = None

    def _init_object(self, arg) -> T:
        """Internal initialization helper."""
        if isinstance(arg, type(self)):
            return arg._object
        return arg

    @property
    def parameters(self):
        """All parameters used to initialize this operand."""
        if self._parameters is None:
            self._parameters = (
                {} if self.isbuiltin
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

    def __eq__(self, other):
        """Called for self == other."""
        return (
            self._object == other._object if isinstance(other, Object)
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


def arg_or_kwarg(args: list, kwargs: dict, name: str):
    """Get the value of a positional or keyword argument, if possible.
    
    This checks for the presence of `name` in `kwargs` rather than calling
    `kwargs.get(name)` in order to avoid prematurely returning `None` before
    trying to retrieve a value from `args`.
    """
    if name in kwargs:
        return kwargs[name]
    with contextlib.suppress(IndexError):
        return args.pop(0)


class Result(iterables.ReprStrMixin):
    """The arguments returned by an operation."""

    def __init__(self, *args, **kwargs) -> None:
        self.args = list(args)
        self.kwargs = kwargs

    def get_values(self, parameters: typing.Mapping[str, inspect.Parameter]):
        """Extract appropriate argument values.
        
        This method will attempt to build appropriate positional and keyword
        arguments from this result, based on the given object signature.
        """
        if not parameters:
            return tuple(self.args), self.kwargs
        args = []
        kwargs = {}
        for name, parameter in parameters.items():
            kind = parameter.kind
            if kind is inspect.Parameter.POSITIONAL_ONLY:
                args.append(self.args.pop(0))
            elif kind is inspect.Parameter.POSITIONAL_OR_KEYWORD:
                args.append(arg_or_kwarg(args, kwargs, name))
            elif kind is inspect.Parameter.KEYWORD_ONLY:
                kwargs[name] = self.kwargs.get(name)
        return tuple(args), kwargs

    def __eq__(self, other) -> bool:
        """Called for self == other.
        
        This method will return ``True`` if any of the following cases is true,
        and will return ``False`` otherwise:

        - `other` is an instance of this class or a subclass, and all its
          arguments equal the corresponding arguments of this instance.
        - this instance has a single argument that is equal to `other`.
        """
        if isinstance(other, Result):
            return self.args == other.args and self.kwargs == other.kwargs
        if len(self.args) == 1 and not self.kwargs:
            return self.args[0] == other
        return False

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


class Rule:
    """A correspondence between operand types and affected parameters."""

    def __init__(
        self,
        __types: Types,
        *parameters: str,
    ) -> None:
        self._types = list(iterables.whole(__types))
        self._ntypes = len(self._types)
        self.parameters = unique(parameters)
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

    def get(self, key: Types, default: Rule=None):
        """Like ``~typing.Mapping.get``, with a modified default value."""
        return super().get(key, default or Rule(key))


class Operation:
    """A general operation context."""

    def __init__(
        self,
        method: typing.Callable[..., T],
        rules: Rules=None,
        result: typing.Union[T, typing.Type[T]]=None,
        **kwargs
    ) -> None:
        self.method = method
        self.rules = rules
        self.result = result
        self.kwargs = kwargs

    def rule(self, *args):
        """Get the rule, if any, for the the given type(s)."""
        types = (type(arg) for arg in args)
        return self.rules.get(types)

    def supports(self, *args):
        """True if this operation supports the given arguments."""
        return bool(self.rule(*args))

    # Developmental idea. Not in use.
    def evaluate(self, name: str, *args, reference=None):
        """Evaluating operands within the context of this operation."""
        rule = self.rule(*args)
        return (
            self.method(
                *[utilities.getattrval(arg, name) for arg in args],
                **self.kwargs
            ) if name in rule.parameters
            else utilities.getattrval(reference, name)
        )

    def format(self, *args, **kwargs) -> T:
        """Convert the result of an operation into the appropriate object.
        
        If `self.result` is a ``type``, this method will return a new instance
        of that type, initialized with the given arguments. If `self.result` is
        an instance of some type, this method will return the instance after
        updating it based on the given arguments.

        Parameters
        ----------
        *args
            The positional arguments used to create the new object.

        **kwargs
            The keyword arguments used to create the new object.

        Returns
        -------
        Any
        """
        reference = Object(self.result)
        if not kwargs and len(args) == 1 and args[0] is NotImplemented:
            return args[0]
        result = Result(*args, **kwargs)
        pos, kwd = result.get_values(reference.parameters)
        if isinstance(self.result, type):
            return self.result(*pos, **kwd)
        for name in reference.parameters:
            value = arg_or_kwarg(pos, kwd, name)
            utilities.setattrval(self.result, name, value)
        return self.result


class OperandTypeError(Exception):
    """These operands are incompatible."""


class Operands(collections.abc.Sequence, iterables.ReprStrMixin):
    """One or more algebraic operands."""

    def __new__(cls, *operands, **kwargs):
        """Prevent empty instance."""
        if not operands:
            raise TypeError(
                f"{cls.__qualname__} requires at least one operand"
            ) from None
        return super().__new__(cls)

    def __init__(
        self,
        *operands: typing.Any,
        reference=None,
    ) -> None:
        self._operands = [Object(operand) for operand in operands]
        self.reference = Object(reference or operands[0])
        """The reference operand."""
        self._types = None
        self._args = None

    @property
    def types(self):
        """The operand(s) object type(s)."""
        if self._types is None:
            self._types = tuple(type(arg) for arg in self.args)
        return self._types

    @property
    def args(self):
        """The equivalent argument(s)."""
        if self._args is None:
            args = (i._object for i in self._operands)
            self._args = tuple(args)
        return self._args

    def __getitem__(self, __i):
        """Access operands by index."""
        index = int(__i)
        if index < 0:
            index += len(self)
        return self._operands[index]

    def __len__(self) -> int:
        """The number of operands. Called for len(self)."""
        return len(self._operands)

    def apply(self, operation: Operation):
        """Evaluate these operands under the given operation."""
        if not operation.supports(self.types):
            return NotImplemented
        rule = operation.rule(self.types)
        if not self.consistent(rule):
            raise OperandTypeError from None
        if (
            not self.reference.parameters
            or rule is None # will this ever happen?
            or operation.result is None
        ): return operation.method(*self.args, **operation.kwargs)
        a = [
            self._evaluate(name, operation)
            for name in self.reference.positional
        ]
        k = {
            name: self._evaluate(name, operation)
            for name in self.reference.keyword
        }
        return operation.format(Result(*a, **k))

    def _evaluate(self, name: str, operation: Operation):
        """Internal method for evaluating operands."""
        rule = operation.rule(*self.args)
        return (
            operation.method(
                *[utilities.getattrval(arg, name) for arg in self.args],
                **operation.kwargs
            ) if name in rule.parameters
            else utilities.getattrval(self.reference, name)
        )

    def agree(self, rule: Rule):
        """True if all operands inter-operate under `rule`."""
        return all(self.consistent(i, rule=rule) for i in self)

    def consistent(self, operand: Object, rule: Rule):
        """True if `operand` inter-operates with the reference operand."""
        names = set(self.reference.parameters) - set(rule.parameters)
        return all(
            hasattr(operand, name)
            and getattr(operand, name) == getattr(self.reference, name)
            for name in names
        )

    def __str__(self) -> str:
        return ', '.join(str(i) for i in self)


class Context:
    """The context of an algebraic operation."""

    def __init__(
        self,
        rules: Rules,
        result: typing.Union[T, typing.Type[T]]=None,
        **kwargs
    ) -> None:
        self.rules = rules
        """The operand-update rules."""
        self.result = result
        """The type or instance in which to store the result."""
        self.kwargs = kwargs
        """Keyword arguments to pass to the operator method."""

    def rule(self, *args):
        """Get the rule, if any, for the the given type(s)."""
        types = (type(arg) for arg in args)
        return self.rules.get(types)

    def supports(self, *args):
        """True if this operation supports the given arguments."""
        return bool(self.rule(*args))


class Operator:
    """An algebraic operator."""

    def __init__(
        self,
        method: typing.Callable[..., T],
        rules: Rules,
    ) -> None:
        self.method = method
        """This operator's evaluation method."""
        self.rules = rules
        """The type-signature rules for this operator."""

    def evaluate(self, operands: Operands, out=None, **kwargs):
        """Call the operator method on these arguments."""
        reference = operands.reference
        if out is None or not reference.parameters:
            return self.method(*operands.args, **kwargs)
        rule = self.rule(*operands.args)
        if not all(operands.agree(rule)):
            raise OperandTypeError(*operands.args) from None
        a = [
            self.compute(name, **kwargs)
            for name in reference.positional
        ]
        k = {
            name: self.compute(name, **kwargs)
            for name in reference.keyword
        }

    def compute(self, __name: str, **kwargs):
        """Compute the value of the named attribute under this operation."""
        # This is copied from an old class
        # - `args` is `Operands.args`
        # - `parameters` is `Rule.parameters`
        # - `reference` is `Operands.reference`
        return (
            self.method(
                *[utilities.getattrval(arg, __name) for arg in self.args],
                **kwargs
            ) if __name in self.parameters
            else utilities.getattrval(self.reference, __name)
        )

    def rule(self, *args):
        """Get the rule, if any, for the the given operands."""
        types = (type(arg) for arg in args)
        return self.rules.get(types)

    def supports(self, *args):
        """True if this operation supports the given arguments."""
        return bool(self.rule(*args))


class Operation:
    """An algebraic operation."""

    def __new__(cls, operator: Operator, operands: Operands):
        """Prevent operations with incompatible operands."""
        rule = operator.rule(*operands.args)
        if not all(operands.agree(rule)):
            raise OperandTypeError(*operands.args) from None
        return super().__new__(cls)

    def __init__(
        self,
        operator: Operator,
        operands: Operands,
        returned=None,
    ) -> None:
        self.operator = operator
        self.operands = operands
        self.returned = returned
        self._args = None
        self._reference = None
        self._parameters = None

    @property
    def reference(self):
        """The reference operand."""
        if self._reference is None:
            self._reference = self.operands.reference
        return self._reference

    @property
    def args(self):
        """The required arguments in this operation."""
        if self._args is None:
            self._args = self.operands.args
        return self._args

    @property
    def parameters(self):
        """The names of updatable attributes."""
        if self._parameters is None:
            rule = self.operator.rule(*self.args)
            self._parameters = rule.parameters
        return self._parameters

    def compute(self, __name: str, **kwargs):
        """Compute the value of the named attribute under this operation."""
        return (
            self.operator.evaluate(
                *[utilities.getattrval(arg, __name) for arg in self.args],
                **kwargs
            ) if __name in self.parameters
            else utilities.getattrval(self.reference, __name)
        )


class Implementation:
    """The default operation implementation."""
    def __init__(self, method, rules=None) -> None:
        self.operator = Operator(method, rules or Rules())

    def __call__(self, *args, **kwargs):
        """Apply this operation to the given arguments."""
        return self._compute(*args, **kwargs)

    def _compute(self, *args, reference=None, **kwargs):
        """Compute the result of the operation."""
        operands = Operands(*args, reference=reference)
        if not self.operator.supports(operands):
            return NotImplemented
        return self.operator.evaluate(operands, **kwargs)


class Cast(Implementation):
    """An implementation of a type-casting operator."""
    def __call__(self, a):
        """Convert `a` to the appopriate type."""
        return super().__call__(a)


class Unary(Implementation):
    """An implementation of a unary arithmetic operator."""
    def __call__(self, a, **kwargs):
        """Operate on `a` and return an object of the same type."""
        return super().__call__(a, out=type(a), **kwargs)


class Comparison(Implementation):
    """An implementation of a binary comparison operator."""
    def __call__(self, a, b):
        """Compare `a` to `b` and return the unmodified result."""
        return super().__call__(a, b)


class Numeric(Implementation):
    """An implementation of a forward binary numeric operator."""
    def __call__(self, a, b, **kwargs):
        """Operate on `a` and `b` and return an object of `a`'s type."""
        return super().__call__(a, b, reference=a, out=type(a), **kwargs)


class Operation:
    """"""

    def __init__(
        self,
        operator: Operator,
        operands: Operands,
        result: typing.Union[T, typing.Type[T]]=None,
    ) -> None:
        self.operator = operator
        self.operands = operands
        self.result = result

    def evaluate(self, *args, **kwargs):
        """Apply this operation to the given arguments."""


class Implementation:
    """"""

    def __init__(
        self,
        method: typing.Callable[..., T],
        rules: Rules,
    ) -> None:
        self.operator = Operator(method, rules)
        self.reference = None
        self.result = None

    def __call__(self, *args, **kwargs):
        """"""
        operands = Operands(*args, reference=self.reference)
        if not self.operator.supports(operands):
            return NotImplemented
        operation = Operation(self.operator, operands, result=self.result)
        return operation.evaluate(*args, **kwargs)


class Cast(Implementation):
    """"""

    def __call__(self, a, /):
        return super().__call__(a)


class Unary(Implementation):
    """"""

    def __call__(self, a, /, **kwargs):
        self.reference = a
        self.result = type(a)
        return super().__call__(a, **kwargs)


class Comparison(Implementation):
    """"""

    def __call__(self, a, b, /):
        return super().__call__(a, b)


class Numeric(Implementation):
    """"""

    def __call__(self, a, b, /, **kwargs):
        self.reference = a
        self.result = type(a)
        return super().__call__(a, b, **kwargs)


IType = typing.TypeVar('IType', bound=Implementation)


class Application(typing.Generic[IType]):
    """The general application of an operation."""

    def __init__(
        self,
        __category: typing.Type[IType],
        rules=None,
    ) -> None:
        self._implement = __category
        self.rules = rules or Rules()

    def implement(self, method):
        operator = self._implement(method, self.rules)
        operator.__name__ = f"__{method.__name__}__"
        operator.__doc__ = method.__doc__
        return operator



# API:
# - user provides an operation name or alias
# - interface returns the appropriate operation context, which may be the
#   default context
# - user modifies operand rules, if necessary
# - user implements the operator by providing a method
# - the operator checks input arguments against operand rules; if there is no
#   rule for the given argument types, the operator returns `NotImplemented`
# - the default context simply passes all arguments to the given method
#
# NOTES:
# - I would really like non-default operators to have a type signature more
#   specific than `(*args, **kwargs) -> Any`
# - does the interface need to know about the type of object or can that be
#   optional?


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
    def find(self, __name: str) -> Operation:
        """Get an appropriate implementation context for the named operator."""
        if operation := self._implementations.get(__name):
            return operation
        rules = Rules(self.target)
        raise NotImplementedError


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


