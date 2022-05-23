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


class Object(typing.Generic[T], iterables.ReprStrMixin):
    """A wrapper around a single object."""

    def __init__(self, __object: typing.Union[T, 'Object']) -> None:
        self._object = self._init_object(__object)
        self._type = type(self._object)
        self.isbuiltin = self._type.__module__ == 'builtins'
        self._values = None
        self._parameters = None
        self._positional = None
        self._keyword = None

    def _init_object(self, arg) -> T:
        """Internal initialization helper."""
        if isinstance(arg, type(self)):
            return arg._object
        return arg

    @property
    def values(self):
        """All values used to initialize this object."""
        if self._values is None:
            self._values = {
                name: utilities.getattrval(self._object, name)
                for name in self.parameters
            }
        return self._values

    @property
    def parameters(self):
        """All parameters used to initialize this object."""
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
        """The names of positional arguments to this object."""
        if self._positional is None:
            names = [
                name for name, parameter in self.parameters.items()
                if parameter.kind in self._postypes
            ]
            self._positional = tuple(names)
        return self._positional

    @property
    def keyword(self):
        """The names of keyword arguments to this object."""
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


class Rule(iterables.ReprStrMixin):
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
        self.implemented = True

    @property
    def types(self):
        """The operand types that define this rule."""
        return tuple(self._types)

    def __len__(self) -> int:
        """The number of parameters affected by this rule."""
        return len(self.parameters)

    def __contains__(self, __x: typing.Union[type, str]) -> bool:
        """True if `__x` is part of this rule.
        
        Parameters
        ----------
        __x : type or string
            If `__x` is a type, this method will report whether or not `__x` is
            a type in this rule. If `__x` is a string, this method with report
            whether or not `__x` is a parameter affected by this rule.
        """
        return (
            __x in self.types if isinstance(__x, type)
            else __x in self.parameters if isinstance(__x, str)
            else False
        )

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
        """True if this rule is implemented."""
        return self.implemented

    # Consider letting this return `Suppressed` (see above).
    @property
    def suppress(self):
        """Suppress this operand rule.
        
        Invoking this property will signal to operation implementations that
        they should not implement the operator for these operand types.
        """
        self._parameters = None
        self.implemented = False
        return self

    def __str__(self) -> str:
        names = [t.__qualname__ for t in self.types]
        types = names[0] if len(names) == 1 else tuple(names)
        return f"{types}: {self.parameters}"


class Objects(collections.abc.Sequence, iterables.ReprStrMixin):
    """A sequence of `~operations.Object` instances."""

    def __init__(self, *objects) -> None:
        self._objects = [Object(i) for i in objects]
        self._types = None

    def agree(self, *names: str):
        """True if these objects have equal attributes.
        
        This method determines if all the named attributes have the same value.
        The result is always ``True`` for an instance with length 1, since a
        single object trivially agrees with itself.
        """
        if len(self) == 1:
            return True
        reference = self[0]
        return all(
            getattr(obj, name) == getattr(reference, name)
            for name in names for obj in self
        )

    @property
    def types(self):
        """The type of each object."""
        if self._types is None:
            self._types = tuple(i._type for i in self._objects)
        return self._types

    def __getitem__(self, __i):
        """Access objects by index."""
        if isinstance(__i, typing.SupportsIndex):
            return Object(self._objects[__i])
        return Objects(*self._objects[__i])

    def __len__(self) -> int:
        """The number of objects. Called for len(self)."""
        return len(self._objects)

    def __iter__(self) -> typing.Iterator[Object]:
        """Iterate over objects. Called for iter(self)."""
        yield from self._objects

    def __eq__(self, other):
        """Called for self == other."""
        if not len(self) == len(other):
            return False
        return all(i in other for i in self)

    def __str__(self) -> str:
        return ', '.join(str(i) for i in self)


class _RulesType(
    typing.Mapping[Types, Rule],
    collections.abc.Mapping,
    iterables.ReprStrMixin,
): ...


class Rules(_RulesType):
    """A class for managing operand-update rules."""

    def __init__(self, *default: str, nargs: int=None) -> None:
        self.default = list(default)
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
            yield Rule(types, *self.mapping[types])

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

    def get(self, __types: Types, default: Rule=None):
        """Get the rule for these types or a default rule.
        
        This method behaves like ``~typing.Mapping.get`` with a modified default
        value: Instead of returning ``None`` when key-based look-up fails (i.e.,
        there is no rule for the given types), this method returns a rule with
        no constraints for the given types.
        """
        return super().get(__types, default or Rule(__types))

    def __str__(self) -> str:
        return ', '.join(str(rule) for rule in self)


class Arguments:
    """Positional and keyword inputs to an object."""

    def __init__(self, reference: Object, *positional, **keyword) -> None:
        self.reference = reference
        self.positional = positional
        self.keyword = keyword

    def __getitem__(self, __x):
        """Get positional arguments by index or keyword arguments by name."""
        if isinstance(__x, (typing.SupportsIndex, slice)):
            return self.positional[__x]
        if isinstance(__x, str):
            return self.keyword[__x]
        raise TypeError(__x)

    def format(self, target: typing.Union[T, typing.Type[T]]) -> T:
        """Convert the result of an operation into the appropriate object.
        
        Parameters
        ----------
        target
            If `target` is a ``type``, this method will return a new instance of
            that type, initialized with the given arguments. If `target` is an
            instance of some type, this method will return the instance after
            updating it based on the given arguments.
        """
        args, kwargs = self.get_values(list(self.positional), self.keyword)
        if isinstance(target, type):
            return target(*args, **kwargs)
        for name in self.reference.parameters:
            value = arg_or_kwarg(list(args), kwargs, name)
            utilities.setattrval(target, name, value)
        return target

    def get_values(self, args: list, kwargs: dict):
        """Extract appropriate argument values.
        
        This method will attempt to build appropriate positional and keyword
        arguments from this result, based on the given object signature.
        """
        if not self.reference.parameters:
            return tuple(args), kwargs
        pos = []
        kwd = {}
        for name, parameter in self.reference.parameters.items():
            kind = parameter.kind
            if kind is inspect.Parameter.POSITIONAL_ONLY:
                pos.append(args.pop(0))
            elif kind is inspect.Parameter.POSITIONAL_OR_KEYWORD:
                pos.append(arg_or_kwarg(args, kwargs, name))
            elif kind is inspect.Parameter.KEYWORD_ONLY:
                kwd[name] = kwargs.get(name)
        return tuple(pos), kwd


def compatible(*operands, rule: Rule=None):
    """True if the operands are compatible under `rule`.
    
    This method determines if all the attributes that are common to these
    objects and that are not included in the given rule have the same value.
    These are the attributes that the corresponding operation will ignore;
    therefore, different values for the same attribute will lead to ambiguity.
    Objects are always compatible if the given rule is unconstrained.
    """
    if rule is not None and not rule.parameters:
        return True
    objects = Objects(*operands)
    sets = [set(obj.parameters) for obj in objects]
    parameters = set.intersection(*sets)
    names = parameters if rule is None else parameters - set(rule.parameters)
    return objects.agree(*names)


class Operands(Objects):
    """Objects that are part of an operation."""

    def __init__(self, *objects, reference: T=None) -> None:
        super().__init__(*objects)
        self.reference = Object(reference or self[0])

    def get(self, name: str):
        """Get operand values for the named attribute."""
        return [utilities.getattrval(i, name) for i in self]


class Context:
    """The implementation context for an operation."""

    def __init__(
        self,
        *args,
        reference: T=None,
        target: typing.Union[T, typing.Type[T]]=None,
    ) -> None:
        self.operands = Operands(*args, reference=reference)
        """The active and reference operands."""
        self.reference = Object(reference)
        """The object that provides reference parameters and values.
        
        Note that this is different from the reference operand: The `reference`
        attribute of an `Operands` instance defaults to the first argument used
        to initialize that instance if the user doesn't provide a reference
        object, whereas this class's `reference` attribute represents whatever
        object the caller used to initialize this instance.
        """
        self.target = target
        """The object or type of object representing the operation result."""

    def supports(self, rule: Rule):
        """True if the operands are compatible under `rule`."""
        return compatible(*self.operands, rule=rule)

    def apply(self, method: typing.Callable, rule: Rule, **kwargs):
        """Apply a method and rule to this context."""
        if not self.reference.parameters:
            values = [
                self.compute(name, method, rule, **kwargs)
                for name in rule.parameters
            ]
            if self.target is None:
                return values[0] if len(values) == 1 else values
            return self.format(*values)
        pos = [
            self.compute(name, method, rule, **kwargs)
            for name in self.reference.positional
        ]
        if self.target is None:
            return pos[0] if len(pos) == 1 else pos
        kwd = {
            name: self.compute(name, method, rule, **kwargs)
            for name in self.reference.keyword
        }
        return self.format(*pos, **kwd)

    def compute(self, name: str, method, rule: Rule, **kwargs):
        """Compute a value for name or get the default value."""
        return (
            method(*self.operands.get(name), **kwargs)
            if name in rule else self.reference.values.get(name)
        )

    def format(self, *args, **kwargs) -> T:
        """Convert the result of an operation into the appropriate object.
        
        If `self.target` is a ``type``, this method will return a new
        instance of that type, initialized with the given arguments. If
        `self.target` is an instance of some type, this method will return
        the instance after updating it based on the given arguments.
        """
        pos, kwd = self.get_values(list(args), kwargs)
        if isinstance(self.target, type):
            return self.target(*pos, **kwd)
        for name in self.reference.parameters:
            value = arg_or_kwarg(list(pos), kwd, name)
            utilities.setattrval(self.target, name, value)
        return self.target

    def get_values(self, args: list, kwargs: dict):
        """Extract appropriate argument values.
        
        This method will attempt to build appropriate positional and keyword
        arguments from this result, based on the given object signature.
        """
        if not self.reference.parameters:
            return tuple(args), kwargs
        pos = []
        kwd = {}
        for name, parameter in self.reference.parameters.items():
            kind = parameter.kind
            if kind is inspect.Parameter.POSITIONAL_ONLY:
                pos.append(args.pop(0))
            elif kind is inspect.Parameter.POSITIONAL_OR_KEYWORD:
                pos.append(arg_or_kwarg(args, kwargs, name))
            elif kind is inspect.Parameter.KEYWORD_ONLY:
                kwd[name] = kwargs.get(name)
        return tuple(pos), kwd


class OperandTypeError(Exception):
    """These operands are incompatible."""


class Operator(abc.ABC):
    """Abstract base class for operator implementations."""

    def __init__(
        self,
        method: typing.Callable[..., T],
        rules: Rules,
    ) -> None:
        self.method = method
        self.rules = rules

    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        """Call this operator.
        
        Concrete subclasses must implement this method. It should take arguments
        appropriate to the specific operator or operator category and return
        whatever a user could reasonably expect from the equivalent standard
        operator.
        """
        pass

    def compute(self, *args, reference=None, target=None, **kwargs):
        """Compute the result of this operation."""
        rule = self.get_rule(*args)
        if not rule.implemented:
            return NotImplemented
        if not rule.parameters:
            # We don't know which arguments to operate on, so we hand execution
            # over to the given objects, in case they implement this method in
            # their class definitions.
            return self.method(*args, **kwargs)
        context = Context(*args, reference=reference, target=target)
        if not context.supports(rule):
            errmsg = self._operand_errmsg(rule, context.operands)
            raise OperandTypeError(errmsg) from None
        return context.apply(self.method, rule, **kwargs)

    def get_rule(self, *operands):
        """Get the operation rule for these operands' types."""
        types = [type(operand) for operand in operands]
        return self.rules.get(types)

    def _operand_errmsg(self, rule: Rule, operands: Operands):
        """Build an error message based on `rule` and `operands`."""
        method_string = repr(self.method.__qualname__)
        types = operands.types
        types_string = (
            types[0].__qualname__ if len(types) == 1
            else f"({', '.join(t.__qualname__ for t in types)})"
        )
        fixed = tuple(set(operands.reference.parameters) - set(rule.parameters))
        attrs_string = (
            repr(fixed[0]) if len(fixed) == 1
            else f"{fixed[0]!r} and {fixed[1]!r}" if len(fixed) == 2
            else f"{', '.join(fixed[:-1])} and {fixed[-1]}"
        )
        return (
            f"Can't apply operator {method_string} to {types_string}"
            f" with different values of {attrs_string}"
        )


A = typing.TypeVar('A')
B = typing.TypeVar('B')


class Default(Operator):
    """The default operator implementation."""

    def __call__(self, *args, **kwargs):
        """Apply this operation's method to the given arguments."""
        return self.compute(*args, **kwargs)


class Cast(Operator):
    """An implementation of a type-casting operator."""

    def __call__(self, a: A, /):
        """Convert `a` to the appropriate type, if possible."""
        return self.compute(a)


class Unary(Operator):
    """An implementation of a unary arithmetic operator."""

    def __call__(self, a: A, /, **kwargs):
        """Apply this operator's method to `a`."""
        return self.compute(a, reference=a, target=type(a), **kwargs)


class Comparison(Operator):
    """An implementation of a binary comparison operator."""

    def __call__(self, a: A, b: B, /):
        """Compare `a` to `b`."""
        return self.compute(a, b)


class Numeric(Operator):
    """An implementation of a binary numeric operator."""

    def __call__(self, a: A, b: B, /, **kwargs):
        """Apply this operator's method to `a` and `b`."""
        return self.compute(a, b, reference=a, target=type(a), **kwargs)


OType = typing.TypeVar('OType', bound=Operator)


class Category(typing.Generic[OType]):
    """Implementation context for an operation category."""

    def __init__(
        self,
        __type: typing.Type[OType],
        nargs: int=None,
        aliases: typing.Sequence[str]=None,
    ) -> None:
        self._type = __type
        self.nargs = nargs
        self.aliases = list(aliases or [])


default = Category(
    Default,
)
unary = Category(
    Unary,
    nargs=1,
    aliases=['abs', 'pos', 'neg', 'trunc', 'round', 'ceil', 'floor'],
)
cast = Category(
    Cast,
    nargs=1,
    aliases=['int', 'float', 'complex'],
)
comparison = Category(
    Comparison,
    nargs=2,
    aliases=['lt', 'le', 'gt', 'ge', 'eq', 'ne'],
)
numeric = Category(
    Numeric,
    nargs=2,
    aliases=['add', 'sub', 'mul', 'truediv', 'floordiv', 'pow'],
)

_categories = {
    'default': default,
    'unary': unary,
    'cast': cast,
    'comparison': comparison,
    'numeric': numeric,
}


class Operation(typing.Generic[OType]):
    """A general arithmetic operation."""

    def __init__(
        self,
        __category: typing.Type[OType],
        rules: Rules=None,
    ) -> None:
        self._implement = __category
        self.rules = rules or Rules()

    def implement(self, __callable: typing.Callable[..., T]):
        return self._implement(__callable, self.rules)



class Interface:
    """Top-level interface to arithmetic operations."""

    def __init__(self, __type: type, dataname: str=None) -> None:
        self._type = __type
        self.dataname = dataname
        """The name of the data-like attribute."""
        self._implementations = None

    _IT = typing.TypeVar('_IT', bound=typing.MutableMapping)
    _IT = typing.MutableMapping[str, Category[OType]]
    @property
    def implementations(self) -> _IT:
        """All available implementation contexts."""
        if self._implementations is None:
            self._implementations = aliased.MutableMapping(_categories)
        return self._implementations

    @typing.overload
    def create(
        self,
        __name: typing.Literal['cast'],
    ) -> Operation[Cast]: ...

    @typing.overload
    def create(
        self,
        __name: typing.Literal['unary'],
    ) -> Operation[Unary]: ...

    @typing.overload
    def create(
        self,
        __name: typing.Literal['comparison'],
    ) -> Operation[Comparison]: ...

    @typing.overload
    def create(
        self,
        __name: typing.Literal['numeric'],
    ) -> Operation[Numeric]: ...

    @typing.overload
    def create(self, __name: str) -> Operation[Default]: ...

    def create(self, __name):
        """Create an interface to an operation or operation category.
        
        Parameters
        ----------
        __name : str
            The name of an operation or an operation category. If `__name` is
        """
        category = self.implementations.get(__name, default)
        nargs = category.nargs
        rules = Rules(self.dataname, nargs=nargs)
        if nargs:
            rules.register([self._type] * nargs)
        return Operation(category._type, rules=rules)


_operators = [
    ('add', '__add__', 'addition'),
    ('sub', '__sub__', 'subtraction'),
]


