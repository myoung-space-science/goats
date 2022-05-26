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


# Bringing this back as an alternative casting to `Object`.
def get_parameters(__type: type):
    """Determine the initialization parameters of this type, if possible."""
    return (
        {} if __type.__module__ == 'builtins'
        else inspect.signature(__type).parameters
    )


# NOTE: This class relies heavily on the ability to get an accurate signature
# from the underlying object. However, some classes define their `__init__`
# method in terms of variable positional and keyword arguments, which breaks
# that functionality. An alternative could be to define a function that attempts
# to parse parameters and values, and either returns a named tuple with the
# results or raises a `TypeError` if it encounters variable arguments. See
# `get_parameters`, above.
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
        self._parameters = parameters
        self.implemented = True

    def ignore(self, *parameters: str):
        """Ignore these parameters when updating operands."""
        self._parameters = set(self.parameters) - set(parameters)
        return self

    @property
    def parameters(self):
        """The parameters that this rule affects."""
        return unique(self._parameters)

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

    def __eq__(self, __o) -> bool:
        """Called for self == other.
        
        This method returns ``True`` iff each type in the other object is
        strictly equal to the corresponding type in this object under
        element-wise comparison.
        """
        return self._compare(__o, standard.eq)

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
        value = utilities.getattrval
        return all(
            value(obj, name) == value(reference, name)
            for name in names for obj in self[1:]
        )

    def get(self, name: str):
        """Get operand values for the named attribute."""
        return [utilities.getattrval(i, name) for i in self]

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


class NTypesError(Exception):
    """Inconsistent number of types in a new rule."""


class _RulesType(
    typing.Mapping[Types, Rule],
    collections.abc.Mapping,
    iterables.ReprStrMixin,
): ...


class Rules(_RulesType):
    """A class for managing operand-update rules."""

    def __init__(
        self,
        *default: str,
        rules: typing.Iterable[Rule]=None,
    ) -> None:
        """Initialize this instance.
        
        Parameters
        ----------
        *default : str
            Zero or more names of attributes for each rule to update unless
            explicity registered otherwise.

        rules : iterable of `~operations.Rule`, optional
            Existing rules with which to initialize the internal collection.

        Notes
        -----
        Providing the names of all possibly updatable attributes via `*default`
        protects against bugs that arise from naive use of
        ``inspect.signature``. For example, a class's ``__init__`` method may
        accept generic `*args` and `**kwargs`, which it then parses into
        specific attributes. In that case, ``inspect.signature`` will not
        discover the correct names of attributes necessary to initialize a new
        instance after applying a given rule.
        """
        self.default = list(default)
        """The default parameters to update for each rule."""
        self.ntypes = None
        """The number of argument types in these rules."""
        self._rulemap = None
        for rule in rules or []:
            self.register(rule.types, *rule.parameters)

    def constrain(self, types: Types, *parameters: str, mode: str='include'):
        """Restrict the parameters of an existing rule.
        
        Parameters
        ----------
        types : type or tuple of types
            The argument type(s) in the target rule.

        *parameters : str
            Zero or more parameters to include or exclude, depending on `mode`.

        mode : {'include', 'exclude'}
            How to handle the given parameters. If `mode == 'include'` (the
            default), this method will restrict the target rule's parameters to
            `parameters`. If `mode == 'exclude'`, this method will remove
            `parameters` from the target rule's parameters.
        """
        key = tuple(iterables.whole(types))
        if key not in self.mapping:
            raise KeyError(f"Rule for {types!r} does not exist") from None
        if mode == 'include':
            new = parameters
        elif mode == 'exclude':
            rule = self[key]
            new = set(rule.parameters) - set(parameters)
        else:
            raise ValueError(f"Unknown mode {mode!r}")
        self.mapping[key] = self._resolve(*new)

    def register(self, types: Types, *parameters: typing.Optional[str]):
        """Add a rule to the collection."""
        key = tuple(iterables.whole(types))
        ntypes = len(key)
        self._check_ntypes(ntypes)
        if key not in self.mapping:
            self.mapping[key] = self._resolve(*parameters)
            return self
        raise KeyError(f"{types!r} is already in the collection") from None

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

    def _check_ntypes(self, ntypes: int):
        """Helper for enforcing consistency in number of types.
        
        If the internal attribute that keeps track of how many arguments are
        allowed for all rules is `None`, this method will set it on the first
        time through. After that, it will raise an exception if the user tries
        to register a rule with a different number of types.
        """
        if self.ntypes is None:
            self.ntypes = ntypes
            return
        if ntypes != self.ntypes:
            raise NTypesError(
                f"Can't add a length-{ntypes} rule to a collection"
                f" of length-{self.ntypes} rules."
            ) from None

    def __len__(self) -> int:
        """Returns the number of rules. Called for len(self)."""
        return len(self.mapping)

    def __iter__(self) -> typing.Iterator[Rule]:
        """Iterate over rules. Called for iter(self)."""
        for types in self.mapping:
            yield Rule(types, *self.mapping[types])

    def __getitem__(self, __k: Types):
        """Retrieve the operand-update rule for `types`."""
        types = tuple(iterables.whole(__k))
        if types in self.mapping:
            return self._from(types)
        for t in self.mapping:
            if all(issubclass(i, j) for i, j in zip(types, t)):
                return self._from(t)
        raise KeyError(f"No rule for operand type(s) {__k!r}") from None

    def _from(self, types: Types):
        """Build a rule from the given types."""
        parameters = self.mapping.get(types, self.default.copy())
        return Rule(types, *parameters)

    def get(self, __types: Types, default: Rule=None):
        """Get the rule for these types, or a default rule.
        
        This method behaves like ``~typing.Mapping.get`` with a modified default
        value: Instead of returning ``None`` when key-based look-up fails (i.e.,
        there is no rule for the given types), this method returns a rule with
        no constraints for the given types.
        """
        return super().get(__types, default or Rule(__types))

    def copy(self):
        """Create a shallow copy of this instance."""
        return Rules(*self.default, rules=iter(self))

    def __eq__(self, __o) -> bool:
        """True iff two instances have the same default parameters and rules."""
        return (
            isinstance(__o, Rules)
            and __o.default == self.default
            and __o.mapping == self.mapping
        )

    def __str__(self) -> str:
        return ', '.join(str(rule) for rule in self)


class Operands(Objects):
    """Objects that are part of an operation."""

    def __init__(self, *objects, reference: T=None) -> None:
        super().__init__(*objects)
        self.reference = Object(reference or self[0])


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
        # BUG: `Object.parameters` finds variable args/kwargs instead of
        # specific required arguments. Should use default parameters from
        # `Rules`, which means moving this method.
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


Caller = typing.TypeVar('Caller', bound=typing.Callable)
Caller = typing.Callable[..., T]


class Operator:
    """Base class for operator implementations."""

    def __init__(
        self,
        method: Caller,
        rules: Rules,
    ) -> None:
        self.method = method
        self.rules = rules

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
        if not self.consistent(context, rule):
            errmsg = self._operand_errmsg(rule, context.operands)
            raise OperandTypeError(errmsg) from None
        return context.apply(self.method, rule, **kwargs)

    def get_rule(self, *operands):
        """Get the operation rule for these operands' types."""
        types = [type(operand) for operand in operands]
        return self.rules.get(types)

    def consistent(self, context: Context, rule: Rule):
        """True if `context` is consistent with `rule`
        
        This method determines if all the attributes that are common to the
        current operands, and that are not included in the given rule, have the
        same value. These are the attributes that the corresponding operation
        will ignore; therefore, different values for the same attribute
        represent an ambiguity.
        """
        operands = context.operands
        fixed = tuple(set(self.rules.default) - set(rule.parameters))
        return operands.agree(*fixed)

    def _operand_errmsg(self, rule: Rule, operands: Operands):
        """Build an error message based on `rule` and `operands`."""
        method_string = repr(self.method.__qualname__)
        types = operands.types
        types_string = (
            types[0].__qualname__ if len(types) == 1
            else f"({', '.join(t.__qualname__ for t in types)})"
        )
        fixed = tuple(set(self.rules.default) - set(rule.parameters))
        attrs_string = (
            repr(fixed[0]) if len(fixed) == 1
            else f"{fixed[0]!r} and {fixed[1]!r}" if len(fixed) == 2
            else f"{', '.join(fixed[:-1])} and {fixed[-1]}"
        )
        return (
            f"Can't apply operator {method_string} to {types_string}"
            f" with different values of {attrs_string}"
        )


class Operation:
    """A general arithmetic operation."""

    def __init__(self, rules: Rules=None) -> None:
        self.rules = Rules() if rules is None else rules

    def implement(self, __caller: Caller) -> typing.Any:
        """Implement this operation with the given callable object."""
        operator = Operator(__caller, self.rules)
        def operate(*args, **kwargs):
            """Apply this operation to the given arguments."""
            return operator.compute(*args, **kwargs)
        return operate


A = typing.TypeVar('A')
B = typing.TypeVar('B')


OType = typing.TypeVar('OType', bound=Operation)


class Cast(Operation):
    """An implementation of a type-casting operation."""

    def implement(self, __caller: typing.Type[T]):
        """Implement this operation with the given callable object."""
        operator = Operator(__caller, self.rules)
        def operate(a: A) -> T:
            """Convert `a` to the appropriate type, if possible."""
            return operator.compute(a)
        return operate


class Unary(Operation):
    """An implementation of a unary arithmetic operation."""

    def implement(self, __caller: Caller):
        """Implement this operation with the given callable object."""
        operator = Operator(__caller, self.rules)
        def operate(a: A, /, **kwargs) -> A:
            """Apply this operation to `a`."""
            return operator.compute(a, reference=a, target=type(a), **kwargs)
        return operate


class Comparison(Operation):
    """An implementation of a binary comparison operation."""

    def implement(self, __caller: Caller):
        """Implement this operation with the given callable object."""
        operator = Operator(__caller, self.rules)
        def operate(a: A, b: B, /) -> T:
            """Compare `a` to `b`."""
            return operator.compute(a, b)
        return operate


class Numeric(Operation):
    """An implementation of a binary numeric operation."""

    def implement(self, __caller: Caller, mode: str='forward'):
        """Implement this operation with the given callable object."""
        operator = Operator(__caller, self.rules)
        def forward(a: A, b: B, /, **kwargs) -> A:
            """Apply this operation to `a` and `b`."""
            return operator.compute(a, b, reference=a, target=type(a), **kwargs)
        def reverse(a: A, b: B, /, **kwargs) -> B:
            """Apply this operation to `a` and `b` with reflected operands."""
            return operator.compute(a, b, reference=b, target=type(b), **kwargs)
        def inplace(a: A, b: B, /, **kwargs) -> A:
            """Apply this operation to `a` and `b` in-place."""
            return operator.compute(a, b, reference=a, target=a, **kwargs)
        if mode == 'forward':
            return forward
        if mode == 'reverse':
            return reverse
        if mode == 'inplace':
            return inplace
        raise ValueError(f"Unknown implementation mode {mode!r}") from None


_categories = {
    'cast': {
        'operations': ['int', 'float', 'complex'],
        'implementation': Cast,
    },
    'unary': {
        'operations': ['abs', 'pos', 'neg', 'trunc', 'round', 'ceil', 'floor'],
        'implementation': Unary,
    },
    'comparison': {
        'operations': ['lt', 'le', 'gt', 'ge', 'eq', 'ne'],
        'implementation': Comparison,
    },
    'numeric': {
        'operations': ['add', 'sub', 'mul', 'truediv', 'floordiv', 'pow'],
        'implementation': Numeric,
    },
}


class Interface:
    """Top-level interface to arithmetic operations."""

    def __init__(self, __type: type, *parameters) -> None:
        self._type = __type
        self.parameters = parameters
        """The names of all updatable attributes."""
        self._implementations = None
        self.cast = self._create(Cast, nargs=1)
        """An interface to type-casting operations."""
        self.unary = self._create(Unary, nargs=1)
        """An interface to a unary arithmetic operations."""
        self.comparison = self._create(Comparison, nargs=2)
        """An interface to a binary comparison operations."""
        self.numeric = self._create(Numeric, nargs=2)
        """An interface to a binary arithmetic operations."""

    @property
    def implementations(self) -> typing.MutableMapping[str, OType]:
        """All available implementation contexts."""
        if self._implementations is None:
            mapping = aliased.MutableMapping(_categories, 'operations')
            self._implementations = mapping.squeeze()
        return self._implementations

    def create(self, __name: str):
        """Create an arbitrary operation interface.
        
        Parameters
        ----------
        __name : string
            The name of an operation or operation category to implement.

        Returns
        -------
        Operation
        """
        return self._create(self.implementations.get(__name, Operation))

    def _create(
        self,
        __category: typing.Type[OType],
        nargs: int=None,
    ) -> OType:
        """Internal helper for creating operation interfaces."""
        rules = Rules(*self.parameters)
        if nargs is not None:
            rules.register([self._type] * nargs)
        return __category(rules=rules)

_operators = [
    ('add', '__add__', 'addition'),
    ('sub', '__sub__', 'subtraction'),
]



# Not in use but could serve as a template for simplifying `Operator.apply`
class Context:
    """The implementation context for an operation."""

    def __init__(
        self,
        *args,
        defaults: typing.Dict[str, typing.Any]=None,
        target: typing.Union[T, typing.Type[T]]=None,
    ) -> None:
        self.operands = Operands(*args)
        """The active and reference operands."""
        self.defaults = defaults or {}
        self.target = target
        """The object or type of object representing the operation result."""

    def apply(self, method: typing.Callable, rule: Rule, **kwargs):
        """Apply a method and rule to this context."""
        values = self._compute(method, rule, **kwargs)
        if self.target is None:
            return values[0] if len(values) == 1 else values
        if isinstance(self.target, type):
            return self.target(*values)
        zipped = zip(self.defaults, values)
        for name, value in zipped:
            utilities.setattrval(self.target, name, value)
        return self.target

    def _compute(self, method, rule: Rule, **kwargs):
        """Compute values based on parameters."""
        if not self.defaults:
            return [
                method(*self.operands.get(name), **kwargs)
                for name in rule.parameters
            ]
        return [
            method(*self.operands.get(name), **kwargs)
            if name in rule else value
            for name, value in self.defaults.items()
        ]


class Operator:
    """The default operator implementation."""

    def __init__(
        self,
        rules: Rules=None,
    ) -> None:
        self.rules = Rules() if rules is None else rules

    def implement(self, __callable: typing.Callable[..., T]):
        """Implement this operation with the given callable object."""
        def operator(*args, **kwargs) -> T:
            return self.apply(__callable, *args, **kwargs)
        return operator

    # Maybe `apply(self, operation: Operation, *args, **kwargs)`, where
    # `Operation` carries the callable, reference, and target objects, computes
    # default values from the reference object when given an iterable of
    # parameters, and possibly has an `apply` method that this `apply` method
    # calls at the end.
    def apply(self, method, *args, reference=None, target=None, **kwargs):
        """"""
        rule = self.get_rule(*args)
        if not rule.implemented:
            return NotImplemented
        if not rule.parameters:
            # We don't know which arguments to operate on, so we hand execution
            # over to the given objects, in case they implement this method in
            # their class definitions.
            return method(*args, **kwargs)
        operands = Objects(*args)
        fixed = tuple(set(self.rules.default) - set(rule.parameters))
        if not operands.agree(*fixed):
            errmsg = self._operand_errmsg(rule, operands)
            raise OperandTypeError(errmsg) from None
        defaults = {
            name: utilities.getattrval(reference, name)
            for name in self.rules.default
        }
        values = self._compute(method, operands, rule, defaults=defaults, **kwargs)
        if target is None:
            return values[0] if len(values) == 1 else values
        if isinstance(target, type):
            return target(*values)
        zipped = zip(defaults, values)
        for name, value in zipped:
            utilities.setattrval(target, name, value)
        return target

    def get_rule(self, *operands):
        """Get the operation rule for these operands' types."""
        types = [type(operand) for operand in operands]
        return self.rules.get(types)

    def _operand_errmsg(self, method, rule: Rule, operands: Operands):
        """Build an error message based on `rule` and `operands`."""
        method_string = repr(method.__qualname__)
        types = operands.types
        types_string = (
            types[0].__qualname__ if len(types) == 1
            else f"({', '.join(t.__qualname__ for t in types)})"
        )
        fixed = tuple(set(self.rules.default) - set(rule.parameters))
        attrs_string = (
            repr(fixed[0]) if len(fixed) == 1
            else f"{fixed[0]!r} and {fixed[1]!r}" if len(fixed) == 2
            else f"{', '.join(fixed[:-1])} and {fixed[-1]}"
        )
        return (
            f"Can't apply operator {method_string} to {types_string}"
            f" with different values of {attrs_string}"
        )

    def _compute(
        self,
        method,
        operands: Objects,
        rule: Rule,
        defaults: dict=None,
        **kwargs
    ):
        """Compute values based on parameters."""
        if not defaults:
            return [
                method(*operands.get(name), **kwargs)
                for name in rule.parameters
            ]
        return [
            method(*operands.get(name), **kwargs)
            if name in rule else value
            for name, value in defaults.items()
        ]


class Cast(Operator):
    """An implementation of a type-casting operator."""

    def implement(self, __callable: typing.Type[T]):
        """Implement this operation with the given callable object."""
        def operator(a: A) -> T:
            """Convert `a` to the appropriate type, if possible."""
            return self.apply(__callable, a)
        return operator


class Unary(Operator):
    """An implementation of a unary arithmetic operator."""

    CType = typing.TypeVar('CType', bound=typing.Callable)
    CType = typing.Callable[[A], A]

    def implement(self, __callable: CType):
        """Apply this operation to `a`."""
        def operator(a: A, /, **kwargs) -> A:
            return self.apply(__callable, a, reference=a, target=type(a), **kwargs)
        return operator


class Comparison(Operator):
    """An implementation of a binary comparison operator."""

    CType = typing.TypeVar('CType', bound=typing.Callable)
    CType = typing.Callable[[A, B], T]

    def implement(self, __callable: CType):
        """Implement this operation with the given callable object."""
        def operator(a: A, b: B, /) -> T:
            """Compare `a` to `b`."""
            return self.apply(__callable, a, b)
        return operator


class Numeric(Operator):
    """An implementation of a binary numeric operator."""

    CType = typing.TypeVar('CType', bound=typing.Callable)
    CType = typing.Callable[[A, B], T]

    def implement(self, __callable: CType, mode: str='forward'):
        """Implement this operation with the given callable object."""
        def forward(a: A, b: B, /, **kwargs) -> A:
            """Apply this operation to `a` and `b`."""
            return self.apply(__callable, a, b, reference=a, target=type(a), **kwargs)
        def reverse(a: A, b: B, /, **kwargs) -> B:
            """Apply this operation to `a` and `b` with reflected operands."""
            return self.apply(__callable, a, b, reference=b, target=type(b), **kwargs)
        def inplace(a: A, b: B, /, **kwargs) -> A:
            """Apply this operation to `a` and `b` in-place."""
            return self.apply(__callable, a, b, reference=a, target=a, **kwargs)
        if mode == 'forward':
            return forward
        if mode == 'reverse':
            return reverse
        if mode == 'inplace':
            return inplace
        raise ValueError(f"Unknown implementation mode {mode!r}") from None


OT = typing.TypeVar('OT', bound=Operator)


class Factory(typing.Generic[OT]):
    """A factory for creating operations of a given type."""

    def __init__(
        self,
        __class: typing.Type[OT],
        rules: Rules=None,
    ) -> None:
        self._class = __class
        self.rules = Rules() if rules is None else rules

    @property
    def operation(self):
        """Create an operation with this category's rules."""
        return self._class(self.rules)


# Notes:
# - Design Interface for algorithmic use (e.g., `datatypes.Array`).
# - Design Factory for explicit use (e.g., `measurable.Quantity`).


unary = Factory(Unary)
_abs = unary.operation
func = _abs.implement(standard.abs)
r = func(3.4)

numeric = Factory(Numeric)
_add = numeric.operation
func = _add.implement(standard.add)
r = func(2, 5.0)

