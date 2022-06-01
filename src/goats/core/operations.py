import abc
import collections
import collections.abc
import contextlib
import functools
import inspect
import math
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


def isbuiltin(__object):
    """Convenience method for testing if an object is a built-in type."""
    return type(__object).__module__ == 'builtins'


def get_parameters(__object):
    """Determine the initialization parameters for an object, if possible."""
    return (
        {} if isbuiltin(__object)
        else inspect.signature(type(__object)).parameters
    )


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

    def __init__(self, *parameters: str) -> None:
        self._parameters = parameters

    def ignore(self, *parameters: str):
        """Ignore these parameters when updating operands."""
        self._parameters = set(self.parameters) - set(parameters)
        return self

    @property
    def implemented(self):
        """True if this rule's set of parameters is not ``None``."""
        return not (len(self.parameters) == 1 and self.parameters[0] is None)

    @property
    def parameters(self):
        """The parameters that this rule affects."""
        return unique(self._parameters)

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
        return __x in self.parameters

    def __eq__(self, __o) -> bool:
        """Called for self == other.
        
        This method returns ``True`` iff each type in the other object is
        strictly equal to the corresponding type in this object under
        element-wise comparison.
        """
        return (
            __o.parameters == self.parameters
            if isinstance(__o, Rule)
            else __o == self.parameters
        )

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
        return str(self.parameters)


class Operands(collections.abc.Sequence, iterables.ReprStrMixin):
    """A sequence of operands."""

    def __init__(self, *args: T, reference: T=None) -> None:
        self._args = list(args)
        self._reference = reference
        self._types = None

    @property
    def reference(self):
        """The reference object.
        
        Notes
        -----
        - The default reference object is the first argument used to initialize
          this instance.
        - The reference object is not part of the sequence of operands and
          therefore will not explicitly appear when iterating over an instance.
        """
        if self._reference is None:
            self._reference = self[0]
        return self._reference

    def agree(self, *names: str):
        """Compare values of named attribute(s) across operands.
        
        This method determines if all the named attributes have the same value
        when present in an operand. The result is trivially ``True`` for a
        single operand
        """
        if len(self) == 1:
            return True
        if not all(hasattr(self.reference, name) for name in names):
            return False
        others = self.find(*names)
        value = utilities.getattrval
        return all(
            value(target, name) == value(self.reference, name)
            for name in names for target in others
        )

    def find(self, *names: str):
        """Get all operands with the named attributes."""
        return [
            obj for obj in self
            if all(hasattr(obj, name) for name in names)
        ]

    def consistent(self, *names: str):
        """Determine if all operands have the named attributes."""
        return all(hasattr(obj, name) for obj in self for name in names)

    def get(self, name: str):
        """Get operand values for the named attribute."""
        return [utilities.getattrval(i, name) for i in self]

    @property
    def types(self):
        """The type of each object."""
        if self._types is None:
            self._types = tuple(type(i) for i in self)
        return self._types

    def __getitem__(self, __i):
        """Access operands by index."""
        if isinstance(__i, typing.SupportsIndex):
            return self._args[__i]
        return Operands(*self._args[__i])

    def __len__(self) -> int:
        """The number of operands. Called for len(self)."""
        return len(self._args)

    def __iter__(self):
        """Iterate over operands. Called for iter(self)."""
        yield from self._args

    def __eq__(self, other):
        """Called for self == other."""
        if not len(self) == len(other):
            return False
        return all(i in other for i in self)

    def __str__(self) -> str:
        return ', '.join(str(i) for i in self)


class same:
    """A callable class that enforces object consistency.

    When used to decorate a method that takes two arguments, this class will
    ensure that the arguments have equal values of a named attribute. This may
    be useful when writing binary comparison methods that are only valid for two
    objects of the same kind (e.g., physical objects with the same dimension).
    """

    def __init__(
        self,
        *names: str,
    ) -> None:
        self.names = names

    def __call__(self, func: typing.Callable) -> typing.Callable:
        """Ensure argument consistency before calling `func`."""
        if not self.names:
            return func
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if len(args) == 1:
                return func(*args, **kwargs)
            operands = Operands(*args)
            if not operands.consistent(*self.names):
                return NotImplemented
            if operands.agree(*self.names):
                return func(*args, **kwargs)
            raise OperandTypeError(*args)
        return wrapper


class NTypesError(Exception):
    """Inconsistent number of types in a new rule."""


class _RulesType(
    typing.Mapping[Types, Rule],
    collections.abc.Mapping,
    iterables.ReprStrMixin,
): ...


class Rules(_RulesType):
    """A class for managing operand-update rules."""

    def __init__(self, *parameters: str) -> None:
        """Initialize this instance.
        
        Parameters
        ----------
        *parameters : string
            Zero or more names of attributes to associate with unconstrained
            rules. See Notes for more information.

        Notes
        -----
        Providing the names of all possibly updatable attributes via
        `*parameters` protects against bugs that arise from naive use of
        ``inspect.signature``. For example, a class's ``__init__`` method may
        accept generic `*args` and `**kwargs`, which it then parses into
        specific attributes. In that case, ``inspect.signature`` will not
        discover the correct names of attributes necessary to initialize a new
        instance after applying a given rule.
        """
        self.parameters = list(parameters)
        """The default parameters to update for each rule."""
        self.ntypes = None
        """The number of argument types in these rules."""
        self._type = None
        self._implied = []
        self._rulemap = None

    def _parse(self, *default):
        """Parse initialization arguments."""
        if not default:
            return None, []
        if isinstance(default[0], type):
            return default[0], list(default[1:])
        return None, list(default)

    def imply(self, __type: type, *parameters: str):
        """Declare the implicit type and parameters."""
        self._type = __type
        self._implied = list(parameters)
        return self

    @property
    def implicit(self):
        """The implicit update rule."""
        if self._type is not None:
            parameters = self._implied or self.parameters
            return Rule(*parameters.copy())

    def register(self, types: Types, *parameters: typing.Optional[str]):
        """Add a rule to the collection.
        
        Parameters
        ----------
        types : type or tuple of types
            The argument type(s) in the target rule.

        *parameters : str
            Zero or more parameters to update based on `types`.

        Raises
        ------
        KeyError
            There is already a rule in the collection corresponding to `types`.

        See Also
        --------
        `~modify`: Update, restrict, or remove an existing rule's parameters.
        """
        key = tuple(iterables.whole(types))
        ntypes = len(key)
        self._check_ntypes(ntypes)
        if key not in self.mapping:
            self.mapping[key] = self._resolve(*parameters)
            return self
        raise KeyError(f"{types!r} is already in the collection") from None

    def suppress(self, types: Types):
        """Suppress the rule for these types."""
        self.modify(types, None)

    def modify(self, types: Types, *parameters: str, mode: str='update'):
        """Modify the parameters of an existing rule.
        
        Parameters
        ----------
        types : type or tuple of types
            The argument type(s) in the target rule.

        *parameters : str
            Zero or more parameters with which to modify the rule. The use of
            the given parameters depends on `mode`.

        mode : {'update', 'restrict', 'remove'}
            How to handle the given parameters::
        - update (default): Replace the existing parameters with the given
          parameters.
        - restrict: Restrict the target rule's parameters to the given
          parameters, and raise an exception if a parameter is not in the rule.
        - remove: Remove the given parameters from the target rule's parameters.

        Raises
        ------
        KeyError
            There is not an existing rule corresponding to `types`.

        See Also
        --------
        `~register`: Create a rule for `types`.
        """
        key = tuple(iterables.whole(types))
        if key not in self:
            if self._type in key:
                self.register(types)
            raise KeyError(f"Rule for {types!r} does not exist") from None
        if mode == 'update':
            new = parameters
        elif mode == 'restrict':
            rule = self[key]
            for parameter in parameters:
                if parameter not in rule.parameters:
                    raise ValueError(
                        "Can't restrict rule with new parameter"
                        f" {parameter}"
                    ) from None
            new = parameters
        elif mode == 'remove':
            rule = self[key]
            new = set(rule.parameters) - set(parameters)
        else:
            raise ValueError(f"Unknown mode {mode!r}")
        self.mapping[key] = self._resolve(*new)

    def _resolve(self, *parameters) -> typing.List[str]:
        """Determine the affected parameters based on input."""
        if len(parameters) == 1 and parameters[0] is None:
            return []
        if not parameters:
            return self.parameters.copy()
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
            yield (types, Rule(*self.mapping[types]))

    def __contains__(self, __o: Types) -> bool:
        """True if there is an explicit rule for these types."""
        return __o in self.mapping

    def __getitem__(self, __k: Types):
        """Retrieve the operand-update rule for `types`."""
        types = tuple(iterables.whole(__k))
        if types in self.mapping:
            return self._from(types)
        for t in self.mapping:
            if all(issubclass(i, j) for i, j in zip(types, t)):
                return self._from(t)
        if (
            self._type is not None
            and (self._type in types
            or any(issubclass(t, self._type) for t in types))
        ): return self.implicit
        raise KeyError(f"No rule for operand type(s) {__k!r}") from None

    def _from(self, types: Types):
        """Build a rule from the given types."""
        parameters = self.mapping.get(types, self.parameters.copy())
        return Rule(*parameters)

    def get(self, __types: Types, default: Rule=None):
        """Get the rule for these types, or a default rule.
        
        This method behaves like ``~typing.Mapping.get`` with a modified default
        value: Instead of returning ``None`` when key-based look-up fails (i.e.,
        there is no rule for the given types), this method returns a rule with
        no constraints for the given types.
        """
        return super().get(__types, default or Rule())

    def copy(self, implicit: bool=True):
        """Create a shallow copy of this instance.
        
        Parameters
        ----------
        implicit : bool, default=True
            If ``True``, also copy this instance's implict rule.
        """
        new = Rules(*self.parameters)
        new.mapping.update(self.mapping.copy())
        if implicit:
            new.imply(self._type, *self._implied)
        return new

    def __eq__(self, __o) -> bool:
        """True iff two instances have the same default parameters and rules."""
        return (
            isinstance(__o, Rules)
            and __o.parameters == self.parameters
            and __o.mapping == self.mapping
        )

    def __str__(self) -> str:
        return ', '.join(f"{t[0]}: {t[1]}" for t in self)


class OperationError(Exception):
    """An error occurred during an operation."""


class OperandTypeError(Exception):
    """Operands are incompatible for a given operation."""


Caller = typing.TypeVar('Caller', bound=typing.Callable)
Caller = typing.Callable[..., T]


A = typing.TypeVar('A')
B = typing.TypeVar('B')


_operators = [
    ('add', '__add__', 'addition'),
    ('sub', '__sub__', 'subtraction'),
]


class Operation:
    """A general arithmetic operation."""

    def __init__(self, method, rules: Rules) -> None:
        self.method = method
        self.rules = rules.copy()

    def compute(self, *args, reference=None, target=None, **kwargs):
        """Evaluate arguments within this operational context."""
        rule = self.get_rule(*args)
        if not rule.implemented:
            return NotImplemented
        if not rule.parameters:
            # We don't know which arguments to operate on, so we hand execution
            # over to the given operands, in case they implement this operator
            # in their class definitions. Note that this will lead to recursion
            # if they define the operator via this class.
            try:
                return self.method(*args, **kwargs)
            except RecursionError as err:
                raise OperationError(
                    f"Caught {err!r} when attempting to implement {self.method!r}. This may be because one of the operands uses {self!r} to implement {self.method!r} without explicit knowledge of the updatable attributes."
                ) from err
        operands = Operands(*args, reference=reference)
        fixed = tuple(set(self.rules.parameters) - set(rule.parameters))
        if not operands.agree(*fixed):
            errmsg = self._operand_errmsg(rule, operands)
            raise OperandTypeError(errmsg) from None
        defaults = self._get_defaults(reference)
        values = self._compute(operands, rule, defaults=defaults, **kwargs)
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

    def _operand_errmsg(self, rule: Rule, operands: Operands):
        """Build an error message based on `rule` and `operands`."""
        method_string = repr(self.method.__qualname__)
        types = operands.types
        types_string = (
            types[0].__qualname__ if len(types) == 1
            else f"({', '.join(t.__qualname__ for t in types)})"
        )
        fixed = tuple(set(self.rules.parameters) - set(rule.parameters))
        attrs_string = (
            repr(fixed[0]) if len(fixed) == 1
            else f"{fixed[0]!r} and {fixed[1]!r}" if len(fixed) == 2
            else f"{', '.join(fixed[:-1])} and {fixed[-1]}"
        )
        return (
            f"Can't apply operator {method_string} to {types_string}"
            f" with different values of {attrs_string}"
        )

    def _get_defaults(self, reference=None):
        """Get default values for initialization arguments."""
        if reference is None:
            return {}
        parameters = get_parameters(reference)
        if not parameters:
            return {}
        kinds = {parameter.kind for parameter in parameters.values()}
        if kinds == {
            inspect.Parameter.VAR_KEYWORD,
            inspect.Parameter.VAR_POSITIONAL,
        }: return {
            name: utilities.getattrval(reference, name)
            for name in self.rules.parameters
        }
        return {
            name: utilities.getattrval(reference, name)
            for name in parameters
        }

    def _compute(
        self,
        operands: Operands,
        rule: Rule,
        defaults: dict=None,
        **kwargs
    ):
        """Compute values based on parameters."""
        if not defaults:
            return [
                self.method(*operands.get(name), **kwargs)
                for name in rule.parameters
            ]
        return [
            self.method(*operands.get(name), **kwargs)
            if name in rule else value
            for name, value in defaults.items()
        ]


class Context(abc.ABC, iterables.ReprStrMixin):
    """Abstract base class for operation-related contexts."""

    def __init__(self, rules: Rules=None) -> None:
        self.rules = Rules() if rules is None else rules.copy()
        """This application's operand-update rules."""

    def spawn(self):
        """Create a new instance of this context with the current rules."""
        return type(self)(self.rules)

    @abc.abstractmethod
    def apply(self, __callable: typing.Callable):
        """Use the given callable object within this context."""
        pass

    def __str__(self) -> str:
        return str(self.rules)


class Default(Context):
    """A factory for generalized operators."""

    def apply(self, __callable: typing.Callable[..., T]):
        """Implement this operation with the given callable object."""
        operation = Operation(__callable, self.rules)
        def operator(*args, **kwargs) -> T:
            return operation.compute(*args, **kwargs)
        return operator


class Cast(Context):
    """A factory for type-casting operators."""

    def apply(self, __callable: typing.Type[T]):
        operation = Operation(__callable, self.rules)
        def operator(a: A) -> T:
            return operation.compute(a)
        operator.__name__ = f'__{__callable.__name__}__'
        operator.__doc__ = __callable.__doc__
        return operator


class Unary(Context):
    """A factory for unary arithmetic operators."""

    CType = typing.TypeVar('CType', bound=typing.Callable)
    CType = typing.Callable[[A], A]

    def apply(self, __callable: CType):
        operation = Operation(__callable, self.rules)
        def operator(a: A, /, **kwargs) -> A:
            return operation.compute(a, reference=a, target=type(a), **kwargs)
        operator.__name__ = f'__{__callable.__name__}__'
        operator.__doc__ = __callable.__doc__
        return operator


class Comparison(Context):
    """A factory for binary comparison operators."""

    CType = typing.TypeVar('CType', bound=typing.Callable)
    CType = typing.Callable[[A, B], T]

    def apply(self, __callable: CType):
        operation = Operation(__callable, self.rules)
        def operator(a: A, b: B, /) -> T:
            return operation.compute(a, b)
        operator.__name__ = f'__{__callable.__name__}__'
        operator.__doc__ = __callable.__doc__
        return operator


class Forward(Context):
    """A factory for standard binary numeric operators."""

    CType = typing.TypeVar('CType', bound=typing.Callable)
    CType = typing.Callable[[A, B], T]

    def apply(self, __callable: CType):
        operation = Operation(__callable, self.rules)
        def operator(a: A, b: B, /, **kwargs) -> A:
            try:
                result = operation.compute(a, b, reference=a, target=type(a), **kwargs)
            except metric.UnitError as err:
                raise OperandTypeError(err) from err
            else:
                return result
        operator.__name__ = f'__{__callable.__name__}__'
        operator.__doc__ = __callable.__doc__
        return operator


class Reverse(Context):
    """A factory for binary numeric operators with reflected operands."""

    CType = typing.TypeVar('CType', bound=typing.Callable)
    CType = typing.Callable[[A, B], T]

    def apply(self, __callable: CType):
        operation = Operation(__callable, self.rules)
        def operator(b: B, a: A, /, **kwargs) -> B:
            try:
                result = operation.compute(a, b, reference=b, target=type(b), **kwargs)
            except metric.UnitError as err:
                raise OperandTypeError(err) from err
            else:
                return result
        operator.__name__ = f'__r{__callable.__name__}__'
        operator.__doc__ = __callable.__doc__
        return operator


class Inplace(Context):
    """A factory for binary numeric operators with in-place updates."""

    CType = typing.TypeVar('CType', bound=typing.Callable)
    CType = typing.Callable[[A, B], T]

    def apply(self, __callable: CType):
        operation = Operation(__callable, self.rules)
        def operator(a: A, b: B, /, **kwargs) -> A:
            try:
                result = operation.compute(a, b, reference=a, target=a, **kwargs)
            except metric.UnitError as err:
                raise OperandTypeError(err) from err
            else:
                return result
        operator.__name__ = f'__i{__callable.__name__}__'
        operator.__doc__ = __callable.__doc__
        return operator


CATEGORIES: typing.Dict[str, typing.Type[Context]] = {
    'default': Default,
    'cast': Cast,
    'unary': Unary,
    'comparison': Comparison,
    'forward': Forward,
    'reverse': Reverse,
    'inplace': Inplace,
}
"""The canonical operation-category implementations."""


OPERATIONS = {
    'int': {
        'category': 'cast',
        'operator': int,
    },
    'float': {
        'category': 'cast',
        'operator': float,
    },
    'abs': {
        'category': 'unary',
        'operator': abs,
    },
    'neg': {
        'category': 'unary',
        'operator': standard.neg,
    },
    'pos': {
        'category': 'unary',
        'operator': standard.pos,
    },
    'ceil': {
        'category': 'unary',
        'operator': math.ceil,
    },
    'floor': {
        'category': 'unary',
        'operator': math.floor,
    },
    'trunc': {
        'category': 'unary',
        'operator': math.trunc,
    },
    'round': {
        'category': 'unary',
        'operator': round,
    },
    'lt': {
        'category': 'comparison',
        'operator': standard.lt,
    },
    'le': {
        'category': 'comparison',
        'operator': standard.le,
    },
    'gt': {
        'category': 'comparison',
        'operator': standard.gt,
    },
    'ge': {
        'category': 'comparison',
        'operator': standard.ge,
    },
    'eq': {
        'category': 'comparison',
        'operator': standard.eq,
    },
    'ne': {
        'category': 'comparison',
        'operator': standard.ne,
    },
    'add': {
        'category': 'numeric',
        'operator': standard.add,
    },
    'sub': {
        'category': 'numeric',
        'operator': standard.sub,
    },
    'mul': {
        'category': 'numeric',
        'operator': standard.mul,
    },
    'truediv': {
        'category': 'numeric',
        'operator': standard.truediv,
    },
    'pow': {
        'category': 'numeric',
        'operator': pow,
    },
}
"""A mapping of operation name to metadata."""


OPERATORS = {
    f'__{k}__': v.copy() for k, v in OPERATIONS.items()
    if v['category'] != 'numeric'
}
OPERATORS.update(
    {
        f'__{k}__': {'operator': v['operator'], 'category': 'forward'}
        for k, v in OPERATIONS.items() if v['category'] == 'numeric'
    }
)
OPERATORS.update(
    {
        f'__r{k}__': {'operator': v['operator'], 'category': 'reverse'}
        for k, v in OPERATIONS.items() if v['category'] == 'numeric'
    }
)
OPERATORS.update(
    {
        f'__i{k}__': {'operator': v['operator'], 'category': 'inplace'}
        for k, v in OPERATIONS.items() if v['category'] == 'numeric'
    }
)


NAMES = {
    c: [f'__{k}__' for k, v in OPERATIONS.items() if v['category'] == c]
    for c in CATEGORIES if c != 'numeric'
}
_numeric = [k for k, v in OPERATIONS.items() if v['category'] == 'numeric']
NAMES.update(
    {
        'forward': [f'__{i}__' for i in _numeric],
        'reverse': [f'__r{i}__' for i in _numeric],
        'inplace': [f'__i{i}__' for i in _numeric],
    }
)
copied = NAMES.copy()
NAMES['all'] = [
    name for category in copied.values() for name in category
]


class Interface(collections.abc.Mapping):
    """Top-level interface to arithmetic operations."""

    def __init__(
        self,
        *parameters: str,
        default: typing.Sequence[typing.Union[type, str]]=None,
    ) -> None:
        """
        Initialize this instance.

        Parameters
        ----------
        *parameters : string
            Zero or more strings representing the updatable attributes in each
            operand to these operations.

        default : type and strings, optional
            An object type and zero or more names of attributes to update
            whenever the given object type appears in an operation, unless
            overriden by an explicit rule.
        """
        self.rules = Rules(*parameters)
        if default is not None:
            self.rules.imply(*default)
        self.parameters = parameters
        """The names of all updatable attributes"""
        self._categories = {k: v(self.rules) for k, v in CATEGORIES.items()}
        self._operations = None

    def implement(self, __k: str, method: typing.Callable):
        """Implement the named operator."""
        context = self[__k] if __k in self else self._categories['default']
        return context.apply(method)

    def __getitem__(self, __k: str) -> Context:
        """Retrieve the operation context for this operator."""
        if __k in self.operations:
            return self.operations[__k]
        raise KeyError(f"Unknown operator {__k!r}") from None

    def __len__(self) -> int:
        """The number of defined operations."""
        return len(self.operations)

    def __iter__(self):
        """Iterate over operation contexts."""
        return iter(self.operations)

    def mixin(self):
        """Generate a mixin operator class from the current state."""
        ops = {
            f'__{k}__': self[k].apply(v['default'])
            for k, v in OPERATIONS.items()
        }
        reflected = {
            f'__r{k}__': self[k].apply(v['default'], 'reverse')
            for k, v in OPERATIONS.items()
            if v['category'] == 'numeric'
        }
        ops.update(reflected)
        return type('OperatorsMixin', (), ops)

    @property
    def operations(self) -> typing.Dict[str, Context]:
        """The standard operation contexts defined here."""
        if self._operations is None:
            self._operations = {
                k: self._categories[v['category']].spawn()
                for k, v in OPERATORS.items()
            }
        return self._operations

    @property
    def cast(self) -> Cast:
        """An interface to type-casting operations."""
        return self._create_category('cast')

    @property
    def unary(self) -> Unary:
        """An interface to unary arithmetic operations."""
        return self._create_category('unary')

    @property
    def comparison(self) -> Comparison:
        """An interface to binary comparison operations."""
        return self._create_category('comparison')

    @property
    def numeric(self):
        """An interface to binary arithmetic operations."""
        return self._create_category('numeric')

    def _create_category(self, name: str):
        """Update the named category context from the current rules."""
        category = CATEGORIES[name]
        context = category(self.rules)
        if name not in self._categories:
            self._categories[name] = context
            return context
        return self._categories[name]


def augment(
    target: type,
    name: str,
    interface: Interface=None,
    include: typing.Iterable[str]=None,
    exclude: typing.Iterable[str]=None,
    # reverse: bool=False,
    # inplace: bool=False,
) -> typing.Type:
    """Create a subclass of `target` with mixin operators.
    
    Parameters
    ----------
    target : type
        The parent type to use as a base class when creating the new class.

    interface : `~operations.Interface`, optional
        An instance of the class that provides dynamic access to operation
        contexts. If ``None`` (the default), this function will create a plain
        instance with the default type set to `target`.
    """
    if not interface:
        interface = Interface(default=[target])
    # User should be able to include or exclude
    # - cast, unary, or comparison categories
    # - entire numeric category or any of forward, reverse, or inplace
    # - individual operation or operator names (i.e., with or without double
    #   underscores)
    # - logical combinations of the above (e.g., all cast and all unary except
    #   abs)

    # incnames = []
    # for string in include:
    #     if string in CATEGORIES:
    #         incnames.append(s for s in NAMES[string])
    #     else:
    #         incnames.append(string)
    # excnames = []
    # for string in exclude:
    #     if string in _by_category:
    #         excnames.append(s for s in _by_category[string])
    #     else:
    #         excnames.append(string)
    # included = (set(NAMES) - set(excnames)) | set(incnames)
    if not include and not exclude:
        included = list(OPERATORS)
    else:
        tmp = []
        for name in include:
            if name in CATEGORIES:
                tmp.extend(NAMES[name])
            else:
                tmp.append(name)
        current = set(tmp)
        for name in exclude:
            if name in CATEGORIES:
                current -= set(NAMES[name])
            else:
                current -= {name}
        included = list(current)
    operators = {
        k: interface.implement(k, OPERATORS['operator'])
        for k in included
    }
    # operators = {
    #     f'__{k.strip("__")}__': v['operator']
    #     for k, v in OPERATIONS.items()
    #     if k in included
    # }
    # ops = {
    #     f'__{k}__': interface[k].apply(v['default'])
    #     for k, v in OPERATIONS.items()
    # }
    # if reverse:
    #     updates = {
    #         f'__r{k}__': interface[k].apply(v['default'], 'reverse')
    #         for k, v in OPERATIONS.items()
    #         if v['category'] == 'numeric'
    #     }
    #     ops.update(updates)
    # if inplace:
    #     updates = {
    #         f'__i{k}__': interface[k].apply(v['default'], 'inplace')
    #         for k, v in OPERATIONS.items()
    #         if v['category'] == 'numeric'
    #     }
    #     ops.update(updates)
    return type(name, (target,), operators)

