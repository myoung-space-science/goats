import math
import numbers
import operator as standard
import typing

import pytest

from goats.core import operations


def test_unique():
    """Test the function that extracts unique items while preserving order."""
    cases = {
        'a': ['a'],
        ('a', 'b'): ['a', 'b'],
        ('a', 'b', 'a'): ['a', 'b'],
        ('a', 'b', 'a', 'c'): ['a', 'b', 'c'],
        ('a', 'b', 'b', 'a', 'c'): ['a', 'b', 'c'],
    }
    for items, expected in cases.items():
        assert sorted(operations.unique(*items)) == expected


def test_types():
    """Test the collection of operand types."""
    types = operations.Types()
    types.add(int, int)
    assert len(types) == 1
    assert (int, int) in types
    types.add(str, float)
    assert len(types) == 2
    assert (str, float) in types
    types.discard(int, int)
    assert len(types) == 1
    assert (str, float) in types
    types.clear()
    assert len(types) == 0
    types.add(int, float, symmetric=True)
    assert len(types) == 2
    assert (int, float) in types and (float, int) in types
    copied = types.copy()
    assert copied == types
    assert copied is not types
    types.clear()
    types.add(numbers.Real, numbers.Real)
    assert types.supports(numbers.Real, numbers.Real)
    assert types.supports(int, float)
    assert not types.supports(int, str)
    assert types
    types.clear()
    assert not types
    types = operations.Types()
    types.add(int)
    assert int in types


def test_types_add_multiple():
    """Test the ability to add multiple type specifications."""
    types = operations.Types()
    assert len(types) == 0
    user = [
        (int, float),
        (float, float),
        (int, int),
    ]
    types.add(*user)
    assert len(types) == 3
    for these in user:
        assert these in types


def test_types_implied():
    """Test the implied type specification."""
    types = operations.Types(implied=str)
    assert str in types
    assert (str, str) in types
    assert (str, str, str) in types
    assert (str, float) not in types
    types.add(int, float)
    assert str not in types
    assert (str, str) in types
    assert (str, str, str) not in types
    assert (str, float) not in types
    types.discard(str, str)
    assert (str, str) not in types


def test_operands():
    """Test the `Operands` class."""
    inputs = ['a', 1, 2.3]
    operands = operations.Operands(*inputs)
    assert operands.types == (str, int, float)
    subset = operands[:]
    assert isinstance(subset, operations.Operands)
    assert subset == operands
    for index in [-2, -1, 0, 1, 2]:
        assert operands[index] == inputs[index]


class Info:
    """Information about a value."""

    def __init__(self, arg) -> None:
        self._text = arg._text if isinstance(arg, Info) else arg

    __abs__ = operations.identity(abs)
    __pos__ = operations.identity(standard.pos)
    __neg__ = operations.identity(standard.neg)
    __ceil__ = operations.identity(math.ceil)
    __floor__ = operations.identity(math.floor)
    __trunc__ = operations.identity(math.trunc)
    __round__ = operations.identity(round)

    __add__ = operations.identity(standard.add)
    __sub__ = operations.identity(standard.sub)
    __mul__ = operations.identity(standard.mul)
    __truediv__ = operations.identity(standard.truediv)
    __pow__ = operations.identity(standard.pow)

    def __eq__(self, other):
        return isinstance(other, Info) and other._text == self._text

    def __repr__(self):
        return f"{self.__class__.__qualname__}({self._text!r})"


class Base:
    """A base for test classes."""
    def __init__(self, value: numbers.Real, info: str) -> None:
        self.value = value
        """A numerical value."""
        self.info = Info(info)
        """Some information about this instance."""

    def __eq__(self, other):
        """True if two instances have the same value and info."""
        return (
            isinstance(other, type(self))
            and all(
                getattr(self, name) == getattr(other, name)
                for name in ('value', 'info')
            )
        )

    def __repr__(self):
        return f"{self.__class__.__qualname__}({self.value}, {self.info})"


class Score:
    def __init__(self, points: float, kind: str, name: str) -> None:
        self.points = points
        self.kind = kind
        self.name = name


T = typing.TypeVar('T')


def build(__type: typing.Type[T]) -> typing.List[T]:
    """Create and initialize instances of a class for tests."""
    inputs = [
        (1, 'A'),
        (2, 'A'),
        (2, 'B'),
    ]
    return [__type(*args) for args in inputs]


def test_operands_find():
    """Test the ability to collect operands with named attributes."""
    instances = build(Base)
    operands = operations.Operands(*instances)
    assert operands.find('value', 'info') == instances
    operands = operations.Operands(instances[0], 0.0)
    assert operands.find('value', 'info') == [instances[0]]


def test_operands_consistent():
    """Test the ability to check for named attributes across operands."""
    instances = build(Base)
    operands = operations.Operands(*instances)
    assert operands.consistent('value', 'info')
    operands = operations.Operands(instances[0], 0.0)
    assert not operands.consistent('value', 'info')


def test_operands_agree():
    """Users should be able to check consistency of object attributes."""
    instances = build(Base)
    same = {
        'value': [
            [instances[1], instances[2]],
            [instances[1], 0.0], # trivial: nothing to compare
        ],
        'info': [
            [instances[0], instances[1]],
            [instances[0], 0.0], # trivial: nothing to compare
        ],
    }
    for name, inputs in same.items():
        for args in inputs:
            operands = operations.Operands(*args)
            assert operands.agree(name)
    for instance in instances: # single object trivially agrees with itself
        assert operations.Operands(instance).agree(same.keys())
    different = {
        'value': [instances[0], instances[1]],
        'info': [instances[1], instances[2]],
    }
    for name, args in different.items():
        assert not operations.Operands(*args).agree(name)


def test_same():
    """Test the class that enforces object consistency.

    This test first defines a demo class that requires a value, a kind, and a
    name. It then defines simple functions that add the values of two instances
    of that class, with various restrictions on which attributes need to be the
    same.
    """
    scores = [
        Score(2.0, 'cat', 'Squirt'),
        Score(3.0, 'cat', 'Paul'),
        Score(1.0, 'dog', 'Bornk'),
        Score(6.0, 'cat', 'Paul'),
    ]

    # No restrictions:
    def f0(*scores: Score):
        return sum((score.points for score in scores))

    # Instances must have the same kind:
    @operations.same('kind')
    def f1(*args):
        return f0(*args)

    # Instances must have the same kind and the same name:
    @operations.same('kind', 'name')
    def f2(*args):
        return f0(*args)

    # A single instance always passes:
    @operations.same('kind')
    def f3(arg):
        return arg

    # Add two instances with no restrictions.
    assert f0(scores[0], scores[2]) == 3.0

    # Add all instances with no restrictions.
    assert f0(*scores) == 12.0

    # Add two instances with restricted kind.
    assert f1(scores[0], scores[1]) == 5.0

    # Add three instances with restricted kind.
    args = [scores[i] for i in (0, 1, 3)]
    assert f1(*args) == 11.0

    # Try to add two instances with different kind.
    with pytest.raises(operations.OperandTypeError):
        f1(scores[0], scores[2])

    # Try to add an instance to a built-in float.
    assert f1(scores[0], 2.0) == NotImplemented

    # Add two instances with restricted kind and name.
    assert f2(scores[1], scores[3]) == 9.0

    # Try to add two instances with same kind but different name.
    with pytest.raises(operations.OperandTypeError):
        f2(scores[0], scores[1])

    # Test a trivial case.
    assert f3(scores[0]) == scores[0]


def test_cast_builtin():
    """Test a type-cast operation on a built-in object."""
    operation = operations.Cast()
    builtin = int
    operator = operation.apply(builtin)
    value = 3.3
    result = operator(value)
    assert isinstance(result, builtin)
    assert result == builtin(value)


def test_cast_custom():
    """Test a type-cast operation on a custom object."""
    operation = operations.Cast('value', 'info')
    instances = build(Base)
    builtin = int
    operator = operation.apply(builtin)
    with pytest.raises(TypeError):
        builtin(instances[0])
    operation.types.add(Base)
    operator = operation.apply(builtin)
    for instance in instances:
        assert operator(instance) == builtin(instance.value)


def test_unary_builtin():
    """Test a unary arithmetic operation on a built-in object."""
    operation = operations.Unary()
    builtin = round
    operator = operation.apply(builtin)
    value = 3.3
    assert operator(value) == builtin(value)


def test_unary_custom():
    """Test a unary arithmetic operation on a custom object."""
    operation = operations.Unary('value', 'info')
    instances = build(Base)
    builtin = round
    operator = operation.apply(builtin)
    with pytest.raises(TypeError):
        builtin(instances[0])
    operation.types.add(Base)
    operator = operation.apply(builtin)
    for instance in instances:
        expected = Base(builtin(instance.value), instance.info)
        assert operator(instance) == expected


def test_comparison_builtin():
    """Test a binary comparison operation on a built-in object."""
    operation = operations.Comparison()
    builtin = standard.lt
    operator = operation.apply(builtin)
    values = 2, 4
    assert operator(*values) == builtin(*values)


def test_comparison_custom():
    """Test a binary comparison operation on a custom object."""
    operation = operations.Comparison('value', 'info')
    builtin = standard.lt
    instances = build(Base)
    operator = operation.apply(builtin)
    with pytest.raises(TypeError):
        builtin(instances[0], instances[1])
    operation.types.add(Base, Base)
    operator = operation.apply(builtin)
    assert operator(instances[0], instances[1])
    assert not operator(instances[1], instances[0])
    with pytest.raises(operations.OperandTypeError):
        operator(instances[0], instances[2])


def test_numeric_builtin():
    """Test a binary numeric operation on a built-in object."""
    operation = operations.Numeric()
    builtin = standard.add
    operator = operation.apply(builtin)
    values = 2, 4
    types = {
        (int, int): int,
        (float, int): float,
        (int, float): float,
    }
    for casts, rtype in types.items():
        inputs = [cast(value) for cast, value in zip(casts, values)]
        result = operator(*inputs)
        assert isinstance(result, rtype)
        assert result == builtin(*inputs)


def test_numeric_custom():
    """Test a binary numeric operation on a built-in object."""
    operation = operations.Numeric('value', 'info')
    builtin = standard.add
    instances = build(Base)
    operator = operation.apply(builtin)
    with pytest.raises(TypeError):
        builtin(instances[0], instances[1])
    operation.types.add(Base, Base)
    operator = operation.apply(builtin)
    expected = Base(
        builtin(instances[0].value, instances[1].value),
        instances[0].info,
    )
    assert operator(instances[0], instances[1]) == expected
    with pytest.raises(operations.OperandTypeError):
        operator(instances[0], instances[2])


CATEGORIES = {
    'cast': {
        'ntypes': 1,
        'context': operations.Cast,
        'operations': {
            'int': int,
            'float': float,
        },
    },
    'unary': {
        'ntypes': 1,
        'context': operations.Unary,
        'operations': {
            'abs': standard.abs,
            'neg': standard.neg,
            'pos': standard.pos,
            'ceil': math.ceil,
            'floor': math.floor,
            'trunc': math.trunc,
            'round': round,
        },
    },
    'comparison': {
        'ntypes': 2,
        'context': operations.Comparison,
        'operations': {
            'lt': standard.lt,
            'le': standard.le,
            'gt': standard.gt,
            'ge': standard.ge,
            # 'eq': standard.eq,
            # 'ne': standard.ne,
        },
    },
    'numeric': {
        'ntypes': 2,
        'context': operations.Numeric,
        'operations': {
            'add': standard.add,
            'sub': standard.sub,
            'mul': standard.mul,
            'truediv': standard.truediv,
            'pow': pow,
        },
    },
}


@pytest.fixture
def method_names():
    """A dictionary of method names by category."""
    names = {
        k: [f'__{i}__' for i in v['operations']]
        for k, v in CATEGORIES.items() if k != 'numeric'
    }
    v = CATEGORIES['numeric']['operations']
    numeric = {
        'forward': [f'__{i}__' for i in v],
        'reverse': [f'__r{i}__' for i in v],
        'inplace': [f'__i{i}__' for i in v],
    }
    names.update(numeric)
    copied = names.copy()
    names['all'] = [
        name for category in copied.values() for name in category
    ]
    return names


@pytest.fixture
def interface():
    """An operations interface."""
    return operations.Interface(Base, 'value', 'info')


# Attempting to operate on instances of `Base` should raise a `TypeError` if it
# doesn't implement the operator, which will be the case for most. Inheriting
# from a mixin class that implements the missing operator should resolve the
# `TypeError`. The mixin class can come from `Interface.subclass`. We will also
# want to implement a custom operator by passing a callable to
# `Interface.implement` (e.g., in `__array_ufunc__`). Therefore, I think the two
# primary groups of tests should 1) test built-in operators on `Base` and a
# subclass with implemented operators, and 2) test custom operator
# implementations from the interface on that subclass.
# - call built-in unary operators on
#   - `Base` -> `TypeError`
# - call built-in binary operators on
#   - (`Base`, `Base`) -> `TypeError`
#   - (`float`, `Base`) -> `TypeError`
#   - (`Base`, `float`) -> `TypeError`
# - define `Custom(Mixin, Base)`
# - call built-in unary operators on
#   - `Custom` -> result
# - call built-in binary operators on
#   - (`Custom`, `Custom`) -> result
#   - (`float`, `Custom`) -> result
#   - (`Custom`, `float`) -> result
# - call custom unary operators on
#   - `Custom` -> result
# - call custom binary operators on
#   - (`Custom`, `Custom`) -> result
#   - (`float`, `Custom`) -> result
#   - (`Custom`, `float`) -> result


def test_builtin_cast(interface: operations.Interface):
    """Test cast operations on custom objects."""
    builtins = CATEGORIES['cast']['operations'].values()
    instances = build(Base)
    for builtin in builtins:
        for instance in instances:
            with pytest.raises(TypeError):
                builtin(instance)
    Custom = interface.subclass('Custom')
    instances = build(Custom)
    for builtin in builtins:
        for instance in instances:
            expected = builtin(instance.value)
            assert builtin(instance) == expected


def test_builtin_comparison(interface: operations.Interface):
    """Test comparison operations on custom objects."""
    builtins = CATEGORIES['comparison']['operations'].values()
    instances = build(Base)
    targets = instances[0], instances[1]
    for builtin in builtins:
        with pytest.raises(TypeError):
            builtin(*targets)
    Custom = interface.subclass('Custom')
    instances = build(Custom)
    targets = instances[0], instances[1]
    for builtin in builtins:
        assert builtin(*targets) == builtin(*[c.value for c in targets])
        with pytest.raises(operations.OperandTypeError):
            builtin(instances[0], instances[2])


def test_builtin_unary(interface: operations.Interface):
    """Test unary operations on custom objects."""
    builtins = CATEGORIES['unary']['operations'].values()
    instances = build(Base)
    for builtin in builtins:
        for instance in instances:
            with pytest.raises(TypeError):
                builtin(instance)
    Custom = interface.subclass('Custom')
    instances = build(Custom)
    for builtin in builtins:
        for instance in instances:
            expected = Base(builtin(instance.value), instance.info)
            assert builtin(instance) == expected


def test_builtin_numeric(interface: operations.Interface):
    """Test numeric operations on custom objects."""
    builtins = CATEGORIES['numeric']['operations'].values()
    instances = build(Base)
    targets = instances[0], instances[1]
    for builtin in builtins:
        with pytest.raises(TypeError):
            builtin(*targets)
    Custom = interface.subclass('Custom')
    instances = build(Custom)
    targets = instances[0], instances[1]
    for builtin in builtins:
        expected = Base(
            builtin(*[c.value for c in targets]),
            targets[0].info,
        )
        assert builtin(*targets) == expected
        with pytest.raises(operations.OperandTypeError):
            builtin(instances[0], instances[2])


def test_interface_categories(interface: operations.Interface):
    """Test the ability to access and update category contexts."""
    for name, current in CATEGORIES.items():
        category = interface[name]
        assert isinstance(category, current['context'])
        assert len(category.types) == 0
        types = [Base] * current['ntypes']
        category.types.add(*types)
        assert len(category.types) == 1
    assert Base in interface['cast'].types
    assert Base in interface['unary'].types
    assert (Base, Base) in interface['comparison'].types
    assert (Base, Base) in interface['numeric'].types


def test_interface_operations(interface: operations.Interface):
    """Test the ability to implement and cache operations."""
    add = interface['add']
    assert add is interface['add']
    assert add is interface['__add__']
    assert add.types == interface['numeric'].types
    assert (Base, Base) in interface['numeric'].types
    add.types.add(Base, float)
    assert len(interface['add'].types) == 1
    assert (Base, float) in interface['add'].types
    for defined in CATEGORIES.values():
        context = defined['context']
        for k in defined['operations']:
            assert isinstance(interface[k], context)


def test_interface_update_rule(interface: operations.Interface):
    """Make sure we can independently update rules."""
    interface['numeric'].types.add(Base, float)
    assert (Base, float) in interface['add'].types
    interface['__add__'].types.discard(Base, float)
    assert (Base, float) not in interface['add'].types
    assert (Base, float) in interface['numeric'].types
    interface['numeric'].types.discard(Base, Base)
    assert (Base, Base) not in interface['numeric'].types


def test_interface_subclass(
    interface: operations.Interface,
    method_names: dict,
) -> None:
    """Test the ability to generate subclasses from interface methods."""
    New = interface.subclass('New')
    assert issubclass(New, Base)
    c0 = New(1.2, 'this')
    c1 = New(2.1, 'this')
    # Check a unary cast operation.
    assert int(c0) == 1
    # Check a unary arithmetic operation.
    assert -c0 == -1.2
    # Check a binary comparison operation.
    assert c0 < c1
    # Check a binary numeric operation.
    assert c0 + c1 == New(3.3, 'this')
    # Make sure the default case defines all methods.
    assert all(name in dir(New) for name in method_names['all'])

