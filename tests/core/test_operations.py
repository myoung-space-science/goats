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


def test_rule_len():
    """Test the length of an operator rule."""
    for parameters in ([], ['a'], ['a', 'b'], ['a', 'b', 'c']):
        assert len(operations.Rule(*parameters)) == len(parameters)


def test_rule_contains():
    """Check for a type in an operator rule."""
    rule = operations.Rule('a', 'b')
    for name in ['a', 'b']:
        assert name in rule


def test_rules_register():
    """Test the ability to register operand rules."""
    default = ['a', 'b', 'c']
    rules = operations.Rules(*default)
    assert not rules
    assert rules.ntypes is None
    rules.register(int, float, 'a', 'b')
    assert rules
    assert rules.ntypes == 2
    assert rules[(int, float)].parameters == ['a', 'b']
    rules.register(float, float)
    assert rules[(float, float)].parameters == default
    rules.register(int, int, None)
    assert not rules[(int, int)].parameters
    with pytest.raises(operations.NTypesError):
        rules.register(int, 'a')


def test_rules_batch_register():
    """Test the ability to register multiple rules at once."""
    default = ['a', 'b', 'c']
    rules = operations.Rules(*default)
    assert not rules
    assert rules.ntypes is None
    user = [
        (int, float, 'a', 'b'),
        (float, float),
        (int, int, None),
    ]
    rules.register(*user)
    assert rules
    assert rules.ntypes == 2
    assert rules[(int, float)].parameters == ['a', 'b']
    assert rules[(float, float)].parameters == default
    assert not rules[(int, int)].parameters


def test_rules_update_rule():
    """Test the ability to dynamically update a rule in Rules."""
    rules = operations.Rules()
    rules.register(float, float, 'd', 'e', 'f')
    rule = rules[(float, float)]
    assert sorted(rule.parameters) == sorted(['d', 'e', 'f'])
    rule.update('a', 'b')
    assert sorted(rule.parameters) == sorted(['a', 'b'])
    rule.append('c')
    assert sorted(rule.parameters) == sorted(['a', 'b', 'c'])
    rule.restrict('a', 'b')
    assert sorted(rule.parameters) == sorted(['a', 'b'])
    with pytest.raises(ValueError):
        rule.restrict('a', 'd')
    rule.remove('a')
    assert rule.parameters == ['b']
    rule.suppress
    assert not rule.implemented


def test_rules_suppress():
    """Allow the user to suppress a rule with or without registering it."""
    rules = operations.Rules()
    rules.register(float, float, 'd', 'e', 'f')
    assert rules[float, float].implemented
    rules.suppress(float, float)
    assert not rules[float, float].implemented
    with pytest.raises(KeyError):
        rules[float, int]
    rules.suppress(float, int)
    assert not rules[float, int].implemented


def test_rules_copy():
    """Test the ability to copy an instance."""
    default = ['a', 'b', 'c']
    rules = operations.Rules(*default)
    rules.register(int, 'a')
    rules.register(float, 'a', 'b')
    copied = rules.copy()
    assert copied == rules
    assert copied is not rules
    for t in [int, float]:
        assert rules[t] == copied[t]
        assert rules[t] is copied[t]
    copied = rules.copy(deep=True)
    assert copied == rules
    assert copied is not rules
    for t in [int, float]:
        assert rules[t] == copied[t]
        assert rules[t] is not copied[t]


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


class Base:
    """A base for test classes."""
    def __init__(self, value: numbers.Real, info: str) -> None:
        self.value = value
        """A numerical value."""
        self.info = info
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


def test_rules_implicit():
    """Allow the user to specify an implicit default rule."""
    rules = operations.Rules('a', 'b', 'c')
    rules.imply(Base, 'a')
    assert len(rules) == 0
    assert rules[Base, Base].parameters == ['a']
    assert rules[Base, float].parameters == ['a']
    rules.register(Base, Base, 'b')
    assert len(rules) == 1
    assert rules[Base, Base].parameters == ['b']
    assert rules[Base, float].parameters == ['a']


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
    rules = operations.Rules('value', 'info')
    operation = operations.Cast(rules)
    instances = build(Base)
    builtin = int
    operator = operation.apply(builtin)
    with pytest.raises(TypeError):
        operator(instances[0])
    operation.rules.register(Base, 'value')
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
    rules = operations.Rules('value', 'info')
    operation = operations.Unary(rules)
    instances = build(Base)
    builtin = round
    operator = operation.apply(builtin)
    with pytest.raises(TypeError):
        operator(instances[0])
    operation.rules.register(Base, 'value')
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
    rules = operations.Rules('value', 'info')
    operation = operations.Comparison(rules)
    builtin = standard.lt
    instances = build(Base)
    operator = operation.apply(builtin)
    with pytest.raises(TypeError):
        operator(instances[0], instances[1])
    operation.rules.register(Base, Base, 'value')
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
    rules = operations.Rules('value', 'info')
    operation = operations.Numeric(rules)
    builtin = standard.add
    instances = build(Base)
    operator = operation.apply(builtin)
    with pytest.raises(TypeError):
        operator(instances[0], instances[1])
    operation.rules.register(Base, Base, 'value')
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
            'eq': standard.eq,
            'ne': standard.ne,
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
    return operations.Interface('value', 'info', default=[Base, 'value'])


def test_cast_interface(interface: operations.Interface):
    """Test cast operations via the module interface."""
    operation = interface['cast']
    instances = build(Base)
    for builtin in CATEGORIES['cast']['operations'].values():
        operator = operation.apply(builtin)
        for instance in instances:
            expected = builtin(instance.value)
            assert operator(instance) == expected


def test_unary_interface(interface: operations.Interface):
    """Test unary operations via the module interface."""
    operation = interface['unary']
    instances = build(Base)
    for builtin in CATEGORIES['unary']['operations'].values():
        operator = operation.apply(builtin)
        for instance in instances:
            expected = Base(builtin(instance.value), instance.info)
            assert operator(instance) == expected


def test_comparison_interface(interface: operations.Interface):
    """Test comparison operations via the module interface."""
    operation = interface['comparison']
    instances = build(Base)
    targets = instances[0], instances[1]
    for builtin in CATEGORIES['comparison']['operations'].values():
        operator = operation.apply(builtin)
        assert operator(*targets) == builtin(*[c.value for c in targets])
        with pytest.raises(operations.OperandTypeError):
            operator(instances[0], instances[2])


def test_numeric_interface(interface: operations.Interface):
    """Test numeric operations via the module interface."""
    operation = interface['numeric']
    instances = build(Base)
    targets = instances[0], instances[1]
    for builtin in CATEGORIES['numeric']['operations'].values():
        operator = operation.apply(builtin)
        expected = Base(
            builtin(*[c.value for c in targets]),
            targets[0].info,
        )
        assert operator(*targets) == expected
        with pytest.raises(operations.OperandTypeError):
            operator(instances[0], instances[2])


def test_interface_categories(interface: operations.Interface):
    """Test the ability to access and update category contexts."""
    for name, current in CATEGORIES.items():
        category = interface[name]
        assert isinstance(category, current['context'])
        assert len(category.rules) == 0
        types = [Base] * current['ntypes']
        category.rules.register(*types, 'value')
        assert len(category.rules) == 1
    assert interface['cast'].rules[Base].parameters == ['value']
    assert interface['unary'].rules[Base].parameters == ['value']
    assert interface['comparison'].rules[Base, Base].parameters == ['value']
    assert interface['numeric'].rules[Base, Base].parameters == ['value']


def test_interface_operations(interface: operations.Interface):
    """Test the ability to implement and cache operations."""
    add = interface['add']
    assert add is interface['add']
    assert add is interface['__add__']
    assert add.rules == interface['numeric'].rules
    add.rules.register(Base, float, 'value')
    assert interface['numeric'].rules[Base, Base].parameters == ['value']
    assert len(interface['add'].rules) == 1
    assert interface['add'].rules[Base, float].parameters == ['value']
    for defined in CATEGORIES.values():
        context = defined['context']
        for k in defined['operations']:
            assert isinstance(interface[k], context)


def test_interface_update_rule(interface: operations.Interface):
    """Make sure we can independently update rules."""
    interface['numeric'].rules.register(Base, float, 'value')
    assert interface['add'].rules[(Base, float)].implemented
    interface['__add__'].rules[(Base, float)].suppress
    assert not interface['add'].rules[(Base, float)].implemented
    assert interface['numeric'].rules[(Base, float)].implemented


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

