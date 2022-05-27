import math
import numbers
import operator as standard
import typing

import pytest

from goats.core import operations


def test_rule_comparison():
    """Test comparisons between operand-update rules."""
    rule = operations.Rule((numbers.Complex, numbers.Real))
    true = [
        [standard.ne, (complex, float)],
        [standard.gt, (complex, float)],
        [standard.ge, (complex, float)],
        [standard.gt, (float, float)],
        [standard.ge, (float, float)],
        [standard.gt, (float, int)],
        [standard.ge, (float, int)],
        [standard.gt, (numbers.Complex, numbers.Real)],
        [standard.ge, (numbers.Complex, numbers.Real)],
        [standard.eq, (numbers.Complex, numbers.Real)],
    ]
    for method, types in true:
        assert method(rule, types)
    false = [
        [standard.eq, (complex, float)],
        [standard.eq, (numbers.Real, numbers.Real)]
    ]
    for method, types in false:
        assert not method(rule, types)


def test_rule_len():
    """Test the length of an operator rule."""
    for types in (int, [int, float]):
        for parameters in ([], ['a'], ['a', 'b'], ['a', 'b', 'c']):
            assert len(operations.Rule(types, *parameters)) == len(parameters)


def test_rule_contains():
    """Check for a type in an operator rule."""
    assert int in operations.Rule(int)
    assert float not in operations.Rule(int)
    rule = operations.Rule([int, float], 'a', 'b')
    for this in [int, float, 'a', 'b']:
        assert this in rule


def test_rule_ignore():
    """Allow a rule to ignore certain parameters."""
    rule = operations.Rule([int, float], 'a', 'b', 'c')
    assert rule.parameters == ['a', 'b', 'c']
    rule.ignore('b')
    assert sorted(rule.parameters) == sorted(['a', 'c'])


def test_rules_register():
    """Test the ability to register operand rules."""
    default = ['a', 'b', 'c']
    rules = operations.Rules(*default)
    assert not rules
    assert rules.ntypes is None
    rules.register([int, float], 'a', 'b')
    assert rules
    assert rules.ntypes == 2
    assert rules[(int, float)].parameters == ['a', 'b']
    rules.register([float, float])
    assert rules[(float, float)].parameters == default
    rules.register([int, int], None)
    assert not rules[(int, int)].parameters
    with pytest.raises(operations.NTypesError):
        rules.register(int, 'a')


def test_rules_modify():
    """Test the ability to modify parameters in a rule."""
    rules = operations.Rules()
    rules.register([float, float], 'd', 'e', 'f')
    assert sorted(rules[(float, float)].parameters) == sorted(['d', 'e', 'f'])
    rules.modify([float, float], 'a', 'b', 'c')
    assert sorted(rules[(float, float)].parameters) == sorted(['a', 'b', 'c'])
    rules.modify([float, float], 'a', 'b', mode='restrict')
    assert sorted(rules[(float, float)].parameters) == sorted(['a', 'b'])
    rules.modify([float, float], 'a', mode='remove')
    assert rules[(float, float)].parameters == ['b']
    with pytest.raises(ValueError):
        rules.modify([float, float], 'a', mode='restrict')


def test_rules_suppress():
    """Test the ability to suppress a given rule."""
    default = ['a', 'b', 'c']
    rules = operations.Rules(*default)
    rules.register([float, float])
    assert rules[(float, float)].parameters == rules.default
    rules.suppress([float, float])
    assert rules[(float, float)].parameters == []


def test_rules_copy():
    """Test the ability to copy an instance."""
    default = ['a', 'b', 'c']
    init = [operations.Rule(int, 'a'), operations.Rule(float, 'a', 'b')]
    rules = operations.Rules(*default, rules=init)
    copied = rules.copy()
    assert copied == rules
    assert copied is not rules


def test_object_idempotence():
    """Create an `Object` instance from another instance."""
    a = operations.Object(1)
    b = operations.Object(a)
    assert isinstance(b, operations.Object)
    assert b is not a
    assert b == a
    assert b.parameters == a.parameters


def test_objects():
    """Test the `Objects` class."""
    inputs = ['a', 1, 2.3]
    objects = operations.Objects(*inputs)
    assert objects.types == (str, int, float)
    subset = objects[:]
    assert isinstance(subset, operations.Objects)
    assert subset == objects
    for index in [-2, -1, 0, 1, 2]:
        assert objects[index] == inputs[index]


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


T = typing.TypeVar('T')


def build(__type: typing.Type[T]) -> typing.List[T]:
    """Create and initialize instances of a class for tests."""
    inputs = [
        (1, 'A'),
        (2, 'A'),
        (2, 'B'),
    ]
    return [__type(*args) for args in inputs]


def test_objects_agree():
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
            operands = operations.Objects(*args)
            assert operands.agree(name)
    for instance in instances: # single object trivially agrees with itself
        assert operations.Objects(instance).agree(same.keys())
    different = {
        'value': [instances[0], instances[1]],
        'info': [instances[1], instances[2]],
    }
    for name, args in different.items():
        assert not operations.Objects(*args).agree(name)


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
    operation.rules.register([Base, Base], 'value')
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
    operation.rules.register([Base, Base], 'value')
    operator = operation.apply(builtin)
    expected = Base(
        builtin(instances[0].value, instances[1].value),
        instances[0].info,
    )
    assert operator(instances[0], instances[1]) == expected
    with pytest.raises(operations.OperandTypeError):
        operator(instances[0], instances[2])


CAST = {
    'int': int,
    'float': float,
}
UNARY = {
    'abs': standard.abs,
    'neg': standard.neg,
    'pos': standard.pos,
    'ceil': math.ceil,
    'floor': math.floor,
    'trunc': math.trunc,
    'round': round,
}
NUMERIC = {
    'add': standard.add,
    'sub': standard.sub,
    'mul': standard.mul,
    'truediv': standard.truediv,
    'pow': pow,
}
COMPARISON = {
    'lt': standard.lt,
    'le': standard.le,
    'gt': standard.gt,
    'ge': standard.ge,
    'eq': standard.eq,
    'ne': standard.ne,
}


def test_cast_interface():
    """Test cast operations via the module interface."""
    interface = operations.Interface('value', 'info')
    operation = interface.cast
    operation.rules.register(Base, 'value')
    instances = build(Base)
    for builtin in CAST.values():
        operator = operation.apply(builtin)
        for instance in instances:
            expected = builtin(instance.value)
            assert operator(instance) == expected


def test_unary_interface():
    """Test unary operations via the module interface."""
    interface = operations.Interface('value', 'info')
    operation = interface.unary
    operation.rules.register(Base, 'value')
    instances = build(Base)
    for builtin in UNARY.values():
        operator = operation.apply(builtin)
        for instance in instances:
            expected = Base(builtin(instance.value), instance.info)
            assert operator(instance) == expected


def test_comparison_interface():
    """Test comparison operations via the module interface."""
    interface = operations.Interface('value', 'info')
    operation = interface.comparison
    operation.rules.register([Base, Base], 'value')
    instances = build(Base)
    targets = instances[0], instances[1]
    for builtin in COMPARISON.values():
        operator = operation.apply(builtin)
        assert operator(*targets) == builtin(*[c.value for c in targets])
        with pytest.raises(operations.OperandTypeError):
            operator(instances[0], instances[2])


def test_numeric_interface():
    """Test numeric operations via the module interface."""
    interface = operations.Interface('value', 'info')
    operation = interface.numeric
    operation.rules.register([Base, Base], 'value')
    instances = build(Base)
    targets = instances[0], instances[1]
    for builtin in NUMERIC.values():
        operator = operation.apply(builtin)
        expected = Base(
            builtin(*[c.value for c in targets]),
            targets[0].info,
        )
        assert operator(*targets) == expected
        with pytest.raises(operations.OperandTypeError):
            operator(instances[0], instances[2])


class Mixin:
    """A mixin class that provides operator implementations."""

    rules = operations.Rules('value', 'info')

    cast = operations.Cast(rules)
    cast.rules.register(Base, 'value')

    comparison = operations.Comparison(rules)
    comparison.rules.register([Base, Base], 'value')

    unary = operations.Unary(rules)
    unary.rules.register(Base, 'value')

    numeric = operations.Numeric(rules)
    numeric.rules.register([Base, Base], 'value')

    __int__ = cast.apply(int)
    __float__ = cast.apply(float)
    __lt__ = comparison.apply(standard.lt)
    __le__ = comparison.apply(standard.le)
    __gt__ = comparison.apply(standard.gt)
    __ge__ = comparison.apply(standard.ge)
    __eq__ = comparison.apply(standard.eq)
    __ne__ = comparison.apply(standard.ne)
    __abs__ = unary.apply(standard.abs)
    __pos__ = unary.apply(standard.pos)
    __neg__ = unary.apply(standard.neg)
    __ceil__ = unary.apply(math.ceil)
    __floor__ = unary.apply(math.floor)
    __trunc__ = unary.apply(math.trunc)
    __round__ = unary.apply(round)
    __add__ = numeric.apply(standard.add)
    __radd__ = numeric.apply(standard.add, 'reverse')
    __sub__ = numeric.apply(standard.sub)
    __rsub__ = numeric.apply(standard.sub, 'reverse')
    __mul__ = numeric.apply(standard.mul)
    __rmul__ = numeric.apply(standard.mul, 'reverse')
    __truediv__ = numeric.apply(standard.truediv)
    __rtruediv__ = numeric.apply(standard.truediv, 'reverse')
    __pow__ = numeric.apply(pow)
    __rpow__ = numeric.apply(pow, 'reverse')


class MixedIn(Mixin, Base):
    """A test class that uses mixin custom operators."""


def test_cast_mixin():
    """Test the use of the mixin cast operators."""
    instances = build(MixedIn)
    for builtin in CAST.values():
        for instance in instances:
            assert builtin(instance) == builtin(instance.value)


def test_unary_mixin():
    """Test the use of the mixin unary operators."""
    instances = build(MixedIn)
    for builtin in UNARY.values():
        for instance in instances:
            expected = MixedIn(builtin(instance.value), instance.info)
            assert builtin(instance) == expected


def test_comparison_mixin():
    """Test the use of the mixin comparison operators."""
    instances = build(MixedIn)
    targets = instances[0], instances[1]
    for builtin in COMPARISON.values():
        assert builtin(*targets) == builtin(*[c.value for c in targets])
        with pytest.raises(operations.OperandTypeError):
            builtin(instances[0], instances[2])


def test_numeric_mixin():
    """Test the use of the mixin numeric operators."""
    instances = build(MixedIn)
    targets = instances[0], instances[1]
    for builtin in NUMERIC.values():
        expected = MixedIn(
            builtin(*[c.value for c in targets]),
            targets[0].info,
        )
        assert builtin(*targets) == expected
        with pytest.raises(operations.OperandTypeError):
            builtin(instances[0], instances[2])


rules = operations.Rules('value', 'info')

cast = operations.Cast(rules)
cast.rules.register(Base, 'value')

comparison = operations.Comparison(rules)
comparison.rules.register([Base, Base], 'value')

unary = operations.Unary(rules)
unary.rules.register(Base, 'value')

numeric = operations.Numeric(rules)
numeric.rules.register([Base, Base], 'value')


class Defined(Base):
    """A test class that defines custom operators."""

    __int__ = cast.apply(int)
    __float__ = cast.apply(float)
    __lt__ = comparison.apply(standard.lt)
    __le__ = comparison.apply(standard.le)
    __gt__ = comparison.apply(standard.gt)
    __ge__ = comparison.apply(standard.ge)
    __eq__ = comparison.apply(standard.eq)
    __ne__ = comparison.apply(standard.ne)
    __abs__ = unary.apply(standard.abs)
    __pos__ = unary.apply(standard.pos)
    __neg__ = unary.apply(standard.neg)
    __ceil__ = unary.apply(math.ceil)
    __floor__ = unary.apply(math.floor)
    __trunc__ = unary.apply(math.trunc)
    __round__ = unary.apply(round)
    __add__ = numeric.apply(standard.add)
    __radd__ = numeric.apply(standard.add, 'reverse')
    __sub__ = numeric.apply(standard.sub)
    __rsub__ = numeric.apply(standard.sub, 'reverse')
    __mul__ = numeric.apply(standard.mul)
    __rmul__ = numeric.apply(standard.mul, 'reverse')
    __truediv__ = numeric.apply(standard.truediv)
    __rtruediv__ = numeric.apply(standard.truediv, 'reverse')
    __pow__ = numeric.apply(pow)
    __rpow__ = numeric.apply(pow, 'reverse')


def test_cast_defined():
    """Test the use of the defined cast operators."""
    instances = build(Defined)
    for builtin in CAST.values():
        for instance in instances:
            assert builtin(instance) == builtin(instance.value)


def test_unary_defined():
    """Test the use of the defined unary operators."""
    instances = build(Defined)
    for builtin in UNARY.values():
        for instance in instances:
            expected = Defined(builtin(instance.value), instance.info)
            assert builtin(instance) == expected


def test_comparison_defined():
    """Test the use of the defined comparison operators."""
    instances = build(Defined)
    targets = instances[0], instances[1]
    for builtin in COMPARISON.values():
        assert builtin(*targets) == builtin(*[c.value for c in targets])
        with pytest.raises(operations.OperandTypeError):
            builtin(instances[0], instances[2])


def test_numeric_defined():
    """Test the use of the defined numeric operators."""
    instances = build(Defined)
    targets = instances[0], instances[1]
    for builtin in NUMERIC.values():
        expected = Defined(
            builtin(*[c.value for c in targets]),
            targets[0].info,
        )
        assert builtin(*targets) == expected
        with pytest.raises(operations.OperandTypeError):
            builtin(instances[0], instances[2])

