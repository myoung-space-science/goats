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



def test_object_idempotence():
    """Create an `Object` instance from another instance."""
    a = operations.Object(1)
    b = operations.Object(a)
    assert isinstance(b, operations.Object)
    assert b is not a
    assert b == a
    assert b.parameters == a.parameters


def test_operands_init():
    """Initialize an `Operands` instance in various ways."""
    with pytest.raises(TypeError):
        operations.Operands()
    assert operations.Operands(1).reference == operations.Object(1)
    assert operations.Operands(1, 2.3).types == (int, float)


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


class Class:
    """A test class."""
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


@pytest.fixture
def instances():
    """Reusable instances of the test class."""
    return {
        'c0': Class(1, 'same'),
        'c1': Class(3.3, 'same'),
        'c2': Class(3.3, 'different'),
    }


def test_cast_operation(instances: typing.Dict[str, Class]):
    """Test the implementation of type-cast operations."""
    builtin = int
    operation = operations.Operation(operations.Cast)
    operator = operation.implement(builtin)
    value = 3.3
    result = operator(value)
    assert isinstance(result, builtin)
    assert operator(value) == builtin(value)
    with pytest.raises(TypeError):
        operator(instances['c0'])
    operator.rules.register(Class, 'value')
    for instance in instances.values():
        assert operator(instance) == builtin(instance.value)


def test_unary_operation(instances: typing.Dict[str, Class]):
    """Test the implementation of unary arithmetic operations."""
    operation = operations.Operation(operations.Unary)
    builtin = round
    operator = operation.implement(builtin)
    value = 3.3
    assert operator(value) == builtin(value)
    with pytest.raises(TypeError):
        operator(instances['c0'])
    operator.rules.register(Class, 'value')
    for instance in instances.values():
        expected = Class(builtin(instance.value), instance.info)
        assert operator(instance) == expected


def test_comparison_operation(instances: typing.Dict[str, Class]):
    """Test the implementation of binary comparison operations."""
    operation = operations.Operation(operations.Comparison)
    builtin = standard.lt
    operator = operation.implement(builtin)
    values = 2, 4
    assert operator(*values) == builtin(*values)
    with pytest.raises(TypeError):
        operator(instances['c0'], instances['c1'])
    operator.rules.register([Class, Class], 'value')
    assert operator(instances['c0'], instances['c1'])
    assert not operator(instances['c1'], instances['c0'])
    with pytest.raises(operations.OperandTypeError):
        operator(instances['c0'], instances['c2'])


def test_numeric_operation(instances: typing.Dict[str, Class]):
    """Test the implementation of binary numeric operations."""
    operation = operations.Operation(operations.Numeric)
    builtin = standard.add
    operator = operation.implement(builtin)
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
    with pytest.raises(TypeError):
        operator(instances['c0'], instances['c1'])
    operator.rules.register([Class, Class], 'value')
    expected = Class(
        builtin(instances['c0'].value, instances['c1'].value),
        instances['c0'].info,
    )
    assert operator(instances['c0'], instances['c1']) == expected
    with pytest.raises(operations.OperandTypeError):
        operator(instances['c0'], instances['c2'])


def test_implementation():
    """Test the ability to implement an operation."""
    class Base:
        def __init__(
            self,
            __value: numbers.Real,
            unit: str,
            axes: typing.Iterable[str]=None,
        ) -> None:
            self.value = __value
            self.unit = unit
            self.axes = axes or []
            operators = operations.Interface(type(self), target='value')
        def __repr__(self) -> str:
            attrs = (
                str(self.value),
                str(self.unit),
                f"axes={self.axes}",
            )
            return f"{self.__class__.__qualname__}({', '.join(attrs)})"

    class OperatorsMixin:
        """"""

    class Empty(Base):
        """"""

    class Filled(Base, OperatorsMixin):
        """"""

    empty = Empty(1.2, 'u')
    value = 1.2
    for method in CAST.values():
        with pytest.raises(TypeError):
            assert method(empty)
    for method in UNARY.values():
        with pytest.raises(TypeError):
            assert method(empty)
    for name, method in COMPARISON.items():
        if name == 'eq':
            assert not method(empty, value)
        elif name == 'ne':
            assert method(empty, value)
        else:
            with pytest.raises(TypeError):
                assert method(empty, value)
    for method in NUMERIC.values():
        with pytest.raises(TypeError):
            assert method(empty, value)
        with pytest.raises(TypeError):
            assert method(value, empty)

    c0 = Filled(+1.2, 'u', axes=['x', 'y'])
    c1 = Filled(-3.4, 'u', axes=['x', 'y'])
    # assert c0 + c1 == Filled(c0.value + c1.value, 'u', axes=['x', 'y'])



