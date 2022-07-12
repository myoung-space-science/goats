import numbers
import operator as standard
import typing

import pytest

from goats.core import metadata


def test_operation_copy():
    """Test the ability to make a copy of an operation context."""
    operation = metadata.Operation(str)
    operation.supports(int, str)
    copied = operation.copy()
    assert copied == operation
    assert copied is not operation


def test_operation_types():
    """Add and remove operand type specifications."""
    operation = metadata.Operation(str)
    assert operation.supports(str)
    assert operation.supports(str, str)
    assert operation.supports(str, str, str)
    assert operation.supports(str, float)
    assert not operation.supports(int, float)
    operation.support(int, float)
    assert operation.supports(int, float)
    assert not operation.supports(str)
    assert operation.supports(str, str)
    assert not operation.supports(str, str, str)
    assert operation.supports(str, float)
    assert operation.supports(int, float)
    assert not operation.supports(float, int)
    operation.suppress(str, str)
    assert not operation.supports(str, str)
    operation.support(complex, float, symmetric=True)
    assert operation.support(complex, float)
    assert operation.support(float, complex)
    operation.suppress(complex, float, symmetric=True)
    assert not operation.supports(complex, float)
    assert not operation.supports(float, complex)


def test_operation_batch_types():
    """Test the ability to add or remove multiple type specifications."""
    operation = metadata.Operation(str)
    assert not operation.implemented
    user = [
        (int, float),
        (float, float),
        (int, int),
    ]
    operation.support(*user)
    assert operation.implemented
    for these in user:
        assert operation.supports(*these)


class Info:
    """Information about a value."""

    def __init__(self, arg) -> None:
        self._text = arg._text if isinstance(arg, Info) else arg

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


T = typing.TypeVar('T')


def build(__type: typing.Type[T]) -> typing.List[T]:
    """Create and initialize instances of a class for tests."""
    inputs = [
        (1, 'A'),
        (2, 'A'),
        (2, 'B'),
    ]
    return [__type(*args) for args in inputs]


def test_operator_factory():
    """Test the class that creates operators from operation contexts."""
    factory = metadata.OperatorFactory(Base)
    assert not factory.parameters
    factory.register('value', 'info')
    assert factory.parameters == ('value', 'info')
    assert isinstance(factory['mul'], metadata.Operation)
    assert factory['mul'] is factory['multiply']
    keys = ('mul', 'multiply')
    assert all(factory[key].supports(Base, float) for key in keys)
    factory['multiply'].suppress(Base, float)
    for key in keys:
        assert not factory[key].supports(Base, float)
        assert factory[key].supports(Base, Base)
        assert factory[key].supports(float, Base)


def test_unit_add():
    """Test the use of '+' between units."""
    apply_additive(standard.add)


def test_unit_sub():
    """Test the use of '-' between units."""
    apply_additive(standard.sub)


def apply_additive(opr):
    """Apply an additive operator between units."""
    meter = metadata.Unit('m')
    assert opr(meter, metadata.Unit('m')) == metadata.Unit('m')
    for number in [1, 1.0]:
        with pytest.raises(TypeError):
            opr(meter, number)
    for arg in ['J', '1']:
        with pytest.raises(metadata.DimensionMismatch):
            opr(meter, metadata.Unit(arg))
    with pytest.raises(metadata.ScaleMismatch):
        opr(meter, metadata.Unit('cm'))


