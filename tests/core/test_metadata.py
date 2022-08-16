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


def test_dimensions_object():
    """Make sure axes names behave as expected in operations."""
    assert len(metadata.Axes()) == 0
    names = ['a', 'b', 'c']
    for i, name in enumerate(names, start=1):
        subset = names[:i]
        dimensions = metadata.Axes(*subset)
        assert len(dimensions) == i
        assert all(name in dimensions for name in subset)
        assert dimensions[i-1] == name


def test_dimensions_init():
    """Test various ways to initialize dimensions metadata."""
    names = ['a', 'b', 'c']
    assert len(metadata.Axes(names)) == 3
    assert len(metadata.Axes(*names)) == 3
    assert len(metadata.Axes([names])) == 3
    invalid = [
        [1, 2, 3],
        [[1], [2], [3]],
        [['a'], ['b'], ['c']],
    ]
    for case in invalid:
        with pytest.raises(TypeError):
            metadata.Axes(*case)


def test_dimensions_operators():
    """Test built-in operations on dimensions metadata."""
    xy = metadata.Axes('x', 'y')
    yz = metadata.Axes('y', 'z')
    zw = metadata.Axes('z', 'w')
    pairs = {
        (xy, xy): {
            standard.add: xy,
            standard.sub: xy,
            standard.mul: xy,
            standard.truediv: xy,
            standard.pow: TypeError,
        },
        # TODO: continue for (xy, yz), (yz, zw), and (xy, zw).
    }
    # Below are just examples. A more rigorous test should check all instances
    # under unary arithmetic operations and all pairs of instances under binary
    # numeric operations.

    # addition and subtraction are only valid on the same dimensions
    for opr in (standard.add, standard.sub):
        assert opr(xy, xy) == xy
        with pytest.raises(TypeError):
            opr(xy, yz)
    # multiplication and division should concatenate unique dimensions
    for opr in (standard.mul, standard.truediv):
        assert opr(xy, xy) == xy
        assert opr(xy, yz) == metadata.Axes('x', 'y', 'z')
        assert opr(xy, zw) == metadata.Axes('x', 'y', 'z', 'w')
    # exponentiation is not valid
    with pytest.raises(TypeError):
        pow(xy, xy)
    # unary arithmetic operations should preserve dimensions
    for instance in (xy, yz, zw):
        for opr in (abs, standard.pos, standard.neg):
            assert opr(instance) == instance
    # cast and comparison operations don't affect dimensions


def test_dimensions_merge():
    """Test the ability to extract unique dimensions in order."""
    xy = metadata.Axes('x', 'y')
    yz = metadata.Axes('y', 'z')
    zw = metadata.Axes('z', 'w')
    assert xy.merge(xy) == metadata.Axes('x', 'y')
    assert xy.merge(yz) == metadata.Axes('x', 'y', 'z')
    assert yz.merge(xy) == metadata.Axes('y', 'z', 'x')
    assert xy.merge(zw) == metadata.Axes('x', 'y', 'z', 'w')
    assert zw.merge(xy) == metadata.Axes('z', 'w', 'x', 'y')
    assert yz.merge(zw) == metadata.Axes('y', 'z', 'w')
    assert zw.merge(yz) == metadata.Axes('z', 'w', 'y')
    assert xy.merge(yz, zw) == metadata.Axes('x', 'y', 'z', 'w')
    assert xy.merge(1.1) == xy
    assert xy.merge(yz, 1.1) == metadata.Axes('x', 'y', 'z')


def test_name():
    """Test the attribute representing a data quantity's name."""
    name = metadata.Name('a', 'A')
    assert len(name) == 2
    assert sorted(name) == ['A', 'a']
    assert all(i in name for i in ('a', 'A'))


def test_name_builtin():
    """Test operations between a name metadata object and a built-in object."""
    original = metadata.Name('a', 'A')
    values = ['2', 2]
    # Addition and subtraction require two instances.
    cases = {
        standard.add: ' + ',
        standard.sub: ' - ',
    }
    for method in cases:
        for value in values:
            with pytest.raises(TypeError):
                method(original, value)
            with pytest.raises(TypeError):
                method(value, original)
    # Multiplication, division, and exponentiation are valid with numbers.
    cases = {
        standard.mul: ' * ',
        standard.truediv: ' / ',
        pow: '^',
    }
    for method, s in cases.items():
        # Values not equivalent to 1 should appear.
        for value in values:
            result = method(original, value)
            assert isinstance(result, metadata.Name)
            updated = metadata.Name(*[f'{i}{s}{value}' for i in original])
            assert result == updated
            result = method(value, original)
            assert isinstance(result, metadata.Name)
            updated = metadata.Name(*[f'{value}{s}{i}' for i in original])
            assert result == updated
        # Values equivalent to 1 should not appear.
        for value in ['1', 1]:
            result = method(original, value)
            assert isinstance(result, metadata.Name)
            assert result is not original
            assert result == original
            result = method(value, original)
            assert isinstance(result, metadata.Name)
            assert result is not original
            assert result == original
    # TODO: multiplcation, division, and exponentiation by 0


def test_name_name():
    """Test operations between two name metadata objects."""
    name = metadata.Name('a', 'A')
    cases = {
        standard.add: ' + ',
        standard.sub: ' - ',
        standard.mul: ' * ',
        standard.truediv: ' / ',
    }
    for method, s in cases.items():
        other = metadata.Name('b', 'B')
        expected = metadata.Name(*[f'{i}{s}{j}' for i in name for j in other])
        assert method(name, other) == expected


def test_same_name():
    """Test operations on a name metadata object with itself."""
    name = metadata.Name('a', 'A')
    additive = {
        standard.add: ' + ',
        standard.sub: ' - ',
    }
    multiplicative = {
        standard.mul: ' * ',
        standard.truediv: ' / ',
    }
    for method in additive:
        assert method(name, name) == name
    for method, symbol in multiplicative.items():
        expected = metadata.Name(*[f'{i}{symbol}{i}' for i in name])
        assert method(name, name) == expected


def test_empty_names():
    """Make sure operations on empty names produce empty results."""
    n0 = metadata.Name('')
    n1 = metadata.Name('')
    cases = {
        standard.add: ' + ',
        standard.sub: ' - ',
        standard.mul: ' * ',
        standard.truediv: ' / ',
    }
    for method in cases:
        assert method(n0, n1) == metadata.Name('')

