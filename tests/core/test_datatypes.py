import operator
import math
import numbers
import typing

import pytest

from goats.core import datatypes
from goats.core import measurable
from goats.core import metadata


@pytest.mark.scalar
def test_scalar_scalar_comparisons():
    """Test comparisons between two scalars."""
    value = 2.0
    unit = 'm'
    scalar = datatypes.Scalar(value, unit=unit)
    cases = [
        (operator.lt, value + 1),
        (operator.le, value + 1),
        (operator.le, value),
        (operator.gt, value - 1),
        (operator.ge, value - 1),
        (operator.ge, value),
    ]
    for case in cases:
        opr, v = case
        assert opr(scalar, datatypes.Scalar(v, unit=unit))
        with pytest.raises(ValueError):
            opr(scalar, datatypes.Scalar(v, unit='J'))
    assert scalar == datatypes.Scalar(value, unit=unit)
    assert scalar != datatypes.Scalar(value+1, unit=unit)
    assert scalar != datatypes.Scalar(value, unit='J')


@pytest.mark.scalar
def test_scalar_number_comparisons():
    """Test comparisons between a scalar and a number."""
    value = 2.0
    unit = 'm'
    scalar = datatypes.Scalar(value, unit=unit)
    cases = [
        (operator.lt, operator.gt, value + 1),
        (operator.le, operator.ge, value + 1),
        (operator.le, operator.ge, value),
        (operator.eq, operator.eq, value),
        (operator.ne, operator.ne, value + 1),
        (operator.gt, operator.lt, value - 1),
        (operator.ge, operator.le, value - 1),
        (operator.ge, operator.le, value),
    ]
    for case in cases:
        fwd, rev, v = case
        assert fwd(scalar, v)
        assert rev(v, scalar)


@pytest.mark.scalar
def test_scalar_cast():
    """Test numeric casting operations on a scalar."""
    value = 2.0
    scalar = datatypes.Scalar(value, unit='m')
    for dtype in {int, float}:
        number = dtype(scalar)
        assert isinstance(number, dtype)
        assert number == dtype(value)


@pytest.mark.scalar
def test_scalar_unary():
    """Test unary arithmetic operations on a scalar."""
    value = 2.0
    unit = 'm'
    scalar = datatypes.Scalar(value, unit=unit)
    oprs = [
        operator.neg,
        operator.pos,
        abs,
        round,
        math.trunc,
        math.floor,
        math.ceil,
    ]
    for opr in oprs:
        result = opr(scalar)
        assert result == datatypes.Scalar(opr(value), unit=unit)


@pytest.mark.scalar
def test_scalar_binary():
    """Test binary arithmetic operations on a scalar."""
    cases = [
        (2.0, 'm'),
        (3.5, 'm'),
        (2.0, 'J'),
    ]
    instances = {
        args: datatypes.Scalar(args[0], unit=args[1])
        for args in cases
    }
    same_unit = cases[0], cases[1]
    diff_unit = cases[0], cases[2]
    scalars_same = [instances[k] for k in same_unit]
    scalars_diff = [instances[k] for k in diff_unit]
    values_same = [k[0] for k in same_unit]
    values_diff = [k[0] for k in diff_unit]
    scalar = scalars_same[0]
    value = values_same[1]

    # ADDITIVE
    oprs = [
        operator.add,
        operator.sub,
    ]
    unit = 'm'
    for opr in oprs:
        # between two instances with same unit
        expected = datatypes.Scalar(opr(*values_same), unit=unit)
        assert opr(*scalars_same) == expected
        # between an instance and a number
        # ...forward
        expected = datatypes.Scalar(opr(*values_same), unit=unit)
        assert opr(scalar, value) == expected
        # ...reverse
        expected = datatypes.Scalar(opr(*values_same[::-1]), unit=unit)
        assert opr(value, scalar) == expected
    # between two instances with different units
    for opr in oprs:
        with pytest.raises(metadata.UnitError):
            opr(*scalars_diff)

    # MULTIPLICATION
    opr = operator.mul
    # between two instances with same unit
    expected = datatypes.Scalar(opr(*values_same), unit='m^2')
    assert opr(*scalars_same) == expected
    # between an instance and a number
    # ...forward
    expected = datatypes.Scalar(opr(*values_same), unit='m')
    assert opr(scalar, value) == expected
    # reverse
    expected = datatypes.Scalar(opr(*values_same[::-1]), unit='m')
    assert opr(value, scalar) == expected
    # between two instances with different units
    expected = datatypes.Scalar(opr(*values_diff), unit='m * J')
    assert opr(*scalars_diff) == expected

    # DIVISION
    opr = operator.truediv
    # between two instances with same unit
    expected = datatypes.Scalar(opr(*values_same), unit='1')
    assert opr(*scalars_same) == expected
    # between an instance and a number
    # ...forward
    expected = datatypes.Scalar(opr(*values_same), unit='m')
    assert opr(scalar, value) == expected
    # reverse
    with pytest.raises(metadata.OperandTypeError):
        opr(value, scalar)
    # between two instances with different units
    expected = datatypes.Scalar(opr(*values_diff), unit='m / J')
    assert opr(*scalars_diff) == expected

    # EXPONENTIAL
    opr = operator.pow
    # between two instances with the same unit
    with pytest.raises(metadata.OperandTypeError):
        opr(*scalars_same)
    # between an instance and a number
    # ...forward
    expected = datatypes.Scalar(opr(*values_same), unit=f'm^{value}')
    assert opr(scalar, value) == expected
    # ...reverse
    with pytest.raises(metadata.OperandTypeError):
        opr(value, scalar)


@pytest.mark.scalar
def test_scalar_bitwise():
    """bitwise comparison is undefined"""
    scalar = datatypes.Scalar(2)
    with pytest.raises(TypeError):
        scalar & 1
        scalar | 1
        scalar ^ 1


@pytest.mark.vector
def test_vector_operators():
    """Test the updated operators on the vector object."""
    v0 = datatypes.Vector([3.0, 6.0], unit='m')
    v1 = datatypes.Vector([1.0, 3.0], unit='m')
    v2 = datatypes.Vector([1.0, 3.0], unit='J')
    assert v0 + v1 == datatypes.Vector([4.0, 9.0], unit='m')
    assert v0 - v1 == datatypes.Vector([2.0, 3.0], unit='m')
    assert v0 * v1 == datatypes.Vector([3.0, 18.0], unit='m^2')
    assert v0 / v1 == datatypes.Vector([3.0, 2.0], unit='1')
    assert v0 / v2 == datatypes.Vector([3.0, 2.0], unit='m / J')
    assert v0 ** 2 == datatypes.Vector([9.0, 36.0], unit='m^2')
    assert 10.0 * v0 == datatypes.Vector([30.0, 60.0], unit='m')
    assert v0 * 10.0 == datatypes.Vector([30.0, 60.0], unit='m')
    assert v0 / 10.0 == datatypes.Vector([0.3, 0.6], unit='m')
    with pytest.raises(metadata.OperandTypeError):
        1.0 / v0
    with pytest.raises(metadata.UnitError):
        v0 + v2


@pytest.mark.scalar
def test_scalar_display():
    """Test the results of str(self) and repr(self) for a scalar."""
    scalar = datatypes.Scalar(1.234, unit='m')
    assert str(scalar) == "1.234 [m]"
    assert repr(scalar).endswith("Scalar(1.234, unit='m')")
    scalar.convert('cm')
    assert str(scalar) == "123.4 [cm]"
    assert repr(scalar).endswith("Scalar(123.4, unit='cm')")


@pytest.mark.vector
def test_vector_display():
    """Test the results of str(self) and repr(self) for a vector."""
    vector = datatypes.Vector(1.234, unit='m')
    assert str(vector) == "[1.234] [m]"
    assert repr(vector).endswith("Vector([1.234], unit='m')")
    vector.convert('cm')
    assert str(vector) == "[123.4] [cm]"
    assert repr(vector).endswith("Vector([123.4], unit='cm')")


@pytest.mark.vector
def test_vector_init():
    """Test initializing with iterable and non-iterable values."""
    expected = sorted(datatypes.Vector([1.1], unit='m'))
    assert sorted(datatypes.Vector(1.1, unit='m')) == expected


@pytest.mark.scalar
def test_scalar_unit():
    """Get and set the unit on a Scalar."""
    check_units(datatypes.Scalar, 1, 'm', 'cm')


@pytest.mark.vector
def test_vector_unit():
    """Get and set the unit on a Vector."""
    check_units(datatypes.Vector, [1, 2], 'm', 'cm')


Obj = typing.Union[datatypes.Scalar, datatypes.Vector]


def check_units(
    obj: typing.Type[Obj],
    amount: measurable.Real,
    reference: str,
    new: str,
) -> None:
    """Extracted for testing the unit attribute on Measured subclasses."""
    original = obj(amount, unit=reference)
    assert original.unit == reference
    updated = original.convert(new)
    assert updated is original
    assert updated.unit == new
    factor = metadata.Unit(new) // metadata.Unit(reference)
    assert updated == obj(rescale(amount, factor), unit=new)
    assert obj(amount).unit == '1'


def rescale(amount, factor):
    """Multiply amount by factor."""
    if isinstance(amount, numbers.Number):
        return factor * amount
    if isinstance(amount, typing.Iterable):
        return [factor * value for value in amount]


@pytest.mark.xfail
def test_assumption():
    """Test the object that represents a physical assumption."""
    values = [1.0, 2.0]
    unit = 'm'
    aliases = 'this', 'a0'
    assumption = datatypes.Assumption(values, unit, *aliases)
    assert assumption.unit == unit
    assert all(alias in assumption.name for alias in aliases)
    scalars = [datatypes.Scalar(value, unit) for value in values]
    assert assumption[:] == scalars
    converted = assumption.convert('cm')
    assert converted.unit == 'cm'
    assert converted[:] == [100.0 * scalar for scalar in scalars]


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
            operator.add: xy,
            operator.sub: xy,
            operator.mul: xy,
            operator.truediv: xy,
            operator.pow: TypeError,
        },
        # TODO: continue for (xy, yz), (yz, zw), and (xy, zw).
    }
    # Below are just examples. A more rigorous test should check all instances
    # under unary arithmetic operations and all pairs of instances under binary
    # numeric operations.

    # addition and subtraction are only valid on the same dimensions
    for opr in (operator.add, operator.sub):
        assert opr(xy, xy) == xy
        with pytest.raises(TypeError):
            opr(xy, yz)
    # multiplication and division should concatenate unique dimensions
    for opr in (operator.mul, operator.truediv):
        assert opr(xy, xy) == xy
        assert opr(xy, yz) == metadata.Axes('x', 'y', 'z')
        assert opr(xy, zw) == metadata.Axes('x', 'y', 'z', 'w')
    # exponentiation is not valid
    with pytest.raises(TypeError):
        pow(xy, xy)
    # unary arithmetic operations should preserve dimensions
    for instance in (xy, yz, zw):
        for opr in (abs, operator.pos, operator.neg):
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
    name = metadata.Name('a', 'A')
    others = ['2', 2]
    # Addition and subtraction require two instances.
    cases = {
        operator.add: ' + ',
        operator.sub: ' - ',
    }
    for method in cases:
        for other in others:
            with pytest.raises(TypeError):
                method(name, other)
            with pytest.raises(TypeError):
                method(other, name)
    # Multiplication, division, and exponentiation are valid with numbers.
    cases = {
        operator.mul: ' * ',
        operator.truediv: ' / ',
        pow: '^',
    }
    for method, s in cases.items():
        for other in others:
            expected = metadata.Name(*[f'{i}{s}{other}' for i in name])
            assert method(name, other) == expected
            expected = metadata.Name(*[f'{other}{s}{i}' for i in name])
            assert method(other, name) == expected


def test_name_name():
    """Test operations between two name metadata objects."""
    name = metadata.Name('a', 'A')
    cases = {
        operator.add: ' + ',
        operator.sub: ' - ',
        operator.mul: ' * ',
        operator.truediv: ' / ',
    }
    for method, s in cases.items():
        other = metadata.Name('b', 'B')
        expected = metadata.Name(*[f'{i}{s}{j}' for i in name for j in other])
        assert method(name, other) == expected


def test_same_name():
    """Test operations on a name metadata object with itself."""
    name = metadata.Name('a', 'A')
    additive = {
        operator.add: ' + ',
        operator.sub: ' - ',
    }
    multiplicative = {
        operator.mul: ' * ',
        operator.truediv: ' / ',
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
        operator.add: ' + ',
        operator.sub: ' - ',
        operator.mul: ' * ',
        operator.truediv: ' / ',
    }
    for method in cases:
        assert method(n0, n1) == metadata.Name('')

