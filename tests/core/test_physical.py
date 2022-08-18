import operator
import math
import numbers
import typing

import pytest

from goats.core import physical
from goats.core import measurable
from goats.core import metadata


@pytest.mark.scalar
def test_scalar_scalar_comparisons():
    """Test comparisons between two scalars."""
    value = 2.0
    unit = 'm'
    scalar = physical.Scalar(value, unit=unit)
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
        assert opr(scalar, physical.Scalar(v, unit=unit))
        with pytest.raises(ValueError):
            opr(scalar, physical.Scalar(v, unit='J'))
    assert scalar == physical.Scalar(value, unit=unit)
    assert scalar != physical.Scalar(value+1, unit=unit)
    assert scalar != physical.Scalar(value, unit='J')


@pytest.mark.scalar
def test_scalar_number_comparisons():
    """Test comparisons between a scalar and a number."""
    value = 2.0
    unit = 'm'
    scalar = physical.Scalar(value, unit=unit)
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
    scalar = physical.Scalar(value, unit='m')
    for dtype in {int, float}:
        number = dtype(scalar)
        assert isinstance(number, dtype)
        assert number == dtype(value)


@pytest.mark.scalar
def test_scalar_unary():
    """Test unary arithmetic operations on a scalar."""
    value = 2.0
    unit = 'm'
    scalar = physical.Scalar(value, unit=unit)
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
        assert result == physical.Scalar(opr(value), unit=unit)


@pytest.mark.scalar
def test_scalar_binary():
    """Test binary arithmetic operations on a scalar."""
    cases = [
        (2.0, 'm'),
        (3.5, 'm'),
        (2.0, 'J'),
    ]
    instances = {
        args: physical.Scalar(args[0], unit=args[1])
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
        expected = physical.Scalar(opr(*values_same), unit=unit)
        assert opr(*scalars_same) == expected
        # between an instance and a number
        # ...forward
        with pytest.raises(TypeError):
            opr(scalar, value)
        # ...reverse
        with pytest.raises(TypeError):
            opr(value, scalar)
    # between two instances with different units
    for opr in oprs:
        with pytest.raises(metadata.UnitError):
            opr(*scalars_diff)

    # MULTIPLICATION
    opr = operator.mul
    # between two instances with same unit
    expected = physical.Scalar(opr(*values_same), unit='m^2')
    assert opr(*scalars_same) == expected
    # between an instance and a number
    # ...forward
    expected = physical.Scalar(opr(*values_same), unit='m')
    assert opr(scalar, value) == expected
    # reverse
    expected = physical.Scalar(opr(*values_same[::-1]), unit='m')
    assert opr(value, scalar) == expected
    # between two instances with different units
    expected = physical.Scalar(opr(*values_diff), unit='m * J')
    assert opr(*scalars_diff) == expected

    # DIVISION
    opr = operator.truediv
    # between two instances with same unit
    expected = physical.Scalar(opr(*values_same), unit='1')
    assert opr(*scalars_same) == expected
    # between an instance and a number
    # ...forward
    expected = physical.Scalar(opr(*values_same), unit='m')
    assert opr(scalar, value) == expected
    # reverse
    with pytest.raises(metadata.OperandTypeError):
        opr(value, scalar)
    # between two instances with different units
    expected = physical.Scalar(opr(*values_diff), unit='m / J')
    assert opr(*scalars_diff) == expected

    # EXPONENTIAL
    opr = operator.pow
    # between two instances with the same unit
    with pytest.raises(metadata.OperandTypeError):
        opr(*scalars_same)
    # between an instance and a number
    # ...forward
    expected = physical.Scalar(opr(*values_same), unit=f'm^{value}')
    assert opr(scalar, value) == expected
    # ...reverse
    with pytest.raises(metadata.OperandTypeError):
        opr(value, scalar)


@pytest.mark.scalar
def test_scalar_bitwise():
    """bitwise comparison is undefined"""
    scalar = physical.Scalar(2)
    with pytest.raises(TypeError):
        scalar & 1
        scalar | 1
        scalar ^ 1


@pytest.mark.vector
def test_vector_operators():
    """Test the updated operators on the vector object."""
    v0 = physical.Vector([3.0, 6.0], unit='m')
    v1 = physical.Vector([1.0, 3.0], unit='m')
    v2 = physical.Vector([1.0, 3.0], unit='J')
    assert v0 + v1 == physical.Vector([4.0, 9.0], unit='m')
    assert v0 - v1 == physical.Vector([2.0, 3.0], unit='m')
    assert v0 * v1 == physical.Vector([3.0, 18.0], unit='m^2')
    assert v0 / v1 == physical.Vector([3.0, 2.0], unit='1')
    assert v0 / v2 == physical.Vector([3.0, 2.0], unit='m / J')
    assert v0 ** 2 == physical.Vector([9.0, 36.0], unit='m^2')
    assert 10.0 * v0 == physical.Vector([30.0, 60.0], unit='m')
    assert v0 * 10.0 == physical.Vector([30.0, 60.0], unit='m')
    assert v0 / 10.0 == physical.Vector([0.3, 0.6], unit='m')
    with pytest.raises(metadata.OperandTypeError):
        1.0 / v0
    with pytest.raises(metadata.UnitError):
        v0 + v2


@pytest.mark.scalar
def test_scalar_display():
    """Test the results of str(self) and repr(self) for a scalar."""
    scalar = physical.Scalar(1.234, unit='m')
    assert str(scalar) == "1.234 [m]"
    assert repr(scalar).endswith("Scalar(1.234, unit='m')")
    scalar['cm']
    assert str(scalar) == "123.4 [cm]"
    assert repr(scalar).endswith("Scalar(123.4, unit='cm')")


@pytest.mark.vector
def test_vector_display():
    """Test the results of str(self) and repr(self) for a vector."""
    vector = physical.Vector(1.234, unit='m')
    assert str(vector) == "[1.234] [m]"
    assert repr(vector).endswith("Vector([1.234], unit='m')")
    vector['cm']
    assert str(vector) == "[123.4] [cm]"
    assert repr(vector).endswith("Vector([123.4], unit='cm')")


@pytest.mark.vector
def test_vector_init():
    """Test initializing with various arguments."""
    expected = physical.Vector([1.1], unit='m')
    assert physical.Vector(1.1, unit='m') == expected
    measurement = measurable.Measurement([1.1], unit='m')
    assert physical.Vector(measurement) == expected


@pytest.mark.scalar
def test_scalar_unit():
    """Get and set the unit on a Scalar."""
    check_units(physical.Scalar, 1, 'm', 'cm')


@pytest.mark.vector
def test_vector_unit():
    """Get and set the unit on a Vector."""
    check_units(physical.Vector, [1, 2], 'm', 'cm')


Obj = typing.Union[physical.Scalar, physical.Vector]


def check_units(
    obj: typing.Type[Obj],
    amount: measurable.Real,
    reference: str,
    new: str,
) -> None:
    """Extracted for testing the unit attribute on Measured subclasses."""
    original = obj(amount, unit=reference)
    assert original.unit == reference
    updated = original[new]
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


def test_scalar():
    """Test the function that converts an object to a scalar, if possible."""
    reference = physical.Scalar(1.1, unit='m')
    valid = [
        physical.Vector([1.1], unit='m'),
        measurable.Measurement([1.1], unit='m'),
        (1.1, 'm'),
    ]
    for case in valid:
        assert physical.scalar(case) == reference
    error = [
        physical.Vector([1.1, 2.3], unit='m'),
        measurable.Measurement([1.1, 2.3], unit='m'),
        (1.1, 2.3, 'm'),
    ]
    for case in error:
        with pytest.raises(ValueError):
            physical.scalar(case)