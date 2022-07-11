import itertools
import operator
import math
import numbers
import typing

import numpy
import numpy.typing
import pytest

from goats.core import datatypes
from goats.core import measurable
from goats.core import metric
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
        with pytest.raises(metric.UnitError):
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
    with pytest.raises(metric.UnitError):
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
    factor = metric.Unit(new) // metric.Unit(reference)
    assert updated == obj(rescale(amount, factor), unit=new)
    assert obj(amount).unit == '1'


def rescale(amount, factor):
    """Multiply amount by factor."""
    if isinstance(amount, numbers.Number):
        return factor * amount
    if isinstance(amount, typing.Iterable):
        return [factor * value for value in amount]


@pytest.mark.variable
def test_variable_display():
    """Test the results of printing a variable."""
    v = datatypes.Variable([1.2], unit='m', name='V', axes=['x'])
    assert str(v) == "'V': [1.2] [m] axes=['x']"
    assert repr(v).endswith("Variable([1.2], unit='m', name='V', axes=['x'])")


@pytest.mark.variable
def test_variable():
    """Test the object that represents a variable."""
    v0 = datatypes.Variable([3.0, 4.5], unit='m', axes=['x'])
    v1 = datatypes.Variable([[1.0], [2.0]], unit='J', axes=['x', 'y'])
    assert numpy.array_equal(v0, [3.0, 4.5])
    assert v0.unit == metric.Unit('m')
    assert list(v0.axes) == ['x']
    assert v0.naxes == 1
    assert numpy.array_equal(v1, [[1.0], [2.0]])
    assert v1.unit == metric.Unit('J')
    assert list(v1.axes) == ['x', 'y']
    assert v1.naxes == 2
    r = v0 + v0
    expected = [6.0, 9.0]
    assert numpy.array_equal(r, expected)
    assert r.unit == v0.unit
    r = v0 * v1
    expected = [[3.0 * 1.0], [4.5 * 2.0]]
    assert numpy.array_equal(r, expected)
    assert r.unit == metric.Unit('m * J')
    r = v0 / v1
    expected = [[3.0 / 1.0], [4.5 / 2.0]]
    assert numpy.array_equal(r, expected)
    assert r.unit == metric.Unit('m / J')
    r = v0 ** 2
    expected = [3.0 ** 2, 4.5 ** 2]
    assert numpy.array_equal(r, expected)
    assert r.unit == metric.Unit('m^2')
    reference = datatypes.Variable(v0)
    assert reference is not v0
    v0_cm = v0.convert('cm')
    assert v0_cm is v0
    expected = 100 * reference
    assert numpy.array_equal(v0_cm, expected)
    assert v0_cm.unit == metric.Unit('cm')
    assert v0_cm.axes == reference.axes


@pytest.fixture
def arr() -> typing.Dict[str, list]:
    """Arrays (lists of lists) for creating variables."""
    reference = [
        [+1.0, +2.0],
        [+2.0, -3.0],
        [-4.0, +6.0],
    ]
    samedims = [
        [+10.0, +20.0],
        [-20.0, -30.0],
        [+40.0, +60.0],
    ]
    sharedim = [
        [+4.0, -4.0, +4.0, -4.0],
        [-6.0, +6.0, -6.0, +6.0],
    ]
    different = [
        [+1.0, +2.0, +3.0, +4.0, +5.0],
        [-1.0, -2.0, -3.0, -4.0, -5.0],
        [+5.0, +4.0, +3.0, +2.0, +1.0],
        [-5.0, -4.0, -3.0, -2.0, -1.0],
    ]
    return {
        'reference': reference,
        'samedims': samedims,
        'sharedim': sharedim,
        'different': different,
    }

@pytest.fixture
def var(arr: typing.Dict[str, list]) -> typing.Dict[str, datatypes.Variable]:
    """A tuple of test variables."""
    reference = datatypes.Variable(
        arr['reference'].copy(),
        axes=('d0', 'd1'),
        unit='m',
    )
    samedims = datatypes.Variable(
        arr['samedims'].copy(),
        axes=('d0', 'd1'),
        unit='kJ',
    )
    sharedim = datatypes.Variable(
        arr['sharedim'].copy(),
        axes=('d1', 'd2'),
        unit='s',
    )
    different = datatypes.Variable(
        arr['different'].copy(),
        axes=('d2', 'd3'),
        unit='km/s',
    )
    return {
        'reference': reference,
        'samedims': samedims,
        'sharedim': sharedim,
        'different': different,
    }


def reduce(
    a: numpy.typing.ArrayLike,
    b: numpy.typing.ArrayLike,
    opr: typing.Callable,
    axes: typing.Iterable[typing.Iterable[str]]=None,
) -> list:
    """Create an array from `a` and `b` by applying `opr`.

    Parameters
    ----------
    a : array-like
        An array-like object with shape (I, J).

    b : array-like or real
        An array-like object with shape (P, Q) or a real number.

    opr : callable
        An operator that accepts arguments of the types of elements in `a` and
        `b` and returns a single value of any type.

    axes : iterable of iterables of strings, optional
        An two-element iterable containing the axes of `a` followed by the axes
        of `b`. The unique axes determine how this function reduces arrays `a`
        and `b`. This function will simply ignore `axes` if `b` is a number.

    Returns
    -------
    list
        A possibly nested list containing the element-wise result of `opr`. The
        shape of the equivalent array will be the 

    Notes
    -----
    This was created to help generalize tests of `Variable` binary arithmetic
    operators. The way in which it builds arrays is not especially Pythonic;
    instead, the goal is to indicate how the structure of the resultant array
    arises from the structure of the operands.
    """
    I = range(len(a))
    J = range(len(a[0]))
    if isinstance(b, (float, int)):
        return [
            # I x J
            [
                opr(a[i][j], b) for j in J
            ] for i in I
        ]
    P = range(len(b))
    Q = range(len(b[0]))
    a_axes, b_axes = axes
    if a_axes[0] == b_axes[0] and a_axes[1] == b_axes[1]:
        return [
            # I x J
            [
                # J
                opr(a[i][j], b[i][j]) for j in J
            ] for i in I
        ]
    if a_axes[0] == b_axes[0]:
        return [
            # I x J x Q
            [
                # J x Q
                [
                    # Q
                    opr(a[i][j], b[i][q]) for q in Q
                ] for j in J
            ] for i in I
        ]
    if a_axes[1] == b_axes[0]:
        return [
            # I x J x Q
            [
                # J x Q
                [
                    # Q
                    opr(a[i][j], b[j][q]) for q in Q
                ] for j in J
            ] for i in I
        ]
    return [
        # I x J x P x Q
        [
            # J x P x Q
            [
                # P x Q
                [
                    # Q
                    opr(a[i][j], b[p][q]) for q in Q
                ] for p in P
            ] for j in J
        ] for i in I
    ]


@pytest.mark.variable
def test_variable_mul_div(
    var: typing.Dict[str, datatypes.Variable],
    arr: typing.Dict[str, list],
) -> None:
    """Test the ability to multiply two datatypes.Variable instances."""
    groups = {
        '*': operator.mul,
        '/': operator.truediv,
    }
    cases = {
        'same axes': {
            'key': 'samedims',
            'axes': ['d0', 'd1'],
        },
        'one shared axis': {
            'key': 'sharedim',
            'axes': ['d0', 'd1', 'd2'],
        },
        'different axes': {
            'key': 'different',
            'axes': ['d0', 'd1', 'd2', 'd3'],
        },
    }
    v0 = var['reference']
    a0 = arr['reference']
    tests = itertools.product(groups.items(), cases.items())
    for (sym, opr), (name, case) in tests:
        msg = f"Failed for {name} with {opr}"
        v1 = var[case['key']]
        a1 = arr[case['key']]
        new = opr(v0, v1)
        assert isinstance(new, datatypes.Variable), msg
        expected = reduce(a0, a1, opr, axes=(v0.axes, v1.axes))
        assert numpy.array_equal(new, expected), msg
        assert sorted(new.axes) == case['axes'], msg
        algebraic = opr(v0.unit, v1.unit)
        formatted = f'({v0.unit}){sym}({v1.unit})'
        for unit in (algebraic, formatted):
            assert new.unit == unit, msg


@pytest.mark.variable
def test_variable_pow(
    var: typing.Dict[str, datatypes.Variable],
    arr: typing.Dict[str, list],
) -> None:
    """Test the ability to exponentiate a datatypes.Variable instance."""
    v0 = var['reference']
    p = 3
    new = v0 ** p
    assert isinstance(new, datatypes.Variable)
    expected = reduce(numpy.array(v0), p, pow)
    assert numpy.array_equal(new, expected)
    assert new.axes == var['reference'].axes
    algebraic = v0.unit ** p
    formatted = f'({v0.unit})^{p}'
    for unit in algebraic, formatted:
        assert new.unit == unit
    with pytest.raises(metadata.OperandTypeError):
        v0 ** arr['reference']


@pytest.mark.variable
def test_variable_add_sub(
    var: typing.Dict[str, datatypes.Variable],
    arr: typing.Dict[str, list],
) -> None:
    """Test the ability to add two datatypes.Variable instances."""
    v0 = var['reference']
    a0 = arr['reference']
    a1 = arr['samedims']
    v1 = datatypes.Variable(a1, unit=v0.unit, axes=v0.axes)
    v2 = datatypes.Variable(
        arr['different'],
        unit=v0.unit,
        axes=var['different'].axes,
    )
    for opr in (operator.add, operator.sub):
        msg = f"Failed for {opr}"
        new = opr(v0, v1)
        expected = reduce(a0, a1, opr, axes=(v0.axes, v1.axes))
        assert isinstance(new, datatypes.Variable), msg
        assert numpy.array_equal(new, expected), msg
        assert new.unit == v0.unit, msg
        assert new.axes == v0.axes, msg
        with pytest.raises(ValueError): # numpy broadcasting error
            opr(v0, v2)


@pytest.mark.variable
def test_variable_units(var: typing.Dict[str, datatypes.Variable]):
    """Test the ability to update a variable's unit."""
    v0 = var['reference']
    reference = datatypes.Variable(v0)
    v0_km = v0.convert('km')
    assert isinstance(v0_km, datatypes.Variable)
    assert v0_km is v0
    assert v0_km is not reference
    assert v0_km.unit == 'km'
    assert v0_km.axes == reference.axes
    assert numpy.array_equal(v0_km[:], 1e-3 * reference[:])


@pytest.mark.variable
def test_numerical_operations(var: typing.Dict[str, datatypes.Variable]):
    """Test operations between a datatypes.Variable and a number."""

    # multiplication is symmetric
    new = var['reference'] * 10.0
    assert isinstance(new, datatypes.Variable)
    expected = [
        # 3 x 2
        [+(1.0*10.0), +(2.0*10.0)],
        [+(2.0*10.0), -(3.0*10.0)],
        [-(4.0*10.0), +(6.0*10.0)],
    ]
    assert numpy.array_equal(new, expected)
    new = 10.0 * var['reference']
    assert isinstance(new, datatypes.Variable)
    assert numpy.array_equal(new, expected)

    # right-sided division, addition, and subtraction create a new instance
    new = var['reference'] / 10.0
    assert isinstance(new, datatypes.Variable)
    expected = [
        # 3 x 2
        [+(1.0/10.0), +(2.0/10.0)],
        [+(2.0/10.0), -(3.0/10.0)],
        [-(4.0/10.0), +(6.0/10.0)],
    ]
    assert numpy.array_equal(new, expected)
    new = var['reference'] + 10.0
    assert isinstance(new, datatypes.Variable)
    expected = [
        # 3 x 2
        [+1.0+10.0, +2.0+10.0],
        [+2.0+10.0, -3.0+10.0],
        [-4.0+10.0, +6.0+10.0],
    ]
    assert numpy.array_equal(new, expected)
    new = var['reference'] - 10.0
    assert isinstance(new, datatypes.Variable)
    expected = [
        # 3 x 2
        [+1.0-10.0, +2.0-10.0],
        [+2.0-10.0, -3.0-10.0],
        [-4.0-10.0, +6.0-10.0],
    ]
    assert numpy.array_equal(new, expected)

    # left-sided division is not supported because of metadata ambiguity
    with pytest.raises(metadata.OperandTypeError):
        10.0 / var['reference']

    # left-sided addition and subtraction create a new instance
    new = 10.0 + var['reference']
    assert isinstance(new, datatypes.Variable)
    expected = [
        # 3 x 2
        [10.0+1.0, 10.0+2.0],
        [10.0+2.0, 10.0-3.0],
        [10.0-4.0, 10.0+6.0],
    ]
    assert numpy.array_equal(new, expected)
    new = 10.0 - var['reference']
    assert isinstance(new, datatypes.Variable)
    expected = [
        # 3 x 2
        [10.0-1.0, 10.0-2.0],
        [10.0-2.0, 10.0+3.0],
        [10.0+4.0, 10.0-6.0],
    ]
    assert numpy.array_equal(new, expected)


@pytest.mark.variable
def test_variable_array(var: typing.Dict[str, datatypes.Variable]):
    """Natively convert a Variable into a NumPy array."""
    v = var['reference']
    assert isinstance(v, datatypes.Variable)
    a = numpy.array(v)
    assert isinstance(a, numpy.ndarray)
    assert numpy.array_equal(v, a)


@pytest.mark.variable
def test_variable_getitem(var: typing.Dict[str, datatypes.Variable]):
    """Subscript a Variable."""
    # reference = [
    #     [+1.0, +2.0],
    #     [+2.0, -3.0],
    #     [-4.0, +6.0],
    # ]
    v = var['reference']
    for sliced in (v[:], v[...]):
        assert isinstance(sliced, datatypes.Array)
        assert sliced is not v
        expected = numpy.array([[+1.0, +2.0], [+2.0, -3.0], [-4.0, +6.0]])
        assert numpy.array_equal(sliced, expected)
    assert v[0, 0] == datatypes.Scalar(+1.0, unit=v.unit)
    assert numpy.array_equal(v[0, :], [+1.0, +2.0])
    assert numpy.array_equal(v[:, 0], [+1.0, +2.0, -4.0])
    assert numpy.array_equal(v[:, 0:1], [[+1.0], [+2.0], [-4.0]])
    assert numpy.array_equal(v[(0, 1), :], [[+1.0, +2.0], [+2.0, -3.0]])
    expected = numpy.array([[+1.0, +2.0], [+2.0, -3.0], [-4.0, +6.0]])
    assert numpy.array_equal(v[:, (0, 1)], expected)


@pytest.mark.variable
def test_variable_names():
    """A variable may have zero or more names."""
    default = datatypes.Variable([1], unit='m', axes=['d0'])
    assert not default.name
    names = ('v0', 'var')
    variable = datatypes.Variable([1], unit='m', name=names, axes=['d0'])
    assert all(name in variable.name for name in names)


@pytest.mark.variable
def test_variable_rename():
    """A user may rename a variable."""
    v = datatypes.Variable([1], unit='m', name='Name', axes=['d0'])
    assert list(v.name) == ['Name']
    v.alias('var')
    assert all(name in v.name for name in ('Name', 'var'))
    assert list(v.alias('v0', reset=True).name) == ['v0']


@pytest.mark.xfail
@pytest.mark.variable
def test_variable_get_array(var: typing.Dict[str, datatypes.Variable]):
    """Test the internal `_get_array` method to prevent regression."""
    v = var['reference']
    a = v._get_array((0, 0))
    assert a.shape == ()
    assert a == 1.0
    assert v._array is None
    a = v._get_array(0)
    assert a.shape == (2,)
    assert numpy.array_equal(a, [1, 2])
    assert v._array is None
    a = v._get_array()
    assert a.shape == (3, 2)
    assert numpy.array_equal(a, [[+1.0, +2.0], [+2.0, -3.0], [-4.0, +6.0]])
    assert v._array is a


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


# Copied from old test module. There is overlap with existing tests.
@pytest.fixture
def components():
    return [
        {
            'data': 1 + numpy.arange(3 * 4).reshape(3, 4),
            'unit': 'J',
            'axes': ('x', 'y'),
            'name': 'v0',
        },
        {
            'data': 11 + numpy.arange(3 * 4).reshape(3, 4),
            'unit': 'J',
            'axes': ('x', 'y'),
            'name': 'v1',
        },
        {
            'data': 1 + 2*numpy.arange(3 * 5).reshape(3, 5),
            'unit': 'm',
            'axes': ('x', 'z'),
            'name': 'v2',
        },
        {
            'data': 1 + numpy.arange(3 * 4).reshape(3, 1, 4),
            'unit': 'J',
            'axes': ('x', 'y', 'z'),
            'name': 'v3',
        },
        {
            'data': 1 + numpy.arange(3 * 4 * 5).reshape(3, 4, 5),
            'unit': 'J',
            'axes': ('x', 'y', 'z'),
            'name': 'v4',
        },
    ]


def make_variable(**attrs):
    """Helper for making a variable from components."""
    return datatypes.Variable(
        attrs['data'],
        unit=attrs.get('unit'),
        name=attrs.get('name'),
        axes=attrs.get('axes'),
    )


OType = typing.TypeVar('OType', datatypes.Variable, measurable.Real)
OType = typing.Union[
    datatypes.Variable,
    measurable.Real,
]
RType = typing.TypeVar('RType', bound=type)

def call_func(
    func: typing.Callable[[OType], RType],
    rtype: RType,
    *operands: OType,
    expected: numpy.typing.ArrayLike=None,
    attrs: dict=None,
    **kwargs
) -> None:
    """Call a function of one or more variables for testing."""
    types = tuple(type(operand) for operand in operands)
    msg = f"Failed for {func!r} with {types}"
    result = func(*operands, **kwargs)
    assert isinstance(result, rtype), msg
    if attrs is not None:
        for name, value in attrs.items():
            assert getattr(result, name, False) == value, msg
    if expected is not None:
        assert numpy.array_equal(result, expected), msg


@pytest.mark.variable
def test_add_number(components):
    ref = [components[i] for i in (0, 1)]
    var = [make_variable(**component) for component in ref]
    num = 2.3
    operands = [var[0], num]
    expected = ref[0]['data'] + num
    attrs = {k: ref[0][k] for k in ('unit', 'axes')}
    attrs['name'] = ref[0]['name']
    call_func(
        operator.add,
        datatypes.Variable,
        *operands,
        expected=expected,
        attrs=attrs,
    )


@pytest.mark.variable
def test_sub_number(components):
    ref = [components[i] for i in (0, 1)]
    var = [make_variable(**component) for component in ref]
    num = 2.3
    operands = [var[0], num]
    expected = ref[0]['data'] - num
    attrs = {k: ref[0][k] for k in ('unit', 'axes')}
    attrs['name'] = ref[0]['name']
    call_func(
        operator.sub,
        datatypes.Variable,
        *operands,
        expected=expected,
        attrs=attrs,
    )


@pytest.mark.variable
def test_add_variable(components):
    ref = [components[i] for i in (0, 1)]
    var = [make_variable(**component) for component in ref]
    operands = [var[0], var[1]]
    expected = ref[0]['data'] + ref[1]['data']
    attrs = {k: ref[0][k] for k in ('unit', 'axes')}
    attrs['name'] = f"{ref[0]['name']} + {ref[1]['name']}"
    call_func(
        operator.add,
        datatypes.Variable,
        *operands,
        expected=expected,
        attrs=attrs,
    )


@pytest.mark.variable
def test_sub_variable(components):
    ref = [components[i] for i in (0, 1)]
    var = [make_variable(**component) for component in ref]
    operands = [var[0], var[1]]
    expected = ref[0]['data'] - ref[1]['data']
    attrs = {k: ref[0][k] for k in ('unit', 'axes')}
    attrs['name'] = f"{ref[0]['name']} - {ref[1]['name']}"
    call_func(
        operator.sub,
        datatypes.Variable,
        *operands,
        expected=expected,
        attrs=attrs,
    )


@pytest.mark.variable
def test_mul_same_shape(components):
    ref = [components[i] for i in (0, 1)]
    var = [make_variable(**component) for component in ref]
    operands = [var[0], var[1]]
    expected = ref[0]['data'] * ref[1]['data']
    attrs = {
        'unit': f"{ref[0]['unit']} * {ref[1]['unit']}",
        'axes': ('x', 'y'),
        'name': f"{ref[0]['name']} * {ref[1]['name']}",
    }
    call_func(
        operator.mul,
        datatypes.Variable,
        *operands,
        expected=expected,
        attrs=attrs,
    )


@pytest.mark.variable
def test_mul_diff_shape(components):
    ref = [components[i] for i in (0, 2)]
    var = [make_variable(**component) for component in ref]
    operands = [var[0], var[1]]
    opr = operator.mul
    arrays = [r['data'] for r in ref]
    axes = [v.axes for v in var]
    expected = reduce(*arrays, opr, axes=axes)
    attrs = {
        'unit': f"{ref[0]['unit']} * {ref[1]['unit']}",
        'axes': ('x', 'y', 'z'),
        'name': f"{ref[0]['name']} * {ref[1]['name']}",
        'shape': (3, 4, 5),
    }
    call_func(
        opr,
        datatypes.Variable,
        *operands,
        expected=expected,
        attrs=attrs,
    )


@pytest.mark.variable
def test_div_same_shape(components):
    ref = [components[i] for i in (0, 1)]
    var = [make_variable(**component) for component in ref]
    operands = [var[0], var[1]]
    expected = ref[0]['data'] / ref[1]['data']
    attrs = {
        'unit': f"{ref[0]['unit']} / {ref[1]['unit']}",
        'axes': ('x', 'y'),
        'name': f"{ref[0]['name']} / {ref[1]['name']}",
    }
    call_func(
        operator.truediv,
        datatypes.Variable,
        *operands,
        expected=expected,
        attrs=attrs,
    )


@pytest.mark.variable
def test_div_diff_shape(components):
    ref = [components[i] for i in (0, 2)]
    var = [make_variable(**component) for component in ref]
    operands = [var[0], var[1]]
    opr = operator.truediv
    arrays = [r['data'] for r in ref]
    axes = [v.axes for v in var]
    expected = reduce(*arrays, opr, axes=axes)
    attrs = {
        'unit': f"{ref[0]['unit']} / {ref[1]['unit']}",
        'axes': ('x', 'y', 'z'),
        'name': f"{ref[0]['name']} / {ref[1]['name']}",
        'shape': (3, 4, 5),
    }
    call_func(
        opr,
        datatypes.Variable,
        *operands,
        expected=expected,
        attrs=attrs,
    )


@pytest.mark.variable
def test_pow_number(components):
    ref = components[0]
    var = make_variable(**ref)
    num = 2
    operands = [var, num]
    expected = ref['data'] ** num
    attrs = {
        'unit': f"{ref['unit']}^{num}",
        'axes': ref['axes'],
        'name': f"{ref['name']}^{num}",
    }
    call_func(
        operator.pow,
        datatypes.Variable,
        *operands,
        expected=expected,
        attrs=attrs,
    )


@pytest.mark.variable
def test_pow_array(components):
    ref = components[0]
    var = make_variable(**ref)
    operands = [var, ref['data']]
    with pytest.raises(metadata.OperandTypeError):
        var ** ref['data']


@pytest.mark.variable
def test_sqrt(components):
    ref = components[0]
    expected = numpy.sqrt(ref['data'])
    attrs = {
        'unit': f"{ref['unit']}^1/2",
        'axes': ref['axes'],
        'name': f"{ref['name']}^1/2",
    }
    call_func(
        numpy.sqrt,
        datatypes.Variable,
        make_variable(**ref),
        expected=expected,
        attrs=attrs,
    )


@pytest.mark.variable
def test_squeeze(components):
    ref = components[3]
    expected = numpy.squeeze(ref['data'])
    attrs = {
        'unit': ref['unit'],
        'axes': ('x', 'z'),
        'name': ref['name'],
    }
    call_func(
        numpy.squeeze,
        datatypes.Variable,
        make_variable(**ref),
        expected=expected,
        attrs=attrs,
    )


@pytest.mark.variable
def test_axis_mean(components):
    ref = components[4]
    cases = [
        ('y', 'z'),
        ('x', 'z'),
        ('x', 'y'),
    ]
    for axis, axes in enumerate(cases):
        expected = numpy.mean(ref['data'], axis=axis)
        attrs = {
            'unit': ref['unit'],
            'axes': axes,
            'name': f"mean({ref['name']})",
        }
        call_func(
            numpy.mean,
            datatypes.Variable,
            make_variable(**ref),
            expected=expected,
            attrs=attrs,
            axis=axis,
        )


@pytest.mark.variable
def test_full_mean(components):
    ref = components[4]
    call_func(
        numpy.mean,
        float,
        make_variable(**ref),
        expected=numpy.mean(ref['data']),
    )


def test_dimensions_object():
    """Make sure axes names behave as expected in operations."""
    assert len(datatypes.Axes()) == 0
    names = ['a', 'b', 'c']
    for i, name in enumerate(names, start=1):
        subset = names[:i]
        dimensions = datatypes.Axes(*subset)
        assert len(dimensions) == i
        assert all(name in dimensions for name in subset)
        assert dimensions[i-1] == name


def test_dimensions_init():
    """Test various ways to initialize dimensions metadata."""
    names = ['a', 'b', 'c']
    assert len(datatypes.Axes(names)) == 3
    assert len(datatypes.Axes(*names)) == 3
    assert len(datatypes.Axes([names])) == 3
    invalid = [
        [1, 2, 3],
        [[1], [2], [3]],
        [['a'], ['b'], ['c']],
    ]
    for case in invalid:
        with pytest.raises(TypeError):
            datatypes.Axes(*case)


def test_dimensions_operators():
    """Test built-in operations on dimensions metadata."""
    xy = datatypes.Axes('x', 'y')
    yz = datatypes.Axes('y', 'z')
    zw = datatypes.Axes('z', 'w')
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
        assert opr(xy, yz) == datatypes.Axes('x', 'y', 'z')
        assert opr(xy, zw) == datatypes.Axes('x', 'y', 'z', 'w')
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
    xy = datatypes.Axes('x', 'y')
    yz = datatypes.Axes('y', 'z')
    zw = datatypes.Axes('z', 'w')
    assert xy.merge(xy) == datatypes.Axes('x', 'y')
    assert xy.merge(yz) == datatypes.Axes('x', 'y', 'z')
    assert yz.merge(xy) == datatypes.Axes('y', 'z', 'x')
    assert xy.merge(zw) == datatypes.Axes('x', 'y', 'z', 'w')
    assert zw.merge(xy) == datatypes.Axes('z', 'w', 'x', 'y')
    assert yz.merge(zw) == datatypes.Axes('y', 'z', 'w')
    assert zw.merge(yz) == datatypes.Axes('z', 'w', 'y')
    assert xy.merge(yz, zw) == datatypes.Axes('x', 'y', 'z', 'w')
    assert xy.merge(1.1) == xy
    assert xy.merge(yz, 1.1) == datatypes.Axes('x', 'y', 'z')


def test_name():
    """Test the attribute representing a data quantity's name."""
    name = datatypes.Name('a', 'A')
    assert len(name) == 2
    assert sorted(name) == ['A', 'a']
    assert all(i in name for i in ('a', 'A'))


def test_name_builtin():
    """Test operations between a name metadata object and a built-in object."""
    name = datatypes.Name('a', 'A')
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
            expected = datatypes.Name(*[f'{i}{s}{other}' for i in name])
            assert method(name, other) == expected
            expected = datatypes.Name(*[f'{other}{s}{i}' for i in name])
            assert method(other, name) == expected


def test_name_name():
    """Test operations between two name metadata objects."""
    name = datatypes.Name('a', 'A')
    cases = {
        operator.add: ' + ',
        operator.sub: ' - ',
        operator.mul: ' * ',
        operator.truediv: ' / ',
    }
    for method, s in cases.items():
        other = datatypes.Name('b', 'B')
        expected = datatypes.Name(*[f'{i}{s}{j}' for i in name for j in other])
        assert method(name, other) == expected


def test_same_name():
    """Test operations on a name metadata object with itself."""
    name = datatypes.Name('a', 'A')
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
        expected = datatypes.Name(*[f'{i}{symbol}{i}' for i in name])
        assert method(name, name) == expected


def test_empty_names():
    """Make sure operations on empty names produce empty results."""
    n0 = datatypes.Name('')
    n1 = datatypes.Name('')
    cases = {
        operator.add: ' + ',
        operator.sub: ' - ',
        operator.mul: ' * ',
        operator.truediv: ' / ',
    }
    for method in cases:
        assert method(n0, n1) == datatypes.Name('')

