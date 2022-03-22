import itertools
import operator
import typing

import numpy
import pytest

from goats.core import datatypes
from goats.core import quantities


@pytest.mark.variable
def test_variable():
    """Test the object that represents a variable."""
    v0 = datatypes.Variable([3.0, 4.5], 'm', ['x'])
    v1 = datatypes.Variable([[1.0], [2.0]], 'J', ['x', 'y'])
    assert numpy.array_equal(v0, [3.0, 4.5])
    assert v0.unit == quantities.Unit('m')
    assert list(v0.axes) == ['x']
    assert v0.naxes == 1
    assert numpy.array_equal(v1, [[1.0], [2.0]])
    assert v1.unit == quantities.Unit('J')
    assert list(v1.axes) == ['x', 'y']
    assert v1.naxes == 2
    v0_cm = v0.convert_to('cm')
    assert v0_cm is not v0
    assert numpy.array_equal(v0_cm, 100 * v0)
    assert v0_cm.unit == quantities.Unit('cm')
    assert v0_cm.axes == v0.axes
    r = v0 + v0
    expected = [6.0, 9.0]
    assert numpy.array_equal(r, expected)
    assert r.unit == v0.unit
    r = v0 * v1
    expected = [[3.0 * 1.0], [4.5 * 2.0]]
    assert numpy.array_equal(r, expected)
    assert r.unit == quantities.Unit('m * J')
    r = v0 / v1
    expected = [[3.0 / 1.0], [4.5 / 2.0]]
    assert numpy.array_equal(r, expected)
    assert r.unit == quantities.Unit('m / J')
    r = v0 ** 2
    expected = [3.0 ** 2, 4.5 ** 2]
    assert numpy.array_equal(r, expected)
    assert r.unit == quantities.Unit('m^2')


@pytest.mark.variable
def test_variable_measure():
    """Test the use of `~quantities.measure` on a variable."""
    v0 = datatypes.Variable([3.0, 4.5], 'm', ['x'])
    measured = quantities.measure(v0)
    assert measured.values == [3.0, 4.5]
    assert measured.unit == 'm'


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


def reduce(a, b, opr):
    """Create an array from `a` and `b` by applying `opr`.

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
    if I == P and J == Q:
        return [
            # I x J
            [
                # J
                opr(a[i][j], b[i][j]) for j in J
            ] for i in I
        ]
    if J == P:
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
        expected = reduce(a0, a1, opr)
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
    opr = operator.pow
    v0 = var['reference']
    a0 = arr['reference']
    cases = [
        {
            'p': 3,
            'rtype': datatypes.Variable,
        },
        {
            'p': a0,
            'rtype': numpy.ndarray,
        },
    ]
    msg = f"Failed for {opr}"
    for case in cases:
        p = case['p']
        rtype = case['rtype']
        new = opr(v0, p)
        assert isinstance(new, rtype)
        if rtype == datatypes.Variable:
            expected = reduce(numpy.array(v0), p, operator.pow)
            assert numpy.array_equal(new, expected), msg
            assert new.axes == var['reference'].axes, msg
            algebraic = opr(v0.unit, 3)
            formatted = f'({v0.unit})^{p}'
            for unit in (algebraic, formatted):
                assert new.unit == unit, msg
        elif rtype == numpy.ndarray:
            expected = opr(numpy.array(a0), numpy.array(a0))
            assert numpy.array_equal(new, expected), msg
        else:
            raise TypeError(
                f"Unexpected return type {type(new)}"
                f" for operand types {type(v0)} and {type(p)}"
            ) from None


@pytest.mark.variable
def test_variable_add_sub(
    var: typing.Dict[str, datatypes.Variable],
    arr: typing.Dict[str, list],
) -> None:
    """Test the ability to add two datatypes.Variable instances."""
    v0 = var['reference']
    a0 = arr['reference']
    a1 = arr['samedims']
    v1 = datatypes.Variable(a1, v0.unit, v0.axes)
    v2 = datatypes.Variable(arr['different'], v0.unit, var['different'].axes)
    for opr in (operator.add, operator.sub):
        msg = f"Failed for {opr}"
        new = opr(v0, v1)
        expected = reduce(a0, a1, opr)
        assert isinstance(new, datatypes.Variable), msg
        assert numpy.array_equal(new, expected), msg
        assert new.unit == v0.unit, msg
        assert new.axes == v0.axes, msg
        with pytest.raises(ValueError): # numpy broadcasting error
            opr(v0, v2)


@pytest.mark.variable
def test_variable_units(var: typing.Dict[str, datatypes.Variable]):
    """Test the ability to update unit via bracket syntax."""
    v0_km = var['reference'].convert_to('km')
    assert isinstance(v0_km, datatypes.Variable)
    assert v0_km is not var['reference']
    assert v0_km.unit == 'km'
    assert v0_km.axes == var['reference'].axes
    assert numpy.array_equal(v0_km[:], 1e-3 * var['reference'][:])


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

    # left-sided division, addition, and subtraction create a new instance
    new = 10.0 / var['reference']
    assert isinstance(new, numpy.ndarray)
    expected = [
        # 3 x 2
        [+(10.0/1.0), +(10.0/2.0)],
        [+(10.0/2.0), -(10.0/3.0)],
        [-(10.0/4.0), +(10.0/6.0)],
    ]
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
        assert isinstance(sliced, datatypes.Variable)
        assert sliced is not v
        expected = numpy.array([[+1.0, +2.0], [+2.0, -3.0], [-4.0, +6.0]])
        assert numpy.array_equal(sliced, expected)
    assert v[0, 0] == quantities.Scalar(+1.0, v.unit)
    assert numpy.array_equal(v[0, :], [+1.0, +2.0])
    assert numpy.array_equal(v[:, 0], [+1.0, +2.0, -4.0])
    assert numpy.array_equal(v[:, 0:1], [[+1.0], [+2.0], [-4.0]])
    assert numpy.array_equal(v[(0, 1), :], [[+1.0, +2.0], [+2.0, -3.0]])
    expected = numpy.array([[+1.0, +2.0], [+2.0, -3.0], [-4.0, +6.0]])
    assert numpy.array_equal(v[:, (0, 1)], expected)


@pytest.mark.variable
def test_variable_name():
    """A variable may have a given name or be anonymous."""
    default = datatypes.Variable([1], 'm', ['d0'])
    assert default.name == '<anonymous>'
    cases = {
        'test': 'test',
        None: '<anonymous>',
    }
    for name, expected in cases.items():
        variable = datatypes.Variable([1], 'm', ['d0'], name=name)
        assert variable.name == expected


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


def test_add_number(components):
    ref = [components[i] for i in (0, 1)]
    var = [datatypes.Variable(**component) for component in ref]
    num = 2.3
    result = var[0] + num
    assert isinstance(result, datatypes.Variable)
    assert result.unit == ref[0]['unit']
    assert result.axes == ref[0]['axes']
    assert result.name == ref[0]['name']
    expected = ref[0]['data'] + num
    assert numpy.array_equal(result, expected)


def test_sub_number(components):
    ref = [components[i] for i in (0, 1)]
    var = [datatypes.Variable(**component) for component in ref]
    num = 2.3
    result = var[0] - num
    assert isinstance(result, datatypes.Variable)
    assert result.unit == ref[0]['unit']
    assert result.axes == ref[0]['axes']
    assert result.name == ref[0]['name']
    expected = ref[0]['data'] - num
    assert numpy.array_equal(result, expected)


def test_add_variable(components):
    ref = [components[i] for i in (0, 1)]
    var = [datatypes.Variable(**component) for component in ref]
    result = var[0] + var[1]
    assert isinstance(result, datatypes.Variable)
    assert result.unit == ref[0]['unit']
    assert result.axes == ref[0]['axes']
    assert result.name == f"{ref[0]['name']} + {ref[1]['name']}"
    expected = ref[0]['data'] + ref[1]['data']
    assert numpy.array_equal(result, expected)


def test_sub_variable(components):
    ref = [components[i] for i in (0, 1)]
    var = [datatypes.Variable(**component) for component in ref]
    result = var[0] - var[1]
    assert isinstance(result, datatypes.Variable)
    assert result.unit == ref[0]['unit']
    assert result.axes == ref[0]['axes']
    assert result.name == f"{ref[0]['name']} - {ref[1]['name']}"
    expected = ref[0]['data'] - ref[1]['data']
    assert numpy.array_equal(result, expected)


def test_mul_same_shape(components):
    ref = [components[i] for i in (0, 1)]
    var = [datatypes.Variable(**component) for component in ref]
    result = var[0] * var[1]
    assert isinstance(result, datatypes.Variable)
    assert result.unit == f"{ref[0]['unit']} * {ref[1]['unit']}"
    assert result.axes == ('x', 'y')
    assert result.name == f"{ref[0]['name']} * {ref[1]['name']}"
    expected = ref[0]['data'] * ref[1]['data']
    assert numpy.array_equal(result, expected)


def test_mul_diff_shape(components):
    ref = [components[i] for i in (0, 2)]
    var = [datatypes.Variable(**component) for component in ref]
    result = var[0] * var[1]
    assert isinstance(result, datatypes.Variable)
    assert result.unit == f"{ref[0]['unit']} * {ref[1]['unit']}"
    assert result.axes == ('x', 'y', 'z')
    assert result.name == f"{ref[0]['name']} * {ref[1]['name']}"
    assert numpy.array(result).shape == (3, 4, 5)
    # TODO: This is here because numpy can't broadcast the arrays together. The
    # solution is to create the arrays by hand, as in `test_datasets.reduce`.
    with pytest.raises(ValueError):
        expected = ref[0]['data'] * ref[1]['data']
        assert numpy.array_equal(result, expected)


def test_div_same_shape(components):
    ref = [components[i] for i in (0, 1)]
    var = [datatypes.Variable(**component) for component in ref]
    result = var[0] / var[1]
    assert isinstance(result, datatypes.Variable)
    assert result.unit == f"{ref[0]['unit']} / {ref[1]['unit']}"
    assert result.axes == ('x', 'y')
    assert result.name == f"{ref[0]['name']} / {ref[1]['name']}"
    expected = ref[0]['data'] / ref[1]['data']
    assert numpy.array_equal(result, expected)


def test_div_diff_shape(components):
    ref = [components[i] for i in (0, 2)]
    var = [datatypes.Variable(**component) for component in ref]
    result = var[0] / var[1]
    assert isinstance(result, datatypes.Variable)
    assert result.unit == f"{ref[0]['unit']} / {ref[1]['unit']}"
    assert result.axes == ('x', 'y', 'z')
    assert result.name == f"{ref[0]['name']} / {ref[1]['name']}"
    assert numpy.array(result).shape == (3, 4, 5)
    # TODO: This is here because numpy can't broadcast the arrays together. The
    # solution is to create the arrays by hand, as in `test_datasets.reduce`.
    with pytest.raises(ValueError):
        expected = ref[0]['data'] / ref[1]['data']
        assert numpy.array_equal(result, expected)


def test_pow_number(components):
    ref = components[0]
    var = datatypes.Variable(**ref)
    result = var ** 2
    assert isinstance(result, datatypes.Variable)
    assert result.unit == f"{ref['unit']}^2"
    assert result.axes == ref['axes']
    assert result.name == f"{ref['name']}^2"
    expected = ref['data'] ** 2
    assert numpy.array_equal(result, expected)


def test_pow_array(components):
    ref = components[0]
    var = datatypes.Variable(**ref)
    result = var ** ref['data']
    assert isinstance(result, numpy.ndarray)
    expected = ref['data'] ** ref['data']
    assert numpy.array_equal(result, expected)


def test_sqrt(components):
    ref = components[0]
    var = datatypes.Variable(**ref)
    result = numpy.sqrt(var)
    assert isinstance(result, datatypes.Variable)
    assert result.unit == f"sqrt({ref['unit']})"
    assert result.axes == ref['axes']
    assert result.name == f"sqrt({ref['name']})"
    expected = numpy.sqrt(ref['data'])
    assert numpy.array_equal(result, expected)


def test_squeeze(components):
    ref = components[3]
    var = datatypes.Variable(**ref)
    result = numpy.squeeze(var)
    assert isinstance(result, datatypes.Variable)
    assert result.unit == ref['unit']
    assert result.axes == ('x', 'z')
    assert result.name == ref['name']
    expected = numpy.squeeze(ref['data'])
    assert numpy.array_equal(result, expected)


def test_axis_mean(components):
    ref = components[4]
    var = datatypes.Variable(**ref)

    result = numpy.mean(var, axis=0)
    assert isinstance(result, datatypes.Variable)
    assert result.unit == ref['unit']
    assert result.axes == ('y', 'z')
    assert result.name == f"mean({ref['name']})"
    expected = numpy.mean(ref['data'], axis=0)
    assert numpy.array_equal(result, expected)

    result = numpy.mean(var, axis=1)
    assert isinstance(result, datatypes.Variable)
    assert result.unit == ref['unit']
    assert result.axes == ('x', 'z')
    assert result.name == f"mean({ref['name']})"
    expected = numpy.mean(ref['data'], axis=1)
    assert numpy.array_equal(result, expected)

    result = numpy.mean(var, axis=2)
    assert isinstance(result, datatypes.Variable)
    assert result.unit == ref['unit']
    assert result.axes == ('x', 'y')
    assert result.name == f"mean({ref['name']})"
    expected = numpy.mean(ref['data'], axis=2)
    assert numpy.array_equal(result, expected)


def test_full_mean(components):
    ref = components[4]
    var = datatypes.Variable(**ref)
    result = numpy.mean(var)
    assert isinstance(result, float)
    assert result == numpy.mean(ref['data'])

