import itertools
import operator
import typing

import numpy
import numpy.typing
import pytest

from goats.core import datatypes
from goats.core import measurable
from goats.core import metric


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
    v0_cm = v0.convert_to('cm')
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
    v0_km = v0.convert_to('km')
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
        assert isinstance(sliced, datatypes.Array)
        assert sliced is not v
        expected = numpy.array([[+1.0, +2.0], [+2.0, -3.0], [-4.0, +6.0]])
        assert numpy.array_equal(sliced, expected)
    assert v[0, 0] == measurable.Scalar(+1.0, v.unit)
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
    assert not default.names
    names = ('v0', 'var')
    variable = datatypes.Variable([1], *names, unit='m', axes=['d0'])
    assert all(name in variable.names for name in names)


@pytest.mark.variable
def test_variable_rename():
    """A user may rename a variable."""
    v = datatypes.Variable([1], 'Name', unit='m', axes=['d0'])
    assert list(v.names) == ['Name']
    v.rename('var', update=True)
    assert all(name in v.names for name in ('Name', 'var'))
    assert list(v.rename('v0').names) == ['v0']


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
    assert all(alias in assumption.aliases for alias in aliases)
    scalars = [measurable.Scalar(value, unit) for value in values]
    assert assumption[:] == scalars
    converted = assumption.convert_to('cm')
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
        attrs['name'],
        unit=attrs.get('unit'),
        axes=attrs.get('axes'),
    )


OType = typing.TypeVar('OType', datatypes.Variable, measurable.RealValued)
OType = typing.Union[
    datatypes.Variable,
    measurable.RealValued,
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


def test_add_number(components):
    ref = [components[i] for i in (0, 1)]
    var = [make_variable(**component) for component in ref]
    num = 2.3
    operands = [var[0], num]
    expected = ref[0]['data'] + num
    attrs = {k: ref[0][k] for k in ('unit', 'axes')}
    attrs['names'] = {ref[0]['name']}
    call_func(
        operator.add,
        datatypes.Variable,
        *operands,
        expected=expected,
        attrs=attrs,
    )


def test_sub_number(components):
    ref = [components[i] for i in (0, 1)]
    var = [make_variable(**component) for component in ref]
    num = 2.3
    operands = [var[0], num]
    expected = ref[0]['data'] - num
    attrs = {k: ref[0][k] for k in ('unit', 'axes')}
    attrs['names'] = {ref[0]['name']}
    call_func(
        operator.sub,
        datatypes.Variable,
        *operands,
        expected=expected,
        attrs=attrs,
    )


def test_add_variable(components):
    ref = [components[i] for i in (0, 1)]
    var = [make_variable(**component) for component in ref]
    operands = [var[0], var[1]]
    expected = ref[0]['data'] + ref[1]['data']
    attrs = {k: ref[0][k] for k in ('unit', 'axes')}
    attrs['names'] = {f"{ref[0]['name']} + {ref[1]['name']}"}
    call_func(
        operator.add,
        datatypes.Variable,
        *operands,
        expected=expected,
        attrs=attrs,
    )


def test_sub_variable(components):
    ref = [components[i] for i in (0, 1)]
    var = [make_variable(**component) for component in ref]
    operands = [var[0], var[1]]
    expected = ref[0]['data'] - ref[1]['data']
    attrs = {k: ref[0][k] for k in ('unit', 'axes')}
    attrs['names'] = {f"{ref[0]['name']} - {ref[1]['name']}"}
    call_func(
        operator.sub,
        datatypes.Variable,
        *operands,
        expected=expected,
        attrs=attrs,
    )


def test_mul_same_shape(components):
    ref = [components[i] for i in (0, 1)]
    var = [make_variable(**component) for component in ref]
    operands = [var[0], var[1]]
    expected = ref[0]['data'] * ref[1]['data']
    attrs = {
        'unit': f"{ref[0]['unit']} * {ref[1]['unit']}",
        'axes': ('x', 'y'),
        'names': {f"{ref[0]['name']} * {ref[1]['name']}"},
    }
    call_func(
        operator.mul,
        datatypes.Variable,
        *operands,
        expected=expected,
        attrs=attrs,
    )


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
        'names': {f"{ref[0]['name']} * {ref[1]['name']}"},
        'shape': (3, 4, 5),
    }
    call_func(
        opr,
        datatypes.Variable,
        *operands,
        expected=expected,
        attrs=attrs,
    )


def test_div_same_shape(components):
    ref = [components[i] for i in (0, 1)]
    var = [make_variable(**component) for component in ref]
    operands = [var[0], var[1]]
    expected = ref[0]['data'] / ref[1]['data']
    attrs = {
        'unit': f"{ref[0]['unit']} / {ref[1]['unit']}",
        'axes': ('x', 'y'),
        'names': {f"{ref[0]['name']} / {ref[1]['name']}"},
    }
    call_func(
        operator.truediv,
        datatypes.Variable,
        *operands,
        expected=expected,
        attrs=attrs,
    )


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
        'names': {f"{ref[0]['name']} / {ref[1]['name']}"},
        'shape': (3, 4, 5),
    }
    call_func(
        opr,
        datatypes.Variable,
        *operands,
        expected=expected,
        attrs=attrs,
    )


def test_pow_number(components):
    ref = components[0]
    var = make_variable(**ref)
    num = 2
    operands = [var, num]
    expected = ref['data'] ** num
    attrs = {
        'unit': f"{ref['unit']}^{num}",
        'axes': ref['axes'],
        'names': {f"{ref['name']}^{num}"},
    }
    call_func(
        operator.pow,
        datatypes.Variable,
        *operands,
        expected=expected,
        attrs=attrs,
    )


def test_pow_array(components):
    ref = components[0]
    var = make_variable(**ref)
    operands = [var, ref['data']]
    result = var ** ref['data']
    assert isinstance(result, numpy.ndarray)
    expected = ref['data'] ** ref['data']
    call_func(
        operator.pow,
        numpy.ndarray,
        *operands,
        expected=expected,
    )


def test_sqrt(components):
    ref = components[0]
    expected = numpy.sqrt(ref['data'])
    attrs = {
        'unit': f"sqrt({ref['unit']})",
        'axes': ref['axes'],
        'names': {f"sqrt({ref['name']})"},
    }
    call_func(
        numpy.sqrt,
        datatypes.Variable,
        make_variable(**ref),
        expected=expected,
        attrs=attrs,
    )


def test_squeeze(components):
    ref = components[3]
    expected = numpy.squeeze(ref['data'])
    attrs = {
        'unit': ref['unit'],
        'axes': ('x', 'z'),
        'names': {ref['name']},
    }
    call_func(
        numpy.squeeze,
        datatypes.Variable,
        make_variable(**ref),
        expected=expected,
        attrs=attrs,
    )


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
            'names': {f"mean({ref['name']})"},
        }
        call_func(
            numpy.mean,
            datatypes.Variable,
            make_variable(**ref),
            expected=expected,
            attrs=attrs,
            axis=axis,
        )


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
    assert len(datatypes.Dimensions()) == 0
    names = ['a', 'b', 'c']
    for i, name in enumerate(names, start=1):
        subset = names[:i]
        dimensions = datatypes.Dimensions(*subset)
        assert len(dimensions) == i
        assert all(name in dimensions for name in subset)
        assert dimensions[i-1] == name


def test_dimensions_init():
    """Test various ways to initialize dimensions metadata."""
    names = ['a', 'b', 'c']
    assert len(datatypes.Dimensions(names)) == 3
    assert len(datatypes.Dimensions(*names)) == 3
    assert len(datatypes.Dimensions([names])) == 3
    invalid = [
        [1, 2, 3],
        [[1], [2], [3]],
        [['a'], ['b'], ['c']],
    ]
    for case in invalid:
        with pytest.raises(TypeError):
            datatypes.Dimensions(*case)


def test_dimensions_operators():
    """Test built-in operations on dimensions metadata."""
    xy = datatypes.Dimensions('x', 'y')
    yz = datatypes.Dimensions('y', 'z')
    zw = datatypes.Dimensions('z', 'w')
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
        assert opr(xy, yz) == datatypes.Dimensions('x', 'y', 'z')
        assert opr(xy, zw) == datatypes.Dimensions('x', 'y', 'z', 'w')
    # exponentiation is not valid
    with pytest.raises(TypeError):
        pow(xy, xy)
    # unary arithmetic operations should preserve dimensions
    for instance in (xy, yz, zw):
        for opr in (abs, operator.pos, operator.neg):
            assert opr(instance) == instance
    # cast and comparison operations don't affect dimensions


def test_name():
    """Test the attribute representing a data quantity's name."""
    name = datatypes.Name('a', 'A')
    assert len(name) == 2
    assert sorted(name) == ['A', 'a']
    assert all(i in name for i in ('a', 'A'))


def test_name_builtin():
    """Test operations between a name metadata object and a built-in object."""
    name = datatypes.Name('a', 'A')
    cases = {
        operator.add: {'symbol': '+', 'others': ['b', 2]},
        operator.sub: {'symbol': '-', 'others': ['b', 2]},
        operator.mul: {'symbol': '*', 'others': ['b', 2]},
        operator.truediv: {'symbol': '/', 'others': ['b', 2]},
        pow: {'symbol': '^','others': [2]},
    }
    for method, test in cases.items():
        s = test['symbol']
        for other in test['others']:
            expected = datatypes.Name(*[f'{i}{s}{other}' for i in name])
            assert method(name, other) == expected


def test_name_name():
    """Test operations between two name metadata objects."""
    name = datatypes.Name('a', 'A')
    cases = {
        operator.add: '+',
        operator.sub: '-',
        operator.mul: '*',
        operator.truediv: '/',
    }
    for method, s in cases.items():
        other = datatypes.Name('b', 'B')
        expected = datatypes.Name(*[f'{i}{s}{j}' for i in name for j in other])
        assert method(name, other) == expected


def test_same_name():
    """Test operations on a name metadata object with itself."""
    name = datatypes.Name('a', 'A')
    additive = {
        operator.add: '+',
        operator.sub: '-',
    }
    multiplicative = {
        operator.mul: '*',
        operator.truediv: '/',
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
        operator.add: '+',
        operator.sub: '-',
        operator.mul: '*',
        operator.truediv: '/',
    }
    for method in cases:
        assert method(n0, n1) == datatypes.Name('')

