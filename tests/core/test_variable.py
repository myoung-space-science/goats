import itertools
import operator
import random
import typing

import numpy
import numpy.typing
import pytest

from goats.core import algebraic
from goats.core import datafile
from goats.core import variable
from goats.core import physical
from goats.core import measurable
from goats.core import metadata


def get_interface(testdata: dict, name: str) -> datafile.Interface:
    """Get an interface to a dataset file by name."""
    return datafile.Interface(testdata[name]['path'])


def get_reference(
    testdata: typing.Dict[str, dict],
    name: str,
    key: str,
) -> typing.Dict[str, dict]:
    """Get reference values for the named dataset."""
    return testdata[name].get(key, {})


def test_variables(testdata: dict):
    """Test the higher-level variables interface."""
    reference = {
        'time': {
            'unit': 's',
            'axes': ['time'],
        },
        'Vr': {
            'unit': 'm / s',
            'axes': ['time', 'shell'],
        },
        'flux': {
            'unit': 'm^-2 s^-1 sr^-1 J^-1',
            'axes': ['time', 'shell', 'species', 'energy'],
        },
        'dist': {
            'unit': 's^3 m^-6',
            'axes': ['time', 'shell', 'species', 'energy', 'mu'],
        },
    }
    for name in ('eprem-obs', 'eprem-flux'):
        datafile = get_interface(testdata, name)
        variables = variable.Interface(datafile)
        for observable, expected in reference.items():
            if observable in variables:
                v = variables[observable]
                assert v.unit == expected['unit']
                assert sorted(v.axes) == sorted(expected['axes'])
            else:
                with pytest.raises(KeyError):
                    variables[observable]


def test_standardize():
    """Test the helper function that standardizes unit strings."""
    cases = {
        'julian date': 'day',
        'shell': '1',
        'cos(mu)': '1',
        'e-': 'e',
        '# / cm^2 s sr MeV': '# / (cm^2 s sr MeV/nuc)',
    }
    for old, new in cases.items():
        assert variable.standardize(old) == new


@pytest.mark.variable
def test_variable_display():
    """Test the results of printing a variable."""
    v = variable.Quantity([1.2], unit='m', name='V', axes=['x'])
    assert str(v) == "'V': <class 'list'> [m] axes=['x']"
    assert repr(v).endswith("(<class 'list'>, unit='m', name='V', axes=['x'])")


@pytest.mark.variable
def test_variable():
    """Test the object that represents a variable."""
    v0 = variable.Quantity([3.0, 4.5], unit='m', axes=['x'])
    v1 = variable.Quantity([[1.0], [2.0]], unit='J', axes=['x', 'y'])
    assert numpy.array_equal(v0, [3.0, 4.5])
    assert v0.unit == metadata.Unit('m')
    assert list(v0.axes) == ['x']
    assert v0.naxes == 1
    assert numpy.array_equal(v1, [[1.0], [2.0]])
    assert v1.unit == metadata.Unit('J')
    assert list(v1.axes) == ['x', 'y']
    assert v1.naxes == 2
    r = v0 + v0
    expected = [6.0, 9.0]
    assert numpy.array_equal(r, expected)
    assert r.unit == v0.unit
    r = v0 * v1
    expected = [[3.0 * 1.0], [4.5 * 2.0]]
    assert numpy.array_equal(r, expected)
    assert r.unit == metadata.Unit('m * J')
    r = v0 / v1
    expected = [[3.0 / 1.0], [4.5 / 2.0]]
    assert numpy.array_equal(r, expected)
    assert r.unit == metadata.Unit('m / J')
    r = v0 ** 2
    expected = [3.0 ** 2, 4.5 ** 2]
    assert numpy.array_equal(r, expected)
    assert r.unit == metadata.Unit('m^2')
    reference = variable.Quantity(v0)
    assert reference is not v0
    v0_cm = v0['cm']
    assert v0_cm is v0
    expected = 100 * reference
    assert numpy.array_equal(v0_cm, expected)
    assert v0_cm.unit == metadata.Unit('cm')
    assert v0_cm.axes == reference.axes


@pytest.mark.variable
def test_variable_init():
    """Test initializing a variable object."""
    data = [[3.0], [4.5]]
    unit = 'm / s'
    name = 'V'
    axes = ('x', 'y')
    # default metadata
    v0 = variable.Quantity(data)
    assert v0.unit == '1'
    assert v0.name == ''
    assert v0.axes == ('x0', 'x1')
    # default unit
    v0 = variable.Quantity(data, name=name, axes=axes)
    assert v0.unit == '1'
    assert v0.name == name
    assert v0.axes == axes
    # default name
    v0 = variable.Quantity(data, unit=unit, axes=axes)
    assert v0.unit == unit
    assert v0.name == ''
    assert v0.axes == axes
    # default axes
    v0 = variable.Quantity(data, unit=unit, name=name)
    assert v0.unit == unit
    assert v0.name == name
    assert v0.axes == ('x0', 'x1')
    # number of axes must match number of data dimensions
    with pytest.raises(ValueError):
        # (too few)
        variable.Quantity(data, axes=['x'])
    with pytest.raises(ValueError):
        # (too many)
        variable.Quantity(data, axes=['x', 'y', 'z'])
    # new from old is equal but not identical
    v1 = variable.Quantity(v0)
    assert v1 is not v0
    assert v1 == v0


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
def var(arr: typing.Dict[str, list]) -> typing.Dict[str, variable.Quantity]:
    """A tuple of test variables."""
    reference = variable.Quantity(
        arr['reference'].copy(),
        axes=('d0', 'd1'),
        unit='m',
    )
    samedims = variable.Quantity(
        arr['samedims'].copy(),
        axes=('d0', 'd1'),
        unit='kJ',
    )
    sharedim = variable.Quantity(
        arr['sharedim'].copy(),
        axes=('d1', 'd2'),
        unit='s',
    )
    different = variable.Quantity(
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
    var: typing.Dict[str, variable.Quantity],
    arr: typing.Dict[str, list],
) -> None:
    """Test the ability to multiply two variable.Quantity instances."""
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
        assert isinstance(new, variable.Quantity), msg
        expected = reduce(a0, a1, opr, axes=(v0.axes, v1.axes))
        assert numpy.array_equal(new, expected), msg
        assert sorted(new.axes) == case['axes'], msg
        algebraic = opr(v0.unit, v1.unit)
        formatted = f'({v0.unit}){sym}({v1.unit})'
        for unit in (algebraic, formatted):
            assert new.unit == unit, msg


@pytest.mark.variable
def test_variable_pow(
    var: typing.Dict[str, variable.Quantity],
    arr: typing.Dict[str, list],
) -> None:
    """Test the ability to exponentiate a variable.Quantity instance."""
    v0 = var['reference']
    p = 3
    new = v0 ** p
    assert isinstance(new, variable.Quantity)
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
    var: typing.Dict[str, variable.Quantity],
    arr: typing.Dict[str, list],
) -> None:
    """Test the ability to add two variable.Quantity instances."""
    v0 = var['reference']
    a0 = arr['reference']
    a1 = arr['samedims']
    v1 = variable.Quantity(a1, unit=v0.unit, axes=v0.axes)
    v2 = variable.Quantity(
        arr['different'],
        unit=v0.unit,
        axes=var['different'].axes,
    )
    for opr in (operator.add, operator.sub):
        msg = f"Failed for {opr}"
        new = opr(v0, v1)
        expected = reduce(a0, a1, opr, axes=(v0.axes, v1.axes))
        assert isinstance(new, variable.Quantity), msg
        assert numpy.array_equal(new, expected), msg
        assert new.unit == v0.unit, msg
        assert new.axes == v0.axes, msg
        with pytest.raises(ValueError): # numpy broadcasting error
            opr(v0, v2)


@pytest.mark.variable
def test_variable_units(var: typing.Dict[str, variable.Quantity]):
    """Test the ability to update a variable's unit."""
    v0 = var['reference']
    reference = variable.Quantity(v0)
    v0_km = v0['km']
    assert isinstance(v0_km, variable.Quantity)
    assert v0_km is v0
    assert v0_km is not reference
    assert v0_km.unit == 'km'
    assert v0_km.axes == reference.axes
    assert numpy.array_equal(v0_km[:], 1e-3 * reference[:])


@pytest.mark.variable
def test_numerical_operations(var: typing.Dict[str, variable.Quantity]):
    """Test operations between a variable.Quantity and a number."""

    # multiplication is symmetric
    new = var['reference'] * 10.0
    assert isinstance(new, variable.Quantity)
    expected = [
        # 3 x 2
        [+(1.0*10.0), +(2.0*10.0)],
        [+(2.0*10.0), -(3.0*10.0)],
        [-(4.0*10.0), +(6.0*10.0)],
    ]
    assert numpy.array_equal(new, expected)
    new = 10.0 * var['reference']
    assert isinstance(new, variable.Quantity)
    assert numpy.array_equal(new, expected)

    # right-sided division creates a new instance
    new = var['reference'] / 10.0
    assert isinstance(new, variable.Quantity)
    expected = [
        # 3 x 2
        [+(1.0/10.0), +(2.0/10.0)],
        [+(2.0/10.0), -(3.0/10.0)],
        [-(4.0/10.0), +(6.0/10.0)],
    ]
    assert numpy.array_equal(new, expected)

    # left-sided division is not supported because of metadata ambiguity
    with pytest.raises(metadata.OperandTypeError):
        10.0 / var['reference']

    # addition and subtraction are not supported due to unit ambiguity
    with pytest.raises(TypeError):
        var['reference'] + 10.0
    with pytest.raises(TypeError):
        var['reference'] - 10.0
    with pytest.raises(TypeError):
        10.0 + var['reference']
    with pytest.raises(TypeError):
        10.0 - var['reference']


@pytest.mark.variable
def test_variable_array(var: typing.Dict[str, variable.Quantity]):
    """Natively convert a Variable into a NumPy array."""
    v = var['reference']
    assert isinstance(v, variable.Quantity)
    a = numpy.array(v)
    assert isinstance(a, numpy.ndarray)
    assert numpy.array_equal(v, a)


@pytest.mark.variable
def test_variable_getitem(var: typing.Dict[str, variable.Quantity]):
    """Subscript a Variable."""
    # reference = [
    #     [+1.0, +2.0],
    #     [+2.0, -3.0],
    #     [-4.0, +6.0],
    # ]
    v = var['reference']
    for sliced in (v[:], v[...]):
        assert isinstance(sliced, physical.Array)
        assert sliced is not v
        expected = numpy.array([[+1.0, +2.0], [+2.0, -3.0], [-4.0, +6.0]])
        assert numpy.array_equal(sliced, expected)
    assert v[0, 0] == physical.Scalar(+1.0, unit=v.unit)
    assert numpy.array_equal(v[0, :], [+1.0, +2.0])
    assert numpy.array_equal(v[:, 0], [+1.0, +2.0, -4.0])
    assert numpy.array_equal(v[:, 0:1], [[+1.0], [+2.0], [-4.0]])
    assert numpy.array_equal(v[(0, 1), :], [[+1.0, +2.0], [+2.0, -3.0]])
    expected = numpy.array([[+1.0, +2.0], [+2.0, -3.0], [-4.0, +6.0]])
    assert numpy.array_equal(v[:, (0, 1)], expected)
    assert numpy.array_equal(v[:, None, 0], [[+1.0], [+2.0], [-4.0]])
    assert numpy.array_equal(v[None, :, 0], [[+1.0, +2.0, -4.0]])


@pytest.mark.variable
def test_variable_names():
    """A variable may have zero or more names."""
    default = variable.Quantity([1], unit='m', axes=['d0'])
    assert not default.name
    names = ('v0', 'var')
    current = variable.Quantity([1], unit='m', name=names, axes=['d0'])
    assert all(name in current.name for name in names)


@pytest.mark.variable
def test_variable_rename():
    """A user may rename a variable."""
    v = variable.Quantity([1], unit='m', name='Name', axes=['d0'])
    assert list(v.name) == ['Name']
    v.alias('var')
    assert all(name in v.name for name in ('Name', 'var'))
    assert list(v.alias('v0', reset=True).name) == ['v0']


@pytest.mark.xfail
@pytest.mark.variable
def test_variable_get_array(var: typing.Dict[str, variable.Quantity]):
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


def make_variable(**attrs):
    """Helper for making a variable from components."""
    return variable.Quantity(
        attrs['data'],
        unit=attrs.get('unit'),
        name=attrs.get('name'),
        axes=attrs.get('axes'),
    )


OType = typing.TypeVar('OType', variable.Quantity, algebraic.Real)
OType = typing.Union[
    variable.Quantity,
    algebraic.Real,
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
def test_add_variable(components):
    ref = [components[i] for i in (0, 1)]
    var = [make_variable(**component) for component in ref]
    operands = [var[0], var[1]]
    expected = ref[0]['data'] + ref[1]['data']
    attrs = {k: ref[0][k] for k in ('unit', 'axes')}
    attrs['name'] = f"{ref[0]['name']} + {ref[1]['name']}"
    call_func(
        operator.add,
        variable.Quantity,
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
        variable.Quantity,
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
        variable.Quantity,
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
        variable.Quantity,
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
        variable.Quantity,
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
        variable.Quantity,
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
        variable.Quantity,
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
        variable.Quantity,
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
        variable.Quantity,
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
            variable.Quantity,
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


def test_array_contains():
    """Test the function that checks for a value in an array."""
    v = variable.Quantity([1.1e-30, 2e-30], axes=['x'])
    cases = [
        {'value': 1.1e-30, 'in': True, 'contains': True},
        {'value': 2.0e-30, 'in': True, 'contains': True},
    ]
    cases = [
        (1.1e-30, (True, True)),
        (2.0e-30, (True, True)),
        (1.1e+30 / 1.0e+60, (False, True)),
        (1.0e-30 + 1.0e-31, (False, True)),
    ]
    for value, test in cases:
        assert (value in v) == test[0]
        assert v.array_contains(value) == test[1]

