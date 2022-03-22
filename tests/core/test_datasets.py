import operator
import random
import typing

import numpy
import pytest

from goats.core import aliased
from goats.core import datasets
from goats.core import quantities


def get_dataset_view(testdata: dict, name: str) -> datasets.DatasetView:
    """Get a dataset view by name."""
    return datasets.DatasetView(testdata[name]['path'])


def get_dataset(testdata: dict, name: str) -> datasets.Dataset:
    """Get a dataset interface (without axis indexers) by name."""
    return datasets.Dataset(testdata[name]['path'])


def get_reference(
    testdata: typing.Dict[str, dict],
    name: str,
    key: str,
) -> typing.Dict[str, dict]:
    """Get reference values for the named dataset."""
    return testdata[name].get(key, {})


def test_dataset_axes(testdata: dict):
    """Test access to dataset axes."""
    testname = 'basic'
    dataset = get_dataset_view(testdata, testname)
    reference = get_reference(testdata, testname, 'axes')
    assert isinstance(dataset.axes, typing.Mapping)
    for axis in reference:
        assert axis in dataset.axes
    for names, axis in dataset.axes.items(aliased=True):
        assert isinstance(axis, datasets.DatasetAxis)
        assert axis.name in names
        assert axis.size == reference[axis.name]['size']


def test_dataset_variables(testdata: dict):
    """Test access to dataset variables."""
    testname = 'basic'
    dataset = get_dataset_view(testdata, testname)
    reference = get_reference(testdata, testname, 'variables')
    assert isinstance(dataset.variables, typing.Mapping)
    for variable in reference:
        assert variable in dataset.variables
    for names, variable in dataset.variables.items(aliased=True):
        assert isinstance(variable, datasets.DatasetVariable)
        assert variable.name in names
        ref = reference[variable.name]
        assert variable.unit == ref.get('unit')
        assert sorted(variable.axes) == sorted(ref.get('axes', ()))


def test_dataset_view(testdata: dict):
    """Test access to the lower-level dataset view."""
    testname = 'basic'
    dataset = get_dataset_view(testdata, testname)
    variables = dataset.variables
    assert variables['time'] == variables['t']
    available = dataset.available('variables')
    full = ['time', 't', 'times', 'level', 'lat', 'lon', 'temp']
    assert sorted(available.full) == sorted(full)
    aliases = [
        aliased.MappingKey(key)
        for key in ('level', ('time', 't', 'times'), 'lat', 'lon', 'temp')
    ]
    assert sorted(available.aliased) == sorted(aliases)
    canonical = ['level', 'time', 'lat', 'lon', 'temp']
    assert sorted(available.canonical) == sorted(canonical)


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
        dataset = get_dataset_view(testdata, name)
        variables = datasets.Variables(dataset)
        for observable, expected in reference.items():
            if observable in variables:
                variable = variables[observable]
                assert variable.unit == expected['unit']
                assert sorted(variable.axes) == sorted(expected['axes'])
            else:
                with pytest.raises(KeyError):
                    variables[observable]


def test_dataset(testdata: dict):
    """Test the full higher-level dataset interface."""
    reference = {
        'time': {
            'axes': ['time'],
        },
        'Vr': {
            'axes': ['time', 'shell'],
        },
        'flux': {
            'axes': ['time', 'shell', 'species', 'energy'],
        },
        'dist': {
            'axes': ['time', 'shell', 'species', 'energy', 'mu'],
        },
    }
    axes = {
        'time': 20,
        'shell': 100,
        'species': 2,
        'energy': 10,
        'mu': 5,
    }
    for name in ('eprem-obs', 'eprem-flux'):
        dataset = get_dataset(testdata, name)
        assert isinstance(dataset, datasets.Dataset)
        assert isinstance(dataset.variables, typing.Mapping)
        assert isinstance(dataset.axes, typing.Mapping)
        for key, length in axes.items():
            axis = dataset.axes[key]
            assert key in axis.name
            assert list(axis()) == list(range(length))
        for observable, expected in reference.items():
            if observable in dataset.variables:
                iter_axes = dataset.iter_axes(observable)
                assert sorted(iter_axes) == sorted(expected['axes'])
                unordered = random.sample(axes.keys(), len(axes))
                assert dataset.resolve_axes(unordered) == tuple(axes)
            else: # Test both options when variable is not in dataset.
                assert not list(dataset.iter_axes(observable, default=()))
                with pytest.raises(ValueError):
                    dataset.iter_axes(observable)


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
        assert datasets.standardize(old) == new


@pytest.mark.variable
def test_variable():
    """Test the object that represents a variable."""
    v0 = datasets.Variable([3.0, 4.5], 'm', ['x'])
    v1 = datasets.Variable([[1.0], [2.0]], 'J', ['x', 'y'])
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
    v0 = datasets.Variable([3.0, 4.5], 'm', ['x'])
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
def var(arr: typing.Dict[str, list]) -> typing.Dict[str, datasets.Variable]:
    """A tuple of test variables."""
    reference = datasets.Variable(
        arr['reference'].copy(),
        axes=('d0', 'd1'),
        unit='m',
    )
    samedims = datasets.Variable(
        arr['samedims'].copy(),
        axes=('d0', 'd1'),
        unit='kJ',
    )
    sharedim = datasets.Variable(
        arr['sharedim'].copy(),
        axes=('d1', 'd2'),
        unit='s',
    )
    different = datasets.Variable(
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
    var: typing.Dict[str, datasets.Variable],
    arr: typing.Dict[str, list],
) -> None:
    """Test the ability to multiply two datasets.Variable instances."""
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
    for sym, opr in groups.items():
        for name, case in cases.items():
            msg = f"Failed for {name} with {opr}"
            v1 = var[case['key']]
            a1 = arr[case['key']]
            new = opr(v0, v1)
            assert isinstance(new, datasets.Variable), msg
            expected = reduce(a0, a1, opr)
            assert numpy.array_equal(new, expected), msg
            assert sorted(new.axes) == case['axes'], msg
            algebraic = opr(v0.unit, v1.unit)
            formatted = f'({v0.unit}){sym}({v1.unit})'
            for unit in (algebraic, formatted):
                assert new.unit == unit, msg


@pytest.mark.variable
def test_variable_pow(var: typing.Dict[str, datasets.Variable]) -> None:
    """Test the ability to exponentiate a datasets.Variable instance."""
    opr = operator.pow
    v0 = var['reference']
    ex = 3
    msg = f"Failed for {opr}"
    new = opr(v0, ex)
    assert isinstance(new, datasets.Variable)
    expected = reduce(numpy.array(v0), ex, operator.pow)
    assert numpy.array_equal(new, expected), msg
    assert new.axes == var['reference'].axes, msg
    algebraic = opr(v0.unit, 3)
    formatted = f'({v0.unit})^{ex}'
    for unit in (algebraic, formatted):
        assert new.unit == unit, msg


@pytest.mark.variable
def test_variable_add_sub(
    var: typing.Dict[str, datasets.Variable],
    arr: typing.Dict[str, list],
) -> None:
    """Test the ability to add two datasets.Variable instances."""
    v0 = var['reference']
    a0 = arr['reference']
    a1 = arr['samedims']
    v1 = datasets.Variable(a1, v0.unit, v0.axes)
    v2 = datasets.Variable(arr['different'], v0.unit, var['different'].axes)
    for opr in (operator.add, operator.sub):
        msg = f"Failed for {opr}"
        new = opr(v0, v1)
        expected = reduce(a0, a1, opr)
        assert isinstance(new, datasets.Variable), msg
        assert numpy.array_equal(new, expected), msg
        assert new.unit == v0.unit, msg
        assert new.axes == v0.axes, msg
        with pytest.raises(ValueError): # numpy broadcasting error
            opr(v0, v2)


@pytest.mark.variable
def test_variable_units(var: typing.Dict[str, datasets.Variable]):
    """Test the ability to update unit via bracket syntax."""
    v0_km = var['reference'].convert_to('km')
    assert isinstance(v0_km, datasets.Variable)
    assert v0_km is not var['reference']
    assert v0_km.unit == 'km'
    assert v0_km.axes == var['reference'].axes
    assert numpy.array_equal(v0_km[:], 1e-3 * var['reference'][:])


@pytest.mark.variable
def test_numerical_operations(var: typing.Dict[str, datasets.Variable]):
    """Test operations between a datasets.Variable and a number."""

    # multiplication is symmetric
    new = var['reference'] * 10.0
    assert isinstance(new, datasets.Variable)
    expected = [
        # 3 x 2
        [+(1.0*10.0), +(2.0*10.0)],
        [+(2.0*10.0), -(3.0*10.0)],
        [-(4.0*10.0), +(6.0*10.0)],
    ]
    assert numpy.array_equal(new, expected)
    new = 10.0 * var['reference']
    assert isinstance(new, datasets.Variable)
    assert numpy.array_equal(new, expected)

    # right-sided division, addition, and subtraction create a new instance
    new = var['reference'] / 10.0
    assert isinstance(new, datasets.Variable)
    expected = [
        # 3 x 2
        [+(1.0/10.0), +(2.0/10.0)],
        [+(2.0/10.0), -(3.0/10.0)],
        [-(4.0/10.0), +(6.0/10.0)],
    ]
    assert numpy.array_equal(new, expected)
    new = var['reference'] + 10.0
    assert isinstance(new, datasets.Variable)
    expected = [
        # 3 x 2
        [+1.0+10.0, +2.0+10.0],
        [+2.0+10.0, -3.0+10.0],
        [-4.0+10.0, +6.0+10.0],
    ]
    assert numpy.array_equal(new, expected)
    new = var['reference'] - 10.0
    assert isinstance(new, datasets.Variable)
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
    assert isinstance(new, datasets.Variable)
    expected = [
        # 3 x 2
        [10.0+1.0, 10.0+2.0],
        [10.0+2.0, 10.0-3.0],
        [10.0-4.0, 10.0+6.0],
    ]
    assert numpy.array_equal(new, expected)
    new = 10.0 - var['reference']
    assert isinstance(new, datasets.Variable)
    expected = [
        # 3 x 2
        [10.0-1.0, 10.0-2.0],
        [10.0-2.0, 10.0+3.0],
        [10.0+4.0, 10.0-6.0],
    ]
    assert numpy.array_equal(new, expected)


@pytest.mark.variable
def test_variable_array(var: typing.Dict[str, datasets.Variable]):
    """Natively convert a Variable into a NumPy array."""
    v = var['reference']
    assert isinstance(v, datasets.Variable)
    a = numpy.array(v)
    assert isinstance(a, numpy.ndarray)
    assert numpy.array_equal(v, a)


@pytest.mark.variable
def test_variable_getitem(var: typing.Dict[str, datasets.Variable]):
    """Subscript a Variable."""
    # reference = [
    #     [+1.0, +2.0],
    #     [+2.0, -3.0],
    #     [-4.0, +6.0],
    # ]
    v = var['reference']
    for sliced in (v[:], v[...]):
        assert isinstance(sliced, datasets.Variable)
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
    default = datasets.Variable([1], 'm', ['d0'])
    assert default.name == '<anonymous>'
    cases = {
        'test': 'test',
        None: '<anonymous>',
    }
    for name, expected in cases.items():
        variable = datasets.Variable([1], 'm', ['d0'], name=name)
        assert variable.name == expected


@pytest.mark.variable
def test_variable_get_array(var: typing.Dict[str, datasets.Variable]):
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


