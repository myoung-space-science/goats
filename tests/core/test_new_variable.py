import numpy
import pytest

from goats.core._variable_dev import Variable


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


def test_add(components):
    ref = [components[i] for i in (0, 1)]
    var = [Variable(**component) for component in ref]
    result = var[0] + var[1]
    assert isinstance(result, Variable)
    assert result.unit == ref[0]['unit']
    assert result.axes == ref[0]['axes']
    assert result.name == f"{ref[0]['name']} + {ref[1]['name']}"
    expected = ref[0]['data'] + ref[1]['data']
    assert numpy.array_equal(result, expected)


def test_sub(components):
    ref = [components[i] for i in (0, 1)]
    var = [Variable(**component) for component in ref]
    result = var[0] - var[1]
    assert isinstance(result, Variable)
    assert result.unit == ref[0]['unit']
    assert result.axes == ref[0]['axes']
    assert result.name == f"{ref[0]['name']} - {ref[1]['name']}"
    expected = ref[0]['data'] - ref[1]['data']
    assert numpy.array_equal(result, expected)


def test_mul_same_shape(components):
    ref = [components[i] for i in (0, 1)]
    var = [Variable(**component) for component in ref]
    result = var[0] * var[1]
    assert isinstance(result, Variable)
    assert result.unit == f"{ref[0]['unit']} * {ref[1]['unit']}"
    assert result.axes == ('x', 'y')
    assert result.name == f"{ref[0]['name']} * {ref[1]['name']}"
    expected = ref[0]['data'] * ref[1]['data']
    assert numpy.array_equal(result, expected)


def test_mul_diff_shape(components):
    ref = [components[i] for i in (0, 2)]
    var = [Variable(**component) for component in ref]
    result = var[0] * var[1]
    assert isinstance(result, Variable)
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
    var = [Variable(**component) for component in ref]
    result = var[0] / var[1]
    assert isinstance(result, Variable)
    assert result.unit == f"{ref[0]['unit']} / {ref[1]['unit']}"
    assert result.axes == ('x', 'y')
    assert result.name == f"{ref[0]['name']} / {ref[1]['name']}"
    expected = ref[0]['data'] / ref[1]['data']
    assert numpy.array_equal(result, expected)


def test_div_diff_shape(components):
    ref = [components[i] for i in (0, 2)]
    var = [Variable(**component) for component in ref]
    result = var[0] / var[1]
    assert isinstance(result, Variable)
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
    var = Variable(**ref)
    result = var ** 2
    assert isinstance(result, Variable)
    assert result.unit == f"{ref['unit']}^2"
    assert result.axes == ref['axes']
    assert result.name == f"{ref['name']}^2"
    expected = ref['data'] ** 2
    assert numpy.array_equal(result, expected)


@pytest.mark.xfail
def test_pow_array(components):
    ref = components[0]
    var = Variable(**ref)
    result = var ** ref['data']
    assert isinstance(result, numpy.ndarray)
    expected = ref['data'] ** ref['data']
    assert numpy.array_equal(result, expected)


def test_sqrt(components):
    ref = components[0]
    var = Variable(**ref)
    result = numpy.sqrt(var)
    assert isinstance(result, Variable)
    assert result.unit == f"sqrt({ref['unit']})"
    assert result.axes == ref['axes']
    assert result.name == f"sqrt({ref['name']})"
    expected = numpy.sqrt(ref['data'])
    assert numpy.array_equal(result, expected)


def test_squeeze(components):
    ref = components[3]
    var = Variable(**ref)
    result = numpy.squeeze(var)
    assert isinstance(result, Variable)
    assert result.unit == ref['unit']
    assert result.axes == ('x', 'z')
    assert result.name == ref['name']
    expected = numpy.squeeze(ref['data'])
    assert numpy.array_equal(result, expected)


def test_axis_mean(components):
    ref = components[4]
    var = Variable(**ref)

    result = numpy.mean(var, axis=0)
    assert isinstance(result, Variable)
    assert result.unit == ref['unit']
    assert result.axes == ('y', 'z')
    assert result.name == f"mean({ref['name']})"
    expected = numpy.mean(ref['data'], axis=0)
    assert numpy.array_equal(result, expected)

    result = numpy.mean(var, axis=1)
    assert isinstance(result, Variable)
    assert result.unit == ref['unit']
    assert result.axes == ('x', 'z')
    assert result.name == f"mean({ref['name']})"
    expected = numpy.mean(ref['data'], axis=1)
    assert numpy.array_equal(result, expected)

    result = numpy.mean(var, axis=2)
    assert isinstance(result, Variable)
    assert result.unit == ref['unit']
    assert result.axes == ('x', 'y')
    assert result.name == f"mean({ref['name']})"
    expected = numpy.mean(ref['data'], axis=2)
    assert numpy.array_equal(result, expected)


def test_full_mean(components):
    ref = components[4]
    var = Variable(**ref)
    result = numpy.mean(var)
    assert isinstance(result, float)
    assert result == numpy.mean(ref['data'])

