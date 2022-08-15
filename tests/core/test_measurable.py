import operator

import pytest

from goats.core import measurable
from goats.core import metric
from goats.core import metadata


class Quantified(measurable.Quantified):
    """A concrete version of `~measurable.Quantified` for testing."""

    def __eq__(self, other):
        return False

    def _call(self, *args, **kwargs):
        raise NotImplementedError("Just for testing")


def test_quantified_bool():
    """Quantified objects are always truthy."""
    cases = [
        Quantified(1),
        Quantified(0),
    ]
    for case in cases:
        assert bool(case)


@pytest.mark.quantity
def test_quantity_display():
    """Test the results of printing a quantity."""
    q = measurable.Quantity(1.2, unit='m')
    assert str(q) == "1.2 [m]"
    assert repr(q).endswith("Quantity(1.2, unit='m')")


@pytest.mark.quantity
def test_quantity_comparisons():
    """Test comparisons between two default quantities."""
    value = 2.0
    unit = 'm'
    q0 = measurable.Quantity(value, unit=unit)
    cases = [
        (operator.lt, value + 1),
        (operator.le, value + 1),
        (operator.le, value),
        (operator.eq, value),
        (operator.ne, value + 1),
        (operator.gt, value - 1),
        (operator.ge, value - 1),
        (operator.ge, value),
    ]
    for case in cases:
        opr, v = case
        result = opr(q0, measurable.Quantity(v, unit=unit))
        assert isinstance(result, bool)
        assert result
    q1 = measurable.Quantity(v, unit='J')
    assert q0 != q1
    for opr in {operator.lt, operator.le, operator.gt, operator.ge}:
        with pytest.raises(ValueError):
            opr(q0, q1)


@pytest.mark.quantity
def test_quantities_same_unit():
    """Test operations on default quantities with the same unit."""
    values = [2.0, 3.5]
    unit='m'
    quantities = [measurable.Quantity(value, unit=unit) for value in values]

    # ADDITIVE
    oprs = [
        operator.add,
        operator.sub,
    ]
    unit = 'm'
    for opr in oprs:
        expected = measurable.Quantity(opr(*values), unit=unit)
        assert opr(*quantities) == expected

    # MULTIPLICATION
    opr = operator.mul
    expected = measurable.Quantity(opr(*values), unit='m^2')
    assert opr(*quantities) == expected

    # DIVISION
    opr = operator.truediv
    expected = measurable.Quantity(opr(*values), unit='1')
    assert opr(*quantities) == expected

    # EXPONENTIAL
    opr = operator.pow
    with pytest.raises(metadata.OperandTypeError):
        opr(*quantities)


@pytest.mark.quantity
def test_quantities_diff_unit():
    """Test operations on default quantities with different units."""
    value = 2.0
    units = ['m', 'J']
    quantities = [measurable.Quantity(value, unit=unit) for unit in units]

    # ADDITIVE
    oprs = [
        operator.add,
        operator.sub,
    ]
    for opr in oprs:
        with pytest.raises(metadata.UnitError):
            opr(*quantities)

    # MULTIPLICATION
    opr = operator.mul
    expected = measurable.Quantity(opr(value, value), unit='m * J')
    assert opr(*quantities) == expected

    # DIVISION
    opr = operator.truediv
    expected = measurable.Quantity(opr(value, value), unit='m / J')
    assert opr(*quantities) == expected

    # EXPONENTIAL
    opr = operator.pow
    with pytest.raises(metadata.OperandTypeError):
        opr(*quantities)


@pytest.mark.quantity
def test_quantity_number():
    """Test operations on a default quantity and a number."""
    values = [2.0, 3.5]
    unit = 'm'
    quantity = measurable.Quantity(values[0], unit=unit)
    value = values[1]

    # ADDITIVE
    oprs = [
        operator.add,
        operator.sub,
    ]
    for opr in oprs:
        # forward
        with pytest.raises(TypeError):
            opr(quantity, value)
        # reverse
        with pytest.raises(TypeError):
            opr(value, quantity)

    # MULTIPLICATION
    opr = operator.mul
    # forward
    expected = measurable.Quantity(opr(*values), unit=unit)
    assert opr(quantity, value) == expected
    # reverse
    expected = measurable.Quantity(opr(*values[::-1]), unit=unit)
    assert opr(value, quantity) == expected

    # DIVISION
    opr = operator.truediv
    # forward
    expected = measurable.Quantity(opr(*values), unit=unit)
    assert opr(quantity, value) == expected
    # reverse
    with pytest.raises(metadata.OperandTypeError):
        opr(value, quantity)

    # EXPONENTIAL
    opr = operator.pow
    # forward
    expected = measurable.Quantity(opr(*values), unit=f'{unit}^{value}')
    assert opr(quantity, value) == expected
    # reverse
    with pytest.raises(metadata.OperandTypeError):
        opr(value, quantity)


@pytest.mark.quantity
def test_quantity_unit():
    """Test getting and setting a quantity's unit."""
    # NOTE: use an integer to avoid failing due to precision
    q = measurable.Quantity(1, unit='m')
    assert q.unit == 'm'
    assert q.convert('cm') == measurable.Quantity(100, unit='cm')
    with pytest.raises(metric.UnitConversionError):
        q.convert('J')


@pytest.mark.quantity
def test_initialize_quantity():
    """Test the initialization behavior of Quantity."""
    unit = 'm'
    cases = [
        1,
        1.1,
        [1.1],
        [1, 2],
        [1.1, 2.3],
    ]
    for data in cases:
        q = measurable.Quantity(data, unit=unit)
        assert q.data == data
        assert q.unit == unit
        with pytest.raises(TypeError):
            # metadata arguments are keyword only
            measurable.Quantity(data, unit)


@pytest.mark.quantity
def test_quantity_idempotence():
    """Test initializing a concrete quantity from an existing instance."""
    q0 = measurable.Quantity(1.5, unit='m')
    q1 = measurable.Quantity(q0)
    assert q1 is not q0
    assert q1 == q0


def test_scalar_cast():
    """Test the type-casting operators on a scalar quantity."""
    value = 1.5
    scalar = measurable.Scalar(value, unit='m')
    assert float(scalar) == value
    assert int(scalar) == int(value)


def test_scalar_unary():
    """Test the unary operators on a scalar quantity."""
    value = 1.5
    unit = 'm'
    scalar = measurable.Scalar(value, unit=unit)
    assert round(scalar) == measurable.Scalar(round(value), unit=unit)


unity = '1'
unitless = [
    {'test': 1.1, 'full': (1.1, unity), 'dist': [(1.1, unity)]},
    {'test': (1.1,), 'full': (1.1, unity), 'dist': [(1.1, unity)]},
    {'test': [1.1], 'full': (1.1, unity), 'dist': [(1.1, unity)]},
    {
        'test': (1.1, 2.3),
        'full': (1.1, 2.3, unity),
        'dist': [(1.1, unity), (2.3, unity)],
    },
    {
        'test': [1.1, 2.3],
        'full': (1.1, 2.3, unity),
        'dist': [(1.1, unity), (2.3, unity)],
    },
]
meter = 'm'
with_units = [
    {'test': (1.1, meter), 'full': (1.1, meter), 'dist': [(1.1, meter)]},
    {'test': [1.1, meter], 'full': (1.1, meter), 'dist': [(1.1, meter)]},
    {
        'test': (1.1, 2.3, meter),
        'full': (1.1, 2.3, meter),
        'dist': [(1.1, meter), (2.3, meter)]
    },
    {
        'test': [1.1, 2.3, meter],
        'full': (1.1, 2.3, meter),
        'dist': [(1.1, meter), (2.3, meter)],
    },
    {
        'test': [(1.1, 2.3), meter],
        'full': (1.1, 2.3, meter),
        'dist': [(1.1, meter), (2.3, meter)],
    },
    {
        'test': [[1.1, 2.3], meter],
        'full': (1.1, 2.3, meter),
        'dist': [(1.1, meter), (2.3, meter)],
    },
    {
        'test': ((1.1, meter), (2.3, meter)),
        'full': (1.1, 2.3, meter),
        'dist': [(1.1, meter), (2.3, meter)],
    },
    {
        'test': [(1.1, meter), (2.3, meter)],
        'full': (1.1, 2.3, meter),
        'dist': [(1.1, meter), (2.3, meter)],
    },
    {
        'test': [(1.1, meter), (2.3, 5.8, meter)],
        'full': (1.1, 2.3, 5.8, meter),
        'dist': [(1.1, meter), (2.3, meter), (5.8, meter)],
    },
]
builtin_cases = [
    *unitless,
    *with_units,
]


def test_parse_measurable():
    """Test the function that attempts to parse a measurable object."""
    for case in builtin_cases:
        result = measurable.parse_measurable(case['test'])
        expected = case['full']
        assert result == expected
    for case in builtin_cases:
        result = measurable.parse_measurable(case['test'], distribute=True)
        expected = case['dist']
        assert result == expected
    assert measurable.parse_measurable(0) == (0, '1') # zero is measurable!
    with pytest.raises(measurable.Unmeasurable):
        measurable.parse_measurable(None)
    with pytest.raises(measurable.MeasuringTypeError):
        measurable.parse_measurable([1.1, 'm', 2.3, 'cm'])
    with pytest.raises(measurable.MeasuringTypeError):
        measurable.parse_measurable([(1.1, 'm'), (2.3, 5.8, 'cm')])


def test_measurable():
    """Test the function that determines if we can measure and object."""
    cases = [case['test'] for case in builtin_cases]
    for case in cases:
        assert measurable.ismeasurable(case)
    class Test:
        def __measure__(): ...
    assert measurable.ismeasurable(Test())


def test_measure():
    """Test the function that creates a measurement object."""
    for case in builtin_cases:
        result = measurable.measure(case['test'])
        assert isinstance(result, measurable.Measurement)


def test_measurement():
    """Test the measurement object on its own."""
    values = (1.1, 2.3)
    unit = 'm'
    measurement = measurable.Measurement(values, unit)
    assert isinstance(measurement, measurable.Measurement)
    assert measurement.values == values
    assert measurement.unit == unit
    assert len(measurement) == len(values)
    for i, value in enumerate(values):
        assert measurement[i] == [(value, unit)]


@pytest.mark.xfail
def test_single_valued_measurement():
    """Test special properties of a single-valued measurement."""
    unit = 'm'
    values = [1.1]
    measurement = measurable.Measurement(values, unit)
    assert float(measurement) == float(values[0])
    assert int(measurement) == int(values[0])
    values = [1.1, 2.3]
    measurement = measurable.Measurement(values, unit)
    with pytest.raises(TypeError):
        float(measurement)
    with pytest.raises(TypeError):
        int(measurement)


