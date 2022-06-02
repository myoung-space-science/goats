import math
import operator

import pytest

from goats.core import measurable
from goats.core import operations


interface = operations.Interface('_amount', '_metric')
Quantified = interface.subclass('Quantified', measurable.Quantifiable)
"""A concrete version of `~measurable.Quantifiable` for testing."""

interface = operations.Interface(
    'data', 'unit',
    default=[measurable.Quantity, 'data'],
)
rules = {
    'numeric': [
        (measurable.Quantity, measurable.Quantity),
        (measurable.Quantity, measurable.Real, 'data'),
        (measurable.Real, measurable.Quantity, 'data'),
    ]
}
interface['numeric'].rules.register(*rules['numeric'])
interface['__rtruediv__'].rules.suppress(measurable.Real, measurable.Quantity)
interface['__pow__'].rules[measurable.Quantity, measurable.Real].append('unit')
interface['__rpow__'].rules.suppress(measurable.Real, measurable.Quantity)
interface['__pow__'].rules[measurable.Quantity, measurable.Quantity].suppress
Quantity = interface.subclass(
    'Quantity',
    exclude=['cast', '__round__', '__ceil__', '__floor__', '__trunc__']
)
"""A concrete quantity for testing."""


Scalar = interface.subclass('Scalar')
"""A concrete scalar quantity for testing."""


@pytest.mark.quantity
def test_quantity_comparisons():
    """Test comparisons between two default quantities."""
    value = 2.0
    unit = 'm'
    scalar = Quantity(value, unit)
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
        result = opr(scalar, Quantity(v, unit))
        assert isinstance(result, bool)
        assert result
        with pytest.raises(operations.OperandTypeError):
            opr(scalar, Quantity(v, 'J'))


@pytest.mark.quantity
def test_quantities_same_unit():
    """Test operations on default quantities with the same unit."""
    cases = [
        (2.0, 'm'),
        (3.5, 'm'),
    ]
    quantities = [Quantity(*args) for args in cases]
    values = [k[0] for k in cases]

    # ADDITIVE
    oprs = [
        operator.add,
        operator.sub,
    ]
    unit = 'm'
    for opr in oprs:
        expected = Quantity(opr(*values), unit)
        assert opr(*quantities) == expected

    # MULTIPLICATION
    opr = operator.mul
    expected = Quantity(opr(*values), 'm^2')
    assert opr(*quantities) == expected

    # DIVISION
    opr = operator.truediv
    expected = Quantity(opr(*values), '1')
    assert opr(*quantities) == expected

    # EXPONENTIAL
    opr = operator.pow
    with pytest.raises(TypeError):
        opr(*quantities)


@pytest.mark.quantity
def test_quantities_diff_unit():
    """Test operations on default quantities with different units."""
    cases = [
        (2.0, 'm'),
        (2.0, 'J'),
    ]
    quantities = [Quantity(*args) for args in cases]
    values = [k[0] for k in cases]

    # ADDITIVE
    oprs = [
        operator.add,
        operator.sub,
    ]
    for opr in oprs:
        with pytest.raises(operations.OperandTypeError):
            opr(*quantities)

    # MULTIPLICATION
    opr = operator.mul
    expected = Quantity(opr(*values), 'm * J')
    assert opr(*quantities) == expected

    # DIVISION
    opr = operator.truediv
    expected = Quantity(opr(*values), 'm / J')
    assert opr(*quantities) == expected

    # EXPONENTIAL
    opr = operator.pow
    with pytest.raises(TypeError):
        opr(*quantities)


@pytest.mark.quantity
def test_quantity_number():
    """Test operations on a default quantity and a number."""
    values = [2.0, 3.5]
    unit = 'm'
    quantity = Quantity(values[0], unit)
    value = values[1]

    # ADDITIVE
    oprs = [
        operator.add,
        operator.sub,
    ]
    for opr in oprs:
        # forward
        expected = Quantity(opr(*values), unit)
        assert opr(quantity, value) == expected
        # reverse
        expected = Quantity(opr(*values[::-1]), unit)
        assert opr(value, quantity) == expected

    # MULTIPLICATION
    opr = operator.mul
    # forward
    expected = Quantity(opr(*values), unit)
    assert opr(quantity, value) == expected
    # reverse
    expected = Quantity(opr(*values[::-1]), unit)
    assert opr(value, quantity) == expected

    # DIVISION
    opr = operator.truediv
    # forward
    expected = Quantity(opr(*values), unit)
    assert opr(quantity, value) == expected
    # reverse
    with pytest.raises(TypeError):
        opr(value, quantity)

    # EXPONENTIAL
    opr = operator.pow
    # forward
    expected = Quantity(opr(*values), f'{unit}^{value}')
    assert opr(quantity, value) == expected
    # reverse
    with pytest.raises(TypeError):
        opr(value, quantity)


def test_scalar_cast():
    """Test the type-casting operators on a scalar quantity."""
    value = 1.5
    scalar = Scalar(value, 'm')
    assert float(scalar) == value
    assert int(scalar) == int(value)


def test_scalar_unary():
    """Test the unary operators on a scalar quantity."""
    value = 1.5
    unit = 'm'
    scalar = Scalar(value, unit)
    assert round(scalar) == Scalar(round(value), unit)


def test_quantity_idempotence():
    """Test initializing a concrete quantity from an existing instance."""
    q0 = Quantity(1.5, 'm')
    q1 = Quantity(q0)
    assert q1 is not q0
    assert q1 == q0


def test_quantified_bool():
    """Quantified objects are always truthy."""
    cases = [
        Quantified(1, 'quantum'),
        Quantified(0, 'quantum'),
    ]
    for case in cases:
        assert bool(case)


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


