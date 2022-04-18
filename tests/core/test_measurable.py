import operator

import pytest

from goats.core import measurable


class Quantified(measurable.OperatorMixin, measurable.Quantifiable):
    """Concrete version of `~measurable.Quantifiable` for testing."""


class Quantity(measurable.OperatorMixin, measurable.Quantity):
    """Concrete quantity for testing."""


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
        assert opr(scalar, Quantity(v, unit))
        with pytest.raises(TypeError):
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
        with pytest.raises(measurable.OperandError):
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


def test_same():
    """Test the class that enforces object consistency.

    This test first defines a demo class that requires a value, a kind, and a
    name. It then defines simple functions that add the values of two instances
    of that class, with various restrictions on which attributes need to be the
    same.
    """
    class Score:
        def __init__(self, points: float, kind: str, name: str) -> None:
            self.points = points
            self.kind = kind
            self.name = name

    scores = [
        Score(2.0, 'cat', 'Squirt'),
        Score(3.0, 'cat', 'Paul'),
        Score(1.0, 'dog', 'Bornk'),
        Score(6.0, 'cat', 'Paul'),
    ]

    # No restrictions:
    def f0(*scores: Score):
        return sum((score.points for score in scores))

    # Instances must have the same kind:
    @measurable.same('kind')
    def f1(*args):
        return f0(*args)

    # Instances must have the same kind and the same name:
    @measurable.same('kind', 'name')
    def f2(*args):
        return f0(*args)

    # A single instance always passes:
    @measurable.same('kind')
    def f3(arg):
        return arg

    # Add two instances with no restrictions.
    assert f0(scores[0], scores[2]) == 3.0

    # Add all instances with no restrictions.
    assert f0(*scores) == 12.0

    # Add two instances with restricted kind.
    assert f1(scores[0], scores[1]) == 5.0

    # Add three instances with restricted kind.
    args = [scores[i] for i in (0, 1, 3)]
    assert f1(*args) == 11.0

    # Try to add two instances with different kind.
    with pytest.raises(measurable.ComparisonError):
        f1(scores[0], scores[2])

    # Try to add an instance to a built-in float.
    assert f1(scores[0], 2.0) == NotImplemented

    # Add two instances with restricted kind and name.
    assert f2(scores[1], scores[3]) == 9.0

    # Try to add two instances with same kind but different name.
    with pytest.raises(measurable.ComparisonError):
        f2(scores[0], scores[1])

    # Test a trivial case.
    assert f3(scores[0]) == scores[0]


def test_getattrval():
    """Test the function that gets a value based on object type."""
    class Base:
        def __init__(self, value) -> None:
            self._value = value
    class PropertyAttr(Base):
        @property
        def value(self):
            return self._value
    class CallableAttr(Base):
        def value(self, scale=1.0):
            return scale * self._value

    value = 2.5
    instance = PropertyAttr(value)
    assert measurable.getattrval(instance, 'value') == value
    instance = CallableAttr(value)
    assert measurable.getattrval(instance, 'value') == value
    scale = 10.0
    expected = scale * value
    assert measurable.getattrval(instance, 'value', scale) == expected
    assert measurable.getattrval(instance, 'value', scale=scale) == expected
    assert measurable.getattrval(value, 'value') == value


def test_setattrval():
    """Test the function that sets a value based on object type."""
    class Base:
        def __init__(self, value) -> None:
            self._value = value
    class StandardAttr(Base):
        def __init__(self, value) -> None:
            super().__init__(value)
            self.value = self._value
    class PropertyAttr(Base):
        @property
        def value(self):
            return self._value
        @value.setter
        def value(self, value):
            self._value = value
    class CallableAttr(Base):
        def value(self, value=None, scale=1.0):
            if value is None:
                return scale * self._value
            self._value = scale * value
            return self

    old, new, scale = 2.5, 4.0, 10.0
    for Instance in StandardAttr, PropertyAttr, CallableAttr:
        instance = Instance(old)
        assert measurable.getattrval(instance, 'value') == old
        measurable.setattrval(instance, 'value', new)
        assert measurable.getattrval(instance, 'value') == new
        if isinstance(instance, CallableAttr):
            measurable.setattrval(instance, 'value', new, scale)
            assert measurable.getattrval(instance, 'value') == scale * new
            measurable.setattrval(instance, 'value', new, scale=scale)
            assert measurable.getattrval(instance, 'value') == scale * new
        with pytest.raises(AttributeError):
            measurable.setattrval(instance, 'other', None)


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


