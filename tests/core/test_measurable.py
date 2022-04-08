import math
import numbers
import operator
import typing

import pytest

from goats.core import measurable
from goats.core import metric


class Ordered(measurable.Ordered):
    """Test class for ordered quantities."""

    def __init__(self, __value) -> None:
        self._value = __value

    def __lt__(self, other) -> bool:
        return self._value < other

    def __eq__(self, other) -> bool:
        return self._value == other

    def __le__(self, other) -> bool:
        return super().__le__(other)

    def __ge__(self, other) -> bool:
        return super().__ge__(other)

    def __gt__(self, other) -> bool:
        return super().__gt__(other)

    def __ne__(self, other) -> bool:
        return super().__ne__(other)

    def __bool__(self) -> bool:
        return self._value != 0


def test_ordered():
    """Test a concrete version of the ordered type."""
    this = Ordered(2)
    assert this == 2
    assert this != 1
    assert this < 3
    assert this <= 2
    assert this <= 3
    assert this > 1
    assert this >= 2
    assert this >= 1


@pytest.mark.scalar
def test_scalar_scalar_comparisons():
    """Test comparisons between two scalars."""
    value = 2.0
    unit = 'm'
    scalar = measurable.Scalar(value, unit)
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
        assert opr(scalar, measurable.Scalar(v, unit))
        with pytest.raises(TypeError):
            opr(scalar, measurable.Scalar(v, 'J'))


@pytest.mark.scalar
def test_scalar_number_comparisons():
    """Test comparisons between a scalar and a number."""
    value = 2.0
    unit = 'm'
    scalar = measurable.Scalar(value, unit)
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
    scalar = measurable.Scalar(value, 'm')
    for dtype in {int, float}:
        number = dtype(scalar)
        assert isinstance(number, dtype)
        assert number == dtype(value)


@pytest.mark.scalar
def test_scalar_unary():
    """Test unary arithmetic operations on a scalar."""
    value = 2.0
    unit = 'm'
    scalar = measurable.Scalar(value, unit)
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
        assert result == measurable.Scalar(opr(value), unit)


@pytest.mark.scalar
def test_scalar_binary():
    """Test binary arithmetic operations on a scalar."""
    cases = [
        (2.0, 'm'),
        (3.5, 'm'),
        (2.0, 'J'),
    ]
    instances = {args: measurable.Scalar(*args) for args in cases}
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
        expected = measurable.Scalar(opr(*values_same), unit)
        assert opr(*scalars_same) == expected
        # between an instance and a number
        # ...forward
        expected = measurable.Scalar(opr(*values_same), unit)
        assert opr(scalar, value) == expected
        # ...reverse
        expected = measurable.Scalar(opr(*values_same[::-1]), unit)
        assert opr(value, scalar) == expected
    # between two instances with different units
    for opr in oprs:
        with pytest.raises(TypeError):
            opr(*scalars_diff)

    # MULTIPLICATION
    opr = operator.mul
    # between two instances with same unit
    expected = measurable.Scalar(opr(*values_same), 'm^2')
    assert opr(*scalars_same) == expected
    # between an instance and a number
    # ...forward
    expected = measurable.Scalar(opr(*values_same), 'm')
    assert opr(scalar, value) == expected
    # reverse
    expected = measurable.Scalar(opr(*values_same[::-1]), 'm')
    assert opr(value, scalar) == expected
    # between two instances with different units
    expected = measurable.Scalar(opr(*values_diff), 'm * J')
    assert opr(*scalars_diff) == expected

    # DIVISION
    opr = operator.truediv
    # between two instances with same unit
    expected = measurable.Scalar(opr(*values_same), '1')
    assert opr(*scalars_same) == expected
    # between an instance and a number
    # ...forward
    expected = measurable.Scalar(opr(*values_same), 'm')
    assert opr(scalar, value) == expected
    # reverse
    with pytest.raises(TypeError):
        opr(value, scalar)
    # between two instances with different units
    expected = measurable.Scalar(opr(*values_diff), 'm / J')
    assert opr(*scalars_diff) == expected

    # EXPONENTIAL
    opr = operator.pow
    # between two instances with the same unit
    with pytest.raises(TypeError):
        opr(*scalars_same)
    # between an instance and a number
    # ...forward
    expected = measurable.Scalar(opr(*values_same), f'm^{value}')
    assert opr(scalar, value) == expected
    # ...reverse
    with pytest.raises(TypeError):
        opr(value, scalar)


@pytest.mark.scalar
def test_scalar_bitwise():
    """bitwise comparison is undefined"""
    scalar = measurable.Scalar(2)
    with pytest.raises(TypeError):
        scalar & 1
        scalar | 1
        scalar ^ 1


@pytest.mark.vector
def test_vector_operators():
    """Test the updated operators on the vector object."""
    v0 = measurable.Vector([3.0, 6.0], 'm')
    v1 = measurable.Vector([1.0, 3.0], 'm')
    v2 = measurable.Vector([1.0, 3.0], 'J')
    assert vectors_equal(v0 + v1, measurable.Vector([4.0, 9.0], 'm'))
    assert vectors_equal(v0 - v1, measurable.Vector([2.0, 3.0], 'm'))
    assert vectors_equal(v0 * v1, measurable.Vector([3.0, 18.0], 'm^2'))
    assert vectors_equal(v0 / v1, measurable.Vector([3.0, 2.0], '1'))
    assert vectors_equal(v0 / v2, measurable.Vector([3.0, 2.0], 'm / J'))
    assert vectors_equal(v0 ** 2, measurable.Vector([9.0, 36.0], 'm^2'))
    assert vectors_equal(10.0 * v0, measurable.Vector([30.0, 60.0], 'm'))
    assert vectors_equal(v0 * 10.0, measurable.Vector([30.0, 60.0], 'm'))
    assert vectors_equal(v0 / 10.0, measurable.Vector([0.3, 0.6], 'm'))
    with pytest.raises(TypeError):
        1.0 / v0
    with pytest.raises(measurable.ComparisonError):
        v0 + v2


@pytest.mark.scalar
def test_scalar_display():
    """Test the results of str(self) and repr(self) for a scalar."""
    scalar = measurable.Scalar(1.234, unit='m')
    assert str(scalar) == "1.234 [m]"
    assert repr(scalar).endswith("Scalar(1.234, unit='m')")
    scalar.unit('cm')
    assert str(scalar) == "123.4 [cm]"
    assert repr(scalar).endswith("Scalar(123.4, unit='cm')")


@pytest.mark.vector
def test_vector_display():
    """Test the results of str(self) and repr(self) for a vector."""
    vector = measurable.Vector(1.234, unit='m')
    assert str(vector) == "[1.234] [m]"
    assert repr(vector).endswith("Vector([1.234], unit='m')")
    vector.unit('cm')
    assert str(vector) == "[123.4] [cm]"
    assert repr(vector).endswith("Vector([123.4], unit='cm')")


@pytest.mark.vector
def test_vector_init():
    """Test initializing with iterable and non-iterable values."""
    expected = sorted(measurable.Vector([1.1], 'm'))
    assert sorted(measurable.Vector(1.1, 'm')) == expected


@pytest.mark.scalar
def test_scalar_unit():
    """Get and set the unit on a Scalar."""
    check_units(measurable.Scalar, 1, 'm', 'cm')


@pytest.mark.vector
def test_vector_unit():
    """Get and set the unit on a Vector."""
    check_units(measurable.Vector, [1, 2], 'm', 'cm')


Obj = typing.TypeVar(
    'Obj',
    typing.Type[measurable.Scalar],
    typing.Type[measurable.Vector],
)
Obj = typing.Union[
    typing.Type[measurable.Scalar],
    typing.Type[measurable.Vector],
]


def check_units(
    obj: Obj,
    amount: measurable.RealValued,
    reference: str,
    new: str,
) -> None:
    """Extracted for testing the unit attribute on Measured subclasses."""
    original = obj(amount, reference)
    assert original.unit() == reference
    updated = original.unit(new)
    assert updated is original
    assert updated.unit() == new
    factor = metric.Unit(new) // metric.Unit(reference)
    equal = (
        vectors_equal if isinstance(updated, measurable.Vector)
        else operator.eq
    )
    assert equal(updated, obj(rescale(amount, factor), new))
    assert obj(amount).unit() == '1'


def rescale(amount, factor):
    """Multiply amount by factor."""
    if isinstance(amount, numbers.Number):
        return factor * amount
    if isinstance(amount, typing.Iterable):
        return [factor * value for value in amount]


def vectors_equal(v0: measurable.Vector, v1: measurable.Vector):
    """Helper function for determining if two vectors are equal."""
    return all(v0 == v1)


def test_quantified_bool():
    """Quantified objects are always truthy."""
    class Quantified(measurable.OperatorMixin, measurable.Quantity):
        """Concrete version of `~measurable.Quantity` for testing."""
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
    def f0(a: Score, b: Score):
        return a.points + b.points

    # Instances must have the same kind:
    @measurable.same('kind')
    def f1(*args):
        return f0(*args)

    # Instances must have the same kind and the same name:
    @measurable.same('kind', 'name')
    def f2(*args):
        return f0(*args)

    # Add two instances with no restrictions.
    assert f0(scores[0], scores[2]) == 3.0

    # Add two instances with restricted kind.
    assert f1(scores[0], scores[1]) == 5.0

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


