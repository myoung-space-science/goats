import operator
import numbers
import math
import typing

import pytest

from goats.core import measurables
from goats.core import metric


def test_same():
    """Test the decorator class that enforces object consistency.

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
    @measurables.same('kind')
    def f1(*args):
        return f0(*args)

    # Instances must have the same kind and the same name:
    @measurables.same('kind', 'name')
    def f2(*args):
        return f0(*args)

    # Add two instances with no restrictions.
    assert f0(scores[0], scores[2]) == 3.0

    # Add two instances with restricted kind.
    assert f1(scores[0], scores[1]) == 5.0

    # Try to add two instances with different kind.
    with pytest.raises(measurables.ComparisonError):
        f1(scores[0], scores[2])

    # Try to add an instance to a built-in float.
    assert f1(scores[0], 2.0) == NotImplemented

    # Add two instances with restricted kind and name.
    assert f2(scores[1], scores[3]) == 9.0

    # Try to add two instances with same kind but different name.
    with pytest.raises(measurables.ComparisonError):
        f2(scores[0], scores[1])


def test_measured_operators():
    """Test comparison and arithmetic on measured objects."""
    meters = metric.Unit('m')
    joules = metric.Unit('J')
    q0 = measurables.Measured(4, meters)
    q1 = measurables.Measured(5, meters)
    q2 = measurables.Measured(3, meters)
    q3 = measurables.Measured(3, joules)
    assert q0 < q1
    assert q0 <= q1
    assert q0 > q2
    assert q0 >= q2
    assert q0 == measurables.Measured(4, meters)
    assert q0 != q1
    with pytest.raises(TypeError):
        q0 <= 3
    assert abs(q0) == measurables.Measured(4, meters)
    assert -q0 == measurables.Measured(-4, meters)
    assert +q0 == measurables.Measured(4, meters)
    assert q0 + q1 == measurables.Measured(9, meters)
    assert q0 / q3 == measurables.Measured(4 / 3, meters / joules)
    assert q0 * q2 == measurables.Measured(12, meters**2)
    assert q0**2 / q3 == measurables.Measured(16 / 3, meters**2 / joules)
    assert q0**2 / 2 == measurables.Measured(8, meters**2)
    with pytest.raises(TypeError):
        2 / q0
    assert q0.unit('cm') == measurables.Measured(400, 'cm')


def test_measured_bool():
    """Test the truthiness of a measured object."""
    cases = [
        measurables.Measured(1),
        measurables.Measured(1, 'm'),
        measurables.Measured(0),
        measurables.Measured(0, 'm'),
    ]
    for case in cases:
        assert bool(case)


@pytest.mark.scalar
def test_scalar_operators():
    """Test comparison and arithmetic on scalar objects."""
    _value_ = 2.0
    scalar = measurables.Scalar(_value_, '1')
    _unit_ = scalar.unit()
    assert scalar < measurables.Scalar(3, _unit_)
    assert scalar <= measurables.Scalar(3, _unit_)
    assert scalar <= measurables.Scalar(_value_, _unit_)
    assert scalar == measurables.Scalar(_value_, _unit_)
    assert scalar != measurables.Scalar(3, _unit_)
    assert scalar > measurables.Scalar(1, _unit_)
    assert scalar >= measurables.Scalar(1, _unit_)
    assert scalar >= measurables.Scalar(_value_, _unit_)

    f_scalar = float(scalar)
    assert isinstance(f_scalar, float)
    assert f_scalar == 2.0
    i_scalar = int(scalar)
    assert isinstance(i_scalar, int)
    assert i_scalar == 2

    # unary operations that preserve numerical precision
    ops = [
        operator.neg,
        operator.pos,
        abs,
    ]
    for op in ops:
        result = op(scalar)
        assert result == measurables.Scalar(op(_value_), _unit_)

    # unary operations that change numerical precision
    ops = [
        round,
        math.trunc,
        math.floor,
        math.ceil,
    ]
    for op in ops:
        result = op(scalar)
        assert result == op(_value_)
        assert isinstance(result, int)

    # binary operations
    number = 1.1 * _value_
    instance = measurables.Scalar(number, _unit_)
    # additive
    ops = [
        operator.add, # valid between instances
        operator.sub, # valid between instances
    ]
    # with an instance
    other = instance
    for op in ops:
        # forward
        result = op(scalar, other)
        expected = measurables.Scalar(
            op(_value_, float(other)),
            _unit_,
        )
        assert result == expected
        # reverse
        result = op(other, scalar)
        expected = measurables.Scalar(
            op(float(other), _value_),
            _unit_,
        )
        assert result == expected
    # with a number
    other = number
    for op in ops:
        # forward
        with pytest.raises(TypeError):
            op(scalar, other)
        # reverse
        with pytest.raises(TypeError):
            op(other, scalar)
    # multiplicative
    ops = [
        operator.mul, # valid between instances; symmetric with numbers
        operator.truediv, # valid between instances; right-sided with numbers
    ]
    # with an instance
    other = instance
    for op in ops:
        # forward
        result = op(scalar, other)
        expected = measurables.Scalar(
            op(_value_, float(other)),
            op(_unit_, other.unit()),
        )
        assert result == expected
        # reverse
        result = op(other, scalar)
        expected = measurables.Scalar(
            op(float(other), _value_),
            op(other.unit(), _unit_),
        )
        assert result == expected
    # with a number
    other = number
    for op in ops:
        # forward
        result = op(scalar, other)
        expected = measurables.Scalar(
            op(_value_, other),
            _unit_,
        )
        assert result == expected
        # reverse
        if op == operator.mul:
            result = op(other, scalar)
            expected = measurables.Scalar(
                op(other, _value_),
                _unit_,
            )
            assert result == expected
        else:
            with pytest.raises(TypeError):
                op(other, scalar)
    # exponential
    op = operator.pow # right-sided with numbers
    # with an instance
    other = instance
    # forward
    with pytest.raises(TypeError):
        op(scalar, other)
    # reverse
    with pytest.raises(TypeError):
        op(other, scalar)
    # with a number
    other = number
    # forward
    result = op(scalar, other)
    expected = measurables.Scalar(
        op(_value_, other),
        op(_unit_, other),
    )
    assert result == expected
    # reverse
    with pytest.raises(TypeError):
        op(other, scalar)

    # in-place: same as forward (immutable)
    number = 1.1 *_value_
    instance = measurables.Scalar(number,_unit_)
    # additive
    ops = [
        operator.iadd, # valid between instances
        operator.isub, # valid between instances
    ]
    # with an instance
    other = instance
    for op in ops:
        result = op(scalar, other)
        assert float(result) == op(_value_, float(other))
        assert result.unit() == _unit_
        scalar = measurables.Scalar(_value_, _unit_)
    # with a number
    other = number
    for op in ops:
        with pytest.raises(TypeError):
            op(scalar, other)
        scalar = measurables.Scalar(_value_, _unit_)
    # multiplicative
    ops = [
        operator.imul, # valid between instances; symmetric with numbers
        operator.itruediv, # valid between instances; right-sided with numbers
    ]
    # with an instance
    other = instance
    for op in ops:
        result = op(scalar, other)
        assert float(result) == op(_value_, float(other))
        assert result.unit() == op(_unit_, other.unit())
        scalar = measurables.Scalar(_value_, _unit_)
    # with a number
    other = number
    for op in ops:
        result = op(scalar, other)
        assert float(result) == op(_value_, other)
        assert result.unit() == _unit_
        scalar = measurables.Scalar(_value_, _unit_)
    # exponential
    op = operator.ipow # right-sided with numbers
    # with an instance
    other = instance
    with pytest.raises(TypeError):
        op(scalar, other)
    scalar = measurables.Scalar(_value_, _unit_)
    # with a number
    other = number
    result = op(scalar, other)
    assert float(result) == op(_value_, other)
    assert result.unit() == op(_unit_, other)
    scalar = measurables.Scalar(_value_, _unit_)

    # must be hashable
    assert isinstance(hash(scalar), int)

    # bitwise comparison is undefined
    with pytest.raises(TypeError):
        scalar & 1
        scalar | 1
        scalar ^ 1


@pytest.mark.scalar
def test_scalar_number_comparisons():
    """Test comparisons between a Scalar instance and a number."""
    _value_ = 2.0
    scalar = measurables.Scalar(_value_, '1')
    assert scalar < 3
    assert scalar <= 3
    assert scalar <= _value_
    assert scalar != 3
    assert scalar > 1
    assert scalar >= 1
    assert scalar >= _value_
    assert 3 > scalar
    assert 3 >= scalar
    assert _value_ >= scalar
    assert 3 != scalar
    assert 1 < scalar
    assert 1 <= scalar
    assert _value_ <= scalar


@pytest.mark.vector
def test_vector_operators():
    """Test the updated operators on the vector object."""
    v0 = measurables.Vector([3.0, 6.0], 'm')
    v1 = measurables.Vector([1.0, 3.0], 'm')
    v2 = measurables.Vector([1.0, 3.0], 'J')
    assert v0 + v1 == measurables.Vector([4.0, 9.0], 'm')
    assert v0 - v1 == measurables.Vector([2.0, 3.0], 'm')
    assert v0 * v1 == measurables.Vector([3.0, 18.0], 'm^2')
    assert v0 / v1 == measurables.Vector([3.0, 2.0], '1')
    assert v0 / v2 == measurables.Vector([3.0, 2.0], 'm / J')
    assert v0 ** 2 == measurables.Vector([9.0, 36.0], 'm^2')
    assert 10.0 * v0 == measurables.Vector([30.0, 60.0], 'm')
    assert v0 * 10.0 == measurables.Vector([30.0, 60.0], 'm')
    assert v0 / 10.0 == measurables.Vector([0.3, 0.6], 'm')
    with pytest.raises(TypeError):
        1.0 / v0
    with pytest.raises(measurables.ComparisonError):
        v0 + v2


@pytest.mark.vector
def test_vector_init():
    """Test initializing with iterable and non-iterable values."""
    expected = sorted(measurables.Vector([1.1], 'm'))
    assert sorted(measurables.Vector(1.1, 'm')) == expected


@pytest.mark.scalar
def test_scalar_unit():
    """Get and set the unit on a Scalar."""
    check_units(measurables.Scalar, 1, 'm', 'cm')


@pytest.mark.vector
def test_vector_unit():
    """Get and set the unit on a Vector."""
    check_units(measurables.Vector, [1, 2], 'm', 'cm')


Obj = typing.TypeVar(
    'Obj',
    typing.Type[measurables.Scalar],
    typing.Type[measurables.Vector],
)
Obj = typing.Union[
    typing.Type[measurables.Scalar],
    typing.Type[measurables.Vector],
]
def check_units(
    obj: Obj,
    amount: measurables.RealValued,
    reference: str,
    new: str,
) -> None:
    """Extracted for testing the unit attribute on Measured subclasses."""
    original = obj(amount, reference)
    assert original.unit() == reference
    updated = original.unit(new)
    assert updated is not original
    assert updated.unit() == new
    factor = metric.Unit(new) // metric.Unit(reference)
    assert updated == obj(rescale(amount, factor), new)
    assert obj(amount).unit() == '1'


def rescale(amount, factor):
    """Multiply amount by factor."""
    if isinstance(amount, numbers.Number):
        return factor * amount
    if isinstance(amount, typing.Iterable):
        return [factor * value for value in amount]


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
        'full': (1.1, 2.3, 5.8,meter),
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
        result = measurables.parse_measurable(case['test'])
        expected = case['full']
        assert result == expected
    for case in builtin_cases:
        result = measurables.parse_measurable(case['test'], distribute=True)
        expected = case['dist']
        assert result == expected
    assert measurables.parse_measurable(0) == (0, '1') # zero is measurable!
    with pytest.raises(measurables.Unmeasurable):
        measurables.parse_measurable(None)
    with pytest.raises(measurables.MeasuringTypeError):
        measurables.parse_measurable([1.1, 'm', 2.3, 'cm'])
    with pytest.raises(measurables.MeasuringTypeError):
        measurables.parse_measurable([(1.1, 'm'), (2.3, 5.8, 'cm')])


def test_measurable():
    """Test the function that determines if we can measure and object."""
    cases = [case['test'] for case in builtin_cases]
    for case in cases:
        assert measurables.measurable(case)
    class Test:
        def __measure__(): ...
    assert measurables.measurable(Test())


def test_measure():
    """Test the function that creates a measurement object."""
    for case in builtin_cases:
        measured = measurables.measure(case['test'])
        assert isinstance(measured, measurables.Measurement)


def test_measurement():
    """Test the measurement object on its own."""
    values = [1.1, 2.3]
    unit = 'm'
    measurement = measurables.Measurement(values, unit)
    assert isinstance(measurement, measurables.Measurement)
    assert isinstance(measurement, measurables.Vector)
    assert measurement.values == values
    assert measurement.unit == unit
    assert len(measurement) == len(values)
    for i, value in enumerate(values):
        assert measurement[i] == measurables.Scalar(value, unit)


def test_single_valued_measurement():
    """Test special properties of a single-valued measurement."""
    unit = 'm'
    values = [1.1]
    measurement = measurables.Measurement(values, unit)
    assert float(measurement) == float(values[0])
    assert int(measurement) == int(values[0])
    values = [1.1, 2.3]
    measurement = measurables.Measurement(values, unit)
    with pytest.raises(TypeError):
        float(measurement)
    with pytest.raises(TypeError):
        int(measurement)


