import operator
import numbers
import math
import typing

import numpy as np
import pytest

from goats.common import quantities


def test_conversion_factor():
    cases = {
        'momentum': {
            ('kg * m / s', 'g * cm / s'): 1e5,
            ('g * cm / s', 'kg * m / s'): 1e-5,
            ('kg * m / s', 'kg * m / s'): 1.0,
            ('mks', 'g * cm / s'): 1e5, # Tests unit-system substitution
            ('cgs', 'kg * m / s'): 1e-5, # (same)
            ('mks', 'cgs'): 1e5, # (same)
            ('cgs', 'mks'): 1e-5, # (same)
        },
        'energy': {
            ('J', 'erg'): 1e7,
            ('eV', 'J'): 1.6022e-19,
            ('erg', 'J',): 1e-7, # Tests reverse conversion
            ('J', 'eV'): 1 / 1.6022e-19, # (same)
            ('eV', 'erg'): 1.6022e-12, # Tests chained search
            ('erg', 'eV'): 1 / 1.6022e-12, # (same)
        },
        # 'energy density': { # Tests formulaic conversion
        #     ('J / m^3', 'erg / cm^3'): 1e1,
        #     ('erg / cm^3', 'J / m^3'): 1e-1,
        # },
        None: { # This will test the branch that searches all quantities
            ('kg * m / s', 'g * cm / s'): 1e5,
            ('J', 'erg'): 1e7,
            ('eV', 'erg'): 1.6022e-12,
        },
    }
    for quantity, conversion in cases.items():
        for pair, expected in conversion.items():
            result = quantities.get_conversion_factor(pair, quantity)
            assert result == pytest.approx(expected)


def test_get_quantity():
    cases = {
        'current': { # Tests retrieval
            'dimensions': {
                'mks': 'I',
                'cgs': '(M^1/2 * L^3/2) / T^2',
            },
            'units': {
                'mks': 'A',
                'cgs': 'statA',
            },
            'conversions': {
                ('A', 'statA'): 10*quantities.C,
            },
        },
        'current density': { # Tests construction
            'dimensions': {
                'mks': 'I L^-2',
                'cgs': 'M^1/2 L^-1/2 T^-2',
            },
            'units': {
                'mks': 'A m^-2',
                'cgs': 'statA cm^-2',
            },
            'conversions': {},
        }
    }
    for name, expected in cases.items():
        quantity = quantities.get_quantity(name)
        assert quantity == quantities.Quantity(**expected)


def test_build_named_unit():
    cases = {
        'm': { # A simple case
            'name': 'meter',
            'symbol': 'm',
            'scale': 1.0,
            'quantity': 'length',
            'dimension': 'L',
        },
        'cm': { # A non-unity metric scale
            'name': 'centimeter',
            'symbol': 'cm',
            'scale': 1e-2,
            'quantity': 'length',
            'dimension': 'L',
        },
        'J': { # A compound dimension
            'name': 'joule',
            'symbol': 'J',
            'scale': 1.0,
            'quantity': 'energy',
            'dimension': '(M * L^2) / T^2',
        },
    }
    for name, attrs in cases.items():
        unit = quantities.NamedUnit(name)
        for key, value in attrs.items():
            assert getattr(unit, key) == value
    with pytest.raises(quantities.UnitParsingError):
        quantities.NamedUnit('cat')


def test_named_unit_floordiv():
    cases = {
        ('cm', 'm'): 1e2,
        ('m', 'cm'): 1e-2,
        ('cm', 'cm'): 1.0,
        ('m', 'au'): 1.495978707e11,
        ('erg', 'J'): 1e7,
        ('J', 'erg'): 1e-7,
        ('erg', 'joule'): 1e7,
        ('erg', 'eV'): 1.6022e-12,
        ('erg', 'MeV'): 1.6022e-6,
    }
    for (s0, s1), expected in cases.items():
        u0 = quantities.NamedUnit(s0)
        u1 = quantities.NamedUnit(s1)
        u0_per_u1 = u0 // u1
        assert u0_per_u1 == pytest.approx(expected)
        u0_per_s1 = u0 // s1
        assert u0_per_s1 == pytest.approx(expected)
        s0_per_u1 = s0 // u1
        assert s0_per_u1 == pytest.approx(expected)


def test_unit():
    u0 = quantities.Unit('m')
    u1 = quantities.Unit('J')
    assert u0**2 is not u0
    assert u0 * u1 == quantities.Unit('m * J')
    assert u0 / u1 == quantities.Unit('m / J')
    assert u0**2 / u1**3 == quantities.Unit('m^2 / J^3')
    assert (u0 / u1)**2 == quantities.Unit('m^2 / J^2')
    assert quantities.Unit('cm') // u0 == 100
    assert u0.dimension == quantities.Dimension('L')
    assert u1.dimension == quantities.Dimension('M * L^2 / T^2')


# These were copied from test_units.py; there is significant overlap with other
# tests in this module.
strings = {
    'm': {
        'unit': 'm',
        'dimension': 'L',
    },
    'm / s': {
        'unit': 'm s^-1',
        'dimension': 'L T^-1',
    },
    '1 / s': {
        'unit': 's^-1',
        'dimension': 'T^-1',
    },
    '1 / s^2': {
        'unit': 's^-2',
        'dimension': 'T^-2',
    },
    's^3 / km^6': {
        'unit': 's^3 km^-6',
        'dimension': 'T^3 L^-6',
    },
    '# / (cm^2*s*sr*MeV/nuc)': {
        'unit': '# cm^-2 s^-1 sr^-1 (MeV nuc^-1)^-1',
        'dimension': 'L^-2 T^-1 (M L^2 T^-2 M^-1)^-1',
    },
    '# / ((cm^2*s*sr*MeV/nuc))': {
        'unit': '# cm^-2 s^-1 sr^-1 (MeV nuc^-1)^-1',
        'dimension': 'L^-2 T^-1 (M L^2 T^-2 M^-1)^-1',
    },
}
conversions = {
    ('km/s', 'm/s'): 1e3,
    ('km/h', 'km/s'): 1/3600,
    ('s^3/km^6', 's^3/cm^6'): 1e-30,
    ('m/h', 'cm/s'): 1/36,
    ('G', 'nT'): 1e5,
    ('nT', 'G'): 1e-5,
}
multiplications = {
    ('m', 's'): 'm*s',
    ('m/s', 'km/m'): 'km/s',
    ('m', 'm^-1'): '1',
}
divisions = {
    ('m', 's'): 'm/s',
    ('m/s', 'm/km'): 'km/s',
    ('m', 'm'): '1',
}
powers = {
    ('m', 2): 'm^2',
    ('m/s', 3): 'm^3 s^-3',
    ('J*s^2/m^3', -1): 'J^-1 s^-2 m^3',
}


def test_unit_init():
    """Initialize the Unit object with various strings."""
    for arg, expected in strings.items():
        unit = quantities.Unit(arg)
        assert unit == expected['unit']
        assert unit.dimension == quantities.Dimension(expected['dimension'])


def test_multiply():
    """Test the ability to create a new compound instance with '*'."""
    for (this, that), expected in multiplications.items():
        result = quantities.Unit(this) * quantities.Unit(that)
        assert isinstance(result, quantities.Unit)
        assert result == quantities.Unit(expected)
    result = quantities.Unit('m') / quantities.Unit('s')
    wrong = quantities.Unit('km*h')
    assert result != wrong


def test_divide():
    """Test the ability to create a new compound instance with '/'."""
    for (this, that), expected in divisions.items():
        result = quantities.Unit(this) / quantities.Unit(that)
        assert isinstance(result, quantities.Unit)
        assert result == quantities.Unit(expected)
    result = quantities.Unit('m') / quantities.Unit('s')
    wrong = quantities.Unit('km/h')
    assert result != wrong


def test_raise_to_power():
    """Test the ability to create a new compound instance with '**'."""
    for (this, that), expected in powers.items():
        result = quantities.Unit(this) ** that
        assert isinstance(result, quantities.Unit)
        assert result == quantities.Unit(expected)


def test_idempotence():
    """Make sure initializing with a `Unit` creates a new `Unit`."""
    old = quantities.Unit('m')
    new = quantities.Unit(old)
    assert isinstance(new, quantities.Unit)
    assert str(new) == str(old)
    assert repr(new) == repr(old)


def test_equality():
    """Test the definition of strict equality between instances."""
    assert quantities.Unit('m/s') == quantities.Unit('m/s')
    assert quantities.Unit('m/s') == quantities.Unit('m*s^-1')


def test_single_unit_parse():
    """Test the ability to handle arbitrary single units.

    Note that the class that manages single units is primarily an assistant to
    the `Unit` class, so full coverage is not necessary as long as `Unit` is
    well tested.
    """
    order, unit = quantities.NamedUnit.parse('m')
    assert order.symbol == ''
    assert order.name == ''
    assert order.factor == 1.0
    assert unit.symbol == 'm'
    assert unit.name == 'meter'
    assert unit.quantity == 'length'
    order, unit = quantities.NamedUnit.parse('mm')
    assert order.symbol == 'm'
    assert order.name == 'milli'
    assert order.factor == 1e-3
    assert unit.symbol == 'm'
    assert unit.name == 'meter'
    assert unit.quantity == 'length'
    symbolic = quantities.NamedUnit.parse('mm')
    named = quantities.NamedUnit.parse('millimeter')
    assert symbolic == named
    order, unit = quantities.NamedUnit.parse('lm')
    assert order.symbol == ''
    assert order.name == ''
    assert order.factor == 1.0
    assert unit.symbol == 'lm'
    assert unit.name == 'lumen'
    assert unit.quantity == 'luminous flux'
    order, unit = quantities.NamedUnit.parse('MeV')
    assert order.symbol == 'M'
    assert order.name == 'mega'
    assert order.factor == 1e6
    assert unit.symbol == 'eV'
    assert unit.name == 'electronvolt'
    assert unit.quantity == 'energy'
    order, unit = quantities.NamedUnit.parse('μeV')
    assert order.symbol == 'μ'
    assert order.name == 'micro'
    assert order.factor == 1e-6
    assert unit.symbol == 'eV'
    assert unit.name == 'electronvolt'
    assert unit.quantity == 'energy'
    order, unit = quantities.NamedUnit.parse('uerg')
    assert order.symbol == 'μ'
    assert order.name == 'micro'
    assert order.factor == 1e-6
    assert unit.symbol == 'erg'
    assert unit.name == 'erg'
    assert unit.quantity == 'energy'
    order, unit = quantities.NamedUnit.parse('statA')
    assert order.symbol == ''
    assert order.name == ''
    assert order.factor == 1.0
    assert unit.symbol == 'statA'
    assert unit.name == 'statampere'
    assert unit.quantity == 'current'


def test_single_unit_idempotence():
    """Make sure we can create a new instance from an existing instance."""
    old = quantities.NamedUnit('m')
    new = quantities.NamedUnit(old)
    assert isinstance(new, quantities.NamedUnit)
    assert new is old


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
    @quantities.same('kind')
    def f1(*args):
        return f0(*args)

    # Instances must have the same kind and the same name:
    @quantities.same('kind', 'name')
    def f2(*args):
        return f0(*args)

    # Add two instances with no restrictions.
    assert f0(scores[0], scores[2]) == 3.0

    # Add two instances with restricted kind.
    assert f1(scores[0], scores[1]) == 5.0

    # Try to add two instances with different kind.
    with pytest.raises(quantities.ComparisonError):
        f1(scores[0], scores[2])

    # Try to add an instance to a built-in float.
    assert f1(scores[0], 2.0) == NotImplemented

    # Add two instances with restricted kind and name.
    assert f2(scores[1], scores[3]) == 9.0

    # Try to add two instances with same kind but different name.
    with pytest.raises(quantities.ComparisonError):
        f2(scores[0], scores[1])


def test_measured_operators():
    """Test comparison and arithmetic on measured objects."""
    meters = quantities.Unit('m')
    joules = quantities.Unit('J')
    q0 = quantities.Measured(4, meters)
    q1 = quantities.Measured(5, meters)
    q2 = quantities.Measured(3, meters)
    q3 = quantities.Measured(3, joules)
    assert q0 < q1
    assert q0 <= q1
    assert q0 > q2
    assert q0 >= q2
    assert q0 == quantities.Measured(4, meters)
    assert q0 != q1
    with pytest.raises(TypeError):
        q0 <= 3
    assert abs(q0) == quantities.Measured(4, meters)
    assert -q0 == quantities.Measured(-4, meters)
    assert +q0 == quantities.Measured(4, meters)
    assert q0 + q1 == quantities.Measured(9, meters)
    assert q0 / q3 == quantities.Measured(4 / 3, meters / joules)
    assert q0 * q2 == quantities.Measured(12, meters**2)
    assert q0**2 / q3 == quantities.Measured(16 / 3, meters**2 / joules)
    assert q0**2 / 2 == quantities.Measured(8, meters**2)
    with pytest.raises(TypeError):
        2 / q0
    assert q0.unit('cm') == quantities.Measured(400, 'cm')


@pytest.mark.scalar
def test_scalar_operators():
    """Test comparison and arithmetic on scalar objects."""
    _value_ = 2.0
    scalar = quantities.Scalar(_value_, '1')
    _unit_ = scalar.unit()
    assert scalar < quantities.Scalar(3, _unit_)
    assert scalar <= quantities.Scalar(3, _unit_)
    assert scalar <= quantities.Scalar(_value_, _unit_)
    assert scalar == quantities.Scalar(_value_, _unit_)
    assert scalar != quantities.Scalar(3, _unit_)
    assert scalar > quantities.Scalar(1, _unit_)
    assert scalar >= quantities.Scalar(1, _unit_)
    assert scalar >= quantities.Scalar(_value_, _unit_)

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
        assert result == quantities.Scalar(op(_value_), _unit_)

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
    instance = quantities.Scalar(number, _unit_)
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
        expected = quantities.Scalar(
            op(_value_, float(other)),
            _unit_,
        )
        assert result == expected
        # reverse
        result = op(other, scalar)
        expected = quantities.Scalar(
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
        expected = quantities.Scalar(
            op(_value_, float(other)),
            op(_unit_, other.unit()),
        )
        assert result == expected
        # reverse
        result = op(other, scalar)
        expected = quantities.Scalar(
            op(float(other), _value_),
            op(other.unit(), _unit_),
        )
        assert result == expected
    # with a number
    other = number
    for op in ops:
        # forward
        result = op(scalar, other)
        expected = quantities.Scalar(
            op(_value_, other),
            _unit_,
        )
        assert result == expected
        # reverse
        if op == operator.mul:
            result = op(other, scalar)
            expected = quantities.Scalar(
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
    expected = quantities.Scalar(
        op(_value_, other),
        op(_unit_, other),
    )
    assert result == expected
    # reverse
    with pytest.raises(TypeError):
        op(other, scalar)

    # in-place: same as forward (immutable)
    number = 1.1 *_value_
    instance = quantities.Scalar(number,_unit_)
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
        scalar = quantities.Scalar(_value_, _unit_)
    # with a number
    other = number
    for op in ops:
        with pytest.raises(TypeError):
            op(scalar, other)
        scalar = quantities.Scalar(_value_, _unit_)
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
        scalar = quantities.Scalar(_value_, _unit_)
    # with a number
    other = number
    for op in ops:
        result = op(scalar, other)
        assert float(result) == op(_value_, other)
        assert result.unit() == _unit_
        scalar = quantities.Scalar(_value_, _unit_)
    # exponential
    op = operator.ipow # right-sided with numbers
    # with an instance
    other = instance
    with pytest.raises(TypeError):
        op(scalar, other)
    scalar = quantities.Scalar(_value_, _unit_)
    # with a number
    other = number
    result = op(scalar, other)
    assert float(result) == op(_value_, other)
    assert result.unit() == op(_unit_, other)
    scalar = quantities.Scalar(_value_, _unit_)

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
    scalar = quantities.Scalar(_value_, '1')
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
    v0 = quantities.Vector([3.0, 6.0], 'm')
    v1 = quantities.Vector([1.0, 3.0], 'm')
    v2 = quantities.Vector([1.0, 3.0], 'J')
    assert v0 + v1 == quantities.Vector([4.0, 9.0], 'm')
    assert v0 - v1 == quantities.Vector([2.0, 3.0], 'm')
    assert v0 * v1 == quantities.Vector([3.0, 18.0], 'm^2')
    assert v0 / v1 == quantities.Vector([3.0, 2.0], '1')
    assert v0 / v2 == quantities.Vector([3.0, 2.0], 'm / J')
    assert v0 ** 2 == quantities.Vector([9.0, 36.0], 'm^2')
    assert 10.0 * v0 == quantities.Vector([30.0, 60.0], 'm')
    assert v0 * 10.0 == quantities.Vector([30.0, 60.0], 'm')
    assert v0 / 10.0 == quantities.Vector([0.3, 0.6], 'm')
    with pytest.raises(TypeError):
        1.0 / v0
    with pytest.raises(quantities.ComparisonError):
        v0 + v2


@pytest.mark.vector
def test_vector_init():
    """Test initializing with iterable and non-iterable values."""
    expected = sorted(quantities.Vector([1.1], 'm'))
    assert sorted(quantities.Vector(1.1, 'm')) == expected


@pytest.mark.scalar
def test_scalar_unit():
    """Get and set the unit on a Scalar."""
    check_units(quantities.Scalar, 1, 'm', 'cm')


@pytest.mark.vector
def test_vector_unit():
    """Get and set the unit on a Vector."""
    check_units(quantities.Vector, [1, 2], 'm', 'cm')


Obj = typing.TypeVar(
    'Obj',
    typing.Type[quantities.Scalar],
    typing.Type[quantities.Vector],
)
Obj = typing.Union[
    typing.Type[quantities.Scalar],
    typing.Type[quantities.Vector],
]
def check_units(
    obj: Obj,
    amount: quantities.RealValued,
    reference: str,
    new: str,
) -> None:
    """Extracted for testing the unit attribute on Measured subclasses."""
    original = obj(amount, reference)
    assert original.unit() == reference
    updated = original.unit(new)
    assert updated is not original
    assert updated.unit() == new
    factor = quantities.Unit(new) // quantities.Unit(reference)
    assert updated == obj(rescale(amount, factor), new)
    assert obj(amount).unit() == '1'


def rescale(amount, factor):
    """Multiply amount by factor."""
    if isinstance(amount, numbers.Number):
        return factor * amount
    if isinstance(amount, typing.Iterable):
        return [factor * value for value in amount]


@pytest.mark.variable
def test_variable():
    """Test the object that represents a variable."""
    v0 = quantities.Variable([3.0, 4.5], 'm', ['x'])
    v1 = quantities.Variable([[1.0], [2.0]], 'J', ['x', 'y'])
    assert np.array_equal(v0, [3.0, 4.5])
    assert v0.unit() == quantities.Unit('m')
    assert list(v0.axes) == ['x']
    assert v0.naxes == 1
    assert np.array_equal(v1, [[1.0], [2.0]])
    assert v1.unit() == quantities.Unit('J')
    assert list(v1.axes) == ['x', 'y']
    assert v1.naxes == 2
    v0_cm = v0.unit('cm')
    assert v0_cm is not v0
    assert np.array_equal(v0_cm, 100 * v0)
    assert v0_cm.unit() == quantities.Unit('cm')
    assert v0_cm.axes == v0.axes
    r = v0 + v0
    expected = [6.0, 9.0]
    assert np.array_equal(r, expected)
    assert r.unit() == v0.unit()
    r = v0 * v1
    expected = [[3.0 * 1.0], [4.5 * 2.0]]
    assert np.array_equal(r, expected)
    assert r.unit() == quantities.Unit('m * J')
    r = v0 / v1
    expected = [[3.0 / 1.0], [4.5 / 2.0]]
    assert np.array_equal(r, expected)
    assert r.unit() == quantities.Unit('m / J')
    r = v0 ** 2
    expected = [3.0 ** 2, 4.5 ** 2]
    assert np.array_equal(r, expected)
    assert r.unit() == quantities.Unit('m^2')


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
def var(arr: typing.Dict[str, list]) -> typing.Dict[str, quantities.Variable]:
    """A tuple of test variables."""
    reference = quantities.Variable(
        arr['reference'].copy(),
        axes=('d0', 'd1'),
        unit='m',
    )
    samedims = quantities.Variable(
        arr['samedims'].copy(),
        axes=('d0', 'd1'),
        unit='kJ',
    )
    sharedim = quantities.Variable(
        arr['sharedim'].copy(),
        axes=('d1', 'd2'),
        unit='s',
    )
    different = quantities.Variable(
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
    var: typing.Dict[str, quantities.Variable],
    arr: typing.Dict[str, list],
) -> None:
    """Test the ability to multiply two quantities.Variable instances."""
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
            assert isinstance(new, quantities.Variable), msg
            expected = reduce(a0, a1, opr)
            assert np.array_equal(new, expected), msg
            assert sorted(new.axes) == case['axes'], msg
            algebraic = opr(v0.unit(), v1.unit())
            formatted = f'({v0.unit()}){sym}({v1.unit()})'
            for unit in (algebraic, formatted):
                assert new.unit() == unit, msg


@pytest.mark.variable
def test_variable_pow(var: typing.Dict[str, quantities.Variable]) -> None:
    """Test the ability to exponentiate a quantities.Variable instance."""
    opr = operator.pow
    v0 = var['reference']
    ex = 3
    msg = f"Failed for {opr}"
    new = opr(v0, ex)
    assert isinstance(new, quantities.Variable)
    expected = reduce(np.array(v0), ex, operator.pow)
    assert np.array_equal(new, expected), msg
    assert new.axes == var['reference'].axes, msg
    algebraic = opr(v0.unit(), 3)
    formatted = f'({v0.unit()})^{ex}'
    for unit in (algebraic, formatted):
        assert new.unit() == unit, msg


@pytest.mark.variable
def test_variable_add_sub(
    var: typing.Dict[str, quantities.Variable],
    arr: typing.Dict[str, list],
) -> None:
    """Test the ability to add two quantities.Variable instances."""
    v0 = var['reference']
    a0 = arr['reference']
    a1 = arr['samedims']
    v1 = quantities.Variable(a1, v0.unit(), v0.axes)
    v2 = quantities.Variable(arr['different'], v0.unit(), var['different'].axes)
    for opr in (operator.add, operator.sub):
        msg = f"Failed for {opr}"
        new = opr(v0, v1)
        expected = reduce(a0, a1, opr)
        assert isinstance(new, quantities.Variable), msg
        assert np.array_equal(new, expected), msg
        assert new.unit() == v0.unit(), msg
        assert new.axes == v0.axes, msg
        with pytest.raises(TypeError):
            opr(v0, v2)


@pytest.mark.variable
def test_variable_units(var: typing.Dict[str, quantities.Variable]):
    """Test the ability to update unit via bracket syntax."""
    v0_km = var['reference'].unit('km')
    assert isinstance(v0_km, quantities.Variable)
    assert v0_km is not var['reference']
    assert v0_km.unit() == 'km'
    assert v0_km.axes == var['reference'].axes
    assert np.array_equal(v0_km[:], 1e-3 * var['reference'][:])


@pytest.mark.variable
def test_numerical_operations(var: typing.Dict[str, quantities.Variable]):
    """Test operations between a quantities.Variable and a number."""

    # multiplication is symmetric
    new = var['reference'] * 10.0
    assert isinstance(new, quantities.Variable)
    expected = [
        # 3 x 2
        [+(1.0*10.0), +(2.0*10.0)],
        [+(2.0*10.0), -(3.0*10.0)],
        [-(4.0*10.0), +(6.0*10.0)],
    ]
    assert np.array_equal(new, expected)
    new = 10.0 * var['reference']
    assert isinstance(new, quantities.Variable)
    assert np.array_equal(new, expected)

    # division is right-sided
    new = var['reference'] / 10.0
    assert isinstance(new, quantities.Variable)
    expected = [
        # 3 x 2
        [+(1.0/10.0), +(2.0/10.0)],
        [+(2.0/10.0), -(3.0/10.0)],
        [-(4.0/10.0), +(6.0/10.0)],
    ]
    assert np.array_equal(new, expected)
    with pytest.raises(TypeError):
        10.0 / var['reference']

    # addition is right-sided
    new = var['reference'] + 10.0
    assert isinstance(new, quantities.Variable)
    expected = [
        # 3 x 2
        [+1.0+10.0, +2.0+10.0],
        [+2.0+10.0, -3.0+10.0],
        [-4.0+10.0, +6.0+10.0],
    ]
    assert np.array_equal(new, expected)
    with pytest.raises(TypeError):
        10.0 + var['reference']

    # subtraction is right-sided
    new = var['reference'] - 10.0
    assert isinstance(new, quantities.Variable)
    expected = [
        # 3 x 2
        [+1.0-10.0, +2.0-10.0],
        [+2.0-10.0, -3.0-10.0],
        [-4.0-10.0, +6.0-10.0],
    ]
    assert np.array_equal(new, expected)
    with pytest.raises(TypeError):
        10.0 - var['reference']


@pytest.mark.variable
def test_variable_array(var: typing.Dict[str, quantities.Variable]):
    """Natively convert a Variable into a NumPy array."""
    v = var['reference']
    assert isinstance(v, quantities.Variable)
    a = np.array(v)
    assert isinstance(a, np.ndarray)
    assert np.array_equal(v, a)


@pytest.mark.variable
def test_variable_getitem(var: typing.Dict[str, quantities.Variable]):
    """Subscript a Variable."""
    # reference = [
    #     [+1.0, +2.0],
    #     [+2.0, -3.0],
    #     [-4.0, +6.0],
    # ]
    v = var['reference']
    for sliced in (v[:], v[...]):
        assert isinstance(sliced, quantities.Variable)
        assert sliced is not v
        expected = np.array([[+1.0, +2.0], [+2.0, -3.0], [-4.0, +6.0]])
        assert np.array_equal(sliced, expected)
    assert v[0, 0] == quantities.Scalar(+1.0, v.unit())
    assert np.array_equal(v[0, :], [+1.0, +2.0])
    assert np.array_equal(v[:, 0], [+1.0, +2.0, -4.0])
    assert np.array_equal(v[:, 0:1], [[+1.0], [+2.0], [-4.0]])
    assert np.array_equal(v[(0, 1), :], [[+1.0, +2.0], [+2.0, -3.0]])
    expected = np.array([[+1.0, +2.0], [+2.0, -3.0], [-4.0, +6.0]])
    assert np.array_equal(v[:, (0, 1)], expected)


@pytest.mark.variable
def test_variable_name():
    """A variable may have a given name or be anonymous."""
    default = quantities.Variable([1], 'm', ['d0'])
    assert default.name == '<anonymous>'
    cases = {
        'test': 'test',
        None: '<anonymous>',
    }
    for name, expected in cases.items():
        variable = quantities.Variable([1], 'm', ['d0'], name=name)
        assert variable.name == expected


@pytest.mark.variable
def test_variable_get_array(var: typing.Dict[str, quantities.Variable]):
    """Test the internal `_get_array` method to prevent regression."""
    v = var['reference']
    a = v._get_array((0, 0))
    assert a.shape == ()
    assert a == 1.0
    assert v._array is None
    a = v._get_array(0)
    assert a.shape == (2,)
    assert np.array_equal(a, [1, 2])
    assert v._array is None
    a = v._get_array()
    assert a.shape == (3, 2)
    assert np.array_equal(a, [[+1.0, +2.0], [+2.0, -3.0], [-4.0, +6.0]])
    assert v._array is a


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
        result = quantities.parse_measurable(case['test'])
        expected = case['full']
        assert result == expected
    for case in builtin_cases:
        result = quantities.parse_measurable(case['test'], distribute=True)
        expected = case['dist']
        assert result == expected
    assert quantities.parse_measurable(0) == (0, '1') # zero is measurable!
    with pytest.raises(quantities.Unmeasurable):
        quantities.parse_measurable(None)
    with pytest.raises(quantities.MeasuringTypeError):
        quantities.parse_measurable([1.1, 'm', 2.3, 'cm'])
    with pytest.raises(quantities.MeasuringTypeError):
        quantities.parse_measurable([(1.1, 'm'), (2.3, 5.8, 'cm')])


def test_measure():
    """Test the function that creates a measurement object."""
    for case in builtin_cases:
        measurement = quantities.measure(case['test'])
        values = case['full'][:-1]
        unit = case['full'][-1]
        measured = quantities.Vector(values, unit=unit)
        expected = quantities.Measurement(measured)
        assert measurement == expected


class FakeScalar(typing.NamedTuple):
    """A test class that acts like a scalar."""
    value: numbers.Real
    unit: str


class FakeVector(typing.NamedTuple):
    """A test class that acts like a vector."""
    values: typing.Iterable[numbers.Real]
    unit: str


def test_measurement():
    """Test the measurement object on its own."""
    _values_ = [1.1, 2.3]
    _unit_ = 'm'
    cases = [
        {
            'object': quantities.Vector(_values_, unit=_unit_),
            'values': _values_,
            'unit': _unit_,
        },
        {
            'object': FakeVector(_values_, unit=_unit_),
            'values': _values_,
            'unit': _unit_,
        },
        {
            'object': FakeScalar(_values_[0], unit=_unit_),
            'values': [_values_[0]],
            'unit': _unit_,
        },
    ]
    for case in cases:
        measurement = quantities.Measurement(case['object'])
        values = case['values']
        unit = case['unit']
        assert isinstance(measurement, quantities.Measurement)
        assert isinstance(measurement, typing.Sequence)
        assert measurement.values == values
        assert measurement.unit == unit
        assert len(measurement) == len(values)
        for i, value in enumerate(values):
            assert measurement[i] == quantities.Scalar(value, unit)
        assert measurement.asvector == quantities.Vector(values, unit=unit)


def test_system():
    """Test the object that represents a system of quantities."""
    # Cases:
    # - length: same dimension; same unit.
    # - momentum: same dimension; different unit.
    # - magnetic induction: different dimension; different unit.
    systems = {
        'mks': {
            'length': {'dimension': 'L', 'unit': 'm'},
            'momentum': {'dimension': '(M * L) / T', 'unit': 'kg * m / s'},
            'magnetic induction': {'dimension': 'M / (T^2 * I)', 'unit': 'T'},
        },
        'cgs': {
            'length': {'dimension': 'L', 'unit': 'cm'},
            'momentum': {'dimension': '(M * L) / T', 'unit': 'g * cm / s'},
            'magnetic induction': {
                'dimension': 'M^1/2 / (L^1/2 * T)',
                'unit': 'G',
            },
        },
    }
    for name, cases in systems.items():
        lower = name.lower()
        upper = name.upper()
        assert quantities.MetricSystem(lower) == quantities.MetricSystem(upper)
        system = quantities.MetricSystem(lower)
        for key, definition in cases.items():
            assert system[key] == quantities.Metric(**definition)


def test_system_unit_lookup():
    """Test the ability to retrieve the appropriate unit."""
    systems = {
        'mks': {
            'quantity': ('length', 'm'),
            'dimension': ('L', 'm'),
            'unit': ('au', 'm'),
            'dimension': ('1', '1'),
            'unit': ('1', '1'),
        },
        'cgs': {
            'quantity': ('length', 'cm'),
            'dimension': ('L', 'cm'),
            'unit': ('au', 'cm'),
            'dimension': ('1', '1'),
            'unit': ('1', '1'),
        },
    }
    for name, cases in systems.items():
        system = quantities.MetricSystem(name)
        for key, (test, expected) in cases.items():
            search = {key: test}
            assert system.get_unit(**search) == expected
    with pytest.raises(KeyError):
        quantities.MetricSystem('mks').get_unit(unit='Erg')


def test_system_singleton():
    """Metric systems should be singletons of their lower-case name."""
    for system in ('mks', 'cgs'):
        old = quantities.MetricSystem(system)
        new = quantities.MetricSystem(old)
        assert new is old
