import operator
import numbers
import math
import typing

import numpy as np
import pytest

from goats.core import quantities


def test_defined_conversions():
    """Test the collection of defined conversions."""
    assert len(quantities.CONVERSIONS) == 2 * len(quantities._CONVERSIONS)
    for (u0, u1), wt in quantities._CONVERSIONS.items():
        assert (u0, u1) in quantities.CONVERSIONS
        assert quantities.CONVERSIONS.get_weight(u0, u1) == wt
        assert quantities.CONVERSIONS.get_weight(u1, u0) == 1 / wt


def test_conversion_class():
    """Test the unit-conversion class"""
    cases = {
        # Length (common and simple)
        ('m', 'm'): 1.0, # trivial conversion
        ('m', 'cm'): 1e2, # defined metric-system conversion
        ('m', 'km'): 1e-3, # base-unit rescale
        ('m', 'mm'): 1e3, # (same)
        ('mm', 'm'): 1e-3, # (same)
        ('m', 'au'): 1 / 1.495978707e11, # to non-system unit
        ('au', 'm'): 1.495978707e11, # from non-system unit
        ('au', 'au'): 1.0, # trivial non-system conversion
        # Momentum (requires algebraic expressions)
        ('kg * m / s', 'g * cm / s'): 1e5, # defined (forward)
        ('g * cm / s', 'kg * m / s'): 1e-5, # defined (reverse)
        ('g * km / s', 'g * cm / s'): 1e5, # undefined (forward)
        ('g * cm / s', 'g * km / s'): 1e-5, # undefined (reverse)
        ('g * km / day', 'g * cm / s'): 1e5 / 86400, # undefined (forward)
        ('g * cm / s', 'g * km / day'): 86400 / 1e5, # undefined (reverse)
        # Energy (has multiple defined conversions)
        ('J', 'erg'): 1e7,
        ('eV', 'J'): 1.6022e-19,
        ('erg', 'J',): 1e-7, # reverse conversion
        ('J', 'eV'): 1 / 1.6022e-19, # (same)
        ('eV', 'erg'): 1.6022e-12, # chained conversion
        ('erg', 'eV'): 1 / 1.6022e-12, # (same)
        ('MeV', 'J'): 1.6022e-13, # conversion with rescale
        ('J', 'MeV'): 1 / 1.6022e-13, # (same)
        ('MeV', 'erg'): 1.6022e-6, # chained conversion with rescale
        ('erg', 'MeV'): 1 / 1.6022e-6, # (same)
        # Energy density (requires building quantity from formula)
        ('J / m^3', 'erg / cm^3'): 1e1,
        ('erg / cm^3', 'J / m^3'): 1e-1,
        ('J m^-3', 'erg cm^-3'): 1e1,
        ('m^-3 J', 'erg cm^-3'): 1e1,
        ('J m^-3', 'cm^-3 erg'): 1e1,
        # Capacitance
        ('F', 'cm'): 2.99792458e10**2 * 1e-9, # defininition has non-base unit
        # Velocity
        ('km/s', 'm/s'): 1e3,
        ('km/h', 'km/s'): 1/3600, # non-system time unit
        ('m/h', 'cm/s'): 1/36, # variation on above
        # Common or pathological conversions
        ('G', 'nT'): 1e5, # `B` units: cgs -> sim
        ('nT', 'G'): 1e-5, # `B` units: cgs -> sim
        ('s^3/km^6', 's^3/cm^6'): 1e-30, # `dist`: sim -> mks
        ('ms^3 m^-2', 'km^-2 s^3'): 1e-3, # terms in different order
        ('ms^3 m^-2', 's^3 km^-2'): 1e-3, # above conversion, but in order
        ('s^3 m^-6', 'km^-6 s^3'): 1e18, # different order; `dist` units
        (
            'm^-2 sr^-1 s^-1 J^-1',
            'cm^-2 sr^-1 s^-1 (MeV/nuc)^-1',
        ): 1.6022e-17 # `flux`: includes 'nuc' (dimensionless)
    }
    for (u0, u1), factor in cases.items():
        check_conversion(u0, u1, factor)

def check_conversion(u0: str, u1: str, factor: float):
    conversion = quantities.Conversion(u0, u1)
    assert conversion.factor == pytest.approx(factor)


def test_quantity_convert():
    """Test conversions with substitution within a quantity."""
    cases = {
        'length': {
            ('cm', 'mks'): 1e-2,
            ('m', 'cgs'): 1e2,
            ('mks', 'cm'): 1e2,
            ('cgs', 'm'): 1e-2,
            ('mks', 'cgs'): 1e2,
            ('cgs', 'mks'): 1e-2,
        },
        'momentum': {
            ('mks', 'cgs'): 1e5,
            ('cgs', 'mks'): 1e-5,
        },
        'energy': {
            ('mks', 'cgs'): 1e7,
            ('cgs', 'mks'): 1e-7,
            ('eV', 'mks'): 1.6022e-19,
            ('mks', 'eV'): 1 / 1.6022e-19,
            ('eV', 'cgs'): 1.6022e-12,
            ('cgs', 'eV'): 1 / 1.6022e-12,
        },
        'energy density': {
            ('mks', 'cgs'): 1e1,
            ('cgs', 'mks'): 1e-1,
        },
    }
    for name, conversion in cases.items():
        for (u0, u1), expected in conversion.items():
            quantity = quantities.Quantity(name)
            result = quantity.convert(u0).to(u1)
            assert result == pytest.approx(expected)


def test_singletons():
    """Make sure certain objects have a single reference."""
    cases = {
        quantities.Property: ['units'],
        quantities.NamedUnit: ['m', 'meter'],
        quantities.Conversion: [('G', 'T')],
        quantities._Converter: [('m', 'length')],
        quantities.Quantity: ['energy', 'Energy'],
        quantities.Unit: ['m s^-1'],
        quantities.MetricSystem: ['mks', 'MKS'],
    }
    for obj, args in cases.items():
        reference = build_singleton(obj, args[0])
        for arg in args:
            instance = build_singleton(obj, arg)
            assert instance is reference


def build_singleton(obj, arg):
    """Helper for test_singletons."""
    return obj(*arg) if isinstance(arg, tuple) else obj(arg)


def test_dimension():
    """Test the Dimension class."""
    cases = [
        {
            'unit': 'm',
            'quantity': 'length',
            'forms': ['L'],
        },
        {
            'unit': 'm / s',
            'quantity': 'velocity',
            'forms': ['L T^-1', 'T^-1 L'],
        },
        {
            'unit': 'km / s',
            'quantity': 'velocity',
            'forms': ['L T^-1', 'T^-1 L'],
        },
        {
            'unit': 'J',
            'quantity': 'energy',
            'forms': ['M L^2 T^-2', 'M T^-2 L^2', 'L^2 M T^-2'],
        },
    ]
    for current in cases:
        unit = quantities.Unit(current['unit'])
        quantity = quantities.Quantity(current['quantity'])
        forms = current['forms']
        for target in (unit, quantity['mks']):
            dimension = quantities.Dimension(target)
            assert isinstance(dimension, quantities.Dimension)
            for form in forms:
                assert dimension == form


def test_named_unit_knows_about():
    """Test the convenience method for testing possible instances."""
    for unit in quantities.named_units:
        assert quantities.NamedUnit.knows_about(unit)
    for unit in ['m^2', 'm / s', 'H / m', 'dogs^2 * cats']:
        assert not quantities.NamedUnit.knows_about(unit)


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
    """Calling u0 // u1 should compute the relative magnitude."""
    cases = {
        ('cm', 'm'): 1e2,
        ('m', 'cm'): 1e-2,
        ('cm', 'cm'): 1.0,
        ('km', 'm'): 1e-3,
        ('m', 'km'): 1e3,
    }
    for (s0, s1), expected in cases.items():
        u0 = quantities.NamedUnit(s0)
        u1 = quantities.NamedUnit(s1)
        u0_per_u1 = u0 // u1 # defined between instances
        assert u0_per_u1 == pytest.approx(expected)
        u0_per_s1 = u0 // s1 # defined for instance // string
        assert u0_per_s1 == pytest.approx(expected)
        s0_per_u1 = s0 // u1 # defined for string // instance
        assert s0_per_u1 == pytest.approx(expected)
    with pytest.raises(ValueError):
        u0 = quantities.NamedUnit('m')
        u1 = quantities.NamedUnit('J')
        u0 // u1 # not defined for different base units


def test_named_unit_decompose():
    """Test the decomposed property of NamedUnit."""
    cases = [
        ('m', [(1e0, 'm', 1)]),
        ('cm', [(1e-2, 'm', 1)]),
        ('km', [(1e3, 'm', 1)]),
        ('J', [(1e3, 'g', 1), (1e0, 'm', 2), (1e0, 's', -2)]),
        ('mJ', [(1e0, 'g', 1), (1e-3, 'm', 2), (1e-3, 's', -2)]),
        ('erg', [(1e0, 'g', 1), (1e-2, 'm', 2), (1e0, 's', -2)]),
        ('merg', [(1e-3, 'g', 1), (1e-5, 'm', 2), (1e-3, 's', -2)]),
    ]
    for (unit, expected) in cases:
        decomposed = quantities.NamedUnit(unit).decomposed
        result = [
            (float(term.coefficient), term.base, int(term.exponent))
            for term in decomposed
        ]
        assert result == expected


def test_unit_algebra():
    """Test algebraic operations on the Unit class."""
    u0 = quantities.Unit('m')
    u1 = quantities.Unit('J')
    assert u0**2 is not u0
    assert u0 * u1 == quantities.Unit('m * J')
    assert u0 / u1 == quantities.Unit('m / J')
    assert u0**2 / u1**3 == quantities.Unit('m^2 / J^3')
    assert (u0 / u1)**2 == quantities.Unit('m^2 / J^2')


def test_unit_floordiv():
    """Test conversion with the Unit class."""
    unit = quantities.Unit('m')
    assert quantities.Unit('cm') // unit == 1e2
    assert unit // 'cm' == 1e-2
    assert 'cm' // unit == 1e2
    unit = quantities.Unit('m / s')
    assert unit // 'km / h' == pytest.approx(1e3 / 3600)
    assert 'km / h' // unit == pytest.approx(3600 / 1e3)


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


def test_measured_bool():
    """Test the truthiness of a measured object."""
    cases = [
        quantities.Measured(1),
        quantities.Measured(1, 'm'),
        quantities.Measured(0),
        quantities.Measured(0, 'm'),
    ]
    for case in cases:
        assert bool(case)


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
        measured = quantities.measure(case['test'])
        assert isinstance(measured, quantities.Measurement)


def test_measurement():
    """Test the measurement object on its own."""
    values = [1.1, 2.3]
    unit = 'm'
    measurement = quantities.Measurement(values, unit)
    assert isinstance(measurement, quantities.Measurement)
    assert isinstance(measurement, quantities.Vector)
    assert measurement.values == values
    assert measurement.unit == unit
    assert len(measurement) == len(values)
    for i, value in enumerate(values):
        assert measurement[i] == quantities.Scalar(value, unit)


def test_single_valued_measurement():
    """Test special properties of a single-valued measurement."""
    unit = 'm'
    values = [1.1]
    measurement = quantities.Measurement(values, unit)
    assert float(measurement) == float(values[0])
    assert int(measurement) == int(values[0])
    values = [1.1, 2.3]
    measurement = quantities.Measurement(values, unit)
    with pytest.raises(TypeError):
        float(measurement)
    with pytest.raises(TypeError):
        int(measurement)


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
        'mks': [
            ('quantity', 'length', 'm'),
            ('dimension', 'L', 'm'),
            ('unit', 'au', 'm'),
            ('dimension', '1', '1'),
            ('unit', '1', '1'),
            ('unit', 'erg', 'J'),
        ],
        'cgs': [
            ('quantity', 'length', 'cm'),
            ('dimension', 'L', 'cm'),
            ('unit', 'au', 'cm'),
            ('dimension', '1', '1'),
            ('unit', '1', '1'),
            ('unit', 'J', 'erg'),
        ],
    }
    for name, cases in systems.items():
        system = quantities.MetricSystem(name)
        for (key, test, expected) in cases:
            search = {key: test}
            assert system.get_unit(**search) == expected


def test_system_singleton():
    """Metric systems should be singletons of their lower-case name."""
    for system in ('mks', 'cgs'):
        old = quantities.MetricSystem(system)
        new = quantities.MetricSystem(old)
        assert new is old
