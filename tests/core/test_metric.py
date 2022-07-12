import operator
import numbers
import math
import typing

import pytest

from goats.core import metric


def test_defined_conversions():
    """Test the collection of defined conversions."""
    assert len(metric.CONVERSIONS) == 2 * len(metric._CONVERSIONS)
    for (u0, u1), wt in metric._CONVERSIONS.items():
        assert (u0, u1) in metric.CONVERSIONS
        assert metric.CONVERSIONS.get_weight(u0, u1) == wt
        assert metric.CONVERSIONS.get_weight(u1, u0) == 1 / wt


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
    conversion = metric.Conversion(u0, u1)
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
            quantity = metric.Quantity(name)
            result = quantity.convert(u0).to(u1)
            assert result == pytest.approx(expected)


def test_singletons():
    """Make sure certain objects have a single reference."""
    cases = {
        metric.Property: ['units'],
        metric.NamedUnit: ['m', 'meter'],
        metric.Conversion: [('G', 'T')],
        metric._Converter: [('m', 'length')],
        metric.Quantity: ['energy', 'Energy'],
        metric.Unit: ['m s^-1'],
        metric.System: ['mks', 'MKS'],
    }
    for obj, args in cases.items():
        reference = build_singleton(obj, args[0])
        for arg in args:
            instance = build_singleton(obj, arg)
            assert instance is reference


def build_singleton(obj, arg):
    """Helper for test_singletons."""
    return obj(*arg) if isinstance(arg, tuple) else obj(arg)


def test_named_unit_knows_about():
    """Test the convenience method for testing possible instances."""
    for unit in metric.named_units:
        assert metric.NamedUnit.knows_about(unit)
    for unit in ['m^2', 'm / s', 'H / m', 'dogs^2 * cats']:
        assert not metric.NamedUnit.knows_about(unit)


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
        unit = metric.NamedUnit(name)
        for key, value in attrs.items():
            assert getattr(unit, key) == value
    with pytest.raises(metric.UnitParsingError):
        metric.NamedUnit('cat')


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
        u0 = metric.NamedUnit(s0)
        u1 = metric.NamedUnit(s1)
        u0_per_u1 = u0 // u1 # defined between instances
        assert u0_per_u1 == pytest.approx(expected)
        u0_per_s1 = u0 // s1 # defined for instance // string
        assert u0_per_s1 == pytest.approx(expected)
        s0_per_u1 = s0 // u1 # defined for string // instance
        assert s0_per_u1 == pytest.approx(expected)
    with pytest.raises(ValueError):
        u0 = metric.NamedUnit('m')
        u1 = metric.NamedUnit('J')
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
        decomposed = metric.NamedUnit(unit).decomposed
        result = [
            (float(term.coefficient), term.base, int(term.exponent))
            for term in decomposed
        ]
        assert result == expected


def test_named_unit_parse():
    """Test the ability to handle arbitrary named units.

    Note that the class that manages single units is primarily an assistant to
    the Unit class, so full coverage is not necessary as long as Unit is
    well tested.
    """
    order, unit = metric.NamedUnit.parse('m')
    assert order.symbol == ''
    assert order.name == ''
    assert order.factor == 1.0
    assert unit.symbol == 'm'
    assert unit.name == 'meter'
    assert unit.quantity == 'length'
    order, unit = metric.NamedUnit.parse('mm')
    assert order.symbol == 'm'
    assert order.name == 'milli'
    assert order.factor == 1e-3
    assert unit.symbol == 'm'
    assert unit.name == 'meter'
    assert unit.quantity == 'length'
    symbolic = metric.NamedUnit.parse('mm')
    named = metric.NamedUnit.parse('millimeter')
    assert symbolic == named
    order, unit = metric.NamedUnit.parse('lm')
    assert order.symbol == ''
    assert order.name == ''
    assert order.factor == 1.0
    assert unit.symbol == 'lm'
    assert unit.name == 'lumen'
    assert unit.quantity == 'luminous flux'
    order, unit = metric.NamedUnit.parse('MeV')
    assert order.symbol == 'M'
    assert order.name == 'mega'
    assert order.factor == 1e6
    assert unit.symbol == 'eV'
    assert unit.name == 'electronvolt'
    assert unit.quantity == 'energy'
    order, unit = metric.NamedUnit.parse('μeV')
    assert order.symbol == 'μ'
    assert order.name == 'micro'
    assert order.factor == 1e-6
    assert unit.symbol == 'eV'
    assert unit.name == 'electronvolt'
    assert unit.quantity == 'energy'
    order, unit = metric.NamedUnit.parse('uerg')
    assert order.symbol == 'μ'
    assert order.name == 'micro'
    assert order.factor == 1e-6
    assert unit.symbol == 'erg'
    assert unit.name == 'erg'
    assert unit.quantity == 'energy'
    order, unit = metric.NamedUnit.parse('statA')
    assert order.symbol == ''
    assert order.name == ''
    assert order.factor == 1.0
    assert unit.symbol == 'statA'
    assert unit.name == 'statampere'
    assert unit.quantity == 'current'


def test_named_unit_idempotence():
    """Make sure we can create a new NamedUnit from an existing instance."""
    old = metric.NamedUnit('m')
    new = metric.NamedUnit(old)
    assert isinstance(new, metric.NamedUnit)
    assert new is old


def test_dimension_init():
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
        unit = metric.Unit(current['unit'])
        quantity = metric.Quantity(current['quantity'])
        forms = current['forms']
        for target in (unit, quantity['mks']):
            dimension = metric.Dimension(target)
            assert isinstance(dimension, metric.Dimension)
            for form in forms:
                assert dimension == form


def test_unit_init():
    """Initialize the Unit object with various strings."""
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
    for arg, expected in strings.items():
        unit = metric.Unit(arg)
        assert unit == expected['unit']


def test_unit_idempotence():
    """Make sure initializing with a Unit creates a new Unit."""
    old = metric.Unit('m')
    new = metric.Unit(old)
    assert isinstance(new, metric.Unit)
    assert str(new) == str(old)
    assert repr(new) == repr(old)


def test_unit_algebra():
    """Test algebraic operations on the Unit class."""
    u0 = metric.Unit('m')
    u1 = metric.Unit('J')
    assert u0**2 is not u0
    assert u0 * u1 == metric.Unit('m * J')
    assert u0 / u1 == metric.Unit('m / J')
    assert u0**2 / u1**3 == metric.Unit('m^2 / J^3')
    assert (u0 / u1)**2 == metric.Unit('m^2 / J^2')


def test_unit_multiply():
    """Test the ability to create a new compound unit with '*'."""
    cases = {
        ('m', 's'): 'm*s',
        ('m/s', 'km/m'): 'km/s',
        ('m', 'm^-1'): '1',
    }
    apply_multiplicative(operator.mul, cases)


def test_unit_divide():
    """Test the ability to create a new compound unit with '/'."""
    cases = {
        ('m', 's'): 'm/s',
        ('m/s', 'm/km'): 'km/s',
        ('m', 'm'): '1',
    }
    apply_multiplicative(operator.truediv, cases)


def apply_multiplicative(opr, cases: dict):
    """Apply a multiplicative operator between units."""
    for (this, that), expected in cases.items():
        result = opr(metric.Unit(this), that)
        assert isinstance(result, metric.Unit)
        assert result == metric.Unit(expected)


def test_unit_floordiv():
    """Test conversion with the Unit class."""
    unit = metric.Unit('m')
    assert metric.Unit('cm') // unit == 1e2
    assert unit // 'cm' == 1e-2
    assert 'cm' // unit == 1e2
    unit = metric.Unit('m / s')
    assert unit // 'km / h' == pytest.approx(1e3 / 3600)
    assert 'km / h' // unit == pytest.approx(3600 / 1e3)


def test_unit_raise():
    """Test the ability to create a new compound unit with '**'."""
    cases = {
        ('m', 2): 'm^2',
        ('m/s', 3): 'm^3 s^-3',
        ('J*s^2/m^3', -1): 'J^-1 s^-2 m^3',
    }
    for (this, that), expected in cases.items():
        result = metric.Unit(this) ** that
        assert isinstance(result, metric.Unit)
        assert result == metric.Unit(expected)


def test_unit_equality():
    """Test the definition of strict equality between units."""
    assert metric.Unit('m/s') == metric.Unit('m/s')
    assert metric.Unit('m/s') == metric.Unit('m*s^-1')


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
        assert metric.System(lower) == metric.System(upper)
        system = metric.System(lower)
        for key, definition in cases.items():
            assert system[key] == metric.Metric(**definition)


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
        system = metric.System(name)
        for (key, test, expected) in cases:
            search = {key: test}
            assert system.get_unit(**search) == expected


def test_system_singleton():
    """Metric systems should be singletons of their lower-case name."""
    for system in ('mks', 'cgs'):
        old = metric.System(system)
        new = metric.System(old)
        assert new is old
