import operator

import pytest

from goats.core import symbolic
from goats.core import metric


def test_defined_conversions():
    """Test the collection of defined conversions."""
    assert len(metric.CONVERSIONS) == 2 * len(metric._CONVERSIONS)
    for (u0, u1), wt in metric._CONVERSIONS.items():
        assert (u0, u1) in metric.CONVERSIONS
        assert metric.CONVERSIONS.get_weight(u0, u1) == wt
        assert metric.CONVERSIONS.get_weight(u1, u0) == 1 / wt


@pytest.fixture
def conversions():
    """Test cases for unit conversions."""
    return {
        # Length (common and simple)
        ('m', 'm'): 1.0, # trivial conversion
        ('m', 'cm'): 1e2, # defined metric-system conversion
        ('m', 'km'): 1e-3, # base-unit rescale
        ('m', 'mm'): 1e3, # (same)
        ('mm', 'm'): 1e-3, # (same)
        ('m', 'au'): 1 / 1.495978707e11, # to non-system unit
        ('au', 'm'): 1.495978707e11, # from non-system unit
        ('au', 'au'): 1.0, # trivial non-system conversion
        # Momentum (requires symbolic expressions)
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
        # Decompositions of equivalent units
        ('N', 'kg m s^-2'): 1.0,
        ('J', 'kg m^2 s^-2'): 1.0,
        ('V / m', 'T m s^-1'): 1.0,
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


def test_conversion_class(conversions: dict):
    """Test the unit-conversion class"""
    for (u0, u1), factor in conversions.items():
        conversion = metric.Conversion(u0, u1)
        assert float(conversion) == pytest.approx(factor)
    with pytest.raises(metric.UnitConversionError):
        metric.Conversion('m', 'J')


def test_conversion_function(conversions: dict):
    """Test the function that wraps the unit-conversion class."""
    for (u0, u1), factor in conversions.items():
        conversion = metric.conversion(u0, u1)
        assert conversion == pytest.approx(factor)
    assert metric.conversion('m', 'J') is None
    with pytest.raises(metric.UnitConversionError):
        metric.conversion('m', 'J', strict=True)


def test_singletons():
    """Make sure certain objects have a single reference."""
    cases = {
        metric._Property: ['units'],
        metric.NamedUnit: ['m', 'meter'],
        metric.Conversion: [('G', 'T')],
        metric._Converter: [('m', 'length')],
        metric.Quantity: ['energy', 'Energy'],
        metric.Unit: [
            'kg m^2 / s^2',
            'kg m^2 s^-2',
            'kg s^-2 m^2',
            'm^2 s^-2 kg',
            'm^2 kg s^-2',
            's^-2 kg m^2',
            's^-2 m^2 kg',
        ],
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


def test_ratio():
    """Test the method that computes the ratio of two units."""
    cases = {
        ('cm', 'm'): 1e-2,
        ('m', 'cm'): 1e2,
        ('cm', 'cm'): 1.0,
        ('km', 'm'): 1e3,
        ('m', 'km'): 1e-3,
    }
    for (u0, u1), result in cases.items():
        assert metric.ratio(u0, u1) == result
    with pytest.raises(ValueError):
        metric.ratio('cm', 'J')


def test_build_named_unit():
    cases = {
        'm': {
            'name': 'meter',
            'symbol': 'm',
            'scale': 1.0,
            'quantity': 'length',
        },
        'cm': {
            'name': 'centimeter',
            'symbol': 'cm',
            'scale': 1e-2,
            'quantity': 'length',
        },
        'J': {
            'name': 'joule',
            'symbol': 'J',
            'scale': 1.0,
            'quantity': 'energy',
        },
    }
    for name, attrs in cases.items():
        unit = metric.NamedUnit(name)
        for key, value in attrs.items():
            assert getattr(unit, key) == value
    with pytest.raises(metric.UnitParsingError):
        metric.NamedUnit('cat')


def test_named_unit_dimensions():
    """Test the dimensions attribute of a NamedUnit."""
    cases = {
        'm': {'mks': 'L', 'cgs': 'L'},
        'cm': {'mks': 'L', 'cgs': 'L'},
        'J': {'mks': '(M * L^2) / T^2', 'cgs': None},
        'erg': {'cgs': '(M * L^2) / T^2', 'mks': None},
        'ohm': {'mks': 'M L^2 T^-3 I^-1', 'cgs': None},
        'au': {'mks': 'L', 'cgs': 'L'},
        'MeV': {'mks': '(M * L^2) / T^2', 'cgs': '(M * L^2) / T^2'},
    }
    for unit, dimensions in cases.items():
        named = metric.NamedUnit(unit)
        assert named.dimensions == dimensions
        named.dimensions.pop('mks')
        named.dimensions.pop('cgs')
        named.dimensions['foo'] = 'bar'
        assert named.dimensions == dimensions


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


@pytest.fixture
def decompositions():
    """Test cases for named-unit decompositions."""
    return {
        # fundamental in mks and cgs
        's': [{'base': 's'}],
        # fundamental in mks
        'm': [{'base': 'm'}],
        'kg': [{'base': 'kg'}],
        'N': [
            {'base': 'kg', 'exponent': 1},
            {'base': 'm', 'exponent': 1},
            {'base': 's', 'exponent': -2},
        ],
        'J': [
            {'base': 'kg', 'exponent': 1},
            {'base': 'm', 'exponent': 2},
            {'base': 's', 'exponent': -2},
        ],
        'ohm': [
            {'base': 'kg', 'exponent': 1},
            {'base': 'm', 'exponent': 2},
            {'base': 's', 'exponent': -3},
            {'base': 'A', 'exponent': -1},
        ],
        # fundamental in cgs
        'cm': [{'base': 'cm'}],
        'g': [{'base': 'g'}],
        'dyn': [
            {'base': 'g', 'exponent': 1},
            {'base': 'cm', 'exponent': 1},
            {'base': 's', 'exponent': -2},
        ],
        'erg': [
            {'base': 'g', 'exponent': 1},
            {'base': 'cm', 'exponent': 2},
            {'base': 's', 'exponent': -2},
        ],
        # not fundamental in any metric system
        'mJ': None,
        'merg': None,
        'km': None,
        'au': None,
    }


def test_named_unit_decompose(decompositions: dict):
    """Test the NamedUnit.decompose method."""
    for unit, expected in decompositions.items():
        named = metric.NamedUnit(unit)
        result = named.decomposed
        if expected is None:
            assert result is None
        else:
            terms = [symbolic.Term(**term) for term in expected]
            assert set(result) == set(terms)


@pytest.fixture
def reductions():
    """Test cases for named-unit reductions."""
    return {
        's': {
            'mks': {
                'scale': 1e0,
                'terms': [{'base': 's'}],
            },
            'cgs': {
                'scale': 1e0,
                'terms': [{'base': 's'}],
            },
        },
        'm': {
            'mks': {
                'scale': 1e0,
                'terms': [{'base': 'm'}],
            },
            'cgs': {
                'scale': 1e2,
                'terms': [{'base': 'cm'}],
            },
        },
        'cm': {
            'mks': {
                'scale': 1e-2,
                'terms': [{'base': 'm'}],
            },
            'cgs': {
                'scale': 1e0,
                'terms': [{'base': 'cm'}],
            },
        },
        'km': {
            'mks': {
                'scale': 1e3,
                'terms': [{'base': 'm'}],
            },
            'cgs': {
                'scale': 1e5,
                'terms': [{'base': 'cm'}],
            },
        },
        'kg': {
            'mks': {
                'scale': 1e0,
                'terms': [{'base': 'kg'}],
            },
            'cgs': {
                'scale': 1e3,
                'terms': [{'base': 'g'}],
            },
        },
        'g': {
            'mks': {
                'scale': 1e-3,
                'terms': [{'base': 'kg'}],
            },
            'cgs': {
                'scale': 1e0,
                'terms': [{'base': 'g'}],
            },
        },
        'J': {
            'mks': {
                'scale': 1e0,
                'terms': [
                    {'base': 'kg', 'exponent': 1},
                    {'base': 'm', 'exponent': 2},
                    {'base': 's', 'exponent': -2},
                ],
            },
            'cgs': None,
        },
        'mJ': {
            'mks': {
                'scale': 1e-3,
                'terms': [
                    {'base': 'kg', 'exponent': 1},
                    {'base': 'm', 'exponent': 2},
                    {'base': 's', 'exponent': -2},
                ],
            },
            'cgs': None,
        },
        'erg': {
            'mks': None,
            'cgs': {
                'scale': 1e0,
                'terms': [
                    {'base': 'g', 'exponent': 1},
                    {'base': 'cm', 'exponent': 2},
                    {'base': 's', 'exponent': -2},
                ],
            },
        },
        'merg': {
            'mks': None,
            'cgs': {
                'scale': 1e-3,
                'terms': [
                    {'base': 'g', 'exponent': 1},
                    {'base': 'cm', 'exponent': 2},
                    {'base': 's', 'exponent': -2},
                ],
            },
        },
        'N': {
            'mks': {
                'scale': 1e0,
                'terms': [
                    {'base': 'kg', 'exponent': 1},
                    {'base': 'm', 'exponent': 1},
                    {'base': 's', 'exponent': -2},
                ],
            },
            'cgs': None,
        },
        'au': {
            'mks': None,
            'cgs': None,
        },
    }


def test_reduction_class():
    """Test the symbolic expression of a unit reduction."""
    this = metric.Reduction(['m^2'], scale=2, system='mks')
    assert this.units == ['m^2']
    assert this.scale == 2.0
    assert this.system == 'mks'
    new = 3 * this
    assert new.units == ['m^2']
    assert new.scale == 6.0
    assert new.system == 'mks'
    new = this * 2
    assert new.units == ['m^2']
    assert new.scale == 4.0
    assert new.system == 'mks'
    new = this / 4
    assert new.units == ['m^2']
    assert new.scale == 0.5
    assert new.system == 'mks'
    with pytest.raises(TypeError):
        5 / this
    new = this ** 2
    assert new.units == ['m^4']
    assert new.scale == 4.0
    assert new.system == 'mks'
    with pytest.raises(TypeError):
        3 ** this


def test_named_unit_norm():
    """A named unit should know its canonical equivalents."""
    cases = {
        'm': {
            'mks': 'm',
            'cgs': 'cm',
        },
        'cm': {
            'mks': 'm',
            'cgs': 'cm',
        },
        'J': {
            'mks': 'J',
            'cgs': 'erg',
        },
        'mJ': {
            'mks': 'J',
            'cgs': 'erg',
        },
        'erg': {
            'mks': 'J',
            'cgs': 'erg',
        },
        'kerg': {
            'mks': 'J',
            'cgs': 'erg',
        },
        'eV': {
            'mks': 'J',
            'cgs': 'erg',
        },
        'MeV': {
            'mks': 'J',
            'cgs': 'erg',
        },
    }
    for unit, systems in cases.items():
        named = metric.NamedUnit(unit)
        for system, expected in systems.items():
            assert named.norm[system] == expected


def test_named_unit_reduce(reductions: dict):
    """Test the NamedUnit.reduce method."""
    for unit, systems in reductions.items():
        named = metric.NamedUnit(unit)
        for system, expected in systems.items():
            result = named.reduce(system)
            if expected is None:
                assert result is None
            else:
                assert result.system == system
                assert result.scale == expected['scale']
                terms = [symbolic.Term(**term) for term in expected['terms']]
                assert set(result.units) == set(terms)


def test_named_unit_reduce_system(reductions: dict):
    """Test reductions with the default metric system."""
    these = {
        'J': 'mks', # only defined in mks
        'erg': 'cgs', # only defined in cgs
        'cm': 'cgs', # fundamental in cgs
        'kg': 'mks', # fundamental in mks
        's': 'mks', # fundamental in both
        'au': 'mks', # fundamental in neither
    }
    for unit, default in these.items():
        case = reductions[unit][default]
        result = metric.NamedUnit(unit).reduce()
        if case is None:
            assert result is None
        else:
            assert result.system == default
            terms = [symbolic.Term(**term) for term in case['terms']]
            assert result.scale == case['scale']
            assert set(result.units) == set(terms)


def test_identify_unit():
    """Test the ability to identify arbitrary named units.

    Note that the primary role of this function is in support of the Unit class,
    so full coverage is not necessary as long as Unit is well tested.
    """
    order, unit = metric.identify('m')
    assert order.symbol == ''
    assert order.name == ''
    assert order.factor == 1.0
    assert unit.symbol == 'm'
    assert unit.name == 'meter'
    assert unit.quantity == 'length'
    order, unit = metric.identify('mm')
    assert order.symbol == 'm'
    assert order.name == 'milli'
    assert order.factor == 1e-3
    assert unit.symbol == 'm'
    assert unit.name == 'meter'
    assert unit.quantity == 'length'
    symbolic = metric.identify('mm')
    named = metric.identify('millimeter')
    assert symbolic == named
    order, unit = metric.identify('lm')
    assert order.symbol == ''
    assert order.name == ''
    assert order.factor == 1.0
    assert unit.symbol == 'lm'
    assert unit.name == 'lumen'
    assert unit.quantity == 'luminous flux'
    order, unit = metric.identify('MeV')
    assert order.symbol == 'M'
    assert order.name == 'mega'
    assert order.factor == 1e6
    assert unit.symbol == 'eV'
    assert unit.name == 'electronvolt'
    assert unit.quantity == 'energy'
    order, unit = metric.identify('μeV')
    assert order.symbol == 'μ'
    assert order.name == 'micro'
    assert order.factor == 1e-6
    assert unit.symbol == 'eV'
    assert unit.name == 'electronvolt'
    assert unit.quantity == 'energy'
    order, unit = metric.identify('uerg')
    assert order.symbol == 'μ'
    assert order.name == 'micro'
    assert order.factor == 1e-6
    assert unit.symbol == 'erg'
    assert unit.name == 'erg'
    assert unit.quantity == 'energy'
    order, unit = metric.identify('statA')
    assert order.symbol == ''
    assert order.name == ''
    assert order.factor == 1.0
    assert unit.symbol == 'statA'
    assert unit.name == 'statampere'
    assert unit.quantity == 'current'


def test_named_unit_systems():
    """Determine which metric systems include a named unit."""
    test = {
        'm': {
            'allowed': {'mks', 'cgs'},
            'defined': {'mks', 'cgs'},
            'fundamental': {'mks'},
        },
        'cm': {
            'allowed': {'mks', 'cgs'},
            'defined': {'mks', 'cgs'},
            'fundamental': {'cgs'},
        },
        'J': {
            'allowed': {'mks'},
            'defined': {'mks'},
            'fundamental': {'mks'},
        },
        'erg': {
            'allowed': {'cgs'},
            'defined': {'cgs'},
            'fundamental': {'cgs'},
        },
        'au': {
            'allowed': {'mks', 'cgs'},
            'defined': set(),
            'fundamental': set(),
        },
        's': {
            'allowed': {'mks', 'cgs'},
            'defined': {'mks', 'cgs'},
            'fundamental': {'mks', 'cgs'},
        },
    }
    for unit, cases in test.items():
        for mode, expected in cases.items():
            named = metric.NamedUnit(unit)
            assert set(named.systems[mode]) == expected
            named.systems.pop('allowed')
            named.systems['foo'] = 'bar'
            assert set(named.systems[mode]) == expected


def test_named_unit_idempotence():
    """Make sure we can create a new NamedUnit from an existing instance."""
    old = metric.NamedUnit('m')
    new = metric.NamedUnit(old)
    assert isinstance(new, metric.NamedUnit)
    assert new is old


def test_create_unit():
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


def test_unit_norm():
    """User must be able to normalize a given unit to a metric system."""
    cases = {
        # high-priority examples from EPREM
        'day': {
            'mks': 's',
            'cgs': 's',
        },
        'au': {
            'mks': 'm',
            'cgs': 'cm',
        },
        'km / s': {
            'mks': 'm / s',
            'cgs': 'cm / s',
        },
        'MeV': {
            'mks': 'J',
            'cgs': 'erg',
        },
        'nT': {
            'mks': 'T',
            'cgs': 'G',
        },
        'cm^-3': {
            'mks': 'm^-3',
            'cgs': 'cm^-3',
        },
        's^3 / km^6': {
            'mks': 's^3 / m^6',
            'cgs': 's^3 / cm^6',
        },
    }
    for name, systems in cases.items():
        unit = metric.Unit(name)
        for system, expected in systems.items():
            assert unit.norm[system] == expected


def test_unit_dimensions():
    """A Unit should know its dimension in all applicable metric systems."""
    test = {
        'm': { # 'm' is defined in both systems
            'mks': 'L',
            'cgs': 'L',
        },
        'm / s': { # 'm' and 's' are defined in both systems
            'mks': 'L T^-1',
            'cgs': 'L T^-1',
        },
        'J': { # 'J' is defined only in mks
            'mks': 'M L^2 T^-2',
            'cgs': None,
        },
        'erg': { # 'erg' is defined only in cgs
            'mks': None,
            'cgs': 'M L^2 T^-2',
        },
        'au': { # 'au' is system-independent
            'mks': 'L',
            'cgs': 'L',
        },
        '# / (cm^2 s sr J)': { # mix of both ('cm') and mks ('J')
            'mks': 'T M^-1 L^-4',
            'cgs': 'T M^-1 L^-4',
        },
        '# / (m^2 s sr erg)': { # mix of both ('m') and cgs ('erg')
            'mks': 'T M^-1 L^-4',
            'cgs': 'T M^-1 L^-4',
        },
        '# / (cm^2 s sr MeV)': { # mix of cgs ('cm') and none ('MeV')
            'mks': 'T M^-1 L^-4',
            'cgs': 'T M^-1 L^-4',
        },
        'au / m': { # dimensionless mix of none ('au') and both ('m')
            'mks': '1',
            'cgs': '1',
        },
        'au / cm': { # dimensionless mix of none ('au') and both ('cm')
            'mks': '1',
            'cgs': '1',
        },
        'J / erg': { # dimensionless mix of mks ('J') and cgs ('erg')
            'mks': '1',
            'cgs': '1',
        },
        'J / eV': { # dimensionless mix of mks ('J') and none ('eV')
            'mks': '1',
            'cgs': '1',
        },
        'erg / eV': { # dimensionless mix of cgs ('erg') and none ('eV')
            'mks': '1',
            'cgs': '1',
        },
    }
    for string, cases in test.items():
        unit = metric.Unit(string)
        for system, expected in cases.items():
            assert unit.dimensions[system] == expected
    # User should not be able to alter dimensions on an instance.
    meter = metric.Unit('m')
    with pytest.raises(AttributeError):
        meter.dimensions.pop('mks')
    with pytest.raises(TypeError):
        meter.dimensions['mks'] = 'Oops!'
    assert meter.dimensions['mks'] == 'L'


def test_decompose_unit():
    """Test the ability to decompose a unit expression."""
    cases = {
        'm': 'm',
        'cm': 'cm',
        'km': 'km',
        'au': 'au',
        'm / cm': 'm / cm',
        'm / au': 'm / au',
        'N * s': 'kg m s^-1',
        'N / s': 'kg m s^-3',
        'N / m': 'kg s^-2',
        'N * s^2': 'kg m',
        'N / cm': 'kg m s^-2 cm^-1',
        'J / erg': 'kg m^2 g^-1 cm^-2',
        'J * (kg / s) / N^2': 's',
    }
    for unit, expected in cases.items():
        decomposed = metric.Unit.decompose(unit)
        expression = symbolic.Expression(expected)
        assert decomposed == expression
        assert metric.Unit(unit).decomposed == decomposed


def test_unit_algebra():
    """Test symbolic operations on the Unit class."""
    u0 = metric.Unit('m')
    u1 = metric.Unit('J')
    assert u0**2 is not u0
    assert u0 * u1 == metric.Unit('m * (kg * m^2 / s^2)')
    assert u0 / u1 == metric.Unit('m / (kg * m^2 / s^2)')
    assert u0**2 / u1**3 == metric.Unit('m^2 / (kg * m^2 / s^2)^3')
    assert (u0 / u1)**2 == metric.Unit('m^2 / (kg * m^2 / s^2)^2')


def test_unit_multiply():
    """Test the ability to create a new compound unit with '*'."""
    cases = {
        ('m', 's'): 'm * s',
        ('m / s', 'km / m'): 'km / s',
        ('m', 'm'): 'm^2',
        ('m', 'm^-1'): '1',
        ('N', '(kg * m / s^2)^-1'): '1',
        ('N', 'kg * m / s^2'): 'kg^2 * m^2 / s^4',
        ('J * cm', 'erg * m'): 'kg m^3 g cm^3 s^-4',
        ('J', 'm'): 'kg m^3 s^-2',
        ('N / erg', 'J / dyn'): 'kg^2 m^3 g^-2 cm^-3',
        ('N / cm^3', 'erg * s / kg'): 'm * g / (s^3 * cm)',
    }
    apply_multiplicative(operator.mul, cases)


def test_unit_divide():
    """Test the ability to create a new compound unit with '/'."""
    cases = {
        ('m', 's'): 'm / s',
        ('m / s', 'm / km'): 'km / s',
        ('m', 'm'): '1',
        ('m', 'm^-1'): 'm^2',
        ('N', '(kg * m / s^2)^-1'): 'kg^2 * m^2 / s^4',
        ('N', 'kg * m / s^2'): '1',
        ('J * cm', 'erg * m'): 'kg m g^-1 cm^-1',
        ('J', 'm'): 'kg m s^-2',
        ('N / erg', 'J / dyn'): 'cm^-1 m^-1',
        ('N / cm^3', 'erg * s / kg'): 'kg^2 * m / (g * cm^5 * s)',
    }
    apply_multiplicative(operator.truediv, cases)


def apply_multiplicative(opr, cases: dict):
    """Apply a multiplicative operator between units."""
    for (this, that), expected in cases.items():
        for other in (that, metric.Unit(that)):
            result = opr(metric.Unit(this), other)
            assert isinstance(result, metric.Unit)
            assert result == expected
            assert result is metric.Unit(expected)


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


def test_unit_consistency():
    """Test the ability to check if two units are mutually consistent."""
    cases = {
        ('m', 'm'): True,
        ('m', 'cm'): True,
        ('m', 'km'): True,
        ('m', 'au'): True,
        ('cm', 'km'): True,
        ('cm', 'au'): True,
        ('J', 'erg'): True,
        ('J', 'eV'): True,
        ('erg', 'eV'): True,
        ('J', 'N * m'): True,
        ('J', 'N'): False,
        ('N', 'm'): False,
        ('V', 'statV'): True,
        ('C', 'statC'): True,
        ('T', 'G'): True,
        ('nuc', 'kg'): True,
    }
    for (u0, u1), truth in cases.items():
        assert (metric.Unit(u0) | metric.Unit(u1)) == truth
        assert (metric.Unit(u0) | u1) == truth
        assert (u0 | metric.Unit(u1)) == truth


def test_unit_equality():
    """Test the definition of strict equality between units."""
    cases = {
        # identity implies equality
        ('m / s', 'm / s'): True,
        # units with equivalent string expressions are equal
        ('m / s', 'm s^-1'): True,
        # units that differ only by dimensionless terms are equal
        ('# / (m^2 s sr J)', 'm^-2 s^-1 sr^-1 J^-1'): True,
        ('# / (m^2 s sr J)', '1 / (m^2 s sr J)'): True,
        # physically equivalent units are equal
        ('N', 'kg m s^-2'): True,
        ('dyn', 'g cm s^-2'): True,
        ('m^3 / (s^2 * kg)', 'm^2 N kg^-2'): True,
        # a ratio of equal units is unitless
        ('J / J', '1'): True,
        # units of the same quantity are not necessarily equal
        ('N', 'dyn'): False,
        ('N', 'g cm s^-2'): False,
        # units that are not interchangeable are not equal
        ('N', 'kg m^2 s^-2'): False,
        # A dimensioned unit is not equal to the dimensionless unit
        ('m', '1'): False,
        ('day', '1'): False,
        # miscellaneous pathological cases
        ('au', 'cd * sr'): False,
    }
    for (u0, u1), truth in cases.items():
        assert (metric.Unit(u0) == metric.Unit(u1)) == truth
        assert (u0 == metric.Unit(u1)) == truth
        assert (metric.Unit(u0) == u1) == truth


def test_unit_identity():
    """Instances that represent the same unit should be identical"""
    cases = [
        ('m', 'meter'),
        ('J', 'joule'),
        ('eV', 'electronvolt'),
        ('G', 'gauss'),
        ('T', 'tesla'),
        ('s', 'second'),
        ('rad', 'radian'),
        ('deg', 'degree'),
        ('Hz', 'hertz'),
    ]
    for (u0, u1) in cases:
        assert metric.Unit(u0) is metric.Unit(u1)


def test_unit_dimensionless():
    """Test the boolean `dimensionless` property."""
    cases = {
        '1': True,
        '#': True,
        'm': False,
        'cm': False,
        'J': False,
        'erg': False,
        'N / N': True,
    }
    for unit, truth in cases.items():
        assert metric.Unit(unit).dimensionless == truth


def test_dimension_fromunit():
    """Test the ability to instantiate a Dimension from a Unit."""
    cases = [
        {
            'unit': 'm',
            'systems': {'mks', 'cgs'},
            'quantity': 'length',
            'forms': ['L'],
        },
        {
            'unit': 'm / s',
            'systems': {'mks', 'cgs'},
            'quantity': 'velocity',
            'forms': ['L T^-1', 'T^-1 L'],
        },
        {
            'unit': 'km / s',
            'systems': {'mks', 'cgs'},
            'quantity': 'velocity',
            'forms': ['L T^-1', 'T^-1 L'],
        },
        {
            'unit': 'J',
            'systems': {'mks'},
            'quantity': 'energy',
            'forms': ['M L^2 T^-2', 'M T^-2 L^2', 'L^2 M T^-2'],
        },
        {
            'unit': 'erg',
            'systems': {'cgs'},
            'quantity': 'energy',
            'forms': ['M L^2 T^-2', 'M T^-2 L^2', 'L^2 M T^-2'],
        },
    ]
    for current in cases:
        unit = metric.Unit(current['unit'])
        forms = current['forms']
        for system in current['systems']:
            dimension = metric.Dimension.fromunit(unit, system)
            assert isinstance(dimension, metric.Dimension)
            for form in forms:
                assert dimension == form


def test_dimension_quantities():
    """Look up quantities in a given metric system for a dimension."""
    cases = {
        'L': {
            'mks': {'length'},
            'cgs': {'capacitance', 'length'},
        },
        'L / T': {
            'mks': {'velocity', 'speed'},
            'cgs': {'conductance', 'velocity', 'speed'},
        },
        'M * L / T^2': {
            'mks': {'force'},
            'cgs': {'force'},
        },
        'M * L^2 / T^2': {
            'mks': {'energy', 'work'},
            'cgs': {'energy', 'work'},
        },
    }
    for string, systems in cases.items():
        dimension = metric.Dimension(string)
        for system, quantities in systems.items():
            assert dimension.quantities(system) == quantities


def test_system():
    """Test the object that represents a system of quantities."""
    # Cases:
    # - length: same dimension; same unit.
    # - momentum: same dimension; different unit.
    # - magnetic induction: different dimension; different unit.
    cases = {
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
    for string, quantities in cases.items():
        lower = string.lower()
        upper = string.upper()
        assert metric.System(lower) == metric.System(upper)
        system = metric.System(lower)
        for name, canonical in quantities.items():
            quantity = system[name]
            assert quantity.unit == canonical['unit']
            assert quantity.dimension == canonical['dimension']


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


@pytest.mark.xfail
def test_decomposition():
    """Test the module-level unit-decomposing function."""
    cases = {
        'm': {
            'mks': (1.0, 'm'),
            'cgs': (1e2, 'cm'),
        },
        'cm': {
            'mks': (1e-2, 'm'),
            'cgs': (1.0, 'cm'),
        },
        'J': {
            'mks': (1.0, 'kg m^2 s^-2'),
            'cgs': (1e1, 'g cm^2 s^-2'),
        },
        'erg': {
            'mks': (1e-1, 'kg m^2 s^-2'),
            'cgs': (1.0, 'g cm^2 s^-2'),
        },
        'N / cm^2': { # kg m s^-1 cm^-2
            'mks': (1e-4, 'kg s^-1 m^-1'), # kg m s^-1 (1e-2m)^-2
            'cgs': (1e-1, 'g s^-1 cm^-1'), # 1e-3g (1e2cm) s^-1 cm^-2
        },
        'erg * s / kg': {
            'mks': (),
            'cgs': (),
        },
    }
    for unit, systems in cases.items():
        for name, expected in systems.items():
            result = metric.reduction(unit, system=name)
            if expected is None:
                assert result is None
            else:
                (scale, expression) = expected
                assert symbolic.Expression(result.units) == expression
                assert result.scale == scale
                assert result.system == name


def test_derived_quantity():
    """Make sure derived quantities have the correct attributes.
    
    A derived quantity (at least for the purposes of this test) is a
    symbolically defined metric quantity. For example, 'speed = length / time'
    is a derived quantity. Note that the definition of a derived quantity may
    contain derived quantities.
    """
    # TODO: Extend to all derived quantities.
    cases = {
        'area': {
            'mks': {
                'dimension': 'L^2',
                'unit': 'm^2',
            },
            'cgs': {
                'dimension': 'L^2',
                'unit': 'cm^2',
            },
        },
        'electric field': {
            'mks': {
                'dimension': '(M * L) / (T^3 * I)',
                'unit': 'V / m',
            },
            'cgs': {
                'dimension': 'M^1/2 / (T * L^1/2)',
                'unit': 'statV / cm',
            },
        },
        'fluence': {
            'mks': {
                'dimension': 'T^2 M^-1 L^-4',
                'unit': 'J^-1 sr^-1 m^-2',
            },
            'cgs': {
                'dimension': 'T^2 M^-1 L^-4',
                'unit': 'erg^-1 sr^-1 cm^-2',
            },
        },
        'flux': {
            'mks': {
                'dimension': 'T M^-1 L^-4',
                'unit': 'J^-1 s^-1 sr^-1 m^-2',
            },
            'cgs': {
                'dimension': 'T M^-1 L^-4',
                'unit': 'erg^-1 s^-1 sr^-1 cm^-2',
            },
        },
        'integral flux': {
            'mks': {
                'dimension': 'T^-1 L^-2',
                'unit': 's^-1 sr^-1 m^-2',
            },
            'cgs': {
                'dimension': 'T^-1 L^-2',
                'unit': 's^-1 sr^-1 cm^-2',
            },
        },
        'number density': {
            'mks': {
                'dimension': '1 / L^3',
                'unit': 'm^-3',
            },
            'cgs': {
                'dimension': '1 / L^3',
                'unit': 'cm^-3',
            },
        },
        'speed': {
            'mks': {
                'dimension': 'L / T',
                'unit': 'm / s',
            },
            'cgs': {
                'dimension': 'L / T',
                'unit': 'cm / s',
            },
        },
    }
    for name, systems in cases.items():
        quantity = metric.Quantity(name)
        for system, expected in systems.items():
            assert quantity[system].dimension == expected['dimension']
            assert quantity[system].unit == expected['unit']


def test_create_quantity():
    """Test the ability to represent arbitrary metric quantities.
    
    This test exists to test new derived quantities that have been built from
    those defined in the module.
    """
    q = metric.Quantity('length / magnetic field')
    assert q['mks'].unit == 'm T^-1'


def test_quantity_equality():
    """Test the definition of equality for a quantity."""
    names = [
        'length',
        'energy',
        'length / velocity',
    ]
    for name in names:
        q0 = metric.Quantity(name)
        # It should be equal to its string name.
        assert q0 == name
        # Idempotence -> identity => equality.
        assert q0 == metric.Quantity(name)


def test_quantity_equivalence():
    """Test the definition of equivalence between quantities."""
    pairs = [
        ('energy', 'work'),
        ('force * area / mass^2', 'volume / (time^2 * mass)'),
        ('frequency', 'vorticity'),
    ]
    for (s0, s1) in pairs:
        q0 = metric.Quantity(s0)
        q1 = metric.Quantity(s1)
        assert q0 != q1
        assert q0 | q1


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


