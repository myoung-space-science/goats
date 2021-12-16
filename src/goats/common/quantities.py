import abc
import collections.abc
import inspect
import functools
import math
import numbers
from typing import *

import numpy as np

from goats.common import *
from goats.common import algebra, arrays, iterables


_prefixes = [
    {'symbol': 'Y', 'name': 'yotta', 'factor': 1e+24},
    {'symbol': 'Z', 'name': 'zetta', 'factor': 1e+21},
    {'symbol': 'E', 'name': 'exa', 'factor': 1e+18},
    {'symbol': 'P', 'name': 'peta', 'factor': 1e+15},
    {'symbol': 'T', 'name': 'tera', 'factor': 1e+12},
    {'symbol': 'G', 'name': 'giga', 'factor': 1e+9},
    {'symbol': 'M', 'name': 'mega', 'factor': 1e+6},
    {'symbol': 'k', 'name': 'kilo', 'factor': 1e+3},
    {'symbol': 'h', 'name': 'hecto', 'factor': 1e+2},
    {'symbol': 'da', 'name': 'deca', 'factor': 1e+1},
    {'symbol': '', 'name': '', 'factor': 1e0},
    {'symbol': ' ', 'name': None, 'factor': 0.0},
    {'symbol': 'd', 'name': 'deci', 'factor': 1e-1},
    {'symbol': 'c', 'name': 'centi', 'factor': 1e-2},
    {'symbol': 'm', 'name': 'milli', 'factor': 1e-3},
    {'symbol': 'μ', 'name': 'micro', 'factor': 1e-6},
    {'symbol': 'n', 'name': 'nano', 'factor': 1e-9},
    {'symbol': 'p', 'name': 'pico', 'factor': 1e-12},
    {'symbol': 'f', 'name': 'femto', 'factor': 1e-15},
    {'symbol': 'a', 'name': 'atto', 'factor': 1e-18},
    {'symbol': 'z', 'name': 'zepto', 'factor': 1e-21},
    {'symbol': 'y', 'name': 'yocto', 'factor': 1e-24},
]

prefixes = iterables.Table(_prefixes)


_units = [
    {
        'symbol': 'm',
        'name': 'meter',
        'quantity': 'length',
        'system': 'mks',
    },
    {
        'symbol': 'au',
        'name': 'astronomical unit',
        'quantity': 'length',
    },
    {
        'symbol': 'g',
        'name': 'gram',
        'quantity': 'mass',
        'system': 'cgs',
    },
    {
        'symbol': 'nuc',
        'name': 'nucleon',
        'quantity': 'mass',
    },
    {
        'symbol': 'amu',
        'name': 'atomic mass unit',
        'quantity': 'mass',
    },
    {
        'symbol': 's',
        'name': 'second',
        'quantity': 'time',
    },
    {
        'symbol': 'min',
        'name': 'minute',
        'quantity': 'time',
    },
    {
        'symbol': 'h',
        'name': 'hour',
        'quantity': 'time',
    },
    {
        'symbol': 'd',
        'name': 'day',
        'quantity': 'time',
    },
    {
        'symbol': 'A',
        'name': 'ampere',
        'quantity': 'current',
        'system': 'mks',
    },
    {
        'symbol': 'K',
        'name': 'kelvin',
        'quantity': 'temperature',
    },
    {
        'symbol': 'mol',
        'name': 'mole',
        'quantity': 'amount',
    },
    {
        'symbol': '#',
        'name': 'count',
        'quantity': 'number',
    },
    {
        'symbol': 'cd',
        'name': 'candela',
        'quantity': 'luminous intensity',
    },
    {
        'symbol': 'rad',
        'name': 'radian',
        'quantity': 'plane angle',
    },
    {
        'symbol': 'deg',
        'name': 'degree',
        'quantity': 'plane angle',
    },
    {
        'symbol': 'sr',
        'name': 'steradian',
        'quantity': 'solid angle',
    },
    {
        'symbol': 'Hz',
        'name': 'hertz',
        'quantity': 'frequency',
    },
    {
        'symbol': 'J',
        'name': 'joule',
        'quantity': 'energy',
        'system': 'mks',
    },
    {
        'symbol': 'erg',
        'name': 'erg',
        'quantity': 'energy',
        'system': 'cgs',
    },
    {
        'symbol': 'eV',
        'name': 'electronvolt',
        'quantity': 'energy',
    },
    {
        'symbol': 'N',
        'name': 'newton',
        'quantity': 'force',
        'system': 'mks',
    },
    {
        'symbol': 'dyn',
        'name': 'dyne',
        'quantity': 'force',
        'system': 'cgs',
    },
    {
        'symbol': 'Pa',
        'name': 'pascal',
        'quantity': 'pressure',
        'system': 'mks',
    },
    {
        'symbol': 'W',
        'name': 'watt',
        'quantity': 'power',
        'system': 'mks',
    },
    {
        'symbol': 'C',
        'name': 'coulomb',
        'quantity': 'charge',
        'system': 'mks',
    },
    {
        'symbol': 'statC',
        'name': 'statcoulomb',
        'quantity': 'charge',
        'system': 'cgs',
    },
    {
        'symbol': 'statA',
        'name': 'statampere',
        'quantity': 'current',
        'system': 'cgs',
    },
    {
        'symbol': 'statV',
        'name': 'statvolt',
        'quantity': 'potential',
        'system': 'cgs',
    },
    {
        'symbol': 'e',
        'name': 'fundamental charge',
        'quantity': 'charge',
    },
    {
        'symbol': 'V',
        'name': 'volt',
        'quantity': 'potential',
        'system': 'mks',
    },
    {
        'symbol': 'Ω',
        'name': 'ohm',
        'quantity': 'resistance',
        'system': 'mks',
    },
    {
        'symbol': 'S',
        'name': 'seimens',
        'quantity': 'conductance',
        'system': 'mks',
    },
    {
        'symbol': 'F',
        'name': 'farad',
        'quantity': 'capacitance',
        'system': 'mks',
    },
    {
        'symbol': 'Wb',
        'name': 'weber',
        'quantity': 'magnetic flux',
        'system': 'mks',
    },
    {
        'symbol': 'Mx',
        'name': 'maxwell',
        'quantity': 'magnetic flux',
        'system': 'cgs',
    },
    {
        'symbol': 'H',
        'name': 'henry',
        'quantity': 'inductance',
        'system': 'mks',
    },
    {
        'symbol': 'T',
        'name': 'tesla',
        'quantity': 'induction',
        'system': 'mks',
    },
    {
        'symbol': 'G',
        'name': 'gauss',
        'quantity': 'induction',
        'system': 'cgs',
    },
    {
        'symbol': 'lm',
        'name': 'lumen',
        'quantity': 'luminous flux',
    },
    {
        'symbol': 'lx',
        'name': 'lux',
        'quantity': 'illuminance',
    },
    {
        'symbol': 'Bq',
        'name': 'becquerel',
        'quantity': 'radioactivity',
    },
    {
        'symbol': 'Gy',
        'name': 'gray',
        'quantity': 'dosage',
        'system': 'mks',
    },
    {
        'symbol': 'P',
        'name': 'poise',
        'quantity': 'viscosity',
        'system': 'cgs',
    },
    {
        'symbol': '1',
        'name': 'unitless',
        'quantity': 'identity',
    },
]

units = iterables.Table(_units)


# A note about angles: Kalinin (2019) "On the status of plane and solid in the
# International System of Units (SI)" makes a very compelling argument that the
# plane- and solid-angle units should not be '1'; instead:
# - the plane-angle unit should be 'radian' ('rad'),
# - the solid-angle unit should be 'steradian' ('sr'), and
# - the plane angle should be a base quantity,
# - 'radian' should be considered a base unit,
# - 'steradian' should be considered a derived unit, with 'sr = rad^2'.

# References and notes on quantities, dimensions, and units:
# - https://en.wikipedia.org/wiki/International_System_of_Quantities#Base_quantities
# - https://www.nist.gov/pml/weights-and-measures/metric-si/si-units
# - This module uses 'H' to represent the dimension of temperature because the
#   SI character, $\Theta$, is not an ASCII character. I also considered 'O'
#   because of its similarity to $\Theta$, but it looks too much like '0'
#   (zero), which, ironically, looks more like $\Theta$ than 'O'.
# - This module adds an identity quantity with '1' as its dimension and unit

base_quantities = iterables.Table(
    [
        {'name': 'identity', 'dimension': '1', 'unit': '1'},
        {'name': 'amount', 'dimension': 'N', 'unit': 'mol'},
        {'name': 'current', 'dimension': 'I', 'unit': 'A'},
        {'name': 'length', 'dimension': 'L', 'unit': 'm'},
        {'name': 'luminous intensity', 'dimension': 'J', 'unit': 'cd'},
        {'name': 'mass', 'dimension': 'M', 'unit': 'kg'},
        {'name': 'temperature', 'dimension': 'H', 'unit': 'K'},
        {'name': 'time', 'dimension': 'T', 'unit': 's'},
    ]
)

# NOTE: We need to define these here, rather than via constants.py, in order to
# avoid the circular import that results from constants.Constant being a
# subclass of quantities.Scalar.
C = 2.99792458e10
"""The speed of light in cm/s."""
PI = np.pi
"""The ratio of a circle's circumference to its diameter."""

definitions = {
    'amount': {
        'dimensions': {
            'mks': 'N',
            'cgs': 'N',
        },
        'units': {
            'mks': 'mol',
            'cgs': 'mol',
        },
        'conversions': {},
    },
    'area': 'length^2',
    'capacitance': {
        'dimensions': {
            'mks': '(T^2 * I)^2 / (M * L^2)',
            'cgs': 'L',
        },
        'units': {
            'mks': 'F',
            'cgs': 'cm',
        },
        'conversions': {
            ('F', 'cm'): C**2 * 1e-9,
        },
    },
    'charge': {
        'dimensions': {
            'mks': 'I * T',
            'cgs': '(M^1/2 * L^3/2) / T',
        },
        'units': {
            'mks': 'C',
            'cgs': 'statC',
        },
        'conversions': {
            ('C', 'statC'): 10*C,
            ('e', 'C'): 1.6022e-19,
        },
    },
    'charge density': 'charge / volume',
    'conductance': {
        'dimensions': {
            'mks': '(T^3 * I^2) / (M * L^2)',
            'cgs': 'L / T',
        },
        'units': {
            'mks': 'S',
            'cgs': 'cm / s',
        },
        'conversions': {
            ('S', 'cm / s'): C**2 * 1e-5,
        },
    },
    'conductivity': 'conductance / length',
    'current': {
        'dimensions': {
            'mks': 'I',
            'cgs': '(M^1/2 * L^3/2) / T^2',
        },
        'units': {
            'mks': 'A',
            'cgs': 'statA',
        },
        'conversions': {
            ('A', 'statA'): 10*C,
        },
    },
    'current density': 'current / area',
    'displacement': {
        'dimensions': {
            'mks': 'I * T / L^2',
            'cgs': 'M^1/2 / (L^1/2 * T)',
        },
        'units': {
            'mks': 'C / m^2',
            'cgs': 'statC / m^2',
        },
        'conversions': {
            ('C / m^2', 'statC / m^2'): 4*PI * C * 1e-3,
        },
    },
    'dosage': {
        'dimensions': {
            'mks': 'L^2 / T^2',
            'cgs': 'L^2 / T^2',
        },
        'units': {
            'mks': 'Gy',
            'cgs': 'erg / g', # Historically, 'rad', but that's in use.
        },
        'conversions': {
            ('Gy', 'erg / g'): 1e4,
        },
    },
    'electric charge': 'charge',
    'electric field': 'potential / length',
    'electromotance': 'potential',
    'energy': {
        'dimensions': {
            'mks': '(M * L^2) / T^2',
            'cgs': '(M * L^2) / T^2',
        },
        'units': {
            'mks': 'J',
            'cgs': 'erg',
        },
        'conversions': {
            ('J', 'erg'): 1e7,
            ('eV', 'J'): 1.6022e-19,
        },
    },
    'energy density': 'energy / volume',
    'force': {
        'dimensions': {
            'mks': '(M * L) / T^2',
            'cgs': '(M * L) / T^2',
        },
        'units': {
            'mks': 'N',
            'cgs': 'dyn',
        },
        'conversions': {
            ('N', 'dyn'): 1e5,
        },
    },
    'frequency': {
        'dimensions': {
            'mks': '1 / T',
            'cgs': '1 / T',
        },
        'units': {
            'mks': 'Hz',
            'cgs': 'Hz',
        },
        'conversions': {},
    },
    'identity': {
        'dimensions': {
            'mks': '1',
            'cgs': '1',
        },
        'units': {
            'mks': '1',
            'cgs': '1',
        },
        'conversions': {},
    },
    'illumunance': { # See note about radian (Kalinin 2019).
        'dimensions': {
            'mks': 'J / L^2',
            'cgs': 'J / L^2',
        },
        'units': {
            'mks': 'cd * sr / m^2',
            'cgs': 'cd * sr / cm^2',
        },
        'conversions': {},
    },
    'impedance': {
        'dimensions': {
            'mks': '(M * L^2) / (T^3 * I)',
            'cgs': 'T / L',
        },
        'units': {
            'mks': 'ohm',
            'cgs': 's / cm',
        },
        'conversions': {
            ('ohm', 's / cm'): 1e5 / C**2,
        },
    },
    'inductance': {
        'dimensions': {
            'mks': '(M * L^2) / (I * T)^2',
            'cgs': 'T^2 / L',
        },
        'units': {
            'mks': 'H',
            'cgs': 's^2 / cm',
        },
        'conversions': {
            ('H', 's^2 / cm'): 1e5 / C**2,
        },
    },
    'induction': 'magnetic induction',
    'length': {
        'dimensions': {
            'mks': 'L',
            'cgs': 'L',
        },
        'units': {
            'mks': 'm',
            'cgs': 'cm',
        },
        'conversions': {
            ('m', 'cm'): 1e2,
            ('au', 'm'): 1.495978707e11,
        },
    },
    'luminous flux': { # See note about radian (Kalinin 2019).
        'dimensions': {
            'mks': 'J',
            'cgs': 'J',
        },
        'units': {
            'mks': 'cd * sr',
            'cgs': 'cd * sr',
        },
        'conversions': {},
    },
    'luminous intensity': {
        'dimensions': {
            'mks': 'J',
            'cgs': 'J',
        },
        'units': {
            'mks': 'cd',
            'cgs': 'cd',
        },
        'conversions': {},
    },
    'magnetic field': 'magnetic induction',
    'magnetic flux': {
        'dimensions': {
            'mks': '(M * L^2) / (T^2 * I)',
            'cgs': '(M^1/2 * L^3/2) / T',
        },
        'units': {
            'mks': 'Wb',
            'cgs': 'Mx',
        },
        'conversions': {
            ('Wb', 'Mx'): 1e8,
        },
    },
    'magnetic induction': {
        'dimensions': {
            'mks': 'M / (T^2 * I)',
            'cgs': 'M^1/2 / (L^1/2 * T)',
        },
        'units': {
            'mks': 'T',
            'cgs': 'G',
        },
        'conversions': {
            ('T', 'G'): 1e4,
        },
    },
    'magnetic intensity': {
        'dimensions': {
            'mks': 'I / L',
            'cgs': 'M^1/2 / (L^1/2 * T)',
        },
        'units': {
            'mks': 'A / m',
            'cgs': 'Oe',
        },
        'conversions': {
            ('A / m', 'Oe'): 4*PI * 1e-3,
        },
    },
    'magnetic moment': {
        'dimensions': {
            'mks': 'I * L^2',
            'cgs': '(M^1/2 * L^5/2) / T',
        },
        'units': {
            'mks': 'A * m^2',
            'cgs': 'Oe * cm^3',
        },
        'conversions': {
            ('A * m^2', 'Oe * cm^3'): 1e-3,
        },
    },
    'magnetization': 'magnetic intensity',
    'magnetomotance': 'current',
    'mass': {
        'dimensions': {
            'mks': 'M',
            'cgs': 'M',
        },
        'units': {
            'mks': 'kg',
            'cgs': 'g',
        },
        'conversions': {
            ('kg', 'g'): 1e3,
            ('nuc', 'kg'): 1.6605e-27,
            ('amu', 'kg'): 1.6605e-27,
        },
    },
    'mass density': 'mass / volume',
    'momentum': {
        'dimensions': {
            'mks': '(M * L) / T',
            'cgs': '(M * L) / T',
        },
        'units': {
            'mks': 'kg * m / s',
            'cgs': 'g * cm / s',
        },
        'conversions': {
            ('kg * m / s', 'g * cm / s'): 1e5,
        },
    },
    'momentum density': 'momentum / volume',
    'number': 'identity',
    'particle distribution': '1 / (length * velocity)^3',
    'permeability': {
        'dimensions': {
            'mks': '(M * L) / (I * T)^2',
            'cgs': '1',
        },
        'units': {
            'mks': 'H / m',
            'cgs': '1',
        },
        'conversions': {
            ('H / m', '1'): 1e7 / 4*PI,
        },
    },
    'permitivity': {
        'dimensions': {
            'mks': 'T^4 * I / (M * L^3)',
            'cgs': '1',
        },
        'units': {
            'mks': 'F / m',
            'cgs': '1',
        },
        'conversions': {
            ('F / m', '1'): 36*PI * 1e9,
        },
    },
    'plane angle': { # See note about radian (Kalinin 2019).
        'dimensions': {
            'mks': '1',
            'cgs': '1',
        },
        'units': {
            'mks': 'rad',
            'cgs': 'rad',
        },
        'conversions': {
            ('rad', 'deg'): 180 / PI,
        },
    },
    'polarization': 'charge / area',
    'potential': {
        'dimensions': {
            'mks': '(M * L^2) / (T^3 * I)',
            'cgs': '(M^1/2 * L^1/2) / T',
        },
        'units': {
            'mks': 'V',
            'cgs': 'statV',
        },
        'conversions': {
            ('V', 'statV'): 1e6 / C,
        },
    },
    'power': {
        'dimensions': {
            'mks': 'M * L^2 / T^3',
            'cgs': 'M * L^2 / T^3',
        },
        'units': {
            'mks': 'W',
            'cgs': 'erg / s',
        },
        'conversions': {
            ('W', 'erg / s'): 1e7,
        },
    },
    'power density': 'power / volume',
    'pressure': {
        'dimensions': {
            'mks': 'M / (L * T^2)',
            'cgs': 'M / (L * T^2)',
        },
        'units': {
            'mks': 'Pa',
            'cgs': 'dyn / cm^2', # also barye (Ba)?
        },
        'conversions': {
            ('Pa', 'dyn / cm^2'): 1e1,
        },
    },
    'radioactivity': {
        'dimensions': {
            'mks': '1 / T',
            'cgs': '1 / T',
        },
        'units': {
            'mks': 'Bq',
            'cgs': 'Ci',
        },
        'conversions': {
            ('Bq', 'Ci'): 1.0 / 3.7e10,
        },
    },
    'reluctance': {
        'dimensions': {
            'mks': '(I * T)^2 / (M * L^2)',
            'cgs': '1 / L',
        },
        'units': {
            'mks': 'A / Wb',
            'cgs': '1 / cm',
        },
        'conversions': {
            ('A / Wb', '1 / cm'): 4*PI * 1e-9,
        },
    },
    'resistance': 'impedance',
    'resistivity': 'resistance * length',
    'temperature': {
        'dimensions': {
            'mks': 'H',
            'cgs': 'H',
        },
        'units': {
            'mks': 'K',
            'cgs': 'K',
        },
        'conversions': {},
    },
    'thermal conductivity': 'power / (length * temperature)',
    'time': {
        'dimensions': {
            'mks': 'T',
            'cgs': 'T',
        },
        'units': {
            'mks': 's',
            'cgs': 's',
        },
        'conversions': {
            ('s', 'min'): 1.0 / 60.0,
            ('s', 'h'): 1.0 / 3600.0,
            ('s', 'd'): 1.0 / 86400.0,
        },
    },
    'solid angle': { # See note about radian (Kalinin 2019).
        'dimensions': {
            'mks': '1',
            'cgs': '1',
        },
        'units': {
            'mks': 'sr',
            'cgs': 'sr',
        },
        'conversions': {},
    },
    'vector potential': {
        'dimensions': {
            'mks': '(M * L) / (T^2 * I)',
            'cgs': '(M^1/2 * L^1/2) / T',
        },
        'units': {
            'mks': 'Wb / m',
            'cgs': 'G * cm',
        },
        'conversions': {
            ('Wb / m', 'G * cm'): 1e6,
        },
    },
    'velocity': 'length / time',
    'viscosity': {
        'dimensions': {
            'mks': 'M / (L * T)',
            'cgs': 'M / (L * T)',
        },
        'units': {
            'mks': 'kg / (m * s)',
            'cgs': 'P',
        },
        'conversions': {
            ('kg / (m * s)', 'P'): 1e1,
        },
    },
    'volume': 'length^3',
    'vorticity': 'frequency',
    'work': 'energy',
}


class Dimension(algebra.Expression):
    """An algebraic expression representing a physical dimension."""

    def __init__(
        self,
        expression: Union[str, iterables.Separable],
        **kwargs,
    ) -> None:
        if isinstance(expression, iterables.Separable):
            terms = [self._get_term(term) for term in expression]
            return super().__init__(terms, **kwargs)
        return super().__init__(expression, **kwargs)

    def _get_term(self, obj):
        """Create an appropriate algebraic term from input."""
        if base := getattr(obj, 'dimension', None):
            exponent = getattr(obj, 'exponent', 1)
            return algebra.Term(base, exponent)
        if isinstance(obj, algebra.Term):
            return obj
        return str(obj)


class UnitParsingError(Exception):
    """Error when attempting to parse string into unit."""

    def __init__(self, string: str) -> None:
        self.string = string

    def __str__(self) -> str:
        return f"Could not determine unit and magnitude of '{self.string}'"


class UnitConversionError(Exception):
    """Unknown unit conversion."""

    def __init__(self, u0: str, u1: str) -> None:
        self._from = u0
        self._to = u1

    def __str__(self) -> str:
        return f"Can't convert {self._from} to {self._to}"


class MetricPrefix(NamedTuple):
    """Metadata for a metric order-of-magnitude prefix."""

    symbol: str
    name: str
    factor: float


class BaseUnit(NamedTuple):
    """Metadata for a named unit without metric prefix."""

    symbol: str
    name: str
    quantity: str
    system: str=None
    aliases: iterables.Separable[str]=None


class NamedUnit(iterables.ReprStrMixin):
    """A single named unit and corresponding metadata."""

    _instances = {}
    _latest = None

    @classmethod
    def parse(cls, string: str):
        unit = named_units[string]
        magnitude = MetricPrefix(**unit['prefix'])
        reference = BaseUnit(**unit['base'])
        return magnitude, reference


    # The point of this somewhat awkward __new__ / __init__ combination is to
    # create a singleton instance for each unique named unit but to avoid
    # parsing the input twice -- first in __new__, to build the instance key,
    # then in __init__, to set the instance attributes. This approach is almost
    # certainly NOT thread safe.

    def __new__(cls, arg: Union[str, 'NamedUnit']):
        """Create a new instance or return an existing one."""
        if isinstance(arg, NamedUnit):
            return arg
        string = str(arg)
        try:
            key = cls.parse(string)
            available = cls._instances.get(key)
            if not available:
                cls._instances[key] = super().__new__(cls)
            cls._latest = key
        except KeyError:
            raise UnitParsingError(string)
        else:
            return cls._instances[key]

    # This will be true after the first pass through __init__ for a given unit.
    # The instance __getattribute__ uses it to control access to certain
    # attributes that are only necessary during instantiation.
    _init = False

    def __init__(self, arg: Union[str, 'NamedUnit']) -> None:
        self._arg = arg
        if not self._init and self._latest:
            self._magnitude, self._reference = self._latest
            self._latest = None
        self._name = None
        self._symbol = None
        self._scale = None
        self._quantity = None
        self._dimension = None
        self._init = True

    def __getattribute__(self, name: str) -> Any:
        if name == '_latest' and self._init:
            raise AttributeError(f"{name!r} is not accessible on instances")
        return super().__getattribute__(name)

    @property
    def name(self) -> str:
        """The full name of this unit."""
        if self._name is None:
            self._name = f"{self._magnitude.name}{self._reference.name}"
        return self._name

    @property
    def symbol(self) -> str:
        """The abbreviated symbol for this unit."""
        if self._symbol is None:
            self._symbol = f"{self._magnitude.symbol}{self._reference.symbol}"
        return self._symbol

    @property
    def scale(self) -> float:
        """The metric scale factor of this unit."""
        if self._scale is None:
            self._scale = self._magnitude.factor
        return self._scale

    @property
    def quantity(self) -> str:
        """The physical quantity of this unit."""
        if self._quantity is None:
            self._quantity = self._reference.quantity
        return self._quantity

    @property
    def dimension(self) -> str:
        """The physical dimension of this unit."""
        if self._dimension is None:
            system = self._reference.system or 'mks'
            dimensions = get_property(self.quantity, 'dimensions')
            self._dimension = dimensions[system]
        return self._dimension

    def __floordiv__(self, target: Union[str, 'NamedUnit']) -> float:
        """Compute the magnitude of this unit relative to another.

        Examples
        --------
        The following are all equivalent to the statement that there are 100
        centimeters per meter:

            >>> NamedUnit('centimeter') // NamedUnit('meter')
            100.0
            >>> NamedUnit('cm') // NamedUnit('m')
            100.0
            >>> NamedUnit('meter') // NamedUnit('centimeter')
            0.01

        Only one of the operands need be an instance of this type:

            >>> NamedUnit('second') // NamedUnit('day')
            86400.0
            >>> NamedUnit('second') // 'day'
            86400.0
            >>> 'second' // NamedUnit('day')
            86400.0

        Attempting this operation between two units with different dimensions
        will raise an exception:

            >>> NamedUnit('meter') // NamedUnit('second')
            <raises UnitConversionError>
        """
        other = type(self)(target)
        if self == other:
            return 1.0
        ratio = other.scale / self.scale
        if other._reference == self._reference:
            return ratio
        if other.quantity == self.quantity:
            pair = (other._reference.symbol, self._reference.symbol)
            factor = get_conversion_factor(pair, self.quantity)
            if factor:
                return ratio * factor
        raise UnitConversionError(self.name, other.name) from None

    def __rfloordiv__(self, target):
        """Support target // self. See `__floordiv__` for details."""
        return 1.0 / self.__floordiv__(target)

    def __eq__(self, other) -> bool:
        """True if two representations have equal magnitude and reference."""
        that = type(self)(other)
        same_magnitude = (self._magnitude == that._magnitude)
        same_reference = (self._reference == that._reference)
        return same_magnitude and same_reference

    def __str__(self) -> str:
        """A printable representation of this unit."""
        return f"'{self.name} | {self.symbol}'"


class UnitTerm(algebra.Term):
    """An algebraic term containing a single named unit."""

    def __init__(
        self,
        arg: Union[str, algebra.Term],
        exponent: Union[str, int]=None,
    ) -> None:
        super().__init__(arg, exponent)
        self._unit = NamedUnit(self.base)
        self.dimension = self._unit.dimension
        self.quantity = self._unit.quantity

    def __floordiv__(self, other):
        if isinstance(other, UnitTerm):
            if other.exponent == self.exponent:
                return (self._unit // other._unit) ** self.exponent
            raise ValueError(
                f"Can't compute {self} // {other}"
                ". Ratio must be unitless."
            )
        return NotImplemented


class Unit(algebra.Expression):
    """An algebraic expression representing a physical unit."""

    def __init__(
        self,
        expression: Union[str, iterables.Separable],
        **kwargs,
    ) -> None:
        super().__init__(expression, **kwargs)
        self._dimension = None

    @property
    def terms(self) -> List[UnitTerm]:
        return [UnitTerm(term) for term in super().terms]

    @property
    def dimension(self):
        """The dimension of this unit."""
        if self._dimension is None:
            self._dimension = Dimension(self.terms)
        return self._dimension

    def __floordiv__(self, other: 'Unit') -> float:
        """Compute the magnitude of this unit relative to another.

        This method essentially computes the amount of `self` per `other`. The
        result is the numerical factor, N, necessary to convert a quantity with
        unit `other` to the equivalent quantity with unit `self`. In algebraic
        terms, suppose you have an amount q0 of some physical quantity when
        expressed in unit u0. The equivalent amount when expressed in unit u1 is
        q0 [u0] = q0 * (u1 // u0) [u1] = q1 [u1].

        Examples
        --------
        Create two unit objects representing a meter and a centimeter, and
        compute their relative magnitude:

            >>> m = Unit('m')
            >>> cm = Unit('cm')
            >>> m // cm
            0.01
            >>> cm // m
            100.0

        These results are equivalent to the statement that there are 100
        centimeters in a meter.
        """
        self_dim = self.dimension.reduced
        other_dim = other.dimension.reduced
        if self_dim != other_dim:
            raise TypeError(
                "Can't compute a numerical factor from"
                f" {other.format(separator=' * ')!r}"
                f" (dimension {other_dim})"
                " with"
                f" {self.format(separator=' * ')!r}"
                f" (dimension {self_dim})"
                ". You can use the '/' operator if you want to"
                " create a new unit representing the ratio."
            )
        ratio = self / other
        factor = 1.0
        for term in ratio:
            quantity = get_quantity(term.quantity)
            reference = UnitTerm(quantity.units['mks'], term.exponent)
            factor *= term // reference
        return factor


def get_property(name: str, key: str):
    """Get a named property of a defined quantity."""
    if name not in definitions:
        raise KeyError(f"No definition for '{name}'")
    q = definitions[name]
    if isinstance(q, dict):
        return q.get(key, {})
    if not isinstance(q, str):
        raise TypeError(f"Expected {name} to be a string")
    return parse_quantity(q, key)


def parse_quantity(string: str, key: str):
    """Parse a string representing a compound quantity."""
    if ' ' in string and all(c not in string for c in ['*', '/']):
        string = string.replace(' ', '_')
    expr = algebra.Expression(string)
    parts = []
    for term in expr:
        prop = get_property(term.base.replace('_', ' '), key)
        tmp = {
            k: algebra.Term(v, term.exponent)
            for k, v in prop.items()
        }
        parts.append(tmp)
    keys = {key for part in parts for key in part.keys()}
    merged = {key: [] for key in keys}
    for part in parts:
        for key, value in part.items():
            merged[key].append(value)
    return {k: str(algebra.Expression(v).reduced) for k, v in merged.items()}


class Metric(iterables.ReprStrMixin):
    """A canonical physical quantity within a named system."""

    def __init__(self, dimension, unit) -> None:
        self.dimension = Dimension(dimension)
        self.unit = Unit(unit)

    def __eq__(self, other: Any) -> bool:
        """True if two instances have the same unit and dimension."""
        if isinstance(other, Metric):
            return self.unit == other.unit and self.dimension == other.dimension
        return NotImplemented

    def __str__(self) -> str:
        """A simplified representation of this object."""
        return f"{self.unit=} {self.dimension=}"


class Quantity(iterables.ReprStrMixin):
    """A single physical quantity."""

    def __init__(
        self,
        dimensions: Mapping[str, str],
        units: Mapping[str, str],
        conversions: Mapping[Tuple[str], float]=None,
    ) -> None:
        self.dimensions = dimensions
        self.conversions = conversions or {}
        self.units = {
            **units,
            'alt': [
                unit for key in conversions.keys() for unit in key
                if unit not in units.values()
            ],
        }

    def __getitem__(self, system: str):
        """Get this quantity's representation in the named metric system."""
        try:
            name = system.lower()
            dimension = self.dimensions[name]
            unit = self.units[name]
        except KeyError:
            raise KeyError(f"No metric available for system '{system}'")
        else:
            return Metric(dimension=dimension, unit=unit)

    def __eq__(self, other) -> bool:
        """True if two quantities have equal attributes."""
        if isinstance(other, Quantity):
            attrs = ['dimensions', 'units', 'conversions']
            try:
                equal = [
                    getattr(self, attr) == getattr(other, attr)
                    for attr in attrs
                ]
            except AttributeError:
                return False
            else:
                return all(equal)
        return NotImplemented

    def __str__(self) -> str:
        """A simplified representation of this object."""
        attrs = ['dimensions', 'units', 'conversions']
        return ', '.join(f"{attr}={getattr(self, attr)}" for attr in attrs)


def get_quantity(name: str):
    """Retrieve a named quantity or build it from a formula."""
    result = {
        key: get_property(name, key=key)
        for key in ['dimensions', 'units']
    }
    definition = definitions[name]
    try:
        options = definition['conversions']
    except (KeyError, TypeError):
        options = {}
    result['conversions'] = options
    return Quantity(**result)


class MetricKeyError(KeyError):
    """Metric-system mapping-key error."""
    pass


class MetricTypeError(TypeError):
    """Metric-system argument-type error."""
    pass


class MetricSearchError(KeyError):
    """Error while searching for a requested metric."""
    pass


class MetricSystem(iterables.MappingBase, iterables.ReprStrMixin):
    """Representations of physical quantities within a given metric system."""

    def __init__(self, name: str) -> None:
        """Initialize this instance.

        Parameters
        ----------
        name : str
            The name of a known metric system (e.g., 'mks').
        """
        self.name = name.lower()
        super().__init__(definitions.keys())
        self.dimensions = {
            get_property(q, 'dimensions')[self.name]: q
            for q in self
        }

    def __getitem__(self, key: str):
        """Get the metric for the requested quantity in this system."""
        try:
            quantity = get_quantity(key)
        except KeyError:
            raise MetricKeyError(f"No known quantity called '{key}'")
        else:
            return quantity[self.name]

    def get_unit(
        self,
        unit: Union[str, Unit]=None,
        dimension: Union[str, Dimension]=None,
        quantity: Union[str, Quantity]=None,
    ) -> Optional[Unit]:
        """Get a canonical unit from a given unit, dimension, or quantity.

        This method will search for the unit in the current metric system based
        on `unit`, `dimension`, or `quantity`, in that order. All arguments
        default to `None`. If `unit` is not `None`, this method will attempt to
        return the equivalent canonical unit; if either `dimension` or
        `quantity` is not `None`, this method will attempt to return the unique
        corresponding unit.

        Parameters
        ----------
        unit : string or Unit
            A unit of measure in any system.

        dimension : string or Dimension
            A physical dimension.

        quantity : string or Quantity
            A physical quantity.

        Returns
        -------
        Unit
            The corresponding unit in the current metric system.

        """
        methods = {
            k: getattr(self, f'_unit_from_{k}')
            for k in ('unit', 'dimension', 'quantity')
        }
        targets = {
            'unit': unit,
            'dimension': dimension,
            'quantity': quantity,
        }
        return self._get_unit(methods, targets)

    T = TypeVar('T', Unit, Dimension, Quantity)
    T = Union[Unit, Dimension, Quantity]

    def _get_unit(
        self,
        methods: Dict[str, Callable[[T], Unit]],
        targets: Dict[str, T],
    ) -> Unit:
        """Search logic for `get_unit`."""
        nonnull = {k: v for k, v in targets.items() if v}
        cases = [(methods[k], v) for k, v in nonnull.items()]
        for (method, arg) in cases:
            if str(arg) == '1':
                return self['identity'].unit
            if result := method(arg):
                return result
        args = self._format_targets(nonnull)
        errmsg = f"Could not determine unit in {self.name} from {args}"
        raise MetricSearchError(errmsg)

    def _format_targets(self, targets: Dict[str, T]):
        """Format `get_unit` targets for printing."""
        if not targets:
            return "nothing"
        args = [f'{k}={v!r}' for k, v in targets.items()]
        if 0 < len(args) < 3:
            return ' or '.join(args)
        return f"{', '.join(str(arg) for arg in args[:-1])}, or {args[-1]}"

    def _unit_from_unit(self, unit) -> Unit:
        """Get the canonical unit corresponding to the given unit."""
        string = str(unit)
        if string == '1':
            return self['identity'].unit
        try:
            quantities = [
                NamedUnit(term.base).quantity
                for term in Unit(string)
            ]
        except UnitParsingError:
            pass
        else:
            if all(quantity in self for quantity in quantities):
                bases = [self[quantity].unit for quantity in quantities]
                exponents = [term.exponent for term in Unit(string)]
                expression = [
                    base ** exponent
                    for base, exponent in zip(bases, exponents)
                ]
                return Unit(expression)

    def _unit_from_dimension(self, dimension) -> Unit:
        """Get the canonical unit corresponding to the given dimension."""
        string = str(dimension)
        if string == '1':
            return self['identity'].unit
        if string in self.dimensions:
            dimension = self.dimensions[string]
            return self[dimension].unit

    def _unit_from_quantity(self, quantity) -> Unit:
        """Get the canonical unit corresponding to the given quantity."""
        string = str(quantity)
        try:
            unit = self[string].unit
        except MetricKeyError:
            parsed = parse_quantity(string, 'units')
            unit = parsed[self.name]
        return unit

    def __eq__(self, other) -> bool:
        """True if two systems have the same `name` attribute."""
        if isinstance(other, MetricSystem):
            return other.name == self.name
        return NotImplemented

    def keys(self) -> KeysView[str]:
        return super().keys()

    def values(self) -> ValuesView[Metric]:
        return super().values()

    def items(self) -> ItemsView[str, Metric]:
        return super().items()

    def __str__(self) -> str:
        """A simplified representation of this object."""
        return str(self.name)


def get_conversion_factor(pair: Tuple[str], quantity: str=None):
    """Get the conversion factor for the given pair of units."""
    if quantity:
        return _search_conversions(*pair, quantity)
    for quantity in definitions:
        if f := _search_conversions(*pair, quantity):
            return f


def _search_conversions(u0: str, u1: str, name: str):
    """Helper function for getting a conversion factor."""
    defined = get_quantity(name)
    if not (options := defined.conversions):
        return
    units = {'u0': u0, 'u1': u1}
    for k, v in units.items():
        if v in defined.units:
            units[k] = defined.units[v]
    u0, u1 = units['u0'], units['u1']
    if u0 == u1:
        return 1.0
    forward, reverse = (u0, u1), (u1, u0)
    if forward in options:
        return options[forward]
    if reverse in options:
        return 1.0 / options[reverse]
    result = 1.0
    ux = None
    for pair, factor in options.items():
        if u0 in pair:
            if u0 == pair[0]:
                result *= factor
                ux = pair[1]
            else:
                result /= factor
                ux = pair[0]
    forward, reverse = (ux, u1), (u1, ux)
    if forward in options:
        return result * options[forward]
    if reverse in options:
        return result / options[reverse]


def build_unit_aliases(prefix, unit):
    """Define all aliases for the given metric prefix and base unit."""
    key = [f"{prefix[k]}{unit[k]}" for k in ['name', 'symbol']]
    if prefix['symbol'] == 'μ':
        key += [f"u{unit['symbol']}"]
    return tuple(key)


# Tables may not be necessary with this.
named_units = iterables.AliasedMapping(
    {
        build_unit_aliases(prefix, unit): {'base': unit, 'prefix': prefix}
        for prefix in prefixes for unit in units
    }
)


class ComparisonError(TypeError):
    """Incomparable instances of the same type."""

    def __init__(self, __this: Any, __that: Any, name: str):
        self.this = getattr(__this, name, None)
        self.that = getattr(__that, name, None)

    def __str__(self) -> str:
        return f"Can't compare '{self.this}' to '{self.that}'"


class same:
    """A decorator class that enforces object consistency.

    When used to decorate a method that takes two arguments, this class will
    ensure that the arguments have equal values of a named attribute. This may
    be useful when writing binary comparison methods that are only valid for two
    objects of the same kind (e.g., physical objects with the same dimension).
    """

    def __init__(
        self,
        *names: str,
        allowed: Iterable[Type]=None,
    ) -> None:
        self.names = names
        self.allowed = iterables.Separable(allowed)

    def __call__(self, func: Callable) -> Callable:
        """Ensure attribute consistency before calling `func`."""
        if not self.names:
            return func
        @functools.wraps(func)
        def wrapper(this, that):
            allowed = (type(this), *self.allowed)
            if not isinstance(that, allowed):
                return NotImplemented
            if isinstance(that, type(this)):
                for name in self.names:
                    if not self._comparable(this, that, name):
                        raise ComparisonError(this, that, name) from None
            return func(this, that)
        return wrapper

    def _comparable(self, this: Any, that: Any, name: str) -> bool:
        """Check whether the instances are comparable."""
        return getattr(this, name) == getattr(that, name)


class Comparable(metaclass=abc.ABCMeta):
    """The base class for all comparable objects.

    Comparable objects support relative ordering. Concrete implementations of
    this class must define the six binary comparison operators (a.k.a "rich
    comparison" operators): `__lt__`, `__gt__`, `__le__`, `__ge__`, `__eq__`,
    and `__ne__`.
    """

    __slots__ = ()

    __hash__ = None

    @abc.abstractmethod
    def __lt__(self, other) -> bool:
        """True if self < other."""
        pass

    @abc.abstractmethod
    def __le__(self, other) -> bool:
        """True if self <= other."""
        pass

    @abc.abstractmethod
    def __gt__(self, other) -> bool:
        """True if self > other."""
        pass

    @abc.abstractmethod
    def __ge__(self, other) -> bool:
        """True if self >= other."""
        pass

    @abc.abstractmethod
    def __eq__(self, other) -> bool:
        """True if self == other."""
        pass

    @abc.abstractmethod
    def __ne__(self, other) -> bool:
        """True if self != other."""
        pass


class RealValued(Comparable):
    """A comparable object with one or more numerical values.

    This class borrows from base classes in the ``numbers`` module but it isn't
    in the numerical hierarchy because it doesn't require conversion to a single
    numerical value. However, it does register ``numbers.Real`` as a virtual
    subclass.
    """

    @abc.abstractmethod
    def __bool__(self) -> bool:
        """The truth value of this object as returned by bool(self)."""
        pass

    @abc.abstractmethod
    def __abs__(self):
        """Implements abs(self)."""
        pass

    @abc.abstractmethod
    def __neg__(self):
        """Called for -self."""
        pass

    @abc.abstractmethod
    def __pos__(self):
        """Called for +self."""
        pass

    @abc.abstractmethod
    def __add__(self, other):
        """Called for self + other."""
        pass

    @abc.abstractmethod
    def __radd__(self, other):
        """Called for other + self."""
        pass

    def __iadd__(self, other):
        """Called for self += other."""
        return NotImplemented

    def __sub__(self, other):
        """Called for self - other."""
        return self + -other

    def __rsub__(self, other):
        """Called for other - self."""
        return -self + other

    def __isub__(self, other):
        """Called for self -= other."""
        return NotImplemented

    @abc.abstractmethod
    def __mul__(self, other):
        """Called for self * other."""
        pass

    @abc.abstractmethod
    def __rmul__(self, other):
        """Called for other * self."""
        pass

    def __imul__(self, other):
        """Called for self *= other."""
        return NotImplemented

    @abc.abstractmethod
    def __truediv__(self, other):
        """Called for self / other."""
        pass

    @abc.abstractmethod
    def __rtruediv__(self, other):
        """Called for other / self."""
        pass

    def __itruediv__(self, other):
        """Called for self /= other."""
        return NotImplemented

    @abc.abstractmethod
    def __pow__(self, other):
        """Called for self ** other or pow(self, other)."""
        pass

    @abc.abstractmethod
    def __rpow__(self, other):
        """Called for other ** self or pow(other, self)."""
        pass

    def __ipow__(self, other):
        """Called for self **= other."""
        return NotImplemented

RealValued.register(numbers.Real)
RealValued.register(arrays.NumericalSequence)


class Quantified(RealValued, iterables.ReprStrMixin):
    """The base class for all quantified objects.

    A quantified object has an amount and a quantity. This class does not place
    any restrictions on the type of either attribute.

    This class declares which operators require that their operands have
    consistent quantities, and enforces those requirements on subclasses.
    """

    _quantified = (
        '__lt__',
        '__le__',
        '__gt__',
        '__ge__',
        '__eq__',
        '__ne__',
        '__add__',
        '__sub__',
        '__iadd__',
        '__isub__',
    )

    def __init_subclass__(cls, **kwargs) -> None:
        """Enforce quantifiability on method implementations.

        This implementation uses an instance of the `same` decorator class to
        ensure that the operands to each method named in the `_quantified` class
        attribute have the same quantity (i.e., the same value of their
        `quantity` attribute). Subclasses may individually override this feature
        by passing the name of the method to the `overrides` keyword in their
        class definition.
        """
        allowed = kwargs.get('allowed', {})
        for method in cls._quantified:
            current = getattr(cls, method, None)
            if current:
                update = same('quantity', allowed=allowed.get(method))
                updated = update(current)
                setattr(cls, method, updated)

    def __init__(self, amount: Any, quantity: Any) -> None:
        self.amount = amount
        self.quantity = quantity

    def __str__(self) -> str:
        """A simplified representation of this object."""
        return f"{self.amount} {self.quantity}"


class Ordered(Quantified):
    """A quantified object that supports comparisons

    An ordered object has an amount and a quantity. The amount must be formally
    comparable -- that is, a comparison to another amount using one of the six
    binary relations (i.e., <, >, <=, >=, ==, !=) must produce well-defined
    results. The quantity may be anything that supports equality comparison,
    which should be true unless the object explicitly disables `__eq__`.
    """

    def __init__(self, amount: Comparable, quantity: Any) -> None:
        super().__init__(amount, quantity)

    def __lt__(self, other: 'Ordered') -> bool:
        return self.amount < other.amount

    def __le__(self, other: 'Ordered') -> bool:
        return self.amount <= other.amount

    def __gt__(self, other: 'Ordered') -> bool:
        return self.amount > other.amount

    def __ge__(self, other: 'Ordered') -> bool:
        return self.amount >= other.amount

    def __eq__(self, other: 'Ordered') -> bool:
        return self.amount == other.amount

    def __ne__(self, other) -> bool:
        """True if self != other.

        Explicitly defined with respect to `self.__eq__` to promote consistency
        in subclasses that overload `__eq__`.
        """
        return not self.__eq__(other)

    def __str__(self) -> str:
        """A simplified representation of this object."""
        return f"{self.amount} {self.quantity}"


class Measured(Ordered):
    """An ordered object with a unit.

    Building on the `Ordered` class, a measured object must have an amount and a
    unit. The amount must be formally real-valued in the sense that the
    following arithmetic operations must produce well-defined results:
    - unary `-`, `+`, and `abs`
    - binary `+` and `-` between two instances with an identical unit
    - binary `*` and `/` between two instances
    - symmetric binary `*` between an instance and a number
    - right-sided `/` and `**` between an instance and a number

    Notes on allowed binary arithmetic operations:
    - This class does not support floor division (`//`) in any form because of
      the ambiguity it would create with `Unit` floor division.
    - This class does not support floating-point division (`/`) in which the
      left operand is not the same type or a subtype. The reason for this choice
      is that the result may be ambiguous. For example, suppose we have an
      instance called ``d`` with values ``[10.0, 20.0]`` and unit ``cm``.
      Whereas the result of ``d / 2.0`` should clearly be a new instance with
      values ``[5.0, 10.0]`` and the same unit, it is unclear whether the values
      of ``2.0 / d`` should be element-wise ratios (i.e., ``[0.2, 0.1]``) or a
      single value (e.g., ``2.0 / ||d||``) and it is not at all obvious what the
      unit or dimensions should be.
    """

    def __init__(
        self,
        amount: RealValued,
        unit: Union[str, Unit]=None,
    ) -> None:
        # Implementation note: converting unit to a Unit, then assigning to
        # self.unit, and finally converting to a string while passing to the
        # parent class not only fixes the types but also normalizes the argument
        # value via `algebra.Expression`.
        self.unit = Unit(unit or '1')
        super().__init__(amount, str(self.unit))

    def to(self, new: Union[str, Unit]):
        """Create a copy of this instance converted to the new unit."""
        scale = Unit(new) // self.unit
        amount = (scale * self).amount
        return self._new(amount=amount, unit=new)

    def _new(self, **updated):
        """Create a new instance with updated attributes."""
        attrs = self._update_attrs(updated)
        return type(self)(**attrs)

    def _update_attrs(self, updates: dict):
        """Compute attribute updates. Extracted for overloading."""
        init = list(inspect.signature(self.__init__).parameters)
        attrs = {p: getattr(self, p) for p in init}
        attrs.update(updates)
        return attrs

    def __bool__(self) -> bool:
        return bool(self.amount)

    def __abs__(self):
        return self._new(amount=abs(self.amount), unit=self.unit)

    def __neg__(self):
        return self._new(amount=-self.amount, unit=self.unit)

    def __pos__(self):
        return self._new(amount=+self.amount, unit=self.unit)

    def __add__(self, other: 'Measured'):
        return self._new(amount=self.amount + other.amount, unit=self.unit)

    def __radd__(self, other: Any):
        return NotImplemented

    def __sub__(self, other: 'Measured'):
        return self._new(amount=self.amount - other.amount, unit=self.unit)

    def __rsub__(self, other: Any):
        return NotImplemented

    def __mul__(self, other: Any):
        if isinstance(other, Measured):
            amount = self.amount * other.amount
            unit = self.unit * other.unit
            return self._new(amount=amount, unit=unit)
        if isinstance(other, numbers.Number):
            return self._new(amount=self.amount * other, unit=self.unit)
        return NotImplemented

    def __rmul__(self, other: Any):
        if isinstance(other, numbers.Number):
            return self._new(amount=other * self.amount, unit=self.unit)
        return NotImplemented

    def __truediv__(self, other: Any):
        if isinstance(other, Measured):
            amount = self.amount / other.amount
            unit = self.unit / other.unit
            return self._new(amount=amount, unit=unit)
        if isinstance(other, numbers.Number):
            return self._new(amount=self.amount / other, unit=self.unit)
        return NotImplemented

    def __rtruediv__(self, other: Any):
        return NotImplemented

    def __pow__(self, other: Any):
        if isinstance(other, numbers.Number):
            return self._new(amount=self.amount ** other, unit=self.unit ** other)
        return NotImplemented

    def __rpow__(self, other: Any):
        return NotImplemented

    def __str__(self) -> str:
        return f"{self.amount} [{self.unit}]"

    def copy(self, **updates):
        """Create a shallow copy of this object with optional updates."""
        return self._new(**updates)


allowed = {m: numbers.Real for m in ['__lt__', '__le__', '__gt__', '__ge__']}
class Scalar(Measured, allowed=allowed):
    """A single-valued measured object with an associated unit."""

    @property
    def value(self) -> RealValued:
        """The current numerical value."""
        return self.amount

    def __lt__(self, other: Ordered) -> bool:
        if isinstance(other, Comparable):
            return self.amount < other
        return super().__lt__(other)

    def __le__(self, other: Ordered) -> bool:
        if isinstance(other, Comparable):
            return self.amount <= other
        return super().__le__(other)

    def __gt__(self, other: Ordered) -> bool:
        if isinstance(other, Comparable):
            return self.amount > other
        return super().__gt__(other)

    def __ge__(self, other: Ordered) -> bool:
        if isinstance(other, Comparable):
            return self.amount >= other
        return super().__ge__(other)

    def __float__(self):
        """Called for float(self)."""
        return float(self.value)

    def __int__(self):
        """Called for int(self)."""
        return int(self.value)

    def __round__(self, ndigits: int=None):
        """Called for round(self)."""
        return round(self.value, ndigits=ndigits)

    def __floor__(self):
        """Called for math.floor(self)."""
        return math.floor(self.value)

    def __ceil__(self):
        """Called for math.ceil(self)."""
        return math.ceil(self.value)

    def __trunc__(self):
        """Called for math.trunc(self)."""
        return math.trunc(self.value)

    def __mul__(self, other: Any):
        if isinstance(other, Variable):
            return NotImplemented
        return super().__mul__(other)

    def __truediv__(self, other: Any):
        if isinstance(other, Variable):
            return NotImplemented
        return super().__truediv__(other)

    def __iadd__(self, other: 'Scalar'):
        self.amount = self.amount + other.amount
        return self

    def __isub__(self, other: 'Scalar'):
        self.amount = self.amount - other.amount
        return self

    def __imul__(self, other: Any):
        if isinstance(other, Measured):
            self.amount = self.amount * other.amount
            self.unit = self.unit * other.unit
            return self
        if isinstance(other, numbers.Number):
            self.amount = self.amount * other
            return self
        return NotImplemented

    def __itruediv__(self, other: Any):
        if isinstance(other, Measured):
            self.amount = self.amount / other.amount
            self.unit = self.unit / other.unit
            return self
        if isinstance(other, numbers.Number):
            self.amount = self.amount / other
            return self
        return NotImplemented

    def __ipow__(self, other: Any):
        if isinstance(other, numbers.Number):
            self.amount = self.amount ** other
            self.unit = self.unit ** other
            return self
        return NotImplemented

    def __hash__(self):
        """Called for hash(self)."""
        return hash((self.value, str(self.unit)))

    def copy(self, **updates):
        if 'value' in updates:
            updates['amount'] = updates.pop('value')
        return super().copy(**updates)


class Vector(Measured):
    """A multi-valued measured object with a single unit."""

    def __init__(
        self,
        amount: RealValued,
        unit: Union[str, Unit]=None,
    ) -> None:
        super().__init__(amount, unit=unit)
        self._values = None

    @property
    def values(self) -> RealValued:
        """The current numerical values."""
        if self._values is None:
            self._values = list(iterables.Separable(self.amount))
        return self._values

    def __len__(self):
        """Called for len(self)."""
        return len(self.values)

    def __iter__(self):
        """Called for iter(self)."""
        return iter(self.values)

    def __contains__(self, item):
        """Called for item in self."""
        return item in self.values

    def __add__(self, other: Any):
        if isinstance(other, Vector):
            values = [s + o for s, o in zip(self.values, other.values)]
            return self._new(amount=values, unit=self.unit)
        if isinstance(other, Measured):
            values = [s + other for s in self.values]
            return self._new(amount=values, unit=self.unit)
        return NotImplemented

    def __sub__(self, other: Any):
        if isinstance(other, Vector):
            values = [s - o for s, o in zip(self.values, other.values)]
            return self._new(amount=values, unit=self.unit)
        if isinstance(other, Measured):
            values = [s - other for s in self.values]
            return self._new(amount=values, unit=self.unit)
        return NotImplemented

    def __mul__(self, other: Any):
        if isinstance(other, Vector):
            values = [s * o for s, o in zip(self.values, other.values)]
            unit = self.unit * other.unit
            return self._new(amount=values, unit=unit)
        if isinstance(other, Scalar):
            other = float(other)
        if isinstance(other, RealValued):
            values = [s * other for s in self.values]
            return self._new(amount=values, unit=self.unit)
        return NotImplemented

    def __rmul__(self, other: Any):
        if isinstance(other, Scalar):
            other = float(other)
        if isinstance(other, numbers.Number):
            values = [other * s for s in self.values]
            return self._new(amount=values, unit=self.unit)
        return NotImplemented

    def __truediv__(self, other: Any):
        if isinstance(other, Vector):
            values = [s / o for s, o in zip(self.values, other.values)]
            unit = self.unit / other.unit
            return self._new(amount=values, unit=unit)
        if isinstance(other, Scalar):
            other = float(other)
        if isinstance(other, RealValued):
            values = [s / other for s in self.values]
            return self._new(amount=values, unit=self.unit)
        return NotImplemented

    def __pow__(self, other: Any):
        if isinstance(other, numbers.Number):
            values = [s ** other for s in self.values]
            unit = self.unit ** other
            return self._new(amount=values, unit=unit)
        return NotImplemented

    def copy(self, **updates):
        if 'values' in updates:
            updates['amount'] = updates.pop('values')
        return super().copy(**updates)


IndexLike = TypeVar('IndexLike', Iterable[int], slice, type(Ellipsis))
IndexLike = Union[Iterable[int], slice, type(Ellipsis)]

UnitLike = TypeVar('UnitLike', str, Unit)
UnitLike = Union[str, Unit]


allowed = {'__add__': float, '__sub__': float}
class Variable(Vector, arrays.Array, allowed=allowed):
    """A vector with values stored in a numerical array.

    The result of binary arithmetic operations on instances of this class are
    similar to those of `Vector`, but differ in the following ways:
    1. Multiplication (`*`) and division (`/`) accept operands with different
       axes, as long as any repeated axes have the same length in both operands.
       The result will contain all unique axes from its operands.
    2. Addition (`+`) and subtraction (`-`) accept real numbers as right-sided
       operands. The result is a new instance with the operation applied to the
       underlying array.
    """

    def __init__(
        self,
        values: Iterable[numbers.Number],
        unit: str,
        axes: Iterable[str],
    ) -> None:
        self.array = arrays.Array(values, axes)
        super().__init__(self.array, unit)

    def __getitem__(self, *args: IndexLike):
        """Create a new instance from a subset of data."""
        builtin = (int, slice, type(...))
        unwrapped = iterables.unwrap(args)
        standard = (
            isinstance(unwrapped, builtin)
            or all(isinstance(arg, builtin) for arg in unwrapped)
        )
        if standard:
            # Handles v[:], v[...], v[i, :], v[:, j], and v[i, j] (i, j ints)
            return self._new(values=self.array[unwrapped])
        expanded = self._expand_ellipsis(unwrapped)
        shape = self.data.shape
        idx = [
            range(shape[i])
            if isinstance(arg, slice) else arg
            for i, arg in enumerate(expanded)
        ]
        indices = np.ix_(*[index for index in idx])
        return self._new(values=self.array[indices])

    def _expand_ellipsis(self, user: Tuple[Any, ...]) -> Tuple[slice, ...]:
        """Expand an ``Ellipsis`` into one or more ``slice`` objects."""
        if Ellipsis not in user:
            return user
        length = self.naxes - len(user) + 1
        start = user.index(Ellipsis)
        return tuple([
            *user[slice(start)],
            *([slice(None)] * length),
            *user[slice(start+length, self.naxes)],
        ])

    def __getattr__(self, name: str):
        """Get an attribute from the base array object."""
        return getattr(self.array, name)

    def __eq__(self, other: Any):
        """True if two instances have the same values and attributes."""
        if not isinstance(other, Variable):
            return NotImplemented
        if not self._equal_attrs(other):
            return False
        return np.array_equal(other, self)

    def _equal_attrs(self, other: 'Variable'):
        """True if two instances have the same attributes."""
        return all(
            getattr(other, attr) == getattr(self, attr)
            for attr in {'unit', 'axes'}
        )

    def __add__(self, other: Any):
        if isinstance(other, numbers.Real):
            return self._new(values=self.data + other)
        if isinstance(other, Variable) and self.axes != other.axes:
            return NotImplemented
        return super().__add__(other)

    def __sub__(self, other: Any):
        if isinstance(other, numbers.Real):
            return self._new(values=self.data - other)
        if isinstance(other, Variable) and self.axes != other.axes:
            return NotImplemented
        return super().__sub__(other)

    def __mul__(self, other: Any):
        if isinstance(other, Variable):
            axes = sorted(tuple(set(self.axes + other.axes)))
            sarr, oarr = self._extend_arrays(other, axes)
            amount = sarr * oarr
            unit = self.unit * other.unit
            return self._new(values=amount, unit=unit, axes=axes)
        return super().__mul__(other)

    def __truediv__(self, other: Any):
        if isinstance(other, Variable):
            axes = sorted(tuple(set(self.axes + other.axes)))
            sarr, oarr = self._extend_arrays(other, axes)
            amount = sarr / oarr
            unit = self.unit / other.unit
            return self._new(values=amount, unit=unit, axes=axes)
        return super().__truediv__(other)

    @property
    def shape_dict(self) -> Dict[str, int]:
        """Label and size for each axis."""
        return {a: n for a, n in zip(self.axes, self.data.shape)}

    def _extend_arrays(
        self,
        other: 'Variable',
        axes: Tuple[str],
    ) -> Tuple[np.ndarray]:
        """Extract arrays with extended axes.

        This method determines the set of unique axes shared by this
        instance and `other`, then extracts arrays suitable for computing a
        product or ratio that has the full set of axes.
        """
        tmp = {**other.shape_dict, **self.shape_dict}
        full_shape = tuple(tmp[d] for d in axes)
        idx = np.ix_(*[range(i) for i in full_shape])
        self_idx = tuple(idx[axes.index(d)] for d in self.shape_dict)
        self_arr = self.data[self_idx]
        other_idx = tuple(idx[axes.index(d)] for d in other.shape_dict)
        other_arr = other.data[other_idx]
        return self_arr, other_arr

    def _update_attrs(self, updates: dict):
        if 'amount' in updates:
            updates['values'] = updates.pop('amount')
        return super()._update_attrs(updates)

    def _new_from_func(self, result):
        new = super()._new_from_func(result)
        if isinstance(new, arrays.Array):
            return Variable(new.values, self.unit, new.axes)
        return new

    def __str__(self) -> str:
        attrs = [
            f"axes=({', '.join(self.axes)})",
            f"shape={self.data.shape}",
            f"unit='{self.unit}'",
        ]
        return ', '.join(attrs)


class Measurement(collections.abc.Sequence, iterables.ReprStrMixin):
    """The result of measuring an object.

    While it is possible to directly instantiate this class, it serves primarily
    as the return type of `quantities.measure`, which accepts a much wider
    domain of arguments.
    """

    def __init__(self, arg) -> None:
        self.values, self.unit = self._get_attrs(arg)

    def _get_attrs(self, arg):
        """Extract initializing attributes from input, if possible."""
        if isinstance(arg, Measured):
            values = iterables.Separable(arg.amount)
            unit = arg.unit
            return list(values), unit
        try:
            values = getattr(arg, 'values', None) or getattr(arg, 'value')
            unit = getattr(arg, 'unit')
        except AttributeError:
            raise TypeError(arg)
        else:
            return list(iterables.Separable(values)), unit

    @property
    def asvector(self):
        """A new `Vector` object equivalent to this measurement."""
        return Vector(self.values, unit=self.unit)

    def __getitem__(self, index):
        """Called for index-based value access."""
        if index < 0:
            index += len(self)
        return Scalar(self.values[index], self.unit)

    def __len__(self) -> int:
        """The number of values. Called for len(self)."""
        return len(self.values)

    def __eq__(self, other) -> bool:
        """True if `other` has equivalent values and unit."""
        try:
            values, unit = self._get_attrs(other)
        except AttributeError:
            False
        else:
            return self.values == values and self.unit == unit

    def __str__(self) -> str:
        """A simplified representation of this object."""
        return f"{self.values} [{self.unit}]"


@runtime_checkable
class Measurable(Protocol):
    """Protocol defining a formally measurable object."""

    __slots__ = ()

    @abc.abstractmethod
    def __measure__(self) -> Measurement:
        """Create a measured object from input."""
        pass


class Unmeasurable(Exception):
    """Cannot measure this type of object."""

    def __init__(self, arg: object) -> None:
        self.arg = arg

    def __str__(self) -> str:
        return f"Cannot measure {self.arg!r}"


class MeasuringTypeError(TypeError):
    """A type-related error occurred while trying to measure this object."""


def measure(*args):
    """Create a measurement from a measurable object.

    This function will first check whether `args` is a single object that
    conforms to the `Measurable` protocol, and call a special method if so.
    Otherwise, it will attempt to parse `args` into one or more values and a
    corresponding unit.
    """
    if len(args) == 1 and isinstance(args[0], Measurable):
        return args[0].__measure__()
    parsed = parse_measurable(args, distribute=False)
    measured = Vector(parsed[:-1], unit=parsed[-1])
    return Measurement(measured)


def parse_measurable(args, distribute: bool=False):
    """Extract one or more values and an optional unit from `args`.
    
    See Also
    --------
    measure : returns the parsed object as a `Measurement`.
    """

    # Strip redundant lists and tuples.
    unwrapped = iterables.unwrap(args)

    # Raise an error for null input.
    if not unwrapped:
        raise Unmeasurable(unwrapped) from None

    # Handle a single numerical value:
    if isinstance(unwrapped, numbers.Number):
        result = (unwrapped, '1')
        return [result] if distribute else result

    # Count the number of distinct unit-like objects.
    types = [type(arg) for arg in unwrapped]
    n_units = sum(types.count(t) for t in (str, Unit))

    # Raise an error for multiple units.
    if n_units > 1:
        errmsg = "You may only specify one unit."
        raise MeasuringTypeError(errmsg) from None

    # TODO: The structure below suggests that there may be available
    # refactorings, though they may require first redefining or dismantling
    # `_callback_parse`.

    # Handle flat numerical iterables, like (1.1,) or (1.1, 2.3).
    if all(isinstance(arg, numbers.Number) for arg in unwrapped):
        return _wrap_measurable(unwrapped, '1', distribute)

    # Recursively handle an iterable of separable items.
    if all(isinstance(arg, iterables.Separable) for arg in unwrapped):
        return _callback_parse(unwrapped, distribute)

    # Ensure an explicit unit-like object
    unit = ensure_unit(unwrapped)

    # Handle flat iterables with a unit, like (1.1, 'm') or (1.1, 2.3, 'm').
    if all(isinstance(arg, numbers.Number) for arg in unwrapped[:-1]):
        return _wrap_measurable(unwrapped[:-1], unit, distribute)

    # Handle iterable values with a unit, like [(1.1, 2.3), 'm'].
    if isinstance(unwrapped[0], (list, tuple, range)):
        return _wrap_measurable(unwrapped[0], unit, distribute)


def _wrap_measurable(values, unit, distribute: bool):
    """Wrap a parsed measurable and return to caller."""
    if distribute:
        return list(iterables.distribute(values, unit))
    return (*values, unit)


def _callback_parse(unwrapped, distribute: bool):
    """Parse the measurable by calling back to `parse_measurable`."""
    if distribute:
        return [
            item
            for arg in unwrapped
            for item in parse_measurable(arg, distribute=True)
        ]
    parsed = [
        parse_measurable(arg, distribute=False) for arg in unwrapped
    ]
    units = [item[-1] for item in parsed]
    if any(unit != units[0] for unit in units):
        errmsg = "Can't combine measurements with different units."
        raise MeasuringTypeError(errmsg)
    values = [i for item in parsed for i in item[:-1]]
    unit = units[0]
    return (*values, unit)


def ensure_unit(args):
    """Extract the given unit or assume the quantity is unitless."""
    last = args[-1]
    implicit = not any(isinstance(arg, (str, Unit)) for arg in args)
    explicit = last in ['1', Unit('1')]
    if implicit or explicit:
        return '1'
    return str(last)


