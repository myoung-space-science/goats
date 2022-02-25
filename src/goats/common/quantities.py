import abc
import collections.abc
import functools
import math
import numbers
import operator
import typing

import numpy as np
import numpy.typing

from goats.common import algebra, aliased, iterables


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

_PREFIXES = iterables.Table(_prefixes)


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

_UNITS = iterables.Table(_units)


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

_BASE_QUANTITIES = iterables.Table(
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

_QUANTITIES = {
    'amount': {
        'dimensions': {
            'mks': 'N',
            'cgs': 'N',
        },
        'units': {
            'mks': 'mol',
            'cgs': 'mol',
        },
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
    },
    'momentum density': 'momentum / volume',
    'number': 'identity',
    'number density': '1 / volume',
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
    },
    'ratio': 'identity',
    'reluctance': {
        'dimensions': {
            'mks': '(I * T)^2 / (M * L^2)',
            'cgs': '1 / L',
        },
        'units': {
            'mks': 'A / Wb',
            'cgs': '1 / cm',
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
    },
    'volume': 'length^3',
    'vorticity': 'frequency',
    'work': 'energy',
}


# NOTE: Defined here to avoid a circular import with physical.py.
C = 2.99792458e10
"""The speed of light in cm/s."""
PI = np.pi
"""The ratio of a circle's circumference to its diameter."""


_CONVERSIONS = {
    ('F', 'cm'): C**2 * 1e-9,
    ('C', 'statC'): 10*C,
    ('e', 'C'): 1.6022e-19,
    ('S', 'cm / s'): C**2 * 1e-5,
    ('A', 'statA'): 10*C,
    # ('C / m^2', 'statC / m^2'): 4*PI * C * 1e-3,
    ('Gy', 'erg / g'): 1e4,
    ('J', 'erg'): 1e7,
    ('eV', 'J'): 1.6022e-19,
    ('N', 'dyn'): 1e5,
    ('ohm', 's / cm'): 1e5 / C**2,
    ('H', 's^2 / cm'): 1e5 / C**2,
    # ('m', 'cm'): 1e2,
    ('au', 'm'): 1.495978707e11,
    ('Wb', 'Mx'): 1e8,
    ('T', 'G'): 1e4,
    ('A / m', 'Oe'): 4*PI * 1e-3,
    # ('A * m^2', 'Oe * cm^3'): 1e-3,
    # ('kg', 'g'): 1e3,
    ('nuc', 'kg'): 1.6605e-27,
    ('amu', 'kg'): 1.6605e-27,
    # ('kg * m / s', 'g * cm / s'): 1e5,
    ('H / m', '1'): 1e7 / 4*PI,
    ('F / m', '1'): 36*PI * 1e9,
    ('rad', 'deg'): 180 / PI,
    ('V', 'statV'): 1e6 / C,
    ('W', 'erg / s'): 1e7,
    ('Pa', 'dyn / cm^2'): 1e1,
    ('Bq', 'Ci'): 1.0 / 3.7e10,
    ('A / Wb', '1 / cm'): 4*PI * 1e-9,
    ('s', 'min'): 1.0 / 60.0,
    ('s', 'h'): 1.0 / 3600.0,
    ('s', 'd'): 1.0 / 86400.0,
    ('Wb / m', 'G * cm'): 1e6,
    ('kg / (m * s)', 'P'): 1e1,
}


CONVERSIONS = iterables.Connections(_CONVERSIONS)


Instance = typing.TypeVar('Instance', bound='Property')


class Property(collections.abc.Mapping, iterables.ReprStrMixin):
    """All definitions of a single metric property."""

    _instances = {}
    _supported = (
        'dimensions',
        'units',
    )

    key: str=None

    def __new__(
        cls: typing.Type[Instance],
        arg: typing.Union[str, Instance],
    ) -> Instance:
        """Create a new instance or return an existing one.
        
        Parameters
        ----------
        arg : string or instance
            A string representing the metric property to create, or an existing
            instance of this class.
        """
        if isinstance(arg, cls):
            return arg
        key = str(arg)
        if key not in cls._supported:
            raise ValueError(f"Unsupported property: {key}") from None
        if available := cls._instances.get(key):
            return available
        self = super().__new__(cls)
        self.key = key
        return self

    LEN = len(_QUANTITIES) # No need to compute every time.
    def __len__(self) -> int:
        """The number of defined quantities. Called for len(self)."""
        return self.LEN

    def __iter__(self) -> typing.Iterator[str]:
        """Iterate over names of defined quantities. Called for iter(self)."""
        return iter(_QUANTITIES)

    def __getitem__(self, name: str):
        """Get a named property of a defined quantity.
        
        This method will search for `name` in the module-level collection of
        defined quantities. If it finds an `dict` entry, it will attempt to
        extract the values corresponding to this property's `key`. If it finds a
        `str` entry, it will attempt to create the equivalent `dict` by
        algebraically evaluating the terms in the entry.
        """
        if name not in _QUANTITIES:
            raise KeyError(f"No definition for '{name}'")
        q = _QUANTITIES[name]
        if isinstance(q, dict):
            return q.get(self.key, {})
        if not isinstance(q, str):
            raise TypeError(f"Expected {name} to be a string")
        # TODO: Cache computed results.
        return self._parse(q)

    def _parse(self, string: str):
        """Parse a string representing a compound quantity."""
        if ' ' in string and all(c not in string for c in ['*', '/']):
            string = string.replace(' ', '_')
        parts = [self._expand(term) for term in algebra.Expression(string)]
        keys = {key for part in parts for key in part.keys()}
        merged = {key: [] for key in keys}
        for part in parts:
            for key, value in part.items():
                merged[key].append(value)
        return {
            k: str(algebra.Expression(v))
            for k, v in merged.items()
        }

    _operand = algebra.OperandFactory()
    def _expand(self, term: algebra.Term):
        """Create a `dict` of operands from this term."""
        return {
            k: self._operand.create(v, term.exponent)
            for k, v in self[term.base.replace('_', ' ')].items()
        }

    def __str__(self) -> str:
        """A simplified representation of this object."""
        return self.key


class Metric(typing.NamedTuple):
    """A canonical physical quantity within a named metric system."""

    # NOTE: These types are not `Unit` and `Dimension` because we need
    # `Quantity` and `Metric` to define `Unit`.
    unit: str=None
    dimension: str=None


def build_unit_aliases(prefix, unit):
    """Define all aliases for the given metric prefix and base unit."""
    key = [f"{prefix[k]}{unit[k]}" for k in ['name', 'symbol']]
    if prefix['symbol'] == 'μ':
        key += [f"u{unit['symbol']}"]
    return tuple(key)


# Tables may not be necessary with this.
named_units = aliased.Mapping(
    {
        build_unit_aliases(prefix, unit): {'base': unit, 'prefix': prefix}
        for prefix in _PREFIXES for unit in _UNITS
    }
)


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


class MetricPrefix(typing.NamedTuple):
    """Metadata for a metric order-of-magnitude prefix."""

    symbol: str
    name: str
    factor: float


class BaseUnit(typing.NamedTuple):
    """Metadata for a named unit without metric prefix."""

    symbol: str
    name: str
    quantity: str
    system: str=None


Instance = typing.TypeVar('Instance', bound='NamedUnit')


class NamedUnit(iterables.ReprStrMixin):
    """A single named unit and corresponding metadata."""

    @typing.overload
    def __new__(
        cls: typing.Type[Instance],
        unit: str,
    ) -> Instance:
        """Create a new instance or return an existing one.
        
        Parameters
        ----------
        unit : string
            A string representing the metric unit to create.
        """

    @typing.overload
    def __new__(
        cls: typing.Type[Instance],
        instance: Instance,
    ) -> Instance:
        """Create a new instance or return an existing one.
        
        Parameters
        ----------
        instance : `~quantities.NamedUnit`
            An existing instance of this class.
        """

    _dimensions = Property('dimensions')

    _instances = {}

    prefix: MetricPrefix=None
    base: BaseUnit=None
    name: str=None
    """The full name of this unit."""
    symbol: str=None
    """The abbreviated symbol for this unit."""
    scale: float=None
    """The metric scale factor of this unit."""
    quantity: str=None
    """The physical quantity of this unit."""
    dimension: str=None
    """The physical dimension of this unit."""

    def __new__(cls, arg):
        """Concrete implementation."""
        if isinstance(arg, cls):
            return arg
        string = str(arg)
        magnitude, reference = cls.parse(string)
        key = (magnitude, reference)
        if available := cls._instances.get(key):
            return available
        self = super().__new__(cls)
        self.prefix = magnitude
        self.base = reference
        self.name = f"{magnitude.name}{reference.name}"
        self.symbol = f"{magnitude.symbol}{reference.symbol}"
        self.scale = magnitude.factor
        self.quantity = reference.quantity
        dimensions = cls._dimensions[self.quantity]
        system = self.base.system or 'mks'
        self.dimension = dimensions[system]
        cls._instances[key] = self
        return self

    @classmethod
    def parse(cls, string: str):
        """Determine the magnitude and reference of a unit.
        
        Parameters
        ----------
        string : str
            A string representing a metric unit.

        Returns
        -------
        tuple
            A 2-tuple in which the first element is a `~quantities.MetricPrefix`
            representing the order-of-magnitude of the given unit and the second
            element is a `~quantities.BaseUnit` representing the unscaled (i.e.,
            order-unity) metric unit.

        Examples
        --------
        >>> mag, ref = NamedUnit.parse('km')
        >>> mag
        MetricPrefix(symbol='k', name='kilo', factor=1000.0)
        >>> ref
        BaseUnit(symbol='m', name='meter', quantity='length', system='mks')
        """
        try:
            unit = named_units[string]
        except KeyError as err:
            raise UnitParsingError(string) from err
        magnitude = MetricPrefix(**unit['prefix'])
        reference = BaseUnit(**unit['base'])
        return magnitude, reference

    @classmethod
    def knows_about(cls, unit: str):
        """True if `unit` is a known named unit.
        
        This class method provides a self-consistent way to check if calling
        code can expect to create an instance of this class. It may provide an
        inexpensive alternative to `try...except` blocks.

        Parameters
        ----------
        unit : string
            The string to test as a possible instance.

        Returns
        -------
        bool
        """
        return unit in named_units

    @property
    def decomposed(self):
        """The representation of this unit in base-quantity units."""
        terms = algebra.Expression(self.dimension)
        quantities = [
            _BASE_QUANTITIES.find(term.base)[0]['name']
            for term in terms
        ]
        units = [
            _QUANTITIES[quantity]['units'][self.base.system]
            for quantity in quantities
        ]
        parsed = [self.parse(unit) for unit in units]
        return [
            algebra.Term(
                coefficient=part[0].factor*self.scale,
                base=part[1].symbol,
                exponent=term.exponent,
            ) for part, term in zip(parsed, terms)
        ]

    def __eq__(self, other) -> bool:
        """True if two representations have equal magnitude and base unit."""
        that = type(self)(other)
        same_magnitude = (self.prefix == that.prefix)
        same_reference = (self.base == that.base)
        return same_magnitude and same_reference

    def __hash__(self) -> int:
        """Called for hash(self). Supports use as mapping key."""
        return hash((self.base, self.prefix))

    def __floordiv__(self, other):
        """The magnitude of self relative to other.

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

            >>> NamedUnit('m') // 'cm'
            100.0
            >>> 'm' // NamedUnit('cm')
            100.0

        Attempting this operation between two units with different bases
        will raise an exception:

            >>> NamedUnit('m') // NamedUnit('au')
            <raises ValueError>
            >>> NamedUnit('m') // NamedUnit('s')
            <raises ValueError>

        Therefore, this operation is not intended for unit conversion.
        """
        if not isinstance(other, (str, NamedUnit)):
            return NotImplemented
        that = type(self)(other)
        if that == self:
            return 1.0
        if that.base != self.base:
            raise ValueError(
                f"Can't compute magnitude of {self!r} relative to {other!r}"
            ) from None
        return that.scale / self.scale

    def __rfloordiv__(self, other):
        """The magnitude of other relative to self."""
        return 1.0 / (self // other)

    def __str__(self) -> str:
        """A printable representation of this unit."""
        return f"'{self.name} | {self.symbol}'"



Instance = typing.TypeVar('Instance', bound='Conversion')


class Conversion(iterables.ReprStrMixin):
    """A single defined unit conversion."""

    @typing.overload
    def __new__(
        cls: typing.Type[Instance],
        u0: str,
        u1: str,
    ) -> Instance:
        """Create a new instance or return an existing one.
        
        Parameters
        ----------
        u0 : string
            The unit from which to convert.

        u1 : string
            The unit to which to convert.
        """

    @typing.overload
    def __new__(
        cls: typing.Type[Instance],
        instance: Instance,
    ) -> Instance:
        """Create a new instance or return an existing one.
        
        Parameters
        ----------
        instance
            An existing instance of this class.
        """

    _instances = {}

    u0: str=None
    u1: str=None
    _factor: float=None
    _path: list=None

    def __new__(cls, *args):
        """Concrete implementation."""
        if len(args) == 1 and isinstance(args[0], cls):
            return args[0]
        key = tuple(args)
        if available := cls._instances.get(key):
            return available
        u0, u1 = key
        self = super().__new__(cls)
        self.u0 = u0
        self.u1 = u1
        self._factor = None
        self._checked = None
        return self

    @property
    def inverse(self):
        """The conversion from `u1` to `u0`."""
        return type(self)(self.u1, self.u0)

    @property
    def factor(self):
        """The numerical scale associated with this conversion."""
        if self._factor is None:
            self._checked = []
            self._factor = self._convert(self.u0, self.u1)
        return self._factor

    def _convert(self, u0: str, u1: str, scale: float=1.0):
        """Convert `u0` to `u1`, if possible.
        
        This method will attempt to look up or compute the appropriate numerical
        conversion factor via various strategies, including searching for
        equivalent conversions or iteratively building a new conversion from
        existing conversions.
        """
        self._checked += [u0]
        if u0 == u1:
            return scale
        if found := self._search(u0, u1):
            return scale * found
        if ratio := self._compute_ratio(u0, u1):
            return scale * ratio
        if modified := self._compute_modified(u0, u1):
            return scale * modified
        conversions = CONVERSIONS.get_adjacencies(u0).items()
        for unit, weight in conversions:
            if unit not in self._checked:
                return weight * self._convert(unit, u1, scale=scale)
        raise UnitConversionError(self.u0, self.u1)

    available = NamedUnit.knows_about
    """Local copy of `~quantities.NamedUnit.knows_about`."""

    def _search(cls, u0: str, u1: str):
        """Search the defined conversions.
        
        This method will first search for a defined conversion from `u0` to
        `u1`. If it can't find one, it will search for a defined conversion from
        an alias of `u0` to an alias of `u1`. See `~_get_aliases_of` for more
        information.
        """
        if (u0, u1) in CONVERSIONS:
            return CONVERSIONS.get_weight(u0, u1)
        starts = cls._get_aliases_of(u0)
        ends = cls._get_aliases_of(u1)
        for ux in starts:
            for uy in ends:
                if (ux, uy) in CONVERSIONS:
                    return CONVERSIONS.get_weight(ux, uy)

    def _get_aliases_of(cls, unit: str):
        """Build a list of possible variations of this unit string.
        
        The aliases of `unit` comprise the given string, as well as the
        canonical name and symbol of the corresponding named unit, if one
        exists.
        """
        built = [unit]
        if cls.available(unit):
            known = NamedUnit(unit)
            built.extend([known.symbol, known.name])
        return built

    def _compute_ratio(self, u0: str, u1: str):
        """Return the relative magnitude of `u1` to `u0`, if possible."""
        try:
            scale = NamedUnit(u1) // NamedUnit(u0)
        except (ValueError, UnitParsingError):
            scale = None
        return scale

    def _compute_modified(self, u0: str, u1: str):
        """Compute the conversion after modifying `u0` or `u1`.
        
        Notes
        -----
        It is possible to have `u0` or `u1` individually in `CONVERSIONS.nodes`
        despite the fact that `(u0, u1)` is not in `CONVERSIONS`. For example,
        there are nodes for both 'min' (minute) and 'd' (day), each with a
        conversion to 's' (second), but there is no direct conversion from 'min'
        to 'd'.
        """
        units = CONVERSIONS.nodes
        if u0 not in units or u1 not in units:
            if u0 not in units:
                if rescaled := self._rescale(u0, u1):
                    return rescaled
            if u1 not in units:
                if rescaled := self._rescale(u1, u0):
                    return 1 / rescaled
            if expanded := self._expand(u0, u1):
                return expanded

    def _rescale(self, u0: str, u1: str):
        """Compute a new conversion after rescaling `u0`.
        
        This method will look for a unit, `ux`, in `~quantities.CONVERSIONS`
        that has the same base unit as `u0`. If it finds one, it will attempt to
        convert `ux` to `u1`, and finally multiply the result by the relative
        magnitude of `u0` to `ux`. In other words, it attempts to compute ``(u0
        // ux) * (ux -> u1)`` in place of ``(u0 -> u1)``.
        """
        if not self.available(u0):
            return
        n0 = NamedUnit(u0)
        for ux in CONVERSIONS.nodes:
            if self.available(ux):
                nx = NamedUnit(ux)
                if nx.base == n0.base:
                    return (nx // n0) * self._search(ux, u1)

    def _expand(self, u0: str, u1: str):
        """Convert complex unit expressions term-by-term."""
        e0 = algebra.Expression(u0)
        e1 = algebra.Expression(u1)
        if len(e0) != len(e1):
            # This will happen if `u0` and `u1` are not simply expressions of
            # the same units, with possibly different metric prefixes (e.g.,
            # `(kg / s -> g / ms)`). We eventually want to handle those more
            # complex conversions (e.g., `(kg * m^2 / s^2 -> erg)`).
            #
            # Consider using new `NamedUnit.decompose()`.
            raise ValueError(
                f"Can't parse pairs from {u0} / {u1}"
            ) from None
        if factor := self._inner_product(e0, e1):
            return factor
        if factor := self._outer_product(e0, e1):
            return factor

    def _inner_product(self, e0: algebra.Expression, e1: algebra.Expression):
        """Compute ratios of all pairs of terms, if possible.
        
        This method attempts to compute a pseudo-inner product of `e0` with `e1`
        by computing individual ratios of sequential pairs of terms taken from
        both expressions. Success requires that the expressions' terms are
        ordered such that each pair has a common base. If the expressions do not
        meet that criterion, this method will return `None` in order to give
        subsequent methods a chance.
        """
        done = [self._reduce_pair(*pair) for pair in zip(e0, e1)]
        if all(done):
            return functools.reduce(lambda x, y: x*y, done)

    def _outer_product(self, e0: algebra.Expression, e1: algebra.Expression):
        """Iteratively compute ratios of terms.
        
        This method attempts to compute a pseudo-outer product of `e0` with `e1`
        by iteratively comparing each term in `e0` to terms in `e1` until it
        finds a term with matching base.
        """
        l0 = list(e0)
        l1 = list(e1)
        factor = 1.0
        while l0:
            t0 = l0.pop()
            for t1 in l1:
                if value := self._reduce_pair(t0, t1):
                    factor *= value
                    l1.remove(t1)
        return factor

    def _reduce_pair(self, t0: algebra.Term, t1: algebra.Term):
        """Compute the ratio of units with a common exponent."""
        if t0.exponent == t1.exponent:
            return self._convert(t0.base, t1.base) ** t0.exponent

    def __bool__(self):
        """True if this conversion exists."""
        # NOTE: This may be expensive.
        try:
            factor = self.factor
        except:
            return False
        return bool(factor)

    def __str__(self) -> str:
        """A simplified representation of this object."""
        return f"({self.u0!r} -> {self.u1!r}): {self.factor!r}"


Instance = typing.TypeVar('Instance', bound='_Converter')


class _Converter:
    """Unit conversions for a known physical quantity."""

    @typing.overload
    def __new__(
        cls: typing.Type[Instance],
        unit: str,
        quantity: str,
    ) -> Instance:
        """Create a new instance or return an existing one.
        
        Parameters
        ----------
        unit : string
            The unit to be converted.

        quantity : string
            The physical quantity of `unit`.
        """

    @typing.overload
    def __new__(
        cls: typing.Type[Instance],
        instance: Instance,
    ) -> Instance:
        """Create a new instance or return an existing one.
        
        Parameters
        ----------
        instance
            An existing instance of this class.
        """

    _instances = {}

    _units = Property('units')

    unit: str=None
    quantity: str=None
    _substitutions: typing.Dict[str, str]=None
    _defined: typing.Dict[str, typing.Dict[str, float]]=None

    def __new__(cls, *args):
        """Concrete implementation."""
        if len(args) == 1 and isinstance(args[0], cls):
            return args[0]
        unit, quantity = (str(arg) for arg in args)
        if available := cls._instances.get(unit):
            return available
        self = super().__new__(cls)
        self._substitutions = self._units[quantity]
        self.unit = self._substitutions.get(unit) or unit
        self._defined = None
        return self

    @property
    def defined(self):
        """An expanded mapping of defined conversions."""
        if self._defined is None:
            conversions = {}
            for (u0, u1), factor in _CONVERSIONS.items():
                forward = (u0, u1, factor)
                reverse = (u1, u0, 1 / factor)
                for ux, uy, s in (forward, reverse):
                    if ux not in conversions:
                        conversions[ux] = {}
                    if uy not in conversions[ux]:
                        conversions[ux][uy] = s
            self._defined = conversions
        return self._defined

    def to(self, target: str):
        """Compute the conversion from `self.unit` to `target`.
        
        Parameters
        ----------
        target : string
            The unit or keyword to which to convert the current unit. See Notes
            for details on how this method handles `target`.

        Notes
        -----
        This method proceeds as follows:
        
        1. Substitute a known unit for `target`, if necessary.
        1. Return 1.0 if the target unit and the current unit are the same.
        1. Search for the forward or reverse conversion and, if found, return
           the corresponding numerical factor.
        1. Return the ratio of metric scale factors between the target unit and
           the current unit if they have the same base unit (e.g., 'm' for 'cm'
           and 'km'). This happens later in the process because it requires two
           conversions to `~quantities.NamedUnit` and an arithmetic operation,
           whereas previous cases only require a mapping look-up.
        1. Search for the forward or reverse conversion between base units and,
           if found, return the corresponding numerical factor after rescaling
           by the metric ratio.
        1. Raise an exception to alert the caller that the conversion is
           undefined.
        """
        unit = self._substitutions.get(target) or target
        if self.unit == unit:
            return 1.0
        if conversion := Conversion(self.unit, unit):
            return conversion.factor
        raise ValueError(
            f"Unknown conversion from {self.unit!r} to {unit!r}."
        ) from None


Instance = typing.TypeVar('Instance', bound='Quantity')


class Quantity(iterables.ReprStrMixin):
    """A single physical quantity."""

    _properties = {k: Property(k) for k in ('units', 'dimensions')}

    _instances = {}

    Attr = typing.TypeVar('Attr', bound=dict)
    Attr = typing.Mapping[str, typing.Dict[str, str]]

    name: str=None
    units: Attr=None
    dimensions: Attr=None

    def __new__(
        cls: typing.Type[Instance],
        arg: typing.Union[str, Instance],
    ) -> Instance:
        """Create a new instance or return an existing one.
        
        Parameters
        ----------
        arg : string or instance
            A string representing the physical quantity to create, or an
            existing instance of this class.
        """
        if isinstance(arg, cls):
            return arg
        name = str(arg).lower()
        if available := cls._instances.get(name):
            return available
        self = super().__new__(cls)
        self.name = name
        """The name of this physical quantity."""
        self.units = cls._properties['units'][self.name]
        self.dimensions = cls._properties['dimensions'][self.name]
        cls._instances[name] = self
        return self

    def __getitem__(self, system: str):
        """Get this quantity's representation in the named metric system."""
        try:
            name = system.lower()
            dimension = self.dimensions[name]
            unit = self.units[name]
        except KeyError as err:
            raise KeyError(
                f"No metric available for system '{system}'"
            ) from err
        else:
            return Metric(dimension=dimension, unit=unit)

    # NOTE: This is here because unit conversions are only defined within their
    # respective quantities, even though two quantities may have identical
    # conversions (e.g., frequency and vorticity).
    def convert(self, unit: str) -> _Converter:
        """Create a conversion object for `unit`."""
        return _Converter(unit, self.name)

    _attrs = ('dimensions', 'units')

    def __eq__(self, other) -> bool:
        """True if two quantities have equal attributes."""
        if isinstance(other, Quantity):
            try:
                equal = [
                    getattr(self, attr) == getattr(other, attr)
                    for attr in self._attrs
                ]
            except AttributeError:
                return False
            else:
                return all(equal)
        return NotImplemented

    def __str__(self) -> str:
        """A simplified representation of this object."""
        return self.name


Instance = typing.TypeVar('Instance', bound='Unit')


# TODO: Cache instances if they correspond to a defined quantity.
class Unit(algebra.Expression):
    """An algebraic expression representing a physical unit."""

    _dimension=None
    _quantity=None

    def __new__(
        cls: typing.Type[Instance],
        expression: typing.Union['Unit', str, iterables.whole],
        **kwargs
    ) -> Instance:
        """Create a new unit from `expression`."""
        self = super().__new__(cls, expression, **kwargs)
        self._dimension = None
        self._quantity = None
        return self

    @property
    def dimension(self):
        """The physical dimension of this unit."""
        if self._dimension is None:
            self._dimension = Dimension(self)
        return self._dimension

    @property
    def quantity(self):
        """The physical quantity of this unit, if defined."""
        if self._quantity is None:
            if quantity := self._get_quantity():
                self._quantity = Quantity(quantity)
        return self._quantity

    def _get_quantity(self):
        """Determine this unit's physical quantity, if possible."""
        this_unit = str(self)
        if NamedUnit.knows_about(this_unit):
            return NamedUnit(this_unit).quantity
        for name in _QUANTITIES:
            quantity = Quantity(name)
            if this_unit in quantity.units.values():
                return quantity

    def __floordiv__(self, other):
        """Compute the magnitude of this unit relative to another.

        This method essentially computes the amount of `self` per `other`. The
        result is the numerical factor, N, necessary to convert a quantity with
        unit `other` to the equivalent quantity with unit `self`. In algebraic
        terms, suppose you have an amount q0 of some physical quantity when
        expressed in unit u0. The equivalent amount when expressed in unit u1 is
        q1 = (u1 // u0) * q0.

        Examples
        --------
        Create two unit objects representing a meter and a centimeter, and
        compute their relative magnitude::

            >>> m = Unit('m')
            >>> cm = Unit('cm')
            >>> m // cm
            0.01
            >>> cm // m
            100.0

        As with `~quantities.NamedUnit`, these results are equivalent to the
        statement that there are 100 centimeters in a meter. However, this class
        also supports more complex unit expressions, and can therefore compute
        more complex ratios::

            >>> Unit('kg * m^2 / s^2') // Unit('g * m^2 / s^2')
            0.001
            >>> Unit('kg * m^2 / s^2') // Unit('g * mm^2 / s^2')
            1e-09
            >>> Unit('kg * m^2 / s^2') // Unit('g * m^2 / day^2')
            1.3395919067215364e-13
            >>> Unit('kg * m^2 / s^2') // Unit('g * au^2 / day^2')
            2997942777.7207007
        """
        if not isinstance(other, (str, Unit)):
            return NotImplemented
        return Conversion(str(other), str(self)).factor

    def __rfloordiv__(self, other):
        """Compute the inverse of self // other."""
        return 1.0 / self.__floordiv__(other)

    def to(self, system: str):
        """Convert this unit to the named system, if possible."""
        if self.quantity:
            return self.quantity.convert(str(self)).to(system)


Instance = typing.TypeVar('Instance', bound='Dimension')


class Dimension(algebra.Expression):
    """An algebraic expression representing a physical dimension.
    
    This class exists to support the `dimension` property of `~quantities.Unit`.
    It is essentially a thin wrapper around `~algebra.Expression` with logic to
    compute the dimension of a `~quantities.Unit` instance or equivalent object.
    """

    def __new__(
        cls: typing.Type[Instance],
        unit: typing.Union[Unit, str, iterables.whole],
    ) -> Instance:
        expression = algebra.Expression(unit)
        terms = [
            algebra.Term(
                base=NamedUnit(term.base).dimension,
                exponent=term.exponent,
            ) for term in expression
        ]
        return super().__new__(cls, terms)


class MetricKeyError(KeyError):
    """Metric-system mapping-key error."""
    pass


class MetricTypeError(TypeError):
    """Metric-system argument-type error."""
    pass


class MetricSearchError(KeyError):
    """Error while searching for a requested metric."""
    pass


# TODO: This class will need significant updates after changes to `Unit`,
# `Quantity`, and `Metric`. Its main role is exposing `get_unit` for
# algorithmically updating units on variables (cf. `eprem.datasets`).
class MetricSystem(collections.abc.Mapping, iterables.ReprStrMixin):
    """Representations of physical quantities within a given metric system."""

    _instances = {}
    name: str=None
    dimensions: typing.Dict[str, str]=None

    def __new__(cls, arg: typing.Union[str, 'MetricSystem']):
        """Return an existing instance or create a new one.

        Parameters
        ----------
        arg : str or `~quantities.MetricSystem`
            The name of the metric system to represent (e.g., 'mks') or an
            existing instance. Names are case sensitive. Instances are
            singletons.
        """
        if isinstance(arg, cls):
            return arg
        name = arg.lower()
        if instance := cls._instances.get(name):
            return instance
        new = super().__new__(cls)
        new.name = name
        new.dimensions = Property('dimensions')
        cls._instances[name] = new
        return new

    def __len__(self) -> int:
        """The number of quantities defined in this metric system."""
        return _QUANTITIES.__len__()

    def __iter__(self) -> typing.Iterator:
        """Iterate over defined metric quantities."""
        return _QUANTITIES.__iter__()

    def __getitem__(self, key: str):
        """Get the metric for the requested quantity in this system."""
        try:
            quantity = Quantity(key)
        except ValueError as err:
            raise MetricKeyError(
                f"No known quantity called '{key}'"
            ) from err
        else:
            return quantity[self.name]

    def get_unit(
        self,
        unit: typing.Union[str, Unit]=None,
        dimension: typing.Union[str, Dimension]=None,
        quantity: typing.Union[str, Quantity]=None,
    ) -> typing.Optional[Unit]:
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

    T = typing.TypeVar('T', Unit, Dimension, Quantity)
    T = typing.Union[Unit, Dimension, Quantity]

    def _get_unit(
        self,
        methods: typing.Dict[str, typing.Callable[[T], Unit]],
        targets: typing.Dict[str, T],
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

    def _format_targets(self, targets: typing.Dict[str, T]):
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
        return self[str(quantity)].unit

    def __eq__(self, other) -> bool:
        """True if two systems have the same `name` attribute."""
        if isinstance(other, MetricSystem):
            return other.name == self.name
        return NotImplemented

    def __bool__(self) -> bool:
        """A defined metric system is always truthy."""
        return True

    def keys(self) -> typing.KeysView[str]:
        return super().keys()

    def values(self) -> typing.ValuesView[Metric]:
        return super().values()

    def items(self) -> typing.ItemsView[str, Metric]:
        return super().items()

    def __str__(self) -> str:
        """A simplified representation of this object."""
        return str(self.name)


class ComparisonError(TypeError):
    """Incomparable instances of the same type."""

    def __init__(self, __this: typing.Any, __that: typing.Any, name: str):
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
        allowed: typing.Iterable[typing.Type]=None,
    ) -> None:
        self.names = names
        self.allowed = iterables.whole(allowed)

    def __call__(self, func: typing.Callable) -> typing.Callable:
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

    def _comparable(
        self,
        this: typing.Any,
        that: typing.Any,
        name: str,
    ) -> bool:
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
RealValued.register(np.ndarray)


Instance = typing.TypeVar('Instance', bound='Quantified')


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
        `quantity` attribute). Subclasses may individually customize this
        behavior via the keyword arguments described below

        Parameters
        ----------
        allowed : mapping from string to type
            A mapping (e.g., `dict`) from method name to a type or iterable of
            types in addition to instances of the decorated class that the named
            method should accept. The `same` class will not check objects of
            these additional types for sameness. For example, to indicate that a
            subclass accepts integers to its addition and subtraction methods,
            pass `allowed={'__add__': int, '__sub__': int}` to the class
            constructor.
        """
        allowed = kwargs.get('allowed', {})
        for method in cls._quantified:
            if current:= getattr(cls, method, None):
                update = same('quantity', allowed=allowed.get(method))
                updated = update(current)
                setattr(cls, method, updated)

    @typing.overload
    def __new__(
        cls: typing.Type[Instance],
        amount: typing.Any,
        quantity: typing.Any,
    ) -> Instance: ...

    @typing.overload
    def __new__(
        cls: typing.Type[Instance],
        instance: Instance,
    ) -> Instance: ...

    _amount: typing.Any=None
    _quantity: typing.Any=None

    def __new__(cls, *args):
        """Create a new instance of `cls`."""
        self = super().__new__(cls)
        if len(args) == 1 and isinstance(args[0], cls):
            instance = args[0]
            self._amount = instance._amount
            self._quantity = instance._quantity
            return self
        if len(args) == 2:
            self._amount, self._quantity = args
            return self
        raise TypeError(
            f"Can't instantiate {cls} from {args}"
        ) from None

    @property
    def quantity(self):
        """The type of thing that this object represents."""
        return self._quantity

    def __str__(self) -> str:
        """A simplified representation of this object."""
        return f"{self._amount} {self._quantity}"


Instance = typing.TypeVar('Instance', bound='Ordered')


class Ordered(Quantified):
    """A quantified object that supports comparisons

    An ordered object has an amount and a quantity. The amount must be formally
    comparable -- that is, a comparison to another amount using one of the six
    binary relations (i.e., <, >, <=, >=, ==, !=) must produce well-defined
    results. The quantity may be anything that supports equality comparison,
    which should be true unless the object explicitly disables `__eq__`.
    """

    @typing.overload
    def __new__(
        cls: typing.Type[Instance],
        amount: Comparable,
        quantity: typing.Any,
    ) -> Instance: ...

    @typing.overload
    def __new__(
        cls: typing.Type[Instance],
        instance: Instance,
    ) -> Instance: ...

    def __new__(cls, *args):
        return super().__new__(cls, *args)

    _amount: Comparable=None

    def __lt__(self, other: 'Ordered') -> bool:
        return self._amount < other._amount

    def __le__(self, other: 'Ordered') -> bool:
        return self._amount <= other._amount

    def __gt__(self, other: 'Ordered') -> bool:
        return self._amount > other._amount

    def __ge__(self, other: 'Ordered') -> bool:
        return self._amount >= other._amount

    def __eq__(self, other: 'Ordered') -> bool:
        return self._amount == other._amount

    def __ne__(self, other) -> bool:
        """True if self != other.

        Explicitly defined with respect to `self.__eq__` to promote consistency
        in subclasses that overload `__eq__`.
        """
        return not self.__eq__(other)

    def __str__(self) -> str:
        """A simplified representation of this object."""
        return f"{self._amount} {self._quantity}"


Instance = typing.TypeVar('Instance', bound='Measured')


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

    @typing.overload
    def __new__(
        cls: typing.Type[Instance],
        amount: RealValued,
    ) -> Instance:
        """Create a new measured object.
        
        Parameters
        ----------
        amount : real-valued
            The measured amount.
        """

    @typing.overload
    def __new__(
        cls: typing.Type[Instance],
        amount: RealValued,
        unit: typing.Union[str, Unit],
    ) -> Instance:
        """Create a new measured object.
        
        Parameters
        ----------
        amount : real-valued
            The measured amount.

        unit : string or `~quantities.Unit`
            The unit in which `amount` is measured.
        """

    @typing.overload
    def __new__(
        cls: typing.Type[Instance],
        instance: Instance,
    ) -> Instance:
        """Create a new measured object.
        
        Parameters
        ----------
        instance : `~quantities.Measured`
            An existing instance of this class.
        """

    _amount: RealValued=None
    _unit: Unit=None

    def __new__(cls, *args, **kwargs):
        """The concrete implementation of `~quantities.Measured.__new__`.
        
        Notes
        -----
        This method first extracts a local `unit` in order to pass it as a `str`
        to its parent class's `_quantity` attribute while returning it as a
        `~quantities.Unit` for initializing the `_unit` attribute of the current
        instance.
        """
        if not kwargs and len(args) == 1 and isinstance(args[0], cls):
            instance = args[0]
            amount = instance._amount
            unit = instance.unit()
        else:
            attrs = list(args)
            attr_dict = {
                k: attrs.pop(0) if attrs
                else kwargs.pop(k, None)
                for k in ('amount', 'unit')
            }
            amount = attr_dict['amount']
            unit = Unit(attr_dict['unit'] or '1')
        self = super().__new__(cls, amount, str(unit))
        self._unit = unit
        return self

    @classmethod
    def _new(cls: typing.Type[Instance], *args, **kwargs) -> Instance:
        """Create a new instance from updated attributes."""
        return cls(*args, **kwargs)

    @typing.overload
    def unit(self: Instance) -> Unit:
        """Get this object's unit of measurement.
        
        Parameters
        ----------
        None

        Returns
        -------
        `~quantities.Unit`
            The current unit of `amount`.
        """

    @typing.overload
    def unit(
        self: Instance,
        new: typing.Union[str, Unit],
    ) -> Instance:
        """Update this object's unit of measurement.

        Parameters
        ----------
        new : string or `~quantities.Unit`
            The new unit in which to measure `amount`.

        Returns
        -------
        Subclass of `~quantities.Measured`
            A new instance of this class.
        """

    def unit(self, new=None):
        """Concrete implementation."""
        if not new:
            return self._unit
        scale = Unit(new) // self._unit
        amount = (scale * self)._amount
        return self._new(amount=amount, unit=new)

    def __bool__(self) -> bool:
        return bool(self._amount)

    def __abs__(self):
        return self._new(
            amount=abs(self._amount),
            unit=self._unit,
        )

    def __neg__(self):
        return self._new(
            amount=-self._amount,
            unit=self._unit,
        )

    def __pos__(self):
        return self._new(
            amount=+self._amount,
            unit=self._unit,
        )

    def __add__(self, other: 'Measured'):
        return self._new(
            amount=self._amount + other._amount,
            unit=self._unit,
        )

    def __radd__(self, other: typing.Any):
        return NotImplemented

    def __sub__(self, other: 'Measured'):
        return self._new(
            amount=self._amount - other._amount,
            unit=self._unit,
        )

    def __rsub__(self, other: typing.Any):
        return NotImplemented

    def __mul__(self, other: typing.Any):
        if isinstance(other, Measured):
            amount = self._amount * other._amount
            unit = self._unit * other._unit
            return self._new(
                amount=amount,
                unit=unit,
            )
        if isinstance(other, numbers.Number):
            return self._new(
                amount=self._amount * other,
                unit=self._unit,
            )
        return NotImplemented

    def __rmul__(self, other: typing.Any):
        if isinstance(other, numbers.Number):
            return self._new(
                amount=other * self._amount,
                unit=self._unit,
            )
        return NotImplemented

    def __truediv__(self, other: typing.Any):
        if isinstance(other, Measured):
            amount = self._amount / other._amount
            unit = self._unit / other._unit
            return self._new(
                amount=amount,
                unit=unit,
            )
        if isinstance(other, numbers.Number):
            return self._new(
                amount=self._amount / other,
                unit=self._unit,
            )
        return NotImplemented

    def __rtruediv__(self, other: typing.Any):
        return NotImplemented

    def __pow__(self, other: typing.Any):
        if isinstance(other, numbers.Number):
            return self._new(
                amount=self._amount ** other,
                unit=self._unit ** other
            )
        return NotImplemented

    def __rpow__(self, other: typing.Any):
        return NotImplemented

    def __str__(self) -> str:
        return f"{self._amount} [{self._unit}]"


Instance = typing.TypeVar('Instance', bound='Measured')


VT = typing.TypeVar('VT', bound=RealValued)
VT = RealValued


allowed = {m: numbers.Real for m in ['__lt__', '__le__', '__gt__', '__ge__']}
class Scalar(Measured, allowed=allowed):
    """A single numerical value and associated unit."""

    @typing.overload
    def __new__(
        cls: typing.Type[Instance],
        value: VT,
    ) -> Instance:
        """Create a new scalar object.
        
        Parameters
        ----------
        value : real number
            The numerical value of this scalar. The argument must implement the
            `~numbers.Real` interface.
        """

    @typing.overload
    def __new__(
        cls: typing.Type[Instance],
        value: VT,
        unit: typing.Union[str, Unit],
    ) -> Instance:
        """Create a new scalar object.
        
        Parameters
        ----------
        value : real number
            The numerical value of this scalar. The argument must implement the
            `~numbers.Real` interface.

        unit : string or  `~quantities.Unit`
            The metric unit of `value`.
        """

    @typing.overload
    def __new__(
        cls: typing.Type[Instance],
        instance: Instance,
    ) -> Instance:
        """Create a new scalar object.
        
        Parameters
        ----------
        instance : `~quantities.Scalar`
            An existing instance of this class.
        """

    _value: VT

    def __new__(cls, *args, **kwargs):
        """The concrete implementation of `~quantities.Scalar.__new__`."""
        if not kwargs and len(args) == 1 and isinstance(args[0], cls):
            instance = args[0]
            value = instance._value
            unit = instance.unit()
        else:
            attrs = list(args)
            if 'amount' in kwargs:
                kwargs['value'] = kwargs.pop('amount')
            attr_dict = {
                k: attrs.pop(0) if attrs
                else kwargs.pop(k, None)
                for k in ('value', 'unit')
            }
            value = attr_dict['value']
            unit = attr_dict['unit']
        self = super().__new__(cls, value, unit=unit)
        self._value = value
        return self

    def __lt__(self, other: Ordered) -> bool:
        if isinstance(other, Comparable):
            return self._amount < other
        return super().__lt__(other)

    def __le__(self, other: Ordered) -> bool:
        if isinstance(other, Comparable):
            return self._amount <= other
        return super().__le__(other)

    def __gt__(self, other: Ordered) -> bool:
        if isinstance(other, Comparable):
            return self._amount > other
        return super().__gt__(other)

    def __ge__(self, other: Ordered) -> bool:
        if isinstance(other, Comparable):
            return self._amount >= other
        return super().__ge__(other)

    def __float__(self):
        """Called for float(self)."""
        return float(self._value)

    def __int__(self):
        """Called for int(self)."""
        return int(self._value)

    def __round__(self, ndigits: int=None):
        """Called for round(self)."""
        return round(self._value, ndigits=ndigits)

    def __floor__(self):
        """Called for math.floor(self)."""
        return math.floor(self._value)

    def __ceil__(self):
        """Called for math.ceil(self)."""
        return math.ceil(self._value)

    def __trunc__(self):
        """Called for math.trunc(self)."""
        return math.trunc(self._value)

    def __mul__(self, other: typing.Any):
        if isinstance(other, Variable):
            return NotImplemented
        return super().__mul__(other)

    def __truediv__(self, other: typing.Any):
        if isinstance(other, Variable):
            return NotImplemented
        return super().__truediv__(other)

    # NOTE: This class is immutable, so in-place operations defer to forward
    # operations. That automatically happens for `__iadd__`, `__isub__`,
    # `__imul__`, and `__itruediv__`, but not for `__ipow__`, so we need to
    # explicitly define a trivial implementation here.
    def __ipow__(self, other: typing.Any):
        return super().__pow__(other)

    def __hash__(self):
        """Called for hash(self)."""
        return hash((self._value, str(self._unit)))


Instance = typing.TypeVar('Instance', bound='Measured')


class Vector(Measured):
    """Multiple numerical values and their associated unit."""

    _values: typing.Iterable[VT]

    @typing.overload
    def __new__(
        cls: typing.Type[Instance],
        values: typing.Iterable[VT],
    ) -> Instance:
        """Create a new vector object.
        
        Parameters
        ----------
        values : iterable of real numbers
            The numerical values of this vector. Each member must implement the
            `~numbers.Real` interface.
        """

    @typing.overload
    def __new__(
        cls: typing.Type[Instance],
        values: typing.Iterable[VT],
        unit: typing.Union[str, Unit],
    ) -> Instance:
        """Create a new vector object.
        
        Parameters
        ----------
        values : iterable of real numbers
            The numerical values of this vector. Each member must implement the
            `~numbers.Real` interface.

        unit : string or `~quantities.Unit`
            The metric unit of `values`.
        """

    @typing.overload
    def __new__(
        cls: typing.Type[Instance],
        instance: Instance,
    ) -> Instance:
        """Create a new vector object.

        Parameters
        ----------
        instance : `~quantities.Vector`
            An existing instance of this class.
        """

    def __new__(cls, *args, **kwargs):
        """The concrete implementation of `~quantities.Vector.__new__`."""
        if not kwargs and len(args) == 1 and isinstance(args[0], cls):
            instance = args[0]
            values = instance._values
            unit = instance.unit()
        else:
            attrs = list(args)
            if 'amount' in kwargs:
                kwargs['values'] = kwargs.pop('amount')
            attr_dict = {
                k: attrs.pop(0) if attrs
                else kwargs.pop(k, None)
                for k in ('values', 'unit')
            }
            values = attr_dict['values']
            unit = attr_dict['unit']
        self = super().__new__(cls, values, unit=unit)
        self._values = list(iterables.whole(self._amount))
        return self

    def __len__(self):
        """Called for len(self)."""
        return len(self._values)

    def __iter__(self):
        """Called for iter(self)."""
        return iter(self._values)

    def __contains__(self, item):
        """Called for item in self."""
        return item in self._values

    def __add__(self, other: typing.Any):
        if isinstance(other, Vector):
            values = [s + o for s, o in zip(self._values, other._values)]
            return self._new(amount=values, unit=self._unit)
        if isinstance(other, Measured):
            values = [s + other for s in self._values]
            return self._new(amount=values, unit=self._unit)
        return NotImplemented

    def __sub__(self, other: typing.Any):
        if isinstance(other, Vector):
            values = [s - o for s, o in zip(self._values, other._values)]
            return self._new(amount=values, unit=self._unit)
        if isinstance(other, Measured):
            values = [s - other for s in self._values]
            return self._new(amount=values, unit=self._unit)
        return NotImplemented

    def __mul__(self, other: typing.Any):
        if isinstance(other, Vector):
            values = [s * o for s, o in zip(self._values, other._values)]
            unit = self._unit * other._unit
            return self._new(amount=values, unit=unit)
        if isinstance(other, Scalar):
            other = float(other)
        if isinstance(other, RealValued):
            values = [s * other for s in self._values]
            return self._new(amount=values, unit=self._unit)
        return NotImplemented

    def __rmul__(self, other: typing.Any):
        if isinstance(other, Scalar):
            other = float(other)
        if isinstance(other, numbers.Number):
            values = [other * s for s in self._values]
            return self._new(amount=values, unit=self._unit)
        return NotImplemented

    def __truediv__(self, other: typing.Any):
        if isinstance(other, Vector):
            values = [s / o for s, o in zip(self._values, other._values)]
            unit = self._unit / other._unit
            return self._new(amount=values, unit=unit)
        if isinstance(other, Scalar):
            other = float(other)
        if isinstance(other, RealValued):
            values = [s / other for s in self._values]
            return self._new(amount=values, unit=self._unit)
        return NotImplemented

    def __pow__(self, other: typing.Any):
        if isinstance(other, numbers.Number):
            values = [s ** other for s in self._values]
            unit = self._unit ** other
            return self._new(amount=values, unit=unit)
        return NotImplemented


IndexLike = typing.TypeVar(
    'IndexLike',
    typing.Iterable[int],
    slice,
    type(Ellipsis),
)
IndexLike = typing.Union[typing.Iterable[int], slice, type(Ellipsis)]

UnitLike = typing.TypeVar('UnitLike', str, Unit)
UnitLike = typing.Union[str, Unit]


allowed = {'__add__': float, '__sub__': float}
class Variable(Measured, np.lib.mixins.NDArrayOperatorsMixin, allowed=allowed):
    """A measured object with data stored in a numerical array.

    The result of binary arithmetic operations on instances of this class are
    similar to those of `Vector`, but differ in the following ways:
    1. Multiplication (`*`) and division (`/`) accept operands with different
       axes, as long as any repeated axes have the same length in both operands.
       The result will contain all unique axes from its operands.
    2. Addition (`+`) and subtraction (`-`) accept real numbers as right-sided
       operands. The result is a new instance with the operation applied to the
       underlying array.
    """

    @typing.overload
    def __new__(
        cls: typing.Type[Instance],
        data: typing.Iterable[numbers.Number],
        unit: typing.Union[str, Unit],
        axes: typing.Iterable[str],
    ) -> Instance:
        """Create a new variable object.
        
        Parameters
        ----------
        data : array-like
            The numerical data of this variable.

        unit : string or `~quantities.Unit`
            The metric unit of `data`.

        axes : iterable of strings
            The names of this variable's indexable axes.
        """

    @typing.overload
    def __new__(
        cls: typing.Type[Instance],
        data: typing.Iterable[numbers.Number],
        unit: typing.Union[str, Unit],
        axes: typing.Iterable[str],
        name: str='<anonymous>',
    ) -> Instance:
        """Create a new variable object.
        
        Parameters
        ----------
        data : array-like
            The numerical data of this variable.

        unit : string or `~quantities.Unit`
            The metric unit of `data`.

        axes : iterable of strings
            The names of this variable's indexable axes.

        name : string, default='<anonymous>'
            The optional name of this variable.
        """

    @typing.overload
    def __new__(
        cls: typing.Type[Instance],
        instance: Instance,
    ) -> Instance:
        """Create a new variable object.
        
        Parameters
        ----------
        instance : `~quantities.Variable`
            An existing instance of this class.
        """

    _amount: numpy.typing.ArrayLike
    axes: typing.Tuple[str]=None
    naxes: int=None
    name: str=None
    _scale: float=None
    _array: np.ndarray=None

    def __new__(cls, *args, **kwargs):
        """The concrete implementation of `~quantities.Variable.__new__`."""
        if not kwargs and len(args) == 1 and isinstance(args[0], cls):
            instance = args[0]
            data = instance._amount
            unit = instance.unit()
            axes = instance.axes
            name = instance.name
            scale = instance._scale
        else:
            if 'amount' in kwargs:
                kwargs['data'] = kwargs.pop('amount')
            attrs = list(args)
            attr_dict = {
                k: attrs.pop(0) if attrs
                else kwargs.pop(k, None)
                for k in ('data', 'unit', 'axes', 'name')
            }
            data = attr_dict['data']
            unit = attr_dict['unit']
            axes = attr_dict['axes'] or ()
            name = attr_dict['name'] or '<anonymous>'
            scale = kwargs.get('scale') or 1.0
        self = super().__new__(cls, data, unit=unit)
        self.axes = tuple(axes)
        """The names of indexable axes in this variable's array."""
        self.naxes = len(axes)
        """The number of indexable axes in this variable's array."""
        self.name = name
        """The name of this variable, if available."""
        self._scale = scale
        self._array = None
        return self

    @typing.overload
    def unit(self: Instance) -> Unit:
        """Get this object's unit of measurement.
        
        Parameters
        ----------
        None

        Returns
        -------
        `~quantities.Unit`
            The current unit of `amount`.
        """

    @typing.overload
    def unit(
        self: Instance,
        new: typing.Union[str, Unit],
    ) -> Instance:
        """Update this object's unit of measurement.

        Parameters
        ----------
        new : string or `~quantities.Unit`
            The new unit in which to measure `amount`.

        Returns
        -------
        Subclass of `~quantities.Measured`
            A new instance of this class.
        """

    def unit(self, new=None):
        """Concrete implementation."""
        if not new:
            return self._unit
        scale = (Unit(new) // self._unit) * self._scale
        return self._new(
            data=self._amount,
            unit=new,
            axes=self.axes,
            name=self.name,
            scale=scale,
        )

    def __len__(self):
        """Called for len(self)."""
        return self._get_data('size')

    def __iter__(self):
        """Called for iter(self)."""
        if method := self._get_data('__iter__'):
            return method()
        return iter(self._get_data())

    def __contains__(self, item):
        """Called for `item` in self."""
        return item in self._amount or item in self._get_data()

    _builtin = (int, slice, type(...))

    def __getitem__(self, *args: IndexLike):
        """Create a new instance from a subset of data."""
        unwrapped = iterables.unwrap(args)
        if self._types_match(unwrapped, self._builtin):
            return self._subscript_standard(unwrapped)
        return self._subscript_custom(unwrapped)

    def _types_match(self, args, types):
        """True if `args` is one `types` or a collection of `types`."""
        return (
            isinstance(args, types)
            or all(isinstance(arg, types) for arg in args)
        )

    def _subscript_standard(self, indices):
        """Perform standard array subscription.

        This method handles cases involving slices, an ellipsis, or integers,
        including v[:], v[...], v[i, :], v[:, j], and v[i, j], where i and j are
        integers.
        """
        result = self._get_data(indices)
        if isinstance(result, numbers.Number):
            return Scalar(result, unit=self.unit())
        return self._new(
            data=result,
            unit=self.unit(),
            axes=self.axes,
            name=self.name,
        )

    def _subscript_custom(self, args):
        """Perform array subscription specific to this object.

        This method handles all cases that don't meet the criteria for
        `_subscript_standard`.
        """
        if not isinstance(args, (tuple, list)):
            args = [args]
        expanded = self._expand_ellipsis(args)
        shape = self._get_data('shape')
        idx = [
            range(shape[i])
            if isinstance(arg, slice) else arg
            for i, arg in enumerate(expanded)
        ]
        indices = np.ix_(*list(idx))
        return self._new(
            data=self._get_data(indices),
            unit=self.unit(),
            axes=self.axes,
            name=self.name,
        )

    def _expand_ellipsis(
        self,
        user: typing.Sequence,
    ) -> typing.Tuple[slice, ...]:
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

    _HANDLED_TYPES = (np.ndarray, numbers.Number, list)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Provide support for `numpy` universal functions.

        See https://numpy.org/doc/stable/reference/arrays.classes.html for more
        information on use of this special method.

        See
        https://numpy.org/doc/stable/reference/generated/numpy.lib.mixins.NDArrayOperatorsMixin.html#numpy.lib.mixins.NDArrayOperatorsMixin
        for the specific implementation example that this class follows.

        This method first ensures that the input types (as well as the type of
        `out`, if given) are supported types.
        """
        out = kwargs.get('out', ())
        for x in inputs + out:
            if not isinstance(x, self._HANDLED_TYPES + (type(self),)):
                return NotImplemented
        inputs = tuple(
            x._get_data() if isinstance(x, type(self))
            else x for x in inputs
        )
        if out:
            kwargs['out'] = tuple(
                x._get_data() if isinstance(x, type(self))
                else x for x in out
            )
        result = getattr(ufunc, method)(*inputs, **kwargs)
        if type(result) is tuple:
            return tuple(self._new_from_func(x) for x in result)
        if method == 'at':
            return None
        return self._new_from_func(result)

    _HANDLED_FUNCTIONS = {}

    def __array_function__(self, func, types, args, kwargs):
        """Provide support for functions in the `numpy` public API.

        See https://numpy.org/doc/stable/reference/arrays.classes.html for more
        information of use of this special method. The implementation shown here
        is a combination of the example on that page and code from the
        definition of `EncapsulateNDArray.__array_function__` in
        https://github.com/dask/dask/blob/main/dask/array/tests/test_dispatch.py

        The initial `issubclass` check allows subclasses that don't override
        `__array_function__` to handle objects of this type.
        """
        accepted = (type(self), np.ndarray, np.ScalarType)
        if not all(issubclass(ti, accepted) for ti in types):
            return NotImplemented
        if func in self._HANDLED_FUNCTIONS:
            arr = self._HANDLED_FUNCTIONS[func](*args, **kwargs)
            return self._new_from_func(arr)
        args = tuple(
            arg._get_data() if isinstance(arg, type(self))
            else arg for arg in args
        )
        types = tuple(
            ti for ti in types
            if not issubclass(ti, type(self))
        )
        data = self._get_data()
        arr = data.__array_function__(func, types, args, kwargs)
        return self._new_from_func(arr)

    def _new_from_func(self, result):
        """Create a new instance from the result of a `numpy` function."""
        if not isinstance(result, np.ndarray):
            return result
        return self._new(
            data=result,
            unit=self.unit(),
            axes=self.axes,
            name=self.name,
        )

    def __eq__(self, other: typing.Any):
        """True if two instances have the same data and attributes."""
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

    def __add__(self, other: typing.Any):
        if self._add_sub_okay(other):
            data = self._get_data().__add__(other)
            return self._new(
                data=data,
                unit=self.unit(),
                axes=self.axes,
                name=self.name,
            )
        return NotImplemented

    def __sub__(self, other: typing.Any):
        if self._add_sub_okay(other):
            data = self._get_data().__sub__(other)
            return self._new(
                data=data,
                unit=self.unit(),
                axes=self.axes,
                name=self.name,
            )
        return NotImplemented

    def _add_sub_okay(self, other):
        if isinstance(other, numbers.Real):
            return True
        if isinstance(other, Variable) and self.axes == other.axes:
            return True
        return False

    def __mul__(self, other: typing.Any):
        if isinstance(other, Variable):
            axes = sorted(tuple(set(self.axes + other.axes)))
            sarr, oarr = self._extend_arrays(other, axes)
            data = sarr * oarr
            unit = self.unit() * other.unit()
            name = f"{self.name} * {other.name}"
            return self._new(
                data=data,
                unit=unit,
                axes=axes,
                name=name,
            )
        data = self._get_data().__mul__(other)
        return self._new(
            data=data,
            unit=self.unit(),
            axes=self.axes,
            name=self.name,
        )

    def __rmul__(self, other: typing.Any):
        data = self._get_data().__rmul__(other)
        return self._new(
            data=data,
            unit=self.unit(),
            axes=self.axes,
            name=self.name,
        )

    def __truediv__(self, other: typing.Any):
        if isinstance(other, Variable):
            axes = sorted(tuple(set(self.axes + other.axes)))
            sarr, oarr = self._extend_arrays(other, axes)
            data = sarr / oarr
            unit = self.unit() / other.unit()
            name = f"{self.name} / {other.name}"
            return self._new(
                data=data,
                unit=unit,
                axes=axes,
                name=name,
            )
        data = self._get_data().__truediv__(other)
        return self._new(
            data=data,
            unit=self.unit(),
            axes=self.axes,
            name=self.name,
        )

    def __pow__(self, other: typing.Any):
        if isinstance(other, numbers.Real):
            data = self._get_data().__pow__(other)
            unit = self.unit().__pow__(other)
            return self._new(
                data=data,
                unit=unit,
                axes=self.axes,
                name=self.name,
            )
        return NotImplemented

    @property
    def shape_dict(self) -> typing.Dict[str, int]:
        """Label and size for each axis."""
        return dict(zip(self.axes, self._get_data('shape')))

    def _extend_arrays(
        self,
        other: 'Variable',
        axes: typing.Tuple[str],
    ) -> typing.Tuple[np.ndarray]:
        """Extract arrays with extended axes.

        This method determines the set of unique axes shared by this
        instance and `other`, then extracts arrays suitable for computing a
        product or ratio that has the full set of axes.
        """
        tmp = {**other.shape_dict, **self.shape_dict}
        full_shape = tuple(tmp[d] for d in axes)
        idx = np.ix_(*[range(i) for i in full_shape])
        self_idx = tuple(idx[axes.index(d)] for d in self.shape_dict)
        self_arr = self._get_data(self_idx)
        other_idx = tuple(idx[axes.index(d)] for d in other.shape_dict)
        other_arr = other._get_data(other_idx)
        return self_arr, other_arr

    def __array__(self, *args, **kwargs) -> np.ndarray:
        """Support casting to `numpy` array types.
        
        Notes
        -----
        This will first cast `self._amount` (inherited from
        `~quantities.Measured`) on its own to a `numpy.ndarray`, before applying
        `*args` and `**kwargs`, in order to avoid a `TypeError` when using
        `netCDF4.Dataset`. See
        https://github.com/mcgibbon/python-examples/blob/master/scripts/file-io/load_netCDF4_full.py
        """
        data = self._get_array()
        return np.asanyarray(data, *args, **kwargs)

    def _get_data(self, arg: typing.Union[str, IndexLike]=None):
        """Access the data array or a dataset attribute.
        
        If `arg` is not a string, this method will assume it is an index and
        will attempt to return the relevant portion of the dataset array (after
        loading from disk, if necessary). If `arg` is a string, this method will
        first search `_amount` for the named attribute, to take advantage of
        viewers that provide metadata without loading the full dataset. If that
        search fails, this method will attempt to retrieve the named attribute
        from the full array.
        """
        if not isinstance(arg, str):
            return self._get_array(index=arg)
        if attr := getattr(self._amount, arg, None):
            return attr
        return getattr(self._get_array(), arg)

    def _get_array(self, index: IndexLike=None):
        """Access array data via index or slice notation.
        
        Notes
        -----
        If `index` is not `None`, this method will create the requested subarray
        from `self._amount` and directly return it. If `index` is `None`, this
        method will load the entire array and let execution proceed to the
        following block, which will immediately return the array. It will then
        subscript the pre-loaded array on subsequent calls. The reasoning behind
        this algorithm is as follows: If we need to load the full array at any
        point, we may as well save it because subscripting an in-memory
        `numpy.ndarray` is much faster than re-reading from disk for large
        arrays. However, we should avoid reading in the full array if the caller
        only wants a small portion of it, and in those cases, keeping the loaded
        data in memory will lead to incorrect results when attempting to access
        a different portion of the full array because the indices will be
        different. The worst-case scenario will occur when the caller repeatedly
        tries to access a large portion of the full array; this is a possible
        area for optimization.
        """
        if self._array is None:
            array = self._load_array(index) * self._scale
            if index is not None:
                return array
            self._array = array
        if iterables.missing(index):
            return self._array
        idx = np.index_exp[index]
        return self._array[idx]

    def _load_array(self, index: IndexLike=None):
        """Read the array data from disk.
        
        If `index` is "missing" in the sense defined by `~iterables.missing`
        this method will load and return the full array. If `index` is not
        missing, this method will first attempt to subscript `self._amount`
        before converting it to an array and returning it. If it catches either
        a `TypeError` or an `IndexError`, it will create the full array before
        subscripting and returning it. The former may occur if `self._amount` is
        a sequence type like `list`, `tuple`, or `range`; the latter may occur
        when attempting to subscript certain array-like objects (e.g.,
        `netCDF4._netCDF4.Variable`) with valid `numpy` index expressions.
        """
        if not iterables.missing(index):
            idx = np.index_exp[index]
            try:
                return np.asarray(self._amount[idx])
            except (TypeError, IndexError):
                return np.asarray(self._amount)[idx]
        return np.asarray(self._amount)

    def __str__(self) -> str:
        """A simplified representation of this object."""
        attrs = [
            f"shape={self.shape_dict}",
            f"unit={self.unit()}",
            f"name='{self.name}'",
        ]
        return ', '.join(attrs)

    @classmethod
    def implements(cls, numpy_function):
        """Register an `__array_function__` implementation for this class.

        See https://numpy.org/doc/stable/reference/arrays.classes.html for the
        suggestion on which this method is based.

        EXAMPLE
        -------
        Overload `numpy.mean` for an existing class called ``Array`` with a
        version that accepts no keyword arguments::

            @Array.implements(np.mean)
            def mean(a: Array, **kwargs) -> Array:
                if kwargs:
                    msg = "Cannot pass keywords to numpy.mean with Array"
                    raise TypeError(msg)
                return np.sum(a) / len(a)

        This will compute the mean of the underlying data when called with no
        arguments, but will raise an exception when called with arguments:

            >>> v = Array([[1, 2], [3, 4]])
            >>> np.mean(v)
            5.0
            >>> np.mean(v, axis=0)
            ...
            TypeError: Cannot pass keywords to numpy.mean with Array
        """
        def decorator(func):
            cls._HANDLED_FUNCTIONS[numpy_function] = func
            return func
        return decorator


@Variable.implements(np.mean)
def _array_mean(a: Variable, **kwargs):
    """Compute the mean and update array dimensions, if necessary."""
    data = a._get_data().mean(**kwargs)
    if (axis := kwargs.get('axis')) is not None:
        a.axes = tuple(
            d for d in a.axes
            if a.axes.index(d) != axis
        )
    return data


class Measurement(Vector):
    """The result of measuring an object.

    While it is possible to directly instantiate this class, it serves primarily
    as the return type of `quantities.measure`, which accepts a much wider
    domain of arguments.
    """

    def __getitem__(self, index):
        """Called for index-based value access."""
        if index < 0:
            index += len(self)
        return Scalar(self._values[index], self.unit)

    @property
    def values(self):
        """The numerical values of this measurement."""
        return self._values

    @property
    def unit(self):
        """The metric unit of this measurement.
        
        Unlike instances of `~quantities.Vector`, this object does not support
        updates to `unit`.
        """
        return super().unit()

    def __float__(self) -> float:
        """Represent a single-valued measurement as a `float`."""
        return self._cast_to(float)

    def __int__(self) -> int:
        """Represent a single-valued measurement as a `int`."""
        return self._cast_to(int)

    _T = typing.TypeVar('_T', int, float)
    def _cast_to(self, __type: _T) -> _T:
        """Internal method for casting to numeric type."""
        nv = len(self.values)
        if nv == 1:
            return __type(self.values[0])
        errmsg = f"Can't convert measurement with {nv!r} values to {__type}"
        raise TypeError(errmsg) from None

    def __str__(self) -> str:
        """A simplified representation of this object."""
        return f"{self.values} [{self.unit}]"


@typing.runtime_checkable
class Measurable(typing.Protocol):
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
    return Measurement(parsed[:-1], parsed[-1])


def parse_measurable(args, distribute: bool=False):
    """Extract one or more values and an optional unit from `args`.
    
    See Also
    --------
    measure : returns the parsed object as a `Measurement`.
    """

    # Strip redundant lists and tuples.
    unwrapped = iterables.unwrap(args)

    # Raise an error for null input.
    if iterables.missing(unwrapped):
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

    # Recursively handle an iterable of whole (distinct) items.
    if all(isinstance(arg, iterables.whole) for arg in unwrapped):
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


