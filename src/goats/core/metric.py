import abc
import collections.abc
import typing

import numpy

from goats.core import algebraic
from goats.core import aliased
from goats.core import iterables


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
        'quantity': 'mass number',
    },
    {
        'symbol': 'amu',
        'name': 'atomic mass unit',
        'quantity': 'mass number',
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
        'symbol': 'Oe',
        'name': 'Oersted',
        'quantity': 'magnetic intensity',
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
        'system': 'mks',
    },
    {
        'symbol': 'Ci',
        'name': 'Curie',
        'quantity': 'radioactivity',
        'system': 'cgs',
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
    'mass number': 'number',
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
PI = numpy.pi
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
    _cache: dict=None

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
        self._cache = {}
        cls._instances[key] = self
        return self

    def system(self, system: str):
        """Get all definitions of this property for `system`."""
        return {k: v[system] for k, v in self.items()}

    LEN = len(_QUANTITIES) # No need to compute every time.
    def __len__(self) -> int:
        """The number of defined quantities. Called for len(self)."""
        return self.LEN

    def __iter__(self) -> typing.Iterator[str]:
        """Iterate over names of defined quantities. Called for iter(self)."""
        return iter(_QUANTITIES)

    def __getitem__(self, quantity: str):
        """Create or retrieve a named property."""
        if quantity in self._cache:
            return self._cache[quantity]
        if new := self._get_property(quantity):
            self._cache[quantity] = new
            return new
        raise KeyError(f"Unknown quantity {quantity}") from None

    def _get_property(self, quantity: str):
        """Get a named property of a defined quantity.
        
        This method will search for `quantity` in the module-level collection of
        defined quantities. If it doesn't find an entry, it will attempt to
        parse `quantity` into known quantities. If it finds a `dict` entry, it
        will attempt to extract the values corresponding to this property's
        `key`. If it finds a `str` entry, it will attempt to create the
        equivalent `dict` by algebraically evaluating the terms in the entry.
        """
        if quantity not in _QUANTITIES:
            return self._parse(quantity)
        q = _QUANTITIES[quantity]
        if isinstance(q, dict):
            return q.get(self.key, {})
        if not isinstance(q, str):
            raise TypeError(f"Expected {quantity} to be a string") from None
        return self._parse(q)

    def _parse(self, string: str):
        """Parse a string representing a compound quantity."""
        for k in _QUANTITIES:
            string = string.replace(k, k.replace(' ', '_'))
        parts = [self._expand(term) for term in algebraic.Expression(string)]
        keys = {key for part in parts for key in part.keys()}
        merged = {key: [] for key in keys}
        for part in parts:
            for key, value in part.items():
                merged[key].append(value)
        return {
            k: str(algebraic.Expression(v))
            for k, v in merged.items()
        }

    _operand = algebraic.OperandFactory()
    def _expand(self, term: algebraic.Term):
        """Create a `dict` of operands from this term."""
        return {
            k: self._operand.create(v, term.exponent)
            for k, v in self[term.base.replace('_', ' ')].items()
        }

    def __str__(self) -> str:
        """A simplified representation of this object."""
        return self.key


class Attributes(typing.NamedTuple):
    """Canonical values of a quantity within a metric system."""

    # NOTE: These types are not `Unit` and `Dimension` because we need
    # `Quantity` and `Attributes` to define `Unit`.
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


class Prefix(typing.NamedTuple):
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
        instance : `~metric.NamedUnit`
            An existing instance of this class.
        """

    _dimensions = Property('dimensions')

    _instances = {}

    prefix: Prefix=None
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
            A 2-tuple in which the first element is a `~metric.Prefix`
            representing the order-of-magnitude of the given unit and the second
            element is a `~metric.BaseUnit` representing the unscaled (i.e.,
            order-unity) metric unit.

        Examples
        --------
        >>> mag, ref = NamedUnit.parse('km')
        >>> mag
        Prefix(symbol='k', name='kilo', factor=1000.0)
        >>> ref
        BaseUnit(symbol='m', name='meter', quantity='length', system='mks')
        """
        try:
            unit = named_units[string]
        except KeyError as err:
            raise UnitParsingError(string) from err
        magnitude = Prefix(**unit['prefix'])
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
        terms = algebraic.Expression(self.dimension)
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
            algebraic.Term(
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
    factor: float=None

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
        methods = (
            self._convert_strings,
            self._convert_expressions,
        )
        self.factor = self._compute(methods)
        cls._instances[key] = self
        return self

    @property
    def inverse(self):
        """The conversion from `u1` to `u0`."""
        return type(self)(self.u1, self.u0)

    def _compute(
        self,
        methods: typing.Iterable[typing.Callable[[str, str], float]],
    ) -> typing.Optional[float]:
        """Call conversion methods on this instance's units."""
        for method in methods:
            self._checked = []
            if conversion := method(self.u0, self.u1):
                return conversion
        raise UnitConversionError(self.u0, self.u1)

    def _convert_strings(self, u0: str, u1: str, scale: float=1.0):
        """Attempt to convert `u0` to `u1` as strings.
        
        This method will attempt to look up or compute the appropriate numerical
        conversion factor via various strategies, including searching for
        equivalent conversions or iteratively building a new conversion from
        existing conversions.
        """
        self._checked += [u0]
        if conversion := self._simple_conversion(u0, u1):
            return conversion
        if conversion := self._complex_conversion(u0, u1):
            return conversion

    def _simple_conversion(cls, u0: str, u1: str, scale: float=1.0):
        """Attempt to compute a simple conversion from `u0` to `u1`.
        
        This method will attempt the following conversions, in the order listed:
        
        * the identity conversion (i.e., `u0 == u1`);
        * a defined conversion from `u0` to `u1`;
        * the metric ratio of `u1` to `u0` (e.g., `'km'` to `'m'`);

        If it does not succeed, it will return ``None`` in order to allow other
        methods an opportunity to convert `u0` to `u1`.
        """
        if u0 == u1:
            return scale
        if found := cls._search(u0, u1):
            return scale * found
        try:
            ratio = NamedUnit(u1) // NamedUnit(u0)
        except (ValueError, UnitParsingError):
            return
        return scale * ratio

    available = NamedUnit.knows_about
    """Local copy of `~metric.NamedUnit.knows_about`."""

    @classmethod
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

    @classmethod
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

    units = CONVERSIONS.nodes
    """Local copy of `~metric.CONVERSIONS.nodes`."""

    def _complex_conversion(self, u0: str, u1: str, scale: float=1.0):
        """Attempt to compute a complex conversion from `u0` to `u1`.

        This method will attempt the following conversions, in the order listed:

        * a defined conversion to `u1` from a unit that has the same base unit
          as, but different metric scale than, `u0` (see
          `~Conversion._rescale`);
        * the inverse of the above process applied to the conversion from `u1`
          to `u0`;
        * a multi-step conversion recursively built by calling back to
          `~Conversions._convert_strings`

        If it does not succeed, it will return ``None`` in order to allow other
        methods an opportunity to convert `u0` to `u1`.
        
        Notes
        -----
        It is possible for `u0` or `u1` to individually be in
        `CONVERSIONS.nodes` even when `(u0, u1)` is not in `CONVERSIONS`. For
        example, there are nodes for both 'min' (minute) and 'd' (day), each
        with a conversion to 's' (second), but there is no direct conversion
        from 'min' to 'd'.
        """
        if u0 not in self.units:
            if computed := self._rescale(u0, u1):
                return scale * computed
        if u1 not in self.units:
            if computed := self._rescale(u1, u0):
                return scale / computed
        conversions = CONVERSIONS.get_adjacencies(u0).items()
        for unit, weight in conversions:
            if unit not in self._checked:
                if value := self._convert_strings(unit, u1, scale=scale):
                    return weight * value

    @classmethod
    def _rescale(cls, u0: str, u1: str):
        """Compute a new conversion after rescaling `u0`.
        
        This method will look for a unit, `ux`, in `~metric.CONVERSIONS`
        that has the same base unit as `u0`. If it finds one, it will attempt to
        convert `ux` to `u1`, and finally multiply the result by the relative
        magnitude of `u0` to `ux`. In other words, it attempts to compute ``(u0
        // ux) * (ux -> u1)`` in place of ``(u0 -> u1)``.
        """
        if not cls.available(u0):
            return
        n0 = NamedUnit(u0)
        for ux in cls.units:
            if cls.available(ux):
                nx = NamedUnit(ux)
                if nx.base == n0.base:
                    if found := cls._search(ux, u1):
                        return (nx // n0) * found

    def _convert_expressions(self, u0: str, u1: str):
        """Convert complex unit expressions term-by-term."""
        e0, e1 = (algebraic.Expression(unit) for unit in (u0, u1))
        if e0 == e1:
            return 1.0
        identities = {'#', '1'}
        ratio = [term for term in e0 / e1 if term.base not in identities]
        factor = 1.0
        matched = []
        unmatched = ratio.copy()
        for target in ratio:
            if target not in matched:
                if match := self._match_terms(target, unmatched):
                    value, term = match
                    if term != target:
                        for this in (target, term):
                            matched.append(this)
                            unmatched.remove(this)
                    else:
                        matched.append(target)
                        unmatched.remove(target)
                    factor *= value
        if not unmatched:
            return factor
        raise UnitConversionError(self.u0, self.u1)

    def _match_terms(
        self,
        target: algebraic.Term,
        terms: typing.Iterable[algebraic.Term],
    ) -> typing.Optional[typing.Union[float, algebraic.Term]]:
        """Attempt to convert `target` to a term in `terms`."""
        u0 = target.base
        exponent = target.exponent
        like_terms = [
            term for term in terms
            if term != target and term.exponent == -exponent
        ]
        for term in like_terms:
            u1 = term.base
            if conversion := self._convert_strings(u0, u1):
                return conversion ** exponent, term
        if self.available(u0) and NamedUnit(u0).dimension == '1':
            return 1.0, target

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


class _Converter(iterables.ReprStrMixin):
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
        key = tuple(str(arg) for arg in args)
        if available := cls._instances.get(key):
            return available
        unit, quantity = key
        self = super().__new__(cls)
        self.quantity = quantity
        self._substitutions = self._units[self.quantity]
        self.unit = self._substitutions.get(unit) or unit
        self._defined = None
        cls._instances[key] = self
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
           conversions to `~metric.NamedUnit` and an arithmetic operation,
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

    def __str__(self) -> str:
        return f"{self.unit!r} [{self.quantity!r}]"


Instance = typing.TypeVar('Instance', bound='Quantity')


class Quantity(iterables.ReprStrMixin):
    """A single metric quantity."""

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
            return Attributes(dimension=dimension, unit=unit)

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


class Unit(algebraic.Expression):
    """An algebraic expression representing a physical unit."""

    _instances = {}

    _dimension=None
    _quantity=None

    def __new__(
        cls: typing.Type[Instance],
        arg: typing.Union['Unit', str, iterables.whole],
        **kwargs
    ) -> Instance:
        """Create a new unit from `arg`."""
        if isinstance(arg, cls):
            return arg
        if isinstance(arg, str) and arg in cls._instances:
            return cls._instances[arg]
        self = super().__new__(cls, arg, **kwargs)
        self._dimension = None
        self._quantity = None
        cls._instances[str(self)] = self
        return self

    @property
    def dimension(self):
        """The physical dimension of this unit."""
        if self._dimension is None:
            self._dimension = Dimension(self)
        return self._dimension

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

        As with `~metric.NamedUnit`, these results are equivalent to the
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
        return (
            Conversion(str(other), str(self)).factor
            if isinstance(other, (str, Unit))
            else NotImplemented
        )

    def __rfloordiv__(self, other):
        """Compute the inverse of self // other."""
        return 1.0 / self.__floordiv__(other)


Instance = typing.TypeVar('Instance', bound='Dimension')


class Dimension(algebraic.Expression):
    """An algebraic expression representing a physical dimension.
    
    This class exists to support the `dimension` property of `~metric.Unit`.
    It is essentially a thin wrapper around `~algebra.Expression` with logic to
    compute the dimension of a `~metric.Unit` instance or equivalent object.
    """

    def __new__(
        cls: typing.Type[Instance],
        arg: typing.Union[Unit, Attributes, str, iterables.whole],
    ) -> Instance:
        if isinstance(arg, Unit):
            terms = [
                algebraic.Term(
                    base=NamedUnit(term.base).dimension,
                    exponent=term.exponent,
                ) for term in algebraic.Expression(arg)
            ]
            return super().__new__(cls, terms)
        if isinstance(arg, Attributes):
            return super().__new__(cls, arg.dimension)
        return super().__new__(cls, arg)


class SearchError(KeyError):
    """Error while searching for a requested metric."""
    pass


class System(collections.abc.Mapping, iterables.ReprStrMixin):
    """Representations of physical quantities within a given metric system."""

    _instances = {}
    name: str=None
    dimensions: typing.Dict[str, str]=None
    units: typing.Dict[str, str]=None

    def __new__(cls, arg: typing.Union[str, 'System']):
        """Return an existing instance or create a new one.

        Parameters
        ----------
        arg : str or `~metric.System`
            The name of the metric system to represent (e.g., 'mks') or an
            existing instance. Names are case sensitive. Instances are
            singletons.
        """
        if isinstance(arg, cls):
            return arg
        name = arg.lower()
        if instance := cls._instances.get(name):
            return instance
        self = super().__new__(cls)
        self.name = name
        self.dimensions = Property('dimensions').system(name)
        self.units = Property('units').system(name)
        cls._instances[name] = self
        return self

    def __len__(self) -> int:
        """The number of quantities defined in this metric system."""
        return _QUANTITIES.__len__()

    def __iter__(self) -> typing.Iterator:
        """Iterate over defined metric quantities."""
        return _QUANTITIES.__iter__()

    def __getitem__(self, key: typing.Union[str, Quantity]):
        """Get the metric for the requested quantity in this system."""
        try:
            quantity = Quantity(key)
        except ValueError as err:
            raise KeyError(f"No known quantity called '{key}'") from err
        else:
            return quantity[self.name]

    def get_unit(
        self,
        *,
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

    T = typing.TypeVar('T', str, Unit, Dimension, Quantity)
    T = typing.Union[str, Unit, Dimension, Quantity]

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
                return Unit(self['identity'].unit)
            if result := method(arg):
                return result
        args = self._format_targets(nonnull)
        errmsg = f"Could not determine unit in {self.name} from {args}"
        raise SearchError(errmsg)

    def _format_targets(self, targets: typing.Dict[str, T]):
        """Format `get_unit` targets for printing."""
        if not targets:
            return "nothing"
        args = [f'{k}={v!r}' for k, v in targets.items()]
        if 0 < len(args) < 3:
            return ' or '.join(args)
        return f"{', '.join(str(arg) for arg in args[:-1])}, or {args[-1]}"

    def _unit_from_unit(
        self,
        target: typing.Union[str, Unit],
    ) -> Unit:
        """Get the canonical unit corresponding to the given unit."""
        unit = Unit(target)
        if unit == '1' or unit in self.units.values():
            return unit
        return self._unit_from_dimension(unit.dimension)

    def _unit_from_dimension(
        self,
        target: typing.Union[str, Dimension],
    ) -> Unit:
        """Get the canonical unit corresponding to the given dimension."""
        for quantity, dimension in self.dimensions.items():
            if dimension == target:
                return Unit(self.units[quantity])

    def _unit_from_quantity(
        self,
        quantity: typing.Union[str, Quantity],
    ) -> Unit:
        """Get the canonical unit corresponding to the given quantity."""
        return Unit(self[quantity].unit)

    def __eq__(self, other) -> bool:
        """True if two systems have the same `name` attribute."""
        return (
            other.name == self.name if isinstance(other, System)
            else NotImplemented
        )

    def __bool__(self) -> bool:
        """A defined metric system is always truthy."""
        return True

    def keys(self) -> typing.KeysView[str]:
        return super().keys()

    def values(self) -> typing.ValuesView[Attributes]:
        return super().values()

    def items(self) -> typing.ItemsView[str, Attributes]:
        return super().items()

    def __str__(self) -> str:
        """A simplified representation of this object."""
        return str(self.name)

