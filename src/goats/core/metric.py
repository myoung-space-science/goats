import abc
import collections.abc
import contextlib
import numbers
import typing

import numpy

from goats.core import symbolic
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

_PREFIXES_TABLE = iterables.Table(_prefixes)


UNITY = {'#', '1'}
"""Strings that represent dimensionless units."""


SYSTEMS = {'mks', 'cgs'}
"""The metric systems known to this module."""


_units = [
    {
        'symbol': 'm',
        'name': 'meter',
        'quantity': 'length',
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
    },
    {
        'symbol': 'erg',
        'name': 'erg',
        'quantity': 'energy',
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
    },
    {
        'symbol': 'dyn',
        'name': 'dyne',
        'quantity': 'force',
    },
    {
        'symbol': 'Pa',
        'name': 'pascal',
        'quantity': 'pressure',
    },
    {
        'symbol': 'W',
        'name': 'watt',
        'quantity': 'power',
    },
    {
        'symbol': 'C',
        'name': 'coulomb',
        'quantity': 'charge',
    },
    {
        'symbol': 'statC',
        'name': 'statcoulomb',
        'quantity': 'charge',
    },
    {
        'symbol': 'statA',
        'name': 'statampere',
        'quantity': 'current',
    },
    {
        'symbol': 'statV',
        'name': 'statvolt',
        'quantity': 'potential',
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
    },
    {
        'symbol': 'Ω',
        'name': 'ohm',
        'quantity': 'resistance',
    },
    {
        'symbol': 'S',
        'name': 'seimens',
        'quantity': 'conductance',
    },
    {
        'symbol': 'F',
        'name': 'farad',
        'quantity': 'capacitance',
    },
    {
        'symbol': 'Wb',
        'name': 'weber',
        'quantity': 'magnetic flux',
    },
    {
        'symbol': 'Mx',
        'name': 'maxwell',
        'quantity': 'magnetic flux',
    },
    {
        'symbol': 'Oe',
        'name': 'Oersted',
        'quantity': 'magnetic intensity',
    },
    {
        'symbol': 'H',
        'name': 'henry',
        'quantity': 'inductance',
    },
    {
        'symbol': 'T',
        'name': 'tesla',
        'quantity': 'induction',
    },
    {
        'symbol': 'G',
        'name': 'gauss',
        'quantity': 'induction',
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
        'symbol': 'Ci',
        'name': 'Curie',
        'quantity': 'radioactivity',
    },
    {
        'symbol': 'Gy',
        'name': 'gray',
        'quantity': 'dosage',
    },
    {
        'symbol': 'P',
        'name': 'poise',
        'quantity': 'viscosity',
    },
    {
        'symbol': '1',
        'name': 'unitless',
        'quantity': 'identity',
    },
]

_UNITS_TABLE = iterables.Table(_units)


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
    'acceleration': 'velocity / time',
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
    'fluence': 'particle fluence',
    'flux': 'particle flux',
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
    'integral flux': 'flux * energy',
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
    'particle fluence': 'number / (area * solid_angle * energy / mass_number)',
    'particle flux': 'fluence / time',
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
    'permittivity': {
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
    'speed': 'velocity',
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
    'wavenumber': '1 / length',
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


Instance = typing.TypeVar('Instance', bound='_Property')


class _Property(collections.abc.Mapping, iterables.ReprStrMixin):
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

    def __getitem__(self, quantity: str) -> typing.Dict[str, str]:
        """Create or retrieve a named property."""
        if quantity in self._cache:
            return self._cache[quantity]
        if new := self._get_property(quantity):
            self._cache[quantity] = new
            return new
        raise KeyError(f"Unknown quantity {quantity}") from None

    def _get_property(self, quantity: str) -> typing.Dict[str, str]:
        """Get this property for a defined quantity.
        
        This method will search for `quantity` in the module-level collection of
        defined quantities. If it doesn't find an entry, it will attempt to
        parse `quantity` into known quantities. If it finds a `dict` entry, it
        will attempt to extract the values corresponding to this property's
        ``key`` attribute (i.e., 'units' or 'dimensions'). If it finds a `str`
        entry, it will attempt to create the equivalent `dict` by symbolically
        evaluating the terms in the entry.
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
        parts = [self._expand(term) for term in symbolic.Expression(string)]
        keys = {key for part in parts for key in part.keys()}
        merged = {key: [] for key in keys}
        for part in parts:
            for key, value in part.items():
                merged[key].append(value)
        return {
            k: str(symbolic.Expression(v))
            for k, v in merged.items()
        }

    # TODO: 
    # - Define a function in `symbolic` that is equivalent to calling
    #   `symbolic.OperandFactory().create(...)`.
    # - Refactor this method.
    _operand = symbolic.OperandFactory()
    def _expand(self, term: symbolic.Term):
        """Create a `dict` of operands from this term."""
        return {
            k: self._operand.create(v, term.exponent)
            for k, v in self[term.base.replace('_', ' ')].items()
        }

    def __str__(self) -> str:
        """A simplified representation of this object."""
        return self.key


# NOTE: Defining mappings from unit or dimension to quantity is a bad idea
# because some quantities have the same unit or dimension in a given system.
# This makes the mapping ill-defined. Python dictionaries simply use the latest
# entry for a repeated key, which means some quantities would overwrite others.
# The following objects rely on mappings from quantity to unit or dimension,
# which are always well defined.


DIMENSIONS = _Property('dimensions')
"""All defined metric dimensions.

This mapping is keyed by physical quantity followed by metric system.
"""


UNITS = _Property('units')
"""All defined metric units.

This mapping is keyed by physical quantity followed by metric system.
"""


CANONICAL = {
    k: {
        system: _Property(k).system(system) for system in ('mks', 'cgs')
    } for k in ('dimensions', 'units')
}
"""Canonical metric properties in each known metric system.

This mapping is keyed by {'dimensions', 'units'}, followed by metric system, and
finally by physical quantity.
"""


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
        for prefix in _PREFIXES_TABLE for unit in _UNITS_TABLE
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
        return f"Can't convert {self._from!r} to {self._to!r}"


class SystemAmbiguityError(Exception):
    """The metric system is ambiguous."""


class UnitSystemError(Exception):
    """The metric system does not contain this unit."""


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


class Reduction(iterables.ReprStrMixin):
    """The components of a reduced unit expression."""

    def __init__(
        self,
        terms: symbolic.Expressable,
        scale: float=1.0,
        system: str=None,
    ) -> None:
        self._expression = scale * symbolic.Expression(terms)
        self.system = system
        self.scale = scale
        self._units = None

    @property
    def units(self) -> typing.List[symbolic.Term]:
        """The unit terms in this reduction."""
        if self._units is None:
            self._units = [
                unit for unit in self._expression
                if unit.base != '1'
            ]
        return self._units

    def __mul__(self, other):
        """Called for self * other."""
        if not isinstance(other, numbers.Real):
            return NotImplemented
        scale = self.scale * other
        terms = list(scale * self._expression)
        return type(self)(terms, scale=scale, system=self.system)

    __rmul__ = __mul__
    """Called for other * self."""

    def __truediv__(self, other):
        """Called for self / other."""
        if not isinstance(other, numbers.Real):
            return NotImplemented
        scale = self.scale / other
        terms = list(scale * self._expression)
        return type(self)(terms, scale=scale, system=self.system)

    def __pow__(self, other):
        """Called for self ** other."""
        if not isinstance(other, numbers.Real):
            return NotImplemented
        scale = self.scale ** other
        terms = list(scale * self._expression**other)
        return type(self)(terms, scale=scale, system=self.system)

    def __str__(self) -> str:
        return f"{self._expression} [{self.system!r}]"


def identify(string: str):
    """Determine the magnitude and reference unit, if possible.
    
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
    >>> mag, ref = identify('km')
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


Instance = typing.TypeVar('Instance', bound='NamedUnit')


class _NamedUnitMeta(abc.ABCMeta):
    """Internal metaclass for `~metric.NamedUnit`.
    
    This class exists to create singleton instances of `~metric.NamedUnit`
    without needing to overload `__new__` on that class or its base class(es).
    """

    _instances = aliased.MutableMapping()
    _attributes = {}

    def __call__(
        cls,
        arg: typing.Union[Instance, symbolic.Expressable],
    ) -> Instance:
        """Create a new instance or return an existing one."""
        if isinstance(arg, cls):
            # If the argument is already an instance, return it.
            return arg
        string = str(arg)
        if available := cls._instances.get(string):
            # If the argument maps to an existing unit, return that unit.
            return available
        # First time through: identify the base unit and prefix.
        magnitude, reference = identify(string)
        # Store the information to initialize a new instance.
        name = f"{magnitude.name}{reference.name}"
        symbol = f"{magnitude.symbol}{reference.symbol}"
        cls._attributes[string] = {
            'prefix': magnitude,
            'base': reference,
            'name': name,
            'symbol': symbol,
            'scale': magnitude.factor,
            'quantity': reference.quantity,
        }
        # Create the new instance. This will ultimately pass control to
        # `NamedUnit.__init__`, which will initialize the newly instantiated
        # instance with the stored attributes corresponding to `str(arg)`.
        instance = super().__call__(arg)
        cls._instances[(name, symbol)] = instance
        return instance


class NamedUnit(iterables.ReprStrMixin, metaclass=_NamedUnitMeta):
    """A single named unit and corresponding metadata."""

    @typing.overload
    def __init__(
        self: Instance,
        unit: str,
    ) -> Instance:
        """Create a new instance or return an existing one.
        
        Parameters
        ----------
        unit : string
            A string representing the metric unit to create.
        """

    @typing.overload
    def __init__(
        self: Instance,
        instance: Instance,
    ) -> Instance:
        """Create a new instance or return an existing one.
        
        Parameters
        ----------
        instance : `~metric.NamedUnit`
            An existing instance of this class.
        """

    def __init__(self, arg) -> None:
        self._parsed = self.__class__._attributes[str(arg)]
        self._prefix = None
        self._base = None
        self._name = None
        self._symbol = None
        self._scale = None
        self._quantity = None
        self._systems = None
        self._dimensions = None
        self._decomposed = None
        self._norm = None
        self._reductions = dict.fromkeys(SYSTEMS)

    @property
    def norm(self):
        """The equivalent unit, represented in base units of `system`.
        
        Notes
        -----
        This property returns a copy of the original `dict` of normalized units
        in order to prevent modifying singleton instances.
        """
        if self._norm is None:
            self._norm = {
                system: type(self)(UNITS[self.quantity][system])
                for system in SYSTEMS
            }
        return self._norm.copy()

    @property
    def prefix(self) -> Prefix:
        """The order of magnitide of this unit's metric prefix."""
        if self._prefix is None:
            self._prefix = self._parsed["prefix"]
        return self._prefix

    @property
    def base(self) -> BaseUnit:
        """The reference unit without metric prefix."""
        if self._base is None:
            self._base = self._parsed["base"]
        return self._base

    @property
    def name(self) -> str:
        """The full name of this unit."""
        if self._name is None:
            self._name = self._parsed["name"]
        return self._name

    @property
    def symbol(self) -> str:
        """The abbreviated symbol for this unit."""
        if self._symbol is None:
            self._symbol = self._parsed["symbol"]
        return self._symbol

    @property
    def scale(self) -> float:
        """The metric scale factor of this unit."""
        if self._scale is None:
            self._scale = self._parsed["scale"]
        return self._scale

    @property
    def quantity(self) -> str:
        """The physical quantity of this unit."""
        if self._quantity is None:
            self._quantity = self._parsed["quantity"]
        return self._quantity

    @property
    def systems(self):
        """The metric systems that use this unit.
        
        This property uses the criteria described in
        `~metric.NamedUnit.is_allowed_in` to build the collection of metric
        systems, most notably that named units not defined in any metric system
        are allowed in all metric systems.
        """
        if self._systems is None:
            modes = {
                k: []
                for k in {'allowed', 'defined', 'fundamental'}
            }
            for system in SYSTEMS:
                if self.is_fundamental_in(system):
                    modes['fundamental'].append(system)
                if self.is_defined_in(system):
                    modes['defined'].append(system)
                if self.is_allowed_in(system):
                    modes['allowed'].append(system)
            self._systems = {k: tuple(v) for k, v in modes.items()}
        return self._systems.copy()

    @property
    def decomposed(self):
        """The representation of this unit in base units, if possible."""
        if self._decomposed is None:
            with contextlib.suppress(StopIteration):
                system = next(
                    system for system in SYSTEMS
                    if self.is_fundamental_in(system)
                )
                self._decomposed = self._decompose(system)
        return self._decomposed

    def _decompose(self, system: typing.Literal['mks', 'cgs']):
        """Internal logic for `NamedUnit.decomposed`."""
        if not self.is_defined_in(system):
            # If this unit is not defined in this metric system, we can't
            # decompose it.
            return
        dimension = self.dimensions[system]
        expression = symbolic.Expression(dimension)
        if len(dimension) == 1:
            # If this unit's dimension is irreducible, there's no point in going
            # through all the decomposition logic.
            return [symbolic.Term(self.symbol)]
        quantities = [
            _BASE_QUANTITIES.find(term.base)[0]['name']
            for term in expression
        ]
        units = [
            _QUANTITIES[quantity]['units'][system]
            for quantity in quantities
        ]
        return [
            symbolic.Term(base=unit, exponent=term.exponent)
            for unit, term in zip(units, expression)
        ]

    def reduce(self, system: str=None) -> typing.Optional[Reduction]:
        """Convert this unit to base units of `system`, if possible."""
        s = self._resolve_system(system)
        if self._reductions[s]:
            return self._reductions[s]
        result = self._reduce(s)
        self._reductions[s] = result
        return result

    def _resolve_system(self, system: typing.Optional[str]):
        """Determine the appropriate metric system to use, if possible."""
        if isinstance(system, str) and system.lower() in SYSTEMS:
            # trivial case
            return system.lower()
        systems = [s for s in SYSTEMS if self.is_fundamental_in(s)]
        if len(systems) == 1:
            # use canonical system if possible
            return systems[0]
        if self.dimensions['mks'] == self.dimensions['cgs']:
            # system-independent: use mks by default
            return 'mks'
        if self.dimensions['mks'] is None:
            # only defined in cgs
            return 'cgs'
        if self.dimensions['cgs'] is None:
            # only defined in mks
            return 'mks'
        # system-dependent but we don't know the system
        raise SystemAmbiguityError(str(self))

    def _reduce(self, system: typing.Literal['mks', 'cgs']):
        """Internal logic for `~NamedUnit.reduce`."""
        if not self.is_defined_in(system):
            # If this unit is not defined in this metric system, we can't
            # reduce it.
            return
        dimension = self.dimensions[system]
        expression = symbolic.Expression(dimension)
        if len(expression) == 1:
            # If this unit's dimension is irreducible, there's no point in going
            # through all the reduction logic.
            canonical = CANONICAL['units'][system][self.quantity]
            if self.symbol == canonical:
                # If this is the canonical unit for its quantity in `system`,
                # return it with a scale of unity.
                return Reduction(
                    [symbolic.Term(self.symbol)],
                    system=system,
                )
            # If not, return the canonical unit with the appropriate scale
            # factor.
            return Reduction(
                [symbolic.Term(canonical)],
                scale=(canonical // self),
                system=system,
            )
        quantities = [
            _BASE_QUANTITIES.find(term.base)[0]['name']
            for term in expression
        ]
        units = [
            _QUANTITIES[quantity]['units'][system]
            for quantity in quantities
        ]
        terms = [
            symbolic.Term(base=unit, exponent=term.exponent)
            for unit, term in zip(units, expression)
        ]
        return Reduction(terms, scale=self.scale, system=system)

    @property
    def dimensions(self) -> typing.Dict[str, typing.Optional[str]]:
        """The physical dimension of this unit in each metric system.
        
        Notes
        -----
        This property returns a copy of an internal `dict` in order to prevent
        accidentally changing the instance dimensions through an otherwise valid
        `dict` operation. Such changes are irreversible since each
        `~metric.NamedUnit` instance is a singleton.
        """
        if self._dimensions is None:
            systems = {system for system in self.systems['defined']}
            self._dimensions = {
                k: symbolic.Expression(v)
                for k, v in self._get_dimensions(systems).items()
            }
        return self._dimensions.copy()

    def _get_dimensions(self, systems: typing.Set[str]):
        """Helper for computing dimensions of this named unit.
        
        Notes
        -----
        This method requires the full set of applicable metric systems (rather
        than one system at a time) because it will return all available
        dimensions if either 1) there are no applicable systems, or 2) the set
        of applicable systems is equal to the set of all known systems.
        """
        dimensions = DIMENSIONS[self.quantity]
        if not systems or (systems == SYSTEMS):
            return dimensions.copy()
        base = dict.fromkeys(SYSTEMS)
        if len(systems) == 1:
            system = systems.pop()
            base[system] = dimensions[system]
            return base
        raise SystemAmbiguityError

    def is_allowed_in(self, system: typing.Literal['mks', 'cgs']):
        """True if this named unit inter-operates with units in `system`.
        
        A named unit is allowed in some or all metric systems, but never none.
        The reason for this is that a named unit that is not defined in any
        metric system is effectively independent of all metric systems, so
        attempting to restrict its use to a subset of metric systems is
        fruitless.

        See Also
        --------
        `~is_defined_in`
            True if the given metric system formally contains this named unit.

        `~is_fundamental_in`
            True if this named unit is the fundamental unit for its dimension in
            the given metric system.
        """
        systems = {s for s in SYSTEMS if self.is_defined_in(s)} or SYSTEMS
        return system in systems

    def is_defined_in(self, system: typing.Literal['mks', 'cgs']):
        """True if this named unit is defined in `system`."""
        if self.is_fundamental_in(system):
            return True
        canonical = CANONICAL['units'][system][self.quantity]
        with contextlib.suppress(UnitParsingError):
            reference = type(self)(canonical)
            if self.base == reference.base:
                return True
        return False

    def is_fundamental_in(self, system: typing.Literal['mks', 'cgs']):
        """True if this named unit is the canonical unit in `system`."""
        canonical = CANONICAL['units'][system][self.quantity]
        keys = (self.symbol, self.name)
        for key in keys:
            if key == canonical:
                return True
        return False

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

        This operation computes the numerical factor necessary to convert a
        quantity in the unit of `self` to a quantity in the unit of `other`,
        provided `self` and `other` represent units with the same base unit. It
        is not intended for unit conversion.

        Notes
        --------
        The result of this operation is the inverse of the result of
        `~metric.ratio`:

            >>> metric.ratio('cm', 'm')
            0.01
            >>> metric.NamedUnit('cm') // 'm'
            100.0

        This is so that the resultant numerical value correctly scales a
        quantity in `self` to a quantity in `other`. Using the units in the
        above example, a numerical value of a quantity in centimeters is 100.0
        times the numerical value of the same quantity in meters.
        """
        return ratio(other, self)

    def __rfloordiv__(self, other):
        """The magnitude of other relative to self.
        
        Notes
        -----
        This method used to return the inverse of the forward operation:

            def __rfloordiv__(self, other):
                return 1.0 / (self // other)

        However, that approach resulted in degraded precision. It now simply
        calls `~metric.ratio` with operands reversed relative to the forward
        operation.
        """
        return ratio(self, other)

    def __str__(self) -> str:
        """A printable representation of this unit."""
        return f"'{self.name} | {self.symbol}'"


def ratio(
    this: typing.Union[str, NamedUnit],
    that: typing.Union[str, NamedUnit],
) -> float:
    """Compute the magnitude of `this` relative to `that`.

    Parameters
    ----------
    this : string or `~metric.NamedUnit`
        The reference unit.

    that : string or `~metric.NamedUnit`
        The unit to compare to `this`. Must have the same base unit as `this`.

    Examples
    --------
    The following are all equivalent to the fact that a meter represents
    100 centimeters:

        >>> ratio('meter', 'centimeter')
        100.0
        >>> ratio('centimeter', 'meter')
        0.01
        >>> ratio('m', 'cm')
        100.0
        >>> ratio('cm', 'm')
        0.01

    Attempting this operation between two units with different base units will
    raise an exception,

        >>> ratio('m', 's')
        <raises ValueError>

    even if they represent the same quantity.

        >>> ratio('m', 'au')
        <raises ValueError>

    Therefore, this function is not intended for unit conversion.
    """
    u = [NamedUnit(i) if isinstance(i, str) else i for i in (this, that)]
    if not all(isinstance(i, NamedUnit) for i in u):
        raise TypeError(
            f"Each unit must be an instance of {str!r} or {NamedUnit!r}"
        ) from None
    u0, u1 = u
    if u1 == u0:
        return 1.0
    if u1.base != u0.base:
        units = ' to '.join(f"{i.symbol!r} ({i.name})" for i in u)
        raise ValueError(f"Can't compare {units}") from None
    return u0.scale / u1.scale


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

    def __new__(cls, *args):
        """Concrete implementation."""
        if len(args) == 1 and isinstance(args[0], cls):
            return args[0]
        key = tuple(args)
        # TODO: Can we use the existence of a particular conversion to more
        # efficiently create its inverse?
        if available := cls._instances.get(key):
            return available
        u0, u1 = key
        self = super().__new__(cls)
        self.u0 = u0
        self.u1 = u1
        methods = (
            self._convert_as_strings,
            self._convert_as_expressions,
        )
        self._factor = self._compute(methods)
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

    def _convert_as_strings(self, u0: str, u1: str, scale: float=1.0):
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
        if unit in named_units:
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
          `~Conversions._convert_as_strings`

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
                if value := self._convert_as_strings(unit, u1, scale=scale):
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
        if not u0 in named_units:
            return
        n0 = NamedUnit(u0)
        for ux in cls.units:
            if ux in named_units:
                nx = NamedUnit(ux)
                if nx.base == n0.base:
                    if found := cls._search(ux, u1):
                        return (nx // n0) * found

    def _convert_as_expressions(self, u0: str, u1: str):
        """Convert complex unit expressions term-by-term."""
        e0, e1 = (symbolic.Expression(unit) for unit in (u0, u1))
        if e0 == e1:
            return 1.0
        terms = [term for term in e0 / e1 if term.base not in UNITY]
        if factor := self._resolve_terms(terms):
            return factor
        if factor := self._convert_by_dimensions(terms):
            return factor
        raise UnitConversionError(self.u0, self.u1)

    def _convert_by_dimensions(self, terms: typing.List[symbolic.Term]):
        """Attempt to compute a conversion via unit dimensions."""
        decomposed = []
        for term in terms:
            reduction = NamedUnit(term.base).reduce()
            if reduction:
                decomposed.extend(
                    [
                        symbolic.Term(
                            coefficient=reduction.scale**term.exponent,
                            base=this.base,
                            exponent=term.exponent*this.exponent,
                        )
                        for this in reduction.units
                    ]
                )
        # TODO: Should we try this in other `_convert_by_expressions` or
        # `_resolve_terms`?
        if symbolic.Expression(decomposed) == '1':
            return 1.0
        return self._resolve_terms(decomposed)

    def _match_terms(
        self,
        target: symbolic.Term,
        terms: typing.Iterable[symbolic.Term],
    ) -> typing.Optional[typing.Union[float, symbolic.Term]]:
        """Attempt to convert `target` to a term in `terms`."""
        u0 = target.base
        exponent = target.exponent
        like_terms = [
            term for term in terms
            if term != target and term.exponent == -exponent
        ]
        for term in like_terms:
            u1 = term.base
            if conversion := self._convert_as_strings(u0, u1):
                return conversion ** exponent, term
        dimensions = NamedUnit(u0).dimensions.values()
        if u0 in named_units and all(d == '1' for d in dimensions):
            return 1.0, target

    def _resolve_terms(self, terms: typing.List[symbolic.Term]):
        """Compute ratios of matching terms, if possible."""
        if len(terms) <= 1:
            # We require at least two terms for a ratio.
            return
        factor = 1.0
        matched = []
        unmatched = terms.copy()
        for target in terms:
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

    def __float__(self) -> float:
        """Reduce this instance to its numerical factor via float(self)."""
        if bool(self):
            return self._factor
        raise TypeError("Conversion is undefined") from None

    def __bool__(self):
        """True if this conversion exists."""
        # NOTE: This may be expensive.
        try:
            factor = self._factor
        except:
            return False
        return bool(factor)

    def __str__(self) -> str:
        """A simplified representation of this object."""
        return f"({self.u0!r} -> {self.u1!r}): {float(self)!r}"


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
        self._substitutions = UNITS[self.quantity]
        self.unit = self._substitutions.get(unit) or unit
        self._defined = None
        cls._instances[key] = self
        return self

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
        1. Attempt the explicit conversion via an instance of
           `~metric.Conversion` and, if successful, return the corresponding
           numerical factor.
        1. Raise an exception to alert the caller that the conversion is
           undefined.
        """
        unit = self._substitutions.get(target) or target
        if self.unit == unit:
            return 1.0
        if conversion := Conversion(self.unit, unit):
            return float(conversion)
        raise ValueError(
            f"Unknown conversion from {self.unit!r} to {unit!r}."
        ) from None

    def __str__(self) -> str:
        return f"{self.unit!r} [{self.quantity!r}]"


Instance = typing.TypeVar('Instance', bound='Unit')


class _UnitMeta(abc.ABCMeta):
    """Internal metaclass for `~metric.Unit`.
    
    This class exists to create singleton instances of `~metric.Unit` without
    needing to overload `__new__` on that class or its base class(es).
    """

    _instances = aliased.MutableMapping()

    def __call__(
        cls,
        arg: typing.Union[Instance, symbolic.Expressable],
        **kwargs
    ) -> Instance:
        """Create a new instance or return an existing one.
        
        The goals of this method (and, effectively, of this class) are:

        - Return the same instance for all possible representations of a unit
          expression. For example: 'm / s', 'm s^-1', and 's^-1 m' should all
          map to the unit that represents "meters per second".
        - Parse a given string representation exactly once.
        """
        if isinstance(arg, cls):
            # If the argument is already an instance, return it.
            return arg
        # Attempt to extract a string representing a single unit.
        if isinstance(arg, str):
            string = arg
        else:
            try:
                string = str(arg[0]) if len(arg) == 1 else None
            except TypeError:
                string = None
        if string in cls._instances:
            # If the argument maps to an existing unit, return that unit.
            return cls._instances[string]
        # First time through: create an symbolic expression from `arg`.
        instance = super().__call__(arg, **kwargs)
        # The canonical string representation is the expression string.
        name = str(instance)
        if name in cls._instances:
            # It turns out that the argument corresponds to an existing unit.
            unit = cls._instances[name]
            if isinstance(arg, str):
                # If the argument is a string, register the argument as an alias
                # for that unit so we can just retrieve it next time.
                cls._instances.alias(name, arg)
            return unit
        # Create the initial mapping aliases for this unit.
        try:
            # If `name` corresponds to a named unit register both the name and
            # symbol (e.g., 'centimeter' and 'cm').
            this = NamedUnit(name)
            key = (this.name, this.symbol)
        except UnitParsingError:
            # If attempting to parse a named unit from `name` failed, register
            # the canonical string and, if applicable, the string argument.
            key = (name, string) if string else name
        # Store and return the new instance.
        cls._instances[key] = instance
        return instance


class Unit(symbolic.Expression, metaclass=_UnitMeta):
    """An symbolic expression representing a physical unit."""

    def __init__(
        self: Instance,
        expression: typing.Union[Instance, str, iterables.whole],
        **kwargs
    ) -> None:
        super().__init__(expression, **kwargs)
        self._dimensions = None
        self._decomposed = None
        self._dimensionless = None
        self._quantity = None
        self._norm = None

    @property
    def norm(self):
        """The equivalent unit, represented in base units of `system`.
        
        Notes
        -----
        This property returns a copy of the original `dict` of normalized units
        in order to prevent modifying singleton instances.
        """
        if self._norm is None:
            self._norm = {
                system: type(self)(
                    symbolic.power(UNITS[term.base][system], term.exponent)
                    for term in self.quantity
                ) for system in SYSTEMS
            }
        return self._norm.copy()

    @property
    def quantity(self):
        """This unit's quantity, derived from its unit terms."""
        if self._quantity is None:
            terms = [
                symbolic.Term(
                    NamedUnit(term.base).quantity.replace(' ', '_')
                ) ** term.exponent
                for term in self
            ]
            self._quantity = symbolic.Expression(terms)
        return self._quantity

    @property
    def dimensionless(self):
        """True if this unit's dimension is '1' in all metric systems.
        
        Notes
        -----
        This property exists as a convenient shortcut for comparing this unit's
        dimension in each metric system to '1'. If you want to check whether
        this unit is dimensionless in a specific metric system, simply check
        the ``dimensions`` property for that system.
        """
        if self._dimensionless is None:
            self._dimensionless = all(
                dimension == '1'
                for dimension in self.dimensions.values()
            )
        return self._dimensionless

    @property
    def dimensions(self):
        """The physical dimension of this unit in each metric system."""
        if self._dimensions is None:
            self._dimensions = Dimensions.fromunit(self)
        return self._dimensions

    def __mul__(self, other):
        """Called for self * other."""
        if self is other or super().__eq__(other):
            return super().__pow__(2)
        return self._apply(symbolic.product, other)

    def __truediv__(self, other):
        """Called for self / other."""
        if self is other or super().__eq__(other):
            return type(self)(1)
        return self._apply(symbolic.ratio, other)

    def __pow__(self, exp: numbers.Real):
        """Called for self ** exp."""
        return self._apply(symbolic.power, exp)

    def _apply(self, operation, other):
        """Apply `operation` to this unit.
        
        This method will attempt to reduce each operand into base units
        before computing the result, in order to reduce the result as much as
        possible
        """
        this = self.decomposed
        if isinstance(other, numbers.Number):
            return type(self)(operation(this, other))
        return type(self)(operation(this, self.decompose(other)))

    @property
    def decomposed(self):
        """This unit's decomposition into base units, where possible."""
        if self._decomposed is None:
            self._decomposed = self.decompose(self)
        return self._decomposed

    @classmethod
    def decompose(cls, unit: symbolic.Expressable):
        """Decompose this unit into base units where possible."""
        decomposed = [
            part
            for term in symbolic.Expression(unit)
            for part in cls._decompose(term)
        ]
        return symbolic.reduce(decomposed)

    @classmethod
    def _decompose(cls, term: symbolic.Term):
        """Internal logic for `~metric.Unit.decompose`."""
        try:
            # Possible outcomes
            # - success: returns list of terms
            # - decomposition failed: returns `None`
            # - parsing failed: raises UnitParsingError
            # - metric system is ambiguous: raises SystemAmbiguityError
            current = NamedUnit(term.base).decomposed
        except (UnitParsingError, SystemAmbiguityError):
            # This effectively reduces the three failure modes listed above into
            # one result.
            current = None
        if current is None:
            # If the attempt to decompose this unit term failed or
            # raised an exception, our only option is to append the
            # existing term to the running list.
            return [term]
        # If the attempt to decompose this unit term succeeded,
        # we need to distribute the term's exponent over the new
        # terms and append each to the running list.
        return [base**term.exponent for base in current]

    def __floordiv__(self, other):
        """Compute the magnitude of this unit relative to `other`.

        This method essentially computes the amount of `self` per `other`. The
        result is the numerical factor, N, necessary to convert a quantity with
        unit `other` to the equivalent quantity with unit `self`.
        
        In symbolic terms, suppose you have an amount q0 of some physical
        quantity when expressed in unit u0. The equivalent amount when expressed
        in unit u1 is q1 = (u1 // u0) * q0.

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

        Notes
        -----
        The result of this operation is the inverse of the result of
        `~metric.conversion`. The justification is equivalent to that described
        in `~metric.NamedUnit.__floordiv__` regarding its relationship to
        `~metric.ratio`.
        """
        return (
            conversion(other, self, strict=True)
            if isinstance(other, (str, Unit))
            else NotImplemented
        )

    def __rfloordiv__(self, other):
        """Compute the magnitude of `other` relative to this unit.
        
        Notes
        -----
        This method does not compute the inverse of the result of
        ``__floordiv__``. The justification is similar to that described in
        `~metric.NamedUnit.__rfloordiv__`.
        """
        return (
            conversion(self, other, strict=True)
            if isinstance(other, (str, Unit))
            else NotImplemented
        )

    def __eq__(self, other) -> bool:
        """Called for self == other.
        
        Two unit expressions are equal if they satisfy one of the following
        conditions, in order of restrictiveness:

        * are identical (e.g., 'N' == 'N')
        * have symbolically equal strings (e.g., 'm / s' == 'm / s')
        * have symbolically equivalent strings (e.g., 'm / s' == 'm s^-1')
        * differ only by dimensionless terms (e.g., '1 / s' == '# / s')
        * have a ratio of unity (e.g., 'N' == 'kg * m / s^2')

        The final two conditions arguably amount to equivalence, rather than
        strict equality, between unit expressions. This class includes
        equivalence in the definition of equality because the physical meaning
        of a unit should not depend on its precise lexical representation.
        Equivalence therefore allows otherwise equal quantities with physically
        equivalent units to compare equal.
        """
        if other is self:
            # If they are identical, they are equal.
            return True
        equal = super().__eq__(other)
        if equal:
            # If the symbolic expressions are equal, the units are equal.
            return True
        unity = (
            str(term) in UNITY
            for term in self.difference(other, symmetric=True)
        )
        if all(unity):
            # If the only terms that differ between the units are dimensionless
            # terms, we can declare them equal by inspection (i.e., without
            # computing their conversion factor).
            return True
        this = type(self)(other)
        if set(self.decomposed) == set(this.decomposed):
            # If their terms are equal, the units are equal.
            return True
        if symbolic.equality(self.decomposed, this.decomposed):
            # If their terms produce equal expressions, the units are equal.
            return True
        if self.dimensionless != this.dimensionless:
            # If one, and only one, of the units is dimensionless, they can't
            # possibly be equal
            return False
        try:
            # If their numerical conversion factor is unity, they are
            # equivalent; for the purposes of this method (see the justification
            # given in the docstring), they are therefore equal.
            return self // this == 1.0
        except UnitConversionError:
            # Everything has failed, so we declare them unequal.
            return False

    def __or__(self, other):
        """Called for self | other.
        
        This method tests whether the given unit is metrically consistent with
        this unit. In order to be metrically consistent, the two units must have
        the same dimension in at least one metric system or have a defined conversion factor.
        """
        that = type(self)(other)
        for system in SYSTEMS:
            defined = self.dimensions[system]
            given = that.dimensions.values()
            if defined and any(d == defined for d in given):
                return True
        if self.quantity == that.quantity:
            return True
        try:
            self // that
        except UnitConversionError:
            return False
        else:
            return True

    __ror__ = __or__
    """Called for other | self.
    
    Notes
    -----
    This is the reflected version of ``~metric.Unit.__or__``. It exists to
    support consistency comparisons between instances of ``~metric.Unit`` and
    objects of other types for which that comparison is meaningful, in cases
    where the other object is the left operand. The semantics are identical.
    """


def conversion(
    u0: typing.Union[str, Unit],
    u1: typing.Union[str, Unit],
    strict: bool=False,
) -> typing.Optional[float]:
    """Compute a numerical unit-conversion factor.

    This function provides a convenient shortcut for computing conversion
    factors via the `~metric.Conversion` class. It also silently returns rather
    than raising an exception when the conversion is not possible.

    Parameters
    ----------
    u0 : string or ~metric.Unit
        The unit from which to convert

    u1 : string or ~metric.Unit
        The unit to which to convert

    strict : bool, default=False
        If true, allow undefined conversions to raise
        `~metric.UnitConversionError`. The default behavior is to return
        ``None``.

    Returns
    -------
    float or None
        The numerical factor required to convert the representation of a
        quantity in unit `u0` to its representation in unit `u1`, if the
        conversion succeeds. The value of `strict` determines the behavior if
        the conversion fails.

    Examples
    --------
    As a trivial first example, suppose you have something that is 2 meters long
    and you want to convert its length to centimeters:

    >>> 2 * metric.conversion('m', 'cm')
    200.0

    Note that because meter and centimeter have the same base unit, this is
    equivalent to computing their ratio:

    >>> 2 * metric.ratio('m', 'cm')
    200.0

    Next, in order to convert an amount of energy flux from 'J / au^2' to 'eV /
    km^2', you must multiply by

    >>> metric.conversion('J / au^2', 'eV / km^2')
    278.8896829059863

    which is equivalent to multiplying by the factor to convert from 'J' to
    'eV', then dividing by the square of the factor to convert from 'au' to
    'km':

    >>> c0 = metric.conversion('J', 'eV')
    >>> c1 = metric.conversion('au', 'km')
    >>> c0
    6.241418050181001e+18
    >>> c1
    149597870.70000002
    >>> c0 / c1**2
    278.8896829059863

    See Also
    --------
    metric.Conversion
    """
    try:
        return float(Conversion(str(u0), str(u1)))
    except UnitConversionError as err:
        if strict:
            raise err


Instance = typing.TypeVar('Instance', bound='Dimension')


class Dimension(symbolic.Expression):
    """An symbolic expression representing a physical dimension."""

    @classmethod
    def fromunit(cls, unit: Unit, system: typing.Literal['mks', 'cgs']):
        """Create an instance from `unit` and `system`."""
        expression = symbolic.Expression('1')
        systems = set()
        for term in unit:
            named = NamedUnit(term.base)
            allowed = named.systems['allowed']
            dimension = (
                named.dimensions[system] if len(allowed) > 1
                else named.dimensions[allowed[0]]
            )
            expression *= symbolic.Expression(dimension) ** term.exponent
            systems.update(allowed)
        if system in systems:
            return cls(expression)
        raise ValueError(
            f"Can't define dimension of {unit!r} in {system!r}"
        ) from None

    def __init__(self, arg: symbolic.Expressable) -> None:
        super().__init__(arg)
        self._quantities = {}

    def quantities(self, system: str) -> typing.Set[str]:
        """The quantities with this dimension in `system`."""
        if system in self._quantities:
            return self._quantities[system]
        canonical = CANONICAL['dimensions'][system]
        found = {k for k, v in canonical.items() if v == self}
        self._quantities[system] = found
        return found


class Dimensions(typing.Mapping, iterables.ReprStrMixin):
    """A collection of symbolic expressions of metric dimensions."""

    @classmethod
    def fromunit(cls, unit: Unit):
        """Compute dimensions based on an instance of `~metric.Unit`."""
        guard = iterables.Guard(Dimension.fromunit)
        guard.catch(ValueError)
        init = {system: guard.call(unit, system) for system in SYSTEMS}
        return cls(**init)

    def __init__(
        self,
        common: symbolic.Expressable=None,
        **systems: symbolic.Expressable
    ) -> None:
        """Initialize from expressable quantities.
        
        Parameters
        ----------
        common : string or iterable or `~symbolic.Expression`
            The dimension to associate with all metric systems.

        **systems
            Zero or more key-value pairs in which the key is the name of a known
            metric system and the value is an object that can instantiate the
            `~symbolic.Expression` class. If present, each value will override
            `common` for the corresponding metric system.
        """
        self._objects = self._init_from(common, **systems)

    def _init_from(
        self,
        common,
        **systems
    ) -> typing.Dict[str, typing.Optional[Dimension]]:
        """Create dimension objects from arguments."""
        created = dict.fromkeys(SYSTEMS)
        default = Dimension(common) if common else None
        updates = {
            system: Dimension(expressable) if expressable else default
            for system, expressable in systems.items()
        }
        created.update(updates)
        if any(created.values()):
            return created
        raise TypeError(
            f"Can't instantiate {self.__class__!r}"
            f" from {common!r} and {systems!r}"
        ) from None

    def __len__(self) -> int:
        return len(self._objects)

    def __iter__(self) -> typing.Iterator:
        return iter(self._objects)

    def __getitem__(self, __k):
        key = str(__k).lower()
        if key in self._objects:
            return self._objects[key]
        raise KeyError(f"No dimension for {__k!r}") from None

    def __str__(self) -> str:
        return ', '.join(
            f"{k!r}: {str(v) if v else v!r}"
            for k, v in self._objects.items()
        )


class Properties(iterables.ReprStrMixin):
    """Canonical properties of a quantity within a metric system."""

    def __init__(self, system: str, unit: typing.Union[str, Unit]) -> None:
        self._system = system.lower()
        self._unit = Unit(unit)
        self._dimension = None

    @property
    def unit(self):
        """The canonical unit of this quantity in this metric system."""
        return self._unit

    @property
    def dimension(self):
        """The dimension of this quantity in this metric system."""
        if self._dimension is None:
            self._dimension = self.unit.dimensions[self._system]
        return self._dimension

    def __eq__(self, __o) -> bool:
        """True if two instances have equal units and dimensions."""
        try:
            equal = [
                getattr(self, attr) == getattr(__o, attr)
                for attr in ('unit', 'dimension')
            ]
            return all(equal)
        except AttributeError:
            return False

    def __str__(self) -> str:
        properties = ', '.join(
            f"{p}={str(getattr(self, p, None))!r}"
            for p in ['unit', 'dimension']
        )
        return f"{properties} [{self._system!r}]"


class QuantityError(Exception):
    """An error occurred in `~metric.Quantity`."""
    pass


Instance = typing.TypeVar('Instance', bound='Quantity')


class Quantity(iterables.ReprStrMixin):
    """A single metric quantity."""

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
        self.units = UNITS[self.name]
        self.dimensions = DIMENSIONS[self.name]
        cls._instances[name] = self
        return self

    def __getitem__(self, system: str):
        """Get this quantity's representation in the named metric system."""
        try:
            unit = self.units[system.lower()]
            return Properties(system, unit)
        except KeyError as err:
            raise QuantityError(
                f"No properties available for {self.name!r} in {system!r}"
            ) from err

    # NOTE: This is here because unit conversions are only defined within their
    # respective quantities, even though two quantities may have identical
    # conversions (e.g., frequency and vorticity).
    def convert(self, unit: str) -> _Converter:
        """Create a conversion object for `unit`."""
        return _Converter(unit, self.name)

    _attrs = ('dimensions', 'units')

    def __eq__(self, other) -> bool:
        """Called to test equality via self == other.
        
        Two instances of this class are either identical, in which case they are
        triviall equal, or they represent distinct quantities, in which they are
        not equal. In addition, an instance of this class will compare equal to
        its case-insensitive name.

        See Also
        --------
        ``~metric.Quantity.__or__`` : test equivalence between quantities.
        """
        if isinstance(other, str):
            return other.lower() == self.name
        return other is self

    def __or__(self, other) -> bool:
        """Called to test equivalence via self | other.
        
        Two metric quantities are equivalent if their dimensions are equal in a
        given metric system. This operation thus provides a way to compare
        unequal quantities to determine if they are have a physical
        correspondence. For example, energy and work are distinct quantities,
        but they have identical dimensions and are linked through the
        work-energy theorem.

        This operation is only defined between two instances of this class.
        """
        if isinstance(other, Quantity):
            for system in SYSTEMS:
                if self[system].dimension != other[system].dimension:
                    return False
            return True
        return NotImplemented

    def __str__(self) -> str:
        """A simplified representation of this object."""
        return self.name


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
        self.dimensions = CANONICAL['dimensions'][name]
        self.units = CANONICAL['units'][name]
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
        return unit.norm[self.name]

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

    def values(self) -> typing.ValuesView[Properties]:
        return super().values()

    def items(self) -> typing.ItemsView[str, Properties]:
        return super().items()

    def __str__(self) -> str:
        """A simplified representation of this object."""
        return str(self.name)


def reduction(unit: symbolic.Expressable, system: str=None):
    """Reduce the given unit expression, if possible.
    
    Notes
    -----
    This function is still experimental.
    """
    expression = symbolic.Expression(unit)
    decomposed = []
    for term in expression:
        try:
            # NOTE: `NamedUnit.reduce` can return `None`
            current = NamedUnit(term.base).reduce(system=system)
        except (UnitParsingError, SystemAmbiguityError):
            current = None
        if current:
            decomposed.append(current**term.exponent)
    if not decomposed:
        return
    result = decomposed[0]
    for other in decomposed[1:]:
        result *= other.scale
    return result

