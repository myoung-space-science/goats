import argparse
import typing

import numpy

from goats.core import aliased
from goats.core import iterables


_constants = {
    'pi': {
        'info': "The ratio of a circle's circumference to its diameter.",
        'all': numpy.pi,
        'quantity': 'number',
    },
    'k': {
        'info': "Boltzmann's constant.",
        'mks': {'unit': 'J / K', 'value': 1.3807e-23},
        'cgs': {'unit': 'erg / K', 'value': 1.3807e-16},
        'quantity': 'energy / temperature'
    },
    'e': {
        'info': "Elementary charge.",
        'mks': {'unit': 'C', 'value': 1.6022e-19},
        'cgs': {'unit': 'statC', 'value': 4.8032e-10},
        'quantity': 'charge',
    },
    'me': {
        'info': "Electron mass.",
        'mks': {'unit': 'kg', 'value': 9.1094e-31},
        'cgs': {'unit': 'g', 'value': 9.1094e-28},
        'quantity': 'mass',
    },
    'mp': {
        'info': "Proton mass.",
        'mks': {'unit': 'kg', 'value': 1.6726e-27},
        'cgs': {'unit': 'g', 'value': 1.6726e-24},
        'quantity': 'mass',
    },
    'G': {
        'info': "Gravitational constant.",
        'mks': {'unit': 'm^3 / (s^2 * kg)', 'value': 6.6726e-11},
        'cgs': {'unit': 'dyn * cm^2 / g^2', 'value': 6.6726e-8},
        'quantity': 'force * area / mass^2',
    },
    'g': {
        'info': "Gravitational acceleration.",
        'mks': {'unit': 'm / s^2', 'value': 9.8067},
        'cgs': {'unit': 'cm / s^2', 'value': 9.8067e2},
        'quantity': 'acceleration',
    },
    'h': {
        'info': "Planck's constant.",
        'mks': {'unit': 'J * s', 'value': 6.6261e-34},
        'cgs': {'unit': 'erg * s', 'value': 6.6261e-27},
        'quantity': 'energy * time',
    },
    'c': {
        'info': "Speed of light in a vacuum.",
        'mks': {'unit': 'm / s', 'value': 2.9979e8},
        'cgs': {'unit': 'cm / s', 'value': 2.9979e10},
        'quantity': 'speed',
    },
    'epsilon0': {
        'info': "Permittivity of free space.",
        'mks': {'unit': 'F / m', 'value': 8.8542e-12},
        'cgs': {'unit': '1', 'value': 1.0},
        'quantity': 'permittivity',
    },
    'mu0': {
        'info': "Permeability of free space.",
        'mks': {'unit': 'H / m', 'value': 4*numpy.pi * 1e-7},
        'cgs': {'unit': '1', 'value': 1.0},
        'quantity': 'permeability',
    },
    'Rinfinity': {
        'info': "Rydberg constant.",
        'mks': {'unit': '1 / m', 'value': 1.0974e7},
        'cgs': {'unit': '1 / cm', 'value': 1.0974e5},
        'quantity': '1 / length',
    },
    'a0': {
        'info': "Bohr radius.",
        'mks': {'unit': 'm', 'value': 5.2918e-11},
        'cgs': {'unit': 'cm', 'value': 5.2918e-9},
        'quantity': 'length',
    },
    're': {
        'info': "Classical electron radius.",
        'mks': {'unit': 'm', 'value': 2.8179e-15},
        'cgs': {'unit': 'cm', 'value': 2.8179e-13},
        'quantity': 'length',
    },
    'alpha': {
        'info': "Fine structure constant.",
        'mks': {'unit': '1', 'value': 7.2974e-3},
        'cgs': {'unit': '1', 'value': 7.2974e-3},
        'quantity': 'number',
    },
    'c1': {
        'info': "First radiation constant.",
        'mks': {'unit': 'W * m^2', 'value': 3.7418e-16},
        'cgs': {'unit': 'erg * cm^2 / s', 'value': 3.7418e-16},
        'quantity': 'power * area',
    },
    'c2': {
        'info': "Second radiation constant.",
        'mks': {'unit': 'm * K', 'value': 1.4388e-2},
        'cgs': {'unit': 'cm * K', 'value': 1.4388},
        'quantity': 'length * temperature',
    },
    'sigma': {
        'info': "Stefan-Boltzmann constant.",
        'mks': {'unit': 'W / (m^2 * K^4)', 'value': 5.6705e-8},
        'cgs': {'unit': '(erg / s) / (cm^2 * K^4)', 'value': 5.6705e-5},
        'quantity': 'power / (area * temperature^4)',
    },
    'eV': {
        'info': "Energy associated with 1 eV.",
        'mks': {'unit': 'J', 'value': 1.6022e-19},
        'cgs': {'unit': 'erg', 'value': 1.6022e-12},
        'quantity': 'energy',
    },
    'amu': {
        'info': "Atomic mass unit.",
        'mks': {'unit': 'kg', 'value': 1.6605e-27},
        'cgs': {'unit': 'g', 'value': 1.6605e-24},
        'quantity': 'mass',
    },
    'au': {
        'info': "Astronomical unit.",
        'mks': {'unit': 'm', 'value': 1.495978707e11},
        'cgs': {'unit': 'cm', 'value': 1.495978707e13},
        'quantity': 'length',
    },
}
_constants['H+'] = {
    'info': "First ionization energy of hydrogen.",
    **{
        k: {
            'unit': _constants['eV'][k]['unit'],
            'value': 13.6 * _constants['eV'][k]['value'],
        }
        for k in ('mks', 'cgs')
    },
    'quantity': 'energy',
}
def _normalize(defined: typing.Dict[str, dict]):
    """Normalize constant definitions."""
    systems = ('mks', 'cgs')
    norm = {}
    for key, definition in defined.items():
        if 'all' in definition:
            value = definition.pop('all')
            new = {k: {'unit': '1', 'value': value} for k in systems}
            definition.update(new)
        norm[key] = definition
    return norm
CONSTANTS = aliased.Mapping(_normalize(_constants))


_elements = [
    {'symbol': 'H', 'name': 'Hydrogen', 'mass': 1.00797},
    {'symbol': 'He', 'name': 'Helium', 'mass': 4.00260},
    {'symbol': 'Li', 'name': 'Lithium', 'mass': 6.941},
    {'symbol': 'Be', 'name': 'Beryllium', 'mass': 9.01218},
    {'symbol': 'B', 'name': 'Boron', 'mass': 10.81},
    {'symbol': 'C', 'name': 'Carbon', 'mass': 12.011},
    {'symbol': 'N', 'name': 'Nitrogen', 'mass': 14.0067},
    {'symbol': 'O', 'name': 'Oxygen', 'mass': 15.9994},
    {'symbol': 'F', 'name': 'Fluorine', 'mass': 18.998403},
    {'symbol': 'Ne', 'name': 'Neon', 'mass': 20.179},
    {'symbol': 'Na', 'name': 'Sodium', 'mass': 22.98977},
    {'symbol': 'Mg', 'name': 'Magnesium', 'mass': 24.305},
    {'symbol': 'Al', 'name': 'Aluminum', 'mass': 26.98154},
    {'symbol': 'Si', 'name': 'Silicon', 'mass': 28.0855},
    {'symbol': 'P', 'name': 'Phosphorus', 'mass': 30.97376},
    {'symbol': 'S', 'name': 'Sulfur', 'mass': 32.06},
    {'symbol': 'Cl', 'name': 'Chlorine', 'mass': 35.453},
    {'symbol': 'K', 'name': 'Potassium', 'mass': 39.0983},
    {'symbol': 'Ar', 'name': 'Argon', 'mass': 39.948},
    {'symbol': 'Ca', 'name': 'Calcium', 'mass': 40.08},
    {'symbol': 'Sc', 'name': 'Scandium', 'mass': 44.9559},
    {'symbol': 'Ti', 'name': 'Titanium', 'mass': 47.90},
    {'symbol': 'V', 'name': 'Vanadium', 'mass': 50.9415},
    {'symbol': 'Cr', 'name': 'Chromium', 'mass': 51.996},
    {'symbol': 'Mn', 'name': 'Manganese', 'mass': 54.9380},
    {'symbol': 'Fe', 'name': 'Iron', 'mass': 55.847},
    {'symbol': 'Ni', 'name': 'Nickel', 'mass': 58.70},
    {'symbol': 'Co', 'name': 'Cobalt', 'mass': 58.9332},
    {'symbol': 'Cu', 'name': 'Copper', 'mass': 63.546},
    {'symbol': 'Zn', 'name': 'Zinc', 'mass': 65.38},
    {'symbol': 'Ga', 'name': 'Gallium', 'mass': 69.72},
    {'symbol': 'Ge', 'name': 'Germanium', 'mass': 72.59},
    {'symbol': 'As', 'name': 'Arsenic', 'mass': 74.9216},
    {'symbol': 'Se', 'name': 'Selenium', 'mass': 78.96},
    {'symbol': 'Br', 'name': 'Bromine', 'mass': 79.904},
    {'symbol': 'Kr', 'name': 'Krypton', 'mass': 83.80},
    {'symbol': 'Rb', 'name': 'Rubidium', 'mass': 85.4678},
    {'symbol': 'Sr', 'name': 'Strontium', 'mass': 87.62},
    {'symbol': 'Y', 'name': 'Yttrium', 'mass': 88.9059},
    {'symbol': 'Zr', 'name': 'Zirconium', 'mass': 91.22},
    {'symbol': 'Nb', 'name': 'Niobium', 'mass': 92.9064},
    {'symbol': 'Mo', 'name': 'Molybdenum', 'mass': 95.94},
    {'symbol': 'Tc', 'name': 'Technetium', 'mass': (98)},
    {'symbol': 'Ru', 'name': 'Ruthenium', 'mass': 101.07},
    {'symbol': 'Rh', 'name': 'Rhodium', 'mass': 102.9055},
    {'symbol': 'Pd', 'name': 'Palladium', 'mass': 106.4},
    {'symbol': 'Ag', 'name': 'Silver', 'mass': 107.868},
    {'symbol': 'Cd', 'name': 'Cadmium', 'mass': 112.41},
    {'symbol': 'In', 'name': 'Indium', 'mass': 114.82},
    {'symbol': 'Sn', 'name': 'Tin', 'mass': 118.69},
    {'symbol': 'Sb', 'name': 'Antimony', 'mass': 121.75},
    {'symbol': 'I', 'name': 'Iodine', 'mass': 126.9045},
    {'symbol': 'Te', 'name': 'Tellurium', 'mass': 127.60},
    {'symbol': 'Xe', 'name': 'Xenon', 'mass': 131.30},
    {'symbol': 'Cs', 'name': 'Cesium', 'mass': 132.9054},
    {'symbol': 'Ba', 'name': 'Barium', 'mass': 137.33},
    {'symbol': 'La', 'name': 'Lanthanum', 'mass': 138.9055},
    {'symbol': 'Ce', 'name': 'Cerium', 'mass': 140.12},
    {'symbol': 'Pr', 'name': 'Praseodymium', 'mass': 140.9077},
    {'symbol': 'Nd', 'name': 'Neodymium', 'mass': 144.24},
    {'symbol': 'Pm', 'name': 'Promethium', 'mass': (145)},
    {'symbol': 'Sm', 'name': 'Samarium', 'mass': 150.4},
    {'symbol': 'Eu', 'name': 'Europium', 'mass': 151.96},
    {'symbol': 'Gd', 'name': 'Gadolinium', 'mass': 157.25},
    {'symbol': 'Tb', 'name': 'Terbium', 'mass': 158.9254},
    {'symbol': 'Dy', 'name': 'Dysprosium', 'mass': 162.50},
    {'symbol': 'Ho', 'name': 'Holmium', 'mass': 164.9304},
    {'symbol': 'Er', 'name': 'Erbium', 'mass': 167.26},
    {'symbol': 'Tm', 'name': 'Thulium', 'mass': 168.9342},
    {'symbol': 'Yb', 'name': 'Ytterbium', 'mass': 173.04},
    {'symbol': 'Lu', 'name': 'Lutetium', 'mass': 174.967},
    {'symbol': 'Hf', 'name': 'Hafnium', 'mass': 178.49},
    {'symbol': 'Ta', 'name': 'Tantalum', 'mass': 180.9479},
    {'symbol': 'W', 'name': 'Tungsten', 'mass': 183.85},
    {'symbol': 'Re', 'name': 'Rhenium', 'mass': 186.207},
    {'symbol': 'Os', 'name': 'Osmium', 'mass': 190.2},
    {'symbol': 'Ir', 'name': 'Iridium', 'mass': 192.22},
    {'symbol': 'Pt', 'name': 'Platinum', 'mass': 195.09},
    {'symbol': 'Au', 'name': 'Gold', 'mass': 196.9665},
    {'symbol': 'Hg', 'name': 'Mercury', 'mass': 200.59},
    {'symbol': 'Tl', 'name': 'Thallium', 'mass': 204.37},
    {'symbol': 'Pb', 'name': 'Lead', 'mass': 207.2},
    {'symbol': 'Bi', 'name': 'Bismuth', 'mass': 208.9804},
    {'symbol': 'Po', 'name': 'Polonium', 'mass': (209)},
    {'symbol': 'At', 'name': 'Astatine', 'mass': (210)},
    {'symbol': 'Rn', 'name': 'Radon', 'mass': (222)},
    {'symbol': 'Fr', 'name': 'Francium', 'mass': (223)},
    {'symbol': 'Ra', 'name': 'Radium', 'mass': 226.0254},
    {'symbol': 'Ac', 'name': 'Actinium', 'mass': 227.0278},
    {'symbol': 'Pa', 'name': 'Protactinium', 'mass': 231.0359},
    {'symbol': 'Th', 'name': 'Thorium', 'mass': 232.0381},
    {'symbol': 'Np', 'name': 'Neptunium', 'mass': 237.0482},
    {'symbol': 'U', 'name': 'Uranium', 'mass': 238.029},
    {'symbol': 'Pu', 'name': 'Plutonium', 'mass': (242)},
    {'symbol': 'Am', 'name': 'Americium', 'mass': (243)},
    {'symbol': 'Bk', 'name': 'Berkelium', 'mass': (247)},
    {'symbol': 'Cm', 'name': 'Curium', 'mass': (247)},
    {'symbol': 'No', 'name': 'Nobelium', 'mass': (250)},
    {'symbol': 'Cf', 'name': 'Californium', 'mass': (251)},
    {'symbol': 'Es', 'name': 'Einsteinium', 'mass': (252)},
    {'symbol': 'Hs', 'name': 'Hassium', 'mass': (255)},
    {'symbol': 'Mt', 'name': 'Meitnerium', 'mass': (256)},
    {'symbol': 'Fm', 'name': 'Fermium', 'mass': (257)},
    {'symbol': 'Md', 'name': 'Mendelevium', 'mass': (258)},
    {'symbol': 'Lr', 'name': 'Lawrencium', 'mass': (260)},
    {'symbol': 'Rf', 'name': 'Rutherfordium', 'mass': (261)},
    {'symbol': 'Bh', 'name': 'Bohrium', 'mass': (262)},
    {'symbol': 'Db', 'name': 'Dubnium', 'mass': (262)},
    {'symbol': 'Sg', 'name': 'Seaborgium', 'mass': (263)},
]

ELEMENTS = iterables.Table(_elements)

_nucleons = {
    element['symbol']: round(element['mass'])
    for element in ELEMENTS
}

def elements(
    mass: typing.Union[int, typing.Iterable[int]],
    charge: typing.Union[int, typing.Iterable[int]],
) -> typing.List[str]:
    """The elemental species symbols, based on masses and charges."""
    _mass = numpy.array(mass, ndmin=1)
    _charge = numpy.array(charge, ndmin=1)
    if len(_mass) != len(_charge):
        message = (
            f"Length of mass ({len(_mass)})"
            f" must equal length of charge ({len(_charge)})"
        )
        raise TypeError(message)
    mass_idx = get_mass_indices(_nucleons, _mass)
    bases = [list(_nucleons.keys())[m] for m in mass_idx]
    signs = [('+' if i > 0 else '-') * abs(int(i)) for i in _charge]
    return [f"{b}{c}" for b, c in zip(bases, signs)]


class MassValueError(Exception):
    """The given mass does not correspond to a known element."""
    def __init__(self, value: int) -> None:
        self.value = value

    def __str__(self) -> str:
        return f"Cannot find an element with atomic mass {self.value}"


def get_mass_indices(
    nucleons: typing.Dict[str, int],
    targets: typing.Iterable,
) -> list:
    """Get the indices in `ELEMENTS` corresponding to the given masses."""
    def get_index(this: list, that: int):
        try:
            return this.index(that)
        except ValueError as err:
            raise MassValueError(that) from err

    values = list(nucleons.values())
    return [get_index(values, round(target)) for target in targets]


def _show_constants():
    """Print all defined physical constants."""
    for key, data in CONSTANTS.items(aliased=True):
        print(f"{key}: {data['info']}")
        for system in ('mks', 'cgs'):
            value = data[system]['value']
            unit = data[system]['unit']
            print(f"\t{system}: {value} [{unit}]")
        print()


def _show_elements():
    """Print all known chemical elements."""
    ELEMENTS.show(names=['symbol', 'name', 'mass'])


def show(*subsets):
    """Print information about physical quantities defined in this module."""
    printers = {
        'constants': _show_constants,
        'elements': _show_elements,
    }
    if not subsets:
        subsets = tuple(printers)
    for subset in subsets:
        if subset in printers:
            printers[subset]()
        else:
            print(f"Nothing to print for {subset!r}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=show.__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        '--constants',
        help=_show_constants.__doc__,
        action='store_true',
    )
    parser.add_argument(
        '--elements',
        help=_show_elements.__doc__,
        action='store_true',
    )
    args = vars(parser.parse_args())
    subsets = (k for k, v in args.items() if v)
    show(*subsets)


