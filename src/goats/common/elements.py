from typing import *

from goats.common import iterables, quantities


_definitions = [
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

definitions = iterables.Table(_definitions)

_nucleons = {
    element['symbol']: round(element['mass'])
    for element in definitions
}

def symbols(mass: Iterable, charge: Iterable) -> List[str]:
    """The elemental species symbols, based on masses and charges."""
    _mass = list(iterables.Separable(mass))
    _charge = list(iterables.Separable(charge))
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


def get_mass_indices(nucleons: Dict[str, int], targets: Iterable) -> list:
    """Get the indices in `definitions` corresponding to the given masses."""
    def get_index(this: list, that: int):
        try:
            return this.index(that)
        except ValueError:
            raise MassValueError(that)

    values = list(nucleons.values())
    return [get_index(values, target) for target in targets]


class PlasmaSpecies:
    """A class to represent a single species in a plasma."""
    def __init__(
        self,
        symbol: str='',
        mass: float=None,
        charge: float=None,
    ) -> None:
        self._symbol = symbol
        self._mass = mass
        self._charge = charge
        if self._symbol is None and self._mass is None and self._charge is None:
            raise ValueError("Element is undefined")

    @property
    def symbol(self) -> str:
        """The elemental symbol of this species."""
        if self._symbol is None:
            s = symbols([self._mass], [self._charge])
            self._symbol = s[0]
        return self._symbol

    @property
    def mass(self) -> quantities.Measurement:
        """The mass of this species."""
        if self._mass is None:
            base = self._symbol.rstrip('+-')
            element = definitions.find(base, unique=True)
            unit = 'nucleon'
            self._mass = quantities.Measurement((element['mass'], unit))
        return self._mass

    @property
    def m(self):
        """Alias for mass."""
        return self.mass

    @property
    def charge(self) -> quantities.Measurement:
        """The charge of this species."""
        if self._charge is None:
            base = self._symbol.rstrip('+-')
            sign = self._symbol.lstrip(base)
            value = sum(float(f"{s}1.0") for s in sign)
            unit = 'e'
            self._charge = quantities.Measurement((value, unit))
        return self._charge

    @property
    def q(self):
        """Alias for charge."""
        return self.charge


def show():
    """Print all known chemical elements."""
    definitions.show(names=['symbol', 'name', 'mass'])


if __name__ == '__main__':
    show()

