import numbers
from typing import *

import numpy as np

from goats.common import iterables
from goats.common import quantities


# TODO: Redefine these items to allow constants that aren't tied to a unit
# system (e.g., pi). The most obvious solution is to swap the order of the
# system name with 'value' and 'unit'. That will mean writing the system name
# twice for the constants defined below but the flexibility may be worth it.
metadata = {
    'k': {
        'info': "Boltzmann's constant.",
        'mks': {'unit': 'J / K', 'value': 1.3807e-23},
        'cgs': {'unit': 'erg / K', 'value': 1.3807e-16},
    },
    'e': {
        'info': "Elementary charge.",
        'mks': {'unit': 'C', 'value': 1.6022e-19},
        'cgs': {'unit': 'statC', 'value': 4.8032e-10},
    },
    'me': {
        'info': "Electron mass.",
        'mks': {'unit': 'kg', 'value': 9.1094e-31},
        'cgs': {'unit': 'g', 'value': 9.1094e-28},
    },
    'mp': {
        'info': "Proton mass.",
        'mks': {'unit': 'kg', 'value': 1.6726e-27},
        'cgs': {'unit': 'g', 'value': 1.6726e-24},
    },
    'G': {
        'info': "Gravitational constant.",
        'mks': {'unit': 'm^3 / (s^2 * kg)', 'value': 6.6726e-11},
        'cgs': {'unit': 'dyn * cm^2 / g^2', 'value': 6.6726e-8},
    },
    'g': {
        'info': "Gravitational acceleration.",
        'mks': {'unit': 'm / s^2', 'value': 9.8067},
        'cgs': {'unit': 'cm / s^2', 'value': 9.8067e2},
    },
    'h': {
        'info': "Planck's constant.",
        'mks': {'unit': 'J * s', 'value': 6.6261e-34},
        'cgs': {'unit': 'erg * s', 'value': 6.6261e-27},
    },
    'c': {
        'info': "Speed of light in a vacuum.",
        'mks': {'unit': 'm / s', 'value': 2.9979e8},
        'cgs': {'unit': 'cm / s', 'value': 2.9979e10},
    },
    'epsilon0': {
        'info': "Permittivity of free space.",
        'mks': {'unit': 'F / m', 'value': 8.8542e-12},
        'cgs': {'unit': '1', 'value': 1.0},
    },
    'mu0': {
        'info': "Permeability of free space.",
        'mks': {'unit': 'H / m', 'value': 4*np.pi * 1e-7},
        'cgs': {'unit': '1', 'value': 1.0},
    },
    'Rinfinity': {
        'info': "Rydberg constant.",
        'mks': {'unit': '1 / m', 'value': 1.0974e7},
        'cgs': {'unit': '1 / cm', 'value': 1.0974e5},
    },
    'a0': {
        'info': "Bohr radius.",
        'mks': {'unit': 'm', 'value': 5.2918e-11},
        'cgs': {'unit': 'cm', 'value': 5.2918e-9},
    },
    're': {
        'info': "Classical electron radius.",
        'mks': {'unit': 'm', 'value': 2.8179e-15},
        'cgs': {'unit': 'cm', 'value': 2.8179e-13},
    },
    'alpha': {
        'info': "Fine structure constant.",
        'mks': {'unit': '1', 'value': 7.2974e-3},
        'cgs': {'unit': '1', 'value': 7.2974e-3},
    },
    'c1': {
        'info': "First radiation constant.",
        'mks': {'unit': 'W * m^2', 'value': 3.7418e-16},
        'cgs': {'unit': 'erg * cm^2 / s', 'value': 3.7418e-16},
    },
    'c2': {
        'info': "Second radiation constant.",
        'mks': {'unit': 'm * K', 'value': 1.4388e-2},
        'cgs': {'unit': 'cm * K', 'value': 1.4388},
    },
    'sigma': {
        'info': "Stefan-Boltzmann constant.",
        'mks': {'unit': 'W / (m^2 * K^4)', 'value': 5.6705e-8},
        'cgs': {'unit': '(erg / s) / (cm^2 * K^4)', 'value': 5.6705e-5},
    },
    'eV': {
        'info': "Energy associated with 1 eV.",
        'mks': {'unit': 'J', 'value': 1.6022e-19},
        'cgs': {'unit': 'erg', 'value': 1.6022e-12},
    },
    'amu': {
        'info': "Atomic mass unit.",
        'mks': {'unit': 'kg', 'value': 1.6605e-27},
        'cgs': {'unit': 'g', 'value': 1.6605e-24},
    },
    'au': {
        'info': "Astronomical unit.",
        'mks': {'unit': 'm', 'value': 1.495978707e11},
        'cgs': {'unit': 'cm', 'value': 1.495978707e13},
    },
}
metadata['H+'] = {
    'info': "First ionization energy of hydrogen.",
    **{
        k: {
            'unit': metadata['eV'][k]['unit'],
            'value': 13.6 * metadata['eV'][k]['value'],
        }
        for k in ('mks', 'cgs')
    }
}


class Constant(quantities.Scalar):
    """A single physical constant."""

    def __init__(
        self,
        value: float,
        unit: Union[str, quantities.Unit] = None,
        info: str=None,
    ) -> None:
        super().__init__(value, unit=unit)
        self.info = info

    @property
    def asscalar(self):
        """A new `Scalar` object equivalent to this constant."""
        return self._new()

    def _new(self, value=None, unit=None):
        """Create a new scalar from this instance.

        This method overloads `quantities.Scalar._new` to explicitly return a
        `quantities.Scalar` instance from an arithmetic operation because the
        original object's `info` attribute may no longer be relevant.
        """
        return quantities.Scalar(
            value or self.value,
            unit=(unit or self.unit),
        )

    def __str__(self) -> str:
        """A simplified representation of this object."""
        return f"{super().__str__()} : {self.info}"


class Constants(iterables.MappingBase):
    """A class to manage sets of physical constants."""
    def __init__(self, system: str) -> None:
        self.system = system.lower()
        self._mapping = metadata.copy()
        super().__init__(self._mapping)

    def __getitem__(self, name: str):
        """Create the named constant or raise an error."""
        if name in self._mapping:
            definition = self._mapping[name][self.system]
            value = definition['value']
            unit = definition['unit']
            info = self._mapping[name].get('info')
            return Constant(value, unit=unit, info=info)
        raise KeyError(name)

    def __repr__(self) -> str:
        """An unambiguous representation of this object."""
        return f"{self.__class__.__qualname__}({self.system})"


cgs = Constants('cgs')
mks = Constants('mks')


def show():
    """Print all defined physical constants."""
    for key, data in metadata.items():
        print(f"{key}: {data['info']}")
        for system in ('mks', 'cgs'):
            value = data[system]['value']
            unit = data[system]['unit']
            print(f"\t{system}: {value} [{unit}]")
        print()


if __name__ == '__main__':
    show()
