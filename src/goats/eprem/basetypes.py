"""
Constants defined in EPREM `src/baseTypes.h`.
"""
import pathlib
import numbers
import typing

from goats.common import algebra
from goats.common import constants
from goats.common import iotools
from goats.common import iterables
from goats.common import numerical


_BASETYPES_H = {
    'T': {
        'info': 'True',
        'type': int,
    },
    'F': {
        'info': 'False',
        'type': int,
    },
    'PI': {
        'info': 'The value of π.',
        'type': float,
    },
    'TWO_PI': {
        'info': 'The value of 2π.',
        'type': float,
    },
    'VERYSMALL': {
        'info': 'A very small value.',
        'type': float,
    },
    'BADVALUE': {
        'info': 'A bad (invalid) float value.',
        'type': float,
    },
    'BADINT': {
        'info': 'A bad (invalid) integer value.',
        'type': int,
    },
    'MP': {
        'info': 'The proton mass.',
        'unit': 'g',
        'type': float,
    },
    'EV': {
        'info': 'The conversion from eVs to ergs.',
        'unit': 'erg/eV',
        'type': float,
    },
    'MEV': {
        'info': 'The conversion from MeVs to ergs.',
        'unit': 'erg/MeV',
        'type': float,
    },
    'GEV': {
        'info': 'The conversion from GeVs to ergs.',
        'unit': 'erg/GeV',
        'type': float,
    },
    'Q': {
        'info': 'The proton charge.',
        'unit': 'statC',
        'type': float,
    },
    'C': {
        'info': 'The speed of light.',
        'unit': 'cm/s',
        'type': float,
    },
    'MZERO': {
        'info': 'The proton rest-mass energy in GeV.',
        'unit': 'GeV',
        'formula': 'MP * C^2 * GEV^-1',
        'type': float,
    },
    'AU': {
        'info': 'One astronomical unit.',
        'unit': 'cm',
        'type': float,
    },
    'RSUN': {
        'info': 'The value of the solar radius.',
        'unit': 'cm',
        'type': float,
    },
    'RSAU': {
        'info': 'The number of solar radii per au.',
        'formula': 'RSUN * AU^-1',
        'type': float,
    },
    'TAU': {
        'info': 'The canonical EPREM time scale.',
        'unit': 's',
        'formula': 'AU * C^-1',
        'type': float,
    },
    'DAY': {
        'info': 'The conversion from EPREM time steps to Julian days.',
        'unit': 'day',
        'formula': 'TAU * 1.1574074074074072e-05', 
        'type': float,
    },
    'MHD_DENSITY_NORM': {
        'info': 'The normalization factor for density.',
        'type': float,
    },
    'MHD_B_NORM': {
        'info': 'The normalization for magnetic fields.',
        'formula': 'MP^0.5 * MHD_DENSITY_NORM^0.5 * C',
        'type': float,
    },
    'OM': {
        'info': 'The normalization for ion gyrofrequency.',
        'formula': 'MHD_B_NORM * Q * AU * MP^-1 * C^-2',
        'type': float,
    },
    'FCONVERT': {
        'info': 'The conversion from distribution to flux.',
        'formula': 'C^-4 * 1e+30',
        'type': float,
    },
    'VOLT': {
        'info': 'The conversion from volts to statvolts.',
        'type': float,
    },
    'THRESH': {
        'info': 'The threshold for perpendicular diffusion.',
        'type': float,
    },
    'MAS_TIME_NORM': {
        'info': 'The MAS time normalization factor.',
        'type': float,
    },
    'MAS_LENGTH_NORM': {
        'info': 'The MAS length normalization factor.',
        'type': float,
    },
    'MAS_RHO_NORM': {
        'info': 'The MAS plasma-density normalization.',
        'type': float,
    },
    'MAS_TIME_CONVERT': {
        'info': 'The time conversion from MAS units.',
        'formula': 'MAS_TIME_NORM * 1.1574074074074072e-05',
        'type': float,
    },
    'MAS_V_CONVERT': {
        'info': 'The velocity conversion from MAS units.',
        'formula': 'MAS_LENGTH_NORM * MAS_TIME_NORM^-1 * C^-1',
        'type': float,
    },
    'MAS_RHO_CONVERT': {
        'info': 'The density conversion from MAS units.',
        'formula': 'MAS_RHO_NORM * MP^-1 * MHD_DENSITY_NORM^-1',
        'type': float,
    },
    'MAS_B_CONVERT': {
        'info': 'The magnetic field conversion from MAS units.',
        'formula': (
            'PI^0.5 * MAS_RHO_NORM^0.5 * MAS_LENGTH_NORM * MAS_TIME_NORM^-1 '
            '* MHD_B_NORM^-1 * 2.0'
        ),
        'type': float,
    },
    'MAX_STRING_SIZE': {
        'type': int,
    },
    'MHD_DEFAULT': {
        'info': 'Use the default MHD solver.',
        'type': int,
    },
    'MHD_ENLIL': {
        'info': 'Use ENLIL for MHD values.',
        'type': int,
    },
    'MHD_LFMH': {
        'info': 'Use LFM for MHD values.',
        'type': int,
    },
    'MHD_BATSRUS': {
        'info': 'Use BATS-R-US for MHD values.',
        'type': int,
    },
    'MHD_MAS': {
        'info': 'Use MAS for MHD values.',
        'type': int,
    },
    'NUM_MPI_BOUNDARY_FLDS': {
        'info': 'Number of MPI psuedo-fields for use in creating MPI typedefs.',
        'type': int,
    },
}


class BaseTypes(iterables.MappingBase):
    """Constants defined in EPREM `src/baseTypes.h`."""

    def __init__(self, src: typing.Union[str, pathlib.Path]=None) -> None:
        self.path = iotools.ReadOnlyPath(src or '.') / 'baseTypes.h'
        self._definitions = None
        self._mapping = {k: self._evaluate(k) for k in self.definitions}
        super().__init__(self._mapping)

    def __getitem__(self, name: str):
        """Access constants by name."""
        if name in self._mapping:
            value = self._mapping[name]
            reference = _BASETYPES_H.get(name)
            unit = reference.get('unit')
            info = reference.get('info')
            return constants.Constant(value, unit=unit, info=info)
        raise KeyError(f"No {name!r} in baseTypes.h")

    @property
    def definitions(self):
        """The definition of each constant in the simulation source code."""
        if self._definitions is None:
            with self.path.open('r') as fp:
                lines = fp.readlines()
            definitions = {}
            for line in lines:
                if line.startswith('#define'):
                    parts = line.strip('#define').rstrip('\n').split(maxsplit=1)
                    if len(parts) == 2:
                        key, value = parts
                        metadata = _BASETYPES_H.get(key, {})
                        if 'formula' in metadata:
                            definitions[key] = algebra.Expression(value)
                        else:
                            cast = metadata.get('type', str)
                            definitions[key] = cast(value)
            self._definitions = definitions
        return self._definitions

    def _evaluate(self, key: str) -> numbers.Real:
        """Compute the value of a defined constant."""
        target = self.definitions[key]
        if isinstance(target, numbers.Real):
            return target
        if isinstance(target, algebra.Expression):
            value = 1.0
            for term in target:
                if term.base in self.definitions:
                    value *= float(term(self._evaluate(term.base)))
                elif term.base == '1':
                    value *= float(term)
            return value
        raise TypeError(target)

    def print(self, tabsize: int=4, stream: typing.TextIO=None):
        """Print formatted reference information."""
        indent = ' ' * tabsize
        print("{", file=stream)
        for key, definition in self.definitions.items():
            formula = (
                definition.format(separator=' * ')
                if isinstance(definition, algebra.Expression)
                else None
            )
            print(
                f"{indent}{key!r}: {'{'}\n"
                f"{indent}{indent}'value': {self[key]},\n"
                f"{indent}{indent}'formula': {formula!r},\n"
                f"{indent}{'},'}",
                file=stream,
            )
        print("}", file=stream)

