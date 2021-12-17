"""
Constants defined in EPREM `src/baseTypes.h`.
"""
import numpy as np

from goats.common import constants
from goats.common import iterables


_primary = {
    'PI': {
        'info': 'The value of π.',
        'unit': '1',
        'value': 3.1415926535897932,
    },
    'TWO_PI': {
        'info': 'The value of 2π.',
        'unit': '1',
        'value': 6.283185307179586,
    },
    'VERYSMALL': {
        'info': 'A very small value.',
        'unit': '1',
        'value': 1.0e-33,
    },
    'BADVALUE': {
        'info': 'A bad (invalid) float value.',
        'unit': '1',
        'value': 1.0e33,
    },
    'BADINT': {
        'info': 'A bad (invalid) integer value.',
        'unit': '1',
        'value': 2147483647,
    },
    'MP': {
        'info': 'The value of the proton mass.',
        'unit': 'g',
        'value': 1.6726e-24,
    },
    'EV': {
        'info': 'The conversion from eVs to ergs.',
        'unit': 'erg/eV',
        'value': 1.6022e-12,
    },
    'MEV': {
        'info': 'The conversion from MeVs to ergs.',
        'unit': 'erg/MeV',
        'value': 1.6022e-6,
    },
    'GEV': {
        'info': 'The conversion from GeVs to ergs.',
        'unit': 'erg/GeV',
        'value': 1.6022e-3,
    },
    'Q': {
        'info': 'The value of the proton charge.',
        'unit': 'statcoul',
        'value': 4.80320425e-10,
    },
    'C': {
        'info': 'The value of the speed of light.',
        'unit': 'cm/s',
        'value': 2.99792458e10,
    },
    'AU': {
        'info': 'The value of 1 au.',
        'unit': 'cm',
        'value': 1.495978707e13,
    },
    'RSUN': {
        'info': 'The value of the solar radius.',
        'unit': 'cm',
        'value': 6.96e10,
    },
    'MHD_DENSITY_NORM': {
        'info': 'The normalization factor for density.',
        'unit': '1',
        'value': 1.0,
    },
    'VOLT': {
        'info': 'The conversion from volts to statvolts.',
        'unit': 'statvolt/V',
        'value': 0.33333e-2,
    },
    'THRESH': {
        'info': 'The threshold for perpendicular diffusion.',
        'unit': '1',
        'value': 0.025,
    },
    'MAS_TIME_NORM': {
        'info': 'The MAS time normalization.',
        'unit': '1',
        'value': 1445.87003080685,
    },
    'MAS_LENGTH_NORM': {
        'info': 'The MAS length normalization.',
        'unit': '1',
        'value': 6.96e10,
    },
    'MAS_RHO_NORM': {
        'info': 'The MAS plasma-density normalization.',
        'unit': '1',
        'value': 1.6726e-16,
    },
    'NUM_FACES': {
        'info': 'The number of faces in the EPREM logical grid.',
        'unit': '1',
        'value': 6,
    },
}

b = {k: v['value'] for k, v in _primary.items()}
_derived = {
    'MZERO': {
        'info': 'The proton rest-mass energy in GeV.',
        'unit': 'GeV',
        'value': b['MP'] * b['C']**2 / b['GEV'],
    },
    'RSAU': {
        'info': 'The solar radius in au.',
        'unit': 'Rs/au',
        'value': b['RSUN'] / b['AU'],
    },
    'TAU': {
        'info': 'The EPREM time scale in seconds.',
        'unit': 's',
        'value': b['AU'] / b['C'],
    },
}
d = {k: v['value'] for k, v in _derived.items()}
_derived.update({
    'DAY': {
        'info': 'The conversion from EPREM time steps to Julian days.',
        'unit': 'days',
        'value': d['TAU'] / (24.0 * 60.0 * 60.0),
    },
    'MHD_B_NORM': {
        'info': 'The normalization for magnetic fields.',
        'unit': '1',
        'value': np.sqrt(b['MP'] * b['MHD_DENSITY_NORM'])* b['C'],
    },
})
d = {k: v['value'] for k, v in _derived.items()}
_derived.update({
    'OM': {
        'info': 'The normalization for ion gyrofrequency.',
        'unit': '1',
        'value': (
            d['MHD_B_NORM'] * b['Q'] * b['AU'] / (b['MP'] * b['C']**2)
        ),
    },
    'FCONVERT': {
        'info': 'The conversion from distribution to flux.',
        'unit': '1',
        'value': 1.0e30 / b['C']**4,
    },
    'MAS_TIME_CONVERT': {
        'info': 'The time conversion from MAS unit.',
        'unit': '1',
        'value': b['MAS_TIME_NORM'] / (24.0 * 60.0 * 60.0),
    },
    'MAS_V_CONVERT': {
        'info': 'The velocity conversion from MAS unit.',
        'unit': '1',
        'value': (
            (b['MAS_LENGTH_NORM'] / b['MAS_TIME_NORM']) / b['C']
        ),
    },
    'MAS_RHO_CONVERT': {
        'info': 'The density conversion from MAS unit.',
        'unit': '1',
        'value': (b['MAS_RHO_NORM'] / b['MP']) / b['MHD_DENSITY_NORM'],
    },
    'MAS_B_CONVERT': {
        'info': 'The magnetic field conversion from MAS unit.',
        'unit': '1',
        'value': (
            np.sqrt(4.0 * np.pi * b['MAS_RHO_NORM'])
            * b['MAS_LENGTH_NORM'] / b['MAS_TIME_NORM']
            / d['MHD_B_NORM']
        ),
    },
})

metadata = {**_primary, **_derived}
"""Metadata for all base-type constants defined here."""


collection = iterables.UniformMapping(metadata, constants.Constant)

