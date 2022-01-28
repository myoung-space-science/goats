import pytest
import numpy

from goats.common import indexing
from goats.common import aliased
from goats.eprem import datasets


@pytest.fixture
def dataset(datadirs: dict):
    """An EPREM dataset."""
    datapath = datadirs['cone']['obs'] / 'obs000000.nc'
    return datasets.Dataset(datapath, 'mks')


def test_axes(dataset: datasets.Dataset):
    """Test the axis-indexing objects."""
    axes = dataset.axes
    cases = {
        'time': {
            'type': indexing.Coordinates,
            'length': 50,
            'test': {
                'user': (0.1, 0.3, 'day'),
                'indices': [0, 2],
                'values': [86400 * i for i in (0.1, 0.3)],
            },
        },
        'shell': {
            'type': indexing.Indices,
            'length': 2000,
            'test': {
                'user': (0, 2),
                'indices': [0, 2],
            },
        },
        'species': {
            'type': indexing.OrderedPairs,
            'length': 1,
            'test': {
                'user': ['H+'],
                'indices': [0],
                'values': ['H+'],
            },
        },
        'energy': {
            'type': indexing.Coordinates,
            'length': 20,
            'test': {
                'user': (1e-1, 1e2, 'MeV'),
                'indices': [0, 19],
                'values': [1.6022e-13 * i for i in (1e-1, 1e2)],
            },
        },
        'mu': {
            'type': indexing.Coordinates,
            'length': 8,
            'test': {
                'user': (-1.0, +1.0),
                'indices': [0, 7],
                'values': [-1.0, +1.0],
            },
        },
    }
    for name, expected in cases.items():
        axis = axes[name]
        full = axis()
        assert isinstance(full, expected['type'])
        assert len(full) == expected['length']
        test = expected['test']
        user = axis(*test['user'])
        assert list(user) == test['indices']
        if isinstance(user, indexing.OrderedPairs):
            assert list(user.values) == test['values']
        if isinstance(user, indexing.Coordinates):
            assert numpy.allclose(user.values, test['values'])
    name = 'energy'
    expected = cases['energy']
    species = axes['species']
    for s in species():
        axis = axes[name]
        full = axis(species=s)
        assert isinstance(full, expected['type'])
        assert len(full) == expected['length']
        test = expected['test']
        user = axis(*test['user'])
        assert list(user) == test['indices']
        assert numpy.allclose(user.values, test['values'])


def test_variables(dataset: datasets.Dataset):
    """Test the dataset variable objects."""
    T, S, P, E, M = 'time', 'shell', 'species', 'energy', 'mu'
    cases = {
        'time': {
            'axes': (T,),
            'unit': {'mks': 's', 'cgs': 's'},
            'aliases': ['t', 'times'],
        },
        'shell': {
            'axes': (S,),
            'unit': {'mks': '1', 'cgs': '1'},
            'aliases': ['shells'],
        },
        'mu': {
            'axes': (M,),
            'unit': {'mks': '1', 'cgs': '1'},
            'aliases': [
                'mus',
                'pitch angle', 'pitch-angle cosine',
                'pitch angles', 'pitch-angle cosines',
            ],
        },
        'mass': {
            'axes': (P,),
            'unit': {'mks': 'kg', 'cgs': 'g'},
            'aliases': ['m'],
        },
        'charge': {
            'axes': (P,),
            'unit': {'mks': 'C', 'cgs': 'statC'},
            'aliases': ['q'],
        },
        'egrid': {
            'axes': (P, E),
            'unit': {'mks': 'J', 'cgs': 'erg'},
            'aliases': ['energy', 'energies', 'E'],
        },
        'vgrid': {
            'axes': (P, E),
            'unit': {'mks': 'm/s', 'cgs': 'cm/s'},
            'aliases': ['speed', 'v', 'vparticle'],
        },
        'R': {
            'axes': (T, S),
            'unit': {'mks': 'm', 'cgs': 'cm'},
            'aliases': ['r', 'radius'],
        },
        'T': {
            'axes': (T, S),
            'unit': {'mks': 'rad', 'cgs': 'rad'},
            'aliases': ['theta'],
        },
        'P': {
            'axes': (T, S),
            'unit': {'mks': 'rad', 'cgs': 'rad'},
            'aliases': ['phi'],
        },
        'Br': {
            'axes': (T, S),
            'unit': {'mks': 'T', 'cgs': 'G'},
            'aliases': ['br'],
        },
        'Bt': {
            'axes': (T, S),
            'unit': {'mks': 'T', 'cgs': 'G'},
            'aliases': ['bt', 'Btheta', 'btheta'],
        },
        'Bp': {
            'axes': (T, S),
            'unit': {'mks': 'T', 'cgs': 'G'},
            'aliases': ['bp', 'Bphi', 'bphi'],
        },
        'Vr': {
            'axes': (T, S),
            'unit': {'mks': 'm/s', 'cgs': 'cm/s'},
            'aliases': ['vr'],
        },
        'Vt': {
            'axes': (T, S),
            'unit': {'mks': 'm/s', 'cgs': 'cm/s'},
            'aliases': ['vt', 'Vtheta', 'vtheta'],
        },
        'Vp': {
            'axes': (T, S),
            'unit': {'mks': 'm/s', 'cgs': 'cm/s'},
            'aliases': ['vp', 'Vphi', 'vphi'],
        },
        'Rho': {
            'axes': (T, S),
            'unit': {'mks': 'm^-3', 'cgs': 'cm^-3'},
            'aliases': ['rho'],
        },
        'Dist': {
            'axes': (T, S, P, E, M),
            'unit': {'mks': 's^3/m^6', 'cgs': 's^3/cm^6'},
            'aliases': ['dist', 'f'],
        },
    }
    variables = dataset.variables
    for name, expected in cases.items():
        variable = variables[name]
        assert variable.axes == expected['axes']
        assert variable.unit == expected['unit']['mks']
        assert variable.name == aliased.MappingKey(name, *expected['aliases'])
