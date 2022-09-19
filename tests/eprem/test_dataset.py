import pytest
import numpy

from goats.core import aliased
from goats.core import axis
from goats.core import datafile
from goats.core import variable
from goats import eprem


@pytest.fixture
def datapath(datadirs: dict):
    """The path to an EPREM test dataset."""
    return datadirs['cone']['obs'] / 'obs000000.nc'


def test_axes(datapath):
    """Test the axis-indexing objects."""
    cases = {
        'time': {
            'length': 50,
            'test': {
                'user': (0.1, 0.3, 'day'),
                'indices': [0, 2],
                'values': [86400 * i for i in (0.1, 0.3)],
            },
        },
        'shell': {
            'length': 2000,
            'test': {
                'user': (0, 2),
                'indices': [0, 2],
            },
        },
        'species': {
            'length': 1,
            'test': {
                'user': ['H+'],
                'indices': [0],
                'values': ['H+'],
            },
        },
        'energy': {
            'length': 20,
            'test': {
                'user': (1e-1, 1e2, 'MeV'),
                'indices': [0, 19],
                'values': [1.6022e-13 * i for i in (1e-1, 1e2)],
            },
        },
        'mu': {
            'length': 8,
            'test': {
                'user': (-1.0, +1.0),
                'indices': [0, 7],
                'values': [-1.0, +1.0],
            },
        },
    }
    data = datafile.Interface(datapath)
    axes = eprem.Axes(data)
    for name, expected in cases.items():
        if name != 'energy':
            this = axes[name]
            full = this.index()
            assert isinstance(full, axis.Index)
            assert len(full) == expected['length']
            test = expected['test']
            user = this.index(*test['user'])
            assert list(user) == test['indices']
            if user.unit is not None:
                assert numpy.allclose(user.values, test['values'])
            elif any(i != j for i, j in zip(user, user.values)):
                assert list(user.values) == test['values']
    name = 'energy'
    expected = cases['energy']
    species = axes['species']
    for s in species.index():
        this = axes[name]
        full = this.index(species=s)
        assert isinstance(full, axis.Index)
        assert len(full) == expected['length']
        test = expected['test']
        user = this.index(*test['user'])
        assert list(user) == test['indices']
        assert numpy.allclose(user.values, test['values'])


def test_single_index(datapath):
    """Users should be able to provide a single numerical value."""
    cases = {
        'time': {
            'input': 8640.0,
            'index': [0],
            'value': [8640.0],
            'unit': 's',
        },
        'shell': {
            'input': 1,
            'index': [1],
        },
        'species': {
            'input': 'H+',
            'index': [0],
            'value': ['H+'],
        },
        'energy': {
            'input': 1.6022e-14,
            'index': [0],
            'value': [1.6022e-14],
            'unit': 'J',
        },
        'mu': {
            'input': -1.0,
            'index': [0],
            'value': [-1.0],
            'unit': '1',
        }
    }
    data = datafile.Interface(datapath)
    axes = eprem.Axes(data)
    for name, expected in cases.items():
        this = axes[name]
        result = this.index(expected['input'])
        assert list(result) == expected['index']
        if 'value' in expected:
            assert list(result.values) == expected['value']
        if 'unit' in expected:
            assert result.unit == expected['unit']


def test_resolve_axes(datapath):
    """Test the method that orders EPREM axes."""
    # This is only a subset of the possible cases and there's probably a more
    # efficient way to build the collection.
    cases = [
        {
            'input': ('shell', 'energy', 'time'),
            'output': ('time', 'shell', 'energy'),
        },
        {
            'input': ('shell', 'energy', 'time', 'extra'),
            'output': ('time', 'shell', 'energy'),
        },
        {
            'input': ('shell', 'energy', 'time'),
            'mode': 'strict',
            'output': ('time', 'shell', 'energy'),
        },
        {
            'input': ('shell', 'energy', 'time', 'extra'),
            'mode': 'strict',
            'output': ('time', 'shell', 'energy'),
        },
        {
            'input': ('shell', 'energy', 'time', 'extra'),
            'mode': 'append',
            'output': ('time', 'shell', 'energy', 'extra'),
        },
        {
            'input': ('extra', 'shell', 'energy', 'time'),
            'mode': 'append',
            'output': ('time', 'shell', 'energy', 'extra'),
        },
        {
            'input': ('shell', 'extra', 'energy', 'time'),
            'mode': 'append',
            'output': ('time', 'shell', 'energy', 'extra'),
        },
    ]
    data = datafile.Interface(datapath)
    axes = eprem.Axes(data)
    for case in cases:
        names = case['input']
        expected = case['output']
        result = (
            axes.resolve(names, mode=case['mode']) if 'mode' in case
            else axes.resolve(names)
        )
        assert result == expected


def test_variables(datapath):
    """Test the dataset variable objects."""
    T, S, P, E, M = 'time', 'shell', 'species', 'energy', 'mu'
    cases = {
        'time': {
            'axes': (T,),
            'unit': 's',
            'aliases': ['t', 'times'],
        },
        'shell': {
            'axes': (S,),
            'unit': '1',
            'aliases': ['shells'],
        },
        'mu': {
            'axes': (M,),
            'unit': '1',
            'aliases': [
                'mu', 'mus',
                'pitch angle', 'pitch-angle', 'pitch-angle cosine',
                'pitch angles', 'pitch-angles', 'pitch-angle cosines',
            ],
        },
        'phiOffset': {
            'axes': (T,),
            'unit': 's',
            'aliases': [],
        },
        'mass': {
            'axes': (P,),
            'unit': 'kg',
            'aliases': ['m'],
        },
        'charge': {
            'axes': (P,),
            'unit': 'C',
            'aliases': ['q'],
        },
        'egrid': {
            'axes': (P, E),
            'unit': 'J',
            'aliases': ['energy', 'energies', 'E'],
        },
        'vgrid': {
            'axes': (P, E),
            'unit': 'm/s',
            'aliases': ['speed', 'vparticle'],
        },
        'R': {
            'axes': (T, S),
            'unit': 'm',
            'aliases': ['r', 'radius'],
        },
        'T': {
            'axes': (T, S),
            'unit': 'rad',
            'aliases': ['theta'],
        },
        'P': {
            'axes': (T, S),
            'unit': 'rad',
            'aliases': ['phi'],
        },
        'Br': {
            'axes': (T, S),
            'unit': 'T',
            'aliases': ['br'],
        },
        'Bt': {
            'axes': (T, S),
            'unit': 'T',
            'aliases': ['bt', 'Btheta', 'btheta'],
        },
        'Bp': {
            'axes': (T, S),
            'unit': 'T',
            'aliases': ['bp', 'Bphi', 'bphi'],
        },
        'Vr': {
            'axes': (T, S),
            'unit': 'm/s',
            'aliases': ['vr'],
        },
        'Vt': {
            'axes': (T, S),
            'unit': 'm/s',
            'aliases': ['vt', 'Vtheta', 'vtheta'],
        },
        'Vp': {
            'axes': (T, S),
            'unit': 'm/s',
            'aliases': ['vp', 'Vphi', 'vphi'],
        },
        'Rho': {
            'axes': (T, S),
            'unit': 'm^-3',
            'aliases': ['rho'],
        },
        'Dist': {
            'axes': (T, S, P, E, M),
            'unit': 's^3/m^6',
            'aliases': ['dist', 'f'],
        },
    }
    data = datafile.Interface(datapath)
    variables = variable.Interface(data)
    for name, expected in cases.items():
        current = variables[name]
        assert current.axes == expected['axes']
        assert current.unit == expected['unit']
        key = aliased.MappingKey(name, *expected['aliases'])
        assert current.name == key


