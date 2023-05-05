import pytest
import numpy

from goats.core import axis
from goats.core import index
from goats.core import datafile
from goats.core import metric
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
            'inputs': (0.1, 0.3, 'day'),
            'expected': {
                'points': [0, 2],
                'values': {
                    'mks': [86400 * i for i in (0.1, 0.3)],
                    'cgs': [86400 * i for i in (0.1, 0.3)],
                },
                'unit': {
                    'mks': 's',
                    'cgs': 's',
                },
            },
        },
        'shell': {
            'length': 2000,
            'inputs': (0, 2),
            'expected': {
                'points': [0, 2],
                'values': {
                    'mks': [0, 2],
                    'cgs': [0, 2],
                },
                'unit': {
                    'mks': '1',
                    'cgs': '1',
                },
            },
        },
        'species': {
            'length': 1,
            'inputs': ['H+'],
            'expected': {
                'points': [0],
                'values': {
                    'mks': ['H+'],
                    'cgs': ['H+'],
                },
                'unit': {
                    'mks': None,
                    'cgs': None,
                },
            },
        },
        'energy': {
            'length': 20,
            'inputs': (1e-1, 1e2, 'MeV'),
            'expected': {
                'points': [0, 19],
                'values': {
                    'mks': [1.6022e-13 * i for i in (1e-1, 1e2)],
                    'cgs': [1.6022e-6 * i for i in (1e-1, 1e2)],
                },
                'unit': {
                    'mks': 'J',
                    'cgs': 'erg',
                },
            },
        },
        'mu': {
            'length': 8,
            'inputs': (-1.0, +1.0),
            'expected': {
                'points': [0, 7],
                'values': {
                    'mks': [-1.0, +1.0],
                    'cgs': [-1.0, +1.0],
                },
                'unit': {
                    'mks': '1',
                    'cgs': '1',
                },
            },
        },
    }
    data = datafile.Interface(datapath)
    for name, parameters in cases.items():
        for system in metric.SYSTEMS:
            axes = eprem.Axes(data, system=system)
            this = axes[name]
            if name == 'energy':
                for species in axes['species'].index():
                    check_axis(this, parameters, system, species=species)
            else:
                check_axis(this, parameters, system)


def check_axis(this: axis.Quantity, parameters, system, **kwargs):
    """Helper for `test_axes`."""
    length = parameters['length']
    inputs = parameters['inputs']
    expected = parameters['expected']
    points = expected['points']
    unit = expected['unit'][system]
    values = expected['values'][system]
    full = this.index(**kwargs)
    assert isinstance(full, index.Quantity)
    assert len(full) == length
    user = this.index(*inputs, **kwargs)
    assert list(user) == points
    assert user[:] == tuple(points)
    assert user[-1] == points[-1]
    assert user.unit == unit
    if user.unit is not None:
        assert numpy.allclose(user.values, values)
    elif any(i != j for i, j in zip(user, user.values)):
        assert list(user.values) == values


def test_egrid_shape(datapath):
    """Test the ability to detect 1D or 2D `egrid` in the dataset.
    
    This test exists because we changed `egrid` to be 1D -- indexed only by
    species -- in April 2023. It had previously been logically 2D -- indexed by
    species and energy -- but the energy dimension was always singular, making
    it effectively 1D.
    """
    paths = [
        datapath,
        '~/emmrem/open/development/eprem/2023-05-05/cone-short/obs000000.nc',
    ]
    for path in paths:
        data = datafile.Interface(path)
        axes = eprem.Axes(data, system='mks')
        assert isinstance(axes['energy'].index(), index.Quantity)


def test_axis_unit(datapath):
    """Users should be able to update the default axis unit."""
    cases = {
        'time': {
            'points': [0, 2],
            'day': [0.1, 0.3],
            'hour': [24 * i for i in [0.1, 0.3]],
        },
        'energy': {
            'points': [0, 19],
            'MeV': [1e-1, 1e2],
            'eV': [1e6 * i for i in [1e-1, 1e2]],
            'erg': [1.6022e-6 * i for i in (1e-1, 1e2)],
        },
    }
    data = datafile.Interface(datapath)
    axes = eprem.Axes(data, system='mks')
    for name, test in cases.items():
        this = axes[name]
        points = test.pop('points')
        for unit, values in test.items():
            converted = this[unit]
            assert converted is not this # only if `unit` != MKS default
            assert converted.unit == unit
            indexed = converted.index(*values)
            assert indexed.unit == unit
            assert list(indexed) == points, f"{name} ({unit})"
            assert numpy.array_equal(indexed.values, values)


def test_axis_reference(datapath):
    """Users should be able to access reference values for an axis."""
    data = datafile.Interface(datapath)
    axes = eprem.Axes(data, system='mks')
    variables = eprem.Variables(data, system='mks')
    cases = {
        'time': {
            'reference': variables['time'],
            'units': ['day', 'hour'],
        },
        'shell': {
            'reference': variables['shell'],
        },
        'species': {
            'reference': ['H+'],
        },
        'energy': {
            'reference': numpy.squeeze(variables['energy'][0]),
            'units': ['eV', 'erg'],
        },
        'mu': {
            'reference': variables['mu'],
        },
    }
    for name, test in cases.items():
        this = axes[name]
        reference = test['reference']
        assert numpy.array_equal(this.reference, reference)
        for unit in test.get('units', []):
            assert numpy.array_equal(this[unit].reference, reference[unit])


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
    axes = eprem.Axes(data, system='mks')
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
    axes = eprem.Axes(data, system='mks')
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
    variables = eprem.Variables(data, system='mks')
    for name, expected in cases.items():
        current = variables[name]
        assert current.dimensions == expected['axes']
        assert current.unit == expected['unit']


def test_standardize():
    """Test the helper function that standardizes unit strings."""
    cases = {
        'julian date': 'day',
        'shell': '1',
        'cos(mu)': '1',
        'e-': 'e',
        '# / cm^2 s sr MeV': '# / (cm^2 s sr MeV/nuc)',
    }
    for old, new in cases.items():
        assert eprem.standardize(old) == new


