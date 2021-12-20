from pathlib import Path

import pytest
import numpy

from goats.common import indexing
from goats.eprem import datasets


def get_rootpath() -> Path:
    """The root path to test data.

    This function gets the current working directory (`cwd`) from the resolved
    file path rather than from `pathlib.Path().cwd()` because the latter
    returns the current working directory of the caller.
    """
    cwd = Path(__file__).expanduser().resolve().parent
    pkgpath = cwd.parent.parent
    return pkgpath / 'data' / 'eprem'


@pytest.fixture
def dataset():
    """An EPREM dataset."""
    dataroot = get_rootpath()
    datapath = dataroot / 'cone' / 'obs' / 'obs000000.nc'
    return datasets.DatasetView(datapath, 'mks')


def test_axes(dataset: datasets.DatasetView):
    """Test the axis-indexing objects."""
    axes = datasets.Axes(dataset)
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


