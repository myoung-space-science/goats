import pathlib

import pytest


@pytest.fixture
def rootpath() -> pathlib.Path:
    """The root path to test data.

    This function gets the current working directory (`cwd`) from the resolved
    file path rather than from `pathlib.Path().cwd()` because the latter
    returns the current working directory of the caller.
    """
    cwd = pathlib.Path(__file__).expanduser().resolve().parent
    pkgpath = cwd.parent.parent
    return pkgpath / 'data' / 'core'


@pytest.fixture
def testdata(rootpath: pathlib.Path):
    """The available test datasets."""
    return {
        key: {
            'path': rootpath / f'{key}.nc',
            **dataset,
        } for key, dataset in _DATASETS.items()
    }


_DATASETS = {
    'eprem-obs': {
        'axes': {
            'time': {'size': 20},
            'shell': {'size': 100},
            'species': {'size': 2},
            'energy': {'size': 10},
            'mu': {'size': 5},
        },
        'variables': {
            'time': {
                'axes': ['time'],
                'unit': 's',
            },
            'Vr': {
                'axes': ['time', 'shell'],
                'unit': 'km / s',
            },
            'Dist': {
                'axes': ['time', 'shell', 'species', 'energy', 'mu'],
                'unit': 's^3 / km^6',
            },
        },
    },
    'eprem-flux': {
        'axes': {
            'time': {'size': 20},
            'shell': {'size': 100},
            'species': {'size': 2},
            'energy': {'size': 10},
            'mu': {'size': 5},
        },
        'variables': {
            'time': {
                'axes': ['time'],
                'unit': 's',
            },
            'Vr': {
                'axes': ['time', 'shell'],
                'unit': 'km / s',
            },
            'flux': {
                'axes': ['time', 'shell', 'species', 'energy'],
                'unit': '# / cm^2 s sr MeV',
            },
        },
    },
    'basic': {
        'axes': {
            'time': {'size': 0},
            'level': {'size': 0},
            'lat': {'size': 73},
            'lon': {'size': 144},
        },
        'variables': {
            'time': {
                'axes': ['time'],
            },
            'level': {
                'axes': ['level'],
            },
            'lat': {
                'axes': ['lat'],
            },
            'lon': {
                'axes': ['lon'],
            },
            'temp': {
                'axes': ['time', 'level', 'lat', 'lon'],
                'unit': 'K',
            },
        }
    }
}

