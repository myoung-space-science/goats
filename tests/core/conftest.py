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
    'basic': {
        'dimensions': {
            'time': {'size': 0},
            'level': {'size': 0},
            'lat': {'size': 73},
            'lon': {'size': 144},
        },
        'variables': {
            'time': {
                'dimensions': ['time'],
            },
            'level': {
                'dimensions': ['level'],
            },
            'lat': {
                'dimensions': ['lat'],
            },
            'lon': {
                'dimensions': ['lon'],
            },
            'temp': {
                'dimensions': ['time', 'level', 'lat', 'lon'],
                'unit': 'K',
            },
        },
    },
}

