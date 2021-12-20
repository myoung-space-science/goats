from pathlib import Path

import pytest

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

