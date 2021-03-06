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
    return pkgpath / 'data' / 'eprem'


@pytest.fixture
def datadirs(rootpath: pathlib.Path):
    """Directories containing datasets."""
    bases = (
        'cone',
        'wind',
    )
    names = (
        'obs',
        'p_obs',
        'flux',
    )
    return {
        base: {
            name: rootpath / base / name
            for name in names
        } for base in bases
    }

