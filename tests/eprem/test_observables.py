"""
This module contains high-level tests of the observer/observable/observation framework. Note that some tests may take especially long to run.
"""

from pathlib import Path
from typing import *

import numpy as np
import pytest

from goats import common
from goats import eprem


def get_rootpath() -> Path:
    """The root path to test data.

    This function gets the current working directory (`cwd`) from the resolved
    file path rather than from `pathlib.Path().cwd()` because the latter
    returns the current working directory of the caller.
    """
    cwd = Path(__file__).expanduser().resolve().parent
    pkgpath = cwd.parent.parent
    return pkgpath / 'data' / 'eprem'


def get_stream():
    """Create a stream observer.

    This is separated out to allow developers to create a stream outside of the
    `stream` fixture. One example application may be adding a `__main__` section
    and calling simple plotting routines for visual end-to-end tests.
    """
    dataroot = get_rootpath()
    datadir = dataroot / 'cone' / 'obs'
    return eprem.Stream(name=0, directory=datadir)


@pytest.fixture
def stream():
    """Provide a stream observer via fixture."""
    return get_stream()


@pytest.fixture
def observables() -> Dict[str, dict]:
    """Information about each observable."""
    t, s, p, e, m = 'time', 'shell', 'species', 'energy', 'mu'
    return {
        'r': {'axes': (t, s)},
        'theta': {'axes': (t, s)},
        'phi': {'axes': (t, s)},
        'Br': {'axes': (t, s)},
        'Btheta': {'axes': (t, s)},
        'Bphi': {'axes': (t, s)},
        'Vr': {'axes': (t, s)},
        'Vtheta': {'axes': (t, s)},
        'Vphi': {'axes': (t, s)},
        'rho': {'axes': (t, s)},
        'dist': {'axes': (t, s, p, e, m)},
        'x': {'axes': (t, s)},
        'y': {'axes': (t, s)},
        'z': {'axes': (t, s)},
        'B': {'axes': (t, s)},
        'V': {'axes': (t, s)},
        'flow angle': {'axes': (t, s)},
        'div(V)': {'axes': (t, s)},
        'density ratio': {'axes': (t, s)},
        'rigidity': {'axes': (p, e)},
        'mean free path': {'axes': (t, s, p, e)},
        'acceleration rate': {'axes': (t, s, p, e)},
        'energy density': {'axes': (t, s, p)},
        'average energy': {'axes': (t, s, p)},
        'isotropic distribution': {'axes': (t, s, p, e)},
        'flux': {'axes': (t, s, p, e)},
        'fluence': {'axes': (s, p, e)},
        'integral flux': {'axes': (t, s, p)},
    }


def test_observable_access(
    stream: eprem.Stream,
    observables: Dict[str, dict],
) -> None:
    """Access all observables."""
    for name in observables:
        observable = stream[name]
        assert isinstance(observable, common.Observable)


def test_create_observation(
    stream: eprem.Stream,
    observables: Dict[str, dict],
) -> None:
    """Create default observation from each observable."""
    for name, expected in observables.items():
        observation = stream[name].observed
        assert isinstance(observation, common.Observation)
        assert all(axis in observation.indices for axis in expected['axes'])

