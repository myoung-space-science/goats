"""
This module contains high-level tests of the observer/observable/observation framework. Note that some tests may take especially long to run.
"""

from pathlib import Path
from typing import *

import numpy as np
import pytest

from goats.common import base
from goats.eprem import observing

observing.env.source('~/emmrem/epicMas/source/eprem/src')


def get_stream(rootpath: Path):
    """Create a stream observer.

    This is separated out to allow developers to create a stream outside of the
    `stream` fixture. One example application may be adding a `__main__` section
    and calling simple plotting routines for visual end-to-end tests.
    """
    datadir = rootpath / 'cone' / 'obs'
    return observing.Stream(name=0, directory=datadir)


@pytest.fixture
def stream(rootpath: Path):
    """Provide a stream observer via fixture."""
    return get_stream(rootpath)


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
    stream: observing.Stream,
    observables: Dict[str, dict],
) -> None:
    """Access all observables."""
    for name in observables:
        observable = stream[name]
        assert isinstance(observable, base.Observable)


def test_create_observation(
    stream: observing.Stream,
    observables: Dict[str, dict],
) -> None:
    """Create default observation from each observable."""
    for name, expected in observables.items():
        observation = stream[name].observed
        assert isinstance(observation, base.Observation)
        assert all(axis in observation.indices for axis in expected['axes'])


def test_observable_aliases(stream: observing.Stream):
    """Explicitly check a few aliases to prevent regression."""
    tests = {
        'r': ['R'],
        'x': ['X'],
        'mean_free_path': ['mfp', 'mean free path'],
        'div_v': ['div(V)', 'divV'],
        'fluence': [],
    }
    for name, aliases in tests.items():
        observable = stream[name]
        for alias in aliases:
            assert stream[alias].name == observable.name

