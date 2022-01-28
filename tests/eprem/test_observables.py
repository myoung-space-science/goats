"""
This module contains high-level tests of the observer/observable/observation framework. Note that some tests may take especially long to run.
"""

from pathlib import Path
from typing import *

import numpy as np
import pytest

from goats.common import base
from goats.eprem import observing


def get_stream(rootpath: Path):
    """Create a stream observer.

    This is separated out to allow developers to create a stream outside of the
    `stream` fixture. One example application may be adding a `__main__` section
    and calling simple plotting routines for visual end-to-end tests.
    """
    datadir = rootpath / 'cone' / 'obs'
    return observing.Stream(name=0, path=datadir)


@pytest.fixture
def stream(rootpath: Path):
    """Provide a stream observer via fixture."""
    return get_stream(rootpath)


@pytest.fixture
def observables() -> Dict[str, dict]:
    """Information about each observable."""
    T, S, P, E, M = 'time', 'shell', 'species', 'energy', 'mu'
    return {
        'r': {'axes': (T, S)},
        'theta': {'axes': (T, S)},
        'phi': {'axes': (T, S)},
        'Br': {'axes': (T, S)},
        'Btheta': {'axes': (T, S)},
        'Bphi': {'axes': (T, S)},
        'Vr': {'axes': (T, S)},
        'Vtheta': {'axes': (T, S)},
        'Vphi': {'axes': (T, S)},
        'rho': {'axes': (T, S)},
        'dist': {'axes': (T, S, P, E, M)},
        'x': {'axes': (T, S)},
        'y': {'axes': (T, S)},
        'z': {'axes': (T, S)},
        'B': {'axes': (T, S)},
        'V': {'axes': (T, S)},
        'flow angle': {'axes': (T, S)},
        'div(V)': {'axes': (T, S)},
        'density ratio': {'axes': (T, S)},
        'rigidity': {'axes': (P, E)},
        'mean free path': {'axes': (T, S, P, E)},
        'acceleration rate': {'axes': (T, S, P, E)},
        'energy density': {'axes': (T, S, P)},
        'average energy': {'axes': (T, S, P)},
        'isotropic distribution': {'axes': (T, S, P, E)},
        'flux': {'axes': (T, S, P, E)},
        'fluence': {'axes': (S, P, E)},
        'integral flux': {'axes': (T, S, P)},
        'Vr / Br': {'axes': (T, S)},
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
        observation = stream[name].observe()
        assert isinstance(observation, base.Observation)
        assert all(axis in observation.indices for axis in expected['axes'])


def test_reset_constraints(stream: observing.Stream):
    """Test the ability to reset observing constraints."""
    observable = stream['dist']
    observation = observable.observe(
        time=[0.1, 0.3, 'day'],
        shell=[10, 11, 12, 13, 14],
        energy=[0.1, 1.0, 5.0, 'MeV'],
        mu=(-1.0, -0.5, 0.5, 1.0),
    )
    assert np.array(observation).shape == (2, 5, 1, 3, 4)
    observation = observable.observe()
    assert np.array(observation).shape == (2, 5, 1, 3, 4)
    observation = observable.observe(time=[0.2, 0.4, 0.5, 'day'])
    assert np.array(observation).shape == (3, 5, 1, 3, 4)
    observable.reset()
    observation = observable.observe()
    assert np.array(observation).shape == (50, 2000, 1, 20, 8)


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


def test_repeated_access(stream: observing.Stream):
    """Test repeatedly accessing the same observable."""
    a = stream['br']
    b = stream['br']
    assert a == b
    assert a is not b
