"""
This module contains high-level tests of the observer/observable/observation framework. Note that some tests may take especially long to run.
"""

from pathlib import Path
import typing

import numpy
import matplotlib.pyplot as plt
import pytest

from goats.core import constant
from goats.core import fundamental
from goats.core import index
from goats.core import observable
from goats.core import observed
from goats.core import observer
from goats import eprem


def get_stream(rootpath: Path):
    """Create a stream observer.

    This is separated out to allow developers to create a stream outside of the
    `stream` fixture. One example application may be adding a `__main__` section
    and calling simple plotting routines for visual end-to-end tests.
    """
    datadir = rootpath / 'cone' / 'obs'
    return eprem.Stream(name=0, path=datadir)


@pytest.fixture
def stream(rootpath: Path):
    """Provide a stream observer via fixture."""
    return get_stream(rootpath)


@pytest.fixture
def observables() -> typing.Dict[str, dict]:
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


def test_init_stream(rootpath: Path):
    """Attempt to initialize a stream observer with various arguments."""
    datadir = rootpath / 'cone' / 'obs'
    # from ID and directory
    stream = eprem.Stream(0, path=datadir)
    assert isinstance(stream, observer.Interface)
    # from full path
    stream = eprem.Stream(path=datadir / 'obs000000.nc')
    assert isinstance(stream, observer.Interface)
    # from only ID, with default path
    stream = eprem.Stream(0)
    assert isinstance(stream, observer.Interface)


def test_observable_access(
    stream: eprem.Stream,
    observables: typing.Dict[str, dict],
) -> None:
    """Access all observables."""
    for name in observables:
        quantity = stream[name]
        assert isinstance(quantity, observable.Quantity)


def test_create_observation_from_quantity(
    stream: eprem.Stream,
    observables: typing.Dict[str, dict],
) -> None:
    """Create the default observation from each observable quantity."""
    for name, expected in observables.items():
        this = stream[name]
        observation = stream.observe(this)
        assert isinstance(observation, observed.Quantity)
        for axis in expected['axes']:
            assert isinstance(observation[axis], index.Quantity), axis


def test_create_observation_by_name(
    stream: eprem.Stream,
    observables: typing.Dict[str, dict],
) -> None:
    """Create the default observation from each observable quantity."""
    for name, expected in observables.items():
        observation = stream.observe(name)
        assert isinstance(observation, observed.Quantity)
        for axis in expected['axes']:
            assert isinstance(observation[axis], index.Quantity), axis


def test_parameter_access(stream: eprem.Stream) -> None:
    """Access runtime parameter arguments."""
    cases = {
        ('lamo', 'lam0', 'lambda0'): {
            'value': 0.1,
            'unit': 'au',
        },
    }
    for aliases, expected in cases.items():
        for alias in aliases:
            argument = stream[alias]
            assert isinstance(argument, constant.Assumption)
            assert alias in argument.name
            assert float(argument) == expected['value']
            assert argument.unit == expected['unit']


def test_observing_unit(stream: eprem.Stream):
    """Change the unit of an observable quantity."""
    r = stream['r']
    assert r.unit == 'm'
    assert r['au'].unit == 'au'
    r.reset()
    assert r.unit == 'm'
    old = stream.observe(r).data
    new = stream.observe(r['au']).data
    assert numpy.allclose(old, new * float(fundamental.mks['au']))


def test_interpolation(stream: eprem.Stream):
    """Interpolate an observation."""
    cases = [
        {
            'index': 'shell',
            'value': 40,
            'color': 'blue',
        },
        {
            'index': 'shell',
            'value': 43,
            'color': 'black',
        },
        {
            'index': 'shell',
            'value': 50,
            'color': 'green',
        },
        {
            'index': 'radius',
            'value': (1.0, 'au'),
            'color': 'black',
        },
    ]
    kwargs = {
        'radius': {'linestyle': 'solid'},
        'shell': {'linestyle': '', 'marker': 'o', 'markevery': 10},
    }
    # TODO: This should test actual values to make sure that the results at
    # difference shells are different.
    for case in cases:
        name = case['index']
        value = case['value']
        observation = stream.observe(
            'integral flux',
            **{name: value},
            species='H+',
        )
        assert observation.data.shape == (50, 1, 1, 1)
        plt.plot(
            observation.data.squeeze(),
            label=f'{name} = {value}',
            color=case['color'],
            **kwargs.get(name, {}),
        )
    plt.yscale('log')
    plt.legend()
    plt.savefig('test_interpolation.png')
    plt.close()


def test_observable_aliases(stream: eprem.Stream):
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


def test_repeated_access(stream: eprem.Stream):
    """Test repeatedly accessing the same observable."""
    a = stream['br']
    b = stream['br']
    assert a == b
    assert a is not b
