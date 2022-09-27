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
from goats.core import metric
from goats.core import observable
from goats.core import observed
from goats.core import observer
from goats.core import physical
from goats import eprem


def get_stream(rootpath: Path):
    """Create a stream observer.

    This is separated out to allow developers to create a stream outside of the
    `stream` fixture. One example application may be adding a `__main__` section
    and calling simple plotting routines for visual end-to-end tests.
    """
    source = rootpath / 'cone' / 'obs'
    return eprem.Stream(0, source=source)


@pytest.fixture
def stream(rootpath: Path):
    """Provide a stream observer via fixture."""
    return get_stream(rootpath)


@pytest.fixture
def observables() -> typing.Dict[str, dict]:
    """Information about each observable."""
    T, S, P, E, M = 'time', 'shell', 'species', 'energy', 'mu'
    return {
        'time': {
            'axes': (T,),
            'unit': {'mks': 's', 'cgs': 's'},
            'aliases': ['t', 'times'],
        },
        'shell': {
            'axes': (S,),
            'unit': {'mks': '1', 'cgs': '1'},
            'aliases': ['shells'],
        },
        'mu': {
            'axes': (M,),
            'unit': {'mks': '1', 'cgs': '1'},
            'aliases': [
                'mu', 'mus',
                'pitch angle', 'pitch-angle', 'pitch-angle cosine',
                'pitch angles', 'pitch-angles', 'pitch-angle cosines',
            ],
        },
        'mass': {
            'axes': (P,),
            'unit': {'mks': 'kg', 'cgs': 'g'},
            'aliases': ['m'],
        },
        'charge': {
            'axes': (P,),
            'unit': {'mks': 'C', 'cgs': 'statC'},
            'aliases': ['q'],
        },
        'egrid': {
            'axes': (P, E),
            'unit': {'mks': 'J', 'cgs': 'erg'},
            'aliases': ['energy', 'energies', 'E'],
        },
        'vgrid': {
            'axes': (P, E),
            'unit': {'mks': 'm/s', 'cgs': 'cm/s'},
            'aliases': ['speed', 'vparticle'],
        },
        'R': {
            'axes': (T, S),
            'unit': {'mks': 'm', 'cgs': 'cm'},
            'aliases': ['r', 'radius'],
        },
        'T': {
            'axes': (T, S),
            'unit': {'mks': 'rad', 'cgs': 'rad'},
            'aliases': ['theta'],
        },
        'P': {
            'axes': (T, S),
            'unit': {'mks': 'rad', 'cgs': 'rad'},
            'aliases': ['phi'],
        },
        'Br': {
            'axes': (T, S),
            'unit': {'mks': 'T', 'cgs': 'G'},
            'aliases': ['br'],
        },
        'Bt': {
            'axes': (T, S),
            'unit': {'mks': 'T', 'cgs': 'G'},
            'aliases': ['bt', 'Btheta', 'btheta'],
        },
        'Bp': {
            'axes': (T, S),
            'unit': {'mks': 'T', 'cgs': 'G'},
            'aliases': ['bp', 'Bphi', 'bphi'],
        },
        'Vr': {
            'axes': (T, S),
            'unit': {'mks': 'm/s', 'cgs': 'cm/s'},
            'aliases': ['vr'],
        },
        'Vt': {
            'axes': (T, S),
            'unit': {'mks': 'm/s', 'cgs': 'cm/s'},
            'aliases': ['vt', 'Vtheta', 'vtheta'],
        },
        'Vp': {
            'axes': (T, S),
            'unit': {'mks': 'm/s', 'cgs': 'cm/s'},
            'aliases': ['vp', 'Vphi', 'vphi'],
        },
        'Rho': {
            'axes': (T, S),
            'unit': {'mks': 'm^-3', 'cgs': 'cm^-3'},
            'aliases': ['rho'],
        },
        'Dist': {
            'axes': (T, S, P, E, M),
            'unit': {'mks': 's^3/m^6', 'cgs': 's^3/cm^6'},
            'aliases': ['dist', 'f'],
        },
        'x': {
            'axes': (T, S),
            'unit': {'mks': 'm', 'cgs': 'cm'},
            'aliases': ['X'],
        },
        'y': {
            'axes': (T, S),
            'unit': {'mks': 'm', 'cgs': 'cm'},
            'aliases': ['Y'],
        },
        'z': {
            'axes': (T, S),
            'unit': {'mks': 'm', 'cgs': 'cm'},
            'aliases': ['Z'],
        },
        'B': {
            'axes': (T, S),
            'unit': {'mks': 'T', 'cgs': 'G'},
            'aliases': ['b_mag', '|B|', 'bmag', 'b mag'],
        },
        'V': {
            'axes': (T, S),
            'unit': {'mks': 'm/s', 'cgs': 'cm/s'},
            'aliases': ['v_mag', '|V|', 'vmag', 'v mag', 'v', '|v|'],
        },
        'BV': {
            'axes': (T, S),
            'unit': {'mks': 'T * m/s', 'cgs': 'G * cm/s'},
            'aliases': ['bv_mag', 'bv', '|bv|', '|BV|'],
        },
        'Vpara': {
            'axes': (T, S),
            'unit': {'mks': 'm/s', 'cgs': 'cm/s'},
            'aliases': ['v_para', 'vpara'],
        },
        'Vperp': {
            'axes': (T, S),
            'unit': {'mks': 'm/s', 'cgs': 'cm/s'},
            'aliases': ['v_perp', 'vperp'],
        },
        'flow angle': {
            'axes': (T, S),
            'unit': {'mks': 'rad', 'cgs': 'rad'},
            'aliases': ['flow_angle', 'angle'],
        },
        'div(V)': {
            'axes': (T, S),
            'unit': {'mks': '1/s', 'cgs': '1/s'},
            'aliases': ['div_v', 'divV', 'divv', 'div V', 'div v', 'div(v)'],
        },
        'density ratio': {
            'axes': (T, S),
            'unit': {'mks': '1', 'cgs': '1'},
            'aliases': ['density_ratio', 'n2/n1', 'n_2/n_1'],
        },
        'rigidity': {
            'axes': (P, E),
            'unit': {'mks': 'kg m / (s C)', 'cgs': 'g cm / (s statC)'},
            'aliases': ['Rg', 'R_g'],
        },
        'mean free path': {
            'axes': (T, S, P, E),
            'unit': {'mks': 'm', 'cgs': 'cm'},
            'aliases': ['mean_free_path', 'mfp'],
        },
        'acceleration rate': {
            'axes': (T, S, P, E),
            'unit': {'mks': '1/s', 'cgs': '1/s'},
            'aliases': ['acceleration_rate'],
        },
        'energy density': {
            'axes': (T, S, P),
            'unit': {'mks': 'J/m^3', 'cgs': 'erg/cm^3'},
            'aliases': ['energy_density'],
        },
        'average energy': {
            'axes': (T, S, P),
            'unit': {'mks': 'J', 'cgs': 'erg'},
            'aliases': ['average_energy'],
        },
        'isotropic distribution': {
            'axes': (T, S, P, E),
            'unit': {'mks': 's^3/m^6', 'cgs': 's^3/cm^6'},
            'aliases': ['isotropic_distribution', 'isodist'],
        },
        'flux': {
            'axes': (T, S, P, E),
            'unit': {
                'mks': '# / (s sr m^2 J)',
                'cgs': '# / (s sr cm^2 erg)',
            },
            'aliases': ['Flux', 'J', 'J(E)', 'j', 'j(E)'],
        },
        'fluence': {
            'axes': (S, P, E),
            'unit': {
                'mks': '# / (sr m^2 J)',
                'cgs': '# / (sr cm^2 erg)',
            },
            'aliases': [],
        },
        'integral flux': {
            'axes': (T, S, P),
            'unit': {
                'mks': '# / (s sr m^2)',
                'cgs': '# / (s sr cm^2)',
            },
            'aliases': ['integral_flux'],
        },
        'Vr / Br': {
            'axes': (T, S),
            'unit': {'mks': 'm / (s T)', 'cgs': 'cm / (s G)'},
            'aliases': [],
        },
    }


def test_create_stream(rootpath: Path):
    """Attempt to initialize a stream observer with various arguments."""
    source = rootpath / 'cone' / 'obs'
    # from ID and directory
    stream = eprem.Stream(0, source=source)
    assert isinstance(stream, observer.Interface)
    # from full path: DEPRECATED
    with pytest.raises(TypeError):
        eprem.Stream(source=source / 'obs000000.nc')
    # from only ID, with default path
    stream = eprem.Stream(0)
    assert isinstance(stream, observer.Interface)


def test_change_source(rootpath: Path):
    """Make sure changing the source paths updates the observer."""
    olddir = rootpath / 'cone' / 'obs'
    stream = eprem.Stream(0, source=olddir)
    assert stream.datapath == olddir / 'obs000000.nc'
    assert stream.confpath == olddir / 'eprem_input_file'
    newdir = rootpath / 'wind' / 'obs'
    stream.readfrom(newdir)
    assert stream.datapath == newdir / 'obs000000.nc'
    assert stream.confpath == newdir / 'eprem_input_file'


def test_observable_access(
    stream: eprem.Stream,
    observables: typing.Dict[str, dict],
) -> None:
    """Access all observables."""
    for name in observables:
        quantity = stream[name]
        assert isinstance(quantity, observable.Quantity)


def test_create_observation(
    stream: eprem.Stream,
    observables: typing.Dict[str, dict],
) -> None:
    """Create the default observation from each observable quantity."""
    for name, expected in observables.items():
        observation = stream[name].observe()
        assert isinstance(observation, observed.Quantity)
        for axis in expected['axes']:
            assert isinstance(observation[axis], physical.Array), axis


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
    assert r.unit == 'm'
    old = stream['r'].observe().array
    new = stream['r']['au'].observe().array
    assert numpy.allclose(old, new * (metric.Unit('m') // metric.Unit('au')))


def test_observation_unit(stream: eprem.Stream):
    """Check and change the unit of an observed quantity."""
    cases = {
        'r': ('m', 'au'),
        'Vr': ('m / s', 'km / s'),
        'flux': ('m^-2 s^-1 sr^-1 J^-1', 'cm^-2 s^-1 sr^-1 MeV^-1'),
        'mfp / Vr': ('s', 'day'),
    }
    for name, (u0, u1) in cases.items():
        observation = stream[name].observe()
        assert observation.data.unit == u0
        converted = observation[u1]
        assert converted is not observation
        assert converted.data.name == observation.data.name
        assert converted.data.axes == observation.data.axes
        assert converted.data.unit == u1


def test_observer_metric_system(
    rootpath: Path,
    observables: typing.Dict[str, dict],
) -> None:
    """Allow users to declare the observer's metric system."""
    source = rootpath / 'cone' / 'obs'
    systems = metric.SYSTEMS
    for system in systems:
        stream = eprem.Stream(0, source=source, system=system)
        for name, expected in observables.items():
            assert stream[name].unit == expected['unit'][system]


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
    time_unit = 'hour'
    data_unit = 'cm^-2 s^-1 sr^-1'
    for case in cases:
        name = case['index']
        value = case['value']
        observation = stream['integral flux'].observe(
            **{name: value},
            species='H+',
        )
        converted = observation[data_unit]
        assert converted.data.shape == (50, 1, 1, 1)
        plt.plot(
            converted['time'][time_unit],
            converted.array,
            label=f'{name} = {value}',
            color=case['color'],
            **kwargs.get(name, {}),
        )
    plt.xlabel(f'Time [{time_unit}]')
    plt.ylabel(f'Integral Flux [{data_unit}]')
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
