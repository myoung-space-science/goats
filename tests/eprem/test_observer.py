"""
This module contains high-level tests of the observer/observable/observation framework. Note that some tests may take especially long to run.
"""

import pathlib
import shutil
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
from goats.core import variable
from goats import eprem


def get_stream(rootpath: pathlib.Path):
    """Create a stream observer.

    This is separated out to allow developers to create a stream outside of the
    `stream` fixture. One example application may be adding a `__main__` section
    and calling simple plotting routines for visual end-to-end tests.
    """
    source = rootpath / 'cone' / 'obs'
    return eprem.Stream(0, source=source)


@pytest.fixture
def stream(rootpath: pathlib.Path):
    """Provide a stream observer via fixture."""
    return get_stream(rootpath)


@pytest.fixture
def axes():
    """Metadata for EPREM dataset axes."""
    return {
        'time': {
            'length': 50,
        },
        'shell': {
            'length': 2000,
        },
        'species': {
            'length': 1,
        },
        'energy': {
            'length': 20,
        },
        'mu': {
            'length': 8,
        },
    }


@pytest.fixture
def quantities() -> typing.Dict[str, dict]:
    """Information about each quantity available to an EPREM observer."""
    T, S, P, E, M = 'time', 'shell', 'species', 'energy', 'mu'
    return {
        'preEruption': {
            'axes': (),
            'unit': {'mks': 's', 'cgs': 's'},
            'aliases': [],
            'observable': False,
        },
        'phiOffset': {
            'axes': (T,),
            'unit': {'mks': 's', 'cgs': 's'},
            'aliases': [],
            'observable': False,
        },
        'time': {
            'axes': (T,),
            'unit': {'mks': 's', 'cgs': 's'},
            'aliases': ['t', 'times'],
            'observable': True,
        },
        'shell': {
            'axes': (S,),
            'unit': {'mks': '1', 'cgs': '1'},
            'aliases': ['shells'],
            'observable': True,
        },
        'mu': {
            'axes': (M,),
            'unit': {'mks': '1', 'cgs': '1'},
            'aliases': [
                'mu', 'mus',
                'pitch angle', 'pitch-angle', 'pitch-angle cosine',
                'pitch angles', 'pitch-angles', 'pitch-angle cosines',
            ],
            'observable': True,
        },
        'mass': {
            'axes': (P,),
            'unit': {'mks': 'kg', 'cgs': 'g'},
            'aliases': ['m'],
            'observable': True,
        },
        'charge': {
            'axes': (P,),
            'unit': {'mks': 'C', 'cgs': 'statC'},
            'aliases': ['q'],
            'observable': True,
        },
        'egrid': {
            'axes': (P, E),
            'unit': {'mks': 'J', 'cgs': 'erg'},
            'aliases': ['energy', 'energies', 'E'],
            'observable': True,
        },
        'vgrid': {
            'axes': (P, E),
            'unit': {'mks': 'm/s', 'cgs': 'cm/s'},
            'aliases': ['speed', 'vparticle'],
            'observable': True,
        },
        'R': {
            'axes': (T, S),
            'unit': {'mks': 'm', 'cgs': 'cm'},
            'aliases': ['r', 'radius'],
            'observable': True,
        },
        'T': {
            'axes': (T, S),
            'unit': {'mks': 'rad', 'cgs': 'rad'},
            'aliases': ['theta'],
            'observable': True,
        },
        'P': {
            'axes': (T, S),
            'unit': {'mks': 'rad', 'cgs': 'rad'},
            'aliases': ['phi'],
            'observable': True,
        },
        'Br': {
            'axes': (T, S),
            'unit': {'mks': 'T', 'cgs': 'G'},
            'aliases': ['br'],
            'observable': True,
        },
        'Bt': {
            'axes': (T, S),
            'unit': {'mks': 'T', 'cgs': 'G'},
            'aliases': ['bt', 'Btheta', 'btheta'],
            'observable': True,
        },
        'Bp': {
            'axes': (T, S),
            'unit': {'mks': 'T', 'cgs': 'G'},
            'aliases': ['bp', 'Bphi', 'bphi'],
            'observable': True,
        },
        'Vr': {
            'axes': (T, S),
            'unit': {'mks': 'm/s', 'cgs': 'cm/s'},
            'aliases': ['vr'],
            'observable': True,
        },
        'Vt': {
            'axes': (T, S),
            'unit': {'mks': 'm/s', 'cgs': 'cm/s'},
            'aliases': ['vt', 'Vtheta', 'vtheta'],
            'observable': True,
        },
        'Vp': {
            'axes': (T, S),
            'unit': {'mks': 'm/s', 'cgs': 'cm/s'},
            'aliases': ['vp', 'Vphi', 'vphi'],
            'observable': True,
        },
        'Rho': {
            'axes': (T, S),
            'unit': {'mks': 'm^-3', 'cgs': 'cm^-3'},
            'aliases': ['rho'],
            'observable': True,
        },
        'Dist': {
            'axes': (T, S, P, E, M),
            'unit': {'mks': 's^3/m^6', 'cgs': 's^3/cm^6'},
            'aliases': ['dist', 'f'],
            'observable': True,
        },
        'x': {
            'axes': (T, S),
            'unit': {'mks': 'm', 'cgs': 'cm'},
            'aliases': ['X'],
            'observable': True,
        },
        'y': {
            'axes': (T, S),
            'unit': {'mks': 'm', 'cgs': 'cm'},
            'aliases': ['Y'],
            'observable': True,
        },
        'z': {
            'axes': (T, S),
            'unit': {'mks': 'm', 'cgs': 'cm'},
            'aliases': ['Z'],
            'observable': True,
        },
        'B': {
            'axes': (T, S),
            'unit': {'mks': 'T', 'cgs': 'G'},
            'aliases': ['b_mag', '|B|', 'bmag', 'b mag'],
            'observable': True,
        },
        'V': {
            'axes': (T, S),
            'unit': {'mks': 'm/s', 'cgs': 'cm/s'},
            'aliases': ['v_mag', '|V|', 'vmag', 'v mag', 'v', '|v|'],
            'observable': True,
        },
        'BV': {
            'axes': (T, S),
            'unit': {'mks': 'T * m/s', 'cgs': 'G * cm/s'},
            'aliases': ['bv_mag', 'bv', '|bv|', '|BV|'],
            'observable': True,
        },
        'Vpara': {
            'axes': (T, S),
            'unit': {'mks': 'm/s', 'cgs': 'cm/s'},
            'aliases': ['v_para', 'vpara'],
            'observable': True,
        },
        'Vperp': {
            'axes': (T, S),
            'unit': {'mks': 'm/s', 'cgs': 'cm/s'},
            'aliases': ['v_perp', 'vperp'],
            'observable': True,
        },
        'flow angle': {
            'axes': (T, S),
            'unit': {'mks': 'rad', 'cgs': 'rad'},
            'aliases': ['flow_angle', 'angle'],
            'observable': True,
        },
        'div(V)': {
            'axes': (T, S),
            'unit': {'mks': '1/s', 'cgs': '1/s'},
            'aliases': ['div_v', 'divV', 'divv', 'div V', 'div v', 'div(v)'],
            'observable': True,
        },
        'density ratio': {
            'axes': (T, S),
            'unit': {'mks': '1', 'cgs': '1'},
            'aliases': ['density_ratio', 'n2/n1', 'n_2/n_1'],
            'observable': True,
        },
        'rigidity': {
            'axes': (P, E),
            'unit': {'mks': 'kg m / (s C)', 'cgs': 'g cm / (s statC)'},
            'aliases': ['Rg', 'R_g'],
            'observable': True,
        },
        'mean free path': {
            'axes': (T, S, P, E),
            'unit': {'mks': 'm', 'cgs': 'cm'},
            'aliases': ['mean_free_path', 'mfp'],
            'observable': True,
        },
        'acceleration rate': {
            'axes': (T, S, P, E),
            'unit': {'mks': '1/s', 'cgs': '1/s'},
            'aliases': ['acceleration_rate'],
            'observable': True,
        },
        'energy density': {
            'axes': (T, S, P),
            'unit': {'mks': 'J/m^3', 'cgs': 'erg/cm^3'},
            'aliases': ['energy_density'],
            'observable': True,
        },
        'average energy': {
            'axes': (T, S, P),
            'unit': {'mks': 'J', 'cgs': 'erg'},
            'aliases': ['average_energy'],
            'observable': True,
        },
        'isotropic distribution': {
            'axes': (T, S, P, E),
            'unit': {'mks': 's^3/m^6', 'cgs': 's^3/cm^6'},
            'aliases': ['isotropic_distribution', 'isodist'],
            'observable': True,
        },
        'flux': {
            'axes': (T, S, P, E),
            'unit': {
                'mks': '# / (s sr m^2 J)',
                'cgs': '# / (s sr cm^2 erg)',
            },
            'aliases': ['Flux', 'J', 'J(E)', 'j', 'j(E)'],
            'observable': True,
        },
        'fluence': {
            'axes': (S, P, E),
            'unit': {
                'mks': '# / (sr m^2 J)',
                'cgs': '# / (sr cm^2 erg)',
            },
            'aliases': [],
            'observable': True,
        },
        'integral flux': {
            'axes': (T, S, P),
            'unit': {
                'mks': '# / (s sr m^2)',
                'cgs': '# / (s sr cm^2)',
            },
            'aliases': ['integral_flux'],
            'observable': True,
        },
        'Vr / Br': {
            'axes': (T, S),
            'unit': {'mks': 'm / (s T)', 'cgs': 'cm / (s G)'},
            'aliases': [],
            'observable': True,
        },
    }


@pytest.fixture
def observables(quantities: typing.Dict[str, dict]):
    """Information about formally observable quantities."""
    return {k: v for k, v in quantities.items() if v['observable']}


def test_create_stream(rootpath: pathlib.Path):
    """Attempt to initialize a stream observer with various arguments."""
    source = rootpath / 'cone' / 'obs'
    datapath = pathlib.Path(source / 'obs000000.nc')
    confpath = pathlib.Path(source / 'eprem_input_file')
    # from ID and absolute directory
    stream = eprem.Stream(0, source=source)
    assert isinstance(stream, observer.Interface)
    assert stream.datapath == datapath
    assert stream.confpath == confpath
    # from ID and relative directory
    dirname = 'local-data'
    testdir = pathlib.Path(dirname)
    if testdir.resolve() != pathlib.Path.cwd():
        testdir.mkdir()
    testdata = pathlib.Path(shutil.copy(datapath, dirname)).resolve()
    assert testdata.parent == pathlib.Path.cwd() / dirname
    testconf = pathlib.Path(shutil.copy(confpath, dirname)).resolve()
    assert testconf.parent == pathlib.Path.cwd() / dirname
    stream = eprem.Stream(0, source=dirname)
    assert isinstance(stream, observer.Interface)
    assert stream.datapath == testdata
    assert stream.confpath == testconf
    testdata.unlink()
    testconf.unlink()
    if testdir.resolve() != pathlib.Path.cwd():
        testdir.rmdir()
    # from only ID
    dirname = '.'
    testdata = pathlib.Path(shutil.copy(datapath, dirname)).resolve()
    assert testdata.parent == pathlib.Path.cwd() / dirname
    testconf = pathlib.Path(shutil.copy(confpath, dirname)).resolve()
    assert testconf.parent == pathlib.Path.cwd() / dirname
    stream = eprem.Stream(0)
    assert isinstance(stream, observer.Interface)
    assert stream.datapath == testdata
    assert stream.confpath == testconf
    testdata.unlink()
    testconf.unlink()
    # from full path: DEPRECATED
    with pytest.raises(TypeError):
        eprem.Stream(source=datapath)


def test_change_source(rootpath: pathlib.Path):
    """Make sure changing the source paths updates the observer."""
    olddir = rootpath / 'cone' / 'obs'
    stream = eprem.Stream(0, source=olddir)
    assert stream.datapath == olddir / 'obs000000.nc'
    assert stream.confpath == olddir / 'eprem_input_file'
    newdir = rootpath / 'wind' / 'obs'
    stream.reset(source=newdir)
    assert stream.datapath == newdir / 'obs000000.nc'
    assert stream.confpath == olddir / 'eprem_input_file'
    stream.reset(config=newdir)
    assert stream.datapath == newdir / 'obs000000.nc'
    assert stream.confpath == newdir / 'eprem_input_file'


def test_quantity_access(
    stream: eprem.Stream,
    quantities: typing.Dict[str, dict],
) -> None:
    """Test access to all non-constant quantities."""
    for name, it_is in quantities.items():
        quantity = stream[name]
        if it_is['observable']:
            assert isinstance(quantity, observable.Quantity), name
        else:
            assert isinstance(quantity, variable.Quantity), name


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
    mfp = stream['mfp']
    assert mfp.unit == 'm'
    assert mfp['au'].unit == 'au'
    assert mfp.unit == 'm'
    old = stream['mfp'].observe().array
    new = stream['mfp']['au'].observe().array
    assert numpy.allclose(old, new * (metric.Unit('m') // metric.Unit('au')))


def test_observation_unit(stream: eprem.Stream):
    """Check and change the unit of an observed quantity."""
    cases = {
        'Vr': ('m / s', 'km / s'),
        'flux': ('m^-2 s^-1 sr^-1 J^-1', 'cm^-2 s^-1 sr^-1 MeV^-1'),
        'mfp': ('m', 'au'),
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
    rootpath: pathlib.Path,
    quantities: typing.Dict[str, dict],
) -> None:
    """Allow users to declare the observer's metric system."""
    source = rootpath / 'cone' / 'obs'
    systems = metric.SYSTEMS
    for system in systems:
        stream = eprem.Stream(0, source=source, system=system)
        for name, expected in quantities.items():
            assert stream[name].unit == expected['unit'][system]


def test_observer_axes(stream: eprem.Stream, axes: dict):
    """Test the observer's axis-like objects."""
    assert len(stream.time) == axes['time']['length']
    assert len(stream.shell) == axes['shell']['length']
    assert len(stream.species) == axes['species']['length']
    assert len(stream.energy) == axes['energy']['length']
    assert len(stream.mu) == axes['mu']['length']
    for unit, scale in {'s': 86400, 'day': 1.0}.items():
        array = numpy.array(stream.time[unit])
        assert array[0] == pytest.approx(0.1 * scale)
        assert array[-1] == pytest.approx(5.0 * scale)
    for unit, scale in {'J': 1.6022e-13, 'MeV': 1.0}.items():
        array = numpy.array(stream.energy[unit])
        assert array[0] == pytest.approx(1e-1 * scale)
        assert array[-1] == pytest.approx(1e2 * scale)
    array = numpy.array(stream.mu)
    assert array[0] == pytest.approx(-1.0)
    assert array[-1] == pytest.approx(1.0)
    array = numpy.array(stream.shell)
    assert array[0] == 0
    assert array[-1] == 1999
    array = numpy.array(stream.species)
    assert array[0] == 'H+'


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
