import random
import typing

import pytest

from goats.core import aliased
from goats.core import datasets


def get_dataset_view(testdata: dict, name: str) -> datasets.DatasetView:
    """Get a dataset view by name."""
    return datasets.DatasetView(testdata[name]['path'])


def get_dataset(testdata: dict, name: str) -> datasets.Dataset:
    """Get a dataset interface (without axis indexers) by name."""
    return datasets.Dataset(testdata[name]['path'])


def get_reference(
    testdata: typing.Dict[str, dict],
    name: str,
    key: str,
) -> typing.Dict[str, dict]:
    """Get reference values for the named dataset."""
    return testdata[name].get(key, {})


def test_dataset_axes(testdata: dict):
    """Test access to dataset axes."""
    testname = 'basic'
    dataset = get_dataset_view(testdata, testname)
    reference = get_reference(testdata, testname, 'axes')
    assert isinstance(dataset.axes, typing.Mapping)
    for axis in reference:
        assert axis in dataset.axes
    for names, axis in dataset.axes.items(aliased=True):
        assert isinstance(axis, datasets.DatasetAxis)
        assert axis.name in names
        assert axis.size == reference[axis.name]['size']


def test_dataset_variables(testdata: dict):
    """Test access to dataset variables."""
    testname = 'basic'
    dataset = get_dataset_view(testdata, testname)
    reference = get_reference(testdata, testname, 'variables')
    assert isinstance(dataset.variables, typing.Mapping)
    for variable in reference:
        assert variable in dataset.variables
    for names, variable in dataset.variables.items(aliased=True):
        assert isinstance(variable, datasets.DatasetVariable)
        assert variable.name in names
        ref = reference[variable.name]
        assert variable.unit == ref.get('unit')
        assert sorted(variable.axes) == sorted(ref.get('axes', ()))


def test_dataset_view(testdata: dict):
    """Test access to the lower-level dataset view."""
    testname = 'basic'
    dataset = get_dataset_view(testdata, testname)
    variables = dataset.variables
    assert variables['time'] == variables['t']
    available = dataset.available('variables')
    full = ['time', 't', 'times', 'level', 'lat', 'lon', 'temp']
    assert sorted(available.full) == sorted(full)
    aliases = [
        aliased.MappingKey(key)
        for key in ('level', ('time', 't', 'times'), 'lat', 'lon', 'temp')
    ]
    assert sorted(available.aliased) == sorted(aliases)
    canonical = ['level', 'time', 'lat', 'lon', 'temp']
    assert sorted(available.canonical) == sorted(canonical)


def test_variables(testdata: dict):
    """Test the higher-level variables interface."""
    reference = {
        'time': {
            'unit': 's',
            'axes': ['time'],
        },
        'Vr': {
            'unit': 'm / s',
            'axes': ['time', 'shell'],
        },
        'flux': {
            'unit': 'm^-2 s^-1 sr^-1 J^-1',
            'axes': ['time', 'shell', 'species', 'energy'],
        },
        'dist': {
            'unit': 's^3 m^-6',
            'axes': ['time', 'shell', 'species', 'energy', 'mu'],
        },
    }
    for name in ('eprem-obs', 'eprem-flux'):
        dataset = get_dataset_view(testdata, name)
        variables = datasets.Variables(dataset)
        for observable, expected in reference.items():
            if observable in variables:
                variable = variables[observable]
                assert variable.unit == expected['unit']
                assert sorted(variable.axes) == sorted(expected['axes'])
            else:
                with pytest.raises(KeyError):
                    variables[observable]


def test_dataset(testdata: dict):
    """Test the full higher-level dataset interface."""
    reference = {
        'time': {
            'axes': ['time'],
        },
        'Vr': {
            'axes': ['time', 'shell'],
        },
        'flux': {
            'axes': ['time', 'shell', 'species', 'energy'],
        },
        'dist': {
            'axes': ['time', 'shell', 'species', 'energy', 'mu'],
        },
    }
    axes = {
        'time': 20,
        'shell': 100,
        'species': 2,
        'energy': 10,
        'mu': 5,
    }
    for name in ('eprem-obs', 'eprem-flux'):
        dataset = get_dataset(testdata, name)
        assert isinstance(dataset, datasets.Dataset)
        assert isinstance(dataset.variables, typing.Mapping)
        assert isinstance(dataset.axes, typing.Mapping)
        for key, length in axes.items():
            axis = dataset.axes[key]
            assert key in axis.names
            assert list(axis()) == list(range(length))
        for observable, expected in reference.items():
            if observable in dataset.variables:
                iter_axes = dataset.iter_axes(observable)
                assert sorted(iter_axes) == sorted(expected['axes'])
                unordered = random.sample(axes.keys(), len(axes))
                assert dataset.resolve_axes(unordered) == tuple(axes)
            else: # Test both options when variable is not in dataset.
                assert not list(dataset.iter_axes(observable, default=()))
                with pytest.raises(ValueError):
                    dataset.iter_axes(observable)


def test_standardize():
    """Test the helper function that standardizes unit strings."""
    cases = {
        'julian date': 'day',
        'shell': '1',
        'cos(mu)': '1',
        'e-': 'e',
        '# / cm^2 s sr MeV': '# / (cm^2 s sr MeV/nuc)',
    }
    for old, new in cases.items():
        assert datasets.standardize(old) == new


