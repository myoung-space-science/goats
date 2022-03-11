import typing

import pytest

from goats.core import aliased
from goats.core import datasets


def get_dataset(testdata: dict, name: str) -> datasets.DatasetView:
    """Get a named test dataset."""
    return datasets.DatasetView(testdata[name]['path'])


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
    dataset = get_dataset(testdata, testname)
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
    dataset = get_dataset(testdata, testname)
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


def test_full_dataset(testdata: dict):
    """Test access to the full dataset."""
    testname = 'basic'
    dataset = get_dataset(testdata, testname)
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
    axes = ['level', 'time', 'lat', 'lon']
    assert sorted(dataset.iter_axes('temp')) == sorted(axes)
    resolved = ('level', 'time', 'lon')
    assert dataset.resolve_axes(['lon', 'time', 'level']) == resolved


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
        variables = datasets.Variables(testdata[name]['path'])
        for observable, expected in reference.items():
            if observable in variables:
                variable = variables[observable]
                assert variable.unit() == expected['unit']
                assert sorted(variable.axes) == sorted(expected['axes'])
            else:
                with pytest.raises(KeyError):
                    variables[observable]
