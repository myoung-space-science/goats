import typing

import pytest

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


def test_axes(testdata: dict):
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


def test_variables(testdata: dict):
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


def test_dataset(testdata: dict):
    """Test access to the full dataset."""
    testname = 'basic'
    dataset = get_dataset(testdata, testname)
    variables = dataset.variables
    assert variables['time'] == variables['t']
