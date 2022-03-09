import typing

from goats.core import datasets
from goats.core import quantities


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
    assert isinstance(dataset.axes, datasets.DataViewer)
    assert sorted(dataset.axes) == sorted(reference)
    for name, axis in dataset.axes.items():
        assert isinstance(axis, datasets.Axis)
        assert axis.name == name
        assert axis.size == reference[name]['size']


def test_variables(testdata: dict):
    """Test access to dataset variables."""
    testname = 'basic'
    dataset = get_dataset(testdata, testname)
    reference = get_reference(testdata, testname, 'variables')
    assert isinstance(dataset.variables, datasets.DataViewer)
    assert sorted(dataset.variables) == sorted(reference)
    for name, variable in dataset.variables.items():
        assert isinstance(variable, datasets.Variable)
        assert variable.name == name
        ref = reference[name]
        assert variable.unit == ref.get('unit')
        assert sorted(variable.axes) == sorted(ref.get('axes', ()))

