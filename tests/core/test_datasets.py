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
) -> dict:
    """Get reference values for the named dataset."""
    return testdata[name].get(key, {})


def test_axes(testdata: dict):
    """Test access to dataset axes."""
    testname = 'basic'
    dataset = get_dataset(testdata, testname)
    reference = get_reference(testdata, testname, 'axes')
    assert isinstance(dataset.axes, tuple)
    assert sorted(dataset.axes) == sorted(reference)
    assert isinstance(dataset.sizes, datasets.DataViewer)
    sizes = {k: v['size'] for k, v in reference.items()}
    assert sorted(dataset.sizes) == sorted(sizes)
    assert sorted(dataset.sizes.values()) == sorted(sizes.values())


def test_variables(testdata: dict):
    """Test access to dataset variables."""
    testname = 'basic'
    dataset = get_dataset(testdata, testname)
    reference = get_reference(testdata, testname, 'variables')
    assert isinstance(dataset.variables, datasets.DataViewer)
    assert sorted(dataset.variables) == sorted(reference)
    for name, variable in dataset.variables.items():
        assert isinstance(variable, quantities.Variable)
        assert variable.name == name

