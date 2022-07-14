import typing

from goats.core import aliased
from goats.core import datafile


def get_interface(testdata: dict, name: str) -> datafile.Interface:
    """Get an interface to a dataset file by name."""
    return datafile.Interface(testdata[name]['path'])


def get_reference(
    testdata: typing.Dict[str, dict],
    name: str,
    key: str,
) -> typing.Dict[str, dict]:
    """Get reference values for the named dataset."""
    return testdata[name].get(key, {})


def test_interface(testdata: dict):
    """Test access to the data-file interface."""
    testname = 'basic'
    dataset = get_interface(testdata, testname)
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


def test_axes(testdata: dict):
    """Test file-level access to dataset axes."""
    testname = 'basic'
    dataset = get_interface(testdata, testname)
    reference = get_reference(testdata, testname, 'axes')
    assert isinstance(dataset.axes, typing.Mapping)
    for axis in reference:
        assert axis in dataset.axes
    for names, axis in dataset.axes.items(aliased=True):
        assert isinstance(axis, datafile.Axis)
        assert axis.name in names
        assert axis.size == reference[axis.name]['size']


def test_variables(testdata: dict):
    """Test file-level access to dataset variables."""
    testname = 'basic'
    dataset = get_interface(testdata, testname)
    reference = get_reference(testdata, testname, 'variables')
    assert isinstance(dataset.variables, typing.Mapping)
    for variable in reference:
        assert variable in dataset.variables
    for names, variable in dataset.variables.items(aliased=True):
        assert isinstance(variable, datafile.Variable)
        assert variable.name in names
        ref = reference[variable.name]
        assert variable.unit == ref.get('unit')
        assert sorted(variable.axes) == sorted(ref.get('axes', ()))

