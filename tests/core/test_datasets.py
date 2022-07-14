import random
import typing

import pytest

from goats.core import datafile
from goats.core import datasets


def get_interface(testdata: dict, name: str) -> datafile.Interface:
    """Get an interface to a dataset file by name."""
    return datafile.Interface(testdata[name]['path'])


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
        datafile = get_interface(testdata, name)
        variables = datasets.Variables(datafile)
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


def test_resolve_axes(testdata: dict):
    """Test the method that orders axes based on the dataset."""
    dataset = get_dataset(testdata, 'eprem-obs')
    # This is only a subset of the possible cases and there's probably a more
    # efficient way to build the collection.
    cases = [
        {
            'input': ('shell', 'energy', 'time'),
            'output': ('time', 'shell', 'energy'),
        },
        {
            'input': ('shell', 'energy', 'time', 'extra'),
            'output': ('time', 'shell', 'energy'),
        },
        {
            'input': ('shell', 'energy', 'time'),
            'mode': 'strict',
            'output': ('time', 'shell', 'energy'),
        },
        {
            'input': ('shell', 'energy', 'time', 'extra'),
            'mode': 'strict',
            'output': ('time', 'shell', 'energy'),
        },
        {
            'input': ('shell', 'energy', 'time', 'extra'),
            'mode': 'append',
            'output': ('time', 'shell', 'energy', 'extra'),
        },
        {
            'input': ('extra', 'shell', 'energy', 'time'),
            'mode': 'append',
            'output': ('time', 'shell', 'energy', 'extra'),
        },
        {
            'input': ('shell', 'extra', 'energy', 'time'),
            'mode': 'append',
            'output': ('time', 'shell', 'energy', 'extra'),
        },
    ]
    for case in cases:
        names = case['input']
        expected = case['output']
        result = (
            dataset.resolve_axes(names, mode=case['mode']) if 'mode' in case
            else dataset.resolve_axes(names)
        )
        assert result == expected


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


