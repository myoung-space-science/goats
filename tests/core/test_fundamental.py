import pytest

from goats.core import fundamental


def test_constants():
    """Test the object that represents physical constants."""
    for key, data in fundamental.CONSTANTS.items(aliased=True):
        for system in ('mks', 'cgs'):
            d = data[system]
            mapping = fundamental.Constants(system)
            c = mapping[key]
            assert float(c) == d['value']
            assert c.unit == d['unit']


def test_elements():
    """Test the function that translates mass and charge to element symbol."""
    # TODO: This needn't test every possible charge state, and testing every
    # known element may be overkill, but it should extend beyond H and He.
    cases = [
        {'in': [1, 0], 'out': ['H']},
        {'in': [[1], [0]], 'out': ['H']},
        {'in': [1, +1], 'out': ['H+']},
        {'in': [1, -1], 'out': ['H-']},
        {'in': [4, 0], 'out': ['He']},
        {'in': [4, +1], 'out': ['He+']},
        {'in': [4, +2], 'out': ['He++']},
        {'in': [[1, 4], [+1, +2]], 'out': ['H+', 'He++']},
    ]
    for case in cases:
        assert fundamental.elements(*case['in']) == case['out']
    with pytest.raises(TypeError):
        fundamental.elements([1], [2, 3])
    with pytest.raises(fundamental.MassValueError):
        fundamental.elements([2], [0])

