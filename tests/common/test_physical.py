import pytest

from goats.common import physical


def test_constants():
    """Test the object that represents physical constants."""
    for key, data in physical._CONSTANTS.items():
        for system in ('mks', 'cgs'):
            d = data[system]
            mapping = physical.Constants(system)
            c = mapping[key]
            assert float(c) == d['value']
            assert c.unit() == d['unit']


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
        assert physical.elements(*case['in']) == case['out']
    with pytest.raises(TypeError):
        physical.elements([1], [2, 3])
    with pytest.raises(physical.MassValueError):
        physical.elements([2], [0])


def test_plasma_species():
    """Test the object that represents a single plasma species."""
    cases = [
        {'symbol': 'H+'},
        {'mass': 1.00797, 'charge': 1.0},
    ]
    for case in cases:
        species = physical.PlasmaSpecies(**case)
        assert species.symbol == 'H+'
        assert all(float(m) == 1.00797 for m in (species.mass, species.m))
        assert all(float(q) == 1.0 for q in (species.charge, species.q))
