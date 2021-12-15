import pytest

from goats.common import elements


def test_symbols():
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
        assert elements.symbols(*case['in']) == case['out']
    with pytest.raises(TypeError):
        elements.symbols([1], [2, 3])
    with pytest.raises(elements.MassValueError):
        elements.symbols([2], [0])

