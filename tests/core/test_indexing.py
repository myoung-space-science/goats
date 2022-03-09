from goats.core import indexing


def test_equality():
    """Test the binary equality operator for various types."""
    indices = ([1, 2], [3, 4])
    orig = indexing.Indices(indices[0])
    same = indexing.Indices(indices[0])
    diff = indexing.Indices(indices[1])
    assert orig == same
    assert orig != diff
    values = ([-1, -2], [-3, -4])
    orig = indexing.OrderedPairs(indices[0], values[0])
    same = indexing.OrderedPairs(indices[0], values[0])
    diff = indexing.OrderedPairs(indices[0], values[1])
    assert orig == same
    assert orig != diff
    diff = indexing.OrderedPairs(indices[1], values[1])
    assert orig != diff
    unit = ('m', 'J')
    orig = indexing.Coordinates(indices[0], values[0], unit[0])
    same = indexing.Coordinates(indices[0], values[0], unit[0])
    diff = indexing.Coordinates(indices[0], values[0], unit[1])
    assert orig == same
    assert orig != diff
    diff = indexing.Coordinates(indices[0], values[1], unit[1])
    assert orig != diff
    diff = indexing.Coordinates(indices[1], values[1], unit[1])
    assert orig != diff
