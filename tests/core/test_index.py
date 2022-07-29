import pytest

from goats.core import index


def test_quantity_init():
    """Initialize an index quantity with various arguments."""
    q = index.create([1, 2])
    assert q.indices == (1, 2)
    assert q.data == (1, 2) # defaults to `indices` values and type
    assert q.unit is None
    q = index.create([1, 2], values=[-1, -2])
    assert q.indices == (1, 2)
    assert q.data == [-1, -2] # preserves own type
    assert q.unit is None
    q = index.create([1, 2], values=[1.2, 3.4], unit='1')
    assert q.indices == (1, 2)
    assert q.data == [1.2, 3.4]
    assert q.unit == '1' # unitless is different from null unit
    with pytest.raises(ValueError):
        index.create([1.2, 3.4])


def test_quantity_equality():
    """Test the binary equality operator for various indices."""
    indices = ([1, 2], [3, 4])
    orig = index.create(indices[0])
    same = index.create(indices[0])
    diff = index.create(indices[1])
    assert orig == same
    assert orig != diff
    values = ([-1, -2], [-3, -4])
    orig = index.create(indices[0], values=values[0])
    same = index.create(indices[0], values=values[0])
    diff = index.create(indices[0], values=values[1])
    assert orig == same
    assert orig != diff
    diff = index.create(indices[1], values=values[1])
    assert orig != diff
    unit = ('m', 'J')
    orig = index.create(indices[0], values=values[0], unit=unit[0])
    same = index.create(indices[0], values=values[0], unit=unit[0])
    diff = index.create(indices[0], values=values[0], unit=unit[1])
    assert orig == same
    assert orig != diff
    diff = index.create(indices[0], values=values[1], unit=unit[1])
    assert orig != diff
    diff = index.create(indices[1], values=values[1], unit=unit[1])
    assert orig != diff

