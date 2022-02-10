import pytest

import numpy

from goats.common import numerical


def test_get_bounding_indices():
    lower = [0.1 * i for i in range(5)]
    upper = [i + 0.2 for i in lower]
    array = numpy.linspace(lower, upper, num=5)
    target = 0.21
    indices = numerical.get_bounding_indices(array, target, axis=1)
    i0 = indices[:, 0]
    i1 = indices[:, 1]
    assert list(i0) == [2, 1, 1, 0, 0]
    assert list(i1) == [3, 2, 2, 1, 1]
    for i, bounds in enumerate(indices):
        j0, j1 = bounds
        v0, v1 = array[i, j0:j1+1]
        assert v0 <= target <= v1 or v1 <= target <= v0


def test_nearest():
    values = [0.1, 0.2, 0.3]
    basic = {
        0.11: (0, 0.1),
        0.15: (0, 0.1),
        0.20: (1, 0.2),
    }
    for target, (index, value) in basic.items():
        found = numerical.find_nearest(values, target)
        assert found.index == index
        assert found.value == value
    for target in [0.21, 0.25, 0.29]:
        found = numerical.find_nearest(values, target, bound='lower')
        assert found.index == 2
        assert found.value == 0.3
        found = numerical.find_nearest(values, target, bound='upper')
        assert found.index == 1
        assert found.value == 0.2


def test_cast():
    """Test the ability to cast an arbitrary object to a numeric type."""
    cases = [
        ('1', int(1)),
        (1, int(1)),
        ('1.1', float(1.1)),
        (1.1, float(1.1)),
        ('1.1+2.2j', complex(1.1, 2.2)),
        (1.1+2.2j, complex(1.1, 2.2)),
    ]
    for (arg, expected) in cases:
        assert numerical.cast(arg) == expected
    assert numerical.cast('a', strict=False) is None
    with pytest.raises(TypeError):
        numerical.cast('a')