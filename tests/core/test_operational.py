import pytest

from goats.core import operational
from goats.core import measurable


def test_argument():
    """Test the object representing an operational argument."""
    cases = [
        [(1,), True],
        [(0,), True],
        [('/file/path',), True],
        [('',), False],
        [(True,), True],
        [(False,), False],
        [(1, 'm'), True],
        [(1.1, 'm'), True],
    ]
    for init, truth in cases:
        arg = operational.Argument(*init)
        data, unit = init if len(init) == 2 else (init[0], None)
        assert arg.data == data
        assert arg.unit == unit
        assert bool(arg) == truth
    with pytest.raises(TypeError):
        operational.Argument(1.1)
    measurement = measurable.measure(1.1, 'm')
    arg = operational.Argument(measurement)
    assert arg.data == measurement.data
    assert arg.unit == measurement.unit

