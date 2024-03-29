import pytest

from goats.core import physical
from goats.core import constant


@pytest.mark.xfail
def test_assumption():
    """Test the object that represents a physical assumption."""
    values = [1.0, 2.0]
    unit = 'm'
    aliases = 'this', 'a0'
    assumption = constant.Assumption(values, unit, *aliases)
    assert assumption.unit == unit
    assert all(alias in assumption.name for alias in aliases)
    scalars = [physical.Scalar(value, unit) for value in values]
    assert assumption[:] == scalars
    converted = assumption.convert('cm')
    assert converted.unit == 'cm'
    assert converted[:] == [100.0 * scalar for scalar in scalars]


