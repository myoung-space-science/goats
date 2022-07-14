import pytest

from goats.core import datatypes
from goats.core import parameter


@pytest.mark.xfail
def test_assumption():
    """Test the object that represents a physical assumption."""
    values = [1.0, 2.0]
    unit = 'm'
    aliases = 'this', 'a0'
    assumption = parameter.Assumption(values, unit, *aliases)
    assert assumption.unit == unit
    assert all(alias in assumption.name for alias in aliases)
    scalars = [datatypes.Scalar(value, unit) for value in values]
    assert assumption[:] == scalars
    converted = assumption.convert('cm')
    assert converted.unit == 'cm'
    assert converted[:] == [100.0 * scalar for scalar in scalars]


