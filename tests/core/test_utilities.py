import pytest

from goats.core import utilities


def test_getattrval():
    """Test the function that gets a value based on object type."""
    class Base:
        def __init__(self, value) -> None:
            self._value = value
    class PropertyAttr(Base):
        @property
        def value(self):
            return self._value
    class CallableAttr(Base):
        def value(self, scale=1.0):
            return scale * self._value

    value = 2.5
    instance = PropertyAttr(value)
    assert utilities.getattrval(instance, 'value') == value
    instance = CallableAttr(value)
    assert utilities.getattrval(instance, 'value') == value
    scale = 10.0
    expected = scale * value
    assert utilities.getattrval(instance, 'value', scale) == expected
    assert utilities.getattrval(instance, 'value', scale=scale) == expected
    assert utilities.getattrval(value, 'value') == value


def test_setattrval():
    """Test the function that sets a value based on object type."""
    class Base:
        def __init__(self, value) -> None:
            self._value = value
    class StandardAttr(Base):
        def __init__(self, value) -> None:
            super().__init__(value)
            self.value = self._value
    class PropertyAttr(Base):
        @property
        def value(self):
            return self._value
        @value.setter
        def value(self, value):
            self._value = value
    class CallableAttr(Base):
        def value(self, value=None, scale=1.0):
            if value is None:
                return scale * self._value
            self._value = scale * value
            return self

    old, new, scale = 2.5, 4.0, 10.0
    for Instance in StandardAttr, PropertyAttr, CallableAttr:
        instance = Instance(old)
        assert utilities.getattrval(instance, 'value') == old
        utilities.setattrval(instance, 'value', new)
        assert utilities.getattrval(instance, 'value') == new
        if isinstance(instance, CallableAttr):
            utilities.setattrval(instance, 'value', new, scale)
            assert utilities.getattrval(instance, 'value') == scale * new
            utilities.setattrval(instance, 'value', new, scale=scale)
            assert utilities.getattrval(instance, 'value') == scale * new
        with pytest.raises(AttributeError):
            utilities.setattrval(instance, 'other', None)


