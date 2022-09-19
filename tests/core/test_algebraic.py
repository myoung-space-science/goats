from goats.core import algebraic


class Ordered(algebraic.Ordered):
    """Test class for ordered quantities."""

    def __init__(self, __value) -> None:
        self._value = __value

    def __lt__(self, other) -> bool:
        return self._value < other

    def __eq__(self, other) -> bool:
        return self._value == other

    def __le__(self, other) -> bool:
        return super().__le__(other)

    def __ge__(self, other) -> bool:
        return super().__ge__(other)

    def __gt__(self, other) -> bool:
        return super().__gt__(other)

    def __ne__(self, other) -> bool:
        return super().__ne__(other)

    def __bool__(self) -> bool:
        return self._value != 0


def test_ordered():
    """Test a concrete version of the ordered type."""
    this = Ordered(2)
    assert this == 2
    assert this != 1
    assert this < 3
    assert this <= 2
    assert this <= 3
    assert this > 1
    assert this >= 2
    assert this >= 1


