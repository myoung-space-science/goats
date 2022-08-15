import typing

from goats.core import variable


class Quantity(variable.Quantity):
    """The result of observing an observable quantity."""

    def __init__(
        self,
        __v: variable.Quantity,
        context: typing.Mapping,
    ) -> None:
        super().__init__(__v)
        self._indices = {
            k: v for k, v in context.items()
            if k in self.axes
        }
        self._scalars = {
            k: v for k, v in context.items()
            if k not in self.axes
        }
        self.parameters = list(self._scalars)
        """The names of scalar assumptions relevant to this observation."""

    def __getitem__(self, __x):
        """Get a scalar by name or array values by index."""
        if isinstance(__x, str) and __x in self._scalars:
            return self._scalars[__x]
        return super().__getitem__(__x)

    def __eq__(self, other) -> bool:
        """True if two instances have equivalent attributes."""
        if not isinstance(other, type(self)):
            return NotImplemented
        if not self._equal_attrs(other, '_indices', '_scalars'):
            return False
        return super().__eq__(other)

    def _equal_attrs(self, other, *names: str):
        """True if two instances have the same attributes."""
        return all(
            getattr(other, name) == getattr(self, name)
            for name in names
        )

    def __str__(self) -> str:
        """A simplified representation of this object."""
        axes = [str(axis) for axis in self.axes]
        parameters = [str(parameter) for parameter in self.parameters]
        attrs = [
            f"unit='{self.unit}'",
            f"axes={axes}",
            f"parameters={parameters}",
        ]
        return f"'{self.name}': {', '.join(attrs)}"

