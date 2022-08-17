import typing

import numpy

from goats.core import index
from goats.core import variable


class Quantity:
    """The result of observing an observable quantity."""

    def __init__(
        self,
        __v: variable.Quantity,
        indices: typing.Mapping[str, index.Quantity],
        **assumptions
    ) -> None:
        self.data = numpy.array(__v.data)
        """The array of this observation's data."""
        self.unit = property(__v.unit)
        """The metric unit of this observation's data values."""
        self.name = property(__v.name)
        """The name(s) of this observation."""
        self.dimensions = property(__v.axes)
        """The names of axes in this observation's data array."""
        self._indices = indices
        self._assumptions = assumptions
        self._parameters = None

    @property
    def parameters(self):
        """The names of scalar assumptions relevant to this observation."""
        if self._parameters is None:
            self._parameters = list(self._assumptions)
        return self._parameters

    def __getitem__(self, __x):
        """Get a scalar by name or array values by index."""
        if isinstance(__x, str):
            if __x in self._indices:
                return self._indices[__x]
            if __x in self._assumptions:
                return self._assumptions[__x]
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
        attrs = [
            f"unit='{self.unit}'",
            f"dimensions={self.dimensions}",
            f"parameters={self.parameters}",
        ]
        return f"'{self.name}': {', '.join(attrs)}"

