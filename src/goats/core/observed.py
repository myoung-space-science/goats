import typing

import numpy

from goats.core import aliased
from goats.core import index
from goats.core import variable
from goats.core import parameter


class Quantity(variable.Quantity):
    """The result of observing an observable quantity."""

    def __init__(
        self,
        __v: variable.Quantity,
        indices: typing.Mapping[str, index.Quantity],
        assumptions: typing.Mapping[str, parameter.Assumption]
    ) -> None:
        self._result = __v
        self._indices = indices
        self._assumptions = assumptions
        # This should provide an aliased access to indices and assumptions. In
        # practice, they have be aliased mappings themselves, but that may not
        # be true in general.

    def __getitem__(self, __k):
        """Get an assumption by name or array values by index."""
        if __k in self._assumptions:
            return self._assumptions[__k]
        if __k in self._indices:
            return self._indices[__k]
        return self._result[__k] # catch `IndexError` as `KeyError`?

    @property
    def axes(self):
        """The indexable axes of this observation's array."""
        if self._axes is None:
            if isinstance(self._indices, aliased.Mapping):
                self._axes = self._indices.keys(aliased=True)
            else:
                self._axes = self._indices.keys()
        return self._axes

    @property
    def parameters(self):
        """The names of assumptions relevant to this observation."""
        if self._parameters is None:
            if isinstance(self._assumptions, aliased.Mapping):
                self._parameters = self._assumptions.keys(aliased=True)
            else:
                self._parameters = self._assumptions.keys()
        return self._parameters

    def __array__(self, *args, **kwargs) -> numpy.ndarray:
        """Support automatic conversion to a `numpy.ndarray`."""
        return self._data.__array__(*args, **kwargs)

    def __eq__(self, other) -> bool:
        """True if two instances have equivalent attributes."""
        if not isinstance(other, type(self)):
            return NotImplemented
        if not self._equal_attrs(other, 'indices', 'assumptions'):
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

