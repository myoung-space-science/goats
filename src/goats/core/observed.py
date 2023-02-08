import typing

import numpy

from goats.core import aliased
from goats.core import iterables
from goats.core import metric
from goats.core import variable


class Quantity(iterables.ReprStrMixin):
    """The result of observing an observable quantity."""

    def __init__(
        self,
        __data: variable.Quantity,
        indices: typing.Mapping,
        assumptions: typing.Mapping=None,
    ) -> None:
        self._data = __data
        self._indices = indices
        self._assumptions = assumptions
        self._array = None
        self._parameters = None

    @property
    def array(self):
        """The observed data array, with singular dimensions removed.
        
        This property primarily provides a shortcut for cases in which the
        result of observing an N-dimensional quantity is an effectively
        M-dimensional quantity, with M<N, but singular dimensions cause it to
        appear to have higher dimensionality.
        """
        if self._array is None:
            self._array = numpy.array(self.data).squeeze()
        return self._array

    @property
    def data(self):
        """The observed variable quantity.
        
        This property provides direct access to the variable-quantity interface,
        as well as to metadata properties of the observed quantity.
        """
        return self._data

    @property
    def parameters(self):
        """The physical parameters relevant to this observation."""
        if self._parameters is None:
            self._parameters = list(self._assumptions)
        return self._parameters

    def __getitem__(self, __x):
        """Get context items or update the unit.
        
        Parameters
        ----------
        __x : string
            If `__x` names a known array axis or physical assumption, return
            that quantity. If `__x` is a valid unit for this observed quantity,
            return a new instance with updated unit.
        """
        if not isinstance(__x, (str, aliased.Group, metric.Unit)):
            raise TypeError(
                f"{__x!r} must name a context item or a unit."
                "Use the array property to access data values."
            ) from None
        if __x in self._indices:
            return self._indices[__x]
        if __x in self._assumptions:
            return self._assumptions[__x]
        return type(self)(self.data, self._indices, self._assumptions)

    def __eq__(self, __o) -> bool:
        """True if two instances have equivalent attributes."""
        if isinstance(__o, Quantity):
            return all(
                getattr(self, attr) == getattr(__o, attr)
                for attr in ('data', 'indices', 'assumptions')
            )
        return NotImplemented

    def __str__(self) -> str:
        """A simplified representation of this object."""
        attrs = [
            f"unit='{self.data.unit}'",
            f"dimensions={self.data.axes}",
            f"parameters={self.parameters}",
        ]
        return f"'{self.data.name}': {', '.join(attrs)}"

