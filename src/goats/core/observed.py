import typing

import numpy

from goats.core import aliased
from goats.core import axis
from goats.core import iterables
from goats.core import constant
from goats.core import physical
from goats.core import variable


class Context:
    """The context of an observation."""

    def __init__(
        self,
        indices: typing.Mapping[str, axis.Index],
        assumptions: typing.Mapping[str, constant.Assumption]=None
    ) -> None:
        self._indices = indices
        self._assumptions = aliased.Mapping(assumptions or {})
        self._axes = None

    @property
    def axes(self):
        """The arrays of axis values relevant to the observation."""
        if self._axes is None:
            items = (
                self._indices.items(aliased=True)
                if isinstance(self._indices, aliased.Mapping)
                else self._indices.items()
            )
            axes = {
                k: physical.Array(
                    index.values,
                    unit=index.unit,
                    name=index.name,
                )
                for k, index in items
            }
            self._axes = aliased.Mapping(axes)
        return self._axes

    @property
    def assumptions(self):
        """The physical assumptions relevant to this observation."""
        return self._assumptions

    def __eq__(self, __o) -> bool:
        """True if two contexts have equal axes and assumptions."""
        if isinstance(__o, Context):
            return all(
                getattr(self, attr) == getattr(__o, attr)
                for attr in ('axes', 'assumptions')
            )
        return NotImplemented


class Quantity(iterables.ReprStrMixin):
    """The result of observing an observable quantity."""

    def __init__(
        self,
        __v: variable.Quantity,
        context: Context,
    ) -> None:
        self._quantity = __v
        self._context = context
        self._array = None
        self._data = None
        self._parameters = None

    @property
    def array(self):
        """The observed data array, with singular dimensions removed.
        
        This property intends to provide a convenient shortcut for cases in
        which the observation result is effectively N-dimensional but singular
        dimensions cause it to appear to have higher dimensionality.
        """
        if self._array is None:
            self._array = numpy.squeeze(self.data)
        return self._array

    @property
    def data(self):
        """The observed variable quantity.
        
        This property provides direct access to the array interface, as well as
        to metadata properties of the observed quantity.
        """
        if self._data is None:
            self._data = self._quantity
        return self._data

    @property
    def parameters(self):
        """The names of scalar assumptions relevant to this observation."""
        if self._parameters is None:
            self._parameters = list(self._context.assumptions)
        return self._parameters

    def __getitem__(self, __x):
        """Get a scalar by name or array values by index."""
        if __x in self._context.axes:
            return self._context.axes[__x]
        if __x in self._context.assumptions:
            return self._context.assumptions[__x]

    def __eq__(self, __o) -> bool:
        """True if two instances have equivalent attributes."""
        if isinstance(__o, Quantity):
            return all(
                getattr(self, attr) == getattr(__o, attr)
                for attr in ('data', 'context')
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

