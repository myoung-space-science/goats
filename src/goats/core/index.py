import collections
import collections.abc
import numbers
import typing

import numpy

from goats.core import iterables
from goats.core import metadata
from goats.core import metric


class Data(iterables.ReprStrMixin):
    """Index points and corresponding values."""

    def __new__(cls, points, **kwargs):
        if not all(isinstance(i, numbers.Integral) for i in points):
            raise ValueError(
                "All index points must have integral type"
            ) from None
        return super().__new__(cls)

    def __init__(
        self,
        points: typing.Iterable[numbers.Integral],
        values: typing.Iterable[typing.Union[numbers.Real, str]]=None,
    ) -> None:
        self.points = tuple(points)
        """The integral index points."""
        self.values = self.points if iterables.missing(values) else values
        """The values associated with index points."""

    def __str__(self) -> str:
        return ', '.join(str((i, j)) for i, j in zip(self.points, self.values))


Instance = typing.TypeVar('Instance', bound='Quantity')


class Quantity(collections.abc.Sequence):
    """A sequence of general axis indices."""

    def __init__(
        self,
        __data: Data,
        unit: typing.Union[str, metric.Unit]=None,
    ) -> None:
        self._points = __data.points
        try:
            __data.values[0]
        except TypeError:
            self._values = tuple(__data.values)
        else:
            self._values = __data.values
        self._unit = metadata.Unit(unit) if unit else None

    def __getitem__(self, __i: typing.SupportsIndex):
        """Get the `__i`-th index point."""
        return self._points[__i]

    def __len__(self) -> int:
        """Called for len(self)."""
        return len(self._points)

    @property
    def values(self):
        """The axis value at each index."""
        return self._values

    @property
    def unit(self):
        """The metric unit of the corresponding values, if any."""
        return self._unit

    def __str__(self) -> str:
        """A simplified representation of this object."""
        return self._as_string()

    def __repr__(self) -> str:
        """An unambiguous representation of this object."""
        module = f"{self.__module__.replace('goats.', '')}."
        name = self.__class__.__qualname__
        return self._as_string(
            prefix=f'{module}{name}(',
            suffix=')',
        )

    def _as_string(self, prefix: str='', suffix: str=''):
        """Create a string representation of this object."""
        try:
            signed = '+' if any(i < 0 for i in self.values) else '-'
        except TypeError: # probably because values are strings
            signed = '-'
        values = numpy.array2string(
            numpy.array(self),
            threshold=4,
            edgeitems=2,
            separator=', ',
            sign=signed,
            precision=3,
            floatmode='maxprec_equal',
            prefix=prefix,
            suffix=suffix,
        )
        parts = [values]
        if self.unit:
            parts.append(f"unit={str(self.unit)!r}")
        string = ', '.join(parts)
        return f"{prefix}{string}{suffix}"


class Array(Quantity):
    """An array of axis index values."""

    @typing.overload
    def __init__(self, __data: Data, **meta) -> None:
        """Create a new array from arguments"""

    @typing.overload
    def __init__(self, quantity: Quantity) -> None:
        """Create a new array from an index quantity."""

    @typing.overload
    def __init__(self: Instance, instance: Instance) -> None:
        """Create a new array from an existing instance."""

    def __new__(cls, arg, **meta):
        if not meta and isinstance(arg, Array):
            return arg
        return super().__new__(cls)

    def __init__(self, arg, **meta) -> None:
        if not meta and isinstance(arg, Quantity):
            data = Data(arg._points, values=arg.values)
            return super().__init__(
                data,
                unit=arg.unit,
            )
        super().__init__(arg, **meta)

    def __array__(self, *args, **kwargs):
        """Called for conversion to numpy array types."""
        return numpy.array(self.values, *args, **kwargs)

    # NOTE: The following unit-related logic includes significant overlap with
    # `measurable.Quantity`.
    def __getitem__(self, arg):
        """Set the unit of this object's values, if applicable.
        
        Notes
        -----
        Using this special method to change the unit supports a simple and
        relatively intuitive syntax but is arguably an abuse of notation.

        Raises
        ------
        TypeError
            User attempted to modify the unit of an unmeasurable quantity.

        ValueError
            The given unit is inconsistent with this quantity. Two units are
            mutually consistent if they have the same dimension in a known
            metric system.
        """
        if not isinstance(arg, metadata.UnitLike):
            return super().__getitem__(arg)
        if self.unit is None:
            raise TypeError(
                "Can't set the unit of an unmeasurable quantity."
            ) from None
        unit = (
            self.unit.norm[arg]
            if str(arg).lower() in metric.SYSTEMS else arg
        )
        if unit == self.unit:
            return self
        new = metadata.Unit(unit)
        values = self.values * (new // self.unit)
        if self.unit | new:
            return type(self)(
                Data(self._points, values=values),
                unit=new,
            )
        raise ValueError(
            f"The unit {str(unit)!r} is inconsistent with {str(self.unit)!r}"
        ) from None

