import collections
import collections.abc
import typing

from goats.core import aliased
from goats.core import datafile
from goats.core import index
from goats.core import iterables
from goats.core import metadata
from goats.core import metric
from goats.core import reference
from goats.core import variable


T = typing.TypeVar('T')


class Indexer(iterables.ReprStrMixin):
    """An object that computes axis indices from user values."""

    def __init__(
        self,
        method: typing.Callable[..., index.Data],
        size: int,
    ) -> None:
        self._method = method
        """The method that converts target values into indices."""
        self.size = size
        """The maximum number of indices."""
        # NOTE: Instances of this class do not expect an array-like reference
        # object at initialization because knowledge of the appropriate
        # reference array is not always possible before calling `method`, and
        # defining reference-based attributes on only certain instances (or
        # subclasses) could create a misleading interface.

    def compute(self, *args, **kwargs):
        """Call the index-computing method."""
        return self._method(*args, **kwargs)

    def normalize(self, *user: T):
        """Convert user input into suitable target values."""
        if not user:
            return range(self.size)
        if isinstance(user[0], slice):
            return iterables.slice_to_range(user[0], stop=self.size)
        if isinstance(user[0], range):
            return user[0]
        return user

    def __str__(self) -> str:
        return f"{self._method.__qualname__}, size={self.size}"


class IndexTypeError(Exception):
    """Invalid index argument."""


def indexer(n: int):
    """Create an instance of the default axis-indexer."""
    def method(targets):
        try:
            indices = [int(arg) for arg in targets]
            if all(0 <= idx < n for idx in indices):
                return index.Data(indices)
        except TypeError as err:
            raise IndexTypeError(
                f"Can't convert {targets!r} to integer indices."
            ) from err
        raise ValueError(
            f"One or more index in {targets} is outside the interval"
            f" [0, {n-1}]"
        ) from None
    method.__qualname__ = "default"
    return Indexer(method, n)


Instance = typing.TypeVar('Instance', bound='Quantity')


class Quantity(iterables.ReprStrMixin):
    """A callable representation of a dataset axis."""

    @typing.overload
    def __init__(
        self,
        __computer: Indexer,
        unit: typing.Union[str, metric.Unit]=None,
    ) -> None:
        """Create a new axis from scratch."""

    @typing.overload
    def __init__(self: Instance, instance: Instance) -> None:
        """Create a new axis from an existing instance."""

    def __new__(cls, __a, **meta):
        """Return an existing instance, if applicable."""
        if not meta and isinstance(__a, cls):
            return __a
        return super().__new__(cls)

    def __init__(self, __a: Indexer, unit=None) -> None:
        self._indexer = __a
        self._unit = metadata.Unit(unit) if unit else None
        self._reference = None

    @property
    def unit(self) -> metadata.Unit:
        """The unit of this axis's values."""
        return self._unit

    @property
    def reference(self):
        """The full array of axis values."""
        if self._reference is None:
            self._reference = index.Array(self.index())
        return self._reference

    def index(self, *args, **kwargs):
        """Convert arguments into an index-like quantity."""
        targets = self._indexer.normalize(*args)
        unit = kwargs.pop('unit', self.unit)
        if unit:
            data = self._indexer.compute(targets, unit, **kwargs)
            return index.Quantity(data, unit=unit)
        data = self._indexer.compute(targets, **kwargs)
        return index.Quantity(data, unit=unit)

    # NOTE: The following unit-related logic includes significant overlap with
    # `measurable.Quantity`.
    def __getitem__(self, arg: typing.Union[str, metric.Unit]):
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
        if self.unit | new:
            return type(self)(self._indexer, unit=new)
        raise ValueError(
            f"The unit {str(unit)!r} is inconsistent with {str(self.unit)!r}"
        ) from None

    def __str__(self) -> str:
        """A simplified representation of this object."""
        return f"{self._indexer}, unit={str(self.unit)!r}"

