import numbers
import typing

import numpy
import numpy.typing

from goats.core import iterables
from goats.core import metadata
from goats.core import physical


class Data:
    """Index points and optional corresponding values."""

    def __new__(cls, __points, **kwargs):
        if not all(isinstance(i, numbers.Integral) for i in __points):
            raise ValueError(
                "All index points must have integral type"
            ) from None
        return super().__new__(cls)

    def __init__(
        self,
        __points: typing.Iterable[numbers.Integral],
        *,
        values: typing.Iterable[numbers.Real]=None,
    ) -> None:
        self.points = tuple(__points)
        """The integral index points."""
        self.values = values or self.points
        """The values associated with index points."""


Instance = typing.TypeVar('Instance', bound='Quantity')


class Quantity(physical.Quantity):
    """A sequence of values that can index a variable."""

    @typing.overload
    def __init__(
        self: Instance,
        __data: Data,
        unit: metadata.UnitLike=None,
        name: typing.Union[str, typing.Iterable[str]]=None,
    ) -> None: ...

    @typing.overload
    def __init__(
        self: Instance,
        instance: Instance,
    ) -> None: ...

    def __init__(self, __data, **meta) -> None:
        """Initialize this instance from arguments or an existing instance."""
        if isinstance(__data, Data):
            indices = __data.points
            __data = __data.values
        elif isinstance(__data, Quantity):
            indices = __data.indices
        super().__init__(__data, **meta)
        self._indices = indices
        # HACK: Reset `_unit` to `None` if the user didn't pass it in. A value
        # of `None` is different from the default value of `'1'`. The latter is
        # only appropriate for dimensionless physical indices (a.k.a
        # coordinates).
        self._unit = meta.get('unit')

    @property
    def indices(self) -> typing.Tuple[numbers.Integral, ...]:
        """The points at which to index a data array."""
        return self._indices

    def __getitem__(self, __i: typing.SupportsIndex):
        """Called for index look-up and iteration."""
        return self.indices[__i]

    def __len__(self):
        """Called for len(self) and iteration."""
        return len(self.indices)

    @property
    def unit(self):
        if self._unit is not None:
            return super().unit

    def apply_conversion(self, new: metadata.Unit):
        if self._unit is not None:
            return super().apply_conversion(new)
        raise TypeError("Can't convert null unit") from None


class Factory:
    """A callable object that generates array indices from user arguments."""

    def __init__(
        self,
        method: typing.Callable[..., Quantity],
        size: int,
        reference: numpy.typing.ArrayLike,
    ) -> None:
        self.method = method
        self.size = size
        self.reference = reference

    def __call__(self, *args, **kwargs):
        """Call the array-indexing method."""
        targets = self.normalize(*args)
        if all(isinstance(value, numbers.Integral) for value in targets):
            return Quantity(targets)
        return self.method(targets, **kwargs)

    def normalize(self, *user):
        """Convert user input into suitable target values."""
        if not user:
            return self.reference
        if isinstance(user[0], slice):
            return iterables.slice_to_range(user[0], stop=self.size)
        if isinstance(user[0], range):
            return user[0]
        return user

