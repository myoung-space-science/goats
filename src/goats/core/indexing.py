import collections.abc
import numbers
import typing

import numpy
import numpy.typing

from goats.core import aliased
from goats.core import iterables
from goats.core import measurable
from goats.core import metadata


class Indices(collections.abc.Sequence, iterables.ReprStrMixin):
    """A sequence of indices into data arrays."""

    __slots__ = ('indices',)

    def __init__(self, indices: typing.Iterable[int]) -> None:
        self.indices = tuple(indices)

    def __getitem__(self, __i: typing.SupportsIndex):
        """Called for index look-up and iteration."""
        return self.indices[__i]

    def __len__(self):
        """Called for len(self) and iteration."""
        return len(self.indices)

    def __eq__(self, other):
        """True if two instances have the same attributes."""
        return (
            self.indices == other.indices if isinstance(other, Indices)
            else NotImplemented
        )

    def __str__(self) -> str:
        """A simplified representation of this object."""
        return iterables.show_at_most(3, self.indices, separator=', ')


class IndexMap(Indices):
    """A sequence of indices in correspondence with values of any type."""

    __slots__ = ('_values',)

    def __init__(
        self,
        indices: typing.Iterable[int],
        values: typing.Iterable[typing.Any],
    ) -> None:
        super().__init__(indices)
        self._values = values
        nv = len(self.values)
        ni = len(self.indices)
        if nv != ni:
            errmsg = f"number of values ({nv}) != number of indices ({ni})"
            raise TypeError(errmsg)

    @property
    def values(self):
        """The values corresponding to indices."""
        return tuple(self._values)

    def __eq__(self, other):
        return (
            super().__eq__(other) and self.values == other.values
            if isinstance(other, IndexMap)
            else NotImplemented
        )

    def __str__(self) -> str:
        """A simplified representation of this object."""
        pairs = [f"{i} | {v!r}" for i, v in zip(self.indices, self.values)]
        return iterables.show_at_most(3, pairs, separator=', ')


class Coordinates(IndexMap):
    """A sequence of indices in correspondence with scalar values."""

    __slots__ = ('_unit',)

    def __init__(
        self,
        indices: typing.Iterable[int],
        values: typing.Iterable[typing.Any],
        unit: metadata.UnitLike,
    ) -> None:
        super().__init__(indices, values)
        self._unit = unit

    @property
    def values(self):
        """The values of this coordinate."""
        return numpy.array(self._values)

    @property
    def unit(self):
        """The metric unit of this coordinate's values."""
        return metadata.Unit(self._unit)

    def convert(self, unit: metadata.UnitLike):
        """Convert this object to the new unit, if possible."""
        if unit == self._unit:
            return self
        scale = metadata.Unit(unit) // self._unit
        self.values = [value * scale for value in self.values]
        self._unit = unit
        return self

    def __eq__(self, other):
        return (
            numpy.array_equal(self.values, other.values)
            and self._unit == other._unit
            if isinstance(other, Coordinates)
            else NotImplemented
        )

    def __str__(self) -> str:
        """A simplified representation of this object."""
        values = iterables.show_at_most(3, self.values, separator=', ')
        return f"{values} [{self._unit}]"


IndexLike = typing.TypeVar('IndexLike', bound=Indices)
IndexLike = typing.Union[Indices, IndexMap, Coordinates]


class Indexer:
    """A callable object that generates array indices from user arguments."""

    def __init__(
        self,
        method: typing.Callable[..., IndexLike],
        reference: numpy.typing.ArrayLike,
    ) -> None:
        self.method = method
        self.reference = reference

    def __call__(self, targets, **kwargs):
        """Call the array-indexing method."""
        return self.method(targets, **kwargs)


Instance = typing.TypeVar('Instance', bound='Axis')


class Axis(iterables.ReprStrMixin):
    """A single dataset axis."""

    @typing.overload
    def __init__(
        self: Instance,
        size: int,
        indexer: Indexer,
        *names: str,
    ) -> None:
        """Create a new axis."""

    @typing.overload
    def __init__(
        self: Instance,
        instance: Instance,
    ) -> None:
        """Create a new axis."""

    def __init__(self, *args, **kwargs) -> None:
        parsed = self._parse(*args, **kwargs)
        size, indexer, names = parsed
        self.size = size
        """The full length of this axis."""
        self.indexer = indexer
        """A callable object that creates indices from user input."""
        self.names = names
        """The valid names for this axis."""
        self.reference = indexer.reference
        """The index reference values."""

    Attrs = typing.TypeVar('Attrs', bound=tuple)
    Attrs = typing.Tuple[
        int,
        Indexer,
        aliased.MappingKey,
    ]

    def _parse(self, *args, **kwargs) -> Attrs:
        """Parse input arguments to initialize this instance."""
        if not kwargs and len(args) == 1 and isinstance(args[0], type(self)):
            instance = args[0]
            return tuple(
                getattr(instance, name)
                for name in ('size', 'indexer', 'names')
            )
        size, indexer, *args = args
        names = aliased.MappingKey(args or ())
        return size, indexer, names

    def __call__(self, *args, **kwargs):
        """Convert user arguments into an index object."""
        targets = self._normalize(*args)
        if all(isinstance(value, numbers.Integral) for value in targets):
            return Indices(targets)
        return self.indexer(targets, **kwargs)

    def _normalize(self, *user):
        """Helper for computing target values from user input."""
        if not user:
            return self.reference
        if isinstance(user[0], slice):
            return iterables.slice_to_range(user[0], stop=self.size)
        if isinstance(user[0], range):
            return user[0]
        return user

    def __len__(self) -> int:
        """The full length of this axis. Called for len(self)."""
        return self.size

    def __str__(self) -> str:
        """A simplified representation of this object."""
        string = f"'{self.names}': size={self.size}"
        unit = (
            str(self.reference.unit)
            if isinstance(self.reference, measurable.Quantity)
            else None
        )
        if unit:
            string += f", unit={unit!r}"
        return string


