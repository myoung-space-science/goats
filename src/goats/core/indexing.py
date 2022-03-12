import abc
import collections.abc
import numbers
import typing

from goats.core.numerical import find_nearest
from goats.core import iterables
from goats.core import quantities


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
        """True if two instances have the same indices."""
        if not isinstance(other, Indices):
            return NotImplemented
        return all(
            getattr(self, attr) == getattr(other, attr)
            for attr in self.__slots__
        )

    def __str__(self) -> str:
        """A simplified representation of this object."""
        return iterables.show_at_most(3, self.indices, separator=', ')


class OrderedPairs(Indices):
    """A sequence of index-value pairs."""

    __slots__ = ('values',)

    def __init__(
        self,
        indices: typing.Iterable[int],
        values: typing.Iterable[typing.Any],
    ) -> None:
        super().__init__(indices)
        self.values = tuple(values)
        nv = len(self.values)
        ni = len(self.indices)
        if nv != ni:
            errmsg = f"number of values ({nv}) != number of indices ({ni})"
            raise TypeError(errmsg)

    def __str__(self) -> str:
        """A simplified representation of this object."""
        pairs = zip(self.indices, self.values)
        return iterables.show_at_most(3, pairs, separator=', ')


class Coordinates(OrderedPairs):
    """A sequence of index-scalar pairs."""

    __slots__ = ('unit',)

    def __init__(
        self,
        indices: typing.Iterable[int],
        values: typing.Iterable[typing.Any],
        unit: typing.Union[str, quantities.Unit],
    ) -> None:
        super().__init__(indices, values)
        self.unit = unit

    def with_unit(self, new: typing.Union[str, quantities.Unit]):
        """Convert this object to the new unit, if possible."""
        scale = quantities.Unit(new) // self.unit
        self.values = [value * scale for value in self.values]
        self.unit = new
        return self

    def __str__(self) -> str:
        """A simplified representation of this object."""
        values = iterables.show_at_most(3, self.values, separator=', ')
        return f"{values} [{self.unit}]"


IndexLike = typing.TypeVar('IndexLike', bound=Indices)
IndexLike = typing.Union[Indices, OrderedPairs, Coordinates]


class Indexer:
    """A callable object that extracts indices from reference values."""

    def __init__(
        self,
        reference: typing.Iterable[typing.Any],
        size: int=None,
    ) -> None:
        self.reference = reference
        self.size = size or len(self.reference)

    def __call__(self, *user):
        """Create an index object from user input."""
        targets = self._normalize(*user)
        return Indices(targets)

    def _normalize(self, *user):
        """Helper for computing target values from user input."""
        if not user:
            return self.reference
        if isinstance(user[0], slice):
            return iterables.slice_to_range(user[0], stop=self.size)
        if isinstance(user[0], range):
            return user[0]
        return user


class IndexMapper(Indexer):
    """A callable object that maps values to indices."""

    def __init__(
        self,
        reference: typing.Iterable[typing.Any],
        size: int=None,
    ) -> None:
        super().__init__(reference, size=size)
        self.reference = tuple(self.reference)

    def __call__(self, *user):
        targets = self._normalize(*user)
        if all(isinstance(value, numbers.Integral) for value in targets):
            return Indices(targets)
        indices = [self.reference.index(target) for target in targets]
        return OrderedPairs(indices, targets)


class IndexComputer(Indexer):
    """A callable object that computes indices from reference values."""

    def __init__(
        self,
        reference: quantities.Measured,
        size: int=None,
    ) -> None:
        super().__init__(reference, size=size)
        self.unit = reference.unit()
        """The unit of the reference values."""

    def __call__(self, *user):
        targets = self._normalize(*user)
        if all(isinstance(value, numbers.Integral) for value in targets):
            return Indices(targets)
        measured = quantities.measure(*targets)
        vector = quantities.Vector(measured.values, measured.unit)
        values = (
            vector.unit(self.unit)
            if vector.unit().dimension == self.unit.dimension
            else vector
        )
        indices = [
            find_nearest(self.reference, float(value)).index
            for value in values
        ]
        return Coordinates(indices, values, self.unit)


class Axis(iterables.ReprStrMixin):
    """A single dataset axis."""

    Idx = typing.TypeVar('Idx', bound=Indexer)
    Idx = typing.Union[Indexer, IndexMapper, IndexComputer]

    def __init__(self, indexer: Idx) -> None:
        self.indexer = indexer
        self.reference = indexer.reference
        """The reference values used to compute indices."""
        self.size = indexer.size
        """The full length of this axis."""

    def __call__(self, *user, **kwargs):
        """Convert user values into an index object."""
        return self.indexer(*user, **kwargs)

    def __len__(self) -> int:
        """The full length of this axis. Called for len(self)."""
        return self.size

    def __str__(self) -> str:
        """A simplified representation of this object."""
        string = f"size={self.size}"
        unit = (
            str(self.reference.unit())
            if isinstance(self.reference, quantities.Measured)
            else None
        )
        return f"{string} unit={unit!r}"

