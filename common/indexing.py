import abc
import collections.abc
import numbers
from typing import *

from goats.common.numerical import find_nearest
from goats.common import iterables
from goats.common import quantities


class Indices(collections.abc.Sequence, iterables.ReprStrMixin):
    """A sequence of indices into data arrays."""

    __slots__ = ('indices',)

    def __init__(self, indices: Iterable[int]) -> None:
        self.indices = tuple(indices)

    def __getitem__(self, __i: SupportsIndex):
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
        indices: Iterable[int],
        values: Iterable[Any],
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
        indices: Iterable[int],
        values: Iterable[Any],
        unit: Union[str, quantities.Unit],
    ) -> None:
        super().__init__(indices, values)
        self.unit = unit

    def to(self, new: Union[str, quantities.Unit]):
        """Convert this object to the new unit, if possible."""
        scale = quantities.Unit(new) // self.unit
        self.values = [value * scale for value in self.values]
        self.unit = new
        return self

    def __str__(self) -> str:
        return f"{super().__str__()} [{self.unit}]"


class Indexer(abc.ABC):
    """Base class for objects that create index objects."""

    def __init__(self, reference: Sized, size: int=None) -> None:
        self.reference = reference
        self.size = size or len(self.reference)

    def __call__(self, *user):
        """Create an index object from user input."""
        targets = self._normalize(*user)
        return self.index(*targets)

    def _normalize(self, *user):
        """Helper for computing target values from user input."""
        if not user:
            return self.reference
        if isinstance(user[0], slice):
            return iterables.slice_to_range(user[0], stop=len(self.reference))
        if isinstance(user[0], range):
            return user[0]
        return user

    @abc.abstractmethod
    def index(self, *targets) -> Indices:
        """Create an index object from user values."""
        return Indices(targets)

    def __repr__(self) -> str:
        """An unambiguous representation of this object."""
        return f"{self.__class__.__qualname__}"


class Trivial(Indexer):
    """"""

    def __init__(self, reference: Iterable[Any], size: int = None) -> None:
        super().__init__(tuple(reference), size=size)

    def index(self, *targets):
        if all(index in self.reference for index in targets):
            return Indices(targets)
        raise IndexError("Invalid target indices.")


class Mapped(Indexer):
    """"""

    def __init__(self, reference: Iterable[Any], size: int = None) -> None:
        super().__init__(reference, size=size)
        self.reference = tuple(self.reference)

    def index(self, *targets):
        indices = [self._get_index(target) for target in targets]
        return OrderedPairs(indices, targets)

    def _get_index(self, value: Any) -> int:
        """Get an appropriate index for the given value."""
        if isinstance(value, numbers.Integral):
            return int(value)
        return self.reference.index(value)


class Measured(Indexer):
    """"""

    def __init__(
        self,
        reference: quantities.Measured,
        size: int=None,
    ) -> None:
        super().__init__(reference, size=size)
        self.unit = reference.unit
        """The unit of the reference values."""

    def index(self, *targets):
        vector = quantities.measure(*targets).asvector
        values = (
            vector.to(self.unit).values
            if vector.unit.dimension == self.unit.dimension
            else vector.values
        )
        indices = [self._get_index(value) for value in values]
        return Coordinates(indices, values, self.unit)

    def _get_index(self, value: Union[SupportsFloat, int]) -> int:
        """Get an appropriate index for the given value."""
        if isinstance(value, numbers.Integral):
            return int(value)
        pair = find_nearest(self.reference, float(value))
        return pair[0]


class Axis(iterables.ReprStrMixin):
    """A single dataset axis."""

    def __init__(self, indexer: Indexer) -> None:
        self.indexer = indexer
        self.reference = indexer.reference
        """The reference values used to compute indices."""
        self.size = indexer.size
        """The full length of this axis."""

    def __call__(self, *user):
        """Convert user values into an index object."""
        return self.indexer(*user)

    def __len__(self) -> int:
        """The full length of this axis. Called for len(self)."""
        return self.size

    def __str__(self) -> str:
        """A simplified representation of this object."""
        return f"size={self.size} type={self.indexer!r}"

