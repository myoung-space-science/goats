import typing

from goats.core import aliased
from goats.core import index
from goats.core import iterables
from goats.core import metadata
from goats.core import observables
from goats.core import variable


Instance = typing.TypeVar('Instance', bound='Quantity')


class Metadata(
    metadata.UnitMixin,
    metadata.NameMixin,
): ...

class Quantity(Metadata, iterables.ReprStrMixin):
    """A callable representation of a single dataset axis."""

    @typing.overload
    def __init__(
        self: Instance,
        __indexer: index.Factory,
        *,
        unit: metadata.UnitLike=None,
        name: typing.Union[str, typing.Iterable[str]]=None,
    ) -> None:
        """Create a new axis from scratch."""

    @typing.overload
    def __init__(
        self: Instance,
        instance: Instance,
    ) -> None:
        """Create a new axis from an existing instance."""

    def __init__(self, __a, **meta) -> None:
        try:
            self._unit = meta.pop('unit')
        except KeyError:
            self._unit = None
        self._indexer, kwd = self.parse_attrs(__a, meta, name='')
        self._name = metadata.Name(kwd['name'])
        self.size = self._indexer.size
        """The full length of this axis."""
        self.reference = self._indexer.reference
        """The index reference values."""

    def parse_attrs(
        self,
        this: typing.Union['Quantity', index.Factory],
        meta: dict,
        **targets
    ) -> typing.Tuple[index.Factory, dict]:
        """Get instance attributes from initialization arguments."""
        if isinstance(this, Quantity):
            return this._indexer, {k: getattr(this, k) for k in targets}
        return this, {k: meta.get(k, v) for k, v in targets.items()}

    @property
    def unit(self):
        if self._unit is not None:
            return super().unit

    def apply_conversion(self, new: metadata.Unit):
        if self._unit is not None:
            return super().apply_conversion(new)
        raise TypeError("Can't convert null unit") from None

    def at(self, *args, **kwargs):
        """Convert arguments into an index quantity."""
        return self._indexer(*args, **kwargs)

    def __len__(self) -> int:
        """The full length of this axis. Called for len(self)."""
        return self.size

    def __str__(self) -> str:
        """A simplified representation of this object."""
        string = f"'{self.name}': size={self.size}"
        if self.unit:
            string += f", unit={self.unit}"
        return string


class Interface(aliased.Mapping):
    """An interface to the array axes available from a dataset."""

    def __init__(
        self,
        variables: variable.Interface,
        indexers: typing.Mapping[str, index.Factory],
    ) -> None:
        self._variables = variables
        super().__init__(indexers, keymap=observables.ALIASES)

    def subscript(self, v: variable.Quantity, **user):
        """Extract the appropriate sub-variable."""
        idx = tuple(self[axis].at(*user.get(axis, ())) for axis in v.axes)
        return v[idx]

    def __getitem__(self, __k: str) -> Quantity:
        """Get the named axis object, if possible."""
        indexer = super().__getitem__(__k)
        data = self._variables.get(__k)
        return Quantity(indexer, unit=data.unit, name=data.name)

