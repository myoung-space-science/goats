import typing

from goats.core import aliased
from goats.core import datafile
from goats.core import index
from goats.core import iterables
from goats.core import metadata
from goats.core import reference
from goats.core import variable


Instance = typing.TypeVar('Instance', bound='Quantity')


class Quantity(metadata.NameMixin, iterables.ReprStrMixin):
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
            return metadata.Unit(self._unit)

    def __getitem__(self, unit: metadata.UnitLike):
        """Set the unit of this object's values.
        
        Notes
        -----
        See note at `~measurable.Quantity.__getitem__`.
        """
        if self._unit is None:
            raise TypeError("Can't convert null unit") from None
        if unit != self._unit:
            self._unit = unit
        return self

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
        indexers: typing.Mapping[str, index.Factory],
        dataset: datafile.Interface,
        system: str=None,
    ) -> None:
        self._variables = variable.Interface(dataset, system)
        super().__init__(indexers, keymap=reference.ALIASES)
        self.dataset = dataset

    def resolve(
        self,
        names: typing.Iterable[str],
        mode: str='strict',
    ) -> typing.Tuple[str]:
        """Compute and order the available axes in `names`."""
        axes = self.dataset.available('axes').canonical
        ordered = tuple(name for name in axes if name in names)
        if mode == 'strict':
            return ordered
        extra = tuple(name for name in names if name not in ordered)
        if not extra:
            return ordered
        if mode == 'append':
            return ordered + extra
        raise ValueError(f"Unrecognized mode {mode!r}")

    def __getitem__(self, __k: str) -> Quantity:
        """Get the named axis object, if possible."""
        indexer = super().__getitem__(__k)
        return Quantity(
            indexer,
            unit=self.get_unit(__k),
            name=self.get_name(__k),
        )

    def get_unit(self, key: str):
        """Get the metric unit corresponding to `key`."""
        try:
            unit = self._variables[key].unit
        except KeyError:
            unit = None
        return unit

    def get_name(self, key: str):
        """Get the set of aliases for `key`."""
        return self.alias(key, include=True)



