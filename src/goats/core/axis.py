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


class Metadata(collections.UserDict):
    """Metadata attributes relevant to a dataset axis."""

    def __init__(
        self,
        name: typing.Union[str, typing.Iterable[str]]=None,
        unit: typing.Union[str, metric.Unit]=None,
    ) -> None:
        attrs = {}
        if name is not None:
            attrs['name'] = metadata.Name(name)
        if unit is not None:
            attrs['unit'] = metadata.Unit(unit)
        super().__init__(attrs)


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


Instance = typing.TypeVar('Instance', bound='Quantity')


class Quantity:
    """A callable representation of a dataset axis."""

    @typing.overload
    def __init__(self, __computer: Indexer, **meta) -> None:
        """Create a new axis from scratch."""

    @typing.overload
    def __init__(self: Instance, instance: Instance) -> None:
        """Create a new axis from an existing instance."""

    def __new__(cls, __a, **meta):
        """Return an existing instance, if applicable."""
        if not meta and isinstance(__a, cls):
            return __a
        return super().__new__(cls)

    def __init__(self, __a: Indexer, **meta) -> None:
        self._indexer = __a
        self._meta = meta
        self._reference = None

    @property
    def name(self) -> metadata.Name:
        """The name of this axis."""
        return self._meta.get('name')

    @property
    def unit(self) -> metadata.Unit:
        """The unit of this axis's values."""
        return self._meta.get('unit')

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
            meta = {**self._meta, 'unit': unit}
            return index.Quantity(data, **meta)
        data = self._indexer.compute(targets, **kwargs)
        return index.Quantity(data, **self._meta)

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
            return type(self)(self._indexer, unit=new, name=self.name)
        raise ValueError(
            f"The unit {str(unit)!r} is inconsistent with {str(self.unit)!r}"
        ) from None

    def __str__(self) -> str:
        """A simplified representation of this object."""
        return ', '.join(f"{k}={str(v)!r}" for k, v in self._meta.items())


class IndexTypeError(Exception):
    """Invalid index argument."""


class Interface(aliased.MutableMapping):
    """An interface to the array axes available from a dataset."""

    def __init__(
        self,
        dataset: datafile.Interface,
        system: typing.Union[str, metric.System]=None,
    ) -> None:
        self.dataset = dataset
        """The dataset to which these axes belong."""
        self._variables = None
        self._system = metric.System(system or 'mks')
        indexers = {
            k: self.build_default(k)
            for k in self.dataset.axes.keys(aliased=True)
        }
        super().__init__(indexers)

    @property
    def variables(self):
        """The variable quantities that these axes support."""
        if self._variables is None:
            self._variables = variable.Interface(
                self.dataset,
                system=self.system,
            )
        return self._variables

    @property
    def system(self):
        """The metric system of these axes."""
        return self._system

    def __getitem__(self, __k: str):
        """Get the named axis object, if possible."""
        try:
            method = self._get_indexer(__k)
        except KeyError as err:
            raise KeyError(
                f"No known indexing method for {__k}"
            ) from err
        meta = self.get_metadata(__k)
        return Quantity(method, **meta)

    def _get_indexer(self, key: str):
        """Get the axis indexer for `key`, or use the default."""
        if key in self.keys():
            return super().__getitem__(key)
        raise KeyError(f"No indexing method for axis {key!r}")

    def build_default(self, key: str):
        """Define the default axis-indexer method."""
        n = self.dataset.axes[key].size
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
                f"One or more index {targets} is outside the interval"
                " [0, {n-1}]"
            ) from None
        return Indexer(method, n)

    def get_metadata(self, key: str):
        """Get metadata attributes corresponding to `key`."""
        name = reference.NAMES.get(key, key)
        try:
            unit = self.variables[key].unit
        except KeyError:
            unit = None
        return Metadata(name=name, unit=unit)

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

