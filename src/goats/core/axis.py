import collections
import numbers
import typing

import numpy

from goats.core import aliased
from goats.core import datafile
from goats.core import iterables
from goats.core import metadata
from goats.core import metric
from goats.core import physical
from goats.core import reference
from goats.core import variable


T = typing.TypeVar('T')


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


class Index(iterables.ReprStrMixin, collections.UserList):
    """A sequence of indices representing axis values."""

    def __init__(
        self,
        data: Data,
        unit: metadata.UnitLike=None,
        name: typing.Union[str, typing.Iterable[str]]=None,
    ) -> None:
        super().__init__(list(data.points))
        self._values = data.values
        self._unit = metadata.Unit(unit) if unit is not None else None
        self._name = name

    @property
    def values(self):
        """The axis value at each index."""
        return self._values

    @property
    def unit(self):
        """The metric unit of the corresponding values, if any."""
        return self._unit

    @property
    def name(self):
        """The name of the axis that these indices represent."""
        return self._name

    def __str__(self) -> str:
        """A simplified representation of this object."""
        string = f"{str(self.name)!r}, values={self.values}"
        if self.unit:
            string += f", unit={str(self.unit)!r}"
        return string


class Indexer(iterables.ReprStrMixin):
    """An object that computes axis indices from user values."""

    def __init__(
        self,
        method: typing.Callable[..., Data],
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


class Quantity(metadata.NameMixin, iterables.ReprStrMixin):
    """A callable representation of a single dataset axis."""

    @typing.overload
    def __init__(
        self: Instance,
        __indexer: Indexer,
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
        self._reference = None

    def parse_attrs(
        self,
        this: typing.Union['Quantity', Indexer],
        meta: dict,
        **targets
    ) -> typing.Tuple[Indexer, dict]:
        """Get instance attributes from initialization arguments."""
        if isinstance(this, Quantity):
            return this._indexer, {k: getattr(this, k) for k in targets}
        return this, {k: meta.get(k, v) for k, v in targets.items()}

    @property
    def reference(self):
        """The full array of axis values."""
        if self._reference is None:
            self._reference = self.index().values
        return self._reference

    @property
    def unit(self):
        if self._unit is not None:
            return metadata.Unit(self._unit)

    # NOTE: Duplicate of `measurable.Quantity.__getitem__`.
    def __getitem__(self, arg: metadata.UnitLike):
        """Set the unit of this object's values.
        
        Notes
        -----
        Using this special method to change the unit supports a simple and
        relatively intuitive syntax but is arguably an abuse of notation.
        """
        unit = (
            self.unit.norm[arg]
            if str(arg).lower() in metric.SYSTEMS else arg
        )
        if unit == self._unit:
            return self
        new = self._validate_unit(metadata.Unit(unit))
        return self.apply_unit(new)

    # NOTE: Significant overlap with `measurable.Quantity.apply_unit`.
    def apply_unit(self, unit: metadata.Unit):
        """Create an instance with the new unit."""
        return type(self)(self._indexer, unit=unit, name=self.name)

    # NOTE: Duplicate of `measurable.Quantity._validate_unit`.
    def _validate_unit(self, unit: metadata.UnitLike):
        """Raise an exception if `unit` is inconsistent with this quantity.
        
        The given unit is consistent if it has the same dimension in a known
        metric system as the existing unit.
        """
        if self.unit | unit:
            return unit
        raise ValueError(
            f"The unit {str(unit)!r} is inconsistent with {str(self.unit)!r}"
        ) from None

    def index(self, *args, **kwargs):
        """Convert arguments into an index-like quantity."""
        targets = self._indexer.normalize(*args)
        if self.unit is None:
            # This axis does not represent a measurable quantity, so the
            # index-computing method should not expect a unit and the resulting
            # index object will not have a unit.
            return Index(
                self._indexer.compute(targets, **kwargs),
                name=self.name,
            )
        # This axis represents a measurable quantity, so we want to make sure
        # the index-computing method and the resulting index object use
        # consistent units.
        unit = kwargs.pop('unit', self.unit)
        return Index(
            self._indexer.compute(targets, unit, **kwargs),
            unit=unit,
            name=self.name
        )

    def __str__(self) -> str:
        """A simplified representation of this object."""
        string = f"{str(self.name)!r}"
        if self.unit:
            string += f", unit={str(self.unit)!r}"
        return string


class List(collections.UserList):
    """A sequence of non-measurable axis values."""

    def __init__(self, __q: Quantity) -> None:
        super().__init__(__q.reference)
        self._q = __q
        self._name = None

    @property
    def name(self):
        """The name of the axis quantity."""
        if self._name is None:
            self._name = self._q.name
        return self._name

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
        string = numpy.array2string(
            numpy.array(self),
            suppress_small=True,
            threshold=4,
            edgeitems=2,
            separator=', ',
            prefix=prefix,
            suffix=suffix,
        )
        return f"{prefix}{string}{suffix}"


class Array(physical.Array):
    """A sequence of measurable axis values."""

    def __init__(self, __q: Quantity) -> None:
        super().__init__(__q.reference, unit=__q.unit, name=__q.name)
        self._q = __q

    def at(self, *values, **kwargs):
        """Compute axis indices at the given values.
        
        Notes
        -----
        Unlike `~axis.Quantity.index`, this assumes that `values` are numerical
        values in the current unit.
        """
        return self._q.index([*values, self.unit], **kwargs)

    def __getitem__(self, unit: metadata.UnitLike):
        """Create a new instance with updated unit."""
        return type(self)(self._q[unit])

    def __str__(self) -> str:
        """A simplified representation of this object."""
        return self._as_string(suffix=f', unit={self.unit}')

    def __repr__(self) -> str:
        """An unambiguous representation of this object."""
        module = f"{self.__module__.replace('goats.', '')}."
        name = self.__class__.__qualname__
        return self._as_string(
            prefix=f'{module}{name}(',
            suffix=f', unit={self.unit})',
        )

    def _as_string(self, prefix: str='', suffix: str=''):
        """Create a string representation of this object."""
        def float_formatter(x):
            if abs(x) <= 1e-4:
                return f"{x:.3e}"
            return f"{x:.3f}"
        signed = any(i < 0 for i in numpy.array(self))
        string = numpy.array2string(
            self,
            suppress_small=True,
            threshold=4,
            edgeitems=2,
            sign='+' if signed else '-',
            separator=', ',
            formatter={'float_kind': float_formatter},
            prefix=prefix,
            suffix=suffix,
        )
        return f"{prefix}{string}{suffix}"


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

    def __getitem__(self, __k: str) -> Quantity:
        """Get the named axis object, if possible."""
        unit = self.get_unit(__k)
        name = self.get_name(__k)
        try:
            return Quantity(
                self._get_indexer(__k),
                unit=unit,
                name=name,
            )
        except KeyError as err:
            raise KeyError(
                f"No known indexing method for {__k}"
            ) from err

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
                    return Data(indices)
            except TypeError as err:
                raise IndexTypeError(
                    f"Can't convert {targets!r} to integer indices."
                ) from err
            raise ValueError(
                f"One or more index {targets} is outside the interval"
                " [0, {n-1}]"
            ) from None
        return Indexer(method, n)

    def get_unit(self, key: str):
        """Get the metric unit corresponding to `key`."""
        try:
            unit = self.variables[key].unit
        except KeyError:
            unit = None
        return unit

    def get_name(self, key: str):
        """Get the set of aliases for `key`."""
        return reference.NAMES.get(key, key)

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



