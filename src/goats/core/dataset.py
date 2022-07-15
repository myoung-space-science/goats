"""Tools for managing datasets."""

import numbers
import typing

import numpy
import numpy.typing

from goats.core import aliased
from goats.core import datafile
from goats.core import iotools
from goats.core import iterables
from goats.core import measurable
from goats.core import metric
from goats.core import metadata
from goats.core import observables
from goats.core import physical


Instance = typing.TypeVar('Instance', bound='Variable')


class Variable(physical.Array, metadata.AxesMixin):
    """A class representing a dataset variable."""

    @typing.overload
    def __init__(
        self: Instance,
        __data: numpy.typing.ArrayLike,
        *,
        axes: typing.Iterable[str],
        unit: metadata.UnitLike=None,
        name: typing.Union[str, typing.Iterable[str]]=None,
    ) -> None:
        """Create a new variable from scratch."""

    @typing.overload
    def __init__(
        self: Instance,
        __data: physical.Array,
        *,
        axes: typing.Iterable[str],
    ) -> None:
        """Create a new variable from an array."""

    @typing.overload
    def __init__(
        self: Instance,
        instance: Instance,
    ) -> None:
        """Create a new variable from an existing variable."""

    def __init__(self, __data, **meta) -> None:
        super().__init__(__data, **meta)
        parsed = self.parse_attrs(__data, meta, axes=())
        self._axes = metadata.Axes(parsed['axes'])
        if not self._axes:
            raise ValueError("Axes are required")
        self.meta.register('axes')
        self.naxes = len(self.axes)
        """The number of indexable axes in this variable's array."""
        if self.naxes != self.ndim:
            raise ValueError(
                f"Number of axes ({self.naxes})"
                f" must equal number of array dimensions ({self.ndim})"
            )
        self.display.register('axes')
        self.display['__str__'].append("axes={axes}")
        self.display['__repr__'].append("axes={axes}")

    def parse_attrs(self, this, meta: dict, **targets):
        if isinstance(this, physical.Array) and not isinstance(this, Variable):
            meta.update({k: getattr(this, k) for k in ('unit', 'name')})
            this = this.data
        return super().parse_attrs(this, meta, **targets)

    def _ufunc_hook(self, ufunc, *inputs):
        """Convert input arrays into arrays appropriate to `ufunc`."""
        multiplicative = ufunc.__name__ in {'multiply', 'divide', 'true_divide'}
        correct_type = all(isinstance(v, type(self)) for v in inputs)
        if multiplicative and correct_type:
            axes = self.axes.merge(*[v.axes for v in inputs])
            tmp = {}
            for v in inputs:
                tmp.update(v.shape_dict)
            full_shape = tuple(tmp[d] for d in axes)
            indices = numpy.ix_(*[range(i) for i in full_shape])
            arrays = []
            for v in inputs:
                idx = tuple(indices[axes.index(d)] for d in v.shape_dict)
                arrays.append(v._get_array(idx))
            return arrays
        return tuple(
            x._array if isinstance(x, type(self))
            else x for x in inputs
        )
    def __getitem__(self, *args):
        result = super().__getitem__(*args)
        if isinstance(result, physical.Array) and result.ndim == self.axes:
            return Variable(result, axes=self.axes)
        return result

    def __eq__(self, other: typing.Any):
        return (
            super().__eq__(other) if getattr(other, 'axes', None) == self.axes
            else False
        )

    @property
    def shape_dict(self) -> typing.Dict[str, int]:
        """Label and size for each axis."""
        return dict(zip(self.axes, self.shape))

    def _copy_with(self, **updates):
        """Create a new instance from the current attributes."""
        array = super()._copy_with(**updates)
        axes = updates.get('axes', self.axes)
        return Variable(array, axes=axes)


@Variable.implements(numpy.squeeze)
def _squeeze(v: Variable, **kwargs):
    """Remove singular axes."""
    data = v._array.squeeze(**kwargs)
    axes = tuple(
        a for a, d in zip(v.axes, v.shape)
        if d != 1
    )
    return Variable(data, unit=v.unit, name=v.name, axes=axes)


@Variable.implements(numpy.mean)
def _mean(v: Variable, **kwargs):
    """Compute the mean of the underlying array."""
    data = v._array.mean(**kwargs)
    axis = kwargs.get('axis')
    if axis is None:
        return data
    axes = tuple(a for a in v.axes if v.axes.index(a) != axis)
    name = [f"mean({name})" for name in v.name]
    return Variable(data, unit=v.unit, name=name, axes=axes)


S = typing.TypeVar('S', bound='Variables')


class Variables(aliased.Mapping):
    """An interface to dataset variables.
    
    This class provides aliased key-based access to all variables in a dataset.
    It converts each requested dataset variable into a `~dataset.Variable`
    instance with the appropriate MKS unit.
    """

    def __init__(self, dataset: datafile.Interface) -> None:
        known = {
            k: v for k, v in dataset.variables.items(aliased=True)
            if k in observables.METADATA
        }
        super().__init__(known)
        self._system = metric.System('mks')
        self._units = None
        self._cache = {}

    @property
    def units(self):
        """The MKS unit of each variable."""
        if self._units is None:
            units = {
                name: self._get_unit(name)
                for name in self.keys(aliased=True)
            }
            self._units = aliased.Mapping(units)
        return self._units

    def _get_unit(self, name: str):
        """Get a standard unit for the named variable."""
        variable = self._system[observables.METADATA[name]['quantity']]
        return metric.Unit(variable.unit)

    def __getitem__(self, key: str):
        """Create the named variable, if possible."""
        if key in self._cache:
            return self._cache[key]
        datavar = super().__getitem__(key)
        variable = Variable(
            datavar.data,
            unit=standardize(datavar.unit),
            axes=datavar.axes,
            name=observables.ALIASES[key],
        )
        result = variable.convert(self.units[key])
        self._cache[key] = result
        return result


Instance = typing.TypeVar('Instance', bound='Indices')


class Indices(physical.Quantity):
    """A sequence of values that can index an array."""

    @typing.overload
    def __init__(
        self: Instance,
        __indices: typing.Iterable[numbers.Integral],
        *,
        values: typing.Iterable[numbers.Real]=None,
        unit: metadata.UnitLike=None,
        name: typing.Union[str, typing.Iterable[str]]=None,
    ) -> None: ...

    @typing.overload
    def __init__(
        self: Instance,
        instance: Instance,
    ) -> None: ...

    def __init__(self, __indices, **meta) -> None:
        """Initialize this instance from arguments or an existing instance."""
        data = meta.pop('values', __indices)
        super().__init__(data, **meta)
        if not all(isinstance(i, numbers.Integral) for i in __indices):
            raise ValueError("All indices must have integral type") from None
        self.indices = tuple(__indices)
        """The integral index values."""
        self._unit = meta.get('unit')

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


class Indexer:
    """A callable object that generates array indices from user arguments."""

    def __init__(
        self,
        method: typing.Callable[..., Indices],
        reference: numpy.typing.ArrayLike,
    ) -> None:
        self.method = method
        self.reference = reference

    def __call__(self, targets, **kwargs):
        """Call the array-indexing method."""
        return self.method(targets, **kwargs)


Instance = typing.TypeVar('Instance', bound='Axis')


class Axis(iterables.ReprStrMixin, metadata.NameMixin):
    """A single dataset axis."""

    @typing.overload
    def __init__(
        self: Instance,
        indexer: Indexer,
        size: int,
        *,
        name: str=None,
    ) -> None:
        """Create a new axis from scratch."""

    @typing.overload
    def __init__(
        self: Instance,
        instance: Instance,
    ) -> None:
        """Create a new axis from an existing instance."""

    def __init__(self, *args, **kwargs) -> None:
        parsed = self._parse(*args, **kwargs)
        indexer, size, name = parsed
        self.indexer = indexer
        """A callable object that creates indices from user input."""
        self.size = size
        """The full length of this axis."""
        self._name = name
        self.reference = indexer.reference
        """The index reference values."""

    Attrs = typing.TypeVar('Attrs', bound=tuple)
    Attrs = typing.Tuple[
        Indexer,
        int,
        aliased.MappingKey,
    ]

    def _parse(self, *args, **kwargs) -> Attrs:
        """Parse input arguments to initialize this instance."""
        if not kwargs and len(args) == 1 and isinstance(args[0], type(self)):
            instance = args[0]
            return tuple(
                getattr(instance, name)
                for name in ('indexer', 'size', 'name')
            )
        indexer, size = args
        name = metadata.Name(kwargs.get('name') or '')
        return indexer, size, name

    def __call__(self, *args, **kwargs) -> Indices:
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
        string = f"'{self.name}': size={self.size}"
        unit = (
            str(self.reference.unit)
            if isinstance(self.reference, measurable.Quantity)
            else None
        )
        if unit:
            string += f", unit={unit!r}"
        return string


class Indexers(aliased.Mapping):
    """The default collection of axis indexers."""

    def __init__(self, dataset: datafile.Interface) -> None:
        self.dataset = dataset
        self.references = {}
        super().__init__(dataset.axes)

    def __getitem__(self, key: str) -> Indexer:
        """Get the default indexer for `key`."""
        axis = super().__getitem__(key)
        reference = range(axis.size)
        method = lambda _: Indices(reference)
        return Indexer(method, reference)


class Axes(aliased.Mapping):
    """An interface to dataset axes."""

    def __init__(
        self,
        dataset: datafile.Interface,
        factory: typing.Type[Indexers],
    ) -> None:
        indexers = factory(dataset)
        super().__init__(indexers)
        self.dataset = dataset

    def __getitem__(self, key: str) -> Axis:
        indexer = super().__getitem__(key)
        size = self.dataset.axes[key].size
        name = observables.ALIASES.get(key, [key])
        return Axis(indexer, size, name=name)


class Interface:
    """The user interface to a dataset."""

    def __init__(self, path: iotools.PathLike) -> None:
        self.path = path
        self.view = datafile.Interface(path)
        self._variables = None

    @property
    def variables(self):
        """Objects representing the variables in this dataset."""
        if self._variables is None:
            self._variables = Variables(self.view)
        return self._variables

    def resolve_axes(
        self,
        names: typing.Iterable[str],
        mode: str='strict',
    ) -> typing.Tuple[str]:
        """Compute and order the available axes in `names`."""
        axes = self.view.available('axes').canonical
        ordered = tuple(name for name in axes if name in names)
        if mode == 'strict':
            return ordered
        extra = tuple(name for name in names if name not in ordered)
        if not extra:
            return ordered
        if mode == 'append':
            return ordered + extra
        raise ValueError(f"Unrecognized mode {mode!r}")


substitutions = {
    'julian date': 'day',
    'shell': '1',
    'cos(mu)': '1',
    'e-': 'e',
    '# / cm^2 s sr MeV': '# / cm^2 s sr MeV/nuc',
}
"""Conversions from non-standard units."""

T = typing.TypeVar('T')

def standardize(unit: T):
    """Replace this unit string with a standard unit string, if possible.

    This function looks for `unit` in the known conversions and returns the
    standard unit string if it exists. If this doesn't find a standard unit
    string, it just returns the input.
    """
    unit = substitutions.get(str(unit), unit)
    if '/' in unit:
        num, den = str(unit).split('/', maxsplit=1)
        unit = ' / '.join((num.strip(), f"({den.strip()})"))
    return unit


