"""Tools for managing datasets."""

import typing

import numpy
import numpy.typing

from goats.core import aliased
from goats.core import datafile
from goats.core import physical
from goats.core import indexing
from goats.core import iotools
from goats.core import observables
from goats.core import metric
from goats.core import metadata


Instance = typing.TypeVar('Instance', bound='Variable')


class Variable(physical.Array, metadata.AxesMixin):
    """A class representing a dataset variable."""

    @typing.overload
    def __init__(
        self: Instance,
        __data: numpy.typing.ArrayLike,
        *,
        unit: metadata.UnitLike=None,
        name: typing.Union[str, typing.Iterable[str]]=None,
        axes: typing.Iterable[str]=None,
    ) -> None:
        """Create a new variable from scratch."""

    @typing.overload
    def __init__(
        self: Instance,
        __data: physical.Array,
        *,
        axes: typing.Iterable[str]=None,
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


class Indexers(aliased.Mapping):
    """The default collection of axis indexers."""

    def __init__(self, dataset: datafile.Interface) -> None:
        self.dataset = dataset
        self.references = {}
        super().__init__(dataset.axes)

    def __getitem__(self, key: str) -> indexing.Indexer:
        """Get the default indexer for `key`."""
        axis = super().__getitem__(key)
        reference = range(axis.size)
        method = lambda _: indexing.Indices(reference)
        return indexing.Indexer(method, reference)


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

    def __getitem__(self, key: str) -> indexing.Axis:
        indexer = super().__getitem__(key)
        size = self.dataset.axes[key].size
        names = observables.ALIASES.get(key, [key])
        return indexing.Axis(size, indexer, *names)


class Interface:
    """The user interface to a dataset."""

    def __init__(
        self,
        path: iotools.PathLike,
        indexers: typing.Type[Indexers]=None,
    ) -> None:
        self.path = path
        self.view = datafile.Interface(path)
        self.indexers = indexers or Indexers
        self._variables = None
        self._axes = None

    @property
    def variables(self):
        """Objects representing the variables in this dataset."""
        if self._variables is None:
            self._variables = Variables(self.view)
        return self._variables

    @property
    def axes(self):
        """Objects representing the axes in this dataset."""
        if self._axes is None:
            self._axes = Axes(self.view, self.indexers)
        return self._axes

    Default = typing.TypeVar('Default')

    def iter_axes(self, name: str, default: Default=None):
        """Iterate over the axes for the named variable."""
        if name in self.variables:
            return iter(self.variables[name].axes)
        if default is not None:
            return default
        raise ValueError(
            f"Can't iterate over axes of missing variable {name!r}"
        ) from None

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

    def get_indices(self, name: str, **user):
        """Extract indexing objects for the named variable."""
        variable = self.variables.get(name)
        if not variable:
            return ()
        return tuple(
            self.axes[axis](*user.get(axis, ()))
            for axis in variable.axes
        )


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


