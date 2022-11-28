"""Tools for managing variable quantities in datasets."""

import abc
import collections.abc
import contextlib
import numbers
import typing

import numpy
import numpy.typing

from goats.core import datafile
from goats.core import metric
from goats.core import metadata
from goats.core import physical


Instance = typing.TypeVar('Instance', bound='Quantity')


class Quantity(physical.Array, metadata.AxesMixin):
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
        axes = (
            metadata.Axes(parsed['axes'])
            or [f'x{i}' for i in range(self.ndim)]
        )
        self._axes = tuple(axes)
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
        if isinstance(this, physical.Array) and not isinstance(this, Quantity):
            attrs = {'unit', 'name'}
            meta.update({k: getattr(this, k) for k in attrs})
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
        return super()._ufunc_hook(ufunc, *inputs)

    def array_contains(self, value: numbers.Real):
        """True if `value` is in this variable quantity's array.

        Parameters
        ----------
        value : real
            The value to check for among this variable's values.
        
        Notes
        -----
        This method exists to handle cases in which floating-point arithmetic
        has caused a numeric operation to return an imprecise result, especially
        for small numbers (e.g., converting energy from eV to J). It will first
        check the built-in `__contains__` method via `in` before attempting to
        determine if `value` is close enough to count, albeit within a very
        strict tolerance.
        """
        if value in self._array:
            return True
        if value < numpy.min(self._array) or value > numpy.max(self._array):
            return False
        return numpy.any([numpy.isclose(value, self._array, atol=0.0)])

    @typing.overload
    def __getitem__(
        self: Instance,
        unit: typing.Union[str, metric.Unit],
    ) -> Instance: ...

    @typing.overload
    def __getitem__(
        self,
        *args: physical.IndexLike,
    ) -> typing.Union[physical.Scalar, physical.Array, Instance]: ...

    def __getitem__(self, *args):
        result = super().__getitem__(*args)
        if isinstance(result, physical.Array) and result.ndim == self.axes:
            return Quantity(result, axes=self.axes)
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


@Quantity.implements(numpy.squeeze)
def _squeeze(v: Quantity, **kwargs):
    """Remove singular axes."""
    data = v._array.squeeze(**kwargs)
    axes = tuple(
        a for a, d in zip(v.axes, v.shape)
        if d != 1
    )
    return Quantity(data, unit=v.unit, name=v.name, axes=axes)


@Quantity.implements(numpy.mean)
def _mean(v: Quantity, **kwargs):
    """Compute the mean of the underlying array."""
    data = v._array.mean(**kwargs)
    axis = kwargs.get('axis')
    if axis is None:
        return data
    axes = tuple(a for a in v.axes if v.axes.index(a) != axis)
    name = [f"mean({name})" for name in v.name]
    return Quantity(data, unit=v.unit, name=name, axes=axes)


S = typing.TypeVar('S', bound='Interface')


class Interface(collections.abc.Mapping):
    """Base class for interfaces to dataset variables.
    
    Concrete subclasses must define the `build` method.
    """

    def __init__(
        self,
        __data: datafile.Interface,
        system: typing.Union[str, metric.System]=None,
    ) -> None:
        self._dataset = __data
        self._system = metric.System(system or 'mks')
        self._cache = {}

    @property
    def system(self):
        """This observer's metric system."""
        return self._system

    def __getitem__(self, __k: str) -> Quantity:
        """Retrieve or create the named quantity, if possible."""
        if __k in self._cache:
            return self._cache[__k]
        with contextlib.suppress(KeyError):
            variable = self._dataset.variables[__k]
            if built := self.build(variable):
                return built
        raise KeyError(
            f"No variable quantity corresponding to {__k!r}"
        ) from None

    @abc.abstractmethod
    def build(self, __v: datafile.Variable) -> typing.Optional[Quantity]:
        """Convert a raw variable into a variable quantity.
        
        Concrete implementations should return `None` upon failure.
        """
