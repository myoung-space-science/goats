import collections.abc
import contextlib
import numbers
import typing

import numpy
import numpy.typing

from goats.core import aliased
from goats.core import iterables
from goats.core import measurable
from goats.core import metadata


Instance = typing.TypeVar('Instance', bound='Quantity')


class Quantity(measurable.Quantity, metadata.NameMixin):
    """A measurable quantity with a name."""

    @typing.overload
    def __init__(
        self: Instance,
        __data: measurable.Real,
        *,
        unit: metadata.UnitLike=None,
        name: typing.Union[str, typing.Iterable[str]]=None,
    ) -> None: ...

    @typing.overload
    def __init__(
        self: Instance,
        instance: Instance,
    ) -> None: ...

    def __init__(self, __data, **meta) -> None:
        """Initialize this instance from arguments or an existing instance."""
        super().__init__(__data, **meta)
        parsed = self.parse_attrs(__data, meta, name='')
        self._name = metadata.Name(parsed['name'])
        self.meta.register('name')
        if self._name:
            self.display.register('name')
            self.display['__str__'].insert(0, "'{name}':")
            self.display['__repr__'].insert(2, "name='{name}'")


class Scalar(Quantity, measurable.ScalarOperators):
    """A single-valued named quantity."""

    @typing.overload
    def __init__(
        self: Instance,
        __data: numbers.Real,
        *,
        unit: metadata.UnitLike=None,
        name: typing.Union[str, typing.Iterable[str]]=None,
    ) -> None: ...

    @typing.overload
    def __init__(
        self: Instance,
        instance: Instance,
    ) -> None: ...

    def __init__(self, __data, **meta) -> None:
        super().__init__(float(__data), **meta)


class Vector(Quantity):
    """A multi-valued named quantity.

    Notes
    -----
    This class converts `data` into a `numpy.ndarray` via a `list` during
    instantiation. It may therefore become a bottleneck for large 1-D objects
    and may produce unexpected results for higher-dimension objects. In those
    cases, an array-like class derived from `~measurable.Quantified` that
    incorporates native `numpy` operators may be more appropriate.
    """

    @typing.overload
    def __init__(
        self,
        __data: typing.Union[measurable.Real, numpy.typing.ArrayLike],
        *,
        unit: metadata.UnitLike=None,
        name: typing.Union[str, typing.Iterable[str]]=None,
    ) -> None: ...

    @typing.overload
    def __init__(
        self: Instance,
        instance: Instance,
    ) -> None: ...

    def __init__(self, __data, **meta) -> None:
        array = numpy.asfarray(list(iterables.whole(__data)))
        super().__init__(array, **meta)

    def __len__(self) -> int:
        """Called for len(self)."""
        return len(self.data)

    def __getitem__(self, index):
        """Called for index-based value access."""
        if isinstance(index, typing.SupportsIndex) and index < 0:
            index += len(self)
        values = self.data[index]
        iter_values = isinstance(values, typing.Iterable)
        unit = self.unit
        return (
            [Scalar(value, unit=unit) for value in values] if iter_values
            else Scalar(values, unit=unit)
        )

    def __eq__(self, other: typing.Any):
        """True if two instances have the same data and attributes."""
        if not isinstance(other, type(self)):
            return NotImplemented
        if not self.unit == other.unit:
            return False
        return numpy.array_equal(other, self)


IndexLike = typing.TypeVar(
    'IndexLike',
    typing.Iterable[int],
    slice,
    type(Ellipsis),
)
IndexLike = typing.Union[typing.Iterable[int], slice, type(Ellipsis)]


Instance = typing.TypeVar('Instance', bound='Array')


class Array(numpy.lib.mixins.NDArrayOperatorsMixin, Quantity):
    """Base class for array-like objects."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._scale = 1.0
        self._cached = {}
        self._ndim = None
        self._shape = None
        self.display.register(data='_array')

    def apply_conversion(self, new: metadata.UnitLike):
        self._cached['scale'] = self._scale
        self._scale *= new // self._unit

    def __measure__(self):
        """Create a measurement from this array's data and unit."""
        # NOTE: This may produce unexpected results when `self.ndim` > 1.
        return measurable.Measurement(self._array, self.unit)

    def __eq__(self, other: typing.Any):
        """True if two instances have the same data and attributes."""
        if not isinstance(other, type(self)):
            return NotImplemented
        if not self.unit == other.unit:
            return False
        return numpy.array_equal(other, self)

    def __len__(self):
        """Called for len(self)."""
        if method := self._get_base_attr('__len__', '_array'):
            return method()
        return len(self.data)

    @property
    def ndim(self) -> int:
        """The number of dimensions in this array."""
        if self._ndim is None:
            self._ndim = self._get_base_attr('ndim', 'data', '_array')
        return self._ndim

    @property
    def shape(self) -> int:
        """The length of each dimension in this array."""
        if self._shape is None:
            self._shape = self._get_base_attr('shape', 'data', '_array')
        return self._shape

    def _get_base_attr(self, name: str, *search: str):
        """Helper method to efficiently access underlying attributes.

        This method will first search the underlying data object for the named
        attribute, to take advantage of viewers that provide metadata without
        loading the full dataset. If that search fails, this method will attempt
        to retrieve the named attribute from the actual dataset array.
        """
        targets = [getattr(self, name) for name in search]
        for target in targets:
            with contextlib.suppress(AttributeError):
                attr = target() if callable(target) else target
                if value := getattr(attr, name):
                    return value

    _builtin = (int, slice, type(...))

    def __getitem__(self, *args: IndexLike):
        """Create a new instance from a subset of data."""
        unwrapped = iterables.unwrap(args)
        if self._types_match(unwrapped, self._builtin):
            return self._subscript_standard(unwrapped)
        return self._subscript_custom(unwrapped)

    def _types_match(self, args, types):
        """True if `args` is one `types` or a collection of `types`."""
        return (
            isinstance(args, types)
            or all(isinstance(arg, types) for arg in args)
        )

    def _subscript_standard(self, indices):
        """Perform standard array subscription.

        This method handles cases involving slices, an ellipsis, or integers,
        including v[:], v[...], v[i, :], v[:, j], and v[i, j], where i and j are
        integers.
        """
        result = self._get_array(indices)
        Type = Scalar if isinstance(result, numbers.Number) else Array
        return Type(result, unit=self.unit, name=self.name)

    def _subscript_custom(self, args):
        """Perform array subscription specific to this object.

        This method handles all cases that don't meet the criteria for
        `_subscript_standard`.
        """
        if not isinstance(args, (tuple, list)):
            args = [args]
        expanded = self._expand_ellipsis(args)
        idx = [
            range(self.shape[i])
            if isinstance(arg, slice) else arg
            for i, arg in enumerate(expanded)
        ]
        indices = numpy.ix_(*list(idx))
        return self._copy_with(data=self._get_array(indices))

    def _expand_ellipsis(
        self,
        user: typing.Sequence,
    ) -> typing.Tuple[slice, ...]:
        """Expand an ``Ellipsis`` into one or more ``slice`` objects."""
        if Ellipsis not in user:
            return user
        length = self.ndim - len(user) + 1
        start = user.index(Ellipsis)
        return (
            *user[slice(start)],
            *([slice(None)] * length),
            *user[slice(start+length, self.ndim)],
        )

    _HANDLED_TYPES = (numpy.ndarray, numbers.Number, list)

    def __array_ufunc__(self, ufunc, method, *args, **kwargs):
        """Provide support for `numpy` universal functions.

        See https://numpy.org/doc/stable/reference/arrays.classes.html for more
        information on use of this special method.

        See
        https://numpy.org/doc/stable/reference/generated/numpy.lib.mixins.NDArrayOperatorsMixin.html#numpy.lib.mixins.NDArrayOperatorsMixin
        for the specific implementation example that this class follows.

        Notes
        -----
        This method first ensures that the input types (as well as the type of
        `out`, if passed via keyword) are supported types. It then extracts
        appropriate operands from any arguments that are an instance of this
        class or a subclass and applies `ufunc`. If `ufunc` is one of a
        pre-defined set of methods that would cause an attribute of this class
        to become ambiguous or undefined, this method return the result of
        applying `ufunc`. Otherwise, this method will return a new instance with
        zero or more attributes that have been modified in a `ufunc`-specific
        way (e.g., multiplying two instances' units when `ufunc` is 'multiply').
        The default behavior is therefore to return a new instance with
        unmodified attributes.
        """
        out = kwargs.get('out', ())
        for x in args + out:
            if not isinstance(x, self._HANDLED_TYPES + (Array,)):
                return NotImplemented
        if out:
            kwargs['out'] = tuple(
                x._array if isinstance(x, Array)
                else x for x in out
            )
        name = ufunc.__name__
        operands = self._ufunc_hook(ufunc, *args)
        compute = getattr(ufunc, method)
        data = compute(*operands, **kwargs)
        evaluate = (
            self.meta[name].evaluate if name in self.meta
            else self.meta.implement(compute)
        )
        kwds = {k: v for k, v in kwargs.items() if k != 'out'}
        meta = evaluate(*args, **kwds)
        if type(data) is tuple:
            return tuple(
                self._new_from_func(x, updates=meta)
                for x in data
            )
        if method == 'at':
            return None
        return self._new_from_func(data, updates=meta)

    def _ufunc_hook(self, ufunc, *inputs):
        """Convert input arrays into arrays appropriate to `ufunc`."""
        return tuple(
            x._array if isinstance(x, Array)
            else x for x in inputs
        )

    _HANDLED_FUNCTIONS = {}

    def __array_function__(self, func, types, args, kwargs):
        """Provide support for functions in the `numpy` public API.

        See https://numpy.org/doc/stable/reference/arrays.classes.html for more
        information of use of this special method. The implementation shown here
        is a combination of the example on that page and code from the
        definition of `EncapsulateNDArray.__array_function__` in
        https://github.com/dask/dask/blob/main/dask/array/tests/test_dispatch.py

        The initial `issubclass` check allows subclasses that don't override
        `__array_function__` to handle objects of this type.
        """
        accepted = (Array, numpy.ndarray, numpy.ScalarType)
        if not all(issubclass(ti, accepted) for ti in types):
            return NotImplemented
        if func in self._HANDLED_FUNCTIONS:
            result = self._HANDLED_FUNCTIONS[func](*args, **kwargs)
            return self._new_from_func(result)
        args = tuple(
            arg._array if isinstance(arg, Array)
            else arg for arg in args
        )
        types = tuple(
            ti for ti in types
            if not issubclass(ti, Array)
        )
        data = self._array
        arr = data.__array_function__(func, types, args, kwargs)
        return self._new_from_func(arr)

    def _new_from_func(self, result, updates: dict=None):
        """Create a new instance from the result of a `numpy` function.
        
        If `result` is a `numpy.ndarray` and `updates` is a (possibly empty)
        `dict`, this method will create a new instance. Otherwise, it will
        return `result` as-is.
        """
        if isinstance(result, numpy.ndarray) and isinstance(updates, dict):
            return self._copy_with(data=result, **updates)
        return result

    def __array__(self, *args, **kwargs) -> numpy.ndarray:
        """Support casting to `numpy` array types.
        
        Notes
        -----
        This will retrieve the underlying array before applying `*args` and
        `**kwargs`, in order to avoid a `TypeError` when using
        `netCDF4.Dataset`. See
        https://github.com/mcgibbon/python-examples/blob/master/scripts/file-io/load_netCDF4_full.py
        """
        data = self._array
        return numpy.asanyarray(data, *args, **kwargs)

    @property
    def _array(self):
        """The full data array for internal use."""
        return self._get_array()

    def _get_array(self, index: IndexLike=None):
        """Access scaled array data via index or slice notation.
        
        Notes
        -----
        - This method represents the internal interface to the underlying array.
        - General users should subscript instances via the standard bracket
          syntax.
        - This method defers the process of loading the appropriate array
          without any numerical scaling to `self._load_array()`, then rescales
          the result if the unit has changed.
        """
        array = self._load_array(index)
        if self._scale == self._cached.get('scale'):
            return array
        return self._scale * array

    def _load_array(self, index=None):
        """Get the unscaled array from disk or memory.

        If `index` is ``None`` or an empty iterable, this method will produce
        the entire array. Otherwise, it will create the requested subarray from
        `self.data`. It will always attempt to use a cached version of the full
        array before loading from disk.
        
        The reasoning behind this algorithm is as follows: If we need to load
        the full array at any point, we may as well save it because subscripting
        an in-memory `numpy.ndarray` is much faster than re-reading from disk
        for large arrays. However, we should avoid reading in the full array if
        the caller only wants a small portion of it. We don't cache these
        subarrays because reusing a subarray is only meaningful if the indices
        haven't changed. Furthermore, accessing a subarray via bracket syntax
        creates a new object, at which point the subarray becomes the new
        object's full array.
        """
        if iterables.missing(index):
            if 'full_array' in self._cached:
                return self._cached['full_array']
            array = self._read_array()
            self._cached['full_array'] = array
            return array
        if 'full_array' in self._cached:
            idx = numpy.index_exp[index]
            return self._cached['full_array'][idx]
        return self._read_array(index)

    def _read_array(self, index: IndexLike=None):
        """Read the array data from disk.
        
        If `index` is "missing" in the sense defined by `~iterables.missing`
        this method will load and return the full array. If `index` is not
        missing, this method will first attempt to subscript `self.data` before
        converting it to an array and returning it. If it catches either a
        `TypeError` or an `IndexError`, it will create the full array before
        subscripting and returning it. The former may occur if `self.data` is a
        sequence type like `list`, `tuple`, or `range`; the latter may occur
        when attempting to subscript certain array-like objects (e.g.,
        `netCDF4._netCDF4.Variable`) with valid `numpy` index expressions.
        """
        if not iterables.missing(index):
            idx = numpy.index_exp[index]
            try:
                return numpy.asarray(self.data[idx])
            except (TypeError, IndexError):
                return numpy.asarray(self.data)[idx]
        return numpy.asarray(self.data)

    def _copy_with(self, **updates):
        """Create a new instance from the current attributes."""
        data = updates.get('data', self.data)
        unit = updates.get('unit', self.unit)
        name = updates.get('name', self.name)
        return Array(data, unit=unit, name=name)

    @classmethod
    def implements(cls, numpy_function):
        """Register an `__array_function__` implementation for this class.

        See https://numpy.org/doc/stable/reference/arrays.classes.html for the
        suggestion on which this method is based.

        EXAMPLE
        -------
        Overload `numpy.mean` for an existing class called ``Array`` with a
        version that accepts no keyword arguments::

            @Array.implements(numpy.mean)
            def mean(a: Array, **kwargs) -> Array:
                if kwargs:
                    msg = "Cannot pass keywords to numpy.mean with Array"
                    raise TypeError(msg)
                return numpy.sum(a) / len(a)

        This will compute the mean of the underlying data when called with no
        arguments, but will raise an exception when called with arguments:

            >>> v = Array([[1, 2], [3, 4]])
            >>> numpy.mean(v)
            5.0
            >>> numpy.mean(v, axis=0)
            ...
            TypeError: Cannot pass keywords to numpy.mean with Array
        """
        def decorator(func):
            cls._HANDLED_FUNCTIONS[numpy_function] = func
            return func
        return decorator


Instance = typing.TypeVar('Instance', bound='Variable')


class Variable(Array, metadata.AxesMixin):
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
        __data: Array,
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
        if isinstance(this, Array) and not isinstance(this, Variable):
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
    def __getitem__(self, *args: IndexLike):
        result = super().__getitem__(*args)
        if isinstance(result, Array) and result.ndim == self.axes:
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


class Assumption(Vector):
    """A measurable parameter argument.
    
    This object behaves like a vector in the sense that it is a multi-valued
    measurable quantity with a unit, but users can also cast a single-valued
    assumption to the built-in `int` and `float` types.
    """

    def __getitem__(self, index):
        values = super().__getitem__(index)
        iter_values = isinstance(values, typing.Iterable)
        return (
            [
                Scalar(value, unit=self.unit, name=self.name)
                for value in values
            ] if iter_values
            else Scalar(values, unit=self.unit, name=self.name)
        )

    def __float__(self):
        """Represent a single-valued measurement as a `float`."""
        return self._cast_to(float)

    def __int__(self):
        """Represent a single-valued measurement as a `int`."""
        return self._cast_to(int)

    Numeric = typing.TypeVar('Numeric', int, float)

    def _cast_to(self, __type: typing.Type[Numeric]) -> Numeric:
        """Internal method for casting to numeric type."""
        nv = len(self.data)
        if nv == 1:
            return __type(self.data[0])
        errmsg = f"Can't convert measurement with {nv!r} values to {__type}"
        raise TypeError(errmsg) from None


class Option(metadata.NameMixin, iterables.ReprStrMixin):
    """An unmeasurable parameter argument."""

    def __init__(
        self,
        __value,
        name: typing.Union[str, typing.Iterable[str]]=None,
    ) -> None:
        self.value = __value
        """The value of this optional parameter."""
        self._name = aliased.MappingKey(name or '')
        self.display.register('value')
        self.display['__str__'] = "{value}"
        self.display['__repr__'] = "{value}"
        self.display['__repr__'].separator = ', '
        if self._name:
            self.display.register('name')
            self.display['__str__'].insert(0, "'{name}': ")
            self.display['__repr__'].insert(1, "'{name}'")

    def __eq__(self, other):
        """True if `other` is equivalent to this option's value."""
        if isinstance(other, Option):
            return other.value == self.value
        return other == self.value


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
            super().__eq__(other) and self._unit == other._unit
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


