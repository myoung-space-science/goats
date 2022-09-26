import collections.abc
import contextlib
import numbers
import typing

import numpy
import numpy.typing

from goats.core import algebraic
from goats.core import fundamental
from goats.core import iterables
from goats.core import measurable
from goats.core import metadata
from goats.core import metric


Instance = typing.TypeVar('Instance', bound='Quantity')


class Quantity(measurable.Quantity, metadata.NameMixin):
    """A measurable quantity with a name."""

    @typing.overload
    def __init__(
        self: Instance,
        __data: algebraic.Real,
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


class Scalar(Quantity, algebraic.Scalar):
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

    def __iter__(self):
        """Explicitly suppress iteration.
        
        This is necessary because `~measurable.Quantity` uses `__getitem__` for
        unit updates. At the same time, built-in code will attempt to use the
        old-style iterator protocol if it doesn't find an `__iter__` method,
        which includes calling `__getitem__` (cf.
        https://docs.python.org/3/library/collections.abc.html). Defining
        `__iter__` as not implemented circumvents that problem.
        """
        raise TypeError(f"Can't iterate over {type(self)!r}") from None


class Constants(collections.abc.Mapping):
    """Fundamental physical constants in a given metric system."""
    def __init__(self, system: str) -> None:
        self.system = system.lower()
        self._mapping = fundamental.CONSTANTS.copy()

    def __len__(self) -> int:
        """The number of defined constants."""
        return len(self._mapping)

    def __iter__(self) -> typing.Iterator:
        """Iterate over defined constants."""
        return iter(self._mapping)

    def __getitem__(self, name: str):
        """Create the named constant or raise an error."""
        if name in self._mapping:
            found = self._get_attributes(name)
            return Scalar(found['value'], unit=found['unit'])
        raise KeyError(name)

    def _get_attributes(self, name: str) -> dict:
        """Get the value and unit for a named constant, if possible."""
        definition = self._mapping[name]
        if 'all' in definition:
            return {'value': definition['all'], 'unit': None}
        if this := definition.get(self.system):
            return this
        raise ValueError(f"Unknown constant: {name!r}")

    def __repr__(self) -> str:
        """An unambiguous representation of this object."""
        return f"{self.__class__.__qualname__}({self.system})"


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
        self: Instance,
        __data: typing.Union[algebraic.Real, numpy.typing.ArrayLike],
        *,
        unit: metadata.UnitLike=None,
        name: typing.Union[str, typing.Iterable[str]]=None,
    ) -> None: ...

    @typing.overload
    def __init__(
        self: Instance,
        __data: measurable.Measurement,
        *,
        name: typing.Union[str, typing.Iterable[str]]=None,
    ) -> None: ...

    @typing.overload
    def __init__(
        self: Instance,
        scalar: Scalar,
    ) -> None: ...

    @typing.overload
    def __init__(
        self: Instance,
        instance: Instance,
    ) -> None: ...

    def __init__(self, __data, **meta) -> None:
        if isinstance(__data, measurable.Measurement):
            meta = {'unit': __data.unit}
            __data = __data.values
        elif isinstance(__data, Scalar):
            meta = {k: getattr(__data, k) for k in ('unit', 'name')}
            __data = __data.data
        array = numpy.asfarray(list(iterables.whole(__data)))
        super().__init__(array, **meta)

    def __iter__(self) -> typing.Iterator:
        """Called for iter(self)."""
        return iter(self.data)

    def __len__(self) -> int:
        """Called for len(self)."""
        return len(self.data)

    def __getitem__(self, index):
        """Called for index-based value access."""
        if isinstance(index, str):
            return super().__getitem__(index)
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


def scalar(this) -> Scalar:
    """Make sure `this` is a `~physical.Scalar`."""
    if isinstance(this, Scalar):
        return this
    if isinstance(this, Vector) and len(this) == 1:
        return this[0]
    if isinstance(this, measurable.Measurement) and len(this) == 1:
        return Scalar(this.values[0], unit=this.unit)
    measured = measurable.measure(this)
    if len(measured) > 1:
        raise ValueError(
            "Can't create a scalar from a multi-valued quantity"
        ) from None
    return scalar(measured)


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

    def __new__(cls, *args, **kwargs):
        """Create an instance of the appropriate object."""
        if len(args) == 1:
            data = args[0]
            if isinstance(data, numbers.Number):
                return Scalar(data, **kwargs)
        return super().__new__(cls)

    def __init__(self, __data, **meta) -> None:
        """Initialize an array from arguments."""
        super().__init__(__data, **meta)
        self._scale = 1.0
        self._full_array = getattr(__data, '_array', None)
        self._ndim = None
        self._shape = None
        self.display.register(data='_get_data_type')

    @property
    def data(self):
        """The initial data.
        
        This attribute represents the data-containing object used to create this
        instance. It does not necessarily represent the current array data,
        which may differ as a result of changing the instance unit, or due to
        other allowed updates defined on subclasses.
        """
        return super().data

    def _get_data_type(self):
        """Internal helper for displaying the type of data object."""
        return self.data.__class__

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
        try:
            method = self._get_base_attr('__len__')
            return method()
        except AttributeError:
            return len(self.data)

    @property
    def ndim(self) -> int:
        """The number of dimensions in this array."""
        if self._ndim is None:
            self._ndim = self._get_base_attr('ndim')
        return self._ndim

    @property
    def shape(self) -> int:
        """The length of each dimension in this array."""
        if self._shape is None:
            self._shape = self._get_base_attr('shape')
        return self._shape

    def _get_base_attr(self, name: str):
        """Helper method to efficiently access underlying attributes.

        This method will first search the underlying data object for the named
        attribute, to take advantage of viewers that provide metadata without
        loading the full dataset. If that search fails, this method will attempt
        to retrieve the named attribute from the actual dataset array, which may
        require loading from disk.
        """
        for this in ['data', '_array']:
            attr = getattr(self, this)
            with contextlib.suppress(AttributeError):
                if hasattr(attr, name):
                    # avoid false negative when value == 0
                    return getattr(attr, name)
        raise AttributeError(f"Could not find an attribute named {name!r}")

    _builtin = (int, slice, type(...), type(None))

    @typing.overload
    def __getitem__(
        self: Instance,
        unit: typing.Union[str, metric.Unit],
    ) -> Instance: ...

    @typing.overload
    def __getitem__(
        self,
        *args: IndexLike,
    ) -> typing.Union[Scalar, 'Array', Instance]: ...

    def __getitem__(self, *args):
        """Create a new instance from a subset of data."""
        if len(args) == 1 and isinstance(args[0], metadata.UnitLike):
            return super().__getitem__(args[0])
        unwrapped = iterables.unwrap(args)
        if iterables.hastype(unwrapped, self._builtin, tuple, strict=True):
            return self._subscript_standard(unwrapped)
        return self._subscript_custom(unwrapped)

    def apply_unit(self, unit: metadata.UnitLike):
        # Create a copy of this instance.
        new = self._copy_with(unit=unit)
        # Update the new instance's internal `scale` attribute.
        new._scale = self._scale * (unit // self._unit)
        # Return the new instance.
        return new

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
        if all(idx is not Ellipsis for idx in user):
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
        """Convert input object types based on `ufunc`."""
        if ufunc.__name__ == 'power':
            inputs = [
                float(x) if isinstance(x, numbers.Real)
                else x for x in inputs
            ]
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
        """Load and scale the array data corresponding to `index`.
        
        Notes
        -----
        - This method represents the internal interface to the underlying array.
        - General users should subscript instances via the standard bracket
          syntax.
        - This method defers the process of loading the appropriate array
          without any numerical scaling to `self._load_array()`, then rescales
          the result if the unit has changed.
        """
        return self._scale * self._load_array(index)

    def _load_array(self, index=None):
        """Get the unscaled array from disk or memory.

        If `index` is ``None`` or an empty iterable, this method will produce
        the entire array. Otherwise, it will create the requested subarray from
        `self.data`. It will always attempt to use a cached version of the full
        array before loading from disk. The specific algorithm is as follows:

        - If `index` is "missing" (i.e., `None` or an empty iterable object, but
          not 0), the caller wants the full array.
        
            - If we already have the full array, return it.
            - Else, read it, save it, and return it.

        - Else, if `index` is not missing, the caller wants a subarray.

            - If we already have the full array, subscript and return it.
            - Else, continue

        - Else, read and subscript it, and return the subarray.

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
            if self._full_array is not None:
                return self._full_array
            array = self._read_array()
            self._full_array = array
            return array
        if self._full_array is not None:
            idx = numpy.index_exp[index]
            return self._full_array[idx]
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
        meta = {
            k: updates.get(k, getattr(self, k))
            for k in self.meta.parameters
        }
        return type(self)(data, **meta)

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


