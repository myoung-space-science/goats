"""Tools for managing variable quantities in datasets."""

import contextlib
import numbers
import typing

import numpy
import numpy.typing

from goats.core import algebraic
from goats.core import aliased
from goats.core import datafile
from goats.core import iterables
from goats.core import measurable
from goats.core import metric
from goats.core import metadata
from goats.core import utilities


IndexLike = typing.TypeVar(
    'IndexLike',
    typing.Iterable[int],
    slice,
    type(Ellipsis),
)
IndexLike = typing.Union[typing.Iterable[int], slice, type(Ellipsis)]


Instance = typing.TypeVar('Instance', bound='Array')


class Array(numpy.lib.mixins.NDArrayOperatorsMixin, measurable.Quantified):
    """The array-like base of each variable quantity."""

    def __init__(self, __interface) -> None:
        # Initialize the parent class.
        super().__init__(
            __interface.data
            if isinstance(__interface, measurable.Quantified)
            else __interface
        )
        # Carry the internal array from an existing instance.
        self._full_array = getattr(__interface, '_array', None)
        # Reset the internal array scale factor.
        self._scale = 1.0
        # Initialize other attributes to `None`.
        self._ndim = None
        self._shape = None
        # Set up algebraic operators for metadata attributes.
        self.meta['true divide'].suppress(algebraic.Real, algebraic.Quantity)
        self.meta['power'].suppress(algebraic.Quantity, algebraic.Quantity)
        self.meta['power'].suppress(algebraic.Real, algebraic.Quantity)
        self.meta['power'].suppress(
            algebraic.Quantity,
            typing.Iterable,
            symmetric=True
        )

    @property
    def data(self):
        """The initial data.
        
        This attribute represents the data-containing object used to create this
        instance. It does not necessarily represent the current array data,
        which may differ as a result of changing the instance unit, or due to
        other allowed updates defined on subclasses.
        """
        return super().data

    def __eq__(self, other: typing.Any):
        """True if two instances have equivalent data arrays."""
        if not isinstance(other, Array):
            return NotImplemented
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

    def __getitem__(self, *args):
        """Extract a subarray."""
        unwrapped = iterables.unwrap(args)
        indices = self._normalize_indices(unwrapped)
        array = self._get_array(indices)
        result = numpy.array(array, ndmin=self.ndim)
        if array.ndim != self.ndim:
            with contextlib.suppress(TypeError):
                shape = [
                    self._get_axis_size(i, v)
                    for i, v in enumerate(indices)
                ]
                return self._copy_with(data=result.reshape(shape))
        return self._copy_with(data=result)

    def _get_axis_size(self, i: int, v):
        """Helper for computing shape in `__getitem__`."""
        if isinstance(v, int):
            return 1
        if isinstance(v, slice):
            return (v.stop or self.shape[i]) - (v.start or 0)
        return self.shape[i]

    def _normalize_indices(self, args):
        """Compute appropriate array indices from `args`.
        
        If the indices in `args` have a standard form involving slices, an
        ellipsis, or integers (including v[:], v[...], v[i, :], v[:, j], and
        v[i, j], where i and j are integers), this method will immediately
        return them. Otherwise, it will extract a sequence of indices that
        represents the original dimensions of the data.
        """
        if iterables.hastype(args, self._builtin, tuple, strict=True):
            return args
        if not isinstance(args, (tuple, list)):
            args = [args]
        expanded = self._expand_ellipsis(args)
        indices = [
            range(self.shape[i])
            if isinstance(arg, slice) else arg
            for i, arg in enumerate(expanded)
        ]
        return numpy.ix_(*list(indices))

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
        """Convert input arrays into arrays appropriate to `ufunc`."""
        if ufunc.__name__ == 'power':
            inputs = [
                float(x) if isinstance(x, numbers.Real)
                else x for x in inputs
            ]
        return tuple(
            x._array if isinstance(x, Array)
            else x for x in inputs
        )

    def array_contains(self, value: numbers.Real):
        """True if `value` is in this variable quantity's array.

        Parameters
        ----------
        value : real
            The value to check for among this variable's values.
        
        Notes
        -----
        * This method exists to handle cases in which floating-point arithmetic
          has caused a numeric operation to return an imprecise result,
          especially for small numbers (e.g., converting energy from eV to J).
          It will first check the built-in `__contains__` method via `in` before
          attempting to determine if `value` is close enough to count, albeit
          within a very strict tolerance.
        """
        if value in self:
            return True
        if value < numpy.min(self._array) or value > numpy.max(self._array):
            return False
        return numpy.any([numpy.isclose(value, self._array, atol=0.0)])

    def __contains__(self, __x):
        """Called for x in self.
        
        See Also
        --------
        array_contains
            Perform additional containment checks beyond this method.
        """
        return __x in self._array

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


Instance = typing.TypeVar('Instance', bound='Quantity')


class Quantity(Array):
    """An array-like quantity with a unit and dimensions."""

    @typing.overload
    def __init__(
        self: Instance,
        __data: numpy.typing.ArrayLike,
        dimensions: typing.Iterable[str]=None,
        unit: metadata.UnitLike=None,
    ) -> None:
        """Create a new variable from scratch."""

    @typing.overload
    def __init__(
        self: Instance,
        instance: Instance,
    ) -> None:
        """Create a new variable from an existing variable."""

    def __init__(
        self,
        __interface,
        dimensions: typing.Iterable[str]=None,
        unit: metadata.UnitLike=None,
    ) -> None:
        # Initialize the parent class.
        super().__init__(__interface)
        # Initialize and register the unit.
        tmp = getattr(__interface, 'unit', unit or '1')
        self._unit = metadata.Unit(tmp)
        self.meta.register('unit')
        # Initialize and register the dimensions.
        tmp = getattr(
            __interface,
            'dimensions',
            dimensions or [f'x{i}' for i in range(self.ndim)]
        )
        self._dimensions = metadata.Dimensions(*tmp)
        self.meta.register('dimensions')
        # Check for consistency between the given dimensions and the dimensions
        # of the data array.
        ndim = len(self._dimensions)
        if ndim != self.ndim:
            raise ValueError(
                f"Number of given dimensions ({ndim})"
                f" must equal number of array dimensions ({self.ndim})"
            )

    def __repr__(self) -> str:
        values = numpy.array2string(
            numpy.array(self),
            threshold=4,
            edgeitems=2,
            separator=', ',
            precision=3,
            floatmode='maxprec_equal',
        )
        parts = [
            f"dimensions={self.dimensions}",
            f"unit={str(self.unit)!r}",
        ]
        return f",\n".join([values, *parts])

    @property
    def unit(self):
        """This quantity's metric unit."""
        return self._unit

    @property
    def dimensions(self):
        """The quantity's indexable dimensions."""
        return self._dimensions

    def __measure__(self):
        """Create a measurement from this variable's data and unit."""
        # NOTE: This may produce unexpected results when `self.ndim` > 1.
        return measurable.Measurement(self._array, self.unit)

    def __eq__(self, other: typing.Any):
        """True if two instances have the same data and attributes."""
        if not isinstance(other, Quantity):
            return NotImplemented
        for name in 'unit', 'dimensions':
            if not utilities.equal_attrs(name, self, other):
                return False
        return super().__eq__(other)

    @typing.overload
    def __getitem__(
        self: Instance,
        unit: typing.Union[str, metric.Unit],
    ) -> Instance: ...

    @typing.overload
    def __getitem__(
        self: Instance,
        *args: IndexLike,
    ) -> Instance: ...

    def __getitem__(self, *args):
        """Set a new unit or extract a subarray.
        
        Notes
        -----
        Using this special method to change the unit supports a simple and
        relatively intuitive syntax but is arguably an abuse of notation.
        """
        if len(args) == 1 and isinstance(args[0], metadata.UnitLike):
            return self._update_unit(args[0])
        return self._copy_with(data=super().__getitem__(*args))

    def _update_unit(self, arg: metadata.UnitLike):
        """Set the unit of this object's values.
        
        Raises
        ------
        ValueError
            The proposed unit is inconsistent with this quantity. A proposed
            unit is consistent if it has the same dimension in a known metric
            system as the existing unit.
        """
        this = (
            self.unit.norm[arg]
            if str(arg).lower() in metric.SYSTEMS else arg
        )
        if this == self.unit:
            # If it's the current unit, there's nothing to update.
            return self
        unit = metadata.Unit(this)
        if not (self.unit | unit):
            # The proposed unit is inconsistent with the current unit.
            raise ValueError(
                f"The unit {str(unit)!r} is inconsistent"
                f" with {str(self.unit)!r}"
            ) from None
        # Create a copy of this instance.
        new = self._copy_with(unit=unit)
        # Update the new instance's internal `scale` attribute.
        new._scale = self._scale * (unit // self._unit)
        # Return the new instance.
        return new

    def _ufunc_hook(self, ufunc, *inputs):
        """Convert input arrays into arrays appropriate to `ufunc`."""
        multiplicative = ufunc.__name__ in {'multiply', 'divide', 'true_divide'}
        correct_type = all(isinstance(v, type(self)) for v in inputs)
        if multiplicative and correct_type:
            dims = self.dimensions.merge(*[v.dimensions for v in inputs])
            tmp = {}
            for v in inputs:
                tmp.update(v.shape_dict)
            full_shape = tuple(tmp[d] for d in dims)
            indices = numpy.ix_(*[range(i) for i in full_shape])
            arrays = []
            for v in inputs:
                idx = tuple(indices[dims.index(d)] for d in v.shape_dict)
                arrays.append(v._get_array(idx))
            return arrays
        return super()._ufunc_hook(ufunc, *inputs)

    @property
    def shape_dict(self) -> typing.Dict[str, int]:
        """Label and size for each axis."""
        return dict(zip(self.dimensions, self.shape))


@Quantity.implements(numpy.squeeze)
def _squeeze(v: Quantity, **kwargs):
    """Remove singular axes."""
    data = v._array.squeeze(**kwargs)
    dimensions = tuple(
        a for a, d in zip(v.dimensions, v.shape)
        if d != 1
    )
    return Quantity(data, dimensions=dimensions, unit=v.unit)


@Quantity.implements(numpy.mean)
def _mean(v: Quantity, **kwargs):
    """Compute the mean of the underlying array."""
    data = v._array.mean(**kwargs)
    axis = kwargs.get('axis')
    if axis is None:
        return data
    dimensions = tuple(
        a for a in v.dimensions if v.dimensions.index(a) != axis
    )
    return Quantity(data, dimensions=dimensions, unit=v.unit)


S = typing.TypeVar('S', bound='Interface')


class Interface(aliased.Mapping):
    """Base class for interfaces to dataset variables."""

    def __init__(
        self,
        __data: datafile.Interface,
        system: typing.Union[str, metric.System]=None,
    ) -> None:
        super().__init__(__data.variables)
        self._system = metric.System(system) if system else None
        self._cache = {}

    def __getitem__(self, __k: str) -> Quantity:
        """Retrieve or create the named quantity, if possible."""
        if __k in self._cache:
            return self._cache[__k]
        with contextlib.suppress(KeyError):
            variable = super().__getitem__(__k)
            if built := self.build(variable):
                self._cache[__k] = built
                return built
        raise KeyError(
            f"No variable quantity corresponding to {__k!r}"
        ) from None

    def build(self, __v: datafile.Variable):
        """Convert a raw variable into a variable quantity.

        Parameters
        ----------
        __v : `~datafile.Variable`
            A variable-like object from the dataset.
        
        Notes
        -----
        * The base implementation initializes and immediately returns an
          instance of `~variable.Quantity`, which it directly initializes from
          attributes of the argument. Observer-specific subclasses may wish to
          overload this method in order to provide additional functionality
          (e.g., unit standardization).
        """
        unit = __v.unit
        return Quantity(
            __v.data,
            dimensions=__v.axes,
            unit=(self.system.get_unit(unit=unit) if self.system else unit),
        )

    @property
    def system(self):
        """The associated metric system, if any."""
        return self._system

