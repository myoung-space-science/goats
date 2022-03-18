import numbers
import typing

import numpy
import numpy.typing

from goats.core import iterables
from goats.core import quantities


Instance = typing.TypeVar('Instance', bound='Variable')


IndexLike = typing.TypeVar(
    'IndexLike',
    typing.Iterable[int],
    slice,
    type(Ellipsis),
)
IndexLike = typing.Union[typing.Iterable[int], slice, type(Ellipsis)]

UnitLike = typing.TypeVar('UnitLike', str, quantities.Unit)
UnitLike = typing.Union[str, quantities.Unit]


Parent = quantities.Measured
Mixin = numpy.lib.mixins.NDArrayOperatorsMixin
allowed = {'__add__': float, '__sub__': float}
class Variable(Parent, Mixin, allowed=allowed):
    """A measured object with data stored in a numerical array.

    The result of binary arithmetic operations on instances of this class are
    similar to those of `Vector`, but differ in the following ways:
    1. Multiplication (`*`) and division (`/`) accept operands with different
       axes, as long as any repeated axes have the same length in both operands.
       The result will contain all unique axes from its operands.
    2. Addition (`+`) and subtraction (`-`) accept real numbers as right-sided
       operands. The result is a new instance with the operation applied to the
       underlying array.
    """

    @typing.overload
    def __new__(
        cls: typing.Type[Instance],
        data: typing.Iterable[numbers.Number],
        unit: typing.Union[str, quantities.Unit],
        axes: typing.Iterable[str],
    ) -> Instance:
        """Create a new variable object.
        
        Parameters
        ----------
        data : array-like
            The numerical data of this variable.

        unit : string or `~quantities.Unit`
            The metric unit of `data`.

        axes : iterable of strings
            The names of this variable's indexable axes.
        """

    @typing.overload
    def __new__(
        cls: typing.Type[Instance],
        data: typing.Iterable[numbers.Number],
        unit: typing.Union[str, quantities.Unit],
        axes: typing.Iterable[str],
        name: str='<anonymous>',
    ) -> Instance:
        """Create a new variable object.
        
        Parameters
        ----------
        data : array-like
            The numerical data of this variable.

        unit : string or `~quantities.Unit`
            The metric unit of `data`.

        axes : iterable of strings
            The names of this variable's indexable axes.

        name : string, default='<anonymous>'
            The optional name of this variable.
        """

    @typing.overload
    def __new__(
        cls: typing.Type[Instance],
        instance: Instance,
    ) -> Instance:
        """Create a new variable object.
        
        Parameters
        ----------
        instance : `~quantities.Variable`
            An existing instance of this class.
        """

    _amount: numpy.typing.ArrayLike
    axes: typing.Tuple[str]=None
    naxes: int=None
    name: str=None
    _scale: float=None
    _array: numpy.ndarray=None

    def __new__(cls, *args, **kwargs):
        """The concrete implementation of `~quantities.Variable.__new__`."""
        if not kwargs and len(args) == 1 and isinstance(args[0], cls):
            instance = args[0]
            data = instance._amount
            unit = instance.unit()
            axes = instance.axes
            name = instance.name
            scale = instance._scale
        else:
            if 'amount' in kwargs:
                kwargs['data'] = kwargs.pop('amount')
            attrs = list(args)
            attr_dict = {
                k: attrs.pop(0) if attrs
                else kwargs.pop(k, None)
                for k in ('data', 'unit', 'axes', 'name')
            }
            data = attr_dict['data']
            unit = attr_dict['unit']
            axes = attr_dict['axes'] or ()
            name = attr_dict['name'] or '<anonymous>'
            scale = kwargs.get('scale') or 1.0
        self = super().__new__(cls, data, unit=unit)
        self.axes = tuple(axes)
        """The names of indexable axes in this variable's array."""
        self.naxes = len(axes)
        """The number of indexable axes in this variable's array."""
        self.name = name
        """The name of this variable, if available."""
        self._scale = scale
        self._array = None
        return self

    @typing.overload
    def unit(self: Instance) -> quantities.Unit:
        """Get this object's unit of measurement.
        
        Parameters
        ----------
        None

        Returns
        -------
        `~quantities.Unit`
            The current unit of `amount`.
        """

    @typing.overload
    def unit(
        self: Instance,
        new: typing.Union[str, quantities.Unit],
    ) -> Instance:
        """Update this object's unit of measurement.

        Parameters
        ----------
        new : string or `~quantities.Unit`
            The new unit in which to measure `amount`.

        Returns
        -------
        Subclass of `~quantities.Measured`
            A new instance of this class.
        """

    def unit(self, new=None):
        """Concrete implementation."""
        if not new:
            return self._unit
        scale = (quantities.Unit(new) // self._unit) * self._scale
        return self._new(
            data=self._amount,
            unit=new,
            axes=self.axes,
            name=self.name,
            scale=scale,
        )

    def __measure__(self):
        """Called for `~quantities.measure(self)`."""
        return quantities.Measurement(self._get_data(), self.unit())

    @property
    def ndim(self) -> int:
        """The number of dimensions in this variable's array."""
        return self.naxes

    def __len__(self):
        """Called for len(self)."""
        return self._get_data('size')

    def __iter__(self):
        """Called for iter(self)."""
        if method := self._get_data('__iter__'):
            return method()
        return iter(self._get_data())

    def __contains__(self, item):
        """Called for `item` in self."""
        return item in self._amount or item in self._get_data()

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
        result = self._get_data(indices)
        if isinstance(result, numbers.Number):
            return quantities.Scalar(result, unit=self.unit())
        return self._new(
            data=result,
            unit=self.unit(),
            axes=self.axes,
            name=self.name,
        )

    def _subscript_custom(self, args):
        """Perform array subscription specific to this object.

        This method handles all cases that don't meet the criteria for
        `_subscript_standard`.
        """
        if not isinstance(args, (tuple, list)):
            args = [args]
        expanded = self._expand_ellipsis(args)
        shape = self._get_data('shape')
        idx = [
            range(shape[i])
            if isinstance(arg, slice) else arg
            for i, arg in enumerate(expanded)
        ]
        indices = numpy.ix_(*list(idx))
        return self._new(
            data=self._get_data(indices),
            unit=self.unit(),
            axes=self.axes,
            name=self.name,
        )

    def _expand_ellipsis(
        self,
        user: typing.Sequence,
    ) -> typing.Tuple[slice, ...]:
        """Expand an ``Ellipsis`` into one or more ``slice`` objects."""
        if Ellipsis not in user:
            return user
        length = self.naxes - len(user) + 1
        start = user.index(Ellipsis)
        return tuple([
            *user[slice(start)],
            *([slice(None)] * length),
            *user[slice(start+length, self.naxes)],
        ])

    _HANDLED_TYPES = (numpy.ndarray, numbers.Number, list)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Provide support for `numpy` universal functions.

        See https://numpy.org/doc/stable/reference/arrays.classes.html for more
        information on use of this special method.

        See
        https://numpy.org/doc/stable/reference/generated/numpy.lib.mixins.NDArrayOperatorsMixin.html#numpy.lib.mixins.NDArrayOperatorsMixin
        for the specific implementation example that this class follows.

        This method first ensures that the input types (as well as the type of
        `out`, if given) are supported types.
        """
        out = kwargs.get('out', ())
        for x in inputs + out:
            if not isinstance(x, self._HANDLED_TYPES + (type(self),)):
                return NotImplemented
        inputs = tuple(
            x._get_data() if isinstance(x, type(self))
            else x for x in inputs
        )
        if out:
            kwargs['out'] = tuple(
                x._get_data() if isinstance(x, type(self))
                else x for x in out
            )
        result = getattr(ufunc, method)(*inputs, **kwargs)
        if type(result) is tuple:
            return tuple(self._new_from_func(x) for x in result)
        if method == 'at':
            return None
        return self._new_from_func(result)

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
        accepted = (type(self), numpy.ndarray, numpy.ScalarType)
        if not all(issubclass(ti, accepted) for ti in types):
            return NotImplemented
        if func in self._HANDLED_FUNCTIONS:
            arr = self._HANDLED_FUNCTIONS[func](*args, **kwargs)
            return self._new_from_func(arr)
        args = tuple(
            arg._get_data() if isinstance(arg, type(self))
            else arg for arg in args
        )
        types = tuple(
            ti for ti in types
            if not issubclass(ti, type(self))
        )
        data = self._get_data()
        arr = data.__array_function__(func, types, args, kwargs)
        return self._new_from_func(arr)

    def _new_from_func(self, result):
        """Create a new instance from the result of a `numpy` function."""
        if not isinstance(result, numpy.ndarray):
            return result
        return self._new(
            data=result,
            unit=self.unit(),
            axes=self.axes,
            name=self.name,
        )

    def __eq__(self, other: typing.Any):
        """True if two instances have the same data and attributes."""
        if not isinstance(other, Variable):
            return NotImplemented
        if not self._equal_attrs(other):
            return False
        return numpy.array_equal(other, self)

    def _equal_attrs(self, other: 'Variable'):
        """True if two instances have the same attributes."""
        return all(
            getattr(other, attr) == getattr(self, attr)
            for attr in {'unit', 'axes'}
        )

    def __add__(self, other: typing.Any):
        if self._add_sub_okay(other):
            data = self._get_data().__add__(other)
            return self._new(
                data=data,
                unit=self.unit(),
                axes=self.axes,
                name=self.name,
            )
        return NotImplemented

    def __sub__(self, other: typing.Any):
        if self._add_sub_okay(other):
            data = self._get_data().__sub__(other)
            return self._new(
                data=data,
                unit=self.unit(),
                axes=self.axes,
                name=self.name,
            )
        return NotImplemented

    def _add_sub_okay(self, other):
        if isinstance(other, numbers.Real):
            return True
        if isinstance(other, Variable) and self.axes == other.axes:
            return True
        return False

    def __mul__(self, other: typing.Any):
        if isinstance(other, Variable):
            axes = sorted(tuple(set(self.axes + other.axes)))
            sarr, oarr = self._extend_arrays(other, axes)
            data = sarr * oarr
            unit = self.unit() * other.unit()
            name = f"{self.name} * {other.name}"
            return self._new(
                data=data,
                unit=unit,
                axes=axes,
                name=name,
            )
        data = self._get_data().__mul__(other)
        return self._new(
            data=data,
            unit=self.unit(),
            axes=self.axes,
            name=self.name,
        )

    def __rmul__(self, other: typing.Any):
        data = self._get_data().__rmul__(other)
        return self._new(
            data=data,
            unit=self.unit(),
            axes=self.axes,
            name=self.name,
        )

    def __truediv__(self, other: typing.Any):
        if isinstance(other, Variable):
            axes = sorted(tuple(set(self.axes + other.axes)))
            sarr, oarr = self._extend_arrays(other, axes)
            data = sarr / oarr
            unit = self.unit() / other.unit()
            name = f"{self.name} / {other.name}"
            return self._new(
                data=data,
                unit=unit,
                axes=axes,
                name=name,
            )
        data = self._get_data().__truediv__(other)
        return self._new(
            data=data,
            unit=self.unit(),
            axes=self.axes,
            name=self.name,
        )

    def __pow__(self, other: typing.Any):
        if isinstance(other, numbers.Real):
            data = self._get_data().__pow__(other)
            unit = self.unit().__pow__(other)
            return self._new(
                data=data,
                unit=unit,
                axes=self.axes,
                name=self.name,
            )
        return NotImplemented

    @property
    def shape_dict(self) -> typing.Dict[str, int]:
        """Label and size for each axis."""
        return dict(zip(self.axes, self._get_data('shape')))

    def _extend_arrays(
        self,
        other: 'Variable',
        axes: typing.Tuple[str],
    ) -> typing.Tuple[numpy.ndarray]:
        """Extract arrays with extended axes.

        This method determines the set of unique axes shared by this
        instance and `other`, then extracts arrays suitable for computing a
        product or ratio that has the full set of axes.
        """
        tmp = {**other.shape_dict, **self.shape_dict}
        full_shape = tuple(tmp[d] for d in axes)
        idx = numpy.ix_(*[range(i) for i in full_shape])
        self_idx = tuple(idx[axes.index(d)] for d in self.shape_dict)
        self_arr = self._get_data(self_idx)
        other_idx = tuple(idx[axes.index(d)] for d in other.shape_dict)
        other_arr = other._get_data(other_idx)
        return self_arr, other_arr

    def __array__(self, *args, **kwargs) -> numpy.ndarray:
        """Support casting to `numpy` array types.
        
        Notes
        -----
        This will first cast `self._amount` (inherited from
        `~quantities.Measured`) on its own to a `numpy.ndarray`, before applying
        `*args` and `**kwargs`, in order to avoid a `TypeError` when using
        `netCDF4.Dataset`. See
        https://github.com/mcgibbon/python-examples/blob/master/scripts/file-io/load_netCDF4_full.py
        """
        data = self._get_array()
        return numpy.asanyarray(data, *args, **kwargs)

    def _get_data(self, arg: typing.Union[str, IndexLike]=None):
        """Access the data array or a dataset attribute.
        
        If `arg` is not a string, this method will assume it is an index and
        will attempt to return the relevant portion of the dataset array (after
        loading from disk, if necessary). If `arg` is a string, this method will
        first search `_amount` for the named attribute, to take advantage of
        viewers that provide metadata without loading the full dataset. If that
        search fails, this method will attempt to retrieve the named attribute
        from the full array.
        """
        if not isinstance(arg, str):
            return self._get_array(index=arg)
        if attr := getattr(self._amount, arg, None):
            return attr
        return getattr(self._get_array(), arg)

    def _get_array(self, index: IndexLike=None):
        """Access array data via index or slice notation.
        
        Notes
        -----
        If `index` is not `None`, this method will create the requested subarray
        from `self._amount` and directly return it. If `index` is `None`, this
        method will load the entire array and let execution proceed to the
        following block, which will immediately return the array. It will then
        subscript the pre-loaded array on subsequent calls. The reasoning behind
        this algorithm is as follows: If we need to load the full array at any
        point, we may as well save it because subscripting an in-memory
        `numpy.ndarray` is much faster than re-reading from disk for large
        arrays. However, we should avoid reading in the full array if the caller
        only wants a small portion of it, and in those cases, keeping the loaded
        data in memory will lead to incorrect results when attempting to access
        a different portion of the full array because the indices will be
        different. The worst-case scenario will occur when the caller repeatedly
        tries to access a large portion of the full array; this is a possible
        area for optimization.
        """
        if self._array is None:
            array = self._load_array(index) * self._scale
            if index is not None:
                return array
            self._array = array
        if iterables.missing(index):
            return self._array
        idx = numpy.index_exp[index]
        return self._array[idx]

    def _load_array(self, index: IndexLike=None):
        """Read the array data from disk.
        
        If `index` is "missing" in the sense defined by `~iterables.missing`
        this method will load and return the full array. If `index` is not
        missing, this method will first attempt to subscript `self._amount`
        before converting it to an array and returning it. If it catches either
        a `TypeError` or an `IndexError`, it will create the full array before
        subscripting and returning it. The former may occur if `self._amount` is
        a sequence type like `list`, `tuple`, or `range`; the latter may occur
        when attempting to subscript certain array-like objects (e.g.,
        `netCDF4._netCDF4.Variable`) with valid `numpy` index expressions.
        """
        if not iterables.missing(index):
            idx = numpy.index_exp[index]
            try:
                return numpy.asarray(self._amount[idx])
            except (TypeError, IndexError):
                return numpy.asarray(self._amount)[idx]
        return numpy.asarray(self._amount)

    def __str__(self) -> str:
        """A simplified representation of this object."""
        attrs = [
            f"shape={self.shape_dict}",
            f"unit='{self.unit()}'",
            f"name='{self.name}'",
        ]
        return ', '.join(attrs)

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


@Variable.implements(numpy.mean)
def _array_mean(a: Variable, **kwargs):
    """Compute the mean and update array dimensions, if necessary."""
    data = a._get_data().mean(**kwargs)
    if (axis := kwargs.get('axis')) is not None:
        a.axes = tuple(
            d for d in a.axes
            if a.axes.index(d) != axis
        )
    return data


