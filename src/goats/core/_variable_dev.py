import numbers
import typing

import numpy
import numpy.typing

from goats.core import quantities
from goats.core import iterables


IndexLike = typing.TypeVar(
    'IndexLike',
    typing.Iterable[int],
    slice,
    type(Ellipsis),
)
IndexLike = typing.Union[typing.Iterable[int], slice, type(Ellipsis)]


class Variable(numpy.lib.mixins.NDArrayOperatorsMixin):
    """A class representing a dataset variable."""

    _data: numpy.typing.ArrayLike
    unit: str=None
    """The unit of this variable's array values."""
    axes: typing.Tuple[str]=None
    """The names of indexable axes in this variable's array."""
    naxes: int=None
    """The number of indexable axes in this variable's array."""
    name: str=None
    """The name of this variable, if available."""
    _scale: float=None
    _array: numpy.ndarray=None

    def __new__(cls, *args, **kwargs):
        """Create a new variable."""
        if not kwargs and len(args) == 1 and isinstance(args[0], cls):
            instance = args[0]
            data = instance._data
            unit = instance.unit
            axes = instance.axes
            name = instance.name
            scale = instance._scale
        else:
            attrs = list(args)
            attr_dict = {
                k: attrs.pop(0) if attrs
                else kwargs.pop(k, None)
                for k in ('data', 'unit', 'axes', 'name')
            }
            data = attr_dict['data']
            unit = attr_dict['unit'] or '1'
            axes = attr_dict['axes'] or ()
            name = attr_dict['name'] or '<anonymous>'
            scale = kwargs.get('scale') or 1.0
        self = super().__new__(cls)
        self._data = data
        self.unit = quantities.Unit(unit)
        self.axes = tuple(axes)
        self.naxes = len(axes)
        self.name = name
        self._scale = scale
        self._array = None
        return self

    def __eq__(self, other: typing.Any):
        """True if two instances have the same data and attributes."""
        if not isinstance(other, Variable):
            return NotImplemented
        if not self.axes == other.axes:
            return False
        if not self.unit == other.unit:
            return False
        return numpy.array_equal(other, self)

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
        print(f"You called __array_ufunc__ for {ufunc=}")
        out = kwargs.get('out', ())
        for x in inputs + out:
            if not isinstance(x, self._HANDLED_TYPES + (type(self),)):
                return NotImplemented
        updater = self._dispatch_updater(ufunc)
        updates = updater(*inputs, **kwargs) if updater else {}
        if 'data' in updates:
            return self._new_from_func(updates.pop('data'), **updates)
        args = tuple(
            x._get_data() if isinstance(x, type(self))
            else x for x in inputs
        )
        if out:
            kwargs['out'] = tuple(
                x._get_data() if isinstance(x, type(self))
                else x for x in out
            )
        result = getattr(ufunc, method)(*args, **kwargs)
        if type(result) is tuple:
            return tuple(self._new_from_func(x, **updates) for x in result)
        if method == 'at':
            return None
        return self._new_from_func(result, **updates)

    def _dispatch_updater(self, ufunc: numpy.ufunc):
        """Get the method that will update variable attributes for `ufunc`."""
        name = ufunc.__name__
        if name == 'add':
            return _add
        if name == 'subtract':
            return _subtract
        if name == 'multiply':
            return _multiply
        if name == 'true_divide':
            return _true_divide

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
        print(f"You called __array_function__ for {func=}")
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

    @property
    def shape_dict(self) -> typing.Dict[str, int]:
        """Label and size for each axis."""
        return dict(zip(self.axes, self._get_data('shape')))

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
        if attr := getattr(self._data, arg, None):
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
                return numpy.asarray(self._data[idx])
            except (TypeError, IndexError):
                return numpy.asarray(self._data)[idx]
        return numpy.asarray(self._data)

    def _new_from_func(self, result, **updates):
        """Create a new instance from the result of a `numpy` function."""
        if not isinstance(result, numpy.ndarray):
            return result
        return self.copy_with(data=result, **updates)

    def convert_to(self, unit: str):
        """Change this variable's unit and update the numerical scale factor."""
        scale = (quantities.Unit(unit) // self.unit) * self._scale
        return self.copy_with(unit=unit, scale=scale)

    def copy_with(self, **updates):
        """Create a new instance with optional parameter updates."""
        if 'data' in updates:
            return type(self)(
                data=updates['data'],
                unit=updates.get('unit', self.unit),
                axes=updates.get('axes', self.axes),
                name=updates.get('name', self.name),
            )
        return type(self)(
            data=self._data,
            unit=updates.get('unit', self.unit),
            axes=updates.get('axes', self.axes),
            name=updates.get('name', self.name),
            scale=updates.get('scale', self._scale),
        )

    def __str__(self) -> str:
        """A simplified representation of this object."""
        attrs = [
            f"{self.name!r}",
            f"unit='{self.unit}'",
            f"axes={self.axes}",
        ]
        return ', '.join(attrs)

    def __repr__(self) -> str:
        """An unambiguous representation of this object."""
        return f"{self.__class__.__qualname__}({self})"


# TODO: Refactor functions to reduce overlap.

def _add(a: Variable, b):
    """Called for a + b."""
    if isinstance(b, quantities.RealValued):
        return {'unit': a.unit, 'axes': a.axes, 'name': a.name}
    if isinstance(b, Variable) and a.axes == b.axes and a.unit == b.unit:
        return {'unit': a.unit, 'axes': a.axes, 'name': f"{a.name} + {b.name}"}

def _subtract(a: Variable, b):
    """Called for a - b."""
    if isinstance(b, quantities.RealValued):
        return {'unit': a.unit, 'axes': a.axes, 'name': a.name}
    if isinstance(b, Variable) and a.axes == b.axes and a.unit == b.unit:
        return {'unit': a.unit, 'axes': a.axes, 'name': f"{a.name} - {b.name}"}

def _multiply(a: Variable, b):
    """Called for a * b."""
    if isinstance(b, quantities.RealValued):
        return {'unit': a.unit, 'axes': a.axes, 'name': a.name}
    if isinstance(b, Variable):
        axes = _get_unique_axes(a.axes + b.axes)
        a_arr, b_arr = _extend_arrays(a, b, axes)
        return {
            'data': a_arr * b_arr,
            'unit': a.unit * b.unit,
            'axes': axes,
            'name': f"{a.name} * {b.name}",
        }

def _true_divide(a: Variable, b):
    """Called for a / b."""
    if isinstance(b, quantities.RealValued):
        return {'unit': a.unit, 'axes': a.axes, 'name': a.name}
    if isinstance(b, Variable):
        axes = _get_unique_axes(a.axes + b.axes)
        a_arr, b_arr = _extend_arrays(a, b, axes)
        return {
            'data': a_arr / b_arr,
            'unit': a.unit / b.unit,
            'axes': axes,
            'name': f"{a.name} / {b.name}",
        }

def _get_unique_axes(axes: typing.Iterable[str]):
    """Extract unique axes while preserving input order."""
    a = []
    for axis in axes:
        if axis not in a:
            a.append(axis)
    return a

def _extend_arrays(
    a: Variable,
    b: Variable,
    axes: typing.Tuple[str],
) -> typing.Tuple[numpy.ndarray]:
    """Extract arrays with extended axes.

    This method determines the set of unique axes shared by variables `a` and
    `b`, then extracts arrays suitable for computing a product or ratio that has
    the full set of axes.
    """
    tmp = {**b.shape_dict, **a.shape_dict}
    full_shape = tuple(tmp[d] for d in axes)
    idx = numpy.ix_(*[range(i) for i in full_shape])
    a_idx = tuple(idx[axes.index(d)] for d in a.shape_dict)
    a_arr = a._get_data(a_idx)
    b_idx = tuple(idx[axes.index(d)] for d in b.shape_dict)
    b_arr = b._get_data(b_idx)
    return a_arr, b_arr

