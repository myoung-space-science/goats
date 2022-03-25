import abc
import collections.abc
import numbers
import typing

import numpy
import numpy.typing

from goats.core import quantities
from goats.core import iterables


class Constraint(abc.ABC):
    """Base class for enforceable operator constraints."""

    def __call__(self, *args, **kwargs):
        """Enforce this constraint."""
        if not self.consistent(*args, **kwargs):
            raise ValueError

    @abc.abstractmethod
    def consistent(self, *args, **kwargs) -> bool:
        """Apply this constraint."""
        return False


class SameAttrs(iterables.ReprStrMixin, Constraint):
    """A class that checks attribute consistency."""

    def __init__(self, *names: str) -> None:
        self.names = names

    def consistent(self, a, b) -> bool:
        """True if the named attributes of `a` and `b` are equal."""
        return all(getattr(a, name) == getattr(b, name) for name in self.names)

    def __str__(self) -> str:
        return ', '.join(self.names)


XT = typing.TypeVar('XT', typing.Callable, str)
def attr_updater(x: XT):
    """Update a `Variable` attribute via `x`."""
    def inner(*v: Variable):
        if callable(x):
            return x(*v)
        if isinstance(x, str):
            return x.format(*v)
    return inner


@typing.runtime_checkable
class HasAxes(typing.Protocol):
    """Abstract protocol for objects with `axes`."""

    @property
    @abc.abstractmethod
    def axes(self) -> typing.Iterable[str]:
        """The names of indexable axes in this object's array."""
        pass


def unique_axes(*args: HasAxes):
    """Compute unique axes while preserving order."""
    axes = (axis for arg in args for axis in arg.axes)
    unique = []
    for axis in axes:
        if axis not in unique:
            unique.append(axis)
    return unique


class Ufunc(iterables.ReprStrMixin):
    """An object that manages use of a `numpy` universal function."""

    def __init__(self, ufunc: numpy.ufunc) -> None:
        self.ufunc = ufunc
        self.name = ufunc.__name__
        self._recipes = None

    @property
    def recipes(self) -> typing.Optional[dict]:
        """The updaters for supported argument types."""
        if self._recipes is None:
            self._recipes = _opr_rules.get(self.name)
        return self._recipes

    def attr_updates(self, *args):
        """Updated attributes for `args`, based on this function."""
        if self.recipes:
            # If we get here, there is an entry for `name`.
            recipe = self._get_recipe(*args)
            if recipe is None:
                return NotImplemented
            # If we get here, there is a recipe for the argument type(s).
            for constraint in recipe.get('constraints', ()):
                constraint(*args)
            updaters = recipe.get('updaters', {})
            return {k: updater(*args) for k, updater in updaters.items()}

    def _get_recipe(self, *args) -> dict:
        """Attempt to find an appropriate recipe for the given args."""
        types = tuple(type(arg) for arg in args)
        if types in self.recipes:
            return self.recipes[types]
        for key, recipe in self.recipes.items():
            if all(issubclass(t, k) for t, k in zip(types, key)):
                return recipe

    def __str__(self) -> str:
        return self.name


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
    unit: quantities.Unit=None
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
            name = attr_dict['name'] or ''
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

    @property
    def ndim(self):
        """Alias for `naxes`."""
        return self.naxes

    @property
    def shape(self):
        """The shape of this instance's array."""
        return self._get_data('shape')

    def __eq__(self, other: typing.Any):
        """True if two instances have the same data and attributes."""
        if not isinstance(other, Variable):
            return NotImplemented
        if not self.axes == other.axes:
            return False
        if not self.unit == other.unit:
            return False
        return numpy.array_equal(other, self)

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
        return item in self._data or item in self._get_data()

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
            return quantities.Scalar(result, unit=self.unit)
        return self.copy_with(data=result)

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
        return self.copy_with(data=self._get_data(indices))

    def _expand_ellipsis(
        self,
        user: typing.Sequence,
    ) -> typing.Tuple[slice, ...]:
        """Expand an ``Ellipsis`` into one or more ``slice`` objects."""
        if Ellipsis not in user:
            return user
        length = self.naxes - len(user) + 1
        start = user.index(Ellipsis)
        return (
            *user[slice(start)],
            *([slice(None)] * length),
            *user[slice(start+length, self.naxes)],
        )

    def __measure__(self):
        """Called for `~quantities.measure(self)`."""
        return quantities.Measurement(self._get_data(), self.unit)

    _HANDLED_TYPES = (numpy.ndarray, numbers.Number, list)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
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
        arrays from any inputs that are an instance of this class or a subclass
        and applies `ufunc`. If `ufunc` is one of a pre-defined set of methods
        that would cause an attribute of this class to become ambiguous or
        undefined, this method will immediately return the result of applying
        `ufunc`. Otherwise, this method will return a new instance with zero or
        more attributes that have been modified in a `ufunc`-specific way (e.g.,
        multiplying two instances' units when `ufunc` is 'multiply'). The
        default behavior is therefore to return a new instance with unmodified
        attributes.
        """
        out = kwargs.get('out', ())
        for x in inputs + out:
            if not isinstance(x, self._HANDLED_TYPES + (type(self),)):
                return NotImplemented
        if out:
            kwargs['out'] = tuple(
                x._get_data() if isinstance(x, type(self))
                else x for x in out
            )
        handler = Ufunc(ufunc)
        args = self._convert_inputs(ufunc, *inputs)
        result = getattr(ufunc, method)(*args, **kwargs)
        if ufunc.__name__ in _native_rtype:
            return result
        updates = handler.attr_updates(*inputs)
        if type(result) is tuple:
            return tuple(
                self._new_from_func(x, updates=updates)
                for x in result
            )
        if method == 'at':
            return None
        return self._new_from_func(result, updates=updates)

    def _convert_inputs(self, ufunc, *inputs):
        """Convert input arrays into arrays appropriate to `ufunc`."""
        multiplicative = ufunc.__name__ in {'multiply', 'divide', 'true_divide'}
        correct_type = all(isinstance(v, type(self)) for v in inputs)
        if multiplicative and correct_type:
            return _extend_arrays(*inputs)
        return tuple(
            x._get_data() if isinstance(x, type(self))
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
        accepted = (type(self), numpy.ndarray, numpy.ScalarType)
        if not all(issubclass(ti, accepted) for ti in types):
            return NotImplemented
        if func in self._HANDLED_FUNCTIONS:
            result = self._HANDLED_FUNCTIONS[func](*args, **kwargs)
            return self._new_from_func(result)
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
        This will retrieve the underlying array before applying `*args` and
        `**kwargs`, in order to avoid a `TypeError` when using
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
        first search `_data` for the named attribute, to take advantage of
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
        from `self._data` and directly return it. If `index` is `None`, this
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
        missing, this method will first attempt to subscript `self._data` before
        converting it to an array and returning it. If it catches either a
        `TypeError` or an `IndexError`, it will create the full array before
        subscripting and returning it. The former may occur if `self._data` is a
        sequence type like `list`, `tuple`, or `range`; the latter may occur
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

    def _new_from_func(self, result, updates: dict=None):
        """Create a new instance from the result of a `numpy` function.
        
        If `result` is a `numpy.ndarray` and `updates` is a (possibly empty)
        `dict`, this method will create a new instance. Otherwise, it will
        return `result` as-is.
        """
        if isinstance(result, numpy.ndarray) and isinstance(updates, dict):
            return self.copy_with(data=result, **updates)
        return result

    def convert_to(self, unit: str):
        """Change this variable's unit and update the numerical scale factor."""
        scale = (quantities.Unit(unit) // self.unit) * self._scale
        return self.copy_with(unit=unit, scale=scale)

    def copy_with(self, **updates):
        """Create a new instance with optional parameter updates."""
        if 'data' in updates: # User shouldn't be able to do this.
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


@Variable.implements(numpy.squeeze)
def _squeeze(v: Variable, **kwargs):
    """Remove singular axes."""
    data = v._get_data().squeeze(**kwargs)
    axes = tuple(
        a for a, d in zip(v.axes, v._get_data('shape'))
        if d != 1
    )
    return Variable(data, unit=v.unit, axes=axes, name=v.name)


@Variable.implements(numpy.mean)
def _mean(v: Variable, **kwargs):
    """Compute the mean of the underlying array."""
    data = v._get_data().mean(**kwargs)
    axis = kwargs.get('axis')
    if axis is None:
        return data
    axes = tuple(a for a in v.axes if v.axes.index(a) != axis)
    name = f"mean({v.name})"
    return Variable(data, unit=v.unit, axes=axes, name=name)


def _extend_arrays(
    a: Variable,
    b: Variable,
) -> typing.Tuple[numpy.ndarray]:
    """Extract arrays with extended axes.

    This method determines the set of unique axes shared by variables `a` and
    `b`, then extracts arrays suitable for computing a product or ratio that has
    the full set of axes.
    """
    axes = unique_axes(a, b)
    tmp = {**b.shape_dict, **a.shape_dict}
    full_shape = tuple(tmp[d] for d in axes)
    idx = numpy.ix_(*[range(i) for i in full_shape])
    a_idx = tuple(idx[axes.index(d)] for d in a.shape_dict)
    a_arr = a._get_data(a_idx)
    b_idx = tuple(idx[axes.index(d)] for d in b.shape_dict)
    b_arr = b._get_data(b_idx)
    return a_arr, b_arr


_opr_rules = {
    'add': {
        (Variable, quantities.RealValued): {},
        (Variable, Variable): {
            'constraints': [SameAttrs('axes', 'unit')],
            'updaters': {
                'name': attr_updater('{0.name} + {1.name}'),
            }
        },
        (quantities.RealValued, Variable): {},
    },
    'subtract': {
        (Variable, quantities.RealValued): {},
        (Variable, Variable): {
            'constraints': [SameAttrs('axes', 'unit')],
            'updaters': {
                'name': attr_updater('{0.name} - {1.name}'),
            }
        },
        (quantities.RealValued, Variable): {},
    },
    'multiply': {
        (Variable, quantities.RealValued): {},
        (Variable, Variable): {
            'updaters': {
                'unit': attr_updater('{0.unit} * {1.unit}'),
                'axes': unique_axes,
                'name': attr_updater('{0.name} * {1.name}'),
            }
        },
        (quantities.RealValued, Variable): {},
    },
    'true_divide': {
        (Variable, quantities.RealValued): {},
        (Variable, Variable): {
            'updaters': {
                'unit': attr_updater('{0.unit} / ({1.unit})'),
                'axes': unique_axes,
                'name': attr_updater('{0.name} / {1.name}'),
            }
        },
    },
    'power': {
        (Variable, numbers.Real): {
            'updaters': {
                'unit': attr_updater('({0.unit})^{1}'),
                'name': attr_updater('{0.name}^{1}'),
            },
        },
    },
    'sqrt': {
        (Variable,): {
            'updaters': {
                'unit': attr_updater('sqrt({0.unit})'),
                'name': attr_updater('sqrt({0.name})'),
            },
        },
    },
}


_native_rtype = {
    'arccos',
    'arccosh',
    'arcsin',
    'arcsinh',
    'arctan',
    'arctan2',
    'arctanh',
    'cos',
    'cosh',
    'sin',
    'sinh',
    'tan',
    'tanh',
    'log',
    'log10',
    'log1p',
    'log2',
    'exp',
    'exp2',
    'expm1',
    'floor_divide',
}
"""Universal functions that result in a `numpy` or built-in return type.

When any of the universal functions in this set act on an instance of
`Variable`, the result will be the same as if it had acted on the underlying
data array. The reason for this behavior may vary from function to function, but
will typically be related to causing an attribute to become ambiguous or
undefined.
"""


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

    __slots__ = ('values',)

    def __init__(
        self,
        indices: typing.Iterable[int],
        values: typing.Iterable[typing.Any],
    ) -> None:
        super().__init__(indices)
        self.values = tuple(values)
        nv = len(self.values)
        ni = len(self.indices)
        if nv != ni:
            errmsg = f"number of values ({nv}) != number of indices ({ni})"
            raise TypeError(errmsg)

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

    __slots__ = ('unit',)

    def __init__(
        self,
        indices: typing.Iterable[int],
        values: typing.Iterable[typing.Any],
        unit: typing.Union[str, quantities.Unit],
    ) -> None:
        super().__init__(indices, values)
        self.unit = unit

    def with_unit(self, new: typing.Union[str, quantities.Unit]):
        """Convert this object to the new unit, if possible."""
        scale = quantities.Unit(new) // self.unit
        self.values = [value * scale for value in self.values]
        self.unit = new
        return self

    def __eq__(self, other):
        return (
            super().__eq__(other) and self.unit == other.unit
            if isinstance(other, Coordinates)
            else NotImplemented
        )

    def __str__(self) -> str:
        """A simplified representation of this object."""
        values = iterables.show_at_most(3, self.values, separator=', ')
        return f"{values} [{self.unit}]"


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


class Axis(iterables.ReprStrMixin):
    """A single dataset axis."""

    def __init__(
        self,
        size: int,
        indexer: Indexer,
        name: str='<anonymous>',
    ) -> None:
        self.size = size
        """The full length of this axis."""
        self.indexer = indexer
        """A callable object that creates indices from user input."""
        self.reference = indexer.reference
        """The index reference values."""
        self.name = name
        """The name of this axis."""

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
        string = f"{self.name}, size={self.size}"
        unit = (
            str(self.reference.unit())
            if isinstance(self.reference, quantities.Measured)
            else None
        )
        if unit:
            string += f", unit={unit!r}"
        return string


