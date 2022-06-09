import abc
import collections.abc
import contextlib
import numbers
import typing

import numpy
import numpy.typing

from goats.core import aliased
from goats.core import iterables
from goats.core import metric
from goats.core import measurable


def same_attrs(*names: str):
    """Enforce attribute equality."""
    def constrain(*v: Variable):
        v0 = v[0]
        if len(v) == 1:
            return True
        vi = v[1:]
        return all(
            getattr(i, name) == getattr(v0, name)
            for name in names for i in vi
        )
    return constrain


def attr_updater(x: typing.Union[typing.Callable, str]):
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


class Name(collections.abc.Collection, iterables.ReprStrMixin):
    """The name attribute of a data quantity."""

    def __init__(self, *aliases: str) -> None:
        self._aliases = aliased.MappingKey(*aliases)

    def __add__(self, other):
        """Called for self + other."""
        if other == self:
            return self
        return Name(*self._combine('+', other))

    def __sub__(self, other):
        """Called for self - other."""
        if other == self:
            return self
        return Name(*self._combine('-', other))

    def __mul__(self, other):
        """Called for self * other."""
        return Name(*self._combine('*', other))

    def __truediv__(self, other):
        """Called for self / other."""
        return Name(*self._combine('/', other))

    def __pow__(self, other):
        """Called for self ** other."""
        return Name(*self._combine('^', fractions.Fraction(other)))

    def _combine(self, symbol: str, other):
        """Symbolically combine `self` with `other`."""
        if not self or not other:
            return ['']
        if self == other:
            return [f'{i}{symbol}{i}' for i in self]
        if isinstance(other, typing.Iterable) and not isinstance(other, str):
            return [f'{i}{symbol}{j}' for i in self for j in other]
        return [f'{i}{symbol}{other}' for i in self._aliases]

    def __bool__(self) -> bool:
        return bool(self._aliases)

    def __contains__(self, __x) -> bool:
        return __x in self._aliases

    def __iter__(self) -> typing.Iterator:
        return iter(self._aliases)

    def __len__(self) -> int:
        return len(self._aliases)

    def __eq__(self, __o) -> bool:
        if isinstance(__o, Name):
            return __o._aliases == self._aliases
        try:
            return __o == self._aliases
        except TypeError:
            return False

    def __str__(self) -> str:
        return str(self._aliases)


Instance = typing.TypeVar('Instance', bound='Quantity')


class Quantity(measurable.OperatorMixin, measurable.Quantity):
    """A measurable quantity with a name.
    
    This class is a concrete implementation of `~measurable.Quantity` that uses
    the default operator implementations in `~measurable.OperatorMixin`.
    """

    @typing.overload
    def __init__(
        self: Instance,
        data: measurable.Real,
        unit: metric.UnitLike=None,
        name: typing.Union[str, typing.Iterable[str]]=None,
    ) -> None:
        """Initialize this instance from arguments."""

    @typing.overload
    def __init__(
        self: Instance,
        instance: Instance,
    ) -> None:
        """Initialize this instance from an existing one."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._name = self._parse_quantity(*args, **kwargs)
        if self._name:
            self.display['__str__']['strings'].insert(0, "'{name}': ")
            self.display['__repr__']['strings'].insert(1, "'{name}'")

    def _parse_quantity(self, *args, **kwargs):
        """Parse input arguments to initialize this instance."""
        if not kwargs and len(args) == 1 and isinstance(args[0], type(self)):
            return args[0].name()
        return aliased.MappingKey(
            args[2] if len(args) == 3
            else kwargs.get('name') or ''
        )

    def name(self, *new: str, reset: bool=False):
        """Get, set, or add to this object's name(s)."""
        if not new:
            return self._name
        name = new if reset else self._name | new
        self._name = aliased.MappingKey(name)
        return self


class Scalar(Quantity):
    """A single-valued data-type quantity.
    
    This class is a virtual concrete implementation of `~measurable.Scalar` that
    uses the results of `~measurable._cast` and `~measurable._unary` to provide
    scalar-specific methods. It defines the magic method `__measure__` in order
    to directly support `~measurables.measure`.
    """

    def __float__(self):
        """Called for float(self)."""
        return float(self.data)

    def __int__(self):
        """Called for int(self)."""
        return int(self.data)

    def __round__(self, ndigits):
        """Called for round(self)."""
        return NotImplemented

    def __ceil__(self):
        """Called for math.ceil(self)."""
        return NotImplemented

    def __floor__(self):
        """Called for math.floor(self)."""
        return NotImplemented

    def __trunk__(self):
        """Called for math.trunk(self)."""
        return NotImplemented

    def __init__(
        self,
        data: numbers.Real,
        unit: metric.UnitLike=None,
        name: typing.Union[str, typing.Iterable[str]]=None,
    ) -> None:
        super().__init__(float(data), unit=unit, name=name)

    def __measure__(self):
        """Create a measurement from this scalar's value and unit."""
        value = iterables.whole(self.data)
        return measurable.Measurement(value, self.unit())


measurable.Scalar.register(Scalar)


class Vector(Quantity):
    """A multi-valued measurable quantity.

    This class is a virtual concrete implementation of `~measurable.Vector` that
    uses local definitions of `__len__` and `__getitem__`. It defines the magic
    method `__measure__` in order to directly support `~measurables.measure`.
    """

    def __init__(
        self,
        data: typing.Union[measurable.Real, numpy.typing.ArrayLike],
        unit: metric.UnitLike=None,
        name: typing.Union[str, typing.Iterable[str]]=None,
    ) -> None:
        array = numpy.asfarray(list(iterables.whole(data)))
        super().__init__(array, unit=unit, name=name)

    def __len__(self) -> int:
        """Called for len(self)."""
        return len(self.data)

    def __getitem__(self, index):
        """Called for index-based value access."""
        if isinstance(index, typing.SupportsIndex) and index < 0:
            index += len(self)
        values = self.data[index]
        iter_values = isinstance(values, typing.Iterable)
        unit = self.unit()
        return (
            [Scalar(value, unit) for value in values] if iter_values
            else Scalar(values, unit)
        )

    def __measure__(self):
        """Create a measurement from this vector's values and unit."""
        return measurable.Measurement(self.data, self.unit())


measurable.Vector.register(Vector)


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
        self._old_scale = self._scale
        self._array = None

    def unit(self, unit: metric.UnitLike=None):
        """Get or set the unit of this object's values."""
        if not unit:
            return self._metric
        if unit == self._metric:
            return self
        new = metric.Unit(unit)
        self._scale *= new // self._metric
        self._metric = new
        return self

    def __eq__(self, other: typing.Any):
        """True if two instances have the same data and attributes."""
        if not isinstance(other, type(self)):
            return NotImplemented
        if not self.unit() == other.unit():
            return False
        return numpy.array_equal(other, self)

    def __len__(self):
        """Called for len(self)."""
        if method := self._get_base_attr('__len__'):
            return method()
        return len(self._amount)

    def __getattr__(self, name: str):
        """Access an attribute of the underlying data object or array."""
        if attr := self._get_base_attr(name):
            self.__dict__[name] = attr
            return attr
        raise AttributeError(name)

    _search_attrs = ('_amount', '_get_array')

    def _get_base_attr(self, name: str):
        """Helper method to efficiently access underlying attributes.

        This method will first search the underlying data object for the named
        attribute, to take advantage of viewers that provide metadata without
        loading the full dataset. If that search fails, this method will attempt
        to retrieve the named attribute from the actual dataset array.
        """
        targets = [getattr(self, name) for name in self._search_attrs]
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
        Type = Value if isinstance(result, numbers.Number) else Array
        return Type(result, unit=self.unit(), name=self.name())

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
                x._get_array() if isinstance(x, type(self))
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
            x._get_array() if isinstance(x, type(self))
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
            arg._get_array() if isinstance(arg, type(self))
            else arg for arg in args
        )
        types = tuple(
            ti for ti in types
            if not issubclass(ti, type(self))
        )
        data = self._get_array()
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
        if self._array is None or self._rescale:
            array = self._load_array(index) * self._scale
            self._rescale = False
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
            return self._copy_with(data=result, **updates)
        return result

    def _copy_with(self, **updates):
        """Create a new instance from the current attributes."""
        data = updates.get('data', self._amount)
        name = updates.get('names', self.name())
        unit = updates.get('unit', self.unit())
        return Array(data, unit=unit, name=name)

    def __str__(self) -> str:
        """A simplified representation of this object."""
        # Consider using the `numpy` string helper functions.
        attrs = [
            f"unit='{self.unit()}'",
        ]
        return f"'{self.name()}': {', '.join(attrs)}"

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


class Dimensions(collections.abc.Sequence, iterables.ReprStrMixin):
    """A representation of one or more axis names."""

    def __init__(self, *names: str) -> None:
        self._names = self._init(*names)

    def _init(self, *args):
        names = iterables.unwrap(args)
        if all(isinstance(name, str) for name in names):
            return names
        raise TypeError(
            f"Can't initialize instance of {type(self)}"
            f" with {names!r}"
        )

    @property
    def names(self): # Try to prevent accidentially changing names.
        """The names of these axes."""
        return self._names

    __abs__ = operations.identity(abs)
    """Called for abs(self)."""
    __pos__ = operations.identity(standard.pos)
    """Called for +self."""
    __neg__ = operations.identity(standard.neg)
    """Called for -self."""

    __add__ = operations.identity(standard.add)
    """Called for self + other."""
    __sub__ = operations.identity(standard.sub)
    """Called for self - other."""

    def merge(a, *others):
        """Return the unique axis names in order."""
        if all(isinstance(b, Dimensions) for b in others):
            names = list(a.names)
            for b in others:
                names.extend(b.names)
            return Dimensions(*operations.unique(*names))
        return NotImplemented

    __mul__ = merge
    """Called for self * other."""
    __truediv__ = merge
    """Called for self / other."""

    def __eq__(self, other):
        """True if two instances represent the same axes."""
        return isinstance(other, Dimensions) and other.names == self.names

    def __hash__(self):
        """Support use as a mapping key."""
        return hash(self.names)

    def __len__(self) -> int:
        """Called for len(self)."""
        return len(self.names)

    def __getitem__(self, __i: typing.SupportsIndex):
        """Called for index-based access."""
        return self.names[__i]

    def __str__(self) -> str:
        return ', '.join(str(name) for name in self.names)


Instance = typing.TypeVar('Instance', bound='Variable')


class Variable(Array):
    """A class representing a dataset variable."""

    @typing.overload
    def __init__(
        self: Instance,
        data: numpy.typing.ArrayLike,
        *names: str,
        unit: typing.Union[str, metric.Unit]=None,
        axes: typing.Iterable[str]=None,
    ) -> None:
        """Create a new variable from scratch."""

    @typing.overload
    def __init__(
        self: Instance,
        array: Array,
        axes: typing.Iterable[str]=None,
    ) -> None:
        """Create a new variable from an array."""

    @typing.overload
    def __init__(
        self: Instance,
        instance: Instance,
    ) -> None:
        """Create a new variable from an existing variable."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.axes = self._parse_variable(*args, **kwargs)
        """The names of indexable axes in this variable's array."""
        self.naxes = len(self.axes)
        """The number of indexable axes in this variable's array."""
        if self.naxes != self.ndim:
            raise ValueError(
                f"Number of axes ({self.naxes})"
                f" must equal number of array dimensions ({self.ndim})"
            )

    def _parse_variable(self, *args, **kwargs) -> typing.Tuple[str]:
        """Parse input arguments to initialize this instance."""
        if not kwargs and len(args) == 1 and isinstance(args[0], type(self)):
            return args[0].axes
        return tuple(kwargs.get('axes', ()))

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

    # TODO: Refactor `Array._copy_with` to allow reuse here.
    def _copy_with(self, **updates):
        """Create a new instance from the current attributes."""
        data = updates.get('data', self._data)
        names = updates.get('names', self.names)
        attrs = {
            name: updates.get(name, getattr(self, name))
            for name in ('unit', 'axes')
        }
        if isinstance(names, str):
            return type(self)(data, names, **attrs)
        return type(self)(data, *names, **attrs)

    def __str__(self) -> str:
        """A simplified representation of this object."""
        attrs = [
            f"unit='{self.unit}'",
            f"axes={self.axes}",
        ]
        return f"'{self.names}': {', '.join(attrs)}"


@Variable.implements(numpy.squeeze)
def _squeeze(v: Variable, **kwargs):
    """Remove singular axes."""
    data = v._get_array().squeeze(**kwargs)
    axes = tuple(
        a for a, d in zip(v.axes, v.shape)
        if d != 1
    )
    return Variable(data, *v.names, unit=v.unit, axes=axes)


@Variable.implements(numpy.mean)
def _mean(v: Variable, **kwargs):
    """Compute the mean of the underlying array."""
    data = v._get_array().mean(**kwargs)
    axis = kwargs.get('axis')
    if axis is None:
        return data
    axes = tuple(a for a in v.axes if v.axes.index(a) != axis)
    names = [f"mean({name})" for name in v.names]
    return Variable(data, *names, unit=v.unit, axes=axes)


measurable.Quantity.register(Variable)


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
    a_arr = a._get_array(a_idx)
    b_idx = tuple(idx[axes.index(d)] for d in b.shape_dict)
    b_arr = b._get_array(b_idx)
    return a_arr, b_arr


_opr_rules = {
    'add': {
        (Variable, measurable.RealValued): {},
        (Variable, Variable): {
            'constraints': [same_attrs('axes', 'unit')],
            'updaters': {
                'names': attr_updater('{0.names} + {1.names}'),
            }
        },
        (measurable.RealValued, Variable): {},
    },
    'subtract': {
        (Variable, measurable.RealValued): {},
        (Variable, Variable): {
            'constraints': [same_attrs('axes', 'unit')],
            'updaters': {
                'names': attr_updater('{0.names} - {1.names}'),
            }
        },
        (measurable.RealValued, Variable): {},
    },
    'multiply': {
        (Variable, measurable.RealValued): {},
        (Variable, Variable): {
            'updaters': {
                'unit': attr_updater('{0.unit} * {1.unit}'),
                'axes': unique_axes,
                'names': attr_updater('{0.names} * {1.names}'),
            }
        },
        (measurable.RealValued, Variable): {},
    },
    'true_divide': {
        (Variable, measurable.RealValued): {},
        (Variable, Variable): {
            'updaters': {
                'unit': attr_updater('{0.unit} / ({1.unit})'),
                'axes': unique_axes,
                'names': attr_updater('{0.names} / {1.names}'),
            }
        },
    },
    'power': {
        (Variable, numbers.Real): {
            'updaters': {
                'unit': attr_updater('({0.unit})^{1}'),
                'names': attr_updater('{0.names}^{1}'),
            },
        },
    },
    'sqrt': {
        (Variable,): {
            'updaters': {
                'unit': attr_updater('sqrt({0.unit})'),
                'names': attr_updater('sqrt({0.names})'),
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
        unit: typing.Union[str, metric.Unit],
    ) -> None:
        super().__init__(indices, values)
        self.unit = unit

    def with_unit(self, new: typing.Union[str, metric.Unit]):
        """Convert this object to the new unit, if possible."""
        scale = metric.Unit(new) // self.unit
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
            str(self.reference.unit())
            if isinstance(self.reference, measurable.Quantity)
            else None
        )
        if unit:
            string += f", unit={unit!r}"
        return string


# I'm going to use the `__int__` and `__float__` logic from this to re-implement
# Assumption.
class X_Measurement:
    """The result of measuring an object."""

    def __getitem__(self, index):
        values = super().__getitem__(index)
        iter_values = isinstance(values, typing.Iterable)
        return (
            [Scalar(value, self._unit) for value in values] if iter_values
            else Scalar(values, self._unit)
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
        nv = len(self._values)
        if nv == 1:
            return __type(self._values[0])
        errmsg = f"Can't convert measurement with {nv!r} values to {__type}"
        raise TypeError(errmsg) from None

    def __str__(self) -> str:
        """A simplified representation of this object."""
        return f"{self._values} [{self._unit}]"


class Assumption(measurable.Measurement):
    """A measurable parameter argument."""

    aliases: typing.Tuple[str, ...] = None
    """The known aliases for this assumption."""

    def __new__(
        cls,
        values,
        unit: str,
        *aliases: str,
    ) -> None:
        self = super().__new__(cls, values, unit)
        self.aliases = aliased.MappingKey(aliases)
        return self

    def __str__(self) -> str:
        values = self.values[0] if len(self) == 1 else self.values
        return (
            f"'{self.aliases}': {values} '{self.unit}'" if self.aliases
            else f"{values} '{self.unit}'"
        )


class Option(iterables.ReprStrMixin):
    """An unmeasurable parameter argument."""

    def __init__(self, value, *aliases: str) -> None:
        self._value = value
        self.aliases = aliased.MappingKey(aliases)

    def __eq__(self, other):
        """True if `other` is equivalent to this option's value."""
        if isinstance(other, Option):
            return other._value == self._value
        return other == self._value

    def __str__(self) -> str:
        return (
            f"'{self.aliases}': {self._value}" if self.aliases
            else str(self._value)
        )


