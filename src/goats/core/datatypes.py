import collections.abc
import contextlib
import fractions
import numbers
import operator as standard
import typing

import numpy
import numpy.typing

from goats.core import aliased
from goats.core import iterables
from goats.core import metric
from goats.core import measurable
from goats.core import metadata


_metadata_mixins = (
    numpy.lib.mixins.NDArrayOperatorsMixin,
    iterables.ReprStrMixin,
)


class Name(collections.abc.Collection, *_metadata_mixins):
    """The name attribute of a data quantity."""

    def __init__(self, *aliases: str) -> None:
        self._aliases = aliased.MappingKey(*aliases)

    def add(self, aliases: typing.Union[str, typing.Iterable[str]]):
        """Add `aliases` to this name."""
        self._aliases = self._aliases | aliases
        return self

    def __array_ufunc__(self, ufunc, method, *args, **kwargs):
        """Provide support for `numpy` universal functions."""
        if func := getattr(self, f'_ufunc_{ufunc.__name__}', None):
            return func(*args, **kwargs)

    def _ufunc_sqrt(self, arg):
        """Implement the square-root function for a name."""
        return arg**'1/2'

    def _implement(symbol: str, reverse: bool=False, strict: bool=False):
        """Implement a symbolic operation.

        This function creates a new function that symbolically represents the
        result of applying an operator to two operands, `a` and `b`.
        
        Parameters
        ----------
        symbol : string
            The string representation of the operator.

        reverse : bool, default=False
            If true, apply the operation to reflected operands.

        strict : bool, default=False
            If true, require that `b` be an instance of `a`'s type or a subtype.

        Raises
        ------
        TypeError
            `strict` is true and `b` is not an instance of `a`'s type or a
            subtype.

        Notes
        -----
        * The nullspace for names is the empty string.
        * 'a | A' + 2 -> undefined
        * 'a | A' + 'a | A' -> 'a | A' when `strict` is true
        * 'a | A' * 'a | A' -> 'a*a | A*A' when `strict` is false
        * 'a | A' + 'b | B' -> 'a+b | a+B | A+b | A+B'
        * 'a | A' * 'b | B' -> 'a*b | a*B | A*b | A*B'
        * 'a | A' * 2 -> 'a*2 | A*2'
        """
        def compute(a, b):
            """Symbolically combine `a` and `b`."""
            x, y = (b, a) if reverse else (a, b)
            if isinstance(y, typing.Iterable) and not isinstance(y, str):
                return [f'{i}{symbol}{j}' for i in x for j in y]
            try:
                fixed = fractions.Fraction(y)
            except ValueError:
                fixed = y
            t = '{1}{s}{0}' if reverse else '{0}{s}{1}'
            return [t.format(i, fixed, s=symbol) for i in x]
        def operator(self, that):
            if not self or not that:
                return ['']
            if strict:
                if not isinstance(that, type(self)):
                    raise TypeError(
                        f"Can't apply {symbol} "
                        f"to {type(self)!r} and {type(that)!r}"
                    ) from None
                if that == self:
                    return self
            if that == self:
                return [f'{i}{symbol}{i}' for i in self]
            return compute(that, self) if reverse else compute(self, that)
        s = f"other {symbol} self" if reverse else f"self {symbol} other"
        operator.__doc__ = f"Called for {s}"
        return operator

    __add__ = _implement(' + ', strict=True)
    __radd__ = _implement(' + ', strict=True, reverse=True)
    __sub__ = _implement(' - ', strict=True)
    __rsub__ = _implement(' - ', strict=True, reverse=True)
    __mul__ = _implement(' * ')
    __rmul__ = _implement(' * ', reverse=True)
    __truediv__ = _implement(' / ')
    __rtruediv__ = _implement(' / ', reverse=True)
    __pow__ = _implement('^')
    __rpow__ = _implement('^', reverse=True)

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
    operator implementations provided by `~measurable.OperatorMixin`.
    """

    __metadata__ = 'name'

    __measurable_operators__ = [
        '__abs__', '__pos__', '__neg__',
        '__lt__', '__le__', '__gt__', '__ge__',
        '__add__', '__radd__',
        '__sub__', '__rsub__',
        '__mul__', '__rmul__',
        '__truediv__', '__rtruediv__',
        '__pow__', '__rpow__',
    ]

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
        if not kwargs and len(args) == 1 and isinstance(args[0], type(self)):
            instance = args[0]
            data, unit, name = instance.data, instance.unit, instance.name
        else:
            pos = list(args)
            data = kwargs.get('data') or pos.pop(0)
            unit = kwargs.pop('unit', None) or self._next_arg(pos, '1')
            name = kwargs.pop('name', None) or self._next_arg(pos, '')
        self._name = Name(name)
        super().__init__(data, unit)
        self._name = Name(name)
        if self._name:
            self.display['__str__']['strings'].insert(0, "'{name}': ")
            self.display['__repr__']['strings'].insert(1, "'{name}'")

    @property
    def name(self):
        """This quantity's name."""
        return self._name

    def alias(self, *updates: str, reset: bool=False):
        """Set or add to this object's name(s)."""
        aliases = updates if reset else self._name.add(updates)
        self._name = Name(*aliases)
        return self


class Scalar(Quantity):
    """A single-valued data-type quantity.
    
    This class is a virtual concrete implementation of `~measurable.Scalar`. It
    defines the magic method `__measure__` in order to directly support
    `~measurables.measure`.
    """

    __measurable_operators__ = [
        '__int__', '__float__',
        '__ceil__', '__floor__', '__round__', '__trunc__',
    ]

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
        return measurable.Measurement(value, self.unit)


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
        unit = self.unit
        return (
            [Scalar(value, unit) for value in values] if iter_values
            else Scalar(values, unit)
        )

    def __measure__(self):
        """Create a measurement from this vector's values and unit."""
        return measurable.Measurement(self.data, self.unit)

    def __eq__(self, other: typing.Any):
        """True if two instances have the same data and attributes."""
        if not isinstance(other, type(self)):
            return NotImplemented
        if not self.unit == other.unit:
            return False
        return numpy.array_equal(other, self)


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
        self._rescale = False
        self._array = None

    def convert(self, unit: metric.UnitLike):
        """Set the unit of this object's values."""
        if unit == self._metric:
            return self
        new = metric.Unit(unit)
        self._scale *= new // self._metric
        self._metric = new
        self._rescale = True
        return self

    def __eq__(self, other: typing.Any):
        """True if two instances have the same data and attributes."""
        if not isinstance(other, type(self)):
            return NotImplemented
        if not self.unit == other.unit:
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
        # NOTE: `NDArrayOperatorsMixin` recommends using the actual class (i.e.,
        # `Array` in this case) instead of `type(self)` to allow subclasses that
        # don't override __array_ufunc__ to support `Array` instances. We could
        # define a base `_HANDLED_TYPES`, then update an instance attribute
        # (e.g., `_ufunc_types`) in `__init__` and refer to that here.
        for x in args + out:
            if not isinstance(x, self._HANDLED_TYPES + (type(self),)):
                return NotImplemented
        if out:
            kwargs['out'] = tuple(
                x._get_array() if isinstance(x, type(self))
                else x for x in out
            )
        name = ufunc.__name__
        operands = self._ufunc_hook(ufunc, *args)
        compute = getattr(ufunc, method)
        data = compute(*operands, **kwargs)
        evaluate = (
            self.metadata[name].evaluate if name in self.metadata
            else self.metadata.implement(compute)
        )
        meta = evaluate(*args, **kwargs)
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
        data = self._get_array()
        return numpy.asanyarray(data, *args, **kwargs)

    def _get_array(self, index: IndexLike=None):
        """Access array data via index or slice notation.
        
        Notes
        -----
        If `index` is not `None`, this method will create the requested subarray
        from `self.data` and directly return it. If `index` is `None`, this
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
        # NOTE: The use of `rescale` is one option. Another would be to keep a
        # copy of `scale` in `old_scale` and enter this block if `scale`
        # changes.
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


# I really want to call this `Axes`, but that will require first renaming the
# existing `Axes` class.
class Dimensions(collections.abc.Sequence, iterables.ReprStrMixin):
    """A representation of one or more axis names."""

    def __init__(self, *names: str) -> None:
        self._names = self._init(*names)

    def _init(self, *args):
        names = iterables.unwrap(args, wrap=tuple)
        if all(isinstance(name, str) for name in names):
            return names
        raise TypeError(
            f"Can't initialize instance of {type(self)}"
            f" with {names!r}"
        )

    @property
    def names(self): # Try to prevent accidentially changing names.
        """The names of these axes."""
        return tuple(self._names)

    __abs__ = metadata.identity(abs)
    """Called for abs(self)."""
    __pos__ = metadata.identity(standard.pos)
    """Called for +self."""
    __neg__ = metadata.identity(standard.neg)
    """Called for -self."""

    __add__ = metadata.identity(standard.add)
    """Called for self + other."""
    __sub__ = metadata.identity(standard.sub)
    """Called for self - other."""

    def merge(a, *others):
        """Return the unique axis names in order."""
        names = list(a.names)
        for b in others:
            if isinstance(b, Dimensions):
                names.extend(b.names)
        return Dimensions(*iterables.unique(*names))

    __mul__ = merge
    """Called for self * other."""
    __rmul__ = merge
    """Called for other * self."""
    __truediv__ = merge
    """Called for self / other."""

    def __eq__(self, other):
        """True if self and other represent the same axes."""
        return (
            isinstance(other, Dimensions) and other.names == self.names
            or (
                isinstance(other, str)
                and len(self) == 1
                and other == self.names[0]
            )
            or (
                isinstance(other, typing.Iterable)
                and len(other) == len(self)
                and all(i in self for i in other)
            )
        )

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

    __metadata__ = 'axes'

    @typing.overload
    def __init__(
        self: Instance,
        data: numpy.typing.ArrayLike,
        unit: typing.Union[str, metric.Unit]=None,
        name: typing.Union[str, typing.Iterable[str]]=None,
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
        if not kwargs and len(args) == 1 and isinstance(args[0], type(self)):
            instance = args[0]
            data, unit, name = instance.data, instance.unit, instance.name
            axes = instance.axes
        else:
            pos = list(args)
            n = len(pos)
            two_args = n == 2 or (n == 1 and len(kwargs) == 1)
            if two_args and isinstance(pos[0], Array):
                array = pos.pop(0)
                args = [
                    getattr(array, name)
                    for name in ('data', 'unit', 'name')
                ]
                pos = list(args) + pos
            data = kwargs.get('data') or pos.pop(0)
            unit = kwargs.pop('unit', None) or self._next_arg(pos, '1')
            name = kwargs.pop('name', None) or self._next_arg(pos, '')
            axes = kwargs.pop('axes', None) or self._next_arg(pos, ())
        super().__init__(data, unit, name)
        self.axes = Dimensions(axes)
        self.naxes = len(self.axes)
        """The number of indexable axes in this variable's array."""
        if self.naxes != self.ndim:
            raise ValueError(
                f"Number of axes ({self.naxes})"
                f" must equal number of array dimensions ({self.ndim})"
            )
        self.display['__str__']['strings'].append("axes={axes}")
        self.display['__repr__']['strings'].append("axes={axes}")

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
            x._get_array() if isinstance(x, type(self))
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
    data = v._get_array().squeeze(**kwargs)
    axes = tuple(
        a for a, d in zip(v.axes, v.shape)
        if d != 1
    )
    return Variable(data, unit=v.unit, name=v.name, axes=axes)


@Variable.implements(numpy.mean)
def _mean(v: Variable, **kwargs):
    """Compute the mean of the underlying array."""
    data = v._get_array().mean(**kwargs)
    axis = kwargs.get('axis')
    if axis is None:
        return data
    axes = tuple(a for a in v.axes if v.axes.index(a) != axis)
    name = [f"mean({name})" for name in v.name]
    return Variable(data, unit=v.unit, name=name, axes=axes)


measurable.Quantity.register(Variable)


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


class Option(iterables.ReprStrMixin):
    """An unmeasurable parameter argument."""

    def __init__(
        self,
        value,
        name: typing.Union[str, typing.Iterable[str]]=None,
    ) -> None:
        self._value = value
        self._name = aliased.MappingKey(name)
        if self._name:
            self.display['__str__']['strings'].insert(0, "'{name}': ")
            self.display['__repr__']['strings'].insert(1, "'{name}'")

    def __eq__(self, other):
        """True if `other` is equivalent to this option's value."""
        if isinstance(other, Option):
            return other._value == self._value
        return other == self._value

    def name(self, *new: str, reset: bool=False):
        """Get, set, or add to this object's name(s)."""
        if not new:
            return self._name
        name = new if reset else self._name | new
        self._name = aliased.MappingKey(name)
        return self



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

    __slots__ = ('_unit',)

    def __init__(
        self,
        indices: typing.Iterable[int],
        values: typing.Iterable[typing.Any],
        unit: typing.Union[str, metric.Unit],
    ) -> None:
        super().__init__(indices, values)
        self._unit = unit

    def unit(self, unit: metric.UnitLike=None):
        """Convert this object to the new unit, if possible."""
        if not unit:
            return self._unit
        if unit == self._unit:
            return self
        scale = metric.Unit(unit) // self._unit
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


