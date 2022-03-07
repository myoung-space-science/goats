import collections.abc
import numbers
import typing

import numpy as np


class NumericalSequence(collections.abc.Sequence):
    """Abstract base class representing a sequence of numerical values.

    This ABC is a partial implementation of `collections.abc.Sequence`, indended
    to serve as a basis for creating array-like objects that interoperate with
    `numpy` arrays. It defines the required `__len__` method but does not define
    the required `__getitem__` method.
    """
    def __init__(self, values: typing.Iterable[numbers.Number]) -> None:
        self.values = values

    def __len__(self) -> int:
        return len(self.values)

    def __array__(self, *args, **kwargs) -> np.ndarray:
        """Support casting to `numpy` array types."""
        return np.asarray(self.values, *args, **kwargs)


class Array(NumericalSequence, np.lib.mixins.NDArrayOperatorsMixin):
    """An indexable numerical sequence."""

    def __init__(
        self,
        values: typing.Iterable[numbers.Number],
        axes: typing.Iterable[str],
    ) -> None:
        super().__init__(values)
        self._axes = axes
        self._data = None

    @property
    def axes(self):
        """The names of this array's indexable axes."""
        return tuple(self._axes)

    @property
    def naxes(self):
        """The number of indexable axes in this array."""
        return len(self.axes)

    @property
    def data(self) -> np.ndarray:
        """The `numpy` representation of this array's values."""
        if self._data is None:
            # FIXME: This is not a rigorous solution to the problem of certain
            # types of `self.values` not producing an array.
            try:
                self._data = np.asarray(self.values)
            except ValueError:
                self._data = np.asarray(self.values[:])
        return self._data

    def __getitem__(self, indices):
        """Access array values via index or slice notation."""
        idx = np.index_exp[indices]
        return self.data[idx]

    _HANDLED_TYPES = (np.ndarray, numbers.Number, list)

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
            if not isinstance(x, self._HANDLED_TYPES + (Array,)):
                return NotImplemented
        inputs = tuple(x.values if isinstance(x, Array) else x for x in inputs)
        if out:
            kwargs['out'] = tuple(
                x.values if isinstance(x, Array) else x for x in out
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
        `__array_function__` to handle `Array` objects.
        """
        accepted = (type(self), np.ndarray) + np.ScalarType
        if not all(issubclass(ti, accepted) for ti in types):
            return NotImplemented
        if func in self._HANDLED_FUNCTIONS:
            arr = self._HANDLED_FUNCTIONS[func](*args, **kwargs)
            return self._new_from_func(arr)
        args = tuple(
            arg
            if not isinstance(arg, type(self)) else arg.values
            for arg in args
        )
        types = tuple(ti for ti in types if not issubclass(ti, type(self)))
        values = self.__array__()
        arr = values.__array_function__(func, types, args, kwargs)
        return self._new_from_func(arr)

    def _new_from_func(self, result):
        """Create a new instance from the result of a `numpy` function."""
        if not isinstance(result, np.ndarray):
            return result
        return Array(result, self.axes)

    def __str__(self) -> str:
        """A simplified representation of this object."""
        return np.array2string(self.data, separator=', ')

    def __repr__(self) -> str:
        """An unambiguous representation of this object.

        See
        https://numpy.org/doc/stable/reference/generated/numpy.array2string.html#numpy.array2string
        for `array2string` options.
        """
        name = self.__class__.__qualname__
        prefix = f"{name}("; suffix = ")"
        string = np.array2string(
            self.data, prefix=prefix, suffix=suffix, separator=', ',
        )
        indent = ' ' * len(prefix)
        return f"{name}{self.axes}(\n{indent}{string}\n)"

    @classmethod
    def implements(cls, numpy_function):
        """Register an `__array_function__` implementation for Array.

        See https://numpy.org/doc/stable/reference/arrays.classes.html for the
        suggestion on which this method is based.

        EXAMPLE
        -------
        Override `numpy.mean` with a version that accepts no keyword arguments::

            @Array.implements(np.mean)
            def mean(a: Array, **kwargs) -> Array:
                if kwargs:
                    msg = "Cannot pass keywords to numpy.mean with Array"
                    raise TypeError(msg)
                return np.sum(a) / len(a)

        This will compute the mean of the underlying values when called with no
        arguments, but will raise an exception when called with arguments:

            >>> v = Array([[1, 2], [3, 4]])
            >>> np.mean(v)
            5.0
            >>> np.mean(v, axis=0)
            ...
            TypeError: Cannot pass keywords to numpy.mean with Array

        """
        def decorator(func):
            cls._HANDLED_FUNCTIONS[numpy_function] = func
            return func
        return decorator


@Array.implements(np.mean)
def _array_mean(a: Array, **kwargs):
    """Compute the mean and update array dimensions, if necessary."""
    values = a.data.mean(**kwargs)
    if (axis := kwargs.get('axis')) is not None:
        a._axes = tuple(
            d for d in a.axes
            if a.axes.index(d) != axis
        )
    return values


