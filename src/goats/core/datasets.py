"""Tools for managing datasets."""

import abc
import collections.abc
import numbers
import typing

import netCDF4
import numpy
import numpy.typing

from goats.core import aliased
from goats.core import iotools
from goats.core import iterables
from goats.core import observables
from goats.core import quantities


class DataViewer(collections.abc.Mapping):
    """An abstract base class for data-viewing objects."""

    def __init__(self, path: iotools.ReadOnlyPath) -> None:
        self.members = self.get_members(path)

    def __iter__(self) -> typing.Iterator:
        return iter(self.members)

    def __len__(self) -> int:
        return len(self.members)

    @abc.abstractmethod
    def get_members(self, path: iotools.ReadOnlyPath) -> typing.Mapping:
        """Get the appropriate members for this viewer."""
        pass

    def __str__(self) -> str:
        """A simplified representation of this object."""
        return '\n\n'.join(f"{k}:\n{v!r}" for k, v in self.items())

    def __repr__(self) -> str:
        """An unambiguous representation of this object."""
        return f"\n::{self.__class__.__qualname__}::\n\n{self}"


class DatasetVariable(typing.NamedTuple):
    """A dataset variable."""

    data: numpy.typing.ArrayLike
    unit: str
    axes: typing.Tuple[str]
    name: str

    def __str__(self) -> str:
        """A simplified representation of this object."""
        return self._display(sep='\n')

    def __repr__(self) -> str:
        """An unambiguous representation of this object."""
        display = self._display(sep='\n', tab=4)
        return f"{self.__class__.__qualname__}(\n{display}\n)"

    def _display(
        self,
        sep: str=', ',
        tab: int=0,
    ) -> str:
        """Helper for `__str__` and `__repr__`."""
        attrs = [
            f"data={type(self.data)}",
            f"unit={self.unit!r}",
            f"axes={self.axes}",
            f"name={self.name!r}",
        ]
        indent = ' ' * tab
        return sep.join(f"{indent}{attr}" for attr in attrs)


class NetCDFVariables(DataViewer):
    """An object for viewing variables in a NetCDF dataset."""

    def get_members(self, path: iotools.ReadOnlyPath) -> typing.Mapping:
        dataset = netCDF4.Dataset(path, 'r')
        return dataset.variables

    def __getitem__(self, name: str) -> DatasetVariable:
        if name in self.members:
            data = self.members[name]
            unit = self._get_unit_from_data(data)
            axes = self._get_axes_from_data(data)
            return DatasetVariable(data, unit, axes, name)
        raise KeyError(f"No variable called '{name}'")

    def _get_axes_from_data(self, data):
        """Compute appropriate variable axes from a dataset object."""
        return tuple(getattr(data, 'dimensions', ()))

    def _get_unit_from_data(self, data):
        """Compute appropriate variable units from a dataset object."""
        available = (
            getattr(data, attr) for attr in ('unit', 'units')
            if hasattr(data, attr)
        )
        return next(available, None)


class DatasetAxis(typing.NamedTuple):
    """A dataset axis."""

    size: int
    name: str

    def __str__(self) -> str:
        """A simplified representation of this object."""
        return f"{self.name!r}, size={self.size}"

    def __repr__(self) -> str:
        """An unambiguous representation of this object."""
        module = f"{self.__module__.replace('goats.', '')}."
        name = self.__class__.__qualname__
        return f"{module}{name}({self})"


class NetCDFAxes(DataViewer):
    """An object for viewing axes in a NetCDF dataset."""

    def get_members(self, path: iotools.ReadOnlyPath) -> typing.Mapping:
        dataset = netCDF4.Dataset(path, 'r')
        return dataset.dimensions

    def __getitem__(self, name: str) -> DatasetAxis:
        if name in self.members:
            data = self.members[name]
            size = getattr(data, 'size', None)
            return DatasetAxis(size, name)
        raise KeyError(f"No axis corresponding to {name!r}")


class ViewerFactory(collections.abc.MutableMapping):
    """A class that creates appropriate viewers for a dataset."""

    _viewer_map = {
        '.nc': {
            'variables': NetCDFVariables,
            'axes': NetCDFAxes,
        }
    }

    def __init__(self, path: iotools.ReadOnlyPath) -> None:
        self._viewers = self._get_viewers(path)
        self.path = path

    def __iter__(self) -> typing.Iterator:
        return iter(self._viewers)

    def __len__(self) -> int:
        return len(self._viewers)

    def _get_viewers(
        self,
        path: iotools.ReadOnlyPath,
    ) -> typing.Dict[str, typing.Type[DataViewer]]:
        """Get the viewers for this file format.

        This may expand to accommodate additional file formats or alternate
        methods.
        """
        try:
            viewers = self._viewer_map[path.suffix]
        except KeyError:
            TypeError(f"Unrecognized file type for {path}")
        else:
            return viewers

    def __getitem__(self, group: str):
        """Get the appropriate viewer for this dataset group."""
        if group in self._viewers:
            viewer = self._viewers[group]
            return viewer(self.path)
        raise KeyError(f"No viewer for group '{group}' in {self.path}")

    def __setitem__(self, group: str, viewer: typing.Type[DataViewer]):
        """Associate a new viewer with this dataset group."""
        self._viewers[group] = viewer

    def __delitem__(self, group: str):
        """Delete the viewer associated with this dataset group."""
        del self._viewers[group]


class SubsetKeys(typing.NamedTuple):
    """A subset of names of attributes."""

    full: typing.Tuple[str]
    aliased: typing.Tuple[aliased.MappingKey]
    canonical: typing.Tuple[str]


class DatasetView(iterables.ReprStrMixin, metaclass=iotools.PathSet):
    """A format-agnostic view of a dataset.
    
    An instance of this class provides aliased access to variables and axes
    defined in a specific dataset, given a path to that dataset. It is designed
    to provide a single interface, regardless of file type, with as little
    overhead as possible. Therefore, it does not attempt to modify attributes
    (e.g., converting variable units), since doing so could result in reading a
    potentially large array from disk.
    """

    def __init__(self, path: iotools.PathLike) -> None:
        self.path = iotools.ReadOnlyPath(path)
        self.viewers = ViewerFactory(self.path)
        self._variables = None
        self._axes = None
        self._units = None

    @property
    def variables(self):
        """The variables in this dataset."""
        if self._variables is None:
            variables = {
                observables.ALIASES.get(name, name): variable
                for name, variable in self.viewers['variables'].items()
            }
            self._variables = aliased.Mapping(variables)
        return self._variables

    @property
    def units(self):
        """The unit of each variable, if available."""
        if self._units is None:
            units = {
                name: variable.unit
                for name, variable in self.variables.items(aliased=True)
            }
            self._units = aliased.Mapping(units)
        return self._units

    @property
    def axes(self):
        """The axes in this dataset."""
        if self._axes is None:
            axes = {
                observables.ALIASES.get(name, name): axis
                for name, axis in self.viewers['axes'].items()
            }
            self._axes = aliased.Mapping(axes)
        return self._axes

    def available(self, key: str):
        """Provide the names of available attributes."""
        if key in {'variable', 'variables'}:
            return SubsetKeys(
                full=tuple(self.variables),
                aliased=tuple(self.variables.keys(aliased=True)),
                canonical=tuple(self.viewers['variables'].keys()),
            )
        if key in {'axis', 'axes'}:
            return SubsetKeys(
                full=tuple(self.axes),
                aliased=tuple(self.axes.keys(aliased=True)),
                canonical=tuple(self.viewers['axes'].keys()),
            )

    def iter_axes(self, name: str):
        """Iterate over the axes for the named variable."""
        this = self.variables[name].axes if name in self.variables else ()
        return iter(this)

    def resolve_axes(self, names: typing.Iterable[str]):
        """Compute and order the available axes in `names`."""
        axes = self.available('axes').canonical
        return tuple(name for name in axes if name in names)

    def use(self, **viewers) -> 'DatasetView':
        """Update the viewers for this instance."""
        self.viewers.update(viewers)
        return self

    def __str__(self) -> str:
        return str(self.path)


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
        return quantities.Measurement(self._amount, self.unit())

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


S = typing.TypeVar('S', bound='Variables')


class Variables(aliased.Mapping):
    """An interface to dataset variables.
    
    This class provides aliased key-based access to all variables in a dataset.
    It converts each requested dataset variable into a `~quantities.Variable`
    instance with the appropriate MKS unit.
    """

    def __init__(self, dataset: DatasetView) -> None:
        known = {
            k: v for k, v in dataset.variables.items(aliased=True)
            if k in observables.METADATA
        }
        super().__init__(known)
        self._system = quantities.MetricSystem('mks')
        self._units = None

    @property
    def units(self):
        """The MKS unit of each variable."""
        if self._units is None:
            units = {
                name: self._get_unit(name)
                for name in self.keys(aliased=True)
            }
            self._units = aliased.Mapping(units)
        return self._units

    def _get_unit(self, name: str):
        """Get a standard unit for the named variable."""
        metric = self._system[observables.METADATA[name]['quantity']]
        return quantities.Unit(metric.unit)

    def __getitem__(self, key: str):
        """Create the named variable, if possible."""
        variable = super().__getitem__(key)
        unit = self.units[key]
        axes = variable.axes
        name = observables.ALIASES[key]
        scale = (unit // standardize(variable.unit))
        data = scale * variable.data[:]
        return Variable(data, unit, axes, name=name)


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
        """True if two instances have the same indices."""
        if not isinstance(other, Indices):
            return NotImplemented
        return all(
            getattr(self, attr) == getattr(other, attr)
            for attr in self.__slots__
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


Indexers = typing.TypeVar('Indexers', bound=typing.Mapping)
Indexers = typing.Mapping[str, Indexer]


class Axes(aliased.Mapping):
    """An interface to dataset axes."""

    def __init__(
        self,
        dataset: DatasetView,
        factory: typing.Type[Indexers],
    ) -> None:
        indexers = factory(dataset)
        super().__init__(indexers)
        self.dataset = dataset

    def __getitem__(self, key: str) -> Axis:
        indexer = super().__getitem__(key)
        size = self.dataset.axes[key].size
        name = f"'{observables.ALIASES.get(key, key)}'"
        return Axis(size, indexer, name=name)


substitutions = {
    'julian date': 'day',
    'shell': '1',
    'cos(mu)': '1',
    'e-': 'e',
    '# / cm^2 s sr MeV': '# / cm^2 s sr MeV/nuc',
}
"""Conversions from non-standard units."""

T = typing.TypeVar('T')

def standardize(unit: T):
    """Replace this unit string with a standard unit string, if possible.

    This function looks for `unit` in the known conversions and returns the
    standard unit string if it exists. If this doesn't find a standard unit
    string, it just returns the input.
    """
    unit = substitutions.get(str(unit), unit)
    if '/' in unit:
        num, den = str(unit).split('/', maxsplit=1)
        unit = ' / '.join((num.strip(), f"({den.strip()})"))
    return unit


