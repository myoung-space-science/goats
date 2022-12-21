"""
Objects that support access to an observer's operational parameters and
arguments.
"""
import abc
import numbers
import typing

import numpy
import numpy.typing

from goats.core import algebraic
from goats.core import iterables
from goats.core import metadata
from goats.core import metric
from goats.core import physical


@typing.runtime_checkable
class _HasDataAndUnit(typing.Protocol):
    """Protocol for objects with `data` and `unit` properties."""

    __slots__ = ()

    @property
    @abc.abstractmethod
    def data(self) -> algebraic.Real:
        pass

    @property
    @abc.abstractmethod
    def unit(self) -> metric.Unit:
        pass


class NumericCastError(Exception):
    """Unable to cast to a numeric type."""


class Argument:
    """An arbitrary operational argument.
    
    Operational arguments comprise two categories: assumptions and options.
    
    - Assumptions represent algebraically real-valued objects (cf.
      `~algebraic.Real`) with an associated metric unit. An assumption with a
      unitless value has a metric unit of '1'. If the argument is multivalued,
      all values must have the same unit.
    
    - Options represent strings, integer-like values, or iterables of those
      types. An option's metric unit is always `None` (as opposed to unitless)
      because the notion of measuring a non-physical quantity is meaningless.
      Integer-like options support conversion to the equivalent integer.
    """

    def __new__(cls, *args, **kwargs):
        """Reject incorrect non-physical data types."""
        if not kwargs and len(args) == 1:
            arg = args[0]
            valid = (numbers.Integral, str, _HasDataAndUnit)
            if not iterables.hastype(arg, valid, strict=True):
                raise TypeError(
                    f"An argument of type {type(arg)} requires a unit"
                ) from None
        return super().__new__(cls)

    @typing.overload
    def __init__(
        self,
        __data: typing.Union[numbers.Integral, str],
    ) -> None:
        """Represent a non-physical argument."""

    @typing.overload
    def __init__(
        self,
        __data: typing.Union[algebraic.Real, numpy.typing.ArrayLike],
        unit: typing.Union[str, metric.Unit],
    ) -> None:
        """Represent a physical argument."""

    @typing.overload
    def __init__(
        self,
        __data: _HasDataAndUnit,
    ) -> None:
        """Convert an object into a physical argument."""

    def __init__(self, *args, **kwargs) -> None:
        data, unit = self._init_from(*args, **kwargs)
        if unit:
            self._data = numpy.array(data, ndmin=1)
            self._unit = metadata.Unit(unit)
        else:
            self._data = [data] if isinstance(data, str) else data
            self._unit = None

    def _init_from(self, *args, **kwargs):
        """Internal initialization logic."""
        if len(args) == 1:
            arg = args[0]
            if isinstance(arg, _HasDataAndUnit):
                return arg.data, arg.unit
            return arg, kwargs.get('unit')
        if len(args) != 2:
            raise TypeError(
                f"Wrong arguments to {self.__class__}"
                f": {args}, {kwargs}"
            ) from None
        if kwargs:
            raise TypeError(
                f"Too many arguments to {self.__class__}"
                f": {args}, {kwargs}"
            ) from None
        return args

    def __bool__(self) -> bool:
        """Called for bool(self).
        
        Notes
        -----
        - If `data` is a number, this object is unconditionally true. This
          design choice is intended to give a value of `0` the same status as
          other numerical values. It therefore requires that any argument
          representing a boolean value use `True` or `False`.
        - If `data` is a string, the truth of this object is equal to the truth
          of that string after stripping leading and trailing whitespace. This
          design choice is intended to treat strings that contain only
          whitespace as the empty string.
        - Otherwise, the truth of this object is equal to the truth of `data`.
        """
        if isinstance(self.data, numbers.Real):
            if isinstance(self.data, bool):
                # because `bool` <: `int` <: `numbers.Real`
                return self.data
            return True
        if isinstance(self.data, str):
            return bool(self.data.strip())
        return bool(self.data)

    def __float__(self):
        """Represent a single-valued physical argument as a `float`."""
        if self.unit:
            return self._cast_to(float)
        return NotImplemented

    def __int__(self):
        """Represent a single-valued argument as a `int`."""
        return self._cast_to(int)

    T = typing.TypeVar('T', int, float)

    def _cast_to(self, __type: typing.Type[T]) -> T:
        """Internal method for casting to numeric type."""
        try:
            nv = len(self.data)
        except TypeError:
            nv = 0
        if nv > 1:
            errmsg = f"Can't convert data with {nv} values to {__type}"
            raise NumericCastError(errmsg) from None
        if nv == 1:
            return __type(self.data[0])
        try: # nv == 0
            return __type(self.data)
        except ValueError as err:
            raise ValueError(f"Can't convert {self!r} to {__type}.") from err

    def __iter__(self) -> typing.Iterator:
        """Called for iter(self)."""
        try:
            return iter(self.data)
        except TypeError:
            return iter([self.data])

    def __len__(self) -> int:
        """Called for len(self)."""
        try:
            return len(self.data)
        except TypeError:
            return 1

    def __getitem__(self, index):
        """Called for index-based value access."""
        if isinstance(index, typing.SupportsIndex) and index < 0:
            index += len(self)
        data = self.data[index]
        try:
            iter(data)
        except TypeError:
            iterable = False
        else:
            iterable = True
        unit = self.unit
        return (
            [physical.Scalar(value, unit=unit) for value in data] if iterable
            else physical.Scalar(data, unit=unit)
        )

    @property
    def data(self):
        """This argument's value(s)."""
        try:
            nv = len(self._data)
        except TypeError:
            nv = 0
        if nv == 1:
            return self._data[0]
        return self._data

    @property
    def unit(self):
        """The unit of `data`, if any."""
        return self._unit

    def __eq__(self, other):
        """Called for self == other."""
        if isinstance(other, Argument):
            same_data = other.data == self.data
            if self.unit:
                return same_data and other.unit == self.unit
            return same_data
        return other == self.data

    def __repr__(self) -> str:
        """An unambiguous representation of this argument."""
        return f"{self.__class__.__qualname__}({self})"

    def __str__(self) -> str:
        """A simplified representation of this argument."""
        if self.unit and self.unit != '1':
            return f"{self.data} {str(self.unit)!r}"
        return str(self.data)
