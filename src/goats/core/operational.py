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


class Argument:
    """Base class for arbitrary operational arguments."""

    def __init__(self, data, unit) -> None:
        self._data = data
        self._unit = unit

    @property
    def data(self):
        """This argument's value(s)."""
        return self._data

    @property
    def unit(self):
        """The unit of `data`, if any."""
        return self._unit


class Assumption(Argument):
    """A physical operational argument.
    
    Assumptions represent algebraically real-valued objects (cf.
    `~algebraic.Real`) with an associated metric unit. An assumption with a
    unitless value has a metric unit of '1'. If the argument is multivalued, all
    values must have the same unit.

    See Also
    --------
    `~Option`
        An argument for which the notion of a metric unit is meaningless.
    """

    @typing.overload
    def __init__(
        self,
        __data: typing.Union[algebraic.Real, numpy.typing.ArrayLike],
        unit: typing.Union[str, metric.Unit]='1',
    ) -> None:
        """Create an instance from data and a unit."""

    @typing.overload
    def __init__(
        self,
        __data: _HasDataAndUnit,
    ) -> None:
        """Create an instance from an object with data and a unit."""

    def __init__(self, *args, **kwargs) -> None:
        data, unit = self._init_from(*args, **kwargs)
        super().__init__(numpy.array(data, ndmin=1), metadata.Unit(unit))

    def _init_from(self, *args, **kwargs):
        """Internal initialization logic."""
        if len(args) == 1:
            arg = args[0]
            if isinstance(arg, _HasDataAndUnit):
                return arg.data, arg.unit
            return arg, kwargs.get('unit') or '1'
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

    def __float__(self):
        """Represent a single-valued measurement as a `float`."""
        return self._cast_to(float)

    def __int__(self):
        """Represent a single-valued measurement as a `int`."""
        return self._cast_to(int)

    T = typing.TypeVar('T', int, float)

    def _cast_to(self, __type: typing.Type[T]) -> T:
        """Internal method for casting to numeric type."""
        nv = len(self.data)
        if nv == 1:
            return __type(self.data[0])
        errmsg = f"Can't convert measurement with {nv!r} values to {__type}"
        raise TypeError(errmsg) from None

    def __eq__(self, other):
        """True if `other` is equivalent to this option's data."""
        if isinstance(other, Assumption):
            return other.data == self.data and other.unit == self.unit
        return other == self.data

    def __repr__(self) -> str:
        """An unambiguous representation of this object."""
        args = f"{self.data}, unit={str(self.unit)!r}"
        return f"{self.__class__.__qualname__}({args})"


class Option(Argument):
    """A non-physical operational argument.
    
    Options represent strings, integer-like values, or iterables of those types.
    An option's metric unit is always `None` (as opposed to unitless) because
    the notion of measuring a non-physical quantity is meaningless. Integer-like
    options support conversion to the equivalent integer.

    See Also
    --------
    `~Assumption`
        A real-valued argument with an associated metric unit.
    """

    def __new__(cls, value):
        """Reject incorrect `value` types."""
        if iterables.hastype(value, (numbers.Integral, str)):
            return super().__new__(cls)
        raise TypeError(
            f"An Option may not represent a value of type {type(value)}"
        ) from None

    def __init__(self, value):
        super().__init__(value, None)

    def __bool__(self) -> bool:
        """Called for bool(self).
        
        Notes
        -----
        - If `data` is an integer, this object is unconditionally true. This
          design choice is intended to give a value of `0` the same status as
          other integral values.
        - If `data` is a string, the truth of this object is equal to the truth
          of that string after stripping leading and trailing whitespace. This
          design choice is intended to treat strings that contain only
          whitespace as the empty string.
        - Otherwise, the truth of this object is equal to the truth of `data`.
        """
        if isinstance(self.data, int):
            return True
        if isinstance(self.data, str):
            return bool(self.data.strip())
        return bool(self.data)

    def __int__(self) -> int:
        """Called for int(self)."""
        try:
            return int(self.data)
        except ValueError as err:
            raise ValueError(
                f"Can't convert {self!r} to integer."
            ) from err

    def __eq__(self, other):
        """True if `other` is equivalent to this option's data."""
        if isinstance(other, Option):
            return other.data == self.data
        return other == self.data

    def __repr__(self) -> str:
        """An unambiguous representation of this object."""
        return f"{self.__class__.__qualname__}({self.data!r})"

