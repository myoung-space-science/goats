import abc
import math
import operator as standard
import typing

from goats.core import algebraic
from goats.core import datatypes
from goats.core import iterables
from goats.core import measurable
from goats.core import metadata
from goats.core import metric


# Goal: Create classes with zero or more metadata attributes, to which they can
# add their own data value(s) or callable. Not all classes will inherit from
# `metadata.Quantified`. For example, an observable quantity takes a callable
# implementation object with which it creates an observed quantity.

class Quantity:
    """An observable quantity."""

    def __init__(
        self,
        __implementation,
        unit: metadata.UnitLike=None,
        name: typing.Union[str, typing.Iterable[str], metadata.Name]=None,
        axes: typing.Union[str, typing.Iterable[str], metadata.Axes]=None,
    ) -> None:
        self._unit = unit
        self._name = name
        self._axes = axes
        super().__init__(__implementation)

    def apply_conversion(self, new: metric.Unit):
        self._unit = new


