import typing

from goats.core import aliased
from goats.core import datatypes
from goats.core import iterables
from goats.core import metadata


class Assumption(datatypes.Vector):
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
                datatypes.Scalar(value, unit=self.unit, name=self.name)
                for value in values
            ] if iter_values
            else datatypes.Scalar(values, unit=self.unit, name=self.name)
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


class Option(metadata.NameMixin, iterables.ReprStrMixin):
    """An unmeasurable parameter argument."""

    def __init__(
        self,
        __value,
        name: typing.Union[str, typing.Iterable[str]]=None,
    ) -> None:
        self.value = __value
        """The value of this optional parameter."""
        self._name = aliased.MappingKey(name or '')
        self.display.register('value')
        self.display['__str__'] = "{value}"
        self.display['__repr__'] = "{value}"
        self.display['__repr__'].separator = ', '
        if self._name:
            self.display.register('name')
            self.display['__str__'].insert(0, "'{name}': ")
            self.display['__repr__'].insert(1, "'{name}'")

    def __eq__(self, other):
        """True if `other` is equivalent to this option's value."""
        if isinstance(other, Option):
            return other.value == self.value
        return other == self.value


