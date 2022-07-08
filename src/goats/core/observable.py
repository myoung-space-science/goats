import math
import operator as standard
import typing

from goats.core import algebraic
from goats.core import datatypes
from goats.core import iterables
from goats.core import measurable
from goats.core import metadata
from goats.core import metric


# This could become `measurable.Quantity`, which would make the name more
# consistent with `algebraic.Quantity`.
class Quantified(algebraic.Quantity, iterables.ReprStrMixin):
    """An object with data and metdata."""

    def __init__(self, __data: measurable.Real, **meta) -> None:
        self._data = __data
        for k, v in meta.items():
            setattr(self, f'_{k}', v)
        self.attrs = list(meta)
        factory = metadata.OperatorFactory(type(self), *meta)
        factory['true divide'].suppress(measurable.Real, Quantified)
        factory['power'].suppress(measurable.Real, Quantified)
        factory['power'].suppress(Quantified, typing.Iterable, symmetric=True)
        self.meta = factory

    def __str__(self) -> str:
        mapped = {k: getattr(self, k, None) for k in self.attrs}
        attrs = [
            f'{self.data}'
        ] + [
            f"{k}={v}" if isinstance(v, (list, tuple)) else f"{k}='{v}'"
            for k, v in mapped.items()
        ]
        return ', '.join(attrs)

    @property
    def data(self):
        """This quantity's data."""
        return self._data

    def __bool__(self) -> bool:
        """Always true for a valid instance."""
        return True

    def __int__(self):
        """Called for int(self)."""
        return self._call(int, 'cast')

    def __float__(self):
        """Called for float(self)."""
        return self._call(float, 'cast')

    def __abs__(self):
        """Called for abs(self)."""
        return self._call(abs, 'arithmetic')

    def __round__(self):
        """Called for round(self)."""
        return self._call(round, 'arithmetic')

    def __pos__(self):
        """Called for +self."""
        return self._call(standard.pos, 'arithmetic')

    def __neg__(self):
        """Called for -self."""
        return self._call(standard.neg, 'arithmetic')

    def __floor__(self):
        """Called for math.floor(self)."""
        return self._call(math.floor, 'arithmetic')

    def __ceil__(self):
        """Called for math.ceil(self)."""
        return self._call(math.ceil, 'arithmetic')

    def __trunc__(self):
        """Called for math.trunc(self)."""
        return self._call(math.trunc, 'arithmetic')

    def __lt__(self, other) -> bool:
        """Called for self < other."""
        return self._call(standard.lt, 'comparison', other)

    def __le__(self, other) -> bool:
        """Called for self <= other."""
        return self._call(standard.le, 'comparison', other)

    def __gt__(self, other) -> bool:
        """Called for self > other."""
        return self._call(standard.gt, 'comparison', other)

    def __ge__(self, other) -> bool:
        """Called for self >= other."""
        return self._call(standard.ge, 'comparison', other)

    def __eq__(self, other) -> bool:
        """Called for self == other."""
        if not isinstance(other, Quantified):
            return other == self.data
        if other.data != self.data:
            return False
        for name in self.meta:
            v = getattr(self, name)
            if hasattr(other, name) and getattr(other, name) != v:
                return False
        return True

    def __ne__(self, other) -> bool:
        """Called for self != other."""
        return not self == other

    def __add__(self, other):
        """Called for self + other."""
        return self._call(standard.add, 'forward', other)

    def __radd__(self, other):
        """Called for other + self."""
        return self._call(standard.add, 'reverse', other)

    def __iadd__(self, other):
        """Called for self += other."""
        return self._call(standard.add, 'inplace', other)

    def __sub__(self, other):
        """Called for self - other."""
        return self._call(standard.sub, 'forward', other)

    def __rsub__(self, other):
        """Called for other - self."""
        return self._call(standard.sub, 'reverse', other)

    def __isub__(self, other):
        """Called for self -= other."""
        return self._call(standard.sub, 'inplace', other)

    def __mul__(self, other):
        """Called for self * other."""
        return self._call(standard.mul, 'forward', other)

    def __rmul__(self, other):
        """Called for other * self."""
        return self._call(standard.mul, 'reverse', other)

    def __imul__(self, other):
        """Called for self *= other."""
        return self._call(standard.mul, 'inplace', other)

    def __truediv__(self, other):
        """Called for self / other."""
        return self._call(standard.truediv, 'forward', other)

    def __rtruediv__(self, other):
        """Called for other / self."""
        return self._call(standard.truediv, 'reverse', other)

    def __itruediv__(self, other):
        """Called for self /= other."""
        return self._call(standard.truediv, 'inplace', other)

    def __pow__(self, other):
        """Called for self ** other."""
        return self._call(standard.pow, 'forward', other)

    def __rpow__(self, other):
        """Called for other ** self."""
        return self._call(standard.pow, 'reverse', other)

    def __ipow__(self, other):
        """Called for self **= other."""
        return self._call(standard.pow, 'inplace', other)

    # Make this abstract and force concrete subclass to define metadata? If so,
    # need to also move `__eq__`.
    def _call(self, func: typing.Callable, mode: str, *others, **kwargs):
        """Implement a standard operation."""
        name = func.__name__
        if mode == 'cast':
            return func(self.data)
        if mode == 'arithmetic':
            data = func(self.data, **kwargs)
            meta = self.meta[name].evaluate(self, **kwargs)
            return type(self)(data, **meta)
        if mode == 'comparison':
            self.meta.check(self, *others)
            operands = [self] + list(*others)
            args = [
                i.data if isinstance(i, Quantified) else i
                for i in operands
            ]
            return func(*args)
        if mode == 'forward':
            operands = [self] + list(others)
            args = [
                i.data if isinstance(i, Quantified) else i
                for i in operands
            ]
            data = func(*args, **kwargs)
            meta = self.meta[name].evaluate(*operands, **kwargs)
            return type(self)(data, **meta)
        if mode == 'reverse':
            operands = others[::-1] + [self]
            args = [
                i.data if isinstance(i, Quantified) else i
                for i in operands
            ]
            data = func(*args, **kwargs)
            meta = self.meta[name].evaluate(*operands, **kwargs)
            return type(self)(data, **meta)
        if mode == 'inplace':
            operands = [self] + list(others)
            args = [
                i.data if isinstance(i, Quantified) else i
                for i in operands
            ]
            data = func(*args, **kwargs)
            meta = self.meta[name].evaluate(*operands, **kwargs)
            self._data = data
            for k, v in meta:
                setattr(self, k, v)
            return self
        raise ValueError(f"Unknown operator mode {mode!r}")


class Measurable:
    """Mixin class for quantities with a unit."""

    _data: measurable.Real=None
    _unit: metric.UnitLike=None

    @property
    def unit(self):
        """This quantity's metric unit."""
        return self._unit

    def convert(self, unit: metric.UnitLike):
        """Set the unit of this object's values."""
        if unit == self._unit:
            return self
        new = metric.Unit(unit)
        self._data *= new // self._unit
        self._unit = new
        return self


class Identifiable:
    """An object with a name."""

    _name: datatypes.Name=None

    @property
    def name(self):
        """This quantity's name."""
        return self._name

    def alias(self, *updates: str, reset: bool=False):
        """Set or add to this object's name(s)."""
        aliases = updates if reset else self._name.add(updates)
        self._name = datatypes.Name(*aliases)
        return self


class Distinguishable(Measurable, Identifiable):
    """A measurable and identifiable object."""


class Locatable:
    """An object with axes."""

    _axes: datatypes.Axes=None

    @property
    def axes(self):
        """This quantity's indexable axes."""
        return self._axes


class Quantity(Quantified, Distinguishable, Locatable):
    """An observable quantity."""

    def __init__(
        self,
        __data: measurable.Real,
        unit: metric.UnitLike=None,
        name: typing.Union[str, typing.Iterable[str], datatypes.Name]=None,
        axes: typing.Union[str, typing.Iterable[str], datatypes.Axes]=None,
    ) -> None:
        meta = {
            'unit': metric.Unit(unit or '1'),
            'name': datatypes.Name(name or ''),
            'axes': datatypes.Axes(axes or ()),
        }
        super().__init__(__data, **meta)


# q = Quantity([1.1], unit='m', name='Q', axes='x')
# q.convert('cm')
# q.alias('q0')
# q.axes

