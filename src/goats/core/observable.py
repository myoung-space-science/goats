import typing

from goats.core import symbolic
from goats.core import iterables
from goats.core import metadata
from goats.core import metric
from goats.core import observing
from goats.core import observed
from goats.core import reference


def iscomposed(this):
    """True if `this` is an symbolic composition of observable quantities.
    
    Parameters
    ----------
    this
        The object to check.

    Notes
    -----
    This is more stringent than simply checking whether `this` can instantiate
    an `~symbolic.Expression` because all names of observable quantities would
    satisfy that condition.
    """
    return (
        isinstance(this, symbolic.Expression)
        or isinstance(this, str) and ('/' in this or '*' in this)
    )


class Interface:
    """An interface to arbitrary observable quantities."""

    def __init__(self, __quantities: observing.Interface) -> None:
        self._quantities = __quantities

    def get_unit(self, key: str):
        """Compute or retrieve the metric unit of a physical quantity."""
        if found := self._quantities.get_unit(key):
            return found
        s = str(key)
        expression = symbolic.Expression(reference.NAMES.get(s, s))
        term = expression[0]
        this = self.get_unit(term.base) ** term.exponent
        if len(expression) > 1:
            for term in expression[1:]:
                this *= self.get_unit(term.base) ** term.exponent
        return metadata.Unit(this)

    def get_dimensions(self, key: str):
        """Compute or retrieve the array dimensions of a physical quantity."""
        if found := self._quantities.get_dimensions(key):
            return found
        s = str(key)
        expression = symbolic.Expression(reference.NAMES.get(s, s))
        term = expression[0]
        this = self.get_dimensions(term.base)
        if len(expression) > 1:
            for term in expression[1:]:
                this *= self.get_dimensions(term.base)
        return metadata.Axes(this)

    def get_parameters(self, key: str):
        """Compute or retrieve the parameters of a physical quantity."""
        if found := self._quantities.get_parameters(key):
            return found
        s = str(key)
        expression = symbolic.Expression(reference.NAMES.get(s, s))
        term = expression[0]
        this = self.get_parameters(term.base)
        if len(expression) > 1:
            for term in expression[1:]:
                this *= self.get_parameters(term.base)
        return observing.Parameters(this)

    def implement(self, key: str, constraints: typing.Mapping):
        """Create an interface to observational data and context."""
        return observing.Implementation(
            self._quantities,
            key,
            constraints,
        )


class Quantity(iterables.ReprStrMixin):
    """A quantity that produces an observation."""

    def __init__(self, __i: observing.Implementation) -> None:
        """
        Initialize this instance.

        Parameters
        ----------
        __i : `~observing.Implementation`
            An implementation of the observer-defined observing interface.
        """
        self._i = __i
        self._name = None
        self._unit = None
        self._dimensions = None
        self._parameters = None

    def __getitem__(self, __x: metadata.UnitLike):
        """Create a quantity with the new unit."""
        unit = (
            self.unit.norm[__x]
            if str(__x).lower() in metric.SYSTEMS else __x
        )
        if unit == self._unit:
            return self
        return Quantity(self._i)

    @property
    def unit(self):
        """This quantity's current metric unit."""
        if self._unit is None:
            self._unit = self._i.unit
        return self._unit

    @property
    def dimensions(self):
        """This quantity's indexable dimensions."""
        if self._dimensions is None:
            self._dimensions = self._i.dimensions
        return self._dimensions

    @property
    def parameters(self):
        """The physical parameters relevant to this quantity."""
        if self._parameters is None:
            self._parameters = self._i.parameters
        return self._parameters

    def observe(self, **constraints):
        """Observe this observable quantity."""
        return self._i.apply(**constraints)

    @property
    def name(self):
        """"""
        return self._name

    _checkable = (
        'name',
        'unit',
        'dimensions',
        'parameters',
    )

    def __eq__(self, __o) -> bool:
        """True if two instances have equivalent attributes."""
        if isinstance(__o, Quantity):
            return all(
                getattr(self, attr) == getattr(__o, attr)
                for attr in self._checkable
            )
        return NotImplemented

    def __str__(self) -> str:
        display = [
            f"{str(self.name)!r}",
            f"unit={str(self.unit)!r}",
            f"dimensions={str(self.dimensions)}",
            f"parameters={str(self.parameters)}",
        ]
        return ', '.join(display)

