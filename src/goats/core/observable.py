import typing

from goats.core import symbolic
from goats.core import iterables
from goats.core import metadata
from goats.core import metric
from goats.core import observing


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


# NOTE: Overloaded by experimental version below.
class Quantity(iterables.ReprStrMixin):
    """A quantity that produces an observation."""

    def __init__(
        self,
        name: typing.Union[str, typing.Iterable[str], metadata.Name],
        context: observing.Context,
    ) -> None:
        self._name = metadata.Name(name)
        self._context = context
        self._unit = None
        self._axes = None
        self.constraints = None

    def __getitem__(self, __x: metadata.UnitLike):
        """Set the unit of this quantity."""
        unit = (
            self.unit.norm[__x]
            if str(__x).lower() in metric.SYSTEMS else __x
        )
        if unit != self._unit:
            self._unit = metadata.Unit(unit)
        return self

    @property
    def unit(self):
        """This quantity's metric unit."""
        if self._unit is None:
            self._unit = self.context.get_unit(self.name)
        return self._unit

    @property
    def axes(self):
        """This quantity's indexable axes."""
        if self._axes is None:
            self._axes = self.context.get_axes(self.name)
        return self._axes

    @property
    def name(self):
        """This quantity's name."""
        return self._name

    @property
    def context(self):
        """This quantity's observing context."""
        if self.constraints:
            self._context.apply(**self.constraints)
        return self._context

    def apply(self, **constraints):
        """Apply the given constraints to observations of this quantity."""
        if self.constraints is None:
            self.constraints = {}
        self.constraints.update(constraints)
        return self

    def reset(self):
        """Clear all user constraints and reset the unit."""
        self.constraints = {}
        self._unit = None
        return self

    def __eq__(self, other):
        """True if two observables have the same name and constraints."""
        if isinstance(other, Quantity):
            attrs = ('unit', 'name', 'axes')
            return all(getattr(self, a) == getattr(other, a) for a in attrs)
        return NotImplemented

    def __str__(self) -> str:
        attrs = {
            'unit': f"'{self.unit}'",
            'name': f"'{self.name}'",
            'axes': str(self.axes),
        }
        return ', '.join(f"{k}={v}" for k, v in attrs.items())


class Quantity(iterables.ReprStrMixin):
    """A quantity that produces an observation."""

    def __init__(
        self,
        name: typing.Union[str, typing.Iterable[str], metadata.Name],
        implementation: observing.Implementation,
        context: observing.Context,
    ) -> None:
        """
        Initialize this instance.

        Parameters
        ----------
        name : string, iterable of strings, or `~metadata.Name`
            The name(s) of this observable quantity.

        implementation : `~observing.Implementation`
            The object that will apply default parameter values and
            user-provided constraints to the target observable quantity.

        context : `~observing.Context`
            An existing observing context to provide default parameter values.
        """
        self._name = metadata.Name(name)
        self._implementation = implementation
        self.context = context
        self._unit = None
        self._axes = None

    def __getitem__(self, __x: metadata.UnitLike):
        """Set the unit of this quantity."""
        unit = (
            self.unit.norm[__x]
            if str(__x).lower() in metric.SYSTEMS else __x
        )
        if unit != self._unit:
            self._unit = metadata.Unit(unit)
        return self

    @property
    def unit(self):
        """This quantity's metric unit."""
        if self._unit is None:
            self._unit = self._implementation.get_unit(self.name)
        return self._unit

    @property
    def axes(self):
        """This quantity's indexable axes."""
        if self._axes is None:
            self._axes = self._implementation.get_axes(self.name)
        return self._axes

    @property
    def name(self):
        """This quantity's name."""
        return self._name

    def observe(self, **constraints):
        """Observe this observable quantity."""
        self.context.apply(**constraints)
        return self._implementation.apply(self.context)

