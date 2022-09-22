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


class Quantity(iterables.ReprStrMixin):
    """A quantity that produces an observation."""

    def __init__(
        self,
        name: typing.Union[str, typing.Iterable[str], metadata.Name],
        implementation: observing.Implementation,
        application: observing.Application,
        unit: metadata.UnitLike=None,
    ) -> None:
        """
        Initialize this instance.

        Parameters
        ----------
        name : string, iterable of strings, or `~metadata.Name`
            The name(s) of this observable quantity.

        implementation : `~observing.Implementation`
            The object that represents the observing interface to the target
            observable quantity.

        application : `~observing.Application`
            An existing observing application that will compute the observed
            quantity, manage user constraints, and provide default parameter
            values.

        unit : unit-like, optional
            The metric unit to which to convert observations of this quantity.
        """
        self._name = metadata.Name(name)
        self._implementation = implementation
        self.application = application
        self._unit = metadata.Unit(unit or self._implementation.get_unit(name))
        self._axes = None

    def __getitem__(self, __x: metadata.UnitLike):
        """Create a quantity with the new unit."""
        unit = (
            self.unit.norm[__x]
            if str(__x).lower() in metric.SYSTEMS else __x
        )
        if unit == self._unit:
            return self
        return Quantity(
            self.name,
            self._implementation,
            self.application,
            unit=metadata.Unit(unit),
        )

    @property
    def unit(self):
        """This quantity's current metric unit."""
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
        self.application.apply(**constraints)
        result = self._implementation.apply(self.application, unit=self.unit)
        self.application.reset()
        return result

    def __eq__(self, __o) -> bool:
        """True if two instances have equivalent attributes."""
        if isinstance(__o, Quantity):
            return all(
                getattr(self, attr) == getattr(__o, attr)
                for attr in ('name', 'unit', 'axes')
            )
        return NotImplemented

    def __str__(self) -> str:
        display = [
            f"{str(self.name)!r}",
            f"unit={str(self.unit)!r}",
            f"axes={str(self.axes)}",
        ]
        return ', '.join(display)

