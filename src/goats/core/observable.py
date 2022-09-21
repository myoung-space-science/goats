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
        """
        self._name = metadata.Name(name)
        self._implementation = implementation
        self.application = application
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
        self.application.apply(**constraints)
        return self._implementation.apply(self.application)

    def __str__(self) -> str:
        display = [
            f"{str(self.name)!r}",
            f"unit={str(self.unit)!r}",
            f"axes={str(self.axes)}",
        ]
        return ', '.join(display)

