import collections.abc
import typing

from goats.core import algebraic
from goats.core import iterables
from goats.core import metadata
from goats.core import observed
from goats.core import observing
from goats.core import reference


def iscomposed(this):
    """True if `this` is an algebraic composition of observable quantities.
    
    Parameters
    ----------
    this
        The object to check.

    Notes
    -----
    This is more stringent than simply checking whether `this` can instantiate
    an `~algebraic.Expression` because all names of observable quantities would
    satisfy that condition.
    """
    return (
        isinstance(this, algebraic.Expression)
        or isinstance(this, str) and ('/' in this or '*' in this)
    )


class Quantity(iterables.ReprStrMixin):
    """A quantity that produces an observation."""

    def __init__(
        self,
        interface: observing.Interface,
        application: typing.Type[observing.Application],
        unit: metadata.UnitLike,
        axes: typing.Union[str, typing.Iterable[str], metadata.Axes],
        name: typing.Union[str, typing.Iterable[str], metadata.Name]=None,
    ) -> None:
        name = name or "Anonymous"
        self.interface = interface
        self._type = application
        self._unit = unit
        self._axes = axes
        self._name = name
        self._cache = None

    def __getitem__(self, __x: metadata.UnitLike):
        """Set the unit of this quantity."""
        unit = metadata.Unit(__x) if __x != self._unit else self._unit
        return type(self)(
            self.interface,
            self._type,
            unit=unit,
            axes=self.axes,
            name=self.name,
        )

    @property
    def unit(self):
        """This quantity's metric unit."""
        return metadata.Unit(self._unit)

    @property
    def name(self):
        """This quantity's name."""
        return metadata.Name(self._name)

    @property
    def axes(self):
        """This quantity's indexable axes."""
        return metadata.Axes(self._axes)

    def at(self, update: bool=False, **constraints) -> observed.Quantity:
        """Create an observation within the given constraints.
        
        This method will create a new observation of this observable quantity by
        applying `constraints`, if given, or the default constraints. The
        default collection of observational constraints uses all relevant axis
        indices and default parameter values.

        Parameters
        ----------
        update : bool, default=False
            If true, update the existing constraints from the given constraints.
            The default behavior is to use only the given constraints.

        **constraints
            Key-value pairs of axes or parameters to update.

        Returns
        -------
        `~observed.Quantity`
            An object representing the resultant observation.
        """
        if self._cache is None:
            self._cache = {}
        current = {**self._cache, **constraints} if update else constraints
        self._cache = current.copy()
        application = self._type(self.interface, **current)
        return application.observe(self.name)

    def __eq__(self, other):
        """True if two observables have the same name and constraints."""
        if isinstance(other, Quantity):
            return self.name == other.name and self._cache == other._cache
        return NotImplemented

    def __str__(self) -> str:
        attrs = {
            'unit': f"'{self.unit}'",
            'name': f"'{self.name}'",
            'axes': str(self.axes),
        }
        return ', '.join(f"{k}={v}" for k, v in attrs.items())


class Interface(collections.abc.Mapping):
    """ABC for interfaces to observable quantities."""

    def __init__(
        self,
        available: observing.Interface,
        application: typing.Type[observing.Application],
        *names: str,
    ) -> None:
        """Initialize this instance.
        
        Parameters
        ----------
        available : `~observing.Interface`
            The interface to all observable quantities available to this
            observer.

        application : type
            A concrete implementation of `~observing.Application` to use when
            creating observed quantities.

        *names : string
            Zero or more names of allowed observable quantities. Then default
            behavior (when `names` is empty) is to use all available quantities.
            This parameter allows observers to limit the allowed observable
            quantities to a subset of those in `available`.
        """
        self.available = available
        self.application = application
        self.names = names or list(available)
        """The names of all observable quantities."""

    def __len__(self) -> int:
        return len(self.names)

    def __iter__(self) -> typing.Iterator:
        return iter(self.names)

    def __getitem__(self, __k: str) -> Quantity:
        """Get the named observable quantity."""
        if not self.knows(__k):
            raise KeyError(f"Cannot observe {__k!r}") from None
        meta = self.available.get_metadata(__k)
        unit = meta.get('unit', '1'),
        axes = meta.get('axes', ())
        if iscomposed(__k):
            return Quantity(
                self.available,
                self.application,
                unit=unit,
                axes=axes,
                name=__k,
            )
        if __k in self.names:
            return Quantity(
                self.available,
                self.application,
                unit=unit,
                axes=axes,
                name=reference.ALIASES[__k],
            )
        # NOTE: If `self.knows` is consistent with the two previous `if` blocks,
        # execution should never reach this point. However, this exception is
        # here to avoid silently failing in case they are inconsistent.
        raise ValueError(
            f"Something went wrong while trying to observe {__k!r}"
        ) from None

    def knows(
        self,
        name: typing.Union[str, typing.Iterable[str], metadata.Name],
    ) -> bool:
        """True if this interface can observe the named quantity."""
        if isinstance(name, str):
            return self._knows(name)
        return next((self._knows(key) for key in name), False)

    def _knows(self, key: str):
        """Internal helper for `~Interface.knows`."""
        if key in self.names:
            return True
        if iscomposed(key):
            expression = algebraic.Expression(key)
            return all(term.base in self.names for term in expression)
        return False

