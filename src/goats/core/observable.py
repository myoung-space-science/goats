import collections.abc
import typing

from goats.core import algebraic
from goats.core import iterables
from goats.core import metadata
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
        name: typing.Union[str, typing.Iterable[str], metadata.Name],
        context: observing.Context,
    ) -> None:
        self._name = name
        self._context = context
        self._unit = None
        self._axes = None
        self.constraints = None

    def __getitem__(self, __x: metadata.UnitLike):
        """Set the unit of this quantity."""
        unit = metadata.Unit(__x) if __x != self._unit else self._unit
        return type(self)(
            unit=unit,
            axes=self.axes,
            name=self.name,
        )

    @property
    def unit(self):
        """This quantity's metric unit."""
        return self.context.get_unit(self.name)

    @property
    def axes(self):
        """This quantity's indexable axes."""
        return self.context.get_axes(self.name)

    @property
    def name(self):
        """This quantity's name."""
        return metadata.Name(self._name)

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
        """Clear all user constraints."""
        self.constraints = {}
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


class Interface(collections.abc.Mapping):
    """ABC for interfaces to observable quantities."""

    def __init__(
        self,
        available: observing.Interface,
        *names: str,
        context: observing.Context=None,
    ) -> None:
        """Initialize this instance.
        
        Parameters
        ----------
        available : `~observing.Interface`
            The interface to all observable quantities available to this
            observer.

        *names : string
            Zero or more names of allowed observable quantities. The default
            behavior (when `names` is empty) is to use all available quantities.
            This parameter allows observers to limit the allowed observable
            quantities to a subset of those in `available`.

        context : `~observing.Context`, optional
            An existing observing context with which to initialize observable
            quantities.
        """
        self.available = available
        self._names = names
        self._context = context

    @property
    def names(self):
        """The names of all observable quantities."""
        return self._names or tuple(self.available)

    @property
    def context(self):
        """The observing context to pass to observable quantities."""
        if self._context is None:
            self._context = observing.Context(self.available)
        return self._context

    def __len__(self) -> int:
        return len(self.names)

    def __iter__(self) -> typing.Iterator:
        return iter(self.names)

    def __getitem__(self, __k: str) -> Quantity:
        """Get the named observable quantity."""
        if not self.knows(__k):
            raise KeyError(f"Cannot observe {__k!r}") from None
        name = __k if iscomposed(__k) else reference.ALIASES[__k]
        return Quantity(name, self.context)

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

