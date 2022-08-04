import collections.abc
import typing

from goats.core import algebraic
from goats.core import metadata
from goats.core import observed
from goats.core import observing


def expression(this):
    """True if `this` has the form of an `~algebraic.Expression`.
    
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


class Metadata(
    metadata.UnitMixin,
    metadata.NameMixin,
    metadata.AxesMixin,
): ...


class Quantity(Metadata):
    """A quantity that produces an observation."""

    def __init__(
        self,
        interface: observing.Interface,
        application: typing.Type[observing.Application],
        name: typing.Union[str, typing.Iterable[str], metadata.Name]=None,
    ) -> None:
        name = name or "Anonymous"
        self.interface = interface
        self._type = application
        self._name = name
        self._unit = interface.get_unit(name)
        self._axes = interface.get_axes(name)
        self._cache = None

    def observe(self, update: bool=False, **constraints) -> observed.Quantity:
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
        result = application.observe(self.name)
        indices = {k: application.indices[k] for k in result.axes}
        scalars = {k: application.scalars[k] for k in result.parameters}
        return observed.Quantity(result.quantity, {**indices, **scalars})

    def __str__(self) -> str:
        attrs = ('unit', 'name', 'axes')
        return ', '.join(f"{a}={getattr(self, a)}" for a in attrs)


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

    def __contains__(self, __o) -> bool:
        """True if `__o` names an observable quantity.
        
        Overloaded to avoid going through `__getitem__`.
        """
        return __o in self.names

    def __len__(self) -> int:
        return len(self.names)

    def __iter__(self) -> typing.Iterator:
        return iter(self.names)

    def __getitem__(self, __k: str) -> Quantity:
        """Get the named observable quantity."""
        return Quantity(self.available, self.application, name=__k)

