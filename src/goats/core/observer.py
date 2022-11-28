import typing

from goats.core import aliased
from goats.core import metadata
from goats.core import metric
from goats.core import observing
from goats.core import spelling
from goats.core import symbolic


A = typing.TypeVar('A', bound=observing.Application)


class Interface:
    """The base class for all observers."""

    def __init__(
        self,
        *unobservable: str,
        system: str='mks',
        application: A=None,
    ) -> None:
        """Initialize this instance.
        
        Parameters
        ----------
        *unobservable : string
            The names of variable quantities from this observer's dataset to
            exclude from the set of formally observable quantities. These
            quantities will still be accessible as variable quantities.

        system : string, default='mks'
            The metric system to use for variable and observable quantities.

        application : subclass of `~observing.Application`, optional
            An instance of the observer-specific observing application. Users
            may set the observing application after initialization via the
            `update` method. This class will raise an exception if the user
            attempts to access physical quantities without a valid observing
            application.
        """
        self._unobservable = unobservable
        self._system = metric.System(system)
        self._application = application
        self._quantities = None
        self._spellcheck = None

    def update(self, __application: A):
        """Use a new observing application."""
        self._application = __application
        self._spellcheck = None
        return self

    @property
    def system(self):
        """This observer's metric system."""
        return self._system

    @property
    def observables(self):
        """The names of formally observable quantities.
        
        This property contains the names of physical quantities from which this
        observer can create an observation. Names are listed as groups of
        aliases for each observable quantity (e.g., 'mfp | mean free path |
        mean_free_path'); any of the listed aliases is a valid key for that
        quantity.

        Note that it is also possible to symbolically compose new observable
        quantities from those listed here (e.g., 'mfp / Vr'). Therefore, this
        collection represents the minimal set of quantities that this observer
        can observe.
        """
        available = aliased.KeyMap(*self.quantities.available)
        return available.without(*self._unobservable)

    @property
    def quantities(self):
        """The interface to available physical quantities.
        
        This collection represents the quantities that this observer will use
        when making observations.
        """
        if self._application is None:
            raise NotImplementedError(
                "This observer does not have an observing application."
            ) from None
        try:
            self._quantities = self._application.quantities
            return self._quantities
        except AttributeError as err:
            raise TypeError(
                "Can't access physical quantities from observing application"
                f" of type {type(self._application)!r}"
            ) from err

    def __getitem__(self, __k: str):
        """Access an observable quantity by keyword, if possible."""
        if self.observes(__k):
            return observing.Implementation(self._application, __k)
        if __k in self.quantities:
            return self.quantities[__k]
        self._check_spelling(__k) # -> None if `__k` is spelled correctly
        raise KeyError(f"No observable for {__k!r}") from None

    def observes(
        self,
        name: typing.Union[str, typing.Iterable[str], metadata.Name],
    ) -> bool:
        """True if this interface can observe the named quantity."""
        if isinstance(name, str):
            return self._observes(name)
        return next((self._observes(key) for key in name), False)

    def _observes(self, key: str):
        """Internal helper for `~Interface.observes`."""
        if key in list(self.observables):
            return True
        if symbolic.composition(key):
            expression = symbolic.Expression(key)
            return all(self._observes(term.base) for term in expression)
        return False

    def _check_spelling(self, key: str):
        """Catch misspelled names of physical quantities, if possible."""
        keys = list(self.quantities)
        if self._spellcheck is None:
            self._spellcheck = spelling.SpellChecker(*keys)
        else:
            self._spellcheck.words |= keys
        return self._spellcheck.check(key)

