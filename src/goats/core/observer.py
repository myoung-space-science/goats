import typing

from goats.core import aliased
from goats.core import metadata
from goats.core import metric
from goats.core import observing
from goats.core import spelling
from goats.core import symbolic


C = typing.TypeVar('C', bound=observing.Context)


class Interface:
    """The base class for all observers."""

    def __init__(
        self,
        *unobservable: str,
        context: C=None,
    ) -> None:
        """Initialize this instance.
        
        Parameters
        ----------
        *unobservable : string
            The names of variable quantities from this observer's dataset to
            exclude from the set of formally observable quantities. These
            quantities will still be accessible as variable quantities.

        context : subclass of `~observing.Context`, optional
            An instance of the observer-specific observing context. Users may
            set the observing context after initialization via the `update`
            method. This class will raise an exception if the user attempts to
            access physical quantities without a valid observing context.
        """
        self._unobservable = unobservable
        self._context = context
        self._spellcheck = None

    def update(self, __context: C):
        """Use a new observing context."""
        self._context = __context
        self._spellcheck = None
        return self

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
        available = aliased.Groups(*self.context.observable)
        return available.without(*self._unobservable)

    def __getitem__(self, __k: str):
        """Access an observable quantity by keyword, if possible."""
        if self.observes(__k):
            return observing.Observable(__k, self.context)
        if __k in self.context:
            return self.context.get_quantity(__k)
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
        if key in self.observables:
            return True
        if symbolic.composition(key):
            expression = symbolic.Expression(key)
            return all(self._observes(term.base) for term in expression)
        return False

    def _check_spelling(self, key: str):
        """Catch misspelled names of physical quantities, if possible."""
        keys = self.context.available
        if self._spellcheck is None:
            self._spellcheck = spelling.SpellChecker(*keys)
        else:
            self._spellcheck.words |= keys
        return self._spellcheck.check(key)

    @property
    def context(self):
        """The current observing context."""
        if self._context is None:
            raise NotImplementedError(
                "This observer does not have an observing context."
            ) from None
        return self._context

