import typing

from goats.core import aliased
from goats.core import metadata
from goats.core import metric
from goats.core import observable
from goats.core import observing
from goats.core import spelling
from goats.core import symbolic


class Interface:
    """The base class for all observers."""

    def __init__(
        self,
        __type: observing.Interface,
        *unobservable: str,
        system: str='mks',
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
        """
        self._type = __type
        self._unobservable = unobservable
        self._system = metric.System(system)
        self._quantities = None
        self._spellcheck = None

    def update(self, __quantities: observing.Quantities):
        """Use a new interface to physical quantities."""
        self._quantities = __quantities
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
        these = aliased.KeyMap(self.quantities.observable)
        return these.without(*self._unobservable)

    @property
    def quantities(self):
        """An interface to this observer's physical quantities.
        
        This property represents the quantities that this observer will use when
        making observations.
        """
        return self._quantities

    def __getitem__(self, __k: str):
        """Access an observable quantity by keyword, if possible."""
        if self.observes(__k):
            implementation = observing.Implementation(
                self._type, __k, self.quantities
            )
            return observable.Quantity(implementation)
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

