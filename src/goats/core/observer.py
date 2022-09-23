import typing

from goats.core import metadata
from goats.core import metric
from goats.core import observable
from goats.core import observing
from goats.core import reference
from goats.core import spelling
from goats.core import symbolic


class Interface:
    """The base class for all observers."""

    _data: observing.Interface=None

    def __init__(
        self,
        *unobservable: str,
        system: str='mks',
        apply: typing.Type[observing.Application]=None,
    ) -> None:
        """Initialize this instance.
        
        Parameters
        ----------
        *unobservable : string
            The names of variable quantities from this observer's dataset to
            exclude from the set of formally observable quantities. These
            quantities will still be accessible in their variable form.

        system : string, default='mks'
            The metric system to use for variable and observable quantities.

        application : type of observing application, optional
            A subclass of `~observing.Application` with which to evaluate
            operational parameters and variable quantities during the observing
            process. If `application` is absent, this class will use an instance
            of the base class.
        """
        self._unobservable = unobservable
        self._system = metric.System(system)
        self._application_type = apply or observing.Application
        self._source = None
        self._application = None
        self._spellcheck = None
        self._primary = None
        self._derived = None
        self._observables = None
        self._keys = None

    # TODO: I'm no longer sure it makes sense to allow the user to update an
    # observer's metric system rather than create a new observer for a new
    # metric system.
    def system(self, new: str=None):
        """Get or set this observer's metric system."""
        if not new:
            return self._system
        self._system = metric.System(new)
        return self

    def readfrom(self, source):
        """Update this observer's data source."""
        self._source = source
        self._application = None
        self._keys = None
        self._primary = None
        self._derived = None
        self._observables = None
        return self

    @property
    def source(self):
        """The source of this observer's data."""
        return self._source

    @property
    def application(self):
        """This observer's observing application."""
        if self._application is None:
            self._application = self._application_type(self.data)
        return self._application

    @property
    def assumptions(self):
        """The operational assumptions available to this observer."""
        return self.data.assumptions

    @property
    def observables(self):
        """The names of formally observable quantities.
        
        This property contains the names of all primary and derived observable
        quantities. See the `primary` and `derived` properties for their
        respective definitions. Names are listed a groups of aliases for each
        observable quantity (e.g., 'mfp | mean free path | mean_free_path'); any
        of the listed aliases is a valid key for that quantity.

        Note that it is also possible to symbolically compose new observable
        quantities from those listed here (e.g., 'mfp / Vr'). Therefore, this
        collection represents the minimal set of quantities that this observer
        can observe.
        """
        if self._observables is None:
            available = {str(name) for name in self.primary + self.derived}
            self._observables = tuple(available - set(self._unobservable))
        return self._observables

    @property
    def keys(self):
        """A flat list of known observables.
        
        This property primarily exists for convenience to internal search
        methods and may be removed in future versions. Please use the
        `observables`, `primary`, and `derived` properties whenever possible.
        """
        if self._keys is None:
            keys = {
                key
                for name in self.primary + self.derived
                for key in name
            }
            self._keys = tuple(keys - set(self._unobservable))
        return self._keys

    @property
    def primary(self):
        """The primary observable quantities.
        
        A primary observable quantity is a formally observable quantity that
        comes directly from this observer's data source.
        """
        if self._primary is None:
            self._primary = tuple(self.data.variables.keys(aliased=True))
        return self._primary

    @property
    def derived(self):
        """The derived observable quantities.
        
        A derived observable quantity is a formally observable quantity that is
        the result of a defined function of other (primary or derived)
        observable quantities, and possibly physical assumptions.
        """
        if self._derived is None:
            self._derived = tuple(self.data.functions.keys(aliased=True))
        return self._derived

    @property
    def data(self):
        """An interface to this observer's dataset quantities."""
        if self._data is None:
            raise NotImplementedError(
                f"Observer requires an observing interface"
            ) from None
        return self._data

    def __getitem__(self, key: str):
        """Access an observable quantity by keyword, if possible."""
        if q := self._get_quantity(key):
            return q
        self._check_spelling(key) # -> None if `key` is spelled correctly
        raise KeyError(f"No observable for {key!r}") from None

    def _get_quantity(self, key: str):
        """Retrieve the named quantity from an interface, if possible."""
        if self.knows(key):
            name = metadata.Name(
                key if observable.iscomposed(key)
                else reference.ALIASES[key]
            )
            return observable.Quantity(
                name,
                observing.Implementation(key, self.data),
                application=self.application,
            )
        if key in self.data.variables:
            return self.data.variables[key]
        if key in self.assumptions:
            return self.assumptions[key]

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
        if key in self.keys:
            return True
        if observable.iscomposed(key):
            expression = symbolic.Expression(key)
            return all(self._knows(term.base) for term in expression)
        return False

    def _check_spelling(self, key: str):
        """Catch misspelled names of observable quantities, if possible."""
        keys = list(self.keys) + list(self.assumptions)
        if self._spellcheck is None:
            self._spellcheck = spelling.SpellChecker(*keys)
        else:
            self._spellcheck.words |= keys
        return self._spellcheck.check(key)

