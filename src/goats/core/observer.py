import typing

from goats.core import aliased
from goats.core import metadata
from goats.core import metric
from goats.core import observable
from goats.core import observing
from goats.core import reference
from goats.core import spelling
from goats.core import symbolic


class Interface:
    """The base class for all observers."""

    def __init__(
        self,
        __data: observing.Dataset,
        *unobservable: str,
        system: str='mks',
        apply: typing.Type[observing.Application]=None,
    ) -> None:
        """Initialize this instance.
        
        Parameters
        ----------
        __data : `~observing.Dataset`
            The observer's dataset.

        *unobservable : string
            The names of variable quantities from this observer's dataset to
            exclude from the set of formally observable quantities. These
            quantities will still be accessible in their variable form.

        system : string, default='mks'
            The metric system to use for variable and observable quantities.

        apply : type of observing application, optional
            A subclass of `~observing.Application` with which to evaluate
            operational parameters and variable quantities during the observing
            process. If `application` is absent, this class will use an instance
            of the base class.
        """
        self._data = __data
        self._unobservable = unobservable
        self._system = metric.System(system)
        self._application_type = apply or observing.Application
        self._quantities = None
        self._application = None
        self._spellcheck = None

    def reset(self, source=None, **kwargs):
        """Reset the dataset interface.
        
        The base implementation resets various data-related attributes to their
        uninitialized values and sets `source`, if given, as the new target from
        which to read data. Concrete subclasses may wish to implement additional
        data-related logic (e.g., reinitialize interfaces), possibly after
        modifying `source` (e.g., to normalize a path).

        Parameters
        ----------
        source, optional
            The new source of this observer's data. The acceptable type(s) will
            be observer-specific.
        """
        self._quantities = None
        self._application = None
        self._spellcheck = None
        if source is not None:
            self._data.readfrom(source, **kwargs)
        return self

    @property
    def system(self):
        """This observer's metric system."""
        return self._system

    @property
    def application(self):
        """This observer's observing application."""
        if self._application is None:
            self._application = self._application_type(self.quantities)
        return self._application

    @property
    def parameters(self):
        """The names of operational arguments relevant to this observer."""
        keys = self.quantities.constants.keys(aliased=True)
        return aliased.KeyMap(*keys)

    @property
    def observables(self):
        """The names of formally observable quantities.
        
        This property contains the names of all primary and derived observable
        quantities. See `~observing.Interface.primary` and
        `~observing.Interface.derived` for their respective definitions. Names
        are listed a groups of aliases for each observable quantity (e.g., 'mfp
        | mean free path | mean_free_path'); any of the listed aliases is a
        valid key for that quantity.

        Note that it is also possible to symbolically compose new observable
        quantities from those listed here (e.g., 'mfp / Vr'). Therefore, this
        collection represents the minimal set of quantities that this observer
        can observe.
        """
        primary = self.quantities.primary
        derived = self.quantities.derived
        available = aliased.KeyMap(*primary, *derived)
        return available.without(*self._unobservable)

    @property
    def quantities(self):
        """An interface to this observer's physical quantities.
        
        This property represents the variable, axis-indexing, and constant
        quantities that are available from this observer's dataset. It
        incorporates this observer's metric system, and exposes objects that
        support arithmetic operations and conversion to numpy arrays.
        """
        if self._quantities is None:
            self._quantities = observing.Quantities(
                self._data,
                system=self.system,
            )
        return self._quantities

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
                observing.Implementation(key, self.quantities),
                application=self.application,
            )
        if key in self.quantities.variables:
            return self.quantities.variables[key]
        if key in self.quantities.constants:
            return self.quantities.constants[key]

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
        if key in list(self.observables):
            return True
        if observable.iscomposed(key):
            expression = symbolic.Expression(key)
            return all(self._knows(term.base) for term in expression)
        return False

    def _check_spelling(self, key: str):
        """Catch misspelled names of observable quantities, if possible."""
        keys = list(self.observables) + list(self.assumptions)
        if self._spellcheck is None:
            self._spellcheck = spelling.SpellChecker(*keys)
        else:
            self._spellcheck.words |= keys
        return self._spellcheck.check(key)

