import typing
import contextlib

from goats.core import observing
from goats.core import spelling


class Interface:
    """The base class for all observers."""

    def __init__(
        self,
        *observables: typing.Mapping[str, observing.Quantity],
    ) -> None:
        """Initialize this instance.
        
        Parameters
        ----------
        *observables
            Zero or more mappings from string key to implementations of
            `~observing.Quantity`. All included quantities will be available to
            users via the standard bracket syntax (i.e., `observer[<key>]`).
        """
        self.observables = observables
        keys = [key for mapping in observables for key in mapping.keys()]
        self._spellcheck = spelling.SpellChecker(keys)

    def __getitem__(self, key: str):
        """Access an observable quantity by keyword, if possible."""
        for mapping in self.observables:
            with contextlib.suppress(KeyError):
                return mapping[key]
        if self._spellcheck.misspelled(key):
            raise spelling.SpellingError(key, self._spellcheck.suggestions)
        raise KeyError(f"No observable for {key!r}") from None

