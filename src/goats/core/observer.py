import abc
import typing
import contextlib

from goats.core import algebraic
from goats.core import computable
from goats.core import index
from goats.core import metadata
from goats.core import observable
from goats.core import observed
from goats.core import observing
from goats.core import physical
from goats.core import reference
from goats.core import variable
from goats.core import spelling


class Interface:
    """The base class for all observers."""

    def __init__(
        self,
        data: observing.Interface,
        names: typing.Iterable[str],
        *others: typing.Mapping[str, typing.Any]
    ) -> None:
        """Initialize this instance.
        
        Parameters
        ----------
        data : `~observing.Interface`
            The interface to variable, axis, and constant quantities available
            to this observer.

        names : iterable of string
            The names of formally observable quantities.

        *others : mappings
            Zero or more mappings from string key to any object that users
            should be able to access from this observer. All included quantities
            will be available to users via the standard bracket syntax (i.e.,
            `observer[<key>]`).
        """
        self.data = data
        self.context = observing.Context(self.data)
        self.observables = observable.Interface(
            data,
            *names,
            context=self.context,
        )
        self.others = others
        keys = list(self.observables)
        keys.append([key for mapping in others for key in mapping.keys()])
        self._spellcheck = spelling.SpellChecker(keys)

    def __getitem__(self, key: str):
        """Access an observable quantity by keyword, if possible."""
        if key in self.observables:
            return self.observables[key]
        for mapping in self.others:
            if key in mapping:
                return mapping[key]
        if self._spellcheck.misspelled(key):
            raise spelling.SpellingError(key, self._spellcheck.suggestions)
        raise KeyError(f"No observable for {key!r}") from None

    def observe(
        self,
        quantity: typing.Union[str, metadata.Name, observable.Quantity],
        **constraints
    ) -> observed.Quantity:
        """Create an observation within the given constraints.
        
        This method will create a new observation of the target observable
        quantity after applying the given constraints. The default collection of
        observational constraints uses all relevant axis indices and default
        parameter values.

        Parameters
        ----------
        quantity : string, `~metadata.Name`, or `~observable.Quantity`
            The quantity to observe. If the argument is a string or instance of
            `~metadata.Name`, this method will first retrieve the corresponding
            `~observable.Quantity`. Users may also pass an existing instance of
            `~observable.Quantity` with updated unit or predefined constraints.

        **constraints
            Key-value pairs of axes or parameters to update. These constraints
            supercede constraints defined on `quantity` if it is an instance of
            `~observable.Quantity`.

        Returns
        -------
        `~observed.Quantity`
            An object representing the resultant observation.
        """
        target = (
            self.observables[quantity] if isinstance(quantity, str)
            else quantity
        )
        self.context.apply(**constraints)
        result = self._observe(target.name)
        indices = {k: self.context.get_index(k) for k in result.axes}
        scalars = {k: self.context.get_value(k) for k in result.parameters}
        return observed.Quantity(result[target.unit], indices, **scalars)

    def _observe(self, name: metadata.Name) -> observing.Quantity:
        """Internal helper for `~Interface.observe`."""
        s = list(name)[0]
        expression = algebraic.Expression(reference.NAMES.get(s, s))
        term = expression[0]
        result = self.get_observable(term.base)
        if len(expression) == 1:
            # We don't need to multiply or divide quantities.
            if term.exponent == 1:
                # We don't even need to raise this quantity to a power.
                return result
            return result ** term.exponent
        q0 = result ** term.exponent
        if len(expression) > 1:
            for term in expression[1:]:
                result = self.get_observable(term.base)
                q0 *= result ** term.exponent
        return q0

    def evaluate(self, q) -> observing.Quantity:
        """Create an observing result based on this quantity."""
        if isinstance(q, computable.Quantity):
            parameters = [p for p in q.parameters if p in self.data.constants]
            return observing.Quantity(self.compute(q), parameters=parameters)
        if isinstance(q, variable.Quantity):
            return observing.Quantity(self.process(q))
        raise ValueError(f"Unknown quantity: {q!r}") from None

    @abc.abstractmethod
    def process(self, name: str) -> variable.Quantity:
        """Compute observer-specific updates to a variable quantity."""
        raise NotImplementedError

    def compute(self, q: computable.Quantity):
        """Determine dependencies and compute the result of this function."""
        dependencies = {p: self.get_dependency(p) for p in q.parameters}
        return q(**dependencies)

    def get_dependency(self, key: str):
        """Get the named constant or variable quantity."""
        if this := self.get_observable(key):
            return this
        return self.context.get_value(key)

    def get_observable(self, key: str):
        """Retrieve and evaluate an observable quantity."""
        if quantity := self.data.get_observable(key):
            return self.evaluate(quantity)

