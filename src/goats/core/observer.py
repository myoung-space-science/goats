import abc
import typing

from goats.core import computed
from goats.core import metadata
from goats.core import metric
from goats.core import observable
from goats.core import observed
from goats.core import observing
from goats.core import reference
from goats.core import spelling
from goats.core import symbolic
from goats.core import variable


class Interface(abc.ABC):
    """The base class for all observers."""

    _data: observing.Interface=None

    def __init__(
        self,
        *unobservable: str,
        system: str='mks',
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
        """
        self._unobservable = unobservable
        self._system = metric.System(system)
        self._source = None
        self._context = None
        self._interface = None
        self._spellcheck = None

    def system(self, new: str=None):
        """Get or set this observer's metric system."""
        if not new:
            return self._system
        self._system = metric.System(new)
        return self

    def readfrom(self, source):
        """Update this observer's data source."""
        self._source = source
        self._context = None
        self._interface = None
        return self

    @property
    def source(self):
        """The source of this observer's data."""
        return self._source

    @property
    def interface(self):
        """An interface to formally observable quantities."""
        if self._interface is None:
            self._interface = observable.Interface(
                self.data,
                *self.observables,
                context=self.context,
            )
        return self._interface

    @property
    def context(self):
        """This observer's observing context."""
        if self._context is None:
            self._context = observing.Context(self.data)
        return self._context

    @property
    def assumptions(self):
        """The operational assumptions available to this observer."""
        return self.data.assumptions

    @property
    def observables(self):
        """The names of formally observable quantities.
        
        This property contains the names of all primary and derived observable
        quantities. A primary derived observable quantity is one that comes
        directly from this observer's dataset; a derived observable quantity is
        one that is the result of a defined function.

        Note that it is also possible to symbolically compose new observable
        quantities from those listed here. Therefore, this collection represents
        the minimal set of quantities that this observer can observe.
        """
        fromdata = list(self.data.variables) + list(self.data.functions)
        available = set(fromdata) - set(self._unobservable)
        return tuple(available)

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
            return q[str(self.system())]
        self._check_spelling(key) # -> None if `key` is spelled correctly
        raise KeyError(f"No observable for {key!r}") from None

    def _get_quantity(self, key: str):
        """Retrieve the named quantity from an interface, if possible."""
        if key in self.interface:
            return self.interface[key]
        if key in self.data.variables:
            return self.data.variables[key]
        if key in self.assumptions:
            return self.assumptions[key]

    def _check_spelling(self, key: str):
        """Catch misspelled names of observable quantities, if possible."""
        keys = list(self.observables) + list(self.assumptions)
        if self._spellcheck is None:
            self._spellcheck = spelling.SpellChecker(*keys)
        else:
            self._spellcheck.words |= keys
        return self._spellcheck.check(key)

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
            self.interface[quantity] if isinstance(quantity, str)
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
        expression = symbolic.Expression(reference.NAMES.get(s, s))
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
        if isinstance(q, computed.Quantity):
            parameters = [p for p in q.parameters if p in self.data.constants]
            return observing.Quantity(self.compute(q), parameters=parameters)
        if isinstance(q, variable.Quantity):
            return observing.Quantity(self.process(q))
        raise ValueError(f"Unknown quantity: {q!r}") from None

    @abc.abstractmethod
    def process(self, name: str) -> variable.Quantity:
        """Compute observer-specific updates to a variable quantity."""
        raise NotImplementedError

    def compute(self, q: computed.Quantity):
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

