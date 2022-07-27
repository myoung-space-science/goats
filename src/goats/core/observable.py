import abc
import collections.abc
import typing

from goats.core import algebraic
from goats.core import aliased
from goats.core import axis
from goats.core import functions
from goats.core import index
from goats.core import iterables
from goats.core import measurable
from goats.core import metadata
from goats.core import observables
from goats.core import observed
from goats.core import parameter
from goats.core import physical
from goats.core import variable


def expression(this):
    """True if `this` has the form of an `~algebraic.Expression`.
    
    Parameters
    ----------
    this
        The object to check.

    Notes
    -----
    This is more stringent than simply checking whether `this` can instantiate
    an `~algebraic.Expression` because all named of observable quantities would
    satisfy that condition.
    """
    return (
        isinstance(this, algebraic.Expression)
        or isinstance(this, str) and ('/' in this or '*' in this)
    )


class Application(abc.ABC):
    """ABC for observing applications.
    
    Concrete subclasses must define a method called `evaluate_variable` that
    takes a string name and returns a `~variable.Quantity`.
    """

    def __init__(
        self,
        dataset: Dataset,
        **constraints
    ) -> None:
        self.dataset = dataset
        self.indices = self.dataset.get_indices(constraints)
        self.scalars = self.dataset.get_scalars(constraints)

    def get_quantity(self, name: str):
        """Retrieve the named quantity from available attributes."""
        if name in self.scalars:
            return self.scalars[name]
        return self.evaluate_variable(name)

    @abc.abstractmethod
    def evaluate_variable(self, name: str) -> variable.Quantity:
        """Retrieve and update a variable quantity from the dataset."""
        raise NotImplementedError

    def evaluate_function(self, name: str) -> variable.Quantity:
        """Create a variable quantity from a function."""
        interface = functions.REGISTRY[name]
        method = interface.pop('method')
        caller = variable.Caller(method, **interface)
        deps = {p: self.get_quantity(p) for p in caller.parameters}
        quantity = observables.METADATA.get(name, {}).get('quantity', None)
        data = caller(**deps)
        return variable.Quantity(
            data,
            axes=self.dataset.get_axes(name),
            unit=self.dataset.get_unit(quantity=quantity),
            name=name,
        )

    def evaluate_expression(self, name: str) -> variable.Quantity:
        """Combine variables and functions based on this expression."""
        expression = algebraic.Expression(name)
        variables = [self.get_quantity(term.base) for term in expression]
        exponents = [term.exponent for term in expression]
        result = variables[0] ** exponents[0]
        for variable, exponent in zip(variables[1:], exponents[1:]):
            result *= variable ** exponent
        return result


class Factory(abc.ABC):
    """ABC for classes that create observed variable quantities."""

    def __init__(self, dataset: Dataset, **constraints) -> None:
        self.application = Application(dataset, **constraints)

    def observe(self, name: str) -> observed.Quantity:
        """Create the named observable quantity."""
        return observed.Quantity(
            self.create(name),
            self.application.indices,
            self.application.scalars,
        )

    @abc.abstractmethod
    def create(self, name: str) -> variable.Quantity:
        """Create a variable quantity for the named observable quantity."""
        raise NotImplementedError


class Variables(Factory):
    """Get an observable quantity from an observer's dataset."""

    def create(self, name: str) -> variable.Quantity:
        return self.application.evaluate_variable(name)


class Functions(Factory):
    """Compute the result of a function of observable quantities."""

    def create(self, name: str) -> variable.Quantity:
        return self.application.evaluate_function(name)


class Expressions(Factory):
    """Evaluate an algebraic expression of other observable quantities."""

    def create(self, name: str) -> variable.Quantity:
        return self.application.evaluate_expression(name)


class Context(abc.ABC):
    """ABC for observing contexts."""

    def __init__(
        self,
        dataset: Dataset,
        factory: typing.Type[Factory],
    ) -> None:
        self.dataset = dataset
        self.factory = factory

    @abc.abstractmethod
    def get_unit(self, name: str) -> metadata.Unit:
        """Get the appropriate unit for this observable quantity."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_axes(self, name: str) -> metadata.Axes:
        """Get the appropriate axes for this observable quantity."""
        raise NotImplementedError

    def apply(self, **user):
        """Apply user constraints to this observing context."""
        return self.factory(self.dataset, **user)


class Primary(Context):
    """The context of a primary observable quantity."""

    def __init__(self, dataset: Dataset) -> None:
        super().__init__(dataset, Variables)

    def get_unit(self, name: str) -> metadata.Unit:
        return self.dataset.variables[name].unit

    def get_axes(self, name: str) -> metadata.Axes:
        return self.dataset.variables[name].axes


class Derived(Context):
    """The context of a derived observable quantity."""

    def __init__(self, dataset: Dataset) -> None:
        super().__init__(dataset, Functions)

    def get_unit(self, name: str) -> metadata.Unit:
        return self.dataset.get_unit(name)

    def get_axes(self, name: str) -> metadata.Axes:
        return self.dataset.get_axes(name)


class Composed(Context):
    """The context of a composed observable quantity."""

    def __init__(self, dataset: Dataset) -> None:
        super().__init__(dataset, Expressions)

    def get_unit(self, name: str) -> metadata.Unit:
        return algebraic.Expression(name).apply(self.dataset.get_unit)

    def get_axes(self, name: str) -> metadata.Axes:
        return algebraic.Expression(name).apply(self.dataset.get_axes)


T = typing.TypeVar('T', bound=Context)


class Implementation(typing.Generic[T]):
    """An implementation of an observable quantity."""

    def __init__(
        self,
        context: T,
        name: typing.Union[str, typing.Iterable[str], metadata.Name]=None,
    ) -> None:
        self.context = context
        self.name = metadata.Name(name)
        self.unit = context.get_unit(name)
        self.axes = context.get_axes(name)


class Metadata(
    metadata.UnitMixin,
    metadata.NameMixin,
    metadata.AxesMixin,
): ...

class Quantity(Metadata, typing.Generic[T]):
    """A quantity that produces an observation."""

    def __init__(
        self,
        context: T,
        name: typing.Union[str, typing.Iterable[str], metadata.Name]=None,
    ) -> None:
        self.context = context
        self._name = metadata.Name(name)
        self._unit = context.get_unit(name)
        self._axes = context.get_axes(name)
        self._constraints = None

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
        if update:
            self._constraints.update(constraints)
        else:
            self._constraints = constraints or {}
        return self.context.apply(**self._constraints).observe(self._name)

    def __str__(self) -> str:
        attrs = ('unit', 'name', 'axes')
        return ', '.join(f"{a}={getattr(self, a)}" for a in attrs)


class Interface(collections.abc.Mapping):
    """ABC for interfaces to observable quantities."""

    def __init__(self, **implemented: Implementation) -> None:
        self.implemented = implemented

    def __len__(self) -> int:
        return len(self.implemented)

    def __iter__(self) -> typing.Iterator:
        return iter(self.implemented)

    def __getitem__(self, __k: str) -> Quantity:
        """Get the named observable quantity."""
        if __k in self.implemented:
            return self.implemented[__k]
        if '/' in __k or '*' in __k:
            expression = algebraic.Expression(__k)
        raise NotImplementedError


