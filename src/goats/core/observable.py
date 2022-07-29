import abc
import collections.abc
import typing

from goats.core import algebraic
from goats.core import aliased
from goats.core import constant
from goats.core import metadata
from goats.core import observed
from goats.core import observer
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
        quantities: observer.Quantities,
        **constraints
    ) -> None:
        self.observer = quantities
        self.user = constraints
        self._scalars = None

    @property
    def scalars(self):
        """This observer's single-valued assumptions."""
        if self._scalars is None:
            assumptions = {
                k: v for k, v in self.observer.constants
                if isinstance(v, constant.Assumption)
            } if self.observer.constants else {}
            defaults = aliased.MutableMapping(assumptions)
            updates = {
                k: self._get_scalar(v)
                for k, v in self.user.items()
                if k not in self.observer.axes
            }
            self._scalars = {**defaults, **updates}
        return self._scalars

    def _get_scalar(self, this):
        """Internal helper for creating `self.scalars`."""
        scalar = constant.scalar(this)
        unit = self.observer.system.get_unit(unit=scalar.unit)
        return scalar.convert(unit)

    def get_quantity(self, name: str):
        """Retrieve the named quantity from available attributes."""
        if name in self.scalars:
            return self.scalars[name]
        if name in self.observer.variables:
            return self.evaluate_variable(name)
        if name in self.observer.functions:
            return self.evaluate_function(name)

    @abc.abstractmethod
    def evaluate_variable(self, name: str) -> variable.Quantity:
        """Retrieve and update a variable quantity from the dataset."""
        raise NotImplementedError

    def evaluate_function(self, name: str) -> variable.Quantity:
        """Create a variable quantity from a function."""
        quantity = self.observer.functions[name]
        dependencies = {p: self.get_quantity(p) for p in quantity.parameters}
        return quantity(dependencies)

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

    def __init__(self, application: Application) -> None:
        self.application = application

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
        quantities: observer.Quantities,
        factory: typing.Type[Factory],
    ) -> None:
        self.quantities = quantities
        self.factory = factory
        self._type = Application

    @abc.abstractmethod
    def get_unit(self, name: str) -> metadata.Unit:
        """Get the appropriate unit for this observable quantity."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_axes(self, name: str) -> metadata.Axes:
        """Get the appropriate axes for this observable quantity."""
        raise NotImplementedError

    def use(self, __type: typing.Type[Application]):
        """Set the application class for this context."""
        self._type = __type

    def apply(self, **user):
        """Apply user constraints to this observing context."""
        application = self._type(self.quantities, **user)
        return self.factory(application)


class Primary(Context):
    """The context of a primary observable quantity."""

    def __init__(self, quantities: observer.Quantities) -> None:
        """Initialize a primary observing context.
        
        This concrete observing context will initialize the base class with a
        variable factory.

        Parameters
        ----------
        quantities : `~observer.Quantities`
            The quantities available within this context.
        """
        super().__init__(quantities, Variables)

    @property
    def variables(self):
        """An interface to the available variable quantities."""
        return self.quantities.variables

    def get_unit(self, name: str) -> metadata.Unit:
        return self.variables[name].unit

    def get_axes(self, name: str) -> metadata.Axes:
        return self.variables[name].axes


class Derived(Context):
    """The context of a derived observable quantity."""

    def __init__(self, quantities: observer.Quantities) -> None:
        """Initialize a derived observing context.
        
        This concrete observing context will initialize the base class with a
        functions factory.

        Parameters
        ----------
        quantities : `~observer.Quantities`
            The quantities available within this context.
        """
        super().__init__(quantities, Functions)

    @property
    def functions(self):
        """An interface to the available computable quantities."""
        return self.quantities.functions

    def get_unit(self, name: str) -> metadata.Unit:
        return self.functions.get_unit(name)

    def get_axes(self, name: str) -> metadata.Axes:
        return self.functions.get_axes(name)


class Composed(Context):
    """The context of a composed observable quantity."""

    def __init__(self, quantities: observer.Quantities) -> None:
        """Initialize a composed observing context.
        
        This concrete observing context will initialize the base class with an
        expressions factory.

        Parameters
        ----------
        quantities : `~observer.Quantities`
            The quantities available within this context.
        """
        super().__init__(quantities, Expressions)

    @property
    def functions(self):
        """An interface to the available computable quantities."""
        return self.quantities.functions

    def get_unit(self, name: str) -> metadata.Unit:
        return algebraic.Expression(name).apply(self.functions.get_unit)

    def get_axes(self, name: str) -> metadata.Axes:
        return algebraic.Expression(name).apply(self.functions.get_axes)


T = typing.TypeVar('T', bound=Context)


class Implementation(typing.Generic[T]):
    """An implementation of an observable quantity."""

    def __init__(self, name: str, context: T) -> None:
        self.name = name
        self.context = context

    def apply(self, **user):
        """Create an observation from user constraints."""
        return self.context.apply(**user).observe(self.name)


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
        name = name or "Anonymous"
        self.implementation = Implementation(name, context)
        self._name = name
        self._unit = context.get_unit(name)
        self._axes = context.get_axes(name)
        self._cache = None

    def convert(self, unit: metadata.UnitLike):
        return super().convert(unit)

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
        return self.implementation.apply(**current)

    def __str__(self) -> str:
        attrs = ('unit', 'name', 'axes')
        return ', '.join(f"{a}={getattr(self, a)}" for a in attrs)


class Interface(collections.abc.Mapping):
    """ABC for interfaces to observable quantities."""

    def __init__(
        self,
        available: observer.Quantities,
        application: typing.Type[Application],
        primary: typing.Iterable[str],
        derived: typing.Iterable[str],
    ) -> None:
        self.available = available
        self.application = application
        self.primary = primary
        """The names of observable quantities in the dataset."""
        self.derived = derived
        """The names of observable quantities computed from variables."""
        self.names = self.primary + self.derived
        """The names of all primary and derived observable quantities."""

    def __len__(self) -> int:
        return len(self.names)

    def __iter__(self) -> typing.Iterator:
        return iter(self.names)

    def __getitem__(self, __k: str) -> Quantity:
        """Get the named observable quantity."""
        return Quantity(self.implement(__k), name=__k)

    def implement(self, name: str):
        """Create the implementation of an observable quantity."""
        Type = self._get_type(name)
        context = Type(self.available)
        context.use(self.application)
        return context

    def _get_type(self, name: str):
        """Get the appropriate implementation type."""
        if name in self.primary:
            return Primary
        if name in self.derived:
            return Derived
        if expression(name):
            return Composed
        raise ValueError(
            f"No implementation available for {name!r}"
        ) from None

