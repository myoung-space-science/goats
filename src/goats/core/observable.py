import abc
import collections.abc
import typing

from goats.core import algebraic
from goats.core import axis
from goats.core import metadata
from goats.core import observed
from goats.core import parameter
from goats.core import variable


class Implementation(abc.ABC):
    """Base class for observable implementations."""

    Arguments = typing.TypeVar('Arguments', bound=typing.Mapping)
    Arguments = typing.Mapping[
        str,
        typing.Union[parameter.Assumption, parameter.Option]
    ]

    def __init__(
        self,
        name: str,
        variables: variable.Interface,
        axes: axis.Interface,
        arguments: Arguments,
    ) -> None:
        self.name = name
        self.variables = variables
        self.axes = axes
        self.arguments = arguments

    @abc.abstractmethod
    def apply(self, **constraints) -> observed.Quantity:
        """Create an observation within the given constraints."""
        pass

    def get_dependency(self, name: str):
        """Get a named dependency of this observable."""
        if name in self.arguments:
            return self.arguments[name]

    def get_variable(self, name: str, constraints: dict):
        """Get the named variable from the dataset, if possible."""
        interpolate = 'radius' in constraints
        # Port code from `eprem.observables.Application`.
        if variable := self.variables.get(name):
            return self.axes.subscript(variable)


class Primary(Implementation):
    """A primary observable."""

    def apply(self, **constraints):
        return self.get_variable(self.name, constraints)


class Derived(Implementation):
    """A derived observable."""


class Compound(Implementation):
    """A compound observable."""

    def apply(self, **constraints):
        expression = algebraic.Expression(self.name)


class Metadata(
    metadata.UnitMixin,
    metadata.NameMixin,
    metadata.AxesMixin,
): ...

class Quantity(Metadata):
    """A quantity that produces an observation."""

    def __init__(
        self,
        __implementation,
        unit: metadata.UnitLike=None,
        name: typing.Union[str, typing.Iterable[str], metadata.Name]=None,
        axes: typing.Union[str, typing.Iterable[str], metadata.Axes]=None,
    ) -> None:
        self.interface = __implementation
        self._unit = unit
        self._name = name
        self._axes = axes
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


