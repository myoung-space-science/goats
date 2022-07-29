import abc
import collections.abc
import typing

from goats.core import algebraic
from goats.core import aliased
from goats.core import axis
from goats.core import computable
from goats.core import constant
from goats.core import index
from goats.core import iterables
from goats.core import metadata
from goats.core import physical
from goats.core import reference
from goats.core import variable


class Attribute(collections.abc.Collection):
    """A collection of observer attributes."""

    def __init__(
        self,
        axes: axis.Interface,
        variables: variable.Interface,
    ) -> None:
        self.axes = axes
        self.variables = variables
        self._defaults = None
        self._defined = False

    @property
    def defaults(self) -> dict:
        """The default attribute values."""
        if not self._defined:
            if self._defaults is None:
                raise TypeError(
                    f"Can't instantiate {type(self)!r} without default values"
                ) from None
            self._defined = True
        return self._defaults

    def __contains__(self, __x) -> bool:
        return __x in self._defaults

    def __iter__(self) -> typing.Iterator:
        """Iterate over attribute names."""
        return iter(self._defaults)

    def __len__(self) -> int:
        """The number of available attributes."""
        return len(self._defaults)

    @abc.abstractmethod
    def convert(self, user: dict):
        """Extract appropriate values from user input."""
        raise NotImplementedError


class Indices(Attribute):
    """An observer's array-indexing objects."""

    def __init__(
        self,
        axes: axis.Interface,
        variables: variable.Interface,
    ) -> None:
        super().__init__(axes, variables)
        self._defaults = aliased.MutableMapping.fromkeys(axes, value=())

    def convert(self, user: dict) -> typing.Dict[str, index.Quantity]:
        """Create the relevant observing indices."""
        updates = {
            k: self._compute_index(k, v)
            for k, v in user.items()
            if k in self.axes
        }
        return {**self.defaults, **updates}

    def _compute_index(self, key: str, this):
        """Compute a single indexing object from input values."""
        target = (
            self.axes[key].at(*iterables.whole(this))
            if not isinstance(this, index.Quantity)
            else this
        )
        if target.unit is not None:
            unit = self.variables.system.get_unit(unit=target.unit)
            return target.convert(unit)
        return target


class Scalars(Attribute):
    """An observer's scalar parameter values."""

    def __init__(
        self,
        axes: axis.Interface,
        variables: variable.Interface,
        constants: constant.Interface,
    ) -> None:
        super().__init__(axes, variables)
        assumptions = {
            k: v for k, v in constants
            if isinstance(v, constant.Assumption)
        }
        self._defaults = aliased.MutableMapping(assumptions)

    def convert(self, user: dict) -> typing.Mapping[str, physical.Scalar]:
        """Extract relevant single-valued assumptions."""
        updates = {
            k: self._get_scalar(v)
            for k, v in user.items()
            if k not in self.axes
        }
        return {**self.defaults, **updates}

    def _get_scalar(self, this):
        """Get a single scalar assumption from user input."""
        scalar = constant.scalar(this)
        unit = self.variables.system.get_unit(unit=scalar.unit)
        return scalar.convert(unit)


class Context(aliased.Mapping):
    """The attributes used when creating a particular observation."""

    # TODO: I would like this to hold a collection of `Attribute` instances
    # (e.g., one for scalars, one for indices, etc.). Individual observers may
    # determine which attributes to include.




class Quantities:
    """The quantities available when creating observations."""

    def __init__(
        self,
        axes: axis.Interface,
        variables: variable.Interface,
        constants: constant.Interface,
    ) -> None:
        self.axes = axes
        """The axis-managing objects available to this observer."""
        self.variables = variables
        """The variable quantities available to this observer."""
        self.constants = constants
        """The constant quantities available to this observer."""
        self._system = None
        self._names = None
        self._functions = None
        self._indices = None
        self._scalars = None

    @property
    def system(self):
        """The metric system of this observer's dataset quantities."""
        if self._system is None:
            self._system = self.variables.system
        return self._system

    @property
    def names(self):
        """The names of all available quantities."""
        if self._names is None:
            self._names = list(self.variables) + list(self.scalars)
        return self._names

    @property
    def functions(self):
        """The computable quantities available to this observer."""
        if self._functions is None:
            self._functions = computable.Interface(self.axes, self.variables)
        return self._functions

    @property
    def indices(self):
        """This observer's array-indexing objects."""
        if self._indices is None:
            self._indices = Indices(self.axes, self.variables)
        return self._indices

    @property
    def scalars(self):
        """This observer's single-valued assumptions."""
        if self._scalars is None:
            self._scalars = Scalars(self.axes, self.variables, self.constants)
        return self._scalars

    def __contains__(self, __k: str):
        """True if `__k` names a variable or an assumption."""
        return __k in self.names


class Application(abc.ABC):
    """ABC for observing applications.
    
    Concrete subclasses must define a method called `evaluate_variable` that
    takes a string name and returns a `~variable.Quantity`.
    """

    def __init__(
        self,
        quantities: Quantities,
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

    def get_context(self, name: str):
        """Compute the observing context for the named quantity."""

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


class Quantity(metadata.NameMixin, metadata.UnitMixin, metadata.AxesMixin):
    """A quantity with one or more name(s), a unit, and axes."""


class Factory(aliased.Mapping):
    """A factory for quantities required to make observations."""

    # TODO: Should this use `computable.Interface` or replace it?

    def __init__(
        self,
        axes: axis.Interface,
        variables: variable.Interface,
    ) -> None:
        super().__init__(computable.registry, keymap=reference.ALIASES)
        self.axes = axes
        self.variables = variables
        self._cache = {}

    def __getitem__(self, __k: str) -> Quantity:
        if 'quantity' not in self._cache:
            self._cache['quantity'] = {}
        if __k in self._cache['quantity']:
            return self._cache['quantity'][__k]
        quantity = computable.Quantity(
            self.get_method(__k),
            axes=self.get_axes(__k),
            unit=self.get_unit(__k),
            name=self.get_name(__k),
        )
        self._cache['quantity'][__k] = quantity
        return quantity

    def get_method(self, key: str):
        """Get a `~computable.Method` for `key`."""
        try:
            this = super().__getitem__(key).copy()
        except KeyError:
            return None
        else:
            method = this.pop('method')
            return computable.Method(method, **this)

    def get_name(self, key: str):
        """Get the set of aliases for `key`."""
        return self.alias(key, include=True)

    def get_unit(self, key: str):
        """Determine the unit of `key` based on its metric quantity."""
        this = reference.METADATA.get(key, {}).get('quantity')
        return self.variables.system.get_unit(quantity=this)

    def get_axes(self, key: str):
        """Compute appropriate axis names for `key`."""
        if 'axes' not in self._cache:
            self._cache['axes'] = {}
        if key in self._cache['axes']:
            return self._cache['axes'][key]
        method = self.get_method(key)
        self._removed = self._get_metadata(method, 'removed')
        self._added = self._get_metadata(method, 'added')
        self._accumulated = []
        axes = self._gather_axes(method)
        self._cache['axes'][key] = axes
        return axes

    def _gather_axes(self, target: computable.Method):
        """Recursively gather appropriate axes."""
        for parameter in target.parameters:
            if parameter in self.variables:
                axes = self.variables[parameter].axes
                self._accumulated.extend(axes)
            elif method := self.get_method(parameter):
                self._removed.extend(self._get_metadata(method, 'removed'))
                self._added.extend(self._get_metadata(method, 'added'))
                self._accumulated.extend(self._gather_axes(method))
        unique = set(self._accumulated) - set(self._removed) | set(self._added)
        return self.axes.resolve(unique, mode='append')

    def _get_metadata(self, method: computable.Method, key: str) -> list:
        """Helper for accessing a method's metadata dictionary."""
        value = method.get(key)
        return list(iterables.whole(value)) if value else []


