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
from goats.core import variable


class Quantity(metadata.NameMixin, metadata.UnitMixin, metadata.AxesMixin):
    """A quantity with one or more name(s), a unit, and axes."""


class Interface(collections.abc.Collection):
    """An interface to quantities required to make observations."""

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
        self._cache = {}
        assumptions = {
            k: v for k, v in constants
            if isinstance(v, constant.Assumption)
        } if constants else {}
        self._default = {
            'indices': aliased.MutableMapping.fromkeys(axes, value=()),
            'scalars': aliased.MutableMapping(assumptions),
        }

    def __contains__(self, __x) -> bool:
        return __x in self.names

    def __iter__(self) -> typing.Iterator:
        return iter(self.names)

    def __len__(self) -> int:
        return len(self.names)

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
            self._names = list(self.variables) + list(self.functions)
        return self._names

    @property
    def functions(self):
        """The computable quantities available to this observer."""
        if self._functions is None:
            self._functions = computable.Interface(self.axes, self.variables)
        return self._functions

    def get_observable(self, key: str):
        """Get the observable quantity corresponding to `key`."""
        for this in (self.variables, self.functions):
            if key in this:
                return this[key]

    def get_unit(self, key: str):
        """Get the metric unit corresponding to `key`."""
        if key in self.variables:
            return self.variables[key].unit
        if key in self.functions:
            return self.functions[key].unit

    def get_axes(self, key: str):
        """Get the axis names corresponding to `key`."""
        if key in self.variables:
            return self.variables[key].axes
        if key in self.functions:
            return self.functions[key].axes

    def compute_indices(self, **user) -> typing.Dict[str, index.Quantity]:
        """Compute all available axis-indexing objects."""
        updates = {
            k: self._compute_index(k, v)
            for k, v in user.items()
            if k in self.axes
        }
        self._indices = {**self._default['indices'], **updates}

    def compute_index(self, key: str, **constraints):
        """Compute the axis-indexing object for `key`."""
        if 'indices' not in self._cache:
            self._cache['indices'] = {}
        if key in self._cache['indices']:
            return self._cache[key]
        if key not in self.axes:
            raise ValueError(f"No axis corresponding to {key!r}") from None
        if key not in constraints:
            return self.axes[key].at()
        idx = self._compute_index(key, constraints[key])
        self._cache['indices'] = idx
        return idx

    def _compute_index(self, key: str, this):
        """Compute a single indexing object from input values."""
        target = (
            this if isinstance(this, index.Quantity)
            else self.axes[key].at(*iterables.whole(this))
        )
        if target.unit is not None:
            unit = self.variables.system.get_unit(unit=target.unit)
            return target.convert(unit)
        return target

    def compute_scalars(self, **user) -> typing.Mapping[str, physical.Scalar]:
        """Compute the available single-valued assumptions."""
        if self._scalars is None:
            updates = {
                k: self._compute_scalar(v)
                for k, v in user.items()
                if k not in self.axes
            }
            self._scalars = {**self._default['scalars'], **updates}
        return self._scalars

    def compute_scalar(self, key: str, **constraints):
        """Create a single-valued physical assumption for `key`."""
        if 'scalars' not in self._cache:
            self._cache['scalars'] = {}
        if key in self._cache['scalars']:
            return self._cache[key]
        if key not in self.constants:
            raise ValueError(
                f"No parameter corresponding to {key!r}"
            ) from None
        val = self._compute_scalar(constraints.get(key, self.constants[key]))
        self._cache['scalars'] = val
        return val

    def _compute_scalar(self, this):
        """Compute a single scalar assumption."""
        scalar = constant.scalar(this)
        unit = self.system.get_unit(unit=scalar.unit)
        return scalar.convert(unit)


class Result(typing.NamedTuple):
    """Container for observing results."""

    quantity: variable.Quantity
    axes: metadata.Axes
    parameters: typing.Tuple[str, ...]=()


class Application(abc.ABC):
    """ABC for observing applications.
    
    Concrete subclasses must define a method called `process` that takes a
    `~variable.Quantity` and returns a `~variable.Quantity` after applying
    observer-specific updates.
    """

    def __init__(
        self,
        interface: Interface,
        **constraints
    ) -> None:
        self.data = interface
        self.user = constraints
        self._indices = None
        self._scalars = None

    @property
    def indices(self) -> typing.Dict[str, index.Quantity]:
        """The available array-indexing objects."""
        if self._indices is None:
            self._indices = self.data.compute_indices(**self.user)
        return self._indices

    @property
    def scalars(self) -> typing.Mapping[str, physical.Scalar]:
        """The available single-valued assumptions."""
        if self._scalars is None:
            self._scalars = self.data.compute_scalars(**self.user)
        return self._scalars

    def observe(self, name: str):
        """Compute the target variable quantity and its observing context."""
        expression = algebraic.Expression(name)
        axes = []
        parameters = []
        term = expression[0]
        quantity = self.data.get_observable(term.base)
        result = self.evaluate(quantity)
        q0 = result.quantity ** term.exponent
        axes.append(result.axes)
        parameters.append(result.parameters)
        if len(expression) > 1:
            for term in expression[1:]:
                quantity = self.data.get_observable(term.base)
                result = self.evaluate(quantity)
                q0 *= result.quantity ** term.exponent
                axes.append(result.axes)
                parameters.append(result.parameters)
        # NOTE: These axes are not guaranteed to be sorted or unique. We could
        # consider defining `Quantity` as a subclass of `variable.Quantity`, so
        # that it correctly updates axes, and define parameter updates via
        # multiplication. The parameters could even be a formal metadata
        # attribute.
        return Result(q0, axes, parameters)

    def evaluate(self, this):
        """Create an observing result based on this quantity."""
        if isinstance(this, computable.Quantity):
            return Result(self.compute(this), this.axes, this.parameters)
        if isinstance(this, variable.Quantity):
            return Result(self.process(this), this.axes)
        raise ValueError(f"Unknown quantity: {this!r}") from None

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
        if key in self.scalars:
            return self.scalars[key]
        return self.observe(key)

