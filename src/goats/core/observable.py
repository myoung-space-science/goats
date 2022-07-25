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


class Context(abc.ABC):
    """ABC for observing contexts."""

    def __init__(
        self,
        axes: axis.Interface,
        variables: variable.Interface,
        assumptions: typing.Mapping[str, parameter.Assumption],
    ) -> None:
        self.axes = axes
        self.variables = variables
        self._cache = {}
        self._default = {
            'indices': aliased.MutableMapping.fromkeys(axes, value=()),
            'scalars': aliased.MutableMapping(assumptions),
        }

    def __contains__(self, __k: str):
        """True if `__k` names a variable or argument."""
        return __k in self.variables or __k in self.arguments

    @abc.abstractmethod
    def get_variable(self, key: str, user: dict) -> variable.Quantity:
        """Retrieve and update a variable quantity as necessary."""
        raise NotImplementedError

    def get_scalars(self, user: dict):
        """Extract relevant single-valued assumptions."""
        updates = {
            k: self._get_assumption(v)
            for k, v in user.items()
            if k not in self.axes
        }
        return {**self._default['scalars'], **updates}

    def _get_assumption(self, this):
        """Get a single assumption from user input."""
        scalar = self._force_scalar(this)
        unit = self.variables.system.get_unit(unit=scalar.unit)
        return scalar.convert(unit)

    def _force_scalar(self, this) -> measurable.Scalar:
        """Make sure `this` is a `~measurable.Scalar`."""
        if isinstance(this, measurable.Scalar):
            return this
        if isinstance(this, parameter.Assumption):
            return this[0]
        if isinstance(this, measurable.Measurement):
            return physical.Scalar(this.values[0], unit=this.unit)
        measured = measurable.measure(this)
        if len(measured) > 1:
            raise ValueError("Can't use a multi-valued assumption") from None
        return self._force_scalar(measured)

    def get_indices(self, user: dict) -> typing.Dict[str, index.Quantity]:
        """Create the relevant observing indices."""
        updates = {
            k: self._compute_index(k, v)
            for k, v in user.items()
            if k in self.axes
        }
        return {**self._default['indices'], **updates}

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

    def get_unit(self, key: str):
        """Determine the appropriate unit based on keyword."""
        this = observables.METADATA.get(key, {}).get('quantity')
        return self.variables.system.get_unit(quantity=this)

    def get_axes(self, key: str):
        """Retrieve or compute the axes corresponding to `key`."""
        if 'axes' not in self._cache:
            self._cache['axes'] = {}
        if key in self._cache['axes']:
            return self._cache['axes'][key]
        method = functions.REGISTRY[key]
        self._removed = self._get_metadata(method, 'removed')
        self._added = self._get_metadata(method, 'added')
        self._accumulated = []
        axes = self._gather_axes(method)
        self._cache['axes'][key] = axes
        return axes

    def _gather_axes(self, target: variable.Caller):
        """Recursively gather appropriate axes."""
        for parameter in target.parameters:
            if parameter in self.variables:
                axes = self.variables[parameter].axes
                self._accumulated.extend(axes)
            elif method := functions.REGISTRY[parameter]:
                self._removed.extend(self._get_metadata(method, 'removed'))
                self._added.extend(self._get_metadata(method, 'added'))
                self._accumulated.extend(self._gather_axes(method))
        unique = set(self._accumulated) - set(self._removed) | set(self._added)
        return self.axes.resolve(unique, mode='append')

    def _get_metadata(self, method: variable.Caller, key: str) -> list:
        """Helper for accessing a method's metadata dictionary."""
        if key not in method.meta:
            return [] # Don't go through the trouble if it's not there.
        value = method.meta[key]
        return list(iterables.whole(value))


class Application(abc.ABC):
    """ABC for observing applications."""

    def __init__(self, context: Context, **constraints) -> None:
        self.context = context
        self.indices = self.context.get_indices(constraints)
        self.scalars = self.context.get_scalars(constraints)

    def get_quantity(self, name: str):
        """Retrieve the named quantity from available attributes."""
        if name in self.scalars:
            return self.scalars[name]
        return self.evaluate_variable(name)

    @abc.abstractmethod
    def evaluate_variable(self, name: str) -> variable.Quantity:
        """Apply user constraints to the named variable quantity."""
        raise NotImplementedError

    def evaluate_function(self, name: str):
        """Create a variable quantity from a function."""
        interface = functions.REGISTRY[name]
        method = interface.pop('method')
        caller = variable.Caller(method, **interface)
        deps = {p: self.get_quantity(p) for p in caller.parameters}
        quantity = observables.METADATA.get(name, {}).get('quantity', None)
        data = caller(**deps)
        return variable.Quantity(
            data,
            axes=self.context.get_axes(name),
            unit=self.context.get_unit(quantity=quantity),
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


class Implementation(abc.ABC):
    """ABC for implementations of observable quantities."""

    def __init__(
        self,
        axes: axis.Interface,
        variables: variable.Interface,
        name: typing.Union[str, typing.Iterable[str], metadata.Name]=None,
    ) -> None:
        self.context = Context(axes, variables)
        self.name = metadata.Name(name)

    def apply(self, **constraints):
        """Apply user-defined observing constraints."""
        applied = Application(self.context, **constraints)
        result = self.compute(applied)
        return observed.Quantity(result, applied.indices, applied.scalars)

    @abc.abstractmethod
    def compute(self, application: Application) -> variable.Quantity:
        """Create the observed variable quantity."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_unit(self) -> metadata.Unit:
        """Determine the appropriate unit."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_axes(self) -> metadata.Axes:
        """Determine the appropriate axes."""
        raise NotImplementedError


class Primary(Implementation):
    """Get an observable quantity from an observer's dataset."""

    def compute(self, application: Application) -> variable.Quantity:
        return application.evaluate_variable(self.name)

    def get_unit(self) -> metadata.Unit:
        return self.context.variables[self.name].unit

    def get_axes(self) -> metadata.Axes:
        return self.context.variables[self.name].axes


class Derived(Implementation):
    """Compute the result of a function of observable quantities."""

    def compute(self, application: Application) -> variable.Quantity:
        return application.evaluate_function(self.name)

    def get_unit(self) -> metadata.Unit:
        return self.context.get_unit(self.name)

    def get_axes(self) -> metadata.Axes:
        return self.context.get_axes(self.name)


class Composed(Implementation):
    """Evaluate an algebraic expression of other observable quantities."""

    def compute(self, application: Application) -> variable.Quantity:
        return application.evaluate_expression(self.name)

    def get_unit(self) -> metadata.Unit:
        return self._build_attr(self.context.get_unit)

    def get_axes(self) -> metadata.Axes:
        return self._build_attr(self.context.get_axes)

    def _build_attr(self, method: typing.Callable):
        """Build an expression for a metadata attribute."""
        expression = algebraic.Expression(self.name)
        bases = [method(term.base) for term in expression]
        exponents = [term.exponent for term in expression]
        result = bases[0] ** exponents[0]
        for base, exponent in zip(bases[1:], exponents[1:]):
            result *= base ** exponent
        return result


class Metadata(
    metadata.UnitMixin,
    metadata.NameMixin,
    metadata.AxesMixin,
): ...

class Quantity(Metadata):
    """A quantity that produces an observation."""

    def __init__(self, __implementation: Implementation) -> None:
        self.interface = __implementation
        self._constraints = None
        self._unit = __implementation.get_unit()
        self._axes = __implementation.get_axes()
        self._name = __implementation.name

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
        return self.interface.apply(constraints)

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


