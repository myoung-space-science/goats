import abc
import collections
import collections.abc
import contextlib
import numbers
import operator as standard
import typing

from goats.core import algebraic
from goats.core import aliased
from goats.core import axis
from goats.core import computable
from goats.core import constant
from goats.core import index
from goats.core import iterables
from goats.core import metadata
from goats.core import observed
from goats.core import physical
from goats.core import reference
from goats.core import variable


class Parameters(collections.abc.Sequence, iterables.ReprStrMixin):
    """The parameters of an `~observing.Quantity`."""

    def __init__(self, *names: str) -> None:
        self._names = self._init(*names)

    def _init(self, *args):
        names = iterables.unique(*iterables.unwrap(args, wrap=tuple))
        if not names or all(isinstance(name, str) for name in names):
            return names
        raise TypeError(
            f"Can't initialize instance of {type(self)}"
            f" with {names!r}"
        )

    __abs__ = metadata.identity(abs)
    """Called for abs(self)."""
    __pos__ = metadata.identity(standard.pos)
    """Called for +self."""
    __neg__ = metadata.identity(standard.neg)
    """Called for -self."""

    __add__ = metadata.identity(standard.add)
    """Called for self + other."""
    __sub__ = metadata.identity(standard.sub)
    """Called for self - other."""

    def merge(a, *others):
        """Return the unique axis names in order."""
        names = list(a._names)
        for b in others:
            if isinstance(b, Parameters):
                names.extend(b._names)
        return Parameters(*set(names))

    __mul__ = merge
    """Called for self * other."""
    __rmul__ = merge
    """Called for other * self."""
    __truediv__ = merge
    """Called for self / other."""

    def __pow__(self, other):
        """Called for self ** other."""
        if isinstance(other, numbers.Real):
            return self
        return NotImplemented

    def __eq__(self, other):
        """True if self and other represent the same axes."""
        return (
            isinstance(other, Parameters) and other._names == self._names
            or (
                isinstance(other, str)
                and len(self) == 1
                and other == self._names[0]
            )
            or (
                isinstance(other, typing.Iterable)
                and len(other) == len(self)
                and all(i in self for i in other)
            )
        )

    def __hash__(self):
        """Support use as a mapping key."""
        return hash(self._names)

    def __len__(self) -> int:
        """Called for len(self)."""
        return len(self._names)

    def __getitem__(self, __i: typing.SupportsIndex):
        """Called for index-based access."""
        return self._names[__i]

    def __str__(self) -> str:
        return f"[{', '.join(repr(name) for name in self._names)}]"


class Quantity(variable.Quantity):
    """A quantity with one or more name(s), a unit, and axes."""

    _parameters: Parameters=None

    def __init__(self, __data, **meta) -> None:
        super().__init__(__data, **meta)
        parsed = self.parse_attrs(__data, meta, parameters=())
        self._parameters = Parameters(parsed['parameters'])
        self.meta.register('parameters')
        self.display.register('parameters')
        self.display['__str__'].append("parameters={parameters}")
        self.display['__repr__'].append("parameters={parameters}")

    def parse_attrs(self, this, meta: dict, **targets):
        if (
            isinstance(this, variable.Quantity)
            and not isinstance(this, Quantity)
        ): # barf
            meta.update({k: getattr(this, k) for k in ('unit', 'name', 'axes')})
            this = this.data
        return super().parse_attrs(this, meta, **targets)

    @property
    def parameters(self):
        """The optional parameters that define this observing quantity."""
        return Parameters(self._parameters)


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
        assumptions = {
            k: v
            for k, v in constants.items(aliased=True)
            if isinstance(v, constant.Assumption)
        } if constants else {}
        self.assumptions = aliased.Mapping(assumptions)
        """The default physical assumptions."""

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

    # TODO: Refactor `get_unit` and `get_axes` to reduce overlap.

    def get_unit(self, key: str):
        """Internal helper for `~Interface.get_unit`."""
        if key in self.variables:
            return self.variables[key].unit
        if key in self.functions:
            return self.functions[key].unit
        s = str(key)
        expression = algebraic.Expression(reference.NAMES.get(s, s))
        term = expression[0]
        this = self.get_unit(term.base) ** term.exponent
        if len(expression) > 1:
            for term in expression[1:]:
                this *= self.get_unit(term.base) ** term.exponent
        return metadata.Unit(this)

    def get_axes(self, key: str):
        """Determine the axes corresponding to `key`."""
        if key in self.variables:
            return self.variables[key].axes
        if key in self.functions:
            return self.functions[key].axes
        s = str(key)
        expression = algebraic.Expression(reference.NAMES.get(s, s))
        term = expression[0]
        this = self.get_axes(term.base)
        if len(expression) > 1:
            for term in expression[1:]:
                this *= self.get_axes(term.base)
        return metadata.Axes(this)

    def compute_index(self, key: str, **constraints) -> index.Quantity:
        """Compute the axis-indexing object for `key`."""
        if key not in self.axes:
            raise ValueError(f"No axis corresponding to {key!r}") from None
        if key not in constraints:
            return self.axes[key].at()
        return self._compute_index(key, constraints[key])

    def _compute_index(self, key: str, this):
        """Compute a single indexing object from input values."""
        target = (
            this if isinstance(this, index.Quantity)
            else self.axes[key].at(*iterables.whole(this))
        )
        if target.unit is not None:
            unit = self.variables.system.get_unit(unit=target.unit)
            return target[unit]
        return target

    def compute_value(self, key: str, **constraints) -> physical.Scalar:
        """Create a parameter value for `key`."""
        if key not in self.constants and key not in constraints:
            raise ValueError(
                f"No parameter corresponding to {key!r}"
            ) from None
        if key in constraints:
            return self._compute_value(constraints[key])
        return self._compute_value(self.constants[key])

    def _compute_value(self, this):
        """Compute a parameter value."""
        # TODO: Generalize beyond scalar parameters.
        scalar = physical.scalar(this)
        unit = self.system.get_unit(unit=scalar.unit)
        return scalar[unit]


class Result(typing.NamedTuple):
    """Container for observing results."""

    quantity: variable.Quantity
    axes: metadata.Axes
    parameters: typing.Tuple[str, ...]=()


class Context:
    """A general observing context."""

    def __init__(self, interface: Interface) -> None:
        self.interface = interface
        """The interface to an observer's dataset."""
        self.user = None
        """The current set of user constraints within this context."""
        self._cache = {}

    def __contains__(self, __k: str):
        """True if this context contains the named constraint."""
        # Should this check all dimensions and parameters or only user
        # constraints?
        return __k in self.user

    # TODO: Extract common methods from here and `observable.Quantity`.
    # - constraints attribute
    # - apply method
    # - reset method

    def apply(self, **constraints):
        """Update the observing constraints.
        
        This method will apply the given constraints when computing axis indices
        and parameter values within this context. It will also clear the
        corresponding items from the instance cache.
        """
        if self.user is None:
            self.user = {}
        for key in constraints:
            for subset in self._cache.values():
                if key in subset:
                    del subset[key]
        self.user.update(constraints)
        return self

    def reset(self, *keys: str):
        """Reset constraints to their default values.
        
        Parameters
        ----------
        *keys : string
            Zero or more names of constraints to reset. This method will also
            remove all named items from the instance cache By default, this
            method will reset all constraints and clear the cache.
        """
        if not keys:
            self.user = {}
            self._cache = {}
        else:
            subsets = self._cache.values()
            for key in keys:
                if key in self.user:
                    del self.user[key]
                for subset in subsets:
                    if key in subset:
                        del subset[key]
        return self

    def get_index(self, key: str) -> index.Quantity:
        """Get the axis-indexing object for `key`."""
        if 'indices' not in self._cache:
            self._cache['indices'] = {}
        if key in self._cache['indices']:
            return self._cache['indices'][key]
        with contextlib.suppress(ValueError):
            idx = self.interface.compute_index(key, **self.user)
            self._cache['indices'][key] = idx
            return idx

    def get_value(self, key: str) -> physical.Scalar:
        """Get the parameter value correpsonding to `key`."""
        if 'values' not in self._cache:
            self._cache['values'] = {}
        if key in self._cache['values']:
            return self._cache['values'][key]
        with contextlib.suppress(ValueError):
            val = self.interface.compute_value(key, **self.user)
            self._cache['values'][key] = val
            return val

    # TODO:
    # - Refactor `get_unit` and `get_axes` to reduce overlap.
    # - Redefine `_cache` as an aliased mapping.

    def get_unit(
        self,
        name: typing.Union[str, typing.Iterable[str], metadata.Name],
    ) -> metadata.Unit:
        """Determine the unit corresponding to `name`."""
        if 'unit' not in self._cache:
            self._cache['unit'] = {}
        if name in self._cache['unit']:
            return self._cache['unit'][name]
        if isinstance(name, str):
            return self.interface.get_unit(name)
        unit = (self.interface.get_unit(key) for key in name)
        return next(unit, None)

    def get_axes(
        self,
        name: typing.Union[str, typing.Iterable[str], metadata.Name],
    ) -> metadata.Axes:
        """Determine the axes corresponding to `name`."""
        if 'axes' not in self._cache:
            self._cache['axes'] = {}
        if name in self._cache['axes']:
            return self._cache['axes'][name]
        if isinstance(name, str):
            return self.interface.get_axes(name)
        axes = (self.interface.get_axes(key) for key in name)
        return next(axes, None)

