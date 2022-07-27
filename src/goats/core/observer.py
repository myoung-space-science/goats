import typing

from goats.core import aliased
from goats.core import axis
from goats.core import base
from goats.core import functions
from goats.core import index
from goats.core import iterables
from goats.core import measurable
from goats.core import metadata
from goats.core import parameter
from goats.core import physical
from goats.core import reference
from goats.core import spelling
from goats.core import variable


class Dataset:
    """Interface to observer dataset quantities."""

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
        self.names = list(variables) + list(assumptions)

    def __contains__(self, __k: str):
        """True if `__k` names a variable or an assumption."""
        return __k in self.names

    def get_unit(self, key: str) -> metadata.Unit:
        """Determine the unit of `key` based on its metric quantity."""
        this = reference.METADATA.get(key, {}).get('quantity')
        return self.variables.system.get_unit(quantity=this)

    def get_axes(self, key: str):
        """Retrieve or compute appropriate axis names for `key`."""
        if key in self.variables:
            return self.variables[key].axes
        if key in functions.REGISTRY:
            return self._compute_axes(key)

    def _compute_axes(self, key: str):
        """Compute appropriate axis names."""
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


class Interface:
    """The base class for all observers."""

    def __init__(
        self,
        *observables: typing.Mapping[str, base.Quantity],
    ) -> None:
        """Initialize this instance.
        
        Parameters
        ----------
        *observables
            Zero or more mappings from string key to `~base.Quantity`. All
            included quantities will be available to users via the standard
            bracket syntax (i.e., `observer[<key>]`).
        """
        self.observables = observables
        keys = [key for mapping in observables for key in mapping.keys()]
        self._spellcheck = spelling.SpellChecker(keys)

    def __getitem__(self, key: str):
        """Access an observable quantity by keyword, if possible."""
        for mapping in self.observables:
            if key in mapping:
                return mapping[key]
        if self._spellcheck.misspelled(key):
            raise spelling.SpellingError(key, self._spellcheck.suggestions)
        raise KeyError(f"No observable for {key!r}") from None

