import abc
import collections.abc
import typing

from goats.core import aliased
from goats.core import axis
from goats.core import base
from goats.core import computable
from goats.core import constant
from goats.core import functions
from goats.core import index
from goats.core import iterables
from goats.core import measurable
from goats.core import metadata
from goats.core import physical
from goats.core import reference
from goats.core import spelling
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


class Quantities:
    """Interface to all quantities available to this observer."""

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

