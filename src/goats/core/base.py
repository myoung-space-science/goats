"""Objects that are available to all goats."""

import abc
from pathlib import Path
import typing

import matplotlib.pyplot as plt
import numpy as np

from goats.core import aliased
from goats.core import datatypes
from goats.core import iterables
from goats.core import quantities
from goats.core import indexing
from goats.core import spelling


@typing.runtime_checkable
class Observed(typing.Protocol):
    """Specification of an observed quantity."""

    unit: quantities.Unit
    """The unit of the observed values"""

    @abc.abstractmethod
    def __array__(self, *args, **kwargs):
        pass


class Observation(iterables.ReprStrMixin):
    """The result of observing an observable object."""

    def __init__(
        self,
        data: datatypes.Variable,
        indices: typing.Mapping[str, indexing.Indices],
        assumptions: typing.Mapping[str, quantities.Scalar]=None,
    ) -> None:
        self._data = data
        self.name = data.name
        self._indices = indices
        self._assumptions = assumptions or {}
        self._axes = None
        self._parameters = None

    def __array__(self, *args, **kwargs) -> np.ndarray:
        """Support automatic conversion to a `numpy.ndarray`."""
        return np.array(self._data, *args, **kwargs)

    def __getitem__(self, item):
        """Get an assumption, an array axis, or array values."""
        if isinstance(item, str):
            if item in self._indices:
                return self._indices[item]
            if item in self._assumptions:
                return self._assumptions[item]
            raise KeyError(item) from None
        return self._data[item]

    @property
    def axes(self):
        """The indexable axes of this observation's array."""
        if self._axes is None:
            if isinstance(self._indices, aliased.Mapping):
                self._axes = self._indices.keys(aliased=True)
            else:
                self._axes = self._indices.keys()
        return self._axes

    @property
    def parameters(self):
        """The names of assumptions relevant to this observation."""
        if self._parameters is None:
            if isinstance(self._assumptions, aliased.Mapping):
                self._parameters = self._assumptions.keys(aliased=True)
            else:
                self._parameters = self._assumptions.keys()
        return self._parameters

    def unit(self, new: typing.Union[str, quantities.Unit]=None):
        """Get or set the unit of this observation's data values."""
        if not new:
            return self._data.unit
        self._data = self._data.convert_to(new)
        return self

    def __eq__(self, other) -> bool:
        """True if two instances have equivalent attributes."""
        if not isinstance(other, Observation):
            return NotImplemented
        if not self._equal_attrs(other):
            return False
        return super().__eq__(other)

    def _equal_attrs(self, other):
        """True if two instances have the same attributes."""
        return all(
            getattr(other, attr) == getattr(self, attr)
            for attr in {'indices', 'assumptions'}
        )

    def __str__(self) -> str:
        """A simplified representation of this object."""
        axes = [str(axis) for axis in self.axes]
        parameters = [str(parameter) for parameter in self.parameters]
        attrs = [
            f"'{self.name}'",
            f"unit='{self.unit()}'",
            f"axes={axes}",
            f"parameters={parameters}",
        ]
        return ', '.join(attrs)

    def plot(
        self,
        *args,
        ax: plt.Axes=None,
        xaxis: str=None,
        show: bool=False,
        path: typing.Union[str, Path]=None,
        **kwargs
    ) -> typing.Optional[plt.Axes]:
        """Plot this observation."""
        data = np.squeeze(self)
        if xaxis in self._indices:
            indices = self._indices[xaxis]
            if isinstance(indices, indexing.Coordinates):
                xarr = np.array(indices.values)
            else:
                xarr = tuple(indices)
        else:
            xarr = np.arange(data.shape[0])
        if ax is not None:
            ax.plot(xarr, data, *args, **kwargs)
            return ax
        lines = plt.plot(xarr, data, *args, **kwargs)
        if not show and not path:
            return lines
        if show:
            plt.show()
        if path:
            savepath = Path(path).expanduser().resolve()
            plt.savefig(savepath)


class Interface(abc.ABC, iterables.ReprStrMixin):
    """Base class for observing interfaces."""

    @abc.abstractmethod
    def apply(self, constraints: typing.Mapping):
        """Apply the given observing constraints to an observable."""
        pass

    @property
    @abc.abstractmethod
    def result(self):
        """The result of this observation."""
        pass

    @property
    @abc.abstractmethod
    def context(self):
        """The context of this observation."""
        pass


S = typing.TypeVar('S', bound='Observable')


class Observable(iterables.ReprStrMixin):
    """An object that, when observed, produces an observation."""

    def __init__(self: S, interface: Interface, name: str) -> None:
        self._interface = interface
        self.name = name
        self._constraints = {}
        self.axes = interface.implementation.axes

    def observe(self, reset: bool=False, **constraints):
        """Create an observation within the given constraints.
        
        This method will create an observation of this observable within the
        current collection of observational constraints. Passing additional
        constraints will update the current set; the caller may reset the
        constraints by passing `reset=True` or by first calling
        `~Observable.reset`. The default collection of observational constraints
        uses all relevant axis indices and default parameter values.

        Parameters
        ----------
        reset : bool, default=false
            If true, discard existing constraints before applying the given
            constraints. This is equivalent to calling `~Observable.reset`
            before calling this method.

        **constraints : dict
            Key-value pairs of axes or parameters to update.

        Returns
        -------
        `~base.Observation`
            An object representing the resultant observation.
        """
        if reset:
            self.reset()
        self._constraints.update(constraints)
        self._interface.apply(self._constraints)
        return Observation(
            self._interface.result,
            **self._interface.context
        )

    def reset(self):
        """Reset the observing constraints in place.
        
        This method will discard constraints accumulated from previous calls to
        `~Observable.observe`. It is equivalent to calling `~Observable.observe`
        with `reset=True`.
        """
        self._constraints = {}

    def __str__(self) -> str:
        """A simplified representation of this object."""
        return f"'{self.name}', axes={list(self.axes)}"

    def __eq__(self, other):
        """True if two observables have the same name and constraints."""
        if isinstance(other, Observable):
            same_name = self.name == other.name
            same_constraints = self._constraints == other._constraints
            return same_name and same_constraints
        return NotImplemented


class Observer:
    """Base class for all observer objects."""

    def __init__(self, *observables: typing.Mapping[str, Observable]) -> None:
        self.observables = observables
        keys = [key for mapping in observables for key in mapping.keys()]
        self._spellcheck = spelling.SpellChecker(keys)

    def __getitem__(self, key: str):
        """Access an observable object by keyword, if possible."""
        for mapping in self.observables:
            if key in mapping:
                return mapping[key]
        if self._spellcheck.misspelled(key):
            raise spelling.SpellingError(key, self._spellcheck.suggestions)
        raise KeyError(f"No observable for {key!r}") from None


