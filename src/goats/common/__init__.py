"""Objects that are available to all goats."""

import abc
from pathlib import Path
from typing import *

import matplotlib.pyplot as plt
import numpy as np

from goats.common import iterables
from goats.common import quantities
from goats.common import indexing
from goats.common import spelling


@runtime_checkable
class Observed(Protocol):
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
        data: Observed,
        indices: Mapping[str, indexing.Indices],
        assumptions: Mapping[str, quantities.Scalar]=None,
    ) -> None:
        self.data = data
        self.unit = data.unit
        """The unit of this observation's values."""
        self.indices = indices
        """The indices of this observation's array."""
        self.assumptions = assumptions or {}
        """The assumptions relevant to this observation."""

    # Allow unit updates as in Measured objects?

    def __array__(self, *args, **kwargs) -> np.ndarray:
        """Support automatic conversion to a `numpy.ndarray`."""
        return np.array(self.data, *args, **kwargs)

    def __eq__(self, other) -> bool:
        """True if two instances have equivalent attributes."""
        if not isinstance(other, Observation):
            return NotImplemented
        if not self._equal_attrs(other):
            return False
        return self.data == other.data

    def _equal_attrs(self, other):
        """True if two instances have the same attributes."""
        return all(
            getattr(other, attr) == getattr(self, attr)
            for attr in {'indices', 'assumptions'}
        )

    def __str__(self) -> str:
        """A simplified representation of this object."""
        attrs = ['data', 'indices', 'assumptions']
        return '\n'.join(f"{attr}={getattr(self, attr)}" for attr in attrs)

    def plot(
        self,
        *args,
        ax: plt.Axes=None,
        xaxis: str=None,
        show: bool=False,
        path: Union[str, Path]=None,
        **kwargs
    ) -> Optional[plt.Axes]:
        """Plot this observation."""
        data = np.squeeze(self)
        if xaxis in self.indices:
            indices = self.indices[xaxis]
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
    def apply(self, constraints: Mapping):
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


class Observable(iterables.ReprStrMixin):
    """An object that, when observed, produces an observation."""

    def __init__(self, interface: Interface, name: str) -> None:
        self._interface = interface
        self.name = name
        self._constraints = {}

    def given(self, **constraints) -> Observation:
        """Create an observation within the given constraints.

        This method updates the user constraints and creates a new observation
        in a single step. It is equivalent to calling the `use` method, then
        accessing the `observed`property.
        """
        return self.use(**constraints).observed

    @property
    def observed(self) -> Observation:
        """An observation within the current user constraints.

        Accessing this property will create an observation of this observable
        within the current observational constraints. The default collection of
        observational constraints uses all relevant indices and default values
        of assumptions. The caller can update the constraints via the `use`
        method.
        """
        self._interface.apply(self._constraints)
        return Observation(
            self._interface.result,
            **self._interface.context
        )

    def use(self, **constraints) -> 'Observable':
        """Update the user constraints.

        Parameters
        ----------
        **constraints : dict
            Key-value pairs of indices or assumptions to update.

        Returns
        -------
        Observable:
            The updated instance.
        """
        self._constraints = constraints.copy()
        return self

    def __str__(self) -> str:
        """A simplified representation of this object."""
        return f"'{self.name}'"


class Observer:
    """Base class for all observer objects."""

    def __init__(self, observables: Mapping[str, Observable]) -> None:
        self.observables = observables
        self._spellcheck = spelling.SpellChecker(observables.keys())

    def __getitem__(self, key: str):
        """Access an observable object by keyword, if possible."""
        if key in self.observables:
            return self.observables[key]
        if self._spellcheck.misspelled(key):
            raise spelling.SpellingError(key, self._spellcheck.suggestions)
        raise KeyError(f"No observable for {key!r}") from None


