"""Objects that are available to all goats."""

import abc
from pathlib import Path
import typing

import matplotlib.pyplot as plt
import numpy as np

from goats.common import iterables
from goats.common import quantities
from goats.common import indexing
from goats.common import spelling


@typing.runtime_checkable
class Observed(typing.Protocol):
    """Specification of an observed quantity."""

    unit: quantities.Unit
    """The unit of the observed values"""

    @abc.abstractmethod
    def __array__(self, *args, **kwargs):
        pass


class Observation(quantities.Variable, iterables.ReprStrMixin):
    """The result of observing an observable object."""

    def __init__(
        self,
        data: quantities.Variable,
        indices: typing.Mapping[str, indexing.Indices],
        assumptions: typing.Mapping[str, quantities.Scalar]=None,
    ) -> None:
        # TODO: Make `Variable` idempotent so we can just pass `data`.
        super().__init__(
            data.values,
            data.unit,
            data.axes,
            name=data.name,
        )
        self.indices = indices
        """The indices of this observation's array."""
        self.assumptions = assumptions or {}
        """The assumptions relevant to this observation."""

    def _new(self, **updated):
        argdict = {
            'values': updated.pop(
                'amount',
                updated.pop('values', self.values),
            ),
            'unit': updated.pop('unit', self.unit),
            'axes': updated.pop('axes', self.axes),
        }
        kwargs = {'name': self.name}
        data = quantities.Variable(*argdict.values(), **kwargs)
        indices = updated.pop('indices', self.indices)
        updated.update(
            data=data,
            indices=indices,
        )
        return type(self)(**updated)

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
        return f"{super().__str__()}, assumptions={self.assumptions}"

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

    def observe(self, **constraints):
        """Create an observation within the given constraints.
        
        This method will create an observation of this observable within the
        current collection of observational constraints. Passing additional
        constraints will update the current set; the caller may reset the
        constraints via `~Observable.reset`. The default collection of
        observational constraints uses all relevant indices and default values
        of assumptions.

        Parameters
        ----------
        **constraints : dict
            Key-value pairs of indices or assumptions to update.

        Returns
        -------
        `~base.Observation`
            An object representing the resultant observation.
        """
        self._constraints.update(constraints)
        self._interface.apply(self._constraints)
        return Observation(
            self._interface.result,
            **self._interface.context
        )

    def reset(self) -> S:
        """Reset the observing constraints."""
        self._constraints = {}
        return self

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

    def __init__(self, observables: typing.Mapping[str, Observable]) -> None:
        self.observables = observables
        self._spellcheck = spelling.SpellChecker(observables.keys())

    def __getitem__(self, key: str):
        """Access an observable object by keyword, if possible."""
        if key in self.observables:
            return self.observables[key]
        if self._spellcheck.misspelled(key):
            raise spelling.SpellingError(key, self._spellcheck.suggestions)
        raise KeyError(f"No observable for {key!r}") from None


