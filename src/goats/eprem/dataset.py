import typing
import numbers

import numpy
import numpy.typing

from goats.core import aliased
from goats.core import axis
from goats.core import datafile
from goats.core import fundamental
from goats.core import iterables
from goats.core import iotools
from goats.core import index
from goats.core import measurable
from goats.core import metric
from goats.core import numerical
from goats.core import physical
from goats.core import variable


class Indexers(iterables.ReprStrMixin, aliased.Mapping):
    """A factory for EPREM array-indexing objects."""

    def __init__(self, data: datafile.Interface) -> None:
        self.variables = variable.Interface(data)
        mass = self.variables['mass'].convert('nuc')
        charge = self.variables['charge'].convert('e')
        self.symbols = fundamental.elements(mass, charge)
        # TODO: Consider using reference arrays in methods, with the possible
        # exception of `_build_shell`.
        indexers = {
            'time': {
                'method': self._build_time,
                'reference': self.variables['time'],
            },
            'shell': {
                'method': self._build_shell,
                'reference': numpy.array(self.variables['shell'], dtype=int),
            },
            'species': {
                'method': self._build_species,
                'reference': self.symbols,
            },
            'energy': {
                'method': self._build_energy,
                'reference': self.variables['energy'],
            },
            'mu': {
                'method': self._build_mu,
                'reference': self.variables['mu'],
            },
        }
        mapping = {
            data.axes.alias(name, include=True): indexer
            for name, indexer in indexers.items()
        }
        super().__init__(mapping)

    def __getitem__(self, key: str) -> index.Factory:
        this = super().__getitem__(key)
        return index.Factory(this['method'], this['reference'])

    def _build_time(self, targets):
        """Build the time-axis indexer."""
        return self._build_coordinates(targets, self.variables['time'])

    def _build_shell(self, targets):
        """Build the shell-axis indexer."""
        return index.Quantity(targets)

    def _build_species(self, targets):
        """Build the species-axis indexer."""
        indices = []
        symbols = []
        for target in targets:
            if isinstance(target, str):
                indices.append(self.symbols.index(target))
                symbols.append(target)
            elif isinstance(target, numbers.Integral):
                indices.append(target)
                symbols.append(self.symbols[target])
        return index.Quantity(indices, values=targets)

    def _build_energy(self, targets, species: typing.Union[str, int]=0):
        """Build the energy-axis indexer."""
        s = self._build_species([species])
        _targets = (
            numpy.squeeze(targets[s, :]) if getattr(targets, 'ndim', None) == 2
            else targets
        )
        _reference = numpy.squeeze(self.variables['energy'][s, :])
        return self._build_coordinates(_targets, _reference)

    def _build_mu(self, targets):
        """Build the mu-axis indexer."""
        return self._build_coordinates(targets, self.variables['mu'])

    def _build_coordinates(
        self,
        targets: numpy.typing.ArrayLike,
        reference: variable.Quantity,
    ) -> index.Quantity:
        """Build an arbitrary coordinate object."""
        result = measurable.measure(targets)
        array = physical.Array(result.values, unit=result.unit)
        values = numpy.array(
            array.convert(reference.unit)
            if array.unit.dimension == reference.unit.dimension
            else array
        )
        indices = [
            numerical.find_nearest(reference, float(value)).index
            for value in values
        ]
        return index.Quantity(indices, values=values, unit=reference.unit)


    def __str__(self) -> str:
        return ', '.join(str(key) for key in self.keys(aliased=True))


class Axes(axis.Interface):
    """Interface to the EPREM axis objects."""

    def __init__(
        self,
        data: datafile.Interface,
        system: str=None,
    ) -> None:
        self.datafile = data
        super().__init__(
            variable.Interface(data, system),
            Indexers(data),
        )

    def resolve(
        self,
        names: typing.Iterable[str],
        mode: str='strict',
    ) -> typing.Tuple[str]:
        """Compute and order the available axes in `names`."""
        axes = self.datafile.available('axes').canonical
        ordered = tuple(name for name in axes if name in names)
        if mode == 'strict':
            return ordered
        extra = tuple(name for name in names if name not in ordered)
        if not extra:
            return ordered
        if mode == 'append':
            return ordered + extra
        raise ValueError(f"Unrecognized mode {mode!r}")


class Interface:
    """Interface to an EPREM dataset."""

    def __init__(
        self,
        path: iotools.PathLike,
        system: str=None,
    ) -> None:
        self.dataset = datafile.Interface(path)
        self.system = metric.System(system or 'mks')
        self._axes = None
        self._variables = None

    @property
    def axes(self):
        """Axis managers for this EPREM dataset."""
        if self._axes is None:
            self._axes = axis.Interface(Indexers, self.dataset, self.system)
        return self._axes

    @property
    def variables(self):
        """Variable quantities in this EPREM dataset."""
        if self._variables is None:
            self._variables = variable.Interface(self.dataset, self.system)
        return self._variables

    def resolve_axes(
        self,
        names: typing.Iterable[str],
        mode: str='strict',
    ) -> typing.Tuple[str]:
        """Compute and order the available axes in `names`."""
        axes = self.dataset.available('axes').canonical
        ordered = tuple(name for name in axes if name in names)
        if mode == 'strict':
            return ordered
        extra = tuple(name for name in names if name not in ordered)
        if not extra:
            return ordered
        if mode == 'append':
            return ordered + extra
        raise ValueError(f"Unrecognized mode {mode!r}")

