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
from goats.core import reference
from goats.core import physical
from goats.core import variable
from goats.eprem import interpolation


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
        self._indices = aliased.MutableMapping.fromkeys(self.axes, value=())

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

    def evaluate(self, name: str, **user):
        """Apply user constraints to the named variable quantity."""
        if name not in self.variables:
            return
        old = self.variables[name]
        indices = self._compute_indices(user)
        if any(self._get_reference(alias) for alias in old.name):
            # This is an axis-reference quantity.
            # - Should this even subscript `old`?
            # - Maybe put a breakpoint here to see if `user` is ever non-empty
            #   when getting an axis-reference quantity. If it is always empty,
            #   we can't be certain; if it is ever non-empty, we know that we
            #   need to subscript `old`.
            return old[indices]
        axes = self._compute_axes(old, user)
        if not axes:
            # There are no axes over which to interpolate.
            return old[indices]
        new = self._interpolate(old, user)
        idx = tuple(
            indices[axis]
            if axis in axes else slice(None)
            for axis in old.axes
        )
        return new[idx]

    def _compute_coordinates(self, user: dict):
        """Determine the measurable observing indices."""
        indices = self._compute_indices(user)
        coordinates = {
            axis: {
                'targets': numpy.array(idx.data),
                'reference': self._get_reference(axis),
            }
            for axis, idx in indices.items()
            if axis in self.axes and idx.unit is not None
        }
        for key in reference.ALIASES['radius']:
            if values := user.get(key):
                radii = iterables.whole(values)
                floats = [float(radius) for radius in radii]
                updates = {
                    'targets': numpy.array(floats),
                    'reference': self._get_reference('radius'),
                }
                coordinates.update(updates)
        return coordinates

    def _compute_axes(self, q: variable.Quantity, user: dict):
        """Determine over which axes to interpolate, if any."""
        indices = self._compute_indices(user)
        crd = {k: v for k, v in indices.items() if v.unit is not None}
        coordinates = {a: crd[a].data for a in q.axes if a in crd}
        references = [self._get_reference(a) for a in q.axes if a in crd]
        axes = [
            a for (a, c), r in zip(coordinates.items(), references)
            if not numpy.all([r.array_contains(target) for target in c])
        ]
        if any(r in user for r in reference.ALIASES['radius']):
            axes.append('radius')
        return list(set(q.axes) - set(axes))

    def _compute_indices(self, user: dict) -> typing.Dict[str, index.Quantity]:
        """Create the relevant observing indices."""
        updates = {
            k: self._compute_index(k, v)
            for k, v in user.items()
            if k in self.axes
        }
        return {**self._indices, **updates}

    def _compute_index(self, key: str, this):
        """Compute a single indexing object from input values."""
        target = (
            self.axes[key].at(*iterables.whole(this))
            if not isinstance(this, index.Quantity)
            else this
        )
        if target.unit is not None:
            unit = self.system.get_unit(unit=target.unit)
            return target.convert(unit)
        return target

    def _get_reference(self, name: str) -> typing.Optional[variable.Quantity]:
        """Get a reference quantity for indexing."""
        if self._references is None:
            axes = {k: v.reference for k, v in self.axes.items(aliased=True)}
            rtp = {
                (k, *self.variables.alias(k, include=True)): self.variables[k]
                for k in {'radius', 'theta', 'phi'}
            }
            self._references = aliased.Mapping({**axes, **rtp})
        return self._references.get(name)

    def _interpolate(
        self,
        q: variable.Quantity,
        user: dict,
    ) -> variable.Quantity:
        """Internal interpolation logic."""
        coordinates = self._compute_coordinates(user)
        array = None
        for coordinate, current in coordinates.items():
            array = self._interpolate_coordinate(
                current['targets'],
                current['reference'],
                coordinate=coordinate,
                workspace=array,
            )
        meta = {k: getattr(q, k, None) for k in {'unit', 'name', 'axes'}}
        return variable.Quantity(array, **meta)

    def _interpolate_coordinate(
        self,
        q: variable.Quantity,
        targets: numpy.ndarray,
        reference: variable.Quantity,
        coordinate: str=None,
        workspace: numpy.ndarray=None,
    ) -> numpy.ndarray:
        """Interpolate a variable array based on a known coordinate."""
        array = numpy.array(q) if workspace is None else workspace
        indices = (q.axes.index(d) for d in reference.axes)
        dst, src = zip(*enumerate(indices))
        reordered = numpy.moveaxis(array, src, dst)
        interpolated = interpolation.apply(
            reordered,
            numpy.array(reference),
            targets,
            coordinate=coordinate,
        )
        return numpy.moveaxis(interpolated, dst, src)

    # We could remove this and let users directly call `axis.Interface.resolve`
    # on the `axes` property.
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

