import typing

import numpy
import numpy.typing
from scipy.interpolate import interp1d

from goats.core import aliased
from goats.core import index
from goats.core import iterables
from goats.core import numerical
from goats.core import reference
from goats.core import variable


def restrict_coordinate(
    array: numpy.ndarray,
    target: float,
    axis: typing.SupportsIndex,
) -> range:
    """Restrict `array` to `target` along `axis`."""
    bounds = numerical.get_bounding_indices(array, target, axis=axis)
    if bounds.ndim == 1:
        return range(bounds[0], bounds[1]+1)
    if bounds.ndim == 2:
        lower = min(bounds[:, 0])
        upper = max(bounds[:, 1])
        return range(lower, upper+1)
    # NOTE: We can get `lower` and `upper` for N-D arrays by replacing `:` with
    # `...`. The problem is that they will be (N-1)-D arrays that we will then
    # need to split into individual index vectors before returning.
    raise NotImplementedError(f"Can't operate on {bounds.ndim}-D array")


class Restriction:
    """A subset of indices with which to create subsets of arrays."""

    def __init__(
        self,
        rule: typing.Callable[..., typing.Iterable[typing.SupportsIndex]],
        *args,
        **kwargs,
    ) -> None:
        """
        Create a new restriction.

        Parameters
        ----------
        rule : callable
            The rule by which to create a subset of array indices. This must be
            a callable object that returns an iterable of objects to be used as
            indices (e.g., `range`).

        *args
            Positional arguments to pass to `rule`.

        **kwargs
            Keyword arguments to pass to `rule`.
        """
        self.indices = rule(*args, **kwargs)

    def apply(self, array: numpy.typing.ArrayLike, axis: int=None):
        """Restrict `array` to a subset along `axis`.
        
        Parameters
        ----------
        array : array-like
            The array from which to extract a subarray corresponding to the
            restricted indices.

        axis : int, default=-1
            The axis of `array` to restrict.

        Returns
        -------
        array-like
            The subarray.
        """
        if axis is None:
            axis = -1
        permuted = numpy.moveaxis(array, axis, 0)
        result = permuted[self.indices, ...]
        return numpy.moveaxis(result, 0, axis)


def apply(
    array: numpy.ndarray,
    reference: numpy.ndarray,
    targets: typing.Iterable[float],
    coordinate: str=None,
) -> numpy.ndarray:
    """Interpolate `array` to target values over `coordinate`."""
    interpolated = [
        _apply_interp1d(array, reference, target, coordinate=coordinate)
        for target in targets
    ]
    if reference.ndim == 2:
        return numpy.swapaxes(interpolated, 0, 1)
    return numpy.array(interpolated)


_AXES = {
    'time': 0,
    'radius': 1,
    'energy': 1,
    'mu': 0,
}


def _apply_interp1d(
    array: numpy.ndarray,
    reference: numpy.ndarray,
    target: float,
    coordinate: str=None,
) -> typing.List[float]:
    """Interpolate data to `target` along the leading axis."""
    if target in reference:
        idx = numerical.find_nearest(reference, target).index
        return array[idx]
    if (axis := _AXES.get(coordinate)) is not None:
        restriction = Restriction(
            restrict_coordinate,
            reference,
            target,
            axis=axis,
        )
        ref = restriction.apply(reference, axis=axis)
        arr = restriction.apply(array, axis=axis)
    else:
        ref = reference
        arr = array
    if reference.ndim == 2:
        interps = [interp1d(x, y, axis=0) for x, y in zip(ref, arr)]
        return [interp(target) for interp in interps]
    interp = interp1d(ref, arr, axis=0)
    return interp(target)


# This will require at least axis and variable interfaces. It may be simpler to
# put this logic in a TBD EPREM dataset class.
class Interface:
    """An object that performs interpolation on EPREM variable quantities."""

    def interpolate(
        self,
        __q: variable.Quantity,
        **user
    ) -> variable.Quantity:
        """Interpolate the given variable quantity."""
        indices = self._compute_indices(user)
        axes = self._compute_axes(__q, user)
        q = self._interpolate(__q, user)
        idx = tuple(
            indices[axis]
            if axis in axes else slice(None)
            for axis in q.axes
        )
        return q[idx]

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

    def _compute_indices(self, user: dict):
        """Create the relevant observing indices."""
        updates = {
            k: self._compute_index(k, v)
            for k, v in user.items()
            if k in self.axes
        }
        return {**self._default_indices, **updates}

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

    def _get_reference(self, name: str) -> variable.Quantity:
        """Get a reference quantity for indexing."""
        if self._references is None:
            axes = {k: v.reference for k, v in self.axes.items(aliased=True)}
            rtp = {
                (k, *self.variables.alias(k, include=True)): self.variables[k]
                for k in {'radius', 'theta', 'phi'}
            }
            self._references = aliased.Mapping({**axes, **rtp})
        return self._references[name]

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
        interpolated = apply(
            reordered,
            numpy.array(reference),
            targets,
            coordinate=coordinate,
        )
        return numpy.moveaxis(interpolated, dst, src)


_PARAMETERS = [
    ('radius', 'r', 'R'),
    ('theta', 'T'),
    ('phi', 'P'),
]
PARAMETERS = aliased.KeyMap(_PARAMETERS)
"""User-constrainable parameters that affect interpolation."""

