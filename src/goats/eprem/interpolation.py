import typing

import numpy as np
import numpy.typing
from scipy.interpolate import interp1d

from goats.core import numerical


def restrict_coordinate(
    array: np.ndarray,
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
        permuted = np.moveaxis(array, axis, 0)
        result = permuted[self.indices, ...]
        return np.moveaxis(result, 0, axis)


def apply(
    array: np.ndarray,
    reference: np.ndarray,
    targets: typing.Iterable[float],
    coordinate: str=None,
) -> np.ndarray:
    """Interpolate `array` to target values over `coordinate`."""
    interpolated = [
        _apply_interp1d(array, reference, target, coordinate=coordinate)
        for target in targets
    ]
    if reference.ndim == 2:
        return np.swapaxes(interpolated, 0, 1)
    return np.array(interpolated)


axes = {
    'time': 0,
    'radius': 1,
    'energy': 1,
    'mu': 0,
}


def _apply_interp1d(
    array: np.ndarray,
    reference: np.ndarray,
    target: float,
    coordinate: str=None,
) -> typing.List[float]:
    """Interpolate data to `target` along the leading axis."""
    if target in reference:
        idx = numerical.find_nearest(reference, target).index
        return array[idx]
    if (axis := axes.get(coordinate)) is not None:
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

