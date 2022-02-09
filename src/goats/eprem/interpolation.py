import typing

import numpy as np
import numpy.typing
from scipy.interpolate import interp1d

from goats.common.numerical import get_bounding_indices


def restrict_shells(r: np.ndarray, r0: float):
    """Restrict the full array of shells to those near `r`."""
    bounds = get_bounding_indices(r, r0, axis=1)
    lower = min(bounds[:, 0])
    upper = max(bounds[:, 1])
    return range(lower, upper+1)


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
        axis = axis or -1
        permuted = np.moveaxis(array, axis, 0)
        result = permuted[self.indices, ...]
        return np.moveaxis(result, 0, axis)


def apply(
    array: np.ndarray,
    reference: np.ndarray,
    target: typing.Iterable[float],
    coordinate: str=None,
) -> np.ndarray:
    """Interpolate `array` to target values over `coordinate`."""
    interpolated = [
        _apply_interp1d(array, reference, value, coordinate=coordinate)
        for value in target
    ]
    if reference.ndim == 2:
        return np.swapaxes(interpolated, 0, 1)
    return np.array(interpolated)


def _apply_interp1d(
    array: np.ndarray,
    reference: np.ndarray,
    target: float,
    coordinate: str=None,
) -> typing.List[float]:
    """Interpolate data to `target` along the leading axis."""
    if reference.ndim == 2:
        if coordinate:
            restriction = Restriction(restrict_shells, reference, target)
            ref = restriction.apply(reference, axis=1)
            arr = restriction.apply(array, axis=1)
        else:
            ref = reference
            arr = array
        interps = [interp1d(x, y, axis=0) for x, y in zip(ref, arr)]
        return [interp(target) for interp in interps]
    interp = interp1d(reference, array, axis=0)
    return interp(target)

