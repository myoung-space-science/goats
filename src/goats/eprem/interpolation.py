import typing

import numpy as np
from scipy.interpolate import interp1d

from goats.common.numerical import get_bounding_indices


def restrict_shells(r: np.ndarray, r0: float):
    """Restrict the full array of shells to those near `r`."""
    bounds = get_bounding_indices(r, r0, axis=1)
    lower = min(bounds[:, 0])
    upper = max(bounds[:, 1])
    return range(lower, upper+1)


def radius(
    array: np.ndarray,
    reference: np.ndarray,
    target: typing.Iterable[float],
) -> np.ndarray:
    """Interpolate the array to one or more radial targets."""
    interpolated = np.array([
        _interp_to_radius(array, reference, value)
        for value in target
    ])
    return np.swapaxes(interpolated, 0, 1)


def standard(
    array: np.ndarray,
    reference: np.ndarray,
    target: typing.Iterable[float],
) -> np.ndarray:
    """Interpolate the array over a standard EPREM coordinate."""
    result = np.array([
        _interp_to_coordinate(array, reference, value)
        for value in target
    ])
    if reference.ndim == 2:
        return np.swapaxes(result, 0, 1)
    return result


def _interp_to_coordinate(
    array: np.ndarray,
    reference: np.ndarray,
    target: float,
) -> typing.List[float]:
    """Interpolate data to a target coordinate value."""
    if reference.ndim == 2:
        interps = [
            interp1d(ref, arr, axis=0)
            for ref, arr in zip(reference, array)
        ]
        return [interp(target) for interp in interps]
    interp = interp1d(reference, array, axis=0)
    return interp(target)


def _interp_to_radius(
    array: np.ndarray,
    radius: np.ndarray,
    target: float,
) -> typing.List[float]:
    """Interpolate data to a single radius."""
    restricted = restrict_shells(radius, target)
    interps = [
        interp1d(x, y, axis=0)
        for x, y in zip(radius[:, restricted], array[:, restricted, ...])
    ]
    return [interp(target) for interp in interps]
