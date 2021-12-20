import numbers
from typing import *

import numpy as np


def get_bounding_indices(
    array: np.ndarray,
    target: float,
    axis: int=0,
) -> np.ndarray:
    """Find the indices bounding the target value."""
    return np.apply_along_axis(find_1d_indices, axis, array, target)


def find_1d_indices(
    array: np.ndarray,
    target: float,
) -> Tuple[int, int]:
    """Find the bounding indices in a 1-D array."""
    leq = array <= target
    lower = np.where(leq)[0].max() if any(leq) else 0
    geq = array >= target
    upper = np.where(geq)[0].min() if any(geq) else len(array)-1
    return lower, upper


_NT = TypeVar('_NT', bound=numbers.Complex)


class Nearest(NamedTuple):
    """The result of searching an array for a target value."""

    index: int
    value: _NT


def find_nearest(
    values: Iterable[_NT],
    target: _NT,
    bound: str=None,
) -> Nearest:
    """Find the value in a collection nearest the target value.
    
    Parameters
    ----------
    values : iterable of numbers
        An iterable collection of numbers to compare to the target value. Must
        support conversion to a `numpy.ndarray`.

    target : number
        A single numerical value for which to search in `values`. Must be
        coercible to the type of `values`.

    bound : {None, 'lower', 'upper'}
        The constraint to apply when finding the nearest value::
        - None: no constraint
        - 'lower': ensure that the nearest value is equal to or greater than the
        target value
        - 'upper': ensure that the nearest value is equal to or less than the
        target value

    Returns
    -------
    Nearest
        A named tuple with `value` and `index` fields, respectively containing
        the value in `values` closest to `target` (given the constraint set by
        `bound`, if any) and the index of `value` in `values`.

    Notes
    -----
    This function is based on the top answer to this StackOverflow question:
    https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
    However, a lower-voted answer (and the comments) has some suggestions for a
    bisection-based method.

    """

    array = np.asarray(values).squeeze()
    index = np.abs(array - target).argmin()
    if bound == 'lower':
        try:
            while array[index] < target:
                index += 1
        except IndexError:
            index = -1
    elif bound == 'upper':
        try:
            while array[index] > target:
                index -= 1
        except IndexError:
            index = 0
    return Nearest(index=index, value=array[index])

