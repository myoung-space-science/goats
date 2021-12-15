import abc
import inspect
import numbers
from typing import *

import numpy as np

from heroes.common.iterables import CollectionMixin
from heroes.common.units import Unit


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
def find_nearest(
    array: Sequence[_NT],
    value: _NT,
    constraint: str=None,
) -> Tuple[int, _NT]:
    """Find the array value nearest the target value.
    
    Returns a 2-tuple whose first element is the array index of the nearest value and second element is the nearest value.

    This function is based on the top answer to this StackOverflow question: 
    https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
    However, a lower-voted answer (and the comments) has some suggestions for a bisection-based method.

    Pass `constraint='lower'` to ensure that the nearest value is equal to or greater than the target value.
    Pass `constraint='upper'` to ensure that the nearest value is equal to or less than the target value.
    """

    array = np.asarray(array)
    array = np.squeeze(array)
    index = np.abs(array - value).argmin()
    if constraint == 'lower':
        try:
            while array[index] < value:
                index += 1
        except IndexError:
            index = -1
    elif constraint == 'upper':
        try:
            while array[index] > value:
                index -= 1
        except IndexError:
            index = 0
    return index, array[index]

