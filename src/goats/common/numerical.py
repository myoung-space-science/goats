import numbers
import typing

import numpy as np
import numpy.typing


def get_bounding_indices(
    array: numpy.typing.ArrayLike,
    target: numbers.Real,
    axis: typing.SupportsIndex=None,
) -> np.ndarray:
    """Find the indices bounding the target value.
    
    Parameters
    ----------
    array : array-like
        The array containing `target`.

    target : real
        The numerical value in `array` whose index to bound.

    axis : integral, default=-1
        The axis in `array` along which to bound `target`.

    Returns
    -------
    `numpy.ndarray`
        An array containing the indices bounding `target` at each slice along
        `axis`. The shape will be the same as that of `array` except for `axis`,
        which will be 2.
    """
    if axis is None:
        axis = -1
    return np.apply_along_axis(
        find_1d_indices,
        int(axis),
        np.asfarray(array),
        float(target),
    )


def find_1d_indices(
    array: np.ndarray,
    target: float,
) -> typing.Tuple[int, int]:
    """Find the bounding indices in a 1-D array."""
    leq = array <= target
    lower = np.where(leq)[0].max() if any(leq) else 0
    geq = array >= target
    upper = np.where(geq)[0].min() if any(geq) else len(array)-1
    return lower, upper


_NT = typing.TypeVar('_NT', bound=numbers.Complex)


class Nearest(typing.NamedTuple):
    """The result of searching an array for a target value."""

    index: int
    value: _NT


def find_nearest(
    values: typing.Iterable[_NT],
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


_T = typing.TypeVar('_T', bound=numbers.Number)

def cast(
    arg: _T,
    strict: bool=True,
) -> typing.Union[int, float, complex, _T]:
    """Attempt to convert `arg` to an appropriate numeric type.
    
    Parameters
    ----------
    arg
        The object to convert. May be of any type. If it has a numeric type,
        this function will immediately return it.

    strict : bool, default=True
        If true (default), this function will raise an exception if it can't
        convert `arg` to a numerical type. If false, this function will silently
        return upon failure to convert.

    Returns
    -------
    number
        The numerical value of `arg`, if possible. See description of `strict`
        for behavior after failed conversion.
    """
    if isinstance(arg, numbers.Number):
        return arg
    types = (
        int,
        float,
        complex,
    )
    for t in types:
        try:
            return t(arg)
        except ValueError:
            pass
    if strict:
        raise TypeError(f"Can't convert {arg!r} to a number") from None


