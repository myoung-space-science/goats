import numbers
import typing
import sys

import numpy
import numpy.typing


def get_bounding_indices(
    array: numpy.typing.ArrayLike,
    target: numbers.Real,
    axis: typing.SupportsIndex=None,
) -> numpy.ndarray:
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
    return numpy.apply_along_axis(
        find_1d_indices,
        int(axis),
        numpy.asfarray(array),
        float(target),
    )


def find_1d_indices(
    array: numpy.ndarray,
    target: float,
) -> typing.Tuple[int, int]:
    """Find the bounding indices in a 1-D array."""
    leq = array <= target
    lower = numpy.where(leq)[0].max() if any(leq) else 0
    geq = array >= target
    upper = numpy.where(geq)[0].min() if any(geq) else len(array)-1
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
        `bound`, if any) and the index of `value` in `values`. If the array
        corresponding to `values` is one-dimensional, `index` will be an
        integer; otherwise, it will be a tuple with one entry for each
        dimension.

    Notes
    -----
    This function is based on the top answer to this StackOverflow question:
    https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
    However, a lower-voted answer (and the comments) has some suggestions for a
    bisection-based method.
    """

    array = numpy.asarray(values)
    index = numpy.abs(array - target).argmin()
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
    if array.ndim > 1:
        index = numpy.unravel_index(index, array.shape)
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


class MissingArgsError(Exception):
    pass


class ArgsNumberError(Exception):
    pass


def xyz2rtp(*args):
    """Convert (x, y, z) to (r, θ, φ).

    This function treats θ as the polar angle and φ as the azimuthal angle.

    Positional Parameters
    ---------------------
    args : ``int``s or ``tuple`` of ``int``s
    
    The x, y, and z values to convert. The user may either pass individual values or a three-tuple of values.

    Returns
    -------
    ``tuple``
    
    A tuple containing the computed r, θ, and φ values.

    Examples
    --------
    ```
    >>> from goats.core.numerical import xyz2rtp
    >>> xyz2rtp(1.0, 0.0, 0.0)
    (1.0, 1.5707963267948966, 0.0)
    >>> xyz2rtp(0.0, 0.0, 1.0)
    (1.0, 0.0, 0.0)
    >>> xyz = (0.0, 0.0, 1.0)
    >>> xyz2rtp(xyz)
    (1.0, 0.0, 0.0)
    ```
    """

    if not args:
        raise MissingArgsError
    elif len(args) == 1:
        x, y, z = args[0]
    elif len(args) == 3:
        x, y, z = args
    else:
        raise ArgsNumberError
    r = numpy.sqrt(x*x + y*y + z*z)
    r[numpy.asarray(numpy.abs(r) < sys.float_info.epsilon).nonzero()] = 0.0
    t = numpy.arccos(z/r)
    t[numpy.asarray(r == 0).nonzero()] = 0.0
    p = numpy.arctan2(y, x)
    p[numpy.asarray(p < 0.0).nonzero()] += 2*numpy.pi
    p[numpy.asarray(
        [i == 0 and j >= 0 for (i, j) in zip(x, y)]
    ).nonzero()] = +0.5*numpy.pi
    p[numpy.asarray(
        [i == 0 and j < 0  for (i, j) in zip(x, y)]
    ).nonzero()] = -0.5*numpy.pi
    return (r, t, p)


def rtp2xyz(*args):
    """Convert (r, θ, φ) to (x, y, z).

    This function treats θ as the polar angle and φ as the azimuthal angle.

    Positional Parameters
    ---------------------
    args : ``int``s or ``tuple`` of ``int``s

    The r, θ, and φ values to convert. The user may either pass individual values or a three-tuple of values.

    Returns
    -------
    ``tuple``

    A tuple containing the computed x, y, and z values.

    Examples
    --------
    ```
    >>> from goats.core.numerical rtp2xyz
    >>> import numpy
    >>> rtp2xyz(1.0, 0.5*numpy.pi, 0)
    (1.0, 0.0, 0.0)
    >>> rtp2xyz(1.0, 0, 0.5*numpy.pi)
    (0.0, 0.0, 1.0)
    >>> rtp = (1.0, 0, 0.5*numpy.pi)
    >>> rtp2xyz(rtp)
    (0.0, 0.0, 1.0)
    ```
    """

    if not args:
        raise MissingArgsError
    elif len(args) == 1:
        r, t, p = args[0]
    elif len(args) == 3:
        r, t, p = args
    else:
        raise ArgsNumberError
    x = r * numpy.sin(t) * numpy.cos(p)
    x = zero_floor(x)
    y = r * numpy.sin(t) * numpy.sin(p)
    y = zero_floor(y)
    z = r * numpy.cos(t)
    z = zero_floor(z)
    return (x, y, z)


def zero_floor(
    value: typing.Union[float, numpy.ndarray],
) -> typing.Union[float, numpy.ndarray]:
    """Round a small number, or array of small numbers, to zero."""
    if value.shape:
        condition = numpy.asarray(
            numpy.abs(value) < sys.float_info.epsilon
        ).nonzero()
        value[condition] = 0.0
    else:
        value = 0.0 if numpy.abs(value) < sys.float_info.epsilon else value
    return value


