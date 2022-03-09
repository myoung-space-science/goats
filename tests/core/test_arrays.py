import numpy as np

from goats.core import arrays


def test_array():
    """Test an implementation of the array object."""
    values = [[1, 2, 3], [10, 20, 30]]
    axes = ('x', 'y')
    array = arrays.Array(values, axes)
    assert isinstance(array, arrays.Array)
    assert isinstance(array[:], np.ndarray)
    assert np.array_equal(array, values)
    assert array.axes == axes
    v_p1 = array + 1.0
    assert isinstance(v_p1, arrays.Array)
    assert np.array_equal(v_p1, [[2, 3, 4], [11, 21, 31]])
    v_avg = np.mean(array)
    assert v_avg == 11.0
    v_avg_d0 = np.mean(array, axis=0)
    assert isinstance(v_avg_d0, arrays.Array)
    assert v_avg_d0.axes == (axes[1],)
