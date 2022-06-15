import numbers

import pytest

from goats.core import metadata


def test_types():
    """Test the collection of operand types."""
    types = metadata.Types()
    assert types.ntypes is None
    types.add(int, int)
    assert types.ntypes == 2
    assert len(types) == 1
    assert (int, int) in types
    types.add(str, float)
    assert types.ntypes == 2
    assert len(types) == 2
    assert (str, float) in types
    types.discard(int, int)
    assert types.ntypes == 2
    assert len(types) == 1
    assert (str, float) in types
    types.clear()
    assert types.ntypes is 2
    assert len(types) == 0
    assert not types
    types.add(numbers.Real, numbers.Real)
    assert (numbers.Real, numbers.Real) in types
    assert (int, float) not in types
    assert types
    types = metadata.Types()
    assert types.ntypes is None
    types.add(int)
    assert types.ntypes == 1
    assert int in types


def test_types_copy():
    """Test the ability to make a copy of a `Types` instance."""
    types = metadata.Types()
    types.add(int, str)
    copied = types.copy()
    assert copied == types
    assert copied is not types


def test_types_add_batch():
    """Test the ability to add multiple type specifications."""
    types = metadata.Types()
    assert len(types) == 0
    user = [
        (int, float),
        (float, float),
        (int, int),
    ]
    types.add(*user)
    assert len(types) == 3
    for these in user:
        assert these in types


def test_types_implied():
    """Test the implied type specification."""
    types = metadata.Types(implied=str)
    assert str in types
    assert (str, str) in types
    assert (str, str, str) in types
    assert (str, float) not in types
    types.add(int, float)
    assert str not in types
    assert (str, str) in types
    assert (str, str, str) not in types
    assert (str, float) not in types
    types.discard(str, str)
    assert (str, str) not in types


def test_operation():
    """Test the general operational context."""
    operation = metadata.Operation()
    operation.support(float, int)
    assert operation.supports(float, int)
    assert not operation.supports(int, float)
    operation.suppress(float, int)
    assert not operation.supports(float, int)
    operation.support(complex, float, symmetric=True)
    assert operation.support(complex, float)
    assert operation.support(float, complex)
    operation.suppress(complex, float, symmetric=True)
    assert not operation.supports(complex, float)
    assert not operation.supports(float, complex)

