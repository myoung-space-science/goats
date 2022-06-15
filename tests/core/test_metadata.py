import numbers

import pytest

from goats.core import metadata


def test_types():
    """Test the collection of operand types."""
    types = metadata.Types()
    types.add(int, int)
    assert len(types) == 1
    assert (int, int) in types
    types.add(str, float)
    assert len(types) == 2
    assert (str, float) in types
    types.discard(int, int)
    assert len(types) == 1
    assert (str, float) in types
    types.clear()
    assert len(types) == 0
    types.add(int, float, symmetric=True)
    assert len(types) == 2
    assert (int, float) in types and (float, int) in types
    copied = types.copy()
    assert copied == types
    assert copied is not types
    types.clear()
    types.add(numbers.Real, numbers.Real)
    assert (numbers.Real, numbers.Real) in types
    assert (int, float) not in types
    assert types
    types.clear()
    assert not types
    types = metadata.Types()
    types.add(int)
    assert int in types


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
    operation.allow(float, int)
    assert operation.supports(float, int)
    assert not operation.supports(int, float)

