import numbers
import operator as standard

from goats.core import operator

def test_rule_comparison():
    """Test comparisons between operand-update rules."""
    rule = operator.Rule((numbers.Complex, numbers.Real))
    true = [
        [standard.ne, (complex, float)],
        [standard.gt, (complex, float)],
        [standard.ge, (complex, float)],
        [standard.gt, (float, float)],
        [standard.ge, (float, float)],
        [standard.gt, (float, int)],
        [standard.ge, (float, int)],
        [standard.gt, (numbers.Complex, numbers.Real)],
        [standard.ge, (numbers.Complex, numbers.Real)],
        [standard.eq, (numbers.Complex, numbers.Real)],
    ]
    for method, types in true:
        assert method(rule, types)
    false = [
        [standard.eq, (complex, float)],
        [standard.eq, (numbers.Real, numbers.Real)]
    ]
    for method, types in false:
        assert not method(rule, types)


def test_rule_len():
    """Test len(operator.Rule)."""
    assert len(operator.Rule(int)) == 1
    assert len(operator.Rule((int, float))) == 2


def test_rule_contains():
    """Test x `in` operator.Rule."""
    assert int in operator.Rule(int)
    assert float not in operator.Rule(int)
    assert int in operator.Rule((int, float))
    assert float in operator.Rule((int, float))

