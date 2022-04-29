import numbers
import operator as standard

from goats.core import operations


def test_rule_comparison():
    """Test comparisons between operand-update rules."""
    rule = operations.Rule((numbers.Complex, numbers.Real))
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
    assert len(operations.Rule(int)) == 1
    assert len(operations.Rule((int, float))) == 2


def test_rule_contains():
    """Test x `in` operator.Rule."""
    assert int in operations.Rule(int)
    assert float not in operations.Rule(int)
    assert int in operations.Rule((int, float))
    assert float in operations.Rule((int, float))


def test_rule_parameters():
    """Check updated and fixed parameters."""
    rule = operations.Rule(int, 'a', 'b', 'c')
    assert rule.updated == ('a', 'b', 'c')
    assert rule.ignored == ()
    rule.remove('b')
    assert rule.updated == ('a', 'c')
    assert rule.ignored == ('b',)


def test_rule_suppress():
    """Allow users to suppress an operator for a given operand rule."""
    rule = operations.Rule(int, 'a', 'b')
    assert rule.updated == ('a', 'b')
    assert rule.ignored == ()
    rule.suppress
    assert rule.updated == ()
    assert rule.ignored == ('a', 'b')


def test_rule_validate():
    """"""


def test_operator():
    """"""
    operator = operations.Operator(standard.add)
    assert operator.evaluate(3, 4) == 7

