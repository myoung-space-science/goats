import pytest
import fractions
import typing

from goats.core import algebraic


class Ordered(algebraic.Ordered):
    """Test class for ordered quantities."""

    def __init__(self, __value) -> None:
        self._value = __value

    def __lt__(self, other) -> bool:
        return self._value < other

    def __eq__(self, other) -> bool:
        return self._value == other

    def __le__(self, other) -> bool:
        return super().__le__(other)

    def __ge__(self, other) -> bool:
        return super().__ge__(other)

    def __gt__(self, other) -> bool:
        return super().__gt__(other)

    def __ne__(self, other) -> bool:
        return super().__ne__(other)

    def __bool__(self) -> bool:
        return self._value != 0


def test_ordered():
    """Test a concrete version of the ordered type."""
    this = Ordered(2)
    assert this == 2
    assert this != 1
    assert this < 3
    assert this <= 2
    assert this <= 3
    assert this > 1
    assert this >= 2
    assert this >= 1


@pytest.mark.term
def test_create_term():
    """Create variable and constant algebraic terms."""
    valid = {
        '1': {'coefficient': 1, 'base': '1', 'exponent': 1},
        '2': {'coefficient': 2, 'base': '1', 'exponent': 1},
        '2^3': {'coefficient': 8, 'base': '1', 'exponent': 1},
        '2^-3': {'coefficient': 1/8, 'base': '1', 'exponent': 1},
        'a': {'coefficient': 1, 'base': 'a', 'exponent': 1},
        'a_b': {'coefficient': 1, 'base': 'a_b', 'exponent': 1},
        'a^2': {'coefficient': 1, 'base': 'a', 'exponent': 2},
        'a^3/2': {'coefficient': 1, 'base': 'a', 'exponent': '3/2'},
        '4a': {'coefficient': 4, 'base': 'a', 'exponent': 1},
        '-4a^3': {'coefficient': -4, 'base': 'a', 'exponent': 3},
        '4a^-3': {'coefficient': 4, 'base': 'a', 'exponent': -3},
        '4a^3/2': {'coefficient': 4, 'base': 'a', 'exponent': '3/2'},
        '4a^+3/2': {'coefficient': 4, 'base': 'a', 'exponent': '3/2'},
        '4a^-3/2': {'coefficient': 4, 'base': 'a', 'exponent': '-3/2'},
        '4a^1.5': {'coefficient': 4, 'base': 'a', 'exponent': 1.5},
        '4a^+1.5': {'coefficient': 4, 'base': 'a', 'exponent': 1.5},
        '4a^-1.5': {'coefficient': 4, 'base': 'a', 'exponent': -1.5},
        '4.1a^3/2': {'coefficient': 4.1, 'base': 'a', 'exponent': '3/2'},
        '4ab^3/2': {'coefficient': 4, 'base': 'ab', 'exponent': '3/2'},
        '4b0^3/2': {'coefficient': 4, 'base': 'b0', 'exponent': '3/2'},
    }
    for string, expected in valid.items():
        term = algebraic.OperandFactory().create(string)
        assert isinstance(term, algebraic.Term)
        assert float(term.coefficient) == float(expected['coefficient'])
        assert term.base == expected['base']
        assert term.exponent == fractions.Fraction(expected['exponent'])
        assert term == string # Probably should be separate test
    invalid = [
        '^3', # exponent only
        'a^', # missing exponent
    ]
    for string in invalid:
        with pytest.raises(algebraic.OperandValueError):
            term = algebraic.OperandFactory().create(string)


@pytest.mark.term
def test_simple_term_operators():
    """Test allowed arithmetic operations on a simple algebraic term."""
    x = algebraic.OperandFactory().create('x')
    assert isinstance(x, algebraic.Term)
    assert x**2 == algebraic.Term(1, 'x', 2)
    assert 3 * x == algebraic.Term(3, 'x', 1)
    assert x * 3 == algebraic.Term(3, 'x', 1)
    assert (3 * x) ** 2 == algebraic.Term(9, 'x', 2)
    y = algebraic.OperandFactory().create('y')
    assert isinstance(y, algebraic.Term)
    y *= 2.5
    assert y == algebraic.Term(2.5, 'y', 1)
    z = algebraic.OperandFactory().create('z')
    assert isinstance(z, algebraic.Term)
    z **= -3
    assert z == algebraic.Term(1, 'z', -3)


@pytest.mark.term
def test_term_cast():
    """Test the ability to cast constant terms to `int` or `float`."""
    constant = algebraic.Term(2, '1', 1)
    assert int(constant) == 2
    assert float(constant) == 2.0


@pytest.mark.term
def test_term_evaluate():
    """Test the ability to evaluate a variable term."""
    variable = algebraic.Term(3, 'a', 2)
    values = [4, -7, 11.3]
    for value in values:
        new = variable(value)
        assert isinstance(new, algebraic.Term)
        assert float(new) == 3 * (value ** 2)
    badtypes = [
        '2.7',
        algebraic.Term(2.7),
        algebraic.Expression('2.7'),
    ]
    for badtype in badtypes:
        with pytest.raises(TypeError):
            variable(badtype)
    with pytest.raises(TypeError):
        constant = algebraic.Term(9)
        constant(2)


def test_create_operand():
    """Test the object representing a part of an expression."""
    cases = {
        (4, '1', 1): [['4'], [1, '4', 1]],
        (1, 'a', 1): [['a']],
        (2, 'a', 1): [['2a'], [1, '2a', 1]],
        (2, 'a', 3): [['2a^3']],
        (2**3, 'a', 3): [['(2a)^3'], [1, '2a', 3]],
        (2**(3/4), 'a', '3/4'): [['(2a)^3/4'], [1, '2a', '3/4']],
        (4 * 2**3, 'a', 3): [['4(2a)^3'], [4, '2a', 3]],
        (2, 'a', -1): [['2a^-1']],
        (1, '2a * b', 3): [['(2a * b)^3']],
        (3, 'a * b', -2): [['3(a * b)^-2']],
        (1, 'a / (b * c)', 1): [['a / (b * c)']],
    }
    for ref, group in cases.items():
        from_ref = algebraic.OperandFactory().create(*ref)
        for args in group:
            from_args = algebraic.OperandFactory().create(*args)
            assert from_ref == from_args
            for part in [from_ref, from_args]:
                assert part.coefficient == ref[0]
                assert part.base == ref[1]
                assert part.exponent == fractions.Fraction(ref[2])


def test_find_bounds():
    """Test OperandFactory.find_bounds."""
    strings = {
        '(a*b)': [(0, 5), '(a*b)'],
        '(a*b)^2': [(0, 5), '(a*b)'],
        '3(a*b)': [(1, 6), '(a*b)'],
        '3(a*b)^2': [(1, 6), '(a*b)'],
        '3(a*b)^2 * (c*d)': [(1, 6), '(a*b)'],
        '4a^4': [None, '4a^4'],
        '3(4a^4)^3': [(1, 7), '(4a^4)'],
        '2(3(4a^4)^3)^2': [(1, 12), '(3(4a^4)^3)'],
    }
    operand = algebraic.OperandFactory()
    for string, (bounds, substring) in strings.items():
        found = operand.find_bounds(string)
        assert found == bounds
        if found:
            start, end = bounds
            assert string[start:end] == substring


def test_strip_separators():
    """Test OperandFactory.strip_separators."""
    strings = {
        '(a*b)': 'a*b',
        '((a*b))': '(a*b)',
        '(a*b)^2': '(a*b)^2',
        '3(a*b)': '3(a*b)',
        '((a^2 / b) / (c^4 / d))': '(a^2 / b) / (c^4 / d)',
    }
    operand = algebraic.OperandFactory()
    for string, stripped in strings.items():
        assert operand.strip_separators(string) == stripped


@pytest.mark.expression
def test_expression_parser():
    """Test the algebraic-expression parser."""
    cases = {
        'a / b': {
            'terms': ['a', 'b^-1'],
        },
        '1 / b': {
            'terms': ['b^-1'],
        },
        'a / (b * c)': {
            'terms': ['a', 'b^-1', 'c^-1'],
        },
        'a / (bc)': {
            'terms': ['a', 'bc^-1'],
        },
        'a / bc': {
            'terms': ['a', 'bc^-1'],
        },
        'a * b / c': {
            'terms': ['a', 'b', 'c^-1'],
        },
        '(a / b) / c': {
            'terms': ['a', 'b^-1', 'c^-1'],
        },
        '(a / b) / (c / d)': {
            'terms': ['a', 'b^-1', 'c^-1', 'd'],
        },
        '(a * b / c) / (d * e / f)': {
            'terms': ['a', 'b', 'c^-1', 'd^-1', 'e^-1', 'f'],
        },
        'a^2 / b^3': {
            'terms': ['a^2', 'b^-3'],
        },
        '(a^2 / b)^5 / (c^4 / d)^3': {
            'terms': ['a^10', 'b^-5', 'c^-12', 'd^3'],
        },
        '((a^2 / b) / (c^4 / d))^3': {
            'terms': ['a^6', 'b^-3', 'c^-12', 'd^3'],
        },
        'a^-2': {
            'terms': ['a^-2'],
        },
        'a^-3 / b^-6': {
            'terms': ['a^-3', 'b^6'],
        },
        '(a * (b * c))': {
            'terms': ['a', 'b', 'c'],
        },
        '(a * (b * c))^2': {
            'terms': ['a^2', 'b^2', 'c^2'],
        },
        '(a * (b * c)^2)': {
            'terms': ['a', 'b^2', 'c^2'],
        },
        '(a / (b * c)^2)': {
            'terms': ['a', 'b^-2', 'c^-2'],
        },
        'a / (b * c * (d / e))': {
            'terms': ['a', 'b^-1', 'c^-1', 'd^-1', 'e'],
        },
        'a0^2 * (a1*a2) / (a3 * a4^2 * (a5/a6))': {
            'terms': ['a0^2', 'a1', 'a2', 'a3^-1', 'a4^-2', 'a5^-1', 'a6'],
        },
        '((a^2 * b^3) / c) * (d^-3)': {
            'terms': ['a^2', 'b^3', 'c^-1', 'd^-3'],
        },
        '3a * b': {
            'terms': ['a', 'b', '3'],
        },
        '3(a * b)': {
            'terms': ['a', 'b', '3'],
        },
        '3a / b': {
            'terms': ['a', 'b^-1', '3'],
        },
        '3(a / b)': {
            'terms': ['a', 'b^-1', '3'],
        },
        'a / (2.5 * 4.0)': {
            'terms': ['a', '0.1'],
        },
        'a / (2.5b * 4.0)': {
            'terms': ['a', 'b^-1', '0.1'],
        },
        'a / ((2.5 * 4.0) * b)': {
            'terms': ['a', 'b^-1', '0.1'],
        },
        'a / (2.5 * 4.0 * b)': {
            'terms': ['a', 'b^-1', '0.1'],
        },
        'a': {
            'terms': ['a'],
        },
        '2a': {
            'terms': ['a', '2'],
        },
        '2a^3': {
            'terms': ['a^3', '2'],
        },
        '1': {
            'terms': ['1'],
        },
        '2^4': {
            'terms': ['2^4'],
        },
        '2^-1': {
            'terms': ['2^-1'],
        },
        '0.5a': {
            'terms': ['a', '0.5'],
        },
        '(a*b)^2': {
            'terms': ['a^2', 'b^2'],
        },
        '3(a*b)^2': {
            'terms': ['a^2', 'b^2', '3'],
        },
        '3(a*2b)^2': {
            'terms': ['a^2', 'b^2', '12'],
        },
        '3(a*2b)^2 / 6c^3': {
            'terms': ['a^2', 'b^2', 'c^-3', '2'],
        },
        '2 * 4a^5 2(2a^3)^6 * (3b * c)^-1': {
            'terms': ['a^23', 'b^-1', 'c^-1', '1024/3'],
        },
        '(3)': {
            'terms': ['3'],
        },
        '(3) * (2a)': {
            'terms': ['a', '6'],
        },
        '4a^4': {
            'terms': ['a^4', '4'],
        },
        '3(4a^4)^3': {
            'terms': ['a^12', '192'],
        },
        '2(3(4a^4)^3)^2': {
            'terms': ['a^24', '73728'],
        },
        '1.0e2': {
            'terms': ['100'],
        }
    }
    for test, expected in cases.items():
        expression = algebraic.Expression(test)
        terms = algebraic.asterms(expected['terms'])
        assert equal_terms(expression, terms)


@pytest.mark.expression
def test_space_multiplies():
    """Test the option to allow whitespace to represent multiplication."""
    cases = {
        'a^2 * b^-2': ['a^2', 'b^-2'],
        'a^2 b^-2': ['a^2', 'b^-2'],
        '(a b^-2) / (c^3 d)': ['a^1', 'b^-2', 'c^-3', 'd^-1'],
    }
    for test, strings in cases.items():
        expression = algebraic.Expression(test)
        expected = [
            algebraic.OperandFactory().create(string)
            for string in strings
        ]
        assert equal_terms(expression, expected)


@pytest.mark.expression
def test_init_collection():
    """Test the ability to initialize an expression from a collection."""
    cases = {
        'a / b': ['a', 'b^-1'],
        'a / (b * c * (d / e))': ['a', 'b^-1', 'c^-1', 'd^-1', 'e'],
        'a0^2 * (a1*a2) / (a3 * a4^2 * (a5/a6))': [
            'a0^2', 'a1', 'a2', 'a3^-1', 'a4^-2', 'a5^-1', 'a6',
        ],
        'a0 * (a1 / a2) * (a3 / a4)': ['a0', 'a1 / a2', 'a3 / a4'],
    }
    for string, terms in cases.items():
        assert algebraic.Expression(string) == algebraic.Expression(terms)
        parts = [algebraic.OperandFactory().create(term) for term in terms]
        assert algebraic.Expression(string) == algebraic.Expression(parts)


@pytest.mark.expression
def test_parse_function():
    """Test the ability to detect a known function in an expression."""
    cases = {
        'sqrt(2a)': ['a^1/2', '2^1/2'],
        ' sqrt(2a) ': ['a^1/2', '2^1/2'],
    }
    for string, terms in cases.items():
        expression = algebraic.Expression(string)
        expected = algebraic.asterms(terms)
        assert expression.terms == expected


@pytest.mark.expression
def test_parser_operators():
    """Test the algebraic parser with non-standard operators."""
    expression = algebraic.Expression('a @ b^2 $ c', multiply='@', divide='$')
    expected = [
        algebraic.OperandFactory().create(term)
        for term in ('a', 'b^2', 'c^-1')
    ]
    assert equal_terms(expression, expected)


@pytest.mark.expression
def test_parser_separators():
    """Test the algebraic parser with non-standard separators."""
    expression = algebraic.Expression('a / [b * c]^2', opening='[', closing=']')
    expected = [
        algebraic.OperandFactory().create(term)
        for term in ('a', 'b^-2', 'c^-2')
    ]
    assert equal_terms(expression, expected)


@pytest.mark.skip(reason="Requires significant refactoring")
@pytest.mark.expression
def test_nonstandard_chars():
    """Test the ability to include non-standard characters in terms."""
    string = '<r> * cm^2 / (<flux> / nuc)'
    expression = algebraic.Expression(string)
    expected = [
        algebraic.Term(1, '<r>', 1),
        algebraic.Term(1, 'cm', 2),
        algebraic.Term(1, '<flux>', -1),
        algebraic.Term(1, 'nuc', 1),
    ]
    assert equal_terms(expression, expected)


@pytest.mark.expression
def test_parsing_errors():
    """Make sure the parser correctly catches errors."""
    with pytest.raises(algebraic.RatioError):
        algebraic.Expression('a/b/c', operator_order='error')
    with pytest.raises(algebraic.ProductError):
        algebraic.Expression('a/b*c', operator_order='error')
    invalid = [
        '(a*b))',
        'a / (a*b',
        'a/',
    ]
    for string in invalid:
        with pytest.raises(algebraic.ParsingValueError):
            algebraic.Expression(string)


@pytest.mark.expression
def test_ignore_operator_order():
    """Test the option to ignore operator order."""
    cases = {
        'a/b/c': ['a', 'b^-1', 'c^-1'],
        'a/b*c': ['a', 'b^-1', 'c'],
    }
    for string, terms in cases.items():
        expression = algebraic.Expression(string, operator_order='ignore')
        expected = algebraic.asterms(terms)
        assert expression.terms == expected

@pytest.mark.expression
def test_formatted_expression():
    """Test the ability to format terms in an algebraic expression."""
    string = 'a0^2 * (a1*a2) / (a3 * a4^2 * (a5/a6))'
    expresssion = algebraic.Expression(string)
    terms = ['a0^2', 'a1', 'a2', 'a3^-1', 'a4^-2', 'a5^-1', 'a6']
    formatted = expresssion.format()
    assert all(term in formatted for term in terms)
    terms = ['a0^{2}', 'a1', 'a2', 'a3^{-1}', 'a4^{-2}', 'a5^{-1}', 'a6']
    formatted = expresssion.format(style='tex')
    assert all(term in formatted for term in terms)


@pytest.mark.expression
def test_expression_equality():
    """Test the definition of equality between two expressions."""
    cases = {
        ('a * b^2', 'a * b^2'): True,
        ('a * b^2', 'b^2 * a'): True,
        ('a * b^2', 'a / b^-2'): True,
        ('a * b^2', '(a * b^2)^1'): True,
        ('a^2 * b^4', '(a * b^2)^2'): True,
        ('a * b^2', 'a * b'): False,
        ('a * b^2', 'a * c^2'): False,
        ('a * b^2', 'a / b^2'): False,
    }
    for args, result in cases.items():
        expressions = [algebraic.Expression(arg) for arg in args]
        for arg, expression in zip(args, expressions):
            assert arg == expression
        assert (args[0] == expressions[1]) == result
        assert (args[1] == expressions[0]) == result
        assert (expressions[0] == expressions[1]) == result


@pytest.mark.expression
def test_expression_algebra():
    """Test algebraic operations between expressions."""
    strings = [
        'a',
        'a * b^2',
        '(c / b)^3',
        'b^-3 * d / a^2',
        'a / b^2',
        'b^2 / a',
        '3a',
    ]
    expressions = [algebraic.Expression(string) for string in strings]

    # Test left and right multiplication and division
    e2 = expressions[2]
    for e1 in (strings[1], expressions[1]):
        expected = algebraic.Expression('a * b^-1 * c^3')
        assert e1 * e2 == expected
        assert e2 * e1 == expected
        expected = algebraic.Expression('a * b^5 * c^-3')
        assert e1 / e2 == expected
        expected = algebraic.Expression('a^-1 * b^-5 * c^3')
        assert e2 / e1 == expected

    # Include a negative exponent
    e2 = expressions[3]
    expected = algebraic.Expression('a^-1 * b^-1 * d')
    for e1 in (strings[1], expressions[1]):
        assert e1 * e2 == expected

    # Identically cancel one of the terms
    e2 = expressions[4]
    expected = algebraic.Expression('a^2')
    for e1 in (strings[1], expressions[1]):
        assert e1 * e2 == expected

    # Identically cancel all terms
    e2 = expressions[5]
    expected = algebraic.Expression('1')
    for e1 in (strings[4], expressions[4]):
        assert e1 * e2 == expected
        assert e2 * e1 == expected

    # Test exponentiation. The second case checks that the previous operation
    # didn't change the term in the expression, which was originally a bug.
    e = expressions[0]
    expected = algebraic.Expression('a^2')
    assert e ** 2 == expected
    expected = algebraic.Expression('a^3')
    assert e ** 3 == expected

    # Test in-place exponentiation
    e = expressions[0]
    e **= 3
    expected = algebraic.Expression('a^3')
    assert e == expected

    # Test exponentiation with multiple terms
    assert expressions[2] ** 3 == algebraic.Expression('c^9 * b^-9')

    # Test exponentiation to a negative power
    assert expressions[3] ** -1 == algebraic.Expression('b^3 * d^-1 * a^2')

    # Test multiplication with a coefficient
    assert expressions[0] * expressions[6] == algebraic.Expression('3a^2')
    assert expressions[6] ** 2 == algebraic.Expression('9a^2')


@pytest.mark.expression
def test_algebra_with_conversion():
    """Test algebraic operations that require conversion to an expression."""
    expr = algebraic.Expression('a^2 * b / c^3')
    expected = algebraic.Expression('a * b^3 / (c^2 * d)')
    assert expr * 'b^2 * c / (a * d)' == expected
    assert expr / 'a * d / (b^2 * c)' == expected


@pytest.mark.expression
def test_reduced_expression():
    """An expression should be equal to its reduced version."""
    expression = algebraic.Expression('a^2 * b / (a * b^3 * c)')
    reduced = algebraic.Expression('a * b^-2 * c^-1')
    assert expression == reduced


def equal_terms(
    expression: algebraic.Expression,
    terms: typing.Iterable[algebraic.Term],
) -> bool:
    """True if an expression's terms equal an iterable of terms."""
    if len(expression.terms) != len(terms):
        return False
    return all(term in expression.terms for term in terms)


@pytest.mark.expression
def test_expression_index():
    """Users should be able to access terms via index notation."""
    expression = algebraic.Expression('a * b^-2 * c^-1')
    terms = [
        algebraic.Term(1, 'a', 1),
        algebraic.Term(1, 'b', -2),
        algebraic.Term(1, 'c', -1),
    ]
    assert expression[0] == terms[0]
    assert expression[:] == terms[:]
    assert expression[:2] == terms[:2]
    assert expression[1:2] == terms[1:2]
    assert expression[-1] == terms[-1]

