import pytest
import fractions

from goats.common import algebra


def test_term():
    """Test the object representing an algebraic term."""
    valid = {
        '1': {'coefficient': 1, 'variable': '1', 'exponent': 1},
        'a': {'coefficient': 1, 'variable': 'a', 'exponent': 1},
        'a^2': {'coefficient': 1, 'variable': 'a', 'exponent': 2},
        'a^3/2': {'coefficient': 1, 'variable': 'a', 'exponent': '3/2'},
        '4a': {'coefficient': 4, 'variable': 'a', 'exponent': 1},
        '-4a^3': {'coefficient': -4, 'variable': 'a', 'exponent': 3},
        '4a^-3': {'coefficient': 4, 'variable': 'a', 'exponent': -3},
        '4a^3/2': {'coefficient': 4, 'variable': 'a', 'exponent': '3/2'},
        '4a^+3/2': {'coefficient': 4, 'variable': 'a', 'exponent': '3/2'},
        '4a^-3/2': {'coefficient': 4, 'variable': 'a', 'exponent': '-3/2'},
        '4a^1.5': {'coefficient': 4, 'variable': 'a', 'exponent': 1.5},
        '4a^+1.5': {'coefficient': 4, 'variable': 'a', 'exponent': 1.5},
        '4a^-1.5': {'coefficient': 4, 'variable': 'a', 'exponent': -1.5},
        '4.1a^3/2': {'coefficient': 4.1, 'variable': 'a', 'exponent': '3/2'},
        '4ab^3/2': {'coefficient': 4, 'variable': 'ab', 'exponent': '3/2'},
        '4b0^3/2': {'coefficient': 4, 'variable': 'b0', 'exponent': '3/2'},
    }
    for string, expected in valid.items():
        term = algebra.Term(string)
        assert term.coefficient == float(expected['coefficient'])
        assert term.variable == expected['variable']
        assert term.exponent == fractions.Fraction(expected['exponent'])
    invalid = [
        '2', # no variable
        '^3', # exponent only
        'a^', # missing exponent

    ]
    for string in invalid:
        with pytest.raises(algebra.TermValueError):
            algebra.Term(string)


def test_term_operators():
    """Test allowed arithmetic operations on an algebraic term."""
    x = algebra.Term('x')
    assert x**2 == algebra.Term('x^2')
    assert 3 * x == algebra.Term('3x')
    assert x * 3 == algebra.Term('3x')
    y = algebra.Term('y')
    y *= 2.5
    assert y == algebra.Term('2.5y')
    z = algebra.Term('z')
    z **= -3
    assert z == algebra.Term('z^-3')


def test_term_idempotence():
    """Make sure we can initialize a term object with an existing instance."""
    term = algebra.Term('a^3')
    assert algebra.Term(term) == term
    assert algebra.Term(term)**2 == algebra.Term('a^6')


def test_component_issimple():
    """Test the check for a 'simple' expression component."""
    cases = {
        '': False,
        'a': True,
        'a^2': True,
        'a^2/3': True,
        '3a^2': True,
        'a * b^2': False,
    }
    for string, expected in cases.items():
        term = algebra.Component(string)
        assert term.issimple == expected


def test_expression_parser():
    """Test the algebraic-expression parser."""
    cases = {
        'a / b': ['a', 'b^-1'],
        '1 / b': ['b^-1'],
        'a / (b * c)': ['a', 'b^-1', 'c^-1'],
        'a / (bc)': ['a', 'bc^-1'],
        'a / bc': ['a', 'bc^-1'],
        'a * b / c': ['a', 'b', 'c^-1'],
        '(a / b) / c': ['a', 'b^-1', 'c^-1'],
        '(a / b) / (c / d)': ['a', 'b^-1', 'c^-1', 'd'],
        '(a * b / c) / (d * e / f)': ['a', 'b', 'c^-1', 'd^-1', 'e^-1', 'f'],
        'a^2 / b^3': ['a^2', 'b^-3'],
        '(a^2 / b)^5 / (c^4 / d)^3': [ 'a^10', 'b^-5', 'c^-12', 'd^3'],
        '((a^2 / b) / (c^4 / d))^3': [ 'a^6', 'b^-3', 'c^-12', 'd^3'],
        'a^-2': ['a^-2', ],
        'a^-3 / b^-6': ['a^-3', 'b^6'],
        '(a * (b * c))': ['a', 'b', 'c'],
        '(a * (b * c))^2': ['a^2', 'b^2', 'c^2'],
        '(a * (b * c)^2)': ['a', 'b^2', 'c^2'],
        '(a / (b * c)^2)': ['a', 'b^-2', 'c^-2'],
        'a / (b * c * (d / e))': [ 'a', 'b^-1', 'c^-1', 'd^-1', 'e'],
        'a0^2 * (a1*a2) / (a3 * a4^2 * (a5/a6))': [
            'a0^2', 'a1', 'a2', 'a3^-1', 'a4^-2', 'a5^-1', 'a6',
        ],
        '((a^2 * b^3) / c) * (d^-3)': [ 'a^2', 'b^3', 'c^-1', 'd^-3'],
    }
    for test, strings in cases.items():
        expression = algebra.Expression(test)
        expected = [algebra.Term(string) for string in strings]
        assert expression.terms == expected


def test_space_multiplies():
    """Test the option to allow whitespace to represent multiplication."""
    cases = {
        'a^2 * b^-2': ['a^2', 'b^-2'],
        'a^2 b^-2': ['a^2', 'b^-2'],
        '(a b^-2) / (c^3 d)': ['a^1', 'b^-2', 'c^-3', 'd^-1'],
    }
    for test, strings in cases.items():
        expression = algebra.Expression(test, space_multiplies=True)
        expected = [algebra.Term(string) for string in strings]
        assert expression.terms == expected


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
    for string, parts in cases.items():
        assert algebra.Expression(string) == algebra.Expression(parts)
        components = [algebra.Component(part) for part in parts]
        assert algebra.Expression(string) == algebra.Expression(components)


def test_parser_operators():
    """Test the algebraic parser with non-standard operators."""
    expression = algebra.Expression('a @ b^2 $ c', multiply='@', divide='$')
    expected = [algebra.Term(term) for term in ('a', 'b^2', 'c^-1')]
    assert expression.terms == expected


def test_parser_separators():
    """Test the algebraic parser with non-standard separators."""
    expression = algebra.Expression('a / [b * c]^2', opening='[', closing=']')
    expected = [algebra.Term(term) for term in ('a', 'b^-2', 'c^-2')]
    assert expression.terms == expected


@pytest.mark.skip(reason="Requires significant refactoring")
def test_nonstandard_chars():
    """Test the ability to include non-standard characters in terms."""
    string = '<r> * cm^2 / (<flux> / nuc)'
    expression = algebra.Expression(string)
    expected = [
        algebra.Term('<r>', 1),
        algebra.Term('cm', 2),
        algebra.Term('<flux>', -1),
        algebra.Term('nuc', 1),
    ]
    assert expression.terms == expected


def test_guess_separators():
    """Test the function that guesses separators in an expression."""
    cases = {
        'a * (b / c)': ('(', ')'),
        'a * (b / c]': ('(', ']'),
        'a * [b / c)': ('[', ')'),
        'a * [b / c]': ('[', ']'),
    }
    for test, expected in cases.items():
        assert algebra.guess_separators(test) == expected


def test_parsing_errors():
    """Make sure the parser raises appropriate exceptions."""
    with pytest.raises(algebra.RatioError):
        algebra.Expression('a/b/c')
    with pytest.raises(algebra.ProductError):
        algebra.Expression('a/b*c')


def test_formatted_expression():
    """Test the ability to format terms in an algebraic expression."""
    string = 'a0^2 * (a1*a2) / (a3 * a4^2 * (a5/a6))'
    expresssion = algebra.Expression(string)
    expected = 'a0^2 a1 a2 a3^-1 a4^-2 a5^-1 a6'
    assert expresssion.format() == expected
    expected = 'a0^{2} a1 a2 a3^{-1} a4^{-2} a5^{-1} a6'
    assert expresssion.format(style='tex') == expected


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
    for (s1, s2), result in cases.items():
        e1 = algebra.Expression(s1)
        e2 = algebra.Expression(s2)
        assert s1 == e1
        assert s2 == e2
        assert (s1 == e2) == result
        assert (s2 == e1) == result
        assert (e1 == e2) == result


def test_expression_collection():
    """Confirm that expressions behave like collections."""
    expression = algebra.Expression('a / (b * c * (d / e))')
    terms = [
        algebra.Term('a'),
        algebra.Term('b^-1'),
        algebra.Term('c^-1'),
        algebra.Term('d^-1'),
        algebra.Term('e'),
    ]
    assert list(expression) == terms
    assert len(expression) == len(terms)
    for term in terms:
        assert term in expression


def test_expression_algebra():
    """Test algebraic operations between expressions."""
    strings = [
        'a',
        'a * b^2',
        '(c / b)^3',
        'b^-3 * d / a^2',
        'a / b^2',
        'b^2 / a',
    ]
    expressions = [algebra.Expression(string) for string in strings]

    # Test left and right multiplication and division
    e2 = expressions[2]
    for e1 in (strings[1], expressions[1]):
        expected = algebra.Expression('a * b^-1 * c^3')
        assert e1 * e2 == expected
        assert e2 * e1 == expected
        expected = algebra.Expression('a * b^5 * c^-3')
        assert e1 / e2 == expected
        expected = algebra.Expression('a^-1 * b^-5 * c^3')
        assert e2 / e1 == expected

    # Include a negative exponent
    e2 = expressions[3]
    expected = algebra.Expression('a^-1 * b^-1 * d')
    for e1 in (strings[1], expressions[1]):
        assert e1 * e2 == expected

    # Identically cancel one of the terms
    e2 = expressions[4]
    expected = algebra.Expression('a^2')
    for e1 in (strings[1], expressions[1]):
        assert e1 * e2 == expected

    # Identically cancel all terms
    e2 = expressions[5]
    expected = algebra.Expression('1')
    for e1 in (strings[4], expressions[4]):
        assert e1 * e2 == expected
        assert e2 * e1 == expected

    # Test exponentiation. The second case checks that the previous operation
    # didn't change the term in the expression, which was originally a bug.
    e = expressions[0]
    expected = algebra.Expression('a^2')
    assert e ** 2 == expected
    expected = algebra.Expression('a^3')
    assert e ** 3 == expected

    # Test in-place exponentiation
    e = expressions[0]
    ec = e.copy()
    ec **= 3
    expected = algebra.Expression('a^3')
    assert ec == expected

    # Test exponentiation with multiple terms
    assert expressions[2] ** 3 == algebra.Expression('c^9 * b^-9')

    # Test exponentiation to a negative power
    assert expressions[3] ** -1 == algebra.Expression('b^3 * d^-1 * a^2')


def test_algebra_with_conversion():
    """Test algebraic operations that require conversion to an expression."""
    expr = algebra.Expression('a^2 * b / c^3')
    expected = algebra.Expression('a * b^3 / (c^2 * d)')
    assert expr * 'b^2 * c / (a * d)' == expected
    assert expr / 'a * d / (b^2 * c)' == expected


def test_expression_copy():
    """Test the method that creates a copy of an expression."""
    expr = algebra.Expression('a * b^2 / c^3')
    assert expr.copy() == expr
    assert expr.copy() is not expr


def test_reduced_expression():
    """Test the property that returns an algebraically reduced expression."""
    expression = algebra.Expression('a^2 * b / (a * b^3 * c)')
    copied = expression.reduced
    expected = algebra.Expression('a * b^-2 * c^-1')
    assert copied == expected
    assert copied is not expression
    assert copied != expression
    inplace = expression.reduce()
    assert inplace == expected
    assert inplace is expression
