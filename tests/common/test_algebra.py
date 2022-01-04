import pytest
import fractions

from goats.common import algebra


# simple_type = (algebra.Variable, algebra.Constant)


@pytest.mark.term
def test_simple_term():
    """Test the object representing a simple algebraic term."""
    valid = {
        '1': {'coefficient': 1, 'base': '1', 'exponent': 1},
        '2': {'coefficient': 2, 'base': '1', 'exponent': 1},
        '2^3': {'coefficient': 2, 'base': '1', 'exponent': 3},
        '2^-3': {'coefficient': 2, 'base': '1', 'exponent': -3},
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
        term = algebra.OperandFactory().create(string)
        assert isinstance(term, algebra.Term)
        assert term.coefficient == float(expected['coefficient'])
        assert term.base == expected['base']
        assert term.exponent == fractions.Fraction(expected['exponent'])
    invalid = [
        '^3', # exponent only
        'a^', # missing exponent
    ]
    for string in invalid:
        with pytest.raises(algebra.PartValueError):
            algebra.Part(string)


@pytest.mark.term
def test_simple_term_init():
    """Initialize a simple term with various arguments."""
    cases = {
        (1, 'a', 1): [['a'], [1, 'a'], ['a', 1], [1, 'a', 1]],
        (2, 'a', 1): [['2a'], [1, '2a'], ['2a', 1], [1, '2a', 1]],
        (2, 'a', 3): [['2a^3'], [2, 'a', 3]],
        (2, 'a', -1): [['2a^-1'], [2, 'a', -1]],
    }
    for reference, groups in cases.items():
        for args in groups:
            term = algebra.Term(*args)
            assert isinstance(term, algebra.Term)
            assert term.coefficient == reference[0]
            assert term.base == reference[1]
            assert term.exponent == reference[2]


@pytest.mark.term
def test_simple_term_format():
    """Test the ability to properly format a simple algebraic term."""
    cases = [
        ('1', '1'),
        ('1^2', '1'),
        ('2^2', '4'),
        ('a', 'a'),
        ('2a', '2a'),
        ('a^2', 'a^2'),
        ('3a^2', '3a^2'),
        ('1a', 'a'),
        ('a^1', 'a'),
        ('1a^1', 'a'),
        ('2a^1', '2a'),
        ('1a^2', 'a^2'),
    ]
    for (arg, expected) in cases:
        assert str(algebra.Part(arg)) == expected


@pytest.mark.term
def test_simple_term_operators():
    """Test allowed arithmetic operations on a simple algebraic term."""
    x = algebra.Part('x')
    assert isinstance(x, algebra.Variable)
    assert x**2 == algebra.Variable(1, 'x', 2)
    assert 3 * x == algebra.Variable(3, 'x', 1)
    assert x * 3 == algebra.Variable(3, 'x', 1)
    assert (3 * x) ** 2 == algebra.Variable(9, 'x', 2)
    y = algebra.Part('y')
    assert isinstance(y, algebra.Variable)
    y *= 2.5
    assert y == algebra.Variable(2.5, 'y', 1)
    z = algebra.Part('z')
    assert isinstance(z, algebra.Variable)
    z **= -3
    assert z == algebra.Variable(1, 'z', -3)


@pytest.mark.part
def test_part_idempotence():
    """Make sure we can initialize a part object with an existing instance."""
    term = algebra.Part('a^3')
    assert isinstance(term, algebra.Variable)
    assert algebra.Part(term) == term


@pytest.mark.part
def test_part_issimple():
    """Test the check for a 'simple' expression part."""
    cases = {
        '': False,
        'a': True,
        'a^2': True,
        'a^2/3': True,
        '3a^2': True,
        'a * b^2': False,
    }
    simple = (algebra.Variable, algebra.Constant)
    for string, expected in cases.items():
        term = algebra.Part(string)
        assert isinstance(term, simple) == expected


@pytest.mark.part
def test_part_init():
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
        from_ref = algebra.Part(*ref)
        for args in group:
            from_args = algebra.Part(*args)
            assert from_ref == from_args
            for part in [from_ref, from_args]:
                assert part.coefficient == ref[0]
                assert part.base == ref[1]
                assert part.exponent == fractions.Fraction(ref[2])


@pytest.mark.expression
def test_expression_parser():
    """Test the algebraic-expression parser."""
    cases = {
        'a / b': {
            'terms': ['1', 'a', 'b^-1'],
        },
        '1 / b': {
            'terms': ['1', 'b^-1'],
        },
        'a / (b * c)': {
            'terms': ['1', 'a', 'b^-1', 'c^-1'],
        },
        'a / (bc)': {
            'terms': ['1', 'a', 'bc^-1'],
        },
        'a / bc': {
            'terms': ['1', 'a', 'bc^-1'],
        },
        'a * b / c': {
            'terms': ['1', 'a', 'b', 'c^-1'],
        },
        '(a / b) / c': {
            'terms': ['1', 'a', 'b^-1', 'c^-1'],
        },
        '(a / b) / (c / d)': {
            'terms': ['1', 'a', 'b^-1', 'c^-1', 'd'],
        },
        '(a * b / c) / (d * e / f)': {
            'terms': ['1', 'a', 'b', 'c^-1', 'd^-1', 'e^-1', 'f'],
        },
        'a^2 / b^3': {
            'terms': ['1', 'a^2', 'b^-3'],
        },
        '(a^2 / b)^5 / (c^4 / d)^3': {
            'terms': ['1', 'a^10', 'b^-5', 'c^-12', 'd^3'],
        },
        '((a^2 / b) / (c^4 / d))^3': {
            'terms': ['1', 'a^6', 'b^-3', 'c^-12', 'd^3'],
        },
        'a^-2': {
            'terms': ['1', 'a^-2'],
        },
        'a^-3 / b^-6': {
            'terms': ['1', 'a^-3', 'b^6'],
        },
        '(a * (b * c))': {
            'terms': ['1', 'a', 'b', 'c'],
        },
        '(a * (b * c))^2': {
            'terms': ['1', 'a^2', 'b^2', 'c^2'],
        },
        '(a * (b * c)^2)': {
            'terms': ['1', 'a', 'b^2', 'c^2'],
        },
        '(a / (b * c)^2)': {
            'terms': ['1', 'a', 'b^-2', 'c^-2'],
        },
        'a / (b * c * (d / e))': {
            'terms': ['1', 'a', 'b^-1', 'c^-1', 'd^-1', 'e'],
        },
        'a0^2 * (a1*a2) / (a3 * a4^2 * (a5/a6))': {
            'terms': ['1', 'a0^2', 'a1', 'a2', 'a3^-1', 'a4^-2', 'a5^-1', 'a6'],
        },
        '((a^2 * b^3) / c) * (d^-3)': {
            'terms': ['1', 'a^2', 'b^3', 'c^-1', 'd^-3'],
        },
        '3a * b': {
            'terms': ['3', 'a', 'b'],
        },
        '3(a * b)': {
            'terms': ['3', 'a', 'b'],
        },
        '3a / b': {
            'terms': ['3', 'a', 'b^-1'],
        },
        '3(a / b)': {
            'terms': ['3', 'a', 'b^-1'],
        },
        'a / (2.5 * 4.0)': {
            'terms': ['0.1', 'a'],
        },
        'a / (2.5b * 4.0)': {
            'terms': ['0.1', 'a', 'b^-1'],
        },
        'a / ((2.5 * 4.0) * b)': {
            'terms': ['0.1', 'a', 'b^-1'],
        },
        'a / (2.5 * 4.0 * b)': {
            'terms': ['0.1', 'a', 'b^-1'],
        },
    }
    for test, expected in cases.items():
        expression = algebra.Expression(test)
        terms = algebra.asterms(expected['terms'])
        assert len(expression.terms) == len(terms)
        assert all(term in expression.terms for term in terms)


@pytest.mark.expression
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
        assert algebra.Expression(string) == algebra.Expression(terms)
        parts = [algebra.Part(term) for term in terms]
        assert algebra.Expression(string) == algebra.Expression(parts)


@pytest.mark.expression
def test_parser_operators():
    """Test the algebraic parser with non-standard operators."""
    expression = algebra.Expression('a @ b^2 $ c', multiply='@', divide='$')
    expected = [algebra.Term(term) for term in ('a', 'b^2', 'c^-1')]
    assert expression.terms == expected


@pytest.mark.expression
def test_parser_separators():
    """Test the algebraic parser with non-standard separators."""
    expression = algebra.Expression('a / [b * c]^2', opening='[', closing=']')
    expected = [algebra.Term(term) for term in ('a', 'b^-2', 'c^-2')]
    assert expression.terms == expected


@pytest.mark.skip(reason="Requires significant refactoring")
@pytest.mark.expression
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


@pytest.mark.expression
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


@pytest.mark.expression
def test_parsing_errors():
    """Make sure the parser raises appropriate exceptions."""
    with pytest.raises(algebra.RatioError):
        algebra.Expression('a/b/c')
    with pytest.raises(algebra.ProductError):
        algebra.Expression('a/b*c')


@pytest.mark.expression
def test_formatted_expression():
    """Test the ability to format terms in an algebraic expression."""
    string = 'a0^2 * (a1*a2) / (a3 * a4^2 * (a5/a6))'
    expresssion = algebra.Expression(string)
    expected = 'a0^2 a1 a2 a3^-1 a4^-2 a5^-1 a6'
    assert expresssion.format() == expected
    expected = 'a0^{2} a1 a2 a3^{-1} a4^{-2} a5^{-1} a6'
    assert expresssion.format(style='tex') == expected


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
        expressions = [algebra.Expression(arg) for arg in args]
        for arg, expression in zip(args, expressions):
            assert arg == expression
        assert (args[0] == expressions[1]) == result
        assert (args[1] == expressions[0]) == result
        assert (expressions[0] == expressions[1]) == result


@pytest.mark.expression
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

    # Test multiplication with a coefficient
    assert expressions[0] * expressions[6] == algebra.Expression('3a^2')
    assert expressions[6] ** 2 == algebra.Expression('9a^2')


@pytest.mark.expression
def test_algebra_with_conversion():
    """Test algebraic operations that require conversion to an expression."""
    expr = algebra.Expression('a^2 * b / c^3')
    expected = algebra.Expression('a * b^3 / (c^2 * d)')
    assert expr * 'b^2 * c / (a * d)' == expected
    assert expr / 'a * d / (b^2 * c)' == expected


@pytest.mark.expression
def test_expression_copy():
    """Test the method that creates a copy of an expression."""
    expr = algebra.Expression('a * b^2 / c^3')
    assert expr.copy() == expr
    assert expr.copy() is not expr


@pytest.mark.expression
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
