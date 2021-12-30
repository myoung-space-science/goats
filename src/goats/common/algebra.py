import abc
import collections.abc
import fractions
import numbers
import operator
import re
from typing import *
from typing import Pattern

from goats.common import iterables
from goats.common import numerical


def strip_outer_parentheses(string: str) -> str:
    """Strip only the left- and right-most parentheses, if they exist.

    This function will first check for opening and closing parentheses at the
    beginning and end of the string. If it finds those, it will make sure that
    they are two ends of a single pair; this avoids stripping the outer
    parentheses off of a string like '(a/b)/(c/d)'. If the string meets both of
    these conditions, this function will remove the outer parentheses.
    Otherwise, it will return the original string.
    """
    inner = string[1:-1]
    outer_pair = (
        # The leftmost character is '(' and the rightmost character is ')'.
        string[0] == '(' and string[-1] == ')'
        and
        # We don't encounter a ')' before the first '('.
        inner.find(')') >= inner.find('(')
    )
    if outer_pair:
        return inner
    return string


def entire(string: str, opening: str, closing: str):
    """True if `string` is completely bounded by separators."""
    counted = False
    count = 0
    if string[0] != opening or string[-1] != closing:
        return False
    for i, c in enumerate(string):
        if c == opening:
            count += 1
            counted = not counted or True # Once true, always true.
        elif c == closing:
            count -= 1
        if counted and count == 0 and i < len(string)-1:
            return False
    return counted and count == 0


def guess_separators(string: str):
    """Attempt to determine which separators the given expression uses."""
    return (
        iterables.unique(string, {'(','[','{'}) or '(',
        iterables.unique(string, {')',']','}'}) or ')',
    )


class RE(NamedTuple):
    """Namespace for regular expressions used in this module."""

    digit = r'[0-9]' # only equal to r'\d' in certain modes
    coefficient = fr'[-+]?{digit}*\.?{digit}+' # '1e2' not included
    base = r'[a-zA-Z#_]+[0-9]*' # digits must follow a known non-digit
    exponent = fr'[-+]?{digit}+(?:[/.]{digit}+)?'
    unity = r'(?<![\d.])1(?![\d.])'
    variable = re.compile(fr'(?:{coefficient})?{base}(?:\^{exponent})?')
    """The regular expression matching a variable term."""
    constant = re.compile(fr'{coefficient}(?:\^{exponent})?')
    """The regular expression matching a constant term."""
    _bases = fr'({coefficient})?((?(1)(?:{base}|\B)?|{base}))'
    _search = re.compile(fr'{_bases}\^?({exponent})?')

    @classmethod
    def parse(cls, s: str):
        """Find a variable or constant term if it exists."""
        return cls._search.findall(s)


class PartTypeError(TypeError):
    pass


class PartValueError(ValueError):
    pass


class Part:
    """Part of an algebraic expression.

    Algebraic parts mainly exist to support the `~algebra.Expression` class.
    They may be simple or complex but always map to an instance of
    `~algebra.Term` or one of its subclasses. In particular, a simple algebraic
    part maps to an instance of either `~algebra.Variable` or
    `~algebra.Constant`.

    A simple algebraic part can have the form [c]b[^e] or c[^e], where `c` is a
    numerical coefficient, `b` is a string base, and `e` is a numerical
    exponent. Braces ('[]') denote an optional component that defaults to a
    value of 1. The form c[^e] represents a constant term and the form [c]b[^e]
    represents a variable term. 

    Examples include::
    * `'1'`: unity / multiplicative identity
    * `'1^n'` (`n` real): constant equivalent to unity
    * `'m^n'` (`m`, `n` real): arbitrary constant
    * `'V'`: variable 'V' with coefficient 1 and exponent 1
    * `'V^n'` (`n` real): variable 'V' with coefficient 1 and exponent n
    * `'mV'` (`m` real): variable 'V' with coefficient m and exponent 1
    * `'mV^n'` (`m`, `n` real): variable 'V' with coefficient m and exponent n

    Note that the base of a variable part may comprise multiple characters as
    long as it does not begin with a digit, which this class will interpret as
    part of the coefficient.

    A complex algebraic part consists of multiple simple parts combined with
    algebraic operators and separators. The following are complex algebraic
    parts::
    * `'a * b^2'` <=> `'(a * b^2)'` <=> `'(a * b^2)^1'`
    * `'(a * b^2)^3'`
    * `'(a * b^2)^3/2'`
    * `'((a / b^2)^3 * c)^2'`
    * `'(a / b^2)^3 * c^2'`
    * `'a / (2 * 4b)'`
    * `'(2a * b)^3 / (4 * c)'`

    There are many more ways to construct a complex part than a simple part. In
    fact, any valid algebraic expression, as well as any number, is a complex
    algebraic part; this is by design, to support the `Expression` class. The
    type of the corresponding `~algebra.Term` indicates whether a part is simple
    or complex.
    """

    @classmethod
    def isvariable(cls, s: str):
        """True if the given string matches the variable pattern."""
        return bool(RE.variable.fullmatch(s))

    @classmethod
    def isconstant(cls, s: str):
        """True if the given string matches the constant pattern."""
        return bool(RE.constant.fullmatch(s))

    @classmethod
    def normalize(cls, *args):
        """Extract attributes from the given argument(s)."""
        try:
            nargs = len(args)
        except TypeError:
            raise PartTypeError(args) from None
        if nargs == 1:
            return 1, str(args[0]), fractions.Fraction(1)
        if nargs == 2:
            # A length-2 `args` may represent either:
            # - (coefficient <Real>, base <str>)
            # - (base <str>, exponent <Real or str>)
            #
            # If it has the first form, assume there is an implied exponent
            # equal to 1. If it has the second form, assume there is an implied
            # coefficient equal to 1. Otherwise, raise an exception.
            argtypes = zip(args, (numbers.Real, str))
            implied_exponent = all(isinstance(a, t) for a, t in argtypes)
            argtypes = zip(args, (str, (numbers.Real, str)))
            implied_coefficient = all(isinstance(a, t) for a, t in argtypes)
            if implied_exponent:
                c = numerical.cast(args[0])
                b = str(args[1])
                e = fractions.Fraction(1)
                return c, b, e
            if implied_coefficient:
                c = 1
                b = str(args[0])
                e = fractions.Fraction(args[1])
                return c, b, e
            badtypes = [type(arg) for arg in args]
            raise PartTypeError(
                "Acceptable two-argument forms are"
                " (coefficient <Real>, base <str>)"
                " and"
                " (base <str>, exponent <Real or str>)"
                " not"
                f"({', '.join(badtypes)})"
            )
        if nargs == 3:
            # NOTE: No need to apply exponent here because the coefficient is
            # outside of the exponential portion by definition in the 3-argument
            # form.
            c = numerical.cast(args[0])
            b = str(args[1])
            e = fractions.Fraction(args[2])
            return c, b, e
        raise PartValueError(
            f"{cls.__qualname__}"
            f" accepts 1, 2, or 3 arguments"
            f" (got {nargs})"
        )

    @classmethod
    def evaluate(cls, *args):
        """Compute coefficient, base, and exponent from input.
        
        This method will create the most complex algebraic part possible from
        the initial string. It will parse a simple algebraic part into a
        coefficient, variable, and exponent but it will not attempt to fully
        parse a complex algebraic part into simple parts (i.e. algebraic terms).
        In other words, it will do as little work as possible to extract a
        coefficient and exponent, and the expression on which they operate. If
        all attempts to determine appropriate attributes fail, it will simply
        return the string representation of the initial argument with
        coefficient and exponent both equal to 1.

        The following examples use the complex algebraic parts from the class
        docstring to illustrate the minimal parsing described above::
        * `'a * b^2'` <=> `'(a * b^2)'` <=> `'(a * b^2)^1'` -> `1, 'a * b^2', 1`
        * `'2a * b^2'` -> `1, '2a * b^2', 1`
        * `'2(a * b^2)'` -> `2, 'a * b^2', 1`
        * `'(a * b^2)^3'` -> `1, 'a * b^2', 3`
        * `'2(a * b^2)^3'` -> `2, 'a * b^2', 3`
        * `'(a * b^2)^3/2'` -> `'a * b^2', '3/2'`
        * `'((a / b^2)^3 * c)^2'` -> `'(a / b^2)^3 * c', 2`
        * `'(a / b^2)^3 * c^2'` -> `'(a / b^2)^3 * c^2', 1`

        Note that this class stores the coefficient as a `float` or `int`, and
        the exponent as a `fractions.Fraction`.
        """
        c0, b0, e0 = cls.normalize(*args)
        if b0.startswith('^') or b0.endswith('^'):
            raise PartValueError(b0) from None
        parsers = (
            cls._simplex,
            cls._complex,
        )
        for parse in parsers:
            if result := parse(c0, b0, e0):
                return result
        return c0, b0, e0

    # It seems like subclasses of Part (Simplex and Complex) would be better
    # here. Simplex.__new__ would create a new Variable or Constant;
    # Complex.__new__ would create a new Term. However, classes may be overkill
    # (or at least non-Pythonic). Does it make more sense to convert this logic
    # into a module-level function that resolves `*args` into an instance of
    # Term or one of its subclasses? If we go with classes, maybe they should
    # have their own instances of `RE` with appropriate patterns (e.g., a
    # variable-like pattern for bounded parts in Complex).

    @classmethod
    def _simplex(cls, c0, b0, e0):
        """Parse a simple algebraic part, if possible.

        A simple algebraic part represents a variable or constant term. This
        method checks for them in that order.
        """
        if not (cls.isconstant(b0) or cls.isvariable(b0)):
            return
        parsed = RE.parse(b0)
        if len(parsed) == 1:
            found = parsed[0]
            c1 = numerical.cast(found[0] or 1)
            coefficient = c0 * (c1 ** e0)
            base = str(found[1] or 1)
            e1 = fractions.Fraction(found[2] or 1)
            exponent = e1 * e0
            return coefficient, base, exponent

    @classmethod
    def _complex(cls, c0, b0, e0):
        """Parse a complex algebraic part, if possible.

        A complex algebraic part is any algebraic part that is neither a term
        not a constant. This method will attempt to match `b0` to a term-like
        regular expression in which a non-empty string is bounded by known
        separator charaters, is possibly preceeded by a coefficient, and is
        possibly followed by an exponent.

        If `b0` matches this RE, it may have the form of a `Term` with a
        potentially complex part in the variable position, but it need not. For
        example, the following strings will both match, but only the first
        contains a coefficient and exponent that apply to all terms:
        - '3(a * b / (c * d))^2'
        - '3(a * b) / (c * d)^2'

        Therefore, we need to perform some additional searching to determine if
        the final closing separator matches the initial opening separator. The
        algorithm essentially consists of computing a running difference between
        the number of opening separators and the number of closing separators.
        If we close the initial opening separator before the end of the
        variable-like substring, the substring is not bounded, so we can't
        extract a coefficient and exponent. If we close the initial opening
        separator right at the end of the substring, the substring is bounded by
        separators, so we can extract and update the coefficient and exponent if
        they exist.

        If `b0` doesn't match the RE or if the variable-like term isn't bounded
        afterall, this method will return `None`.
        """
        sep = guess_separators(b0)
        bounded = fr"\{sep[0]}.+?\{sep[1]}"
        full = fr'^({RE.coefficient})?({bounded})\^?({RE.exponent})?$'
        found = re.findall(full, b0)
        if not found or len(found) != 1:
            return
        parts = found[0]
        string = str(parts[1])
        inside = string[1:-1].strip()
        if not entire(string, *sep):
            return
        c = numerical.cast(parts[0] or 1)
        e = fractions.Fraction(parts[2] or 1)
        coefficient = c0 * (c ** e0)
        exponent = e * e0
        if this := cls._simplex(1, inside, 1):
            exponent *= this[2]
            coefficient *= this[0] ** exponent
            base = this[1]
        else:
            base = inside
        return coefficient, base, exponent

    @classmethod
    def _init(cls, base: str):
        """Return the appropriate type for the given base string."""
        if cls.isvariable(base):
            return Variable
        if cls.isconstant(base):
            return Constant
        return Term

    def __new__(cls, *args):
        """Create a new instance based on input."""
        coefficient, base, exponent = cls.evaluate(*args)
        if cls is Part:
            new = cls._init(base)
            return new(coefficient, base, exponent)
        raise TypeError(f"Can't create new term from {args}")


class Term(iterables.ReprStrMixin):
    """A single term in an algebraic expression."""

    def __init__(self, coefficient, base, exponent) -> None:
        self.coefficient = coefficient
        self.base = base
        self.exponent = exponent

    def __pow__(self, power):
        """Create a new instance, raised to `power`."""
        coefficient = self.coefficient ** power
        exponent = self.exponent * power
        return type(self)(coefficient, self.base, exponent)

    def __ipow__(self, power):
        """Update this instance's exponent."""
        self.coefficient **= power
        self.exponent *= power
        return self

    def __mul__(self, other):
        """Create a new instance, multiplied by `other`."""
        coefficient = self.coefficient * other
        return type(self)(coefficient, self.base, self.exponent)

    __rmul__ = __mul__

    def __imul__(self, other):
        """Update this instance's coefficient."""
        self.coefficient *= other
        return self

    def __eq__(self, other) -> bool:
        """True if two instances' attributes are equal."""
        if not isinstance(other, Term):
            return NotImplemented
        attrs = {'coefficient', 'base', 'exponent'}
        try:
            true = (getattr(self, a) == getattr(other, a) for a in attrs)
            truth = all(true)
        except AttributeError:
            return False
        else:
            return truth

    def __str__(self) -> str:
        """A simplified representation of this object."""
        return self.format()

    def format(self):
        """Format this term."""
        return f"{self.coefficient} * ({self.base})^{self.exponent}"


class Variable(Term):
    """An algebraic term with an irreducible variable base."""

    def format(self, style: str=None):
        """Format this term."""
        coefficient = '' if self.coefficient == 1 else str(self.coefficient)
        exponent = self._format_exponent(style)
        return f"{coefficient}{self.base}{exponent}"

    def _format_exponent(self, style: str):
        """Format the current exponent for printing."""
        if self.exponent == 1:
            return ''
        if not style:
            return f"^{self.exponent}"
        if 'tex' in style.lower():
            return f"^{{{self.exponent}}}"
        raise ValueError(f"Can't format {self.exponent}")


class Constant(Term):
    """An algebraic term representing a constant value."""

    def __init__(self, coefficient, base, exponent) -> None:
        super().__init__(coefficient, base, exponent)
        self._value = self.coefficient ** self.exponent

    def format(self):
        return str(self._value)

    def __float__(self) -> float:
        """Called for float(self)."""
        return float(self._value)

    def __eq__(self, other: SupportsFloat):
        """True if two objects have the same floating-point value."""
        return float(self) == float(other)


class ParsingError(Exception):
    """Base class for exceptions encountered during algebraic parsing."""
    def __init__(self, arg: Any) -> None:
        self.arg = arg


class RatioError(ParsingError):
    """The string contains multiple '/' on a single level."""

    def __str__(self) -> str:
        return (
            f"The expression '{self.arg}' contains ambiguous '/'."
            f" Please refer to the NIST guidelines"
            f" (https://physics.nist.gov/cuu/Units/checklist.html)"
            f" for more information."
        )


class ProductError(ParsingError):
    """The string contains a '*' after a '/'."""

    def __str__(self) -> str:
        return (
            f"The expression '{self.arg}' contains an ambiguous '*'."
            f" Please group '*' in parentheses when following '/'."
        )


class OperandError(TypeError):
    pass


class Expression(collections.abc.Collection):
    """An object representing an algebraic expression."""

    # TODO: Allow arbitrary characters in the term variable as long as they are
    # not one of the operators or separators. This may require refactoring Part
    # and Term; it also may not be feasible at all. One of the first steps may
    # be redefining the full-term RE in terms of a compliment (i.e., what is
    # *not* included in a term rather than what is included). Note that
    # `re.escape()` may be handy here:
    #
    # + escaped = [fr'\{c}' for c in (mul, div, *sep)]
    # + self._base_re = fr'[^\s{"".join(escaped)}]+'
    # + self._full_re = fr'{self._base_re}(?:{Term.expo_re})?'
    #
    # The challenge at this point is that Term will still use its own `base_re`,
    # which will cause expression parsing to be inconsistent. Should we allow
    # updating the Term RE class attributes? The induced coupling may be weird.
    # Should we use a factory class to create Term classes with pre-set REs?

    _multiply = '*'
    _divide = '/'
    _opening = '('
    _closing = ')'

    def __init__(
        self,
        expression: Union[str, iterables.Separable],
        multiply: str='*',
        divide: str='/',
        opening: str='(',
        closing: str=')',
        space_multiplies: bool=False,
    ) -> None:
        """Create a new expression from user input.

        This will initialize a new instance of this class from the given string
        or collection of parts after replacing operators and separators with
        their standard versions, if necessary.

        Parameters
        ----------
        expression : string or collection
            A single string or collection of any type to initialize the new
            instance. If this is a collection, all members must support
            conversion to a string.

        multiply : string, default='*'
            The token that represents multiplication in `expression`.

        divide : string, default='/'
            The token that represents division in `expression`.

        opening : string, default='('
            The token that represents an opening separator in `expression`.

        closing : string, default='('
            The token that represents a closing separator in `expression`.

        space_multiplies : bool, default=False
            NIST guidelines allow for the use of a space (`' '`) to represent
            multiplication. Setting this keyword to `True` will interpret any
            whitespace between terms as multiplication. Because this feature can
            lead to incorrect parsing results when an expression contains errant
            whitespace, it is off by default.
        """
        opr = fr'\s*[\{self._multiply}\{self._divide}]\s*'
        self._opr = fr'{opr}|\s+' if space_multiplies else opr
        op_sep_str = {
            multiply: self._multiply,
            divide: self._divide,
            opening: self._opening,
            closing: self._closing,
        }
        string = self._normalize(expression, op_sep_str)
        self._terms = []
        self._scale = 1.0
        self._parse(Part(string))
        self.scale = self._scale

    def __iter__(self) -> Iterator[Term]:
        return iter(self.terms)

    def __len__(self) -> int:
        return len(self.terms)

    def __contains__(self, key: str) -> bool:
        return key in self.terms

    def __repr__(self) -> str:
        """An unambiguous representation of this instance."""
        return f"{self.__class__.__qualname__}({self})"

    def __str__(self) -> str:
        """A simplified representation of this instance."""
        return self.format()

    def format(self, separator: str=' ', style: str=None):
        """Join algebraic terms into a string."""
        formatted = (term.format(style=style) for term in self)
        return separator.join(formatted)

    def __eq__(self, other: 'Expression') -> bool:
        """True if two expressions have the same algebraic terms.

        This method defines two expressions as equal if they have equivalent
        lists of algebraic terms (a.k.a simple parts), regardless of order,
        after parsing. Two expressions with different numbers of terms are
        always false. If the expressions have the same number of terms, this
        method will sort the triples (first by base, then by exponent, and
        finally by coefficient) and compare the sorted lists. Two expressions
        are equal if and only if their sorted lists of terms are equal.

        If `other` is a string, this method will first attempt to convert it to
        an expression.
        """
        if not isinstance(other, Expression):
            other = self._convert(other)
        if len(self._terms) != len(other._terms):
            return False
        key = operator.attrgetter('base', 'exponent', 'coefficient')
        return sorted(self._terms, key=key) == sorted(other._terms, key=key)

    def __mul__(self, other: 'Expression'):
        """Called for self * other.

        This method implements multiplication between two expressions by
        reducing the exponents of terms with a common base. If `other` is a
        string, it will first attempt to convert it to an `Expression`.
        """
        if not isinstance(other, Expression):
            other = self._convert(other)
        if not other:
            return NotImplemented
        terms = self.reduce(self._terms, other)
        return self._new(terms)

    def __rmul__(self, other: Any):
        """Called for other * self."""
        return self._convert(other).__mul__(self)

    def __truediv__(self, other: 'Expression'):
        """Called for self / other.

        This method implements division between two expressions by raising all
        terms in `other` to -1, then reducing the exponents of terms with a
        common base. If `other` is a string, it will first attempt to convert it
        to an `Expression`.
        """
        if not isinstance(other, Expression):
            other = self._convert(other)
        if not other:
            return NotImplemented
        return self.__mul__([term ** -1 for term in other])

    def __rtruediv__(self, other: Any):
        """Called for other / self."""
        return self._convert(other).__truediv__(self)

    def __pow__(self, exp: numbers.Real):
        """Called for self ** exp.

        This method implements exponentiation of an expression by raising all
        terms to the given power, then reducing exponents of terms with a common
        base. It will first attempt to convert `exp` to a float.
        """
        exp = self._convert(exp, float)
        if not exp:
            return NotImplemented
        terms = [pow(term, exp) for term in self._terms]
        return self._new(self.reduce(terms))

    def __ipow__(self, exp: numbers.Real):
        """Called for self **= exp."""
        for term in self._terms:
            term **= exp
        return self.reduce()

    def _convert(self, other, converter: Callable=None, fatal: bool=False):
        """Convert `other` to a new type, if possible.

        Parameters
        ----------
        other : any
            The object to convert.

        converter : callable, default=None
            A callable object that will perform the conversion. By default, this
            method converts `other` to an instance of this class. 

        fatal : bool, default=False
            If this keyword is set to true, a conversion error will raise an
            exception. If it is set to false (default), this method will
            silently return `None`. The default option allows the calling code
            to decide what to do with a failed conversion.
        """
        convert = converter or self._new
        try:
            converted = convert(other)
        except TypeError:
            if fatal:
                raise OperandError(f"Can't convert {other}")
        else:
            return converted

    def reduce(self, *groups: Iterable[Term]):
        """Algebraically reduce terms with equal bases.
        
        Parameters
        ----------
        *groups : tuple of iterables
            Zero or more iterables of `~algebra.Term` instances. If there are
            zero groups, this method will use the current collection of terms in
            this expression; otherwise, it will combine all terms it finds in
            the full collection of groups.
        """
        if not groups:
            self._terms = self._reduce(self._terms.copy())
            return self
        terms = (term for group in groups for term in group)
        return self._reduce(terms)

    def _reduce(self, terms: Iterable[Term]):
        """Internal helper for `reduce`.

        This method handles the actual logic for combining terms with the same
        base into a single term with a combined exponent.
        """
        reduced = {}
        for term in terms:
            if term.base in reduced:
                reduced[term.base]['coefficient'] *= term.coefficient
                reduced[term.base]['exponent'] += term.exponent
            else:
                attributes = {
                    'coefficient': term.coefficient,
                    'exponent': term.exponent,
                }
                reduced[term.base] = attributes
        return [
            Part(v['coefficient'], k, v['exponent'])
            for k, v in reduced.items() if v['exponent'] != 0
        ]

    @property
    def reduced(self) -> 'Expression':
        """A new instance, with algebraically reduced terms."""
        inst = self.copy()
        return inst.reduce()

    def _normalize(
        self,
        expression: Union[str, iterables.Separable],
        replaceable: Dict[str, str],
    ) -> str:
        """Convert user input to a standard format."""
        string = self._build_string(expression)
        return self._replace_tokens(string, replaceable)

    def _build_string(self, expression: Union[str, iterables.Separable]) -> str:
        """Create a standardized string from user input."""
        if not expression:
            return '1'
        if isinstance(expression, str):
            return expression
        return ' * '.join(f"({part})" for part in expression or ['1'])

    def _replace_tokens(self, string: str, replacement: Dict[str, str]) -> str:
        """Replace user tokens with standard versions."""
        for old, new in replacement.items():
            string = string.replace(old.strip(), new)
        return string

    @classmethod
    def _new(cls, arg: Union[str, iterables.Separable]):
        """Internal helper method for creating a new instance.

        This method is separated out for the sake of modularity, in case of a
        need to add any other functionality when creating a new instance from
        the current one (perhaps in a subclass).
        """
        return cls(arg)

    def copy(self):
        """Create a copy of this instance."""
        return self._new(self._terms)

    @property
    def terms(self) -> List[Term]:
        """The algebraic terms in this expression."""
        return self._terms or [Term('1')]

    def _parse(self, term: Term):
        """Internal parsing logic."""
        self._scale *= term.coefficient
        normalized = Part(term.base, term.exponent)
        if isinstance(normalized, (Variable, Constant)):
            self._terms.append(normalized)
        else:
            resolved = self._resolve_operations(normalized)
            for t in resolved:
                self._parse(t)

    def _resolve_operations(self, term: Term) -> List[Term]:
        """Split the current term into operators and operands."""
        parsed = self._parse_nested(term.base)
        operands = []
        operators = []
        for part in parsed:
            if part in (self._multiply, self._divide):
                operators.append(part)
            else:
                args = (part, term.exponent)
                operands.append(Part(*args))
        if exception := self._check_operators(operators):
            raise exception(term)
        resolved = [operands[0]]
        for operator, operand in zip(operators, operands[1:]):
            if operator == self._divide:
                operand **= -1
            resolved.append(operand)
        return resolved

    def _check_operators(self, operators: List[str]) -> Optional[ParsingError]:
        """Check for operator-related exceptions.

        This method checks for the following errors and returns the appropriate
        exception class if it finds one:
        * Multiple divisions on a single level (e.g., `'a / b / c'`), which
          results in a `RatioError`.
        * Multiplication after division on the same level (e.g., `'a / b * c'`),
          which results in a `ProductError`.

        Both of the examples shown above result in errors because they each
        introduce an ambiguous order of operations. Users can resolve the
        ambiguity by properly grouping terms in the expression. Continuing with
        the above examples, `'a / b / c'` should become `'(a / b) / c'` or `'a /
        (b / c)'`, and `'a / b * c'` should become `'(a / b) * c'` or `'a / (b *
        c)'`.
        """
        n_div = operators.count(self._divide)
        i_div = operators.index(self._divide) if n_div > 0 else -1
        n_mul = operators.count(self._multiply)
        i_mul = operators.index(self._multiply) if n_mul > 0 else -1
        if n_div > 1:
            return RatioError
        if n_div == 1 and i_mul > i_div:
            return ProductError

    def _parse_nested(self, string: str) -> List[str]:
        """Parse an algebraic expression while preserving nested groups."""
        parsed = []
        methods = [ # Order matters!
            self._find_term,
            self._find_number,
            self._find_operator,
            self._find_group,
        ]
        errstart = "Failed to find a match for"
        errfinal = repr(string)
        n = len(string)
        i = 0
        i0 = None
        while i < n:
            i0 = i
            for method in methods:
                result = method(string[i:])
                if result:
                    parsed.append(result[0])
                    i += result[1]
            if i == i0:
                if i != 0:
                    errfinal = f"{string[i:]!r} in {errfinal}"
                raise RecursionError(f"{errstart} {errfinal}") from None
        return parsed

    def _find_number(self, string: str):
        """Check for a pure number at the start of `string`."""
        match = re.match(RE.constant, string)
        if match:
            return match[0], match.end()

    def _find_term(self, string: str):
        """Check for an algebraic term at the start of `string`."""
        match = re.match(RE.variable, string)
        if match:
            return match[0], match.end()

    def _find_operator(self, string: str):
        """Check for a known operator at the start of `string`."""
        match = re.match(self._opr, string)
        if match:
            ischar = any(i in match[0] for i in {self._multiply, self._divide})
            if ischar:
                return match[0].strip(), match.end()
            return self._multiply, match.end()

    def _find_group(self, string: str):
        """Check for a nested group at the start of `string`."""
        if string.startswith(self._opening):
            level = 1
            j = 1
            while level > 0:
                c = string[j]
                if c == self._opening:
                    level += 1
                elif c == self._closing:
                    level -= 1
                j += 1
            exp = re.match(fr'\^{RE.exponent}', string[j:])
            if exp:
                j += exp.end()
            return string[:j], j


class Part(abc.ABC, iterables.ReprStrMixin):
    """Base class for parts of an algebraic expression."""

    __slots__ = ()


class Operator(Part):
    """An operator in an algebraic expression."""

    def __init__(self, operation: str) -> None:
        self.operation = operation

    def __str__(self) -> str:
        """A simplified representation of this object."""
        return self.operation

    def __eq__(self, other) -> bool:
        """True if two operators represent the same operation."""
        if isinstance(other, Operator):
            return other.operation == self.operation
        if isinstance(other, str):
            return other == self.operation
        return NotImplemented


class Operand(Part):
    """An operand in an algebraic expression."""

    def __init__(self, coefficient, base, exponent) -> None:
        self.coefficient = coefficient
        """The numerical coefficient."""
        self.base = base
        """The base term or complex."""
        self.exponent = exponent
        """The numerical exponent."""

    def __pow__(self, power):
        """Create a new operand, raised to `power`."""
        coefficient = self.coefficient ** power
        exponent = self.exponent * power
        return type(self)(coefficient, self.base, exponent)

    def __ipow__(self, power):
        """Update this operand's exponent."""
        self.coefficient **= power
        self.exponent *= power
        return self

    def __mul__(self, other):
        """Create a new operand, multiplied by `other`."""
        coefficient = self.coefficient * other
        return type(self)(coefficient, self.base, self.exponent)

    __rmul__ = __mul__

    def __imul__(self, other):
        """Update this operand's coefficient."""
        self.coefficient *= other
        return self

    def __eq__(self, other) -> bool:
        """True if two operands' attributes are equal."""
        if not isinstance(other, Operand):
            return NotImplemented
        attrs = {'coefficient', 'base', 'exponent'}
        try:
            true = (getattr(self, a) == getattr(other, a) for a in attrs)
            truth = all(true)
        except AttributeError:
            return False
        else:
            return truth

    def __str__(self) -> str:
        """A simplified representation of this object."""
        return self.format()

    def format(self):
        """Format this operand for printing."""
        if self.coefficient == 1 and self.exponent == 1:
            return self.base
        string = f"({self.base})"
        if self.exponent != 1:
            string = f"{string}^{self.exponent}"
        if self.coefficient != 1:
            string = f"{self.coefficient}{self.base}"
        return string


class Term(Operand):
    """An algebraic operand with an irreducible base."""

    def __init__(self, coefficient, base, exponent) -> None:
        super().__init__(coefficient, base, exponent)
        self._value = (
            self.coefficient ** self.exponent
            if self.base == '1' else None
        )

    def format(self, style: str=None):
        """Format this term."""
        exponent = self._format_exponent(style)
        if self.base == '1':
            return f"{self.coefficient}{exponent}"
        coefficient = '' if self.coefficient == 1 else str(self.coefficient)
        return f"{coefficient}{self.base}{exponent}"

    def _format_exponent(self, style: str):
        """Format the current exponent for printing."""
        if self.exponent == 1:
            return ''
        if not style:
            return f"^{self.exponent}"
        if 'tex' in style.lower():
            return f"^{{{self.exponent}}}"
        raise ValueError(f"Can't format {self.exponent}")

    def __float__(self) -> float:
        """Called for float(self)."""
        if self._value:
            return float(self._value)
        errmsg = f"Can't convert term with base {self.base!r} to float"
        raise TypeError(errmsg)

    def __eq__(self, other) -> bool:
        if isinstance(other, numbers.Real):
            return float(self) == float(other)
        return super().__eq__(other)


class OperandTypeError(TypeError):
    pass


class OperandValueError(ValueError):
    pass


class Parsed(NamedTuple):
    """The result of parsing an algebraic string."""

    result: Part
    remainder: str

    def __bool__(self) -> bool:
        """Called for bool(self)."""
        return bool(self.result)


@runtime_checkable
class PartFactory(Protocol):
    """Protocol for algebraic factories."""

    @abc.abstractmethod
    def parse(self) -> Parsed:
        pass


class OperatorFactory(PartFactory):
    """A factory that produces algebraic operators."""

    def __init__(
        self,
        multiply: str='*',
        divide: str='/',
    ) -> None:
        mul = fr'\{multiply}'
        div = fr'\{divide}'
        self.patterns = {
            'multiply': re.compile(
                fr'(?<!{div})(\s*{mul}\s*)(?!{div})'
            ),
            'divide': re.compile(
                fr'(?<!{mul})(\s*{div}\s*)(?!{mul})'
            ),
        }
        """Compiled regular expressions for algebraic operators."""

    def parse(self, string: str):
        """Extract an operator at the start of `string`, possible."""
        for key in ('multiply', 'divide'):
            if match := self.patterns[key].match(string):
                return Parsed(
                    result=Operator(key),
                    remainder=string[match.end():],
                )
        return Parsed(result=None, remainder=string)


class OperandFactory(PartFactory):
    """A factory that produces algebraic operands."""

    digit = r'[0-9]' # only equal to r'\d' in certain modes
    coefficient = fr'[-+]?{digit}*\.?{digit}+' # '1e2' not included
    base = r'[a-zA-Z#_]+[0-9]*' # digits must follow a known non-digit
    exponent = fr'[-+]?{digit}+(?:[/.]{digit}+)?'
    unity = r'(?<![\d.])1(?![\d.])'

    def __init__(
        self,
        opening: str='(',
        closing: str=')',
        raising: str='^',
    ) -> None:
        exponent = fr'\{raising}{self.exponent}'
        self.patterns = {
            'constant': re.compile(
                fr'(?P<coefficient>{self.coefficient})'
                fr'(?P<exponent>{exponent})?'
            ),
            'variable': re.compile(
                fr'(?P<coefficient>{self.coefficient})?'
                fr'(?P<base>{self.base})'
                fr'(?P<exponent>{exponent})?'
            ),
            'complex': re.compile(
                fr'(?P<coefficient>{self.coefficient})?'
                fr'(?P<base>\{opening}.+?\{closing})'
                fr'(?P<exponent>{exponent})?'
            ),
            'exponent': re.compile(exponent),
            'opening': re.compile(fr'\{opening}'),
            'closing': re.compile(fr'\{closing}'),
            'raising': re.compile(fr'\{raising}')
        }
        """Compiled regular expressions for algebraic operands."""

    # NOTE: The difference between `create` and `parse` is: 
    # * `create` resolves input into (coefficient, base, exponent), then creates
    #   an appropriate operand from the base string, applies the input
    #   coefficient and exponent, and finally returns the operand.
    # * `parse` attempts to match an operand at the start of a string, then
    #   creates an appropriate operand from only that substring, and finally
    #   returns the operand and the remainder of the string.
    #
    # TODO: Reduce overlap between `create`, `parse`, and their various helper
    # methods.

    def create(self, *args, strict: bool=False):
        """Create an operand from input.

        Parameters
        ----------
        *args
            The object(s) from which to create an operand, if possible. This may
            take one of the following forms: a single string representing the
            base operand; a numerical coefficient and a base string; a base
            string and a numerical exponent; or a coefficient, base, and
            exponent. A missing coefficient or exponent will default to 1.

        strict : bool, default=false
            If true, this method will return `None` if it is unable to create an
            operand from `*args`. The default behavior is to return the input
            (with default coefficient and exponent, if necessary) as an
            `~algebra.Operand`.

        Returns
        -------
        `~algebra.Operand` or `None`
            An instance of `~algebra.Operand` or one of its subclasses if the
            input arguments represent a valid operand. The `strict` keyword
            dictates the return behavior when input does not produce an operand.

        Notes
        -----
        This method will create the most complex algebraic operand possible from
        the initial string. It will parse a simple algebraic operand into a
        coefficient, variable, and exponent but it will not attempt to fully
        parse a complex algebraic operand into simpler parts (i.e. algebraic
        terms). In other words, it will do as little work as possible to extract
        a coefficient and exponent, and the expression on which they operate. If
        all attempts to determine appropriate attributes fail, it will simply
        return the string representation of the initial argument with
        coefficient and exponent both equal to 1.

        The following examples use the complex algebraic parts from the class
        docstring to illustrate the minimal parsing described above::
        * `'a * b^2'` <=> `'(a * b^2)'` <=> `'(a * b^2)^1'` -> `1, 'a * b^2', 1`
        * `'2a * b^2'` -> `1, '2a * b^2', 1`
        * `'2(a * b^2)'` -> `2, 'a * b^2', 1`
        * `'(a * b^2)^3'` -> `1, 'a * b^2', 3`
        * `'2(a * b^2)^3'` -> `2, 'a * b^2', 3`
        * `'(a * b^2)^3/2'` -> `'a * b^2', '3/2'`
        * `'((a / b^2)^3 * c)^2'` -> `'(a / b^2)^3 * c', 2`
        * `'(a / b^2)^3 * c^2'` -> `'(a / b^2)^3 * c^2', 1`

        Note that this class stores the coefficient as a `float` or `int`, and
        the exponent as a `fractions.Fraction`.
        """
        c0, b0, e0 = self.normalize(*args)
        ends = (b0[0], b0[-1])
        if any(self.patterns['raising'].match(c) for c in ends):
            raise OperandValueError(b0) from None
        parsers = (
            self._simplex,
            self._complex,
        )
        for parse in parsers:
            if result := parse(c0, b0, e0):
                return result
        if not strict:
            return Operand(c0, b0, e0)

    def normalize(self, *args):
        """Extract attributes from the given argument(s)."""
        try:
            nargs = len(args)
        except TypeError:
            raise OperandTypeError(args) from None
        if nargs == 1:
            # A length-1 `args` may represent either:
            # - coefficient <Real>
            # - base <str>
            #
            # If it has one of these forms, substitute the default value for the
            # missing attributes. Otherwise, raise an exception.
            arg = args[0]
            if isinstance(arg, str):
                return self._standard(base=arg)
            if isinstance(arg, numbers.Real):
                return self._standard(coefficient=arg)
            raise OperandTypeError(
                "A single argument may be either"
                " a coefficient <Real> or a base <str>;"
                f" not {type(arg)}"
            )
        if nargs == 2:
            # A length-2 `args` may represent either:
            # - (base <str>, exponent <Real or str>)
            # - (coefficient <Real>, exponent <Real or str>)
            # - (coefficient <Real>, base <str>)
            #
            # If it has one of these forms, substitute the default value for the
            # missing attribute. Otherwise, raise an exception.
            argtypes = zip(args, (str, (numbers.Real, str)))
            implied_coefficient = all(isinstance(a, t) for a, t in argtypes)
            if implied_coefficient:
                return self._standard(
                    base=args[0],
                    exponent=args[1],
                )
            argtypes = zip(args, (numbers.Real, (numbers.Real, str)))
            implied_base = all(isinstance(a, t) for a, t in argtypes)
            if implied_base:
                return self._standard(
                    coefficient=args[0],
                    exponent=args[1],
                )
            argtypes = zip(args, (numbers.Real, str))
            implied_exponent = all(isinstance(a, t) for a, t in argtypes)
            if implied_exponent:
                return self._standard(
                    coefficient=args[0],
                    base=args[1],
                )
            badtypes = [type(arg) for arg in args]
            raise OperandTypeError(
                "Acceptable two-argument forms are"
                " (base <str>, exponent <Real or str>),"
                " (coefficient <Real>, exponent <Real or str>),"
                " or"
                " (coefficient <Real>, base <str>);"
                " not"
                f"({', '.join(str(t) for t in badtypes)})"
            )
        if nargs == 3:
            return self._standard(
                coefficient=args[0],
                base=args[1],
                exponent=args[2],
            )
        raise OperandValueError(
            f"{self.__class__.__qualname__}"
            f" accepts 1, 2, or 3 arguments"
            f" (got {nargs})"
        )

    def parse(self, string: str):
        """Extract an operand at the start of `string`, possible."""
        current = (string[1:-1] if self.entire(string) else string).strip()
        for key in ('complex', 'variable', 'constant'):
            if match := self.patterns[key].match(current):
                args = self._standard(**match.groupdict())
                return Parsed(
                    result=self._create_new(key, *args),
                    remainder=current[match.end():],
                )
        return Parsed(result=None, remainder=current)

    def _create_new(self, key: str, *args):
        """Create a new instance from `args`, based on `key`."""
        if key in {'constant', 'variable'}:
            return Term(*args)
        return Operand(*args)

    def _standard(
        self,
        coefficient: Any=None,
        base: Any=None,
        exponent: Any=None,
    ) -> Tuple[Union[float, int], str, fractions.Fraction]:
        """Cast component(s) to the appropriate type(s) or return default."""
        c = numerical.cast(coefficient or 1)
        b = str(base or 1)
        if isinstance(exponent, str):
            exponent = self.patterns['raising'].sub('', exponent)
        e = fractions.Fraction(exponent or 1)
        return c, b, e

    def findfull(self, string: str, keys: Union[str, Iterable[str]]):
        """Extract components if the string matches a named pattern.

        This method will compare `string` to the pattern(s) indicated by `keys`
        in the order of `keys`. It will only return a non-null result if the
        whole string matches one of the target patterns.
        """
        patterns = (
            pattern for key, pattern in self.patterns.items()
            if key in iterables.Separable(keys)
        )
        for pattern in patterns:
            if match := pattern.fullmatch(string):
                return self._standard(**match.groupdict())

    def _simplex(self, c0, b0, e0):
        """Parse a simple algebraic part, if possible.

        A simple algebraic part represents a variable or constant term. This
        method checks for them in that order.
        """
        keys = ('constant', 'variable')
        found = self.findfull(b0, keys)
        if not found:
            return
        try:
            c1, base, e1 = found
        except ValueError:
            errmsg = f"Expected 3 components in {b0} but found {len(found)}"
            raise OperandValueError(errmsg)
        coefficient = c0 * (c1 ** e0)
        exponent = e1 * e0
        return Term(coefficient, base, exponent)

    def _complex(self, c0, b0, e0):
        """Parse a complex algebraic part, if possible.

        A complex algebraic part is any algebraic part that is neither a term
        not a constant. This method will attempt to match `b0` to a term-like
        regular expression in which a non-empty string is bounded by known
        separator charaters, is possibly preceeded by a coefficient, and is
        possibly followed by an exponent.

        If `b0` matches this RE, it may have the form of a `Term` with a
        potentially complex part in the variable position, but it need not. For
        example, the following strings will both match, but only the first
        contains a coefficient and exponent that apply to all terms:
        - '3(a * b / (c * d))^2'
        - '3(a * b) / (c * d)^2'

        Therefore, we need to perform some additional searching to determine if
        the final closing separator matches the initial opening separator. The
        algorithm essentially consists of computing a running difference between
        the number of opening separators and the number of closing separators.
        If we close the initial opening separator before the end of the
        variable-like substring, the substring is not bounded, so we can't
        extract a coefficient and exponent. If we close the initial opening
        separator right at the end of the substring, the substring is bounded by
        separators, so we can extract and update the coefficient and exponent if
        they exist.

        If `b0` doesn't match the RE or if the variable-like term isn't bounded
        afterall, this method will return `None`.
        """
        found = self.findfull(b0, 'complex')
        if not found:
            return
        try:
            c1, base, e1 = found
        except ValueError:
            errmsg = f"Expected 3 components in {b0} but found {len(found)}"
            raise OperandValueError(errmsg)
        if not self.entire(base):
            return
        inside = base[1:-1].strip()
        coefficient = c0 * (c1 ** e0)
        exponent = e1 * e0
        if this := self._simplex(1, inside, 1):
            exponent *= this[2]
            coefficient *= this[0] ** exponent
            base = this[1]
        else:
            base = inside
        return Operand(coefficient, base, exponent)

    def entire(self, string: str):
        """True if `string` is completely bounded by separators."""
        counted = False
        count = 0
        opened = self.patterns['opening'].match(string[0])
        closed = self.patterns['closing'].match(string[-1])
        if not (opened and closed):
            return False
        for i, c in enumerate(string):
            if self.patterns['opening'].match(c):
                count += 1
                counted = not counted or True # Once true, always true.
            elif self.patterns['closing'].match(c):
                count -= 1
            if counted and count == 0 and i < len(string)-1:
                return False
        return counted and count == 0


class Parser:
    """A tool for parsing algebraic expressions."""

    def __init__(
        self,
        multiply: str='*',
        divide: str='/',
        opening: str='(',
        closing: str=')',
        raising: str='^',
    ) -> None:
        """
        Initialize a parser with fixed tokens.

        Parameters
        ----------
        multiply : string, default='*'
            The token that represents multiplication.

        divide : string, default='/'
            The token that represents division.

        opening : string, default='('
            The token that represents an opening separator.

        closing : string, default='('
            The token that represents a closing separator.

        raising : string, default='^'
            The token that represents raising to a power (exponentiation).
        """
        self.operators = OperatorFactory(multiply, divide)
        self.operands = OperandFactory(opening, closing, raising)
        self.tokens = {
            'multiply': multiply,
            'divide': divide,
            'opening': opening,
            'closing': closing,
            'raising': raising,
        }

    def parse(self, string: str):
        """Resolve the given string into individual terms."""
        standard = iterables.batch_replace(string, self.tokens)
        operand = self.operands.create(standard)
        if not operand:
            return []
        if isinstance(operand, Term):
            return [operand]
        return self._resolve_operations(operand)

    def _insert_multiply(self, parts: Iterable[Part]):
        """Insert a multiplication operator between adjacent terms.

        NIST guidelines permit the use of a space character to indicate
        multiplication between unit symbols, and this is a common practice when
        writing algebraic expressions in general. This method places a
        multiplication operator of the appropriate form between adjacent terms
        in order to make that operation explicit
        """
        result = []
        last = parts[0]
        for this in parts[1:]:
            if all(isinstance(part, Operand) for part in (last, this)):
                result.extend([last, Operator('multiply')])
            else:
                result.append(last)
            last = this
        result.append(last)
        return result

    # TODO: Refactor this method at the comment breaks.
    def _resolve_operations(self, current: Operand):
        """Separate an algebraic group into operators and operands."""
        # Parse the base string into operands and operators.
        parts = self._parse_complex(current.base)
        # Insert explicit multiplication operators where implied.
        tmp = self._insert_multiply(parts)
        # Gather operands, invert if necessary, and catch operator errors.
        exponent = current.exponent
        operands = tmp[::2]
        operators = tmp[1::2]
        if exception := self._check_operators(operators):
            raise exception(current)
        pairs = zip(operands[::-1], operators[::-1])
        resolved = [operands[0] ** +exponent]
        for (operand, operator) in pairs:
            exp = -exponent if operator == 'divide' else +exponent
            resolved.append(operand ** exp)
        # Store terms and parse remaining groups.
        terms = [self.operands.create(current.coefficient)]
        for operand in resolved:
            if isinstance(operand, Term):
                terms.append(operand)
            elif isinstance(operand, Operand):
                terms.extend(self._resolve_operations(operand))
            else:
                raise TypeError(f"Unknown operand type: {operand!r}")
        return terms

    def _parse_complex(self, string: str) -> List[Part]:
        """Parse an algebraic expression while preserving nested groups."""
        errstart = "Failed to find a match for"
        errfinal = repr(string)
        parts = []
        parsers = (self.operands, self.operators)
        init = string
        while string:
            current = string
            for parser in parsers:
                # Could we instead stay in this loop until a parser fails?
                parsed = parser.parse(current)
                if parsed:
                    parts.append(parsed.result)
                    current = parsed.remainder
            if current == string:
                if current == init:
                    errfinal = f"{current!r} in {errfinal}"
                raise RecursionError(f"{errstart} {errfinal}") from None
            string = current
        return parts

    def _check_operators(
        self,
        operators: Sequence[Operator],
    ) -> Optional[ParsingError]:
        """Check for operator-related exceptions.

        This method checks for the following errors and returns the appropriate
        exception class if it finds one:
        * Multiple divisions on a single level (e.g., `'a / b / c'`), which
          results in a `RatioError`.
        * Multiplication after division on the same level (e.g., `'a / b * c'`),
          which results in a `ProductError`.

        Both of the examples shown above result in errors because they each
        introduce an ambiguous order of operations. Users can resolve the
        ambiguity by properly grouping terms in the expression. Continuing with
        the above examples, `'a / b / c'` should become `'(a / b) / c'` or `'a /
        (b / c)'`, and `'a / b * c'` should become `'(a / b) * c'` or `'a / (b *
        c)'`.
        """
        n_div = operators.count('divide')
        i_div = operators.index('divide') if n_div > 0 else -1
        n_mul = operators.count('multiply')
        i_mul = operators.index('multiply') if n_mul > 0 else -1
        if n_div > 1:
            return RatioError
        if n_div == 1 and i_mul > i_div:
            return ProductError


class Expression(collections.abc.Collection, iterables.ReprStrMixin):
    """An object representing an algebraic expression."""

    def __init__(
        self,
        expression: Union[str, iterables.Separable],
        **kwargs
    ) -> None:
        """Create a new expression from user input.

        This will initialize a new instance of this class from the given string
        or collection of parts after replacing operators and separators with
        their standard versions, if necessary.

        Parameters
        ----------
        expression : string or collection
            A single string or collection of any type to initialize the new
            instance. If this is a collection, all members must support
            conversion to a string.

        **kwargs
            Keywords to pass to `~algebra.Parser`.
        """
        self.parser = Parser(**kwargs)
        string = self._standardize(expression)
        self._terms = self.parser.parse(string)

    def _standardize(self, expression: Union[str, iterables.Separable]) -> str:
        """Convert user input to a standard format."""
        if not expression:
            return '1'
        if isinstance(expression, str):
            return expression
        return ' * '.join(f"({part})" for part in expression or ['1'])

    @property
    def terms(self) -> List[Term]:
        """The algebraic terms in this expression."""
        return self._terms or [Term('1')]

    def __iter__(self) -> Iterator[Term]:
        return iter(self.terms)

    def __len__(self) -> int:
        return len(self.terms)

    def __contains__(self, key: str) -> bool:
        return key in self.terms

    def __str__(self) -> str:
        """A simplified representation of this instance."""
        return self.format()

    def format(self, separator: str=' ', style: str=None):
        """Join algebraic terms into a string."""
        formatted = (term.format(style=style) for term in self)
        return separator.join(formatted)

    def __eq__(self, other: 'Expression') -> bool:
        """True if two expressions have the same algebraic terms.

        This method defines two expressions as equal if they have equivalent
        lists of algebraic terms (a.k.a simple parts), regardless of order,
        after parsing. Two expressions with different numbers of terms are
        always false. If the expressions have the same number of terms, this
        method will sort the triples (first by base, then by exponent, and
        finally by coefficient) and compare the sorted lists. Two expressions
        are equal if and only if their sorted lists of terms are equal.

        If `other` is a string, this method will first attempt to convert it to
        an expression.
        """
        if not isinstance(other, Expression):
            other = self._convert(other)
        if len(self._terms) != len(other._terms):
            return False
        key = operator.attrgetter('base', 'exponent', 'coefficient')
        return sorted(self._terms, key=key) == sorted(other._terms, key=key)

    def __mul__(self, other: 'Expression'):
        """Called for self * other.

        This method implements multiplication between two expressions by
        reducing the exponents of terms with a common base. If `other` is a
        string, it will first attempt to convert it to an `Expression`.
        """
        if not isinstance(other, Expression):
            other = self._convert(other)
        if not other:
            return NotImplemented
        terms = self.reduce(self._terms, other)
        return self._new(terms)

    def __rmul__(self, other: Any):
        """Called for other * self."""
        return self._convert(other).__mul__(self)

    def __truediv__(self, other: 'Expression'):
        """Called for self / other.

        This method implements division between two expressions by raising all
        terms in `other` to -1, then reducing the exponents of terms with a
        common base. If `other` is a string, it will first attempt to convert it
        to an `Expression`.
        """
        if not isinstance(other, Expression):
            other = self._convert(other)
        if not other:
            return NotImplemented
        return self.__mul__([term ** -1 for term in other])

    def __rtruediv__(self, other: Any):
        """Called for other / self."""
        return self._convert(other).__truediv__(self)

    def __pow__(self, exp: numbers.Real):
        """Called for self ** exp.

        This method implements exponentiation of an expression by raising all
        terms to the given power, then reducing exponents of terms with a common
        base. It will first attempt to convert `exp` to a float.
        """
        exp = self._convert(exp, float)
        if not exp:
            return NotImplemented
        terms = [pow(term, exp) for term in self._terms]
        return self._new(self.reduce(terms))

    def __ipow__(self, exp: numbers.Real):
        """Called for self **= exp."""
        for term in self._terms:
            term **= exp
        return self.reduce()

    def _convert(self, other, converter: Callable=None, fatal: bool=False):
        """Convert `other` to a new type, if possible.

        Parameters
        ----------
        other : any
            The object to convert.

        converter : callable, default=None
            A callable object that will perform the conversion. By default, this
            method converts `other` to an instance of this class. 

        fatal : bool, default=False
            If this keyword is set to true, a conversion error will raise an
            exception. If it is set to false (default), this method will
            silently return `None`. The default option allows the calling code
            to decide what to do with a failed conversion.
        """
        convert = converter or self._new
        try:
            converted = convert(other)
        except TypeError:
            if fatal:
                raise OperandError(f"Can't convert {other}")
        else:
            return converted

    def reduce(self, *groups: Iterable[Term]):
        """Algebraically reduce terms with equal bases.
        
        Parameters
        ----------
        *groups : tuple of iterables
            Zero or more iterables of `~algebra.Term` instances. If there are
            zero groups, this method will use the current collection of terms in
            this expression; otherwise, it will combine all terms it finds in
            the full collection of groups.
        """
        if not groups:
            self._terms = self._reduce(self._terms.copy())
            return self
        terms = (term for group in groups for term in group)
        return self._reduce(terms)

    def _reduce(self, terms: Iterable[Term]):
        """Internal helper for `reduce`.

        This method handles the actual logic for combining terms with the same
        base into a single term with a combined exponent.
        """
        reduced = {}
        for term in terms:
            if term.base in reduced:
                reduced[term.base]['coefficient'] *= term.coefficient
                reduced[term.base]['exponent'] += term.exponent
            else:
                attributes = {
                    'coefficient': term.coefficient,
                    'exponent': term.exponent,
                }
                reduced[term.base] = attributes
        return [
            Part(v['coefficient'], k, v['exponent'])
            for k, v in reduced.items() if v['exponent'] != 0
        ]

    @property
    def reduced(self) -> 'Expression':
        """A new instance, with algebraically reduced terms."""
        inst = self.copy()
        return inst.reduce()

    def copy(self):
        """Create a copy of this instance."""
        return self._new(self._terms)

    @classmethod
    def _new(cls, arg: Union[str, iterables.Separable]):
        """Internal helper method for creating a new instance.

        This method is separated out for the sake of modularity, in case of a
        need to add any other functionality when creating a new instance from
        the current one (perhaps in a subclass).
        """
        return cls(arg)

