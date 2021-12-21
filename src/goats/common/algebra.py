import abc
import collections.abc
import fractions
import numbers
import operator
import re
from typing import *
from typing import Pattern

from goats.common import iterables


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


class TermOperatorMixin:
    """A mixin class that provides operations on algebraic terms."""

    _base: str = None
    _exponent: fractions.Fraction = None

    def __pow__(self, power: int):
        """Create a new instance, raised to `power`."""
        return type(self)(self._base, self._exponent * power)

    def __ipow__(self, power: int):
        """Update this instance's exponent"""
        self._exponent *= power
        return self


class Term(TermOperatorMixin):
    """An object representing a term in an algebraic expression.

    Algebraic terms, as defined in this module, may be simple or complex. A
    simple algebraic term consists of a single base quantity and an optional
    fractional exponent. The following are simple algebraic terms::
    * `'1'`
    * `'a'`
    * `'a^2'`
    * `'a^3/2'`

    A complex algebraic term consists of multiple simple terms combined with
    algebraic operators and separators. The following are complex algebraic
    terms::
    * `'a * b^2'` <=> `'(a * b^2)'` <=> `'(a * b^2)^1'`
    * `'(a * b^2)^3'`
    * `'(a * b^2)^3/2'`
    * `'((a / b^2)^3 * c)^2'`
    * `'(a / b^2)^3 * c^2'`

    There are many more ways to construct a complex term than a simple term. In
    fact, any valid algebraic expression is a complex algebraic term; this is by
    design, to support the `Expression` class. The `.issimple` property provides
    a way to determine if a given algebraic term is simple.
    """

    base_re = r'[-+\w#]+'
    expo_re = r'\^[-+]?\d+(?:\/\d+)?'
    full_re = fr'{base_re}(?:{expo_re})?'

    def __init__(
        self,
        arg: Union[str, 'Term'],
        exponent: Union[str, int]=None,
    ) -> None:
        __string = str(arg)
        self._sep = guess_separators(__string)
        self._string = __string
        self._base, exp = self._base_exp
        self._exponent = exp * fractions.Fraction(exponent or 1)

    @property
    def _base_exp(self):
        """Internal property that represents this term's base and exponent.

        This logic here will create the most complex algebraic term possible
        from the initial string. It will parse a simple algebraic term into a
        base quantity and an exponent but it will not attempt to parse a complex
        algebraic term into simple terms. If all attempts to determine an
        appropriate base and exponent fail, it will simply return the initial
        string with an exponent of 1. Therefore, it always ensures a base and
        exponent.

        The following examples use the simple and complex algebraic terms from
        the class docstring the illustrate this splitting::
        * `'1'` -> `'1', 1`
        * `'a'` -> `'a', 1`
        * `'a^2'` -> `'a', 2`
        * `'a^3/2'` -> `'a', '3/2'`
        * `'a * b^2'` <=> `'(a * b^2)'` <=> `'(a * b^2)^1'` -> `'a * b^2', 1`
        * `'(a * b^2)^3'` -> `'a * b^2', 3`
        * `'(a * b^2)^3/2'` -> `'a * b^2', '3/2'`
        * `'((a / b^2)^3 * c)^2'` -> `'(a / b^2)^3 * c', 2`
        * `'(a / b^2)^3 * c^2'` -> `'(a / b^2)^3 * c^2', 1`

        Note that this class stores exponents as instances of
        `fractions.Fraction`.
        """
        forms = [
            self._simple_term,
            self._complex_term,
        ]
        for form in forms:
            if form:
                return form
        return self._string, fractions.Fraction(1)

    @property
    def _simple_term(self):
        """The base and exponent of a simple term, if available.

        The logic in this property will extract the base quantity and an
        appropriate exponent (possibly 1) from the initial string if it has the
        form of a simple algebraic term. Otherwise, it will return `None`.
        """
        if self.issimple:
            try:
                base, exp = self._string.rsplit('^', 1)
            except ValueError:
                base, exp = self._string, 1
            return base, fractions.Fraction(exp)

    @property
    def _complex_term(self):
        """The base and exponent of a complex term, if available.

        The logic in this property will check if the initial string contains an
        algebraic expression entirely bound by parentheses and an optional
        exponent. If it finds an unbound algebraic expression, it will simply
        return the initial string along with an exponent of 1. If it finds a
        bound algebraic expression, it will return the expression and the
        outermost exponent (if it exists) or 1. Otherwise, it will return
        `None`.
        """
        pattern = fr"^[^{self._sep[0]}{self._sep[1]}]*$"
        if re.fullmatch(pattern, self._string):
            return self._string, fractions.Fraction(1)
        group = fr"\{self._sep[0]}.+?\{self._sep[1]}"
        pattern = fr'^({group})({self.expo_re})?$'
        found = re.findall(pattern, self._string)
        if found and len(found) == 1:
            pair = found[0]
            base = str(pair[0])
            inside = base[1:-1]
            if inside.find(self._sep[1]) >= inside.find(self._sep[0]):
                # TODO: Extract a method for this logic. There may be some
                # overlap with Expression parsing.
                counted = False
                count = 0
                exponent = fractions.Fraction(str(pair[1]).lstrip('^') or 1)
                for i, c in enumerate(base):
                    if c == self._sep[0]:
                        count += 1
                        counted = not counted or True # Once true, always true.
                    elif c == self._sep[1]:
                        count -= 1
                    if counted and count == 0 and i < len(base)-1:
                        return base, exponent
                if counted and count == 0:
                    return inside, exponent

    @property
    def issimple(self) -> bool:
        """True if this instance has a single base and an optional exponent."""
        return bool(re.fullmatch(self.full_re, self._string))

    @property
    def base(self) -> str:
        """The base quantity."""
        return self._base

    @property
    def exponent(self) -> fractions.Fraction:
        """The numerical exponent."""
        return self._exponent

    @property
    def isunity(self) -> bool:
        """True iff this instance is equivalent to unity.

        This property may be useful when determining whether to keep a term in
        an algebraic expression.
        """
        return self._base == '1'

    def __repr__(self) -> str:
        """The reproducible representation of this instance."""
        args = f"'{self.base}', {self.exponent}"
        return f"{self.__class__.__qualname__}({args})"

    def __str__(self) -> str:
        """A simplified representation of this instance."""
        if self.isunity or self.exponent == 1:
            return self.base
        if self.issimple:
            return f"{self.base}^{self.exponent}"
        return f"({self.base})^{self.exponent}"

    def __eq__(self, other: 'Term') -> bool:
        """True if two instances' bases and exponents are equal."""
        try:
            return self.base == other.base and self.exponent == other.exponent
        except AttributeError:
            return False


class TermABC(abc.ABC):
    """Abstract base class for algebraic terms."""

    def __init__(
        self,
        arg: Any,
        exponent: Union[str, int]=None,
        base_re: Pattern=None,
        expo_re: Pattern=None,
        separators: Iterable[str]=None,
    ) -> None:
        self._string = str(arg)
        self._base_re = base_re or r'[-+\w#]+'
        self._expo_re = expo_re or r'\^[-+]?\d+(?:\/\d+)?'
        self._full_re = fr'{base_re}(?:{expo_re})?'
        self._sep = self._get_separators(separators)

    def _get_separators(self, given: Iterable[str]) -> Tuple[str, str]:
        """Get an appropriate 2-tuple of opening and closing separators."""
        if not given:
            return guess_separators(self._string)
        try:
            nsep = len(given)
        except TypeError:
            nsep = -1
        return tuple(given) if nsep == 2 else ('(',')')


class ParsingError(Exception):
    """Base class for exceptions encountered during algebraic parsing."""
    def __init__(self, string: str) -> None:
        self.string = string


class RatioError(ParsingError):
    """The string contains multiple '/' on a single level."""

    def __str__(self) -> str:
        return (
            f"The expression '{self.string}' contains ambiguous '/'."
            f" Please refer to the NIST guidelines"
            f" (https://physics.nist.gov/cuu/Units/checklist.html)"
            f" for more information."
        )


class ProductError(ParsingError):
    """The string contains a '*' after a '/'."""

    def __str__(self) -> str:
        return (
            f"The expression '{self.string}' contains an ambiguous '*'."
            f" Please group '*' in parentheses when following '/'."
        )


class OperandError(TypeError):
    pass


class Expression(collections.abc.Collection):
    """An object representing an algebraic expression."""

    # TODO: Allow arbitrary characters in the term base as long as they are not
    # one of the operators or separators. This may require refactoring Term; it
    # also may not be feasible at all. One of the first steps may be redefining
    # the full-term RE in terms of a compliment (i.e., what is *not* included in
    # a term rather than what is included):
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
        self._parse(Term(string))

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

    def __eq__(self, other: 'Expression') -> bool:
        """True if two expressions have the same simple algebraic terms.

        This method defines two expressions as equal if they have equivalent
        lists of simple algebraic terms, regardless of order, after parsing. Two
        expressions with different numbers of terms are always false. If the
        expressions have the same number of terms, this method will sort the
        pairs (first by base, then by exponent) and compare the sorted lists.
        Two expressions are equal if and only if their sorted lists of terms are
        equal.

        If `other` is a string, this method will first attempt to convert it to
        an expression.
        """
        if not isinstance(other, Expression):
            other = self._convert(other)
        if len(self._terms) != len(other._terms):
            return False
        key = operator.attrgetter('base', 'exponent')
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
        terms = [pow(term, exp) for term in self]
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

    def reduce(self, *groups: Iterable[Term]) -> List[Term]:
        """Algebraically reduce terms with equal bases."""
        if not groups:
            self._terms = self._reduce(self._terms.copy())
            return self
        terms = (term for group in groups for term in group)
        return self._reduce(terms)

    def _reduce(self, terms: Iterable[Term]):
        """Internal helper for `reduce`."""
        reduced = {}
        for term in terms:
            if term.base in reduced:
                reduced[term.base] += term.exponent
            else:
                reduced[term.base] = term.exponent
        return [Term(b, e) for b, e in reduced.items() if e != 0]

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
        string = (
            expression if isinstance(expression, str)
            else ' * '.join(f"({part})" for part in expression or ['1'])
        )
        for old, new in replaceable.items():
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

    def format(self, separator: str=' ', style: str=None):
        """Join algebraic terms into a string."""
        if style == 'tex':
            return self._format("{base}^{{{exponent}}}", separator)
        return self._format("{base}^{exponent}", separator)

    def _format(self, template: str, separator: str):
        """Helper method for expression formatting."""
        return separator.join(
            template.format(base=term.base, exponent=term.exponent)
            if term.exponent != 1 else f"{term.base}"
            for term in self.terms
        )

    @property
    def terms(self) -> List[Term]:
        """The algebraic terms in this expression."""
        return self._terms or [Term('1')]

    def _parse(self, term: Term):
        """Internal parsing logic."""
        resolved = self._resolve_operations(term)
        for term in resolved:
            if not term.issimple:
                self._parse(term)
            elif not term.isunity:
                self._terms.append(term)

    def _resolve_operations(self, term: Term) -> List[Term]:
        """Split the current term into operators and operands."""
        parts = self._parse_nested(term.base)
        operands = []
        operators = []
        for part in parts:
            if part in {self._multiply, self._divide}:
                operators.append(part)
            else:
                operands.append(Term(part, term.exponent))
        resolved = [operands[0]]
        if exception := self._check_operators(operators):
            raise exception(term)
        for operator, operand in zip(operators, operands[1:]):
            if operator == self._divide:
                resolved.append(operand ** -1)
            else:
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
        parts = []
        i = 0
        methods = [ # Order matters!
            self._find_term,
            self._find_operator,
            self._find_group,
        ]
        while i < len(string):
            for method in methods:
                result = method(string[i:])
                if result:
                    parts.append(result[0])
                    i += result[1]
        return parts

    def _find_term(self, string: str):
        """Find a simple algebraic term in `string`, if possible."""
        match = re.match(Term.full_re, string)
        if match:
            return match[0], match.end()

    def _find_operator(self, string: str):
        """Find a known operator in `string`, if possible."""
        match = re.match(self._opr, string)
        if match:
            ischar = any(i in match[0] for i in {self._multiply, self._divide})
            if ischar:
                return match[0].strip(), match.end()
            return self._multiply, match.end()

    def _find_group(self, string: str):
        """Find a nested group in `string`, if possible."""
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
            exp = re.match(Term.expo_re, string[j:])
            if exp:
                j += exp.end()
            return string[:j], j


def guess_separators(string: str):
    """Attempt to determine which separators the given expression uses."""
    return (
        iterables.unique(string, {'(','[','{'}) or '(',
        iterables.unique(string, {')',']','}'}) or ')',
    )

