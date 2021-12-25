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


class TermValueError(ValueError):
    """Argument not an algebraic term."""

    def __init__(self, arg) -> None:
        self.arg = arg

    def __str__(self) -> str:
        return f"Can't create an algebraic term from {self.arg!r}"


class Term(iterables.ReprStrMixin):
    """An object representing a term in an algebraic expression.
    
    An algebraic term has the form [c]b[^e], where `c` is an optional
    coefficient, `v` is the variable, and `e` is an optional exponent. An
    omitted coefficient or exponent implies a value of 1. The type of `v` is
    non-numeric except for the special case of `'1'` (see below).

    Examples include::
    * `'1'`: unity (multiplicative identity)
    * `'a'`: variable 'a' with coefficient 1 and exponent 1
    * `'a^2'`: variable 'a' with coefficient 1 and exponent 2
    * `'a^3/2'`: variable 'a' with coefficient 1 and exponent 3/2
    * `'2a^3'`: variable 'a' with coefficient 2 and exponent 3
    * `'2a0^3'`: variable 'a0' with coefficient 2 and exponent 3
    """

    d_re = r'[0-9]' # only equal to r'\d' in certain modes
    n_re = fr'[-+]?{d_re}*\.?{d_re}+'
    c_re = n_re # don't allow exponential notation (e.g., 1e2)
    b_re = r'[a-zA-Z#]+[0-9]*' # digits must follow a known non-digit
    e_re = fr'[-+]?{d_re}+(?:[/.]{d_re}+)?'
    # NOTE: Put the unity RE outside for `full_re` because we want it to check
    # that special case first in `fullmatch`, but put it inside the variable RE
    # of `find_re` so the variable portion will always appear at index 1 in the
    # resultant tuple.
    u_re = r'(?<![\d.])1(?![\d.])'
    full_re = fr'{u_re}|(?:{c_re})?{b_re}(?:\^{e_re})?'
    find_re = fr'({c_re})?({u_re}|{b_re})\^?({e_re})?'
    # TODO:
    # - Use compiled versions.
    # - Create class methods to check arbitrary strings for matches. For
    #   example, this would make it easier to include `\^` when asking whether a
    #   component has an exponent while excluding it from the exponent RE.

    def __init__(self, *args) -> None:
        try:
            c, v, e = self._normalize(args)
        except TypeError:
            raise TermValueError(args)
        else:
            self.coefficient = c
            """The coefficient of this term."""
            self.variable = v
            """The variable of this term."""
            self.exponent = e
            """The exponent of this term."""

    def _normalize(self, args):
        """Ensure a three-tuple of (coefficient, variable, exponent)."""
        c, base, e = Component.normalize(args)
        if parsed := self._parse(base):
            coefficient = c * (parsed[0] ** e)
            variable = parsed[1]
            exponent = parsed[2] * e
            return coefficient, variable, exponent
        raise TermValueError(base)

    _RT = TypeVar('_RT', bound=tuple)
    _RT = Tuple[Union[int, float], str, fractions.Fraction]

    def _parse(self, s: str) -> _RT:
        """Extract components from the input string."""
        if re.fullmatch(self.full_re, s):
            found = re.findall(self.find_re, s)
            if len(found) == 1 and isinstance(found[0], tuple):
                c, variable, e = found[0]
                # NOTE: No need to apply exponent to coefficient because initial
                # arguments of the form (cv)^e are not allowed.
                coefficient = numerical.cast(c or 1)
                exponent = fractions.Fraction(e or 1)
                return coefficient, variable, exponent

    def __pow__(self, power):
        """Create a new instance, raised to `power`."""
        new_c = self.coefficient ** power
        new_e = self.exponent * power
        arg = self.format(coefficient=new_c, exponent=new_e)
        return type(self)(arg)

    def __ipow__(self, power):
        """Update this instance's exponent."""
        self.exponent *= power
        return self

    def __mul__(self, other):
        """Create a new instance, multiplied by `other`."""
        new = self.coefficient * other
        arg = self.format(coefficient=new)
        return type(self)(arg)

    __rmul__ = __mul__

    def __imul__(self, other):
        """Update this instance's coefficient."""
        self.coefficient *= other
        return self

    def __eq__(self, other) -> bool:
        """True if two instances' bases and exponents are equal."""
        if not isinstance(other, Term):
            return NotImplemented
        attrs = {'coefficient', 'variable', 'exponent'}
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

    def format(self, style: str=None, **updates):
        """Format this term."""
        c = updates.get('coefficient', self.coefficient)
        e = updates.get('exponent', self.exponent)
        coefficient = self._format_coefficient(c)
        exponent = self._format_exponent(e, style)
        return f"{coefficient}{self.variable}{exponent}"

    def _format_coefficient(self, coefficient: Union[int, float]):
        """Format the current coefficient for string use."""
        if coefficient == 1:
            return ''
        return str(coefficient)

    def _format_exponent(self, exponent: fractions.Fraction, style: str):
        """Format the current exponent for string use."""
        if exponent == 1:
            return ''
        if not style:
            return f"^{exponent}"
        if 'tex' in style.lower():
            return f"^{{{exponent}}}"
        raise ValueError(f"Can't format {exponent}")


class ComponentTypeError(TypeError):
    pass


class ComponentValueError(ValueError):
    pass


# TODO 22Dec2021:
# + Step 1
#  - Redefine this as purely a helper for Expression.
#  - [Take advantage of overlap with Term if possible.]
#  - [Extract parsing from Expression if possible.]
# + Step 2
#  - Parse baseTypes.h `#define` directives
# + Step 3
#  - Finish porting streams3d.py
class Component:
    """An object representing a component of an algebraic expression.

    Algebraic components mainly exist to support the `~algebra.Expression`
    class. They may be simple or complex. A simple algebraic component is
    equivalent to an algebraic term. See `~algebra.Term` for examples.

    A complex algebraic component consists of multiple simple components
    combined with algebraic operators and separators. The following are complex
    algebraic components::
    * `'a * b^2'` <=> `'(a * b^2)'` <=> `'(a * b^2)^1'`
    * `'(a * b^2)^3'`
    * `'(a * b^2)^3/2'`
    * `'((a / b^2)^3 * c)^2'`
    * `'(a / b^2)^3 * c^2'`

    There are many more ways to construct a complex component than a simple
    component. In fact, any valid algebraic expression, as well as any number,
    is a complex algebraic component; this is by design, to support the
    `Expression` class. The `Component.issimple` property provides a way to
    determine if a given algebraic component is simple, and `Component.asterm`
    converts a simple algebraic component into a `Term`.
    """

    CType = TypeVar('CType', bound=numbers.Real)
    CType = Union[int, float]

    BType = TypeVar('BType', bound=str)
    BType = str

    EType = TypeVar('EType', bound=numbers.Real)
    EType = fractions.Fraction

    ArgsType = TypeVar('ArgsType', bound=tuple)
    ArgsType = Tuple[CType, BType, EType]

    def __init__(self, *args) -> None:
        """
        Parameters
        ----------
        *args
            A length-1, -2, or -3 tuple. The 1-element form must be a string
            from which this class can extract a base component (possibly the
            entire string), and optional coefficient and exponent. The 2-element
            form may be either (coefficient, base) or (base, exponent); this
            class will provide a default value for the missing exponent or
            coefficient. The 3-element form must be (coefficient, base,
            exponent). This class will still attempt to extract a base
            component, and optional coefficient and exponent from either the 2-
            or 3-element form, then reduce coefficients and exponents as
            necessary.
        """
        self._issimple = None
        self.coefficient, self.base, self.exponent = self._init(args)

    def _init(self, args) -> ArgsType:
        """Extract appropriate attributes or raise an exception."""
        norm = self.normalize(args)
        if parsed := self._parse(norm):
            return parsed
        raise ComponentValueError(f"Can't initialize from {args}")

    @classmethod
    def normalize(cls, args) -> ArgsType:
        """Extract attributes from the given argument(s)."""
        try:
            nargs = len(args)
        except TypeError:
            raise ComponentTypeError(args) from None
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
            raise ComponentTypeError(
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
        # TODO: Consider putting a check (with this exception) in __new__. We
        # could also do further checking, including:
        # - making sure the length-1 form supports conversion to string;
        # - making sure the length-2 form is either (coefficient-like,
        #   base-like) or (base-like, exponent-like)
        # - making sure the length-3 form is (coefficient-like, base-like,
        #   exponent-like).
        raise ComponentValueError(
            f"{cls.__qualname__}"
            f" accepts 1, 2, or 3 arguments"
            f" (got {nargs})"
        )

    def _parse(self, args: ArgsType) -> ArgsType:
        """Parse appropriate attributes from arguments.
        
        This method will create the most complex algebraic component possible
        from the initial string. It will parse a simple algebraic component into
        a coefficient, variable, and exponent but it will not attempt to fully
        parse a complex algebraic component into simple components (i.e.
        algebraic terms). In other words, it will do as little work as possible
        to extract a coefficient and exponent, and the expression on which they
        operate. If all attempts to determine appropriate attributes fail, it
        will simply return the string representation of the initial argument
        with coefficient and exponent both equal to 1.

        The following examples use the complex algebraic components from the
        class docstring to illustrate the minimal parsing described above::
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
        parsers = (
            self._parse_simple,
            self._parse_complex,
        )
        for parse in parsers:
            if result := parse(*args):
                return result
        return args

    def _parse_simple(
        self,
        _coefficient: CType,
        _base: BType,
        _exponent: EType,
    ) -> ArgsType:
        """Parse a simple algebraic component, if possible.

        A simple algebraic component may be a term or a constant. This method
        checks for them in that order.

        * If we can convert `_base` to an instance of `~algebra.Term`, we can
           use its attributes to compute the coefficient and exponent, then
           return its variable as the base.
        * If `_base` matches the RE for a pure number, we'll shift it to the
           coefficient, define the base to be '1', and pass the exponent along.
        * If neither of those succeed, this method will return its arguments
          unaltered.
        """
        try:
            term = Term(_base)
            self._issimple = True
        except TermValueError:
            self._issimple = False
        else:
            coefficient = _coefficient * (term.coefficient ** _exponent)
            exponent = term.exponent * _exponent
            return coefficient, term.variable, exponent
        if match := re.fullmatch(Term.n_re, _base):
            c = numerical.cast(match[0])
            coefficient = _coefficient * (c ** _exponent)
            return coefficient, '1', _exponent

    def _parse_complex(
        self,
        _coefficient: CType,
        _base: BType,
        _exponent: EType,
    ) -> ArgsType:
        """Parse a complex algebraic component, if possible.

        A complex algebraic component is any algebraic component that is neither
        a term not a constant. This method will attempt to match `_base` to a
        term-like regular expression in which a non-empty string is bounded by
        known separator charaters, is possibly preceeded by a coefficient, and
        is possibly followed by an exponent.

        If `_base` matches this RE, it may have the form of a `Term` with a
        potentially complex component in the variable position, but it need not.
        For example, the following strings will both match, but only the first
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

        If `_base` doesn't match the RE or if the variable-like term isn't
        boudned afterall, this method will return `None`.
        """
        sep = guess_separators(_base)
        bounded = fr"\{sep[0]}.+?\{sep[1]}"
        full = fr'^({Term.c_re})?({bounded})\^?({Term.e_re})?$'
        found = re.findall(full, _base)
        if not found or len(found) != 1:
            return
        parts = found[0]
        string = str(parts[1])
        inside = string[1:-1].strip()
        if not entire(string, *sep):
            return
        c = numerical.cast(parts[0] or 1)
        e = fractions.Fraction(parts[2] or 1)
        coefficient = _coefficient * (c ** _exponent)
        exponent = e * _exponent
        if this := self._parse_simple(1, inside, 1):
            exponent *= this[2]
            coefficient *= this[0] ** exponent
            base = this[1]
        else:
            base = inside
        return coefficient, base, exponent

    @property
    def isconstant(self) -> bool:
        """True if this instance is equivalent to a constant value."""
        try:
            numerical.cast(self.base)
        except ValueError:
            return False
        else:
            return True

    @property
    def issimple(self) -> bool:
        """True if this instance has the form of an algebraic term."""
        if self._issimple is None:
            self._issimple = False
        return self._issimple

    @property
    def isunity(self) -> bool:
        """True iff this instance is equivalent to unity.

        This property may be useful when determining whether to keep a term in
        an algebraic expression.
        """
        return self.base == '1' and self.coefficient == 1

    def __float__(self) -> float:
        """Convert this component to a `float`, if possible."""
        if self.isconstant:
            return self.coefficient * float(self.base) ** self.exponent
        raise ValueError(f"Can't convert {self} to float")

    @property
    def asterm(self) -> Term:
        """Convert this component to a `Term`, if possible."""
        if self.issimple:
            return Term(str(self))
        raise ValueError(f"Can't convert {self} to Term.")

    def __pow__(self, power):
        """Create a new instance, raised to `power`."""
        return type(self)(
            self.coefficient ** power,
            self.base,
            self.exponent * power,
        )

    def __ipow__(self, power):
        """Update this instance's exponent."""
        self.coefficient **= power
        self.exponent *= power
        return self

    def __mul__(self, other):
        """Create a new instance, multiplied by `other`."""
        return type(self)(
            (self.coefficient * other) ** self.exponent,
            self.base,
            self.exponent,
        )

    __rmul__ = __mul__

    def __imul__(self, other):
        """Update this instance's coefficient."""
        self.coefficient *= other
        return self

    def __eq__(self, other: 'Component') -> bool:
        """True if two instances' attributes are equal."""
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
        c = '' if self.coefficient == 1 else str(self.coefficient)
        b = self.base if self.issimple else f'({self.base})'
        e = '' if self.exponent == 1 else f"^{self.exponent}"
        return f"{c}{b}{e}"

    def __repr__(self) -> str:
        """The reproducible representation of this instance."""
        args = f"{self.coefficient}, '{self.base}', {self.exponent}"
        return f"{self.__class__.__qualname__}({args})"


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

    # TODO: Allow arbitrary characters in the term variable as long as they are
    # not one of the operators or separators. This may require refactoring
    # Component and Term; it also may not be feasible at all. One of the first
    # steps may be redefining the full-term RE in terms of a compliment (i.e.,
    # what is *not* included in a term rather than what is included). Note that
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
        self._parse(Component(string))
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
        lists of algebraic terms (a.k.a simple components), regardless of order,
        after parsing. Two expressions with different numbers of terms are
        always false. If the expressions have the same number of terms, this
        method will sort the triples (first by variable, then by exponent, and
        finally by coefficient) and compare the sorted lists. Two expressions
        are equal if and only if their sorted lists of terms are equal.

        If `other` is a string, this method will first attempt to convert it to
        an expression.
        """
        if not isinstance(other, Expression):
            other = self._convert(other)
        if len(self._terms) != len(other._terms):
            return False
        key = operator.attrgetter('variable', 'exponent', 'coefficient')
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

    def reduce(self, *groups: Iterable[Term]):
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
            if term.variable in reduced:
                reduced[term.variable]['coefficient'] *= term.coefficient
                reduced[term.variable]['exponent'] += term.exponent
            else:
                attributes = {
                    'coefficient': term.coefficient,
                    'exponent': term.exponent,
                }
                reduced[term.variable] = attributes
        return [
            Component(v['coefficient'], k, v['exponent']).asterm
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

    @property
    def terms(self) -> List[Term]:
        """The algebraic terms in this expression."""
        return self._terms or [Term('1')]

    def _parse(self, component: Component):
        """Internal parsing logic."""
        resolved = self._resolve_operations(component)
        for component in resolved:
            if component.isconstant:
                self._scale *= float(component)
            elif not component.issimple:
                self._parse(component)
            else:
                term = component.asterm
                self._scale *= float(term.coefficient)
                normalized = Component(term.variable, term.exponent)
                self._terms.append(normalized.asterm)

    def _resolve_operations(self, component: Component) -> List[Component]:
        """Split the current component into operators and operands."""
        self._scale *= component.coefficient
        parts = self._parse_nested(component.base)
        operands = []
        operators = []
        for part in parts:
            if part in (self._multiply, self._divide):
                operators.append(part)
            else:
                args = (part, component.exponent)
                operands.append(Component(*args))
        if exception := self._check_operators(operators):
            raise exception(component)
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
        parts = []
        i = 0
        methods = [ # Order matters!
            self._find_term,
            self._find_number,
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

    def _find_number(self, string: str):
        """Check for a pure number at the start of `string`."""
        match = re.match(Term.n_re, string)
        if match:
            return match[0], match.end()

    def _find_term(self, string: str):
        """Check for an algebraic term at the start of `string`."""
        match = re.match(Term.full_re, string)
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
            exp = re.match(fr'\^{Term.e_re}', string[j:])
            if exp:
                j += exp.end()
            return string[:j], j


def guess_separators(string: str):
    """Attempt to determine which separators the given expression uses."""
    return (
        iterables.unique(string, {'(','[','{'}) or '(',
        iterables.unique(string, {')',']','}'}) or ')',
    )

