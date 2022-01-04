import abc
import collections.abc
import fractions
import functools
import numbers
import operator
import re
from typing import *
from typing import Pattern

from goats.common import iterables


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
    """An operand in an algebraic expression.

    Algebraic operands mainly exist to support the `~algebra.Expression` class.
    They may be simple or general, as described below.

    A simple algebraic operand has the form [c]b[^e] or c[^e], where `c` is a
    numerical coefficient, `b` is a string base, and `e` is a numerical
    exponent. Braces ('[]') denote an optional component that defaults to a
    value of 1. The `~algebra.Term` class formally represents simple algebraic
    operands; the form [c]b[^e] corresponds to a variable term and the form
    c[^e] corresponds to a constant term.

    Examples include::
    * `'1'`: unity / multiplicative identity
    * `'1^n'` (`n` real): constant equivalent to unity
    * `'m^n'` (`m`, `n` real): arbitrary constant
    * `'V'`: variable 'V' with coefficient 1 and exponent 1
    * `'V^n'` (`n` real): variable 'V' with coefficient 1 and exponent n
    * `'mV'` (`m` real): variable 'V' with coefficient m and exponent 1
    * `'mV^n'` (`m`, `n` real): variable 'V' with coefficient m and exponent n

    Note that the base of a variable term may comprise multiple characters as
    long as it does not begin with a digit, which this class will interpret as
    part of the coefficient.

    A general algebraic operand consists of simpler operands (though not
    necessarily formally simple operands) combined with algebraic operators and
    separators. All formally simple operands are general operands. The following
    are examples of (non-simple) general algebraic operands::
    * `'a * b^2'` <=> `'(a * b^2)'` <=> `'(a * b^2)^1'`
    * `'(a * b^2)^3'`
    * `'(a * b^2)^3/2'`
    * `'((a / b^2)^3 * c)^2'`
    * `'(a / b^2)^3 * c^2'`
    * `'a / (2 * 4b)'`
    * `'(2a * b)^3 / (4 * c)'`

    There are many more ways to construct a general operand than a simple
    operand. This is by design, to support building instances of
    `~algebra.Expression` with `~algebra.Parser`.
    """

    def __init__(
        self,
        coefficient: numbers.Real=1,
        base: str='1',
        exponent: numbers.Real=1,
    ) -> None:
        self.coefficient = coefficient
        """The numerical coefficient."""
        self.base = base
        """The base term or complex."""
        self.exponent = exponent
        """The numerical exponent."""

    @property
    def attrs(self):
        """The current coefficient, base, and exponent."""
        return (self.coefficient, self.base, self.exponent)

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
        if not other.base == self.base:
            return False
        return all(
            float(getattr(other, attr)) == float(getattr(self, attr))
            for attr in ('exponent', 'coefficient')
        )

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
            string = f"{self.coefficient}{string}"
        return string


class Term(Operand):
    """An algebraic operand with an irreducible base."""

    def format(self, style: str=None):
        """Format this term."""
        coefficient = self._format_coefficient()
        if self.base == '1':
            return f"{coefficient}"
        exponent = self._format_exponent(style)
        return f"{coefficient}{self.base}{exponent}"

    def _format_coefficient(self):
        """Format the coefficient for printing."""
        if self.base != '1' and self.coefficient == 1:
            return ''
        if float(self.coefficient) == int(self.coefficient):
            return str(int(self.coefficient))
        if (
            isinstance(self.coefficient, fractions.Fraction)
            and
            self.base != '1'
            and
            self.coefficient.denominator != 1
        ): return f"({self.coefficient})"
        return str(self.coefficient)

    def _format_exponent(self, style: str):
        """Format the current exponent for printing."""
        if self.base == '1' or self.exponent == 1:
            return ''
        if not style:
            return f"^{self.exponent}"
        if 'tex' in style.lower():
            return f"^{{{self.exponent}}}"
        raise ValueError(f"Can't format {self.exponent}")

    def __float__(self) -> float:
        """Called for float(self)."""
        if self.base == '1':
            return float(self.coefficient)
        errmsg = f"Can't convert term with base {self.base!r} to float"
        raise TypeError(errmsg)

    def __eq__(self, other) -> bool:
        if isinstance(other, numbers.Real):
            return float(self) == float(other)
        return super().__eq__(other)


def asterms(these: Iterable[str]):
    """Convert strings to terms, if possible."""
    return [OperandFactory().create(this) for this in these]


class OperandTypeError(TypeError):
    pass


class OperandValueError(ValueError):
    pass


class Parsed(NamedTuple):
    """The result of parsing an algebraic string."""

    result: Part
    remainder: str
    start: int=0
    end: int=-1

    def __bool__(self) -> bool:
        """The truth value of this object."""
        return bool(self.result)


@runtime_checkable
class Matched(Protocol):
    """Protocol for pattern-matching results."""

    @abc.abstractmethod
    def groupdict(self) -> Dict[str, Any]:
        pass

    @abc.abstractmethod
    def end(self) -> int:
        pass


class MatchResult(Matched):
    """A simple objects that mimics a `re.Match` object.

    This class exists to provide a single interface for parsing and
    pattern-matching methods, some of which use the `re` module and some of
    which use custom code.
    """

    def __init__(self, match: re.Match=None, **kwargs) -> None:
        self._kwargs = self._update_kwargs(kwargs, match=match)

    def _update_kwargs(self, original: dict, match: re.Match=None):
        """Update the user keyword arguments from the `Match` object."""
        updates = {
            'groupdict': match.groupdict(),
            'end': match.end(),
        } if match else {}
        original.update(**updates)
        return original

    def groupdict(self):
        return self._kwargs.get('groupdict')

    def end(self):
        return self._kwargs.get('end')

    def __bool__(self) -> bool:
        """The truth value of this object."""
        return bool(self._kwargs)


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
                    end=match.end(),
                )
        return Parsed(result=None, remainder=string)


class OperandFactory(PartFactory):
    """A factory that produces algebraic operands."""

    rational = r""" # Modeled after `fractions._RATIONAL_FORMAT`
        [-+]?                 # an optional sign, ...
        (?=\d|\.\d)           # ... only if followed by <digit> or .<digit>
        \d*                   # possibly empty numerator
        (?:                   # followed by ...
            (?:/\d+)?         # ... an optional denominator
        |                     # OR
            (?:\.\d*)?        # ... an optional fractional part
            (?:[eE][-+]?\d+)? #     and optional exponent
        )
    """
    base = r"""
        [a-zA-Z#_]+ # one or more accepted non-digit character(s)
        \d*         # followed by optional digits
    """

    def __init__(
        self,
        opening: str='(',
        closing: str=')',
        raising: str='^',
    ) -> None:
        exponent = fr'\{raising}{self.rational}'
        self.patterns = {
            'constant': re.compile(
                fr'(?P<coefficient>{self.rational})'
                fr'(?P<exponent>{exponent})?',
                re.VERBOSE,
            ),
            'variable': re.compile(
                fr'(?P<coefficient>{self.rational})?'
                fr'(?P<base>{self.base})'
                fr'(?P<exponent>{exponent})?',
                re.VERBOSE,
            ),
            'complex': re.compile(
                fr'(?P<coefficient>{self.rational})?'
                fr'(?P<base>\{opening}.+?\{closing})'
                fr'(?P<exponent>{exponent})?',
                re.VERBOSE,
            ),
            'exponent': re.compile(exponent, re.VERBOSE),
            'opening': re.compile(fr'\{opening}', re.VERBOSE),
            'closing': re.compile(fr'\{closing}', re.VERBOSE),
            'raising': re.compile(fr'\{raising}', re.VERBOSE)
        }
        """Compiled regular expressions for algebraic operands."""
        self.recipes = {
            'simplex': {
                'type': Term,
                'match': self._match_simplex,
                'fullmatch': self._fullmatch_simplex,
            },
            'complex': {
                'type': Operand,
                'match': self._match_complex,
                'fullmatch': self._fullmatch_complex,
            },
        }

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
                return self.standardize(base=arg, fill=True)
            if isinstance(arg, numbers.Real):
                return self.standardize(coefficient=arg, fill=True)
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
                return self.standardize(
                    base=args[0],
                    exponent=args[1],
                    fill=True,
                )
            argtypes = zip(args, (numbers.Real, (numbers.Real, str)))
            implied_base = all(isinstance(a, t) for a, t in argtypes)
            if implied_base:
                return self.standardize(
                    coefficient=args[0],
                    exponent=args[1],
                    fill=True,
                )
            argtypes = zip(args, (numbers.Real, str))
            implied_exponent = all(isinstance(a, t) for a, t in argtypes)
            if implied_exponent:
                return self.standardize(
                    coefficient=args[0],
                    base=args[1],
                    fill=True,
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
            return self.standardize(
                coefficient=args[0],
                base=args[1],
                exponent=args[2],
                fill=True,
            )
        raise OperandValueError(
            f"{self.__class__.__qualname__}"
            f" accepts 1, 2, or 3 arguments"
            f" (got {nargs})"
        )

    # NOTE: The difference between `create` and `parse` is: 
    # * `create` resolves input into (coefficient, base, exponent), then creates
    #   an appropriate operand from the base string, applies the input
    #   coefficient and exponent, and finally returns the operand.
    # * `parse` attempts to match an operand at the start of a string, then
    #   creates an appropriate operand from only that substring, and finally
    #   returns the operand and the remainder of the string.

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
        This method will create the most general algebraic operand possible from
        the initial string. It will parse a simple algebraic operand into a
        coefficient, variable, and exponent but it will not attempt to fully
        parse a general algebraic operand into simpler operands (i.e. algebraic
        terms). In other words, it will do as little work as possible to extract
        a coefficient and exponent, and the expression on which they operate. If
        all attempts to determine appropriate attributes fail, it will simply
        return the string representation of the initial argument with default
        coefficient and exponent.

        The following examples use the general algebraic operands described in
        `~algebra.Operand` to illustrate the minimal parsing described above::
        * `'a * b^2'` <=> `'(a * b^2)'` <=> `'(a * b^2)^1'` -> `1, 'a * b^2', 1`
        * `'2a * b^2'` -> `1, '2a * b^2', 1`
        * `'2(a * b^2)'` -> `2, 'a * b^2', 1`
        * `'(a * b^2)^3'` -> `1, 'a * b^2', 3`
        * `'2(a * b^2)^3'` -> `2, 'a * b^2', 3`
        * `'(a * b^2)^3/2'` -> `'a * b^2', '3/2'`
        * `'((a / b^2)^3 * c)^2'` -> `'(a / b^2)^3 * c', 2`
        * `'(a / b^2)^3 * c^2'` -> `'(a / b^2)^3 * c^2', 1`
        """
        c0, b0, e0 = self.normalize(*args).values()
        ends = (b0[0], b0[-1])
        if any(self.patterns['raising'].match(c) for c in ends):
            raise OperandValueError(b0) from None
        recipe = self.recipes['simplex']
        if match := self.match_maker(b0, recipe['fullmatch']):
            standard = self.standardize(**match.groupdict(), fill=True)
            c1, base, e1 = standard.values()
            if base == '1':
                coefficient = c0 * (c1 ** e1) ** e0
                return recipe['type'](float(coefficient))
            coefficient = c0 * (c1 ** e0)
            exponent = e1 * e0
            return recipe['type'](coefficient, base, exponent)
        recipe = self.recipes['complex']
        if match := self.match_maker(b0, recipe['fullmatch']):
            standard = self.standardize(**match.groupdict(), fill=True)
            c1, base, e1 = standard.values()
            coefficient = c0 * (c1 ** e0)
            exponent = e1 * e0
            # TODO: Generalize the following so we can loop over recipes.
            inside = self.unpack(base)
            if interior := self.create(inside):
                exponent *= interior.exponent
                coefficient *= interior.coefficient ** exponent
                base = interior.base
            else:
                base = inside
            return recipe['type'](coefficient, base, exponent)
        if not strict:
            return Operand(c0, b0, e0)

    def parse(self, string: str):
        """Extract an operand at the start of `string`, possible."""
        stripped = string.strip()
        for recipe in self.recipes.values():
            if match := self.match_maker(stripped, recipe['match']):
                standard = self.standardize(**match.groupdict(), fill=True)
                return Parsed(
                    result=recipe['type'](*standard.values()),
                    remainder=stripped[match.end():],
                    end=match.end(),
                )
        return Parsed(result=None, remainder=string)

    def match_maker(
        self,
        string: str,
        func: Callable[[str], Matched],
    ) -> Optional[Matched]:
        """Makes you a match.

        This method will create a new string by removing bounding parentheses
        (if they exist), then pass the updated string to the given function, and
        returns the result if it satisfies certain criteria.

        Parameters
        ----------
        string
            The string to match.

        func : callable(str) -> Matched or mapping
            A callable object that takes a single string argument and possibly
            returns an object that either conforms to the `~algebra.Matched`
            protocol (e.g., `re.Match` or `~algebra.MatchResult`) or exposes a
            mapping that can initialize an instance of `~algebra.MatchResult`.

        Returns
        -------
        A Matched object or `None`
        """
        result = func(string)
        if not result:
            return
        if isinstance(result, Matched):
            return result
        if isinstance(result, Mapping):
            return MatchResult(**result)

    def _fullmatch_simplex(self, string: str):
        """Attempt to match `string` to a full term pattern."""
        match = self._match_simplex(string)
        if match and match.end() == len(string):
            return match

    def _match_simplex(self, string: str):
        """Attempt to find an irreducible term at the start of `string`.

        Notes
        -----
        This method tries to match the 'variable' pattern before the 'constant'
        pattern because `re.match` will find a match for 'constant' at the start
        of any variable term with an explicit coefficient.
        """
        bounded = self.find_bounded(string, match=True)
        if bounded:
            stripped = self.unpack(bounded.result)
            for key in ('variable', 'constant'):
                if match := self.patterns[key].fullmatch(stripped):
                    return MatchResult(
                        groupdict=match.groupdict(),
                        end=bounded.end,
                    )
        for key in ('variable', 'constant'):
            if match := self.patterns[key].match(string):
                return MatchResult(match=match)

    def _fullmatch_complex(self, string: str):
        """Attempt to match `string` to the form of a complex operand."""
        match = self._match_complex(string)
        if match and match.end() == len(string):
            return match

    def _match_complex(self, string: str):
        """Attempt to find a complex operand at the start of `string`."""
        bounded = self.find_bounded(string, strip=True, match=True)
        if not bounded:
            return
        result = {}
        end = bounded.end
        if leading := self._match_simplex(string):
            standard = self.standardize(**leading.groupdict())
            coefficient = standard['coefficient'] ** standard['exponent']
            result['coefficient'] = coefficient
        matches = tuple(self.patterns['exponent'].finditer(bounded.result))
        if matches:
            final = matches[-1]
            base = bounded.result[:final.start()]
            result['exponent'] = final[0]
        else:
            base = bounded.result
        result['base'] = self.unpack(base)
        # breakpoint()
        return MatchResult(groupdict=result, end=end)

    def standardize(
        self,
        fill: bool=False,
        **given
    ) -> Dict[str, Union[float, int, str, fractions.Fraction]]:
        """Cast to appropriate types and fill in defaults, if necessary."""
        full = {
            'coefficient': {'callable': self._standard_coefficient},
            'base': {'callable': self._standard_base},
            'exponent': {'callable': self._standard_exponent},
        }
        default = self.fill_defaults(**dict.fromkeys(full.keys()))
        updatable = full.copy() if fill else {k: full[k] for k in given}
        return {
            key: attr['callable'](given.get(key) or default[key])
            for key, attr in updatable.items()
        }

    def _standard_coefficient(self, v):
        """Convert input to a standard coefficient."""
        return fractions.Fraction(v or 1)

    def _standard_base(self, v):
        """Convert input to a standard base."""
        return str(v)

    def _standard_exponent(self, v):
        """Convert input to a standard exponent."""
        if isinstance(v, str):
            v = self.patterns['raising'].sub('', v)
        return fractions.Fraction(v or 1)

    def fill_defaults(self, **given):
        """Return the default value for any explicitly null arguments.

        If the given key-value pairs contain an argument with a null value
        (e.g., `None`), this method will replace it with the default value. It
        will pass all other values through unaltered and will not fill in
        default values corresponding to other keys.
        """
        defaults = {
            'coefficient': 1,
            'base': '1',
            'exponent': 1,
        }
        given.update(
            {
                key: defaults[key]
                for key, value in given.items()
                if key in defaults and not value
            }
        )
        return given

    def find_bounded(
        self,
        string: str,
        strip: bool=False,
        match: bool=False,
    ) -> Optional[Parsed]:
        """Find the first bounded operand in `string`.
        
        A bounded operand is any collection of valid operands or operators that
        is bounded on the left by the opening separator and on the right by the
        closing separator. Opening and closing separators are an immutable
        attribute of an instance of this class.

        Parameters
        ----------
        string
            The string in which to search for a bounded operand.

        strip : bool, default=False
            If True, remove the bounding separators from the result.

        match : bool, default=False
            If True, restrict search to beginning of string (similar to
            the `match` functions of the `re` module).

        Returns
        -------
        Parsed
            A `~algebra.Parsed` object built from the result, if found;
            otherwise, `None`.
        """
        initialized = False
        count = 0
        start = 0
        for i, c in enumerate(string):
            if self.patterns['opening'].match(c):
                count += 1
                if not initialized:
                    start = i
                    initialized = True
            elif self.patterns['closing'].match(c):
                count -= 1
            if match and start > 0:
                return
            if initialized and count == 0:
                end = i+1
                if exp := self.patterns['exponent'].match(string[end:]):
                    end += exp.end()
                result = string[start:end]
                remainder = string[end:]
                if strip:
                    result = self.strip_separators(result)
                return Parsed(
                    result=result,
                    remainder=remainder,
                    start=start,
                    end=end,
                )

    def entire(self, string: str):
        """True if `string` is completely bounded by separators."""
        opened = self.patterns['opening'].match(string[0])
        closed = self.patterns['closing'].match(string[-1])
        if not (opened and closed):
            return False
        if bounded := self.find_bounded(string):
            return not bounded.remainder

    def unpack(self, string: str):
        """Remove bounding separators from `string`."""
        if not self.entire(string):
            return string
        inside = self.strip_separators(string)
        while self.entire(inside):
            inside = self.strip_separators(inside)
        return inside

    # TODO: Refactor to reuse code with `entire`.
    def strip_separators(self, string: str):
        """Remove one opening and one closing separator."""
        opened = self.patterns['opening'].match(string[0])
        closed = self.patterns['closing'].match(string[-1])
        if not (opened and closed):
            return string
        string = self.patterns['opening'].sub('', string, count=1)
        string = self.patterns['closing'].sub('', string[::-1], count=1)
        return string[::-1]


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
        # TODO: Consider extracting all coefficients, at least as separate
        # constant terms.
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


class OperandError(TypeError):
    pass


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
        self.reduce()

    def _standardize(self, expression: Union[str, iterables.Separable]) -> str:
        """Convert user input to a standard format."""
        if not expression:
            return '1'
        if isinstance(expression, str):
            return expression
        return ' * '.join(f"({part})" for part in expression)

    @property
    def terms(self) -> List[Term]:
        """The algebraic terms in this expression."""
        return self._terms or []

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
        fracs = [
            fractions.Fraction(v['coefficient'])
            for v in reduced.values()
        ]
        constant = functools.reduce(lambda x, y: x*y, fracs)
        return [Term(coefficient=constant)] + [
            Term(base=k, exponent=v['exponent'])
            for k, v in reduced.items()
            if k != '1' and v['exponent'] != 0
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

