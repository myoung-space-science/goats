import abc
import collections.abc
import fractions
import functools
import itertools
import numbers
import re
from operator import attrgetter
import typing

from goats.core import iterables


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

    def __hash__(self) -> str:
        """Compute instance hash (e.g., for use as `dict` key)."""
        return hash(self.operation)

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

    def __call__(self, value: numbers.Real):
        """Evaluate a variable term at this value.
        
        This method will attempt to substitute `value` for this term's `base`
        attribute. If successful, it will return a constant term that the caller
        may cast to `int` or `float` type.

        Parameters
        ----------
        value : real
            The numerical value at which to evaluate this term.

        Returns
        -------
        `~algebra.Term`
            A new instance of this class equivalent to the constant numerical
            value of this term when `base == value`.
        """
        if not isinstance(value, numbers.Real):
            errmsg = f"Can't evaluate term with value {value!r}"
            raise TypeError(errmsg)
        if self.base != '1':
            coefficient = self.coefficient * (value ** self.exponent)
            return type(self)(coefficient=coefficient)
        errmsg = f"Can't evaluate term with base {self.base!r}"
        raise TypeError(errmsg)

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
        if self._ambiguous_coefficient():
            return f"({self.coefficient})"
        if float(self.coefficient) == int(self.coefficient):
            return str(int(self.coefficient))
        return str(self.coefficient)

    def _ambiguous_coefficient(self):
        """True if this term's coefficient needs to be grouped."""
        return (
            isinstance(self.coefficient, fractions.Fraction)
            and self.base != '1'
            and self.coefficient.denominator != 1
        )

    def _format_exponent(self, style: str):
        """Format the current exponent for printing."""
        if self.base == '1' or self.exponent == 1:
            return ''
        if not style:
            return f"^{self.exponent}"
        if 'tex' in style.lower():
            return f"^{{{self.exponent}}}"
        raise ValueError(f"Can't format {self.exponent}")

    def __int__(self):
        """Called for int(self)."""
        return self._cast_to(int)

    def __float__(self):
        """Called for float(self)."""
        return self._cast_to(float)

    _T = typing.TypeVar('_T', int, float)
    def _cast_to(self, __type: _T) -> _T:
        """Internal method for casting to numeric type."""
        if self.base == '1':
            return __type(self.coefficient)
        errmsg = f"Can't convert term with base {self.base!r} to {__type}"
        raise TypeError(errmsg) from None

    def __eq__(self, other) -> bool:
        if isinstance(other, numbers.Real):
            return float(self) == float(other)
        if isinstance(other, str):
            term = OperandFactory().create(other)
            return super().__eq__(term)
        return super().__eq__(other)


def asterms(these: typing.Iterable[str]):
    """Convert strings to terms, if possible."""
    return [OperandFactory().create(this) for this in these]


class OperandTypeError(TypeError):
    pass


class OperandValueError(ValueError):
    pass


T = typing.TypeVar('T', bound=Part)
class PartMatch(typing.Generic[T], iterables.ReprStrMixin):
    """An object that represents the result of a RE pattern match."""

    def __new__(cls, result, context):
        """Check argument types."""
        if not isinstance(result, Part):
            raise TypeError(
                f"Result must be an Part"
                f", not {type(result)}"
            ) from None
        if not isinstance(context, (re.Match, typing.Mapping)):
            raise TypeError(
                f"Context may be a Match object or a Mapping"
                f", not {type(context)}"
            )
        return super().__new__(cls)

    def __init__(
        self,
        result: T,
        context: typing.Union[re.Match, typing.Mapping],
    ) -> None:
        self.result = result
        """The result of the match attempt."""
        self._context = self._set_context(context)

    def _set_context(self, user: typing.Any) -> dict:
        """Normalize the instance context from user input."""
        if isinstance(user, re.Match):
            return self._set_from_match(user)
        if isinstance(user, typing.Mapping):
            return self._set_from_mapping(user)

    def _set_from_match(self, match: re.Match):
        """Set the instance context from a match object."""
        return {
            'start': match.start(),
            'end': match.end(),
            'string': match.string,
        }

    def _set_from_mapping(self, mapping: typing.Mapping):
        """Set the instance context from a mapping."""
        attrs = (
            'start',
            'end',
            'string',
        )
        return {k: mapping.get(k) for k in attrs}

    @property
    def start(self) -> int:
        """The starting index in `string` of the match."""
        return self._context['start']

    @property
    def end(self) -> int:
        """The ending index in `string` of the match."""
        return self._context['end']

    @property
    def string(self) -> str:
        """The target string."""
        return self._context['string']

    @property
    def remainder(self) -> str:
        """The unparsed portion of `string` after `end`."""
        return self.string[self.end:]

    def __bool__(self) -> bool:
        """Always true, like `re.Match`."""
        return True

    def __str__(self) -> str:
        """A simplified representation of this object."""
        attrs = {
            'result': 'result',
            'context': '_context',
        }
        return ', '.join(f"{k}: {getattr(self, v)}" for k, v in attrs.items())


@typing.runtime_checkable
class PartFactory(typing.Protocol):
    """Protocol for algebraic factories."""

    @abc.abstractmethod
    def parse(self) -> PartMatch:
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
            'sqrt': re.compile(r'\s*sqrt\s*')
        }
        """Compiled regular expressions for algebraic operators."""

    def parse(self, string: str):
        """Extract an operator at the start of `string`, possible."""
        for key in self.patterns:
            if match := self.patterns[key].match(string):
                return PartMatch(
                    result=Operator(key),
                    context=match,
                )


class OperandFactory(PartFactory):
    """A factory that produces algebraic operands."""

    rational = r""" # Modeled after `fractions._RATIONAL_FORMAT`
        [-+]?                 # an optional sign, ...
        (?=\d|\.\d)           # ... only if followed by <digit> or .<digit>
        \d*                   # possibly empty numerator
        (?:                   # followed by ...
            (?:/\d+?)         # ... an optional denominator
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

    _argtypes = {
        'coefficient': numbers.Real,
        'base': str,
        'exponent': (numbers.Real, str),
    }
    @classmethod
    def isvalid(cls, name: str, this: typing.Any):
        """True if `this` is valid for use as the named attribute."""
        return isinstance(this, cls._argtypes[name])

    def normalize(self, *args):
        """Extract attributes from the given argument(s)."""
        try:
            nargs = len(args)
        except TypeError:
            raise OperandTypeError(args) from None
        if nargs == 1:
            return self._length_1(args)
        if nargs == 2:
            return self._length_2(args)
        if nargs == 3:
            return self._length_3(args)
        raise OperandValueError(
            f"{self.__class__.__qualname__}"
            f" accepts 1, 2, or 3 arguments"
            f" (got {nargs})"
        )

    def _length_1(self, args: typing.Any):
        """Normalize a length-1 argument tuple, if possible.

        A length-1 tuple may represent either:
        - coefficient <Real>
        - base <str>

        If it has one of these forms, this method will substitute the default
        value for the missing attributes; otherwise, it will raise an exception.
        """
        arg = args[0]
        names = ('base', 'coefficient')
        for name in names:
            given = {name: arg}
            if self.isvalid(name, arg):
                return self.standardize(fill=True, **given)
        raise OperandTypeError(
            "A single argument may be either"
            " a coefficient <Real> or a base <str>;"
            f" not {type(arg)}"
        )

    def _length_2(self, args: typing.Any):
        """Normalize a length-2 argument tuple, if possible.

        A length-2 tuple may represent either:
        - (base <str>, exponent <Real or str>)
        - (coefficient <Real>, exponent <Real or str>)
        - (coefficient <Real>, base <str>)

        If it has one of these forms, this method will substitute the default
        value for the missing attribute; otherwise, it will raise an exception.
        """
        combinations = itertools.combinations(self._argtypes, 2)
        for names in combinations:
            given = dict(zip(names, args))
            if all(self.isvalid(name, arg) for name, arg in given.items()):
                return self.standardize(fill=True, **given)
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

    def _length_3(self, args: typing.Any):
        """Normalize a length-3 argument tuple.

        A length-3 tuple must have the form (coefficient <Real>, base <str>,
        exponent <Real or str>).
        """
        return self.standardize(
            coefficient=args[0],
            base=args[1],
            exponent=args[2],
            fill=True,
        )

    def create(self, *args, strict: bool=False) -> typing.Optional[Operand]:
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
            (with default coefficient and exponent, if necessary) as an instance
            of `~algebra.Operand`.

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
        all attempts to determine appropriate attributes fail, the value of
        `strict` controls its return behavior.
        
        See note at `parse` for differences between this method and that.

        The following examples use the general algebraic operands described in
        `~algebra.Operand` to illustrate the minimal parsing described above::
        * `'a * b^2'` <=> `'(a * b^2)'` <=> `'(a * b^2)^1'` -> `1, 'a * b^2', 1`
        * `'2a * b^2'` -> `1, '2a * b^2', 1`
        * `'2(a * b^2)'` -> `2, 'a * b^2', 1`
        * `'(a * b^2)^3'` -> `1, 'a * b^2', 3`
        * `'2(a * b^2)^3'` -> `2, 'a * b^2', 3`
        * `'(a * b^2)^3/2'` -> `1, 'a * b^2', '3/2'`
        * `'((a / b^2)^3 * c)^2'` -> `1, '(a / b^2)^3 * c', 2`
        * `'(a / b^2)^3 * c^2'` -> `1, '(a / b^2)^3 * c^2', 1`
        """
        c0, b0, e0 = self.normalize(*args).values()
        ends = (b0[0], b0[-1])
        if any(self.patterns['raising'].match(c) for c in ends):
            raise OperandValueError(b0) from None
        match = self.search(b0, mode='fullmatch')
        if not match:
            if not strict:
                return Operand(c0, b0, e0)
            return
        if not isinstance(match.result, Operand):
            raise TypeError(
                f"Expected Operand but got {type(match.result)}"
            ) from None
        c1, base, e1 = match.result.attrs
        coefficient = c0 * (c1 ** e0)
        exponent = e1 * e0
        if not isinstance(match.result, Term):
            interior = self.create(base)
            if isinstance(interior, Term):
                coefficient *= interior.coefficient ** exponent
                base = interior.base
                exponent *= interior.exponent
                return Term(
                    coefficient=coefficient,
                    base=base,
                    exponent=exponent,
                )
            return Operand(
                coefficient=coefficient,
                base=base,
                exponent=exponent,
            )
        return Term(
            coefficient=coefficient,
            base=base,
            exponent=exponent,
        )

    def parse(self, string: str):
        """Extract an operand at the start of `string`, possible.
        
        Notes
        -----
        The primary difference between `~create` and `~parse` is as follows::
        * `~create` resolves input into (coefficient, base, exponent), then
          creates an appropriate operand from the base string, then applies the
          input coefficient and exponent, and finally returns the operand.
        * `~parse` attempts to match an operand at the start of a string, then
          creates an appropriate operand from only that substring, and finally
          returns the operand and the remainder of the string.
        """
        stripped = string.strip()
        if match := self.search(stripped):
            return match

    def search(self, string: str, **kwargs):
        """Search for an operand in the given string.
        
        Parameters
        ----------
        string
            The string to which to apply pattern-matching methods in an attempt
            to find an appropriate operand.

        **kwargs
            Keyword arguments to pass to the pattern-matching methods.

        Returns
        -------
        `~algebra.PartMatch` or `None`
            An object representing the matched substring and contextual
            information, if any attemp to match was successful. If no method
            found an operand in `string` (subject to any given keyword
            arguments), this method will return `None`.
        """
        methods = (
            self._match_simplex,
            self._match_complex,
        )
        if match := iterables.apply(methods, string, **kwargs):
            return match

    def _match_simplex(
        self,
        string: str,
        mode: str='match',
        start: int=0,
    ) -> typing.Optional[PartMatch[Operand]]:
        """Attempt to find an irreducible term at the start of `string`.

        Notes
        -----
        This method tries to match the 'variable' pattern before the 'constant'
        pattern because `re.match` will find a match for 'constant' at the start
        of any variable term with an explicit coefficient.
        """
        target = string[start:]
        matches = {
            key: self._get_match_method(key, mode)(target)
            for key in ('variable', 'constant')
        }
        if not any(matches.values()):
            return
        if all(matches.values()):
            same = matches['variable'][0] == matches['constant'][0]
            key = 'constant' if same else 'variable'
            build_method = self._get_build_method(key)
            return build_method(matches[key])
        for key, match in matches.items():
            if match:
                return self._get_build_method(key)(match)

    def _get_match_method(
        self,
        pattern: str,
        mode: str,
    ) -> typing.Callable[[str], re.Match]:
        """Look up the appropriate matching method for `pattern` and `mode`."""
        return getattr(self.patterns[pattern], mode)

    def _get_build_method(
        self,
        pattern: str,
    ) -> typing.Callable[[re.Match], PartMatch[Operand]]:
        """Look up the appropriate building method for `pattern`."""
        return getattr(self, f'_build_{pattern}')

    def _build_variable(self, match: re.Match):
        """Build a variable term from a match object."""
        standard = self.standardize(**match.groupdict(), fill=True)
        return PartMatch(
            result=Term(**standard),
            context=match,
        )

    def _build_constant(self, match: re.Match):
        """Build a constant term from a match object."""
        standard = self.standardize(**match.groupdict(), fill=True)
        coefficient = standard['coefficient'] ** standard['exponent']
        return PartMatch(
            result=Term(coefficient=float(coefficient)),
            context=match,
        )

    def _match_complex(
        self,
        string: str,
        mode: str='match',
        start: int=0,
    ) -> typing.Optional[PartMatch[Operand]]:
        """Attempt to match a complex operand at the start of `string`."""
        target = string[start:]
        bounds = self.find_bounds(target)
        if not bounds:
            return
        i0, end = bounds
        result = {'base': target[i0+1:end-1]}
        if match := self._match_simplex(target[:i0], mode='fullmatch'):
            result['coefficient'] = match.result.coefficient
            i0 = 0
        if mode == 'match' and i0 != 0:
            return
        if exp := self.patterns['exponent'].match(target[end:]):
            result['exponent'] = exp[0]
            end += exp.end()
        if mode == 'fullmatch' and (i0, end) != (0, len(target)):
            return
        standard = self.standardize(**result, fill=True)
        return PartMatch(
            result=Operand(**standard),
            context={'end': end, 'start': start, 'string': string},
        )

    def find_bounds(self, string: str):
        """Find the indices of the first bounded substring, if any.
        
        A bounded substring is any portion of `string` that is bounded on the
        left by the opening separator and on the right by the closing separator.
        Opening and closing separators are an immutable attribute of an instance
        of this class.

        Parameters
        ----------
        string
            The string in which to search for a bounded substring.

        Returns
        -------
        tuple of int, or `None`
            The index of the leftmost opening separator and the index of the
            first character beyond the rightmost closing separator (possibly the
            end), if there is a bounded substring; otherwise, `None`. The
            convention is such that if `start, end = find_bounds(string)`,
            `string[start:end]` will produce the bounded substring with bounds.

        Examples
        --------
        Define a list of test strings::

            >>> strings = [
            ...     '(a*b)',
            ...     '(a*b)^2',
            ...     '3(a*b)',
            ...     '3(a*b)^2',
            ...     '3(a*b)^2 * (c*d)',
            ...     '4a^4',
            ...     '3(4a^4)^3',
            ...     '2(3(4a^4)^3)^2',
            ... ]

        Create an instance of this class with the default operators and
        separators::

            >>> operand = algebra.OperandFactory()

        Find the bounding indices of each test string, if any, and display the
        corresponding substring::

            >>> for string in strings:
            ...     bounds = operand.find_bounds(string)
            ...     print(f"{bounds}: {string!r} -> ", end='')
            ...     if bounds:
            ...         start, end = bounds
            ...         result = f"{string[start:end]}"
            ...     else:
            ...         result = string
            ...     print(f"{result!r}")
            ... 
            (0, 5): '(a*b)' -> '(a*b)'
            (0, 5): '(a*b)^2' -> '(a*b)'
            (1, 6): '3(a*b)' -> '(a*b)'
            (1, 6): '3(a*b)^2' -> '(a*b)'
            (1, 6): '3(a*b)^2 * (c*d)' -> '(a*b)'
            None: '4a^4' -> '4a^4'
            (1, 7): '3(4a^4)^3' -> '(4a^4)'
            (1, 12): '2(3(4a^4)^3)^2' -> '(3(4a^4)^3)'
        """
        initialized = False
        count = 0
        i0 = 0
        for i, c in enumerate(string):
            if self.patterns['opening'].match(c):
                count += 1
                if not initialized:
                    i0 = i
                    initialized = True
            elif self.patterns['closing'].match(c):
                count -= 1
            if initialized and count == 0:
                return i0, i+1

    def standardize(
        self,
        fill: bool=False,
        **given
    ) -> typing.Dict[str, typing.Union[float, int, str, fractions.Fraction]]:
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

    def __init__(self, arg: typing.Any) -> None:
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


class ParsingValueError(ValueError):
    """Cannot create an expression from the given string."""
    pass


class Iteration(iterables.ReprStrMixin):
    """An object that keeps track of parsing attributes."""

    __slots__ = ('string', 'operator', 'operand')

    def __init__(
        self,
        string: str,
        operator: Operator=None,
        operand: Operand=None,
    ) -> None:
        self.string = string
        self.operator = operator
        self.operand = operand

    @property
    def _attrs(self):
        """Internal mapping of current attribute values."""
        return {name: getattr(self, name) for name in self.__slots__}

    def copy(self):
        """Make a copy of this instance."""
        return type(self)(**self._attrs)

    def __str__(self):
        """A simplified representation of this object."""
        return ', '.join(f"{k}={v!r}" for k, v in self._attrs.items())


class Parser:
    """A tool for parsing algebraic expressions."""

    def __init__(
        self,
        multiply: str='*',
        divide: str='/',
        opening: str='(',
        closing: str=')',
        raising: str='^',
        operator_order: str='ignore',
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

        operator_order : {'ignore', 'error'}
            Determines how the parser responds when operator order violates NIST
            guidelines. If set to `'ignore'` (default), it will treat operators
            independent of one another. If set to `'error'`, the parser will
            raise an exception based on the type of violation.
        """
        self.operands = OperandFactory(opening, closing, raising)
        self.operators = OperatorFactory(multiply, divide)
        self.parsers = (self.operands, self.operators)
        self.tokens = {
            'multiply': multiply,
            'divide': divide,
            'opening': opening,
            'closing': closing,
            'raising': raising,
        }
        self._operator_order = operator_order

    def parse(self, string: str):
        """Resolve the given string into individual terms."""
        operand = Operand(base=string)
        return self._resolve_operations(operand)

    def _resolve_operations(self, current: Operand) -> typing.List[Term]:
        """Separate an algebraic group into operators and operands."""
        operands = self._parse_operand(current)
        return [
            term for operand in operands
            for term in self._update_terms(operand)
        ] + [Term(coefficient=current.coefficient)]

    def _parse_operand(self, initial: Operand) -> typing.List[Operand]:
        """Resolve a general operand into simpler operands.

        This method parses known operators and operands from the initial operand
        while preserving nested groups in the latter. Calling code may then pass
        those nested groups back in for further parsing.
        """
        operands = []
        current = Iteration(initial.base)
        previous = current.copy()
        while current.string:
            current = self._get_operator(initial, current, previous)
            current = self._get_operand(initial, current)
            if new := self._compute_operand(current):
                operands.append(new)
            previous = current.copy()
            current = Iteration(previous.string)
        return operands

    def _get_operator(
        self,
        initial: Operand,
        current: Iteration,
        previous: Iteration,
    ) -> Iteration:
        """Attempt to parse an operator from the current string."""
        if parsed := self.operators.parse(current.string):
            current.operator = parsed.result
            if exception := self._operator_error(
                    current.operator,
                    previous.operator,
                ): raise exception(initial)
            current.string = parsed.remainder
        return current

    def _get_operand(
        self,
        initial: Operand,
        current: Iteration,
    ) -> Iteration:
        """Attempt to parse an operand from the current string."""
        if parsed := self.operands.parse(current.string):
            current.operand = parsed.result ** initial.exponent
            current.string = parsed.remainder
        return current

    def _compute_operand(self, current: Iteration):
        """Create a new operand from the current iteration."""
        if current.operand and current.operator:
            return self._evaluate(current.operator, current.operand)
        if current.operand:
            return current.operand
        if current.operator:
            raise ParsingValueError("Operator without operand")
        raise ParsingValueError("Failed to parse string")

    def _operator_error(self, current: Operator, previous: Operator):
        """Check for known operator-related errors.
        
        This method checks for the following errors and returns the appropriate
        exception class if it finds one:
        - Multiple divisions on a single level (e.g., `'a / b / c'`), which
          results in a `RatioError`.
        - Multiplication after division on the same level (e.g., `'a / b * c'`),
          which results in a `ProductError`.

        Both of the examples shown above result in errors because they each
        introduce an ambiguous order of operations. Users can resolve the
        ambiguity by properly grouping terms in the expression. Continuing with
        the above examples, `'a / b / c'` should become `'(a / b) / c'` or `'a /
        (b / c)'`, and `'a / b * c'` should become `'(a / b) * c'` or `'a / (b *
        c)'`.
        """
        if self._operator_order == 'ignore':
            return
        if previous == 'divide':
            if current == 'divide':
                return RatioError
            if current == 'multiply':
                return ProductError

    def _evaluate(self, operator: Operator, operand: Operand):
        """Compute the effect of `operator` on `operand`."""
        if operator in {'multiply', 'identity'}:
            return operand
        if operator == 'divide':
            return operand ** -1
        if operator == 'sqrt':
            return operand ** 0.5
        raise ValueError(f"Unrecognized operator {operator!r}")

    def _update_terms(self, operand: Operand):
        """Store a new term or initiate further parsing."""
        # TODO: Consider extracting all coefficients, at least as separate
        # constant terms.
        if isinstance(operand, Term):
            return [operand]
        return self._resolve_operations(operand)


class OperandError(TypeError):
    pass


def standard(this, missing: str='1', joiner: str='*') -> str:
    """Convert a string to a standard format.
    
    Parameters
    ----------
    this : string or iterable
        The object to convert.

    missing : string, default='1'
        The value to use if `this` is null.

    joiner : string, default='*'
        The string token to use when joining parts of an iterable argument.

    See Also
    --------
    `~algebraic.Expression`: A class that represents one or more terms joined by
    algebraic operators and grouped by separator characters. Instances support
    multiplication and division with strings or other instances, and
    exponentiation by real numbers. Instantiation automatically calls this
    function.
    """
    if not this:
        return missing
    if isinstance(this, str):
        return this
    if not isinstance(this, typing.Iterable):
        return str(this)
    return joiner.join(f"({part})" for part in this)


Instance = typing.TypeVar('Instance', bound='Expression')


class Expression(collections.abc.Sequence, iterables.ReprStrMixin):
    """An object representing an algebraic expression."""

    terms: typing.List[Term]=None

    def __new__(
        cls: typing.Type[Instance],
        expression: typing.Union['Expression', str, iterables.whole],
        **kwargs
    ) -> Instance:
        """Create a new expression from user input.

        If `expression` is an instance of this class, this method will simply
        return it. Otherwise, it will create a new instance of this class from
        the given string or collection of parts after replacing operators and
        separators with their standard versions, if necessary.

        Parameters
        ----------
        expression : string or collection
            A single string or collection of any type to initialize the new
            instance. If this is a collection, all members must support
            conversion to a string.

        **kwargs
            Keywords to pass to `~algebra.Parser`.
        """
        if isinstance(expression, cls):
            return expression
        new = super().__new__(cls)
        string = standard(expression, joiner=' * ')
        terms = Parser(**kwargs).parse(string)
        new.terms = reduce(terms)
        """The algebraic terms in this expression."""
        return new

    def __iter__(self) -> typing.Iterator[Term]:
        return iter(self.terms)

    def __len__(self) -> int:
        return len(self.terms)

    @typing.overload
    def __getitem__(self, index: typing.SupportsIndex) -> Term: ...

    @typing.overload
    def __getitem__(self, index: slice) -> typing.Iterable[Term]: ...

    def __getitem__(self, index):
        """Access terms via standard indexing."""
        if isinstance(index, typing.SupportsIndex):
            idx = int(index)
            if idx > len(self):
                raise IndexError(index)
            if idx < 0:
                idx += len(self)
            return self.terms[idx]
        return self.terms[index]

    def __str__(self) -> str:
        """A simplified representation of this instance."""
        return self.format()

    def format(self, separator: str=' ', style: str=None):
        """Join algebraic terms into a string."""
        formatted = (term.format(style=style) for term in self)
        return separator.join(formatted)

    def __eq__(self, other) -> bool:
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
            other = self._new(other)
        if len(self) != len(other):
            return False
        key = attrgetter('base', 'exponent', 'coefficient')
        return sorted(self, key=key) == sorted(other, key=key)

    def __mul__(self, other):
        """Called for self * other.

        This method implements multiplication between two expressions by
        reducing the exponents of terms with a common base. If `other` is a
        string, it will first attempt to convert it to an `Expression`.
        """
        if not isinstance(other, Expression):
            other = self._new(other)
        if not other:
            return NotImplemented
        reduced = reduce(self, other)
        return self._new(reduced)

    def __rmul__(self, other: typing.Any):
        """Called for other * self."""
        return self._new(other).__mul__(self)

    def __truediv__(self, other):
        """Called for self / other.

        This method implements division between two expressions by raising all
        terms in `other` to -1, then reducing the exponents of terms with a
        common base. If `other` is a string, it will first attempt to convert it
        to an `Expression`.
        """
        if not isinstance(other, Expression):
            other = self._new(other)
        if not other:
            return NotImplemented
        return self._new(reduce(self, [term ** -1 for term in other]))

    def __rtruediv__(self, other: typing.Any):
        """Called for other / self."""
        return self._new(other).__truediv__(self)

    def __pow__(self, exp: numbers.Real):
        """Called for self ** exp.

        This method implements exponentiation of an expression by raising all
        terms to the given power, then reducing exponents of terms with a common
        base. It will first attempt to convert `exp` to a float.
        """
        exp = float(exp)
        if not exp:
            return NotImplemented
        terms = [pow(term, exp) for term in self]
        return self._new(reduce(terms))

    @classmethod
    def _new(cls, arg: typing.Union[str, iterables.whole]):
        """Internal helper method for creating a new instance.

        This method is separated out for the sake of modularity, in case of a
        need to add any other functionality when creating a new instance from
        the current one (perhaps in a subclass).
        """
        return cls(arg)

    def apply(self: Instance, update: typing.Callable) -> Instance:
        """Create a new expression by applying the given callable object.
        
        Parameters
        ----------
        update : callable
            The callable object that this method should use to update the base
            of each term in this expression.

        Returns
        -------
        `~algebraic.Expression`
        """
        bases = [update(term.base) for term in self]
        exponents = [term.exponent for term in self]
        result = bases[0] ** exponents[0]
        for base, exponent in zip(bases[1:], exponents[1:]):
            result *= base ** exponent
        return result


def reduce(*groups: typing.Iterable[Term]):
    """Algebraically reduce terms with equal bases.

    Parameters
    ----------
    *groups : tuple of iterables
        One or more iterables of `~algebra.Term` instances. If there are
        multiple groups, this method will combine all terms it finds in the full
        collection of groups.
    """
    terms = [term for group in groups for term in group]
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
    variables = [
        Term(base=k, exponent=v['exponent'])
        for k, v in reduced.items()
        if k != '1' and v['exponent'] != 0
    ]
    c = functools.reduce(lambda x, y: x*y, fracs)
    constant = [Term(coefficient=c)]
    if not variables:
        return constant
    if c == 1:
        return variables
    return variables + constant


@typing.runtime_checkable
class Orderable(typing.Protocol):
    """Protocol for objects that support ordering.
    
    Instance checks against this ABC will return `True` iff the instance
    implements the following methods: `__lt__`, `__gt__`, `__le__`, `__ge__`,
    `__eq__`, and `__ne__`. It exists to support type-checking orderable objects
    outside the `~algebraic.Quantity` framework (e.g., pure numbers).
    """

    __slots__ = ()

    @abc.abstractmethod
    def __lt__(self, other):
        pass

    @abc.abstractmethod
    def __eq__(self, other):
        pass

    @abc.abstractmethod
    def __le__(self, other):
        pass

    @abc.abstractmethod
    def __gt__(self, other):
        pass

    @abc.abstractmethod
    def __ge__(self, other):
        pass

    @abc.abstractmethod
    def __ne__(self, other):
        pass


class Ordered(abc.ABC):
    """Abstract base class for all objects that support relative ordering.

    Concrete implementations of this class must define the six binary comparison
    operators (a.k.a "rich comparison" operators): `__lt__`, `__gt__`, `__le__`,
    `__ge__`, `__eq__`, and `__ne__`.

    The following default implementations are available by calling their
    equivalents on `super()`:

    - `__ne__`: defined as not equal.
    - `__le__`: defined as less than or equal.
    - `__gt__`: defined as not less than and not equal.
    - `__ge__`: defined as not less than.
    """

    __slots__ = ()

    __hash__ = None

    @abc.abstractmethod
    def __lt__(self, other) -> bool:
        """True if self < other."""
        pass

    @abc.abstractmethod
    def __eq__(self, other) -> bool:
        """True if self == other."""
        pass

    @abc.abstractmethod
    def __le__(self, other) -> bool:
        """True if self <= other."""
        return self.__lt__(other) or self.__eq__(other)

    @abc.abstractmethod
    def __gt__(self, other) -> bool:
        """True if self > other."""
        return not self.__le__(other)

    @abc.abstractmethod
    def __ge__(self, other) -> bool:
        """True if self >= other."""
        return not self.__lt__(other)

    @abc.abstractmethod
    def __ne__(self, other) -> bool:
        """True if self != other."""
        return not self.__eq__(other)


Self = typing.TypeVar('Self', bound='Additive')


class Additive(abc.ABC):
    """Abstract base class for additive objects."""

    __slots__ = ()

    @abc.abstractmethod
    def __add__(self: Self, other) -> Self:
        pass

    @abc.abstractmethod
    def __radd__(self: Self, other) -> Self:
        pass

    @abc.abstractmethod
    def __sub__(self: Self, other) -> Self:
        pass

    @abc.abstractmethod
    def __rsub__(self: Self, other) -> Self:
        pass


Self = typing.TypeVar('Self', bound='Multiplicative')


class Multiplicative(abc.ABC):
    """Abstract base class for multiplicative objects."""

    __slots__ = ()

    @abc.abstractmethod
    def __mul__(self: Self, other) -> Self:
        pass

    @abc.abstractmethod
    def __rmul__(self: Self, other) -> Self:
        pass

    @abc.abstractmethod
    def __truediv__(self: Self, other) -> Self:
        pass

    @abc.abstractmethod
    def __rtruediv__(self: Self, other) -> Self:
        pass

    @abc.abstractmethod
    def __pow__(self: Self, other) -> Self:
        pass

    @abc.abstractmethod
    def __rpow__(self: Self, other) -> Self:
        pass


Multiplicative.register(Expression)


class Quantity(Ordered, Additive, Multiplicative):
    """Base class for algebraic quantities.

    Concrete subclasses of this class must implement the six comparison
    operators,
        - `__lt__` (less than; called for `self < other`)
        - `__gt__` (greater than; called for `self > other`)
        - `__le__` (less than or equal to; called for `self <= other`)
        - `__ge__` (greater than or equal to; called for `self >= other`)
        - `__eq__` (equal to; called for `self == other`)
        - `__ne__` (not equal to; called for `self != other`)
    
    the following unary arithmetic operators,
        - `__abs__` (absolute value; called for `abs(self)`)
        - `__neg__` (negative value; called for `-self`)
        - `__pos__` (positive value; called for `+self`)

    and the following binary arithmetic operators,
        - `__add__` (addition; called for `self + other`)
        - `__radd__` (reflected addition; called for `other + self`)
        - `__sub__` (subtraction; called for `self - other`)
        - `__rsub__` (reflected subtraction; called for `other - self`)
        - `__mul__` (multiplication; called for `self * other`)
        - `__rmul__` (reflected multiplication; called for `other * self`)
        - `__truediv__` (division; called for `self / other`)
        - `__rtruediv__` (reflected division; called for `other / self`)
        - `__pow__` (exponentiation; called for `self ** other`)
        - `__rpow__` (reflected exponentiation; called for `other ** self`)

    Any required method may return `NotImplemented`.
    """

    __slots__ = ()

    @abc.abstractmethod
    def __abs__(self):
        """Implements abs(self)."""
        pass

    @abc.abstractmethod
    def __neg__(self):
        """Called for -self."""
        pass

    @abc.abstractmethod
    def __pos__(self):
        """Called for +self."""
        pass


