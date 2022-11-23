import abc
import collections
import collections.abc
import fractions
import functools
import itertools
import numbers
import re
from operator import attrgetter
import typing

from goats.core import algebraic
from goats.core import iterables


# NOTE: This is an experimental attempt to refactor redundant code in `Term` and
# `OperandFactory`. It is currently not in use.
class Patterns(collections.UserDict):
    """Compiled regular expressions for symbolic operands."""

    rational = r""" # Modeled after `fractions._RATIONAL_FORMAT`
        [-+]?                 # an optional sign, ...
        (?=\d|\.\d)           # ... only if followed by <digit> or .<digit>
        \d*                   # and a possibly empty numerator
        (?:                   # followed by ...
            (?:/\d+?)         # ... an optional denominator
        |                     # OR
            (?:\.\d*)?        # ... an optional fractional part,
            (?:[eE][-+]?\d+)? #     and an optional exponent
        )
    """
    base = r"""
        [a-zA-Z#_]+ # one or more accepted non-digit character(s)
        \d*         # followed by optional digits
    """

    def __init__(
        self,
        multiplication: str='*',
        division: str='/',
        opening: str='(',
        closing: str=')',
        raising: str='^',
    ) -> None:
        multiply = fr'\{multiplication}'
        divide = fr'\{division}'
        exponent = fr'\{raising}{self.rational}'
        patterns = {
            'multiply': (
                fr'(?<!{divide})(\s*{multiply}\s*)(?!{divide})'
            ),
            'divide': (
                fr'(?<!{multiply})(\s*{divide}\s*)(?!{multiply})'
            ),
            'sqrt': (r'\s*sqrt\s*'),
            'constant': (
                fr'(?P<coefficient>{self.rational})'
                fr'(?P<exponent>{exponent})?'
            ),
            'variable': (
                fr'(?P<coefficient>{self.rational})?'
                fr'(?P<base>{self.base})'
                fr'(?P<exponent>{exponent})?'
            ),
            'expression': (
                fr'(?P<coefficient>{self.rational})?'
                fr'(?P<base>\{opening}.+?\{closing})'
                fr'(?P<exponent>{exponent})?'
            ),
            'exponent': exponent,
            'opening': fr'\{opening}',
            'closing': fr'\{closing}',
            'raising': fr'\{raising}',
        }
        self._compiled = {}
        super().__init__(patterns)

    def __getitem__(self, __k: str):
        if __k in self._compiled:
            return self._compiled[__k]
        if __k in self.data:
            compiled = re.compile(self.data[__k], re.VERBOSE)
            self._compiled[__k] = compiled
            return compiled
        raise KeyError(f"No pattern for {__k}") from None

    def __setitem__(self, __k: str, __v: str) -> None:
        self._compiled.pop(__k, None)
        return super().__setitem__(__k, __v)

    def __delitem__(self, __k: str) -> None:
        self._compiled.pop(__k, None)
        return super().__delitem__(__k)


class Part(abc.ABC, iterables.ReprStrMixin):
    """Base class for parts of a symbolic expression."""

    __slots__ = ()


class Operator(Part):
    """An operator in a symbolic expression."""

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
    """An operand in a symbolic expression.

    Algebraic operands mainly exist to support the `~algebra.Expression` class.
    They may be simple or general, as described below.

    A simple symbolic operand has the form [c]b[^e] or c[^e], where `c` is a
    numerical coefficient, `b` is a string base, and `e` is a numerical
    exponent. Braces ('[]') denote an optional component that defaults to a
    value of 1. The `~algebra.Term` class formally represents simple symbolic
    operands; the form [c]b[^e] corresponds to a variable term and the form
    c[^e] corresponds to a constant term.

    Examples include:
    
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

    A general symbolic operand consists of simpler operands (though not
    necessarily formally simple operands) combined with symbolic operators and
    separators. All formally simple operands are general operands. The following
    are examples of (non-simple) general symbolic operands:

    * `'a * b^2'` <=> `'(a * b^2)'` <=> `'(a * b^2)^1'`
    * `'(a * b^2)^3'`
    * `'(a * b^2)^3/2'`
    * `'((a / b^2)^3 * c)^2'`
    * `'(a / b^2)^3 * c^2'`
    * `'a / (2 * 4b)'`
    * `'(2a * b)^3 / (4 * c)'`

    There are many more ways to construct a general operand than a simple
    operand. This is by design, to support building instances of
    `~symbolic.Expression` with `~symbolic.Parser`.
    """

    def __init__(
        self,
        coefficient: numbers.Real=None,
        base: str=None,
        exponent: numbers.Real=None,
    ) -> None:
        self.coefficient = fractions.Fraction(coefficient or 1)
        """The numerical coefficient."""
        self.base = base or '1'
        """The base term or complex."""
        self.exponent = fractions.Fraction(exponent or 1)
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

    def __mul__(self, other):
        """Create a new operand, multiplied by `other`."""
        coefficient = self.coefficient * other
        return type(self)(coefficient, self.base, self.exponent)

    __rmul__ = __mul__

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
    """A symbolic operand with an irreducible base.
    
    In the single-argument form, the user provides only the base quantity by
    positional-only argument. In the triple-argument form, the user provides the
    coefficient, base, and exponent by positional or keyword argument(s), with
    the caveat that if one argument is positional, they must all be positional.
    This restriction prevents ambiguity among the possible double-argument
    combinations.
    """

    # NOTE: Currently redundant with `OperandFactory`.

    rational = r""" # Modeled after `fractions._RATIONAL_FORMAT`
        [-+]?                 # an optional sign, ...
        (?=\d|\.\d)           # ... only if followed by <digit> or .<digit>
        \d*                   # and a possibly empty numerator
        (?:                   # followed by ...
            (?:/\d+?)         # ... an optional denominator
        |                     # OR
            (?:\.\d*)?        # ... an optional fractional part,
            (?:[eE][-+]?\d+)? #     and an optional exponent
        )
    """
    base = r"""
        [a-zA-Z#_]+ # one or more accepted non-digit character(s)
        \d*         # followed by optional digits
    """
    _base_re = re.compile(fr'({rational}|{base})', re.VERBOSE)

    @typing.overload
    def __init__(self, base: str, /) -> None:
        """Create a symbolic term from a base quantity.
        
        Parameters
        ----------
        base : string
            The base quantity of this symbolic term.
        """

    @typing.overload
    def __init__(
        self,
        coefficient: numbers.Real=1,
        base: str='1',
        exponent: numbers.Real=1,
    ) -> None:
        """Create a symbolic term from coefficient, base, and exponent.
        
        Parameters
        ----------
        coefficient : real number, default=1
            The numerical coefficient to associate with the base quantity.

        base : string, default='1'
            The base quantity of this symbolic term.

        exponent : real number, default=1
            The exponent of the base quantity.
        """

    def __init__(self, *args, **kwargs) -> None:
        if not kwargs:
            # positional argument(s) only
            if not args:
                # use all default values
                return super().__init__()
            if len(args) == 1:
                # single-argument form (positional base only)
                base = args[0]
                self._validate_base(base)
                return super().__init__(base=base)
            if len(args) == 3:
                # triple-argument form
                base = args[1]
                self._validate_base(base)
                return super().__init__(*args)
        if not args:
            # keyword arguments only
            if 'base' in kwargs:
                # NOTE: We don't use `base = kwargs.get('base')` because we
                # don't want to mistakenly treat `None` as the base when
                # comparing to the base RE pattern. We'll let `Operand`
                # determine the default value.
                base = kwargs['base']
                self._validate_base(base)
            return super().__init__(**kwargs)
        raise TypeError(
            f"Can't instantiate {self.__class__} with {args} and {kwargs}."
        ) from None

    def _validate_base(self, base):
        """Raise an exception if `base` does not match the definition."""
        if not self._base_re.fullmatch(str(base)):
            raise ValueError(
                f"Can't create a symbolic term with base {base!r}"
            ) from None

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

    def __hash__(self) -> int:
        # NOTE: If we decide to implement `Operand.__hash__`, we will still need
        # to explicitly define `Term.__hash__`. See
        # https://docs.python.org/3/reference/datamodel.html#object.__hash__ for
        # an explanation.
        return hash(self.attrs)


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
    """Protocol for symbolic factories."""

    @abc.abstractmethod
    def parse(self) -> PartMatch:
        pass


class OperatorFactory(PartFactory):
    """A factory that produces symbolic operators."""

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
        """Compiled regular expressions for symbolic operators."""

    def parse(self, string: str):
        """Extract an operator at the start of `string`, possible."""
        for key in self.patterns:
            if match := self.patterns[key].match(string):
                return PartMatch(
                    result=Operator(key),
                    context=match,
                )


class OperandFactory(PartFactory):
    """A factory that produces symbolic operands."""

    rational = r""" # Modeled after `fractions._RATIONAL_FORMAT`
        [-+]?                 # an optional sign, ...
        (?=\d|\.\d)           # ... only if followed by <digit> or .<digit>
        \d*                   # and a possibly empty numerator
        (?:                   # followed by ...
            (?:/\d+?)         # ... an optional denominator
        |                     # OR
            (?:\.\d*)?        # ... an optional fractional part,
            (?:[eE][-+]?\d+)? #     and an optional exponent
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
        """Compiled regular expressions for symbolic operands."""

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
        This method will create the most general symbolic operand possible from
        the initial string. It will parse a simple symbolic operand into a
        coefficient, variable, and exponent but it will not attempt to fully
        parse a general symbolic operand into simpler operands (i.e. symbolic
        terms). In other words, it will do as little work as possible to extract
        a coefficient and exponent, and the expression on which they operate. If
        all attempts to determine appropriate attributes fail, the value of
        `strict` controls its return behavior.
        
        See note at `parse` for differences between this method and that.

        The following examples use the general symbolic operands described in
        `~algebra.Operand` to illustrate the minimal parsing described above:

        - `'a * b^2'` <=> `'(a * b^2)'` <=> `'(a * b^2)^1'` -> `1, 'a * b^2', 1`
        - `'2a * b^2'` -> `1, '2a * b^2', 1`
        - `'2(a * b^2)'` -> `2, 'a * b^2', 1`
        - `'(a * b^2)^3'` -> `1, 'a * b^2', 3`
        - `'2(a * b^2)^3'` -> `2, 'a * b^2', 3`
        - `'(a * b^2)^3/2'` -> `1, 'a * b^2', '3/2'`
        - `'((a / b^2)^3 * c)^2'` -> `1, '(a / b^2)^3 * c', 2`
        - `'(a / b^2)^3 * c^2'` -> `1, '(a / b^2)^3 * c^2', 1`
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
        The primary difference between `~create` and `~parse` is as follows:

        - `~create` resolves input into (coefficient, base, exponent), then
          creates an appropriate operand from the base string, then applies the
          input coefficient and exponent, and finally returns the operand.

        - `~parse` attempts to match an operand at the start of a string, then
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
        `~symbolic.PartMatch` or `None`
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
    """Base class for exceptions encountered during symbolic parsing."""

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
    """A tool for parsing symbolic expressions."""

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
        """Separate a symbolic group into operators and operands."""
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
    """Convert `this` to a standard format.
    
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
    `~symbolic.Expression`: A class that represents one or more terms joined by
    symbolic operators and grouped by separator characters. Instances support
    multiplication and division with strings or other instances, and
    exponentiation by real numbers. Instantiation automatically calls this
    function.
    """
    if not this:
        return missing
    if isinstance(this, str):
        return this
    try:
        iter(this)
    except TypeError:
        return str(this)
    else:
        return joiner.join(f"({part})" for part in this)


Instance = typing.TypeVar('Instance', bound='Expression')


class Expression(collections.abc.Sequence, iterables.ReprStrMixin):
    """An object representing a symbolic expression.

    If this class is instantiated with an existing instance, the result will be
    the same instance. Otherwise, it will create a new instance of this class
    from the given string or iterable after replacing operators and separators
    with their standard versions, if necessary.
    """

    def __new__(cls, *args, **kwargs):
        if len(args) == 1:
            arg = args[0]
            if isinstance(arg, cls):
                if not kwargs:
                    return arg
                raise TypeError(
                    "Can't change parsing options on an existing expression"
                ) from None
        return super().__new__(cls)

    @typing.overload
    def __init__(self, expression: str, /, **kwargs) -> None:
        """Create an expression from a single string.
        
        Parameters
        ----------
        expression : string
            A single string to convert into an expression.

        **kwargs
            Keywords to pass to `~algebra.Parser`.

        Examples
        --------
        Create a symbolic expression from a string that represents the result of
        multiplying `a^3/2` by `b`, dividing by `c^1/2`, and squaring the ratio:

        >>> symbolic.Expression('(a^3/2 * b / c^1/2)^2')
        core.symbolic.Expression(a^3 b^2 c^-1)
        """

    @typing.overload
    def __init__(self, expression: typing.Iterable, /, **kwargs) -> None:
        """Create an expression from an iterable of any type.
        
        Parameters
        ----------
        expression : iterable
            An iterable of any type to initialize the new instance. All members
            must support conversion to a string.

        **kwargs
            Keywords to pass to `~algebra.Parser`.

        Examples
        --------
        Create a symbolic expression from a list of the individual string terms
        in the result of multiplying `a^3/2` by `b`, dividing by `c^1/2`, and
        squaring the ratio:

        >>> symbolic.Expression(['a^3', 'b', 'c^-1'])
        core.symbolic.Expression(a^3 b c^-1)
        """

    @typing.overload
    def __init__(self: Instance, expression: Instance, /) -> None:
        """Create an expression from an expression.

        This mode exists to support algorithms that don't know the type of
        argument until runtime. If the type is known, it is simpler to use the
        existing instance.

        Parameters
        ----------
        expression : `~symbolic.Expression`
            An existing instance of this class.

        Examples
        --------
        Create an instance from a string:

        >>> this = symbolic.Expression('a * b / c')

        Pass the first instance to this class:

        >>> that = symbolic.Expression(this)

        Both `this` and `that` represent the same expression...
        
        >>> this
        core.symbolic.Expression(a b c^-1)
        >>> that
        core.symbolic.Expression(a b c^-1)

        ...because they are the same object.

        >>> that is this
        True
        """

    def __init__(self, expression, **kwargs) -> None:
        string = standard(expression, joiner=' * ')
        terms = Parser(**kwargs).parse(string)
        self.terms = reduce(terms)
        """The symbolic terms in this expression."""

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
        """Join symbolic terms into a string."""
        formatted = (term.format(style=style) for term in self)
        return separator.join(formatted)

    def difference(self, other, symmetric: bool=False, split: bool=False):
        """Compute the difference between two expressions.
        
        Parameters
        ----------
        other
            The object with respect to which to compute the difference. If
            `other` is not a `~symbolic.Expression`, this method will convert it
            to one before proceeding.

        symmetric : bool, default=False
            If true, compute the symmetric difference between this expression
            and `other`.

        split : bool, default=False
            If true, return the one-sided differences in a ``list``. The first
            element contains the terms in this expression that are not in
            `other`, and the second element contains the terms in `other` that
            are not in this expression.

        Notes
        -----
        The `split` keyword argument takes precedence over the `symmetric`
        keyword argument because the result of the former contains more
        imformation than the result of the latter. See Examples for a suggestion
        on converting a split result into a symmetric result.

        Examples
        --------
        Consider the following two expressions:
        
        >>> e0 = symbolic.Expression('a * b')
        >>> e1 = symbolic.Expression('a * c')

        Their formal (one-sided) difference is

        >>> e0.difference(e1)
        {core.symbolic.Term(b)}

        Their formal symmetric difference is

        >>> e0.difference(e1, symmetric=True)
        {core.symbolic.Term(b), core.symbolic.Term(c)}

        Passing ``split=True`` produces a ``list`` of ``set``s

        >>> e0.difference(e1, split=True)
        [{core.symbolic.Term(b)}, {core.symbolic.Term(c)}]

        To convert a split result into a symmetric result, simply compute the
        union of the former:

        >>> symmetric = e0.difference(e1, symmetric=True)
        >>> split = e0.difference(e1, split=True)
        >>> set.union(*split) == symmetric
        True
        """
        if not isinstance(other, Expression):
            other = type(self)(other)
        s0 = set(self.terms)
        s1 = set(other.terms)
        if split:
            return [s0 - s1, s1 - s0]
        if symmetric:
            return s0 ^ s1
        return s0 - s1

    def __hash__(self):
        """Compute hash(self)."""
        return hash(tuple(self.terms))

    def __eq__(self, other) -> bool:
        """True if two expressions have the same symbolic terms.

        This method defines two expressions as equal if they have equivalent
        lists of symbolic terms (a.k.a simple parts), regardless of order, after
        parsing. Two expressions with different numbers of terms are always
        false. If the expressions have the same number of terms, this method
        will sort the triples (first by base, then by exponent, and finally by
        coefficient) and compare the sorted lists. Two expressions are equal if
        and only if their sorted lists of terms are equal.

        If `other` is not an instance of this class, this method will first
        attempt to convert it.
        """
        if not isinstance(other, Expression):
            other = type(self)(other)
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
            other = type(self)(other)
        if not other:
            return NotImplemented
        reduced = reduce(self, other)
        return type(self)(reduced)

    def __rmul__(self, other: typing.Any):
        """Called for other * self."""
        return type(self)(other).__mul__(self)

    def __truediv__(self, other):
        """Called for self / other.

        This method implements division between two expressions by raising all
        terms in `other` to -1, then reducing the exponents of terms with a
        common base. If `other` is a string, it will first attempt to convert it
        to an `Expression`.
        """
        if not isinstance(other, Expression):
            other = type(self)(other)
        if not other:
            return NotImplemented
        return type(self)(reduce(self, [term ** -1 for term in other]))

    def __rtruediv__(self, other: typing.Any):
        """Called for other / self."""
        return type(self)(other).__truediv__(self)

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
        return type(self)(reduce(terms))

    def apply(self: Instance, update: typing.Callable) -> Instance:
        """Create a new expression by applying the given callable object.
        
        Parameters
        ----------
        update : callable
            The callable object that this method should use to update the base
            of each term in this expression.

        Returns
        -------
        `~symbolic.Expression`
        """
        bases = [update(term.base) for term in self]
        exponents = [term.exponent for term in self]
        result = bases[0] ** exponents[0]
        for base, exponent in zip(bases[1:], exponents[1:]):
            result *= base ** exponent
        return result


algebraic.Multiplicative.register(Expression)


def composition(this):
    """True if `this` appears to be a symbolic composition of terms.
    
    Parameters
    ----------
    this
        The object to check.

    Notes
    -----
    This is more stringent than simply checking whether `this` can instantiate a
    `~symbolic.Expression` because any string would satisfy that condition.
    """
    return (
        isinstance(this, Expression)
        or isinstance(this, str) and ('/' in this or '*' in this)
    )


Expressable = typing.TypeVar(
    'Expressable',
    str,
    typing.Iterable,
    Expression,
)
Expressable = typing.Union[str, typing.Iterable, Expression]


def reduce(*groups: typing.Iterable[Term]):
    """Algebraically reduce terms with equal bases.

    Parameters
    ----------
    *groups : tuple of iterables
        One or more iterables of `~algebra.Term` instances. If there are
        multiple groups, this method will combine all terms it finds in the full
        collection of groups.

    Notes
    -----
    This function will sort terms in order of ascending exponent, and
    alphabetically for equal exponents.
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
    tmp = [
        Term(base=k, exponent=v['exponent'])
        for k, v in reduced.items()
        if k != '1' and v['exponent'] != 0
    ]
    # Sort: high to low in exponent, followed by alphabetic in base.
    variables = sorted(
        sorted(tmp, key=attrgetter('base')),
        key=attrgetter('exponent'),
        reverse=True,
    )
    c = functools.reduce(lambda x, y: x*y, fracs)
    constant = [Term(coefficient=c)]
    if not variables:
        return constant
    if c == 1:
        return variables
    return constant + variables


def equality(a, b) -> Expression:
    """Symbolically compute a == b."""
    x, y = (Expression(i) for i in (a, b))
    return x == y


def product(a, b) -> Expression:
    """Symbolically compute a * b."""
    x, y = (Expression(i) for i in (a, b))
    return x * y


def ratio(a, b) -> Expression:
    """Symbolically compute a / b."""
    x, y = (Expression(i) for i in (a, b))
    return x / y


def power(a, n) -> Expression:
    """Symbolically compute a ** n."""
    return Expression(a) ** n

