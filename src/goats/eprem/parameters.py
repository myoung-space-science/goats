"""EPREM runtime parameter arguments, definitions, and metadata.

This module includes the following objects::
* `~parameters.BaseTypesH` provides access to all EPREM runtime constants (see
  src/baseTypes.h), given a path to the directory containing EPREM source code.
* `~parameters.ConfigurationC` provides definitions and default arguments of all
  parameters relevant to a simulation run (see src/configuration.c), given a
  path to the directory containing EPREM source code.
* `~parameters.ConfigFile` represents the contents of a named configuration file
  that contains user-provided arguments for a subset of simulation parameters,
  given a path to the file.
* `~parameters.Runtime` provides a unified interface to all parameter arguments
  used in a particular simulation run, given appropriate paths.
* `~parameters.Arguments` is an instance of `aliased.Mapping` that
  supports aliased access to simulation and post-processing parameters, given an
  instance of `~parameters.Runtime`.

Notes
=====

Metadata
--------
* `~parameters.Runtime` represents the full set of parameters relevant to a
  particular simulation run in two regards: 1) the EPREM distribution (e.g.,
  'epicMas' or 'epicEnlil'), and 2) the combination of user-provided and default
  arguments. There is a long-term goal to develop a single modular EPREM
  distribution, completion of which would remove the first item.
* `~parameters._CONFIGURATION_C` contains metadata for a subset of the
  parameters that `~parameters.Runtime` represents; `~parameters.Runtime` is
  always the canonical collection.
* `~parameters._BASETYPES_H` is always a proper subset of `~parameters.Runtime`.
* `~parameters.Runtime` + `~parameters._LOCAL` represents the full set of
  parameters available to post-processing code in this package.

Terminology
-----------
The word "parameter" can have multiple meanings in the contexts to which this
module is relevant. The python built-in module `inspect` uses "parameters" to
mean the names representing positional and keyword arguments. This is consistent
with formal definitions of a function parameter (cf.
https://developer.mozilla.org/en-US/docs/Glossary/Parameter). It is also common
to refer to config-file options as parameters, in a way that is similar but not
identical to the formal software defintion above. This module attempts to
respect the formal distinction between parameters and arguments.
"""

import argparse
import functools
import numbers
import pathlib
import re
import typing
import json

from goats.common import algebra
from goats.common import aliased
from goats.common import iotools
from goats.common import iterables
from goats.common import numerical
from goats.common import quantities


class BaseTypesH(iterables.MappingBase):
    """A representation of EPREM `baseTypes.h`."""

    def __init__(self, src: typing.Union[str, pathlib.Path]=None) -> None:
        self.path = iotools.ReadOnlyPath(src or '.') / 'baseTypes.h'
        self._definitions = None
        super().__init__(tuple(self.definitions))
        self._cache = {}

    def __getitem__(self, name: str):
        """Access constants by name."""
        if name in self._cache:
            return self._cache[name]
        if name in self:
            value = self._compute(name)
            self._cache[name] = value
            return value
        raise KeyError(f"No {name!r} in baseTypes.h")

    @property
    def definitions(self):
        """The definition of each constant in the simulation source code."""
        if self._definitions is None:
            with self.path.open('r') as fp:
                lines = fp.readlines()
            definitions = {}
            for line in lines:
                if line.startswith('#define'):
                    parts = line.strip('#define').rstrip('\n').split(maxsplit=1)
                    if len(parts) == 2:
                        key, value = parts
                        metadata = _BASETYPES_H.get(key, {})
                        if any(c in value for c in {'*', '/', 'sqrt'}):
                            definitions[key] = algebra.Expression(value)
                        else:
                            cast = metadata.get('type', str)
                            definitions[key] = cast(value)
            self._definitions = definitions
        return self._definitions

    def _compute(self, key: str) -> numbers.Real:
        """Compute the value of a defined constant."""
        target = self.definitions[key]
        if isinstance(target, numbers.Real):
            return target
        if isinstance(target, algebra.Expression):
            value = 1.0
            for term in target:
                if term.base in self.definitions:
                    value *= float(term(self._compute(term.base)))
                elif term.base == '1':
                    value *= float(term)
            return value
        raise TypeError(target)

    def dump(self, *args, **kwargs):
        """Serial values and metadata in JSON format.
        
        See `json.dump` for descriptions of accepted arguments.
        """
        formulas = {
            key: definition.format(separator=' * ')
            if isinstance(definition, algebra.Expression)
            else None
            for key, definition in self.definitions.items()
        }
        obj = {
            key: {
                'value': value,
                'type': str(type(value).__qualname__),
                'formula': formulas[key],
            } for key, value in self.items()
        }
        json.dump(obj, *args, **kwargs)


class FunctionCall:
    """Pattern parser for function calls."""

    # TODO: 
    # - __init__ should take arguments that allow the user to define possible
    #   function names (e.g., 'readInt', 'readDouble') and target-variable
    #   prefix (e.g., 'config').
    # - It may be better to just grab everything in parentheses as `args` and
    #   let `parse` sort them out.

    @property
    def pattern(self):
        return re.compile(r"""
            # the start of the string, followed by optional whitespace
            \A\s*
            # the name of the C struct
            config\.
            # the name of the attribute
            (?P<name>\w+)
            # equals sign surrounded by optional whitespace
            \s*=\s*
            # kludge for `(char*)readString(...)`
            (?:\(char\*\))?
            # the name of the file-reading method
            (?P<mode>read(?:Int|Double|DoubleArray|String))
            # beginning of function call
            \(
                # parameter name in config file
                (?P<alias>\"\w*\")
                # optional whitespace followed by a comma
                \s*\,
                # the remaining arguments
                (?P<args>.*?)
            # end of function call
            \)
            # C statement terminator
            \;
            # optional whitespace, followed by the end of the string
            \s*\Z
        """, re.VERBOSE)

    def match(self, line: str):
        """Identify lines that read config-file input."""
        if match := self.pattern.match(line.strip()):
            return match.groupdict()

    def parse(self, parsable: typing.Dict[str, str]):
        """Parse a line that reads config-file input."""
        parsed = {}
        name = parsable['name']
        if alias := parsable['alias']:
            a = alias.strip('"')
            if a != name:
                parsed['alias'] = a
        parsed['mode'] = parsable['mode']
        args = parsable['args'].split(',')
        parsed['args'] = [arg.strip() for arg in args]
        return name, parsed


class VariableDefinition:
    """Pattern parser for variable definitions."""

    # TODO:
    # - __init__ should take arguments that allow the user to define the
    #   variable type (e.g., 'Scalar_t').
    # - Can the pattern identify things like `size` and `value` with optional
    #   user input, or does `parse` need to handle that?

    @property
    def pattern(self):
        return re.compile(r"""
            # the start of the string, followed by optional whitespace
            \A\s*
            # type declaration
            Scalar\_t
            # optional whitespace
            \s*
            # variable name
            (?P<name>\w+)
            # array size
            \[(?P<size>\d+)\]
            # equals sign surrounded by optional whitespace
            \s*=\s*
            # array value(s)
            \{(?P<value>\d*(?:\.\d+)?)\}
            # C statement terminator
            \;
            # optional whitespace, followed by the end of the string
            \s*\Z
        """, re.VERBOSE)

    def match(self, line: str):
        """Identify lines that define a variable."""
        if match := self.pattern.match(line.strip()):
            return match.groupdict()

    def parse(self, parsable: typing.Dict[str, str]):
        """Parse information about a defined variable."""
        parsed = {
            'size': int(parsable['size']),
            'value': parsable['value'],
        }
        return parsable['name'], parsed


_RT = typing.TypeVar('_RT')
def soft_convert(
    string: str,
    convert: typing.Union[_RT, typing.Callable[[str], _RT]],
    acceptable: typing.Union[Exception, typing.Tuple[Exception]]=Exception,
) -> typing.Union[str, _RT]:
    """Convert a string to a different type, if possible.

    This function will use the given callable to attempt to convert the given
    string to a different type. If the conversion fails, this function will
    return the unconverted string.

    Parameters
    ----------
    string
        The string to convert.

    convert : callable
        A callable object that accepts a string argument and returns the
        converted value.

    acceptable : exception or iterable of exceptions
        One or more types of valid exception classes that this function should
        ignore when attempting to convert `string`. The default behavior is to
        accept any subclass of `Exception`. Passing arguments to this keyword
        therefore allows the caller to limit the scope of ignorable errors.

    Returns
    -------
    converted type or string
        The converted value of the given string, if conversion succeeded, or the
        unconverted string.
    """
    try:
        value = convert(string)
    except acceptable or Exception:
        return string
    return value


class ConfigurationC(iterables.MappingBase):
    """A representation of EPREM `configuration.c`."""

    def __init__(self, src: typing.Union[str, pathlib.Path]=None) -> None:
        path = pathlib.Path(src or '.') / 'configuration.c'
        self.file = iotools.TextFile(path)
        self._assignment = FunctionCall()
        self._array_default = VariableDefinition()
        self._definitions = None
        super().__init__(tuple(self.definitions))
        self._path = path

    @property
    def path(self):
        """The path to the relevant EPREM distribution."""
        return iotools.ReadOnlyPath(self._path)

    def __getitem__(self, key: str):
        """Request a reference object by parameter name."""
        if key in self:
            return self.definitions[key]
        raise KeyError(f"No reference information for {key!r}")

    @property
    def definitions(self):
        """Unconverted reference values from the source code."""
        if self._definitions is None:
            assignments = self.file.extract(
                self._assignment.match,
                self._assignment.parse,
            )
            self._definitions = {
                k: self._prepare(v)
                for k, v in assignments.items()
            }
        return self._definitions

    def _prepare(
        self,
        metadata: typing.Dict[str, typing.Union[str, type]],
    ) -> typing.Dict[str, typing.Union[str, type]]:
        """Create a reference object from parameter metadata."""
        mode = metadata['mode']
        if mode == 'readInt':
            keys = ('default', 'minimum', 'maximum')
            args = metadata['args']
            return {'type': int, **dict(zip(keys, args))}
        if mode == 'readDouble':
            keys = ('default', 'minimum', 'maximum')
            args = metadata['args']
            return {'type': float, **dict(zip(keys, args))}
        if mode == 'readString':
            arg = metadata['args'][0]
            return {'type': str, 'default': arg.strip('"')}
        if mode == 'readDoubleArray':
            defaults = self.file.extract(
                self._array_default.match,
                self._array_default.parse,
            )
            args = [
                defaults[arg]['value']
                if arg in defaults else arg
                for arg in metadata['args']
            ]
            keys = ('size', 'default')
            return {'type': list, **dict(zip(keys, args))}
        raise ValueError(f"Unknown configuration mode {mode!r}")

    def print(self, tabsize: int=4, stream: typing.TextIO=None):
        """Print formatted reference information."""
        indent = ' ' * tabsize
        print("{", file=stream)
        for key, defined in self.definitions.items():
            args = ''.join(
                f"{indent}{indent}{k!r}: {v!r},\n"
                if not isinstance(v, type) else
                f"{indent}{indent}{k!r}: {v.__qualname__},\n"
                for k, v in defined.items()
            )
            print(
                f"{indent}{key!r}: {'{'}\n"
                f"{args}",
                f"{indent}{'},'}",
                file=stream,
            )
        print("}", file=stream)

    def dump(self, *args, **kwargs):
        """Serial values and metadata in JSON format.
        
        See `json.dump` for descriptions of accepted arguments.
        """
        obj = {
            key: {
                k: repr(value)
                if not isinstance(value, type)
                else str(value.__qualname__)
                for k, value in defined.items()
            } for key, defined in self.definitions.items()
        }
        json.dump(obj, *args, **kwargs)


class ConfigKeyError(KeyError):
    pass


class ConfigFile(iterables.MappingBase):
    """A class to handle EPREM run configuration files.

    Parameters
    ---------------------
    filepath : string or path
        The full path to the simulation-run config file to read. This class will
        convert to input string or `pathlib.Path` object into a fully-qualified
        read-only path.

    comments : list of strings, default='#'
        List of single-character strings to interpret as signifying a comment
        line. This method will automatically ignore lines that begin with either
        the empty string or the newline character.

    Notes
    -----
    This interface was designed to provide a faithful representation of the
    information in a given configuration file. A few notable consequences are as
    follows:
    - The look-up methods do not accept aliases as keys.
    - The parsing routine does not attempt to cast parsed values from strings to
      their underlying types, meaning all values are internally represented as
      strings. This only applies to each instance as a whole, since the object
      returned by look-up methods may perform type casting.
    - The unit associated with a specific value, when available, are those
      consistent with the convetions of EPREM (cf. configuration.c) and do not
      necessarily conform to a particular unit system. For example, many
      reference lengths or distances (e.g., reference mean free path or observer
      positions) are in au despite the fact that EPREM works with lengths in cm.
    """
    def __init__(
        self,
        filepath: typing.Union[str, pathlib.Path],
        comments: typing.List[str]=None,
    ) -> None:
        self.filepath = iotools.ReadOnlyPath(filepath)
        self.comments = comments or ['#']
        self.KeyError = ConfigKeyError
        self.parsed = self._parse()
        super().__init__(tuple(self.parsed))

    def __getitem__(self, key: str):
        """Get a value and unit for a configuration option."""
        if key in self.parsed:
            return self.parsed[key]
        raise self.KeyError(key)

    def _parse(self) -> typing.Dict[str, str]:
        """Parse an EPREM config file into a dictionary.

        This method opens the file that the given filepath points to and reads
        it line-by-line. It ignores lines that begin with a valid comment
        character and parses the rest into key-value pairs. It will
        automatically strip out inline comments.
        """
        pairs = {}
        with self.filepath.open('r') as fp:
            for line in fp:
                line = line.rstrip('\n')
                if line == '' or line == '\n' or line[0] in self.comments:
                    continue
                key, _tmp = line.split('=')
                value = iotools.strip_inline_comments(_tmp, self.comments)
                pairs[key] = value
        return pairs

    def __str__(self) -> str:
        """A simplified representation of this object."""
        return str(self.parsed)


class Runtime(iterables.MappingBase):
    """Parameter arguments relevant to an EPREM run.

    An instance of this class represents metadata, default values, and user
    arguments for parameters available to a specific EPREM simulation run.
    """

    def __init__(
        self,
        source_path: iotools.PathLike,
        config_path: iotools.PathLike=None,
        **kwargs
    ) -> None:
        """
        Parameters
        ----------
        source_path : path-like
            The location of the directory containing relevant EPREM source code.

        config_path : path-like, default=None
            The location of the EPREM configuration file containing
            user-provided parameter values for a particular EPREM simulation
            run. May be omitted, in which case this class will use default
            argument values.

        Notes
        -----
        Path-like arguments may be any path-like object (e.g., `str`,
        `pathlib.Path`), may be relative, and may contain standard wildcard
        characters (e.g., `~`).
        """
        self._paths = {
            'source': source_path,
            'config': config_path,
        }
        self._user = None
        self._basetypes = None
        self._reference = None
        super().__init__(tuple(self.reference))
        # BUG: These `kwargs` will persist even if the user updates the
        # config-file path, which could create an inconsistency.
        self._kwargs = kwargs
        self._current = None

    def __getitem__(self, name: str):
        """Retrieve a parameter value by name."""
        if name in self:
            return self._evaluate(name)
        raise KeyError(f"Unknown parameter {name!r}")

    def path_to(self, *current: str, **new: iotools.PathLike):
        """Get or set paths."""
        if current and new:
            raise TypeError("Can't simultaneously get and set paths")
        if current:
            if len(current) == 1:
                return self._paths[current[0]]
            return [self._paths[name] for name in current]
        if new:
            self._paths.update(new)
            self._reset(paths=new)
            return self
        raise TypeError("No paths to get or set")

    def _reset(self, paths: typing.Container=None):
        """Reset attributes based on updates to other attributes."""
        paths = paths or {}
        self._current = None
        if 'config' in paths:
            self._user = None
        if 'source' in paths:
            self._basetypes = None
            self._reference = None

    @property
    def basetypes(self):
        """Values of constants defined in `baseTypes.h`."""
        if self._basetypes is None:
            self._basetypes = BaseTypesH(self._paths['source'])
        return self._basetypes

    @property
    def user(self) -> typing.Mapping[str, str]:
        """The user-provided parameter values."""
        if self._user is None:
            if path := self._paths['config']:
                self._user = ConfigFile(path, **self._kwargs)
            else:
                self._user = {}
        return self._user

    @property
    def reference(self):
        """Parameter metadata from `configuration.c`."""
        if self._reference is None:
            self._reference = ConfigurationC(self._paths['source'])
        return self._reference

    _special_cases = {
        'N_PROCS': None,
        'third': 1/3,
    }

    def _evaluate(self, current: typing.Union[str, numbers.Real]):
        """Compute the final value of a current."""
        if isinstance(current, numbers.Real):
            return current
        if value := numerical.cast(current, strict=False):
            return value
        if current in self._special_cases:
            return self._special_cases[current]
        argument = self._get_argument(current)
        if argument is not None:
            definition = argument['definition']
            if isinstance(definition, argument['type']):
                return definition
            return self._evaluate(definition)
        if isinstance(current, str):
            return self._resolve(current)
        raise TypeError(f"Can't evaluate {current!r}") from None

    _struct_member = re.compile(r'\Aconfig\.\w*\Z')

    def _resolve(self, definition: str):
        """Resolve a parameter definition into simpler components."""
        if self._struct_member.match(definition):
            return self._evaluate(definition.replace('config.', ''))
        if result := self._compute_sum(definition):
            return result
        if any(c in definition for c in {'*', '/'}):
            expression = algebra.Expression(definition)
            evaluated = [
                term.coefficient * self._evaluate(term.base)**term.exponent
                for term in expression
            ]
            return functools.reduce(lambda x,y: x*y, evaluated)
        raise TypeError(f"Can't resolve {definition!r}") from None

    def _compute_sum(self, arg: str) -> numbers.Real:
        """Compute the sum of two known parameters."""
        # HACK: This is only designed to handle strings that contain a single
        # additive operator joining two arguments that `_evaluate` already knows
        # how to handle.
        for operator in ('+', '-'):
            if operator in arg:
                terms = [
                    self._evaluate(s.strip())
                    for s in arg.split(operator)
                ]
                return terms[0] + float(f'{operator}1')*terms[1]

    def _get_argument(self, name: str):
        """Get the current definition and type of a parameter."""
        if self._current is None:
            self._current = {
                key: self._convert(
                    self.user.get(key) or parameter['default'],
                    parameter['type'],
                )
                for key, parameter in self.reference.items()
            }
        if name in self._current:
            return {
                'definition': self._current[name],
                'type': self.reference[name]['type']
            }
        if name in self.basetypes:
            return {
                'definition': self.basetypes[name],
                'type': type(self.basetypes[name]),
            }

    def _convert(self, arg: typing.Union[str, _RT], realtype: _RT):
        """Convert `arg` to its real type, if necessary and possible."""
        if isinstance(arg, realtype):
            return arg
        if realtype in {int, float}:
            return soft_convert(arg, realtype)
        if realtype == list:
            return soft_convert(arg, iterables.string_to_list)
        return arg


class Arguments(aliased.Mapping):
    """Aliased access to EPREM parameter arguments."""

    def __init__(
        self,
        runtime: Runtime=None,
        **runtime_init,
    ) -> None:
        """Initialize an instance of this class.

        The caller may pass an existing instance of `~parameters.Runtime` or the
        arguments necessary to create the appropriate instance. See
        documentation at `~parameters.Runtime` for information. Note that this
        class will use an existing instance of `~parameters.Runtime` if
        available, without checking for the presence of additional arguments.

        Parameters
        ----------
        runtime : optional
            An instance of `parameters.Runtime` initialized with paths relevant
            to the simulation run under analysis.

        **runtime_init : optional
            See `~parameters.Runtime`.
        """
        runtime = runtime or Runtime(**runtime_init)
        super().__init__(mapping=self._build_mapping(runtime))

    def __getitem__(self, key: str):
        """Get the value (and unit, if applicable) of a named parameter."""
        try:
            parameter = super().__getitem__(key)
        except KeyError:
            raise KeyError(f"No parameter corresponding to '{key}'") from None
        value = parameter['value']
        unit = parameter['unit']
        if unit:
            if isinstance(value, list):
                return [quantities.Scalar(v, unit) for v in value]
            return quantities.Scalar(value, unit)
        return value

    def _build_mapping(self, runtime: Runtime):
        """Build the mapping of available parameters."""
        values = {
            **{
                key: info.get('default')
                for key, info in _LOCAL.items()
            },
            **dict(runtime),
        }
        keys = tuple(set(tuple(_LOCAL) + tuple(runtime)))
        base = {
            tuple([key, *_ALIASES.get(key, [])]): {
                'unit': _UNITS.get(key),
                'value': values.get(key),
            }
            for key in keys
        }
        return aliased.MutableMapping(base)


_LOCAL = {
    'minimum_energy': {
        'aliases': ['Emin', 'minimum energy'],
        'unit': 'MeV',
        'default': 0.0,
    },
    'reference energy': {
        'aliases': ['energy0'],
        'unit': 'MeV',
        'default': 1.0,
    },
    'reference radius': {
        'aliases': ['r0'],
        'unit': 'au',
        'default': 1.0,
    },
}
"""Metadata for post-processing parameters."""


_CONFIGURATION_C = {
    'boundaryFunctAmplitude': {
        'aliases': ['J0'],
        'unit': '1 / (cm^2 * s * sr * (MeV/nuc))',
    },
    'boundaryFunctXi': {
        'aliases': ['xi'],
        'unit': '1',
    },
    'boundaryFunctBeta': {
        'aliases': ['beta'],
        'unit': '1',
    },
    'boundaryFunctGamma': {
        'aliases': ['gamma'],
        'unit': '1',
    },
    'boundaryFunctEcutoff': {
        'aliases': ['E0'],
        'unit': 'MeV',
    },
    'kperxkpar': {
        'aliases': ['kper_kpar', 'kper/kpar', 'kper / kpar'],
        'unit': '1',
    },
    'lamo': {
        'aliases': ['lam0', 'lambda0'],
        'unit': 'au',
    },
    'mfpRadialPower': {
        'aliases': ['mfp_radial_power'],
        'unit': '1',
    },
    'rigidityPower': {
        'aliases': ['rigidity_power'],
        'unit': '1',
    },
    'flowMag': {
        'aliases': [],
        'unit': 'km/s',
    },
    'mhdDensityAu': {
        'aliases': [],
        'unit': 'cm^-3',
    },
    'mhdBAu': {
        'aliases': [],
        'unit': 'G',
    },
    'omegaSun': {
        'aliases': [],
        'unit': 'rad * cm / (au * s)',
    },
    'mass': {
        'aliases': [],
        'unit': 'nucleon',
    },
    'charge': {
        'aliases': [],
        'unit': 'e',
    },
    'minInjectionEnergy': {
        'aliases': [],
        'unit': 'MeV',
    },
    'maxInjectionEnergy': {
        'aliases': [],
        'unit': 'MeV',
    },
    'shockInjectionFactor': {
        'aliases': [],
        'unit': '1',
    },
    'shockDetectPercent': {
        'aliases': [],
        'unit': '1',
    },
    'rScale': {
        'aliases': [],
        'unit': 'au',
    },
    'simStartTime': {
        'aliases': [],
        'unit': 'day',
    },
    'simStopTime': {
        'aliases': [],
        'unit': 'day',
    },
    'eMin': {
        'aliases': [],
        'unit': 'MeV/nuc',
    },
    'eMax': {
        'aliases': [],
        'unit': 'MeV/nuc',
    },
    'numObservers': {
        'aliases': [],
        'unit': '1',
    },
    'obsR': {
        'aliases': [],
        'unit': 'au',
    },
    'obsTheta': {
        'aliases': [],
        'unit': 'rad',
    },
    'obsPhi': {
        'aliases': [],
        'unit': 'rad',
    },
    'idw_p': {
        'aliases': [],
        'unit': '1',
    },
    'idealShockSharpness': {
        'aliases': [],
        'unit': '1',
    },
    'idealShockScaleLength': {
        'aliases': [],
        'unit': 'au',
    },
    'idealShockJump': {
        'aliases': [],
        'unit': '1',
    },
    'idealShockSpeed': {
        'aliases': [],
        'unit': 'km/s',
    },
    'idealShockInitTime': {
        'aliases': [],
        'unit': 'day',
    },
    'idealShockTheta': {
        'aliases': [],
        'unit': 'rad',
    },
    'idealShockPhi': {
        'aliases': [],
        'unit': 'rad',
    },
    'idealShockWidth': {
        'aliases': [],
        'unit': 'rad',
    },
    'tDel': {
        'aliases': [],
        'unit': 'day',
    },
    'gammaElow': {
        'aliases': [],
        'unit': '1',
    },
    'gammaEhigh': {
        'aliases': [],
        'unit': '1',
    },
    'masInitTimeStep': {
        'aliases': [],
        'unit': 'day',
    },
    'masStartTime': {
        'aliases': [],
        'unit': 'day',
    },
    'epEquilibriumCalcDuration': {
        'aliases': [],
        'unit': 'day',
    },
    'preEruptionDuration': {
        'aliases': [],
        'unit': 'day',
    },
    'pointObserverOutput': {
        'aliases': [],
        'unit': None,
    },
    'enlilCouple': {
        'aliases': [],
        'unit': None,
    },
    'outputFloat': {
        'aliases': [],
        'unit': None,
    },
    'numRowsPerFace': {
        'aliases': [],
        'unit': None,
    },
    'numColumnsPerFace': {
        'aliases': [],
        'unit': None,
    },
    'numNodesPerStream': {
        'aliases': [],
        'unit': None,
    },
    'numEnergySteps': {
        'aliases': [],
        'unit': None,
    },
    'numMuSteps': {
        'aliases': [],
        'unit': None,
    },
    'useDrift': {
        'aliases': [],
        'unit': None,
    },
    'useShellDiffusion': {
        'aliases': [],
        'unit': None,
    },
    'unifiedOutput': {
        'aliases': [],
        'unit': None,
    },
    'streamFluxOutput': {
        'aliases': [],
        'unit': None,
    },
    'epremDomain': {
        'aliases': [],
        'unit': None,
    },
    'dumpFreq': {
        'aliases': [],
        'unit': None,
    },
    'idealShock': {
        'aliases': [],
        'unit': None,
    },
    'shockSolver': {
        'aliases': [],
        'unit': None,
    },
    'fluxLimiter': {
        'aliases': [],
        'unit': None,
    },
    'numEpSteps': {
        'aliases': [],
        'unit': None,
    },
    'useParallelDiffusion': {
        'aliases': [],
        'unit': None,
    },
    'useAdiabaticChange': {
        'aliases': [],
        'unit': None,
    },
    'useAdiabaticFocus': {
        'aliases': [],
        'unit': None,
    },
    'numSpecies': {
        'aliases': [],
        'unit': None,
    },
    'boundaryFunctionInitDomain': {
        'aliases': [],
        'unit': None,
    },
    'checkSeedPopulation': {
        'aliases': [],
        'unit': None,
    },
    'subTimeCouple': {
        'aliases': [],
        'unit': None,
    },
    'FailModeDump': {
        'aliases': [],
        'unit': None,
    },
    'masCouple': {
        'aliases': [],
        'unit': None,
    },
    'masDirectory': {
        'aliases': [],
        'unit': None,
    },
    'masInitFromOuterBoundary': {
        'aliases': [],
        'unit': None,
    },
    'masRotateSolution': {
        'aliases': [],
        'unit': None,
    },
    'useMasSteadyStateDt': {
        'aliases': [],
        'unit': None,
    },
    'masDigits': {
        'aliases': [],
        'unit': None,
    },
}
"""Metadata for parameters defined in `configuration.c`."""

_BASETYPES_H = {
    'T': {
        'info': 'True',
        'unit': None,
        'type': int,
    },
    'F': {
        'info': 'False',
        'unit': None,
        'type': int,
    },
    'PI': {
        'info': 'The value of π.',
        'unit': None,
        'type': float,
    },
    'TWO_PI': {
        'info': 'The value of 2π.',
        'unit': None,
        'type': float,
    },
    'VERYSMALL': {
        'info': 'A very small value.',
        'unit': None,
        'type': float,
    },
    'BADVALUE': {
        'info': 'A bad (invalid) float value.',
        'unit': None,
        'type': float,
    },
    'BADINT': {
        'info': 'A bad (invalid) integer value.',
        'unit': None,
        'type': int,
    },
    'MP': {
        'info': 'The proton mass.',
        'unit': 'g',
        'type': float,
    },
    'EV': {
        'info': 'The conversion from eVs to ergs.',
        'unit': 'erg/eV',
        'type': float,
    },
    'MEV': {
        'info': 'The conversion from MeVs to ergs.',
        'unit': 'erg/MeV',
        'type': float,
    },
    'GEV': {
        'info': 'The conversion from GeVs to ergs.',
        'unit': 'erg/GeV',
        'type': float,
    },
    'Q': {
        'info': 'The proton charge.',
        'unit': 'statC',
        'type': float,
    },
    'C': {
        'info': 'The speed of light.',
        'unit': 'cm/s',
        'type': float,
    },
    'MZERO': {
        'info': 'The proton rest-mass energy in GeV.',
        'unit': 'GeV',
        'type': float,
    },
    'AU': {
        'info': 'One astronomical unit.',
        'unit': 'cm',
        'type': float,
    },
    'RSUN': {
        'info': 'The value of the solar radius.',
        'unit': 'cm',
        'type': float,
    },
    'RSAU': {
        'info': 'The number of solar radii per au.',
        'unit': '1',
        'type': float,
    },
    'TAU': {
        'info': 'The canonical EPREM time scale.',
        'unit': 's',
        'type': float,
    },
    'DAY': {
        'info': 'The conversion from EPREM time steps to Julian days.',
        'unit': 'day',
        'type': float,
    },
    'MHD_DENSITY_NORM': {
        'info': 'The normalization factor for density.',
        'unit': '1',
        'type': float,
    },
    'MHD_B_NORM': {
        'info': 'The normalization for magnetic fields.',
        'unit': '1',
        'type': float,
    },
    'OM': {
        'info': 'The normalization for ion gyrofrequency.',
        'unit': '1',
        'type': float,
    },
    'FCONVERT': {
        'info': 'The conversion from distribution to flux.',
        'unit': '1',
        'type': float,
    },
    'VOLT': {
        'info': 'The conversion from volts to statvolts.',
        'unit': '1',
        'type': float,
    },
    'THRESH': {
        'info': 'The threshold for perpendicular diffusion.',
        'unit': '1',
        'type': float,
    },
    'MAS_TIME_NORM': {
        'info': 'The MAS time normalization factor.',
        'unit': '1',
        'type': float,
    },
    'MAS_LENGTH_NORM': {
        'info': 'The MAS length normalization factor.',
        'unit': '1',
        'type': float,
    },
    'MAS_RHO_NORM': {
        'info': 'The MAS plasma-density normalization.',
        'unit': '1',
        'type': float,
    },
    'MAS_TIME_CONVERT': {
        'info': 'The time conversion from MAS units.',
        'unit': '1',
        'type': float,
    },
    'MAS_V_CONVERT': {
        'info': 'The velocity conversion from MAS units.',
        'unit': '1',
        'type': float,
    },
    'MAS_RHO_CONVERT': {
        'info': 'The density conversion from MAS units.',
        'unit': '1',
        'type': float,
    },
    'MAS_B_CONVERT': {
        'info': 'The magnetic field conversion from MAS units.',
        'unit': '1',
        'type': float,
    },
    'MAX_STRING_SIZE': {
        'info': '',
        'unit': None,
        'type': int,
    },
    'MHD_DEFAULT': {
        'info': 'Use the default MHD solver.',
        'unit': None,
        'type': int,
    },
    'MHD_ENLIL': {
        'info': 'Use ENLIL for MHD values.',
        'unit': None,
        'type': int,
    },
    'MHD_LFMH': {
        'info': 'Use LFM for MHD values.',
        'unit': None,
        'type': int,
    },
    'MHD_BATSRUS': {
        'info': 'Use BATS-R-US for MHD values.',
        'unit': None,
        'type': int,
    },
    'MHD_MAS': {
        'info': 'Use MAS for MHD values.',
        'unit': None,
        'type': int,
    },
    'NUM_MPI_BOUNDARY_FLDS': {
        'info': 'Number of MPI psuedo-fields for use in creating MPI typedefs.',
        'unit': None,
        'type': int,
    },
}
"""Metadata for parameters defined in `baseTypes.h`."""

_metadata = {**_LOCAL, **_CONFIGURATION_C, **_BASETYPES_H}
"""Combined metadata dictionary for internal use."""

_ALIASES = {
    key: info.get('aliases')
    for key, info in _metadata.items()
}
"""Collection of aliases from metadata."""

_UNITS = {
    key: info.get('unit')
    for key, info in _metadata.items()
}
"""Collection of units from metadata."""


def generate_defaults(path: iotools.PathLike):
    """Generate default arguments from the EPREM source code in `path`."""
    targets = {
        '_BASETYPES_H': BaseTypesH(path),
        '_CONFIGURATION_C': ConfigurationC(path)
    }
    filedir = pathlib.Path(__file__).expanduser().resolve().parent
    for name, target in targets.items():
        outpath = filedir / f'{name}.json'
        with outpath.open('w') as fp:
            target.dump(fp, indent=4, sort_keys=True)


if __name__ == '__main__':
    doclines = __doc__.split('\n')
    parser = argparse.ArgumentParser(
        description=doclines[0],
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        '-r',
        '--generate_defaults',
        help=generate_defaults.__doc__.replace('`path`', 'SRC'),
        metavar='SRC',
    )
    args = parser.parse_args()
    kwargs = vars(args)
    if 'generate_defaults' in kwargs:
        generate_defaults(kwargs['generate_defaults'])

