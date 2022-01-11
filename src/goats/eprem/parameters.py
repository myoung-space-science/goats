"""Simulation configuration parameters and user-defined physical quantities.

This module provides direct access to EPREM simulation runtime configuration
files (a.k.a config files) via the `ConfigManager` class, as well as objects to manage user-defined physical quantities with associated units.

Note that the word "parameter" can have multiple meanings in the contexts to
which this module is relevant. The python built-in module `inspect` uses
"parameters" to mean the names representing positional and keyword arguments.
This is consistent with formal definitions of a function parameter (cf.
https://developer.mozilla.org/en-US/docs/Glossary/Parameter). It is also common
to refer to config-file options as parameters, in a way that is similar but not
identical to the formal software defintion above. This module represents objects
of the latter type, and user code should take care to distuinguish between the
various meanings.
"""

import functools
import numbers
import pathlib
import re
import typing

from goats.common import algebra
from goats.common import constants
from goats.common import iotools
from goats.common import iterables
from goats.common import quantities


_assumptions = {
    'boundaryFunctAmplitude': {
        'aliases': ('J0',),
        'type': float,
        'default': (1.0, '1 / (cm^2 * s * sr * (MeV/nuc))'),
    },
    'boundaryFunctXi': {
        'aliases': ('xi',),
        'type': float,
        'default': 1.0,
    },
    'boundaryFunctBeta': {
        'aliases': ('beta',),
        'type': float,
        'default': 2.0,
    },
    'boundaryFunctGamma': {
        'aliases': ('gamma',),
        'type': float,
        'default': 1.5,
    },
    'boundaryFunctEcutoff': {
        'aliases': ('E0',),
        'type': float,
        'default': (1.0, 'MeV'),
    },
    'kperxkpar': {
        'aliases': ('kper_kpar', 'kper/kpar', 'kper / kpar'),
        'default': 0.01,
    },
    'lamo': {
        'aliases': ('lam0', 'lambda0',),
        'type': float,
        'default': (1.0, 'au'),
    },
    'minimum_energy': {
        'aliases': ('Emin', 'minimum energy'),
        'type': float,
        'default': (0.0, 'MeV'),
    },
    'mfpRadialPower': {
        'aliases': ('mfp_radial_power',),
        'type': float,
        'default': 2.0,
    },
    'rigidityPower': {
        'aliases': ('rigidity_power',),
        'type': float,
        'default': 1/3,
    },
    'flowMag': {
        'type': float,
        'default': (400.0, 'km/s'),
    },
    'mhdDensityAu': {
        'type': float,
        'default': (5.0, 'cm^-3'),
    },
    'mhdBAu': {
        'type': float,
        'default': (1e-5, 'G'),
    },
    'omegaSun': {
        'type': float,
        'default': (1e-3, 'rad * cm / (au * s)'),
    },
    'reference energy': {
        'aliases': ('energy0',),
        'type': float,
        'default': (1.0, 'MeV'),
    },
    'reference radius': {
        'aliases': ('r0',),
        'type': float,
        'default': (1.0, 'au'),
    },
}
"""Physical quantities with default values and units."""

_constraints = {
    'mass': {
        'type': list,
        'default': (None, 'nucleon'),
    },
    'charge': {
        'type': list,
        'default': (None, 'e'),
    },
    'minInjectionEnergy': {
        'type': float,
        'default': (None, 'MeV'),
    },
    'maxInjectionEnergy': {
        'type': float,
        'default': (None, 'MeV'),
    },
    'shockInjectionFactor': {
        'type': float,
        'default': (None, '1'),
    },
    'shockDetectPercent': {
        'type': float,
        'default': (None, '1'),
    },
    'rScale': {
        'type': float,
        'default': (None, 'au'),
    },
    'simStartTime': {
        'type': float,
        'default': (None, 'day'),
    },
    'simStopTime': {
        'type': float,
        'default': (None, 'day'),
    },
    'eMin': {
        'type': float,
        'default': (None, 'MeV/nuc'),
    },
    'eMax': {
        'type': float,
        'default': (None, 'MeV/nuc'),
    },
    'numObservers': {
        'type': int,
        'default': (None, '1'),
    },
    'obsR': {
        'type': list,
        'default': (None, 'au'),
    },
    'obsTheta': {
        'type': list,
        'default': (None, 'rad'),
    },
    'obsPhi': {
        'type': list,
        'default': (None, 'rad'),
    },
    'idw_p': {
        'type': float,
        'default': (None, '1'),
    },
    'idealShockSharpness': {
        'type': float,
        'default': (None, '1'),
    },
    'idealShockScaleLength': {
        'type': float,
        'default': (None, 'au'),
    },
    'idealShockJump': {
        'type': float,
        'default': (None, '1'),
    },
    'idealShockSpeed': {
        'type': float,
        'default': (None, 'km/s'),
    },
    'idealShockInitTime': {
        'type': float,
        'default': (None, 'day'),
    },
    'idealShockTheta': {
        'type': float,
        'default': (None, 'rad'),
    },
    'idealShockPhi': {
        'type': float,
        'default': (None, 'rad'),
    },
    'idealShockWidth': {
        'type': float,
        'default': (None, 'rad'),
    },
    'tDel': {
        'type': float,
        'default': (None, 'day'),
    },
    'gammaElow': {
        'type': float,
        'default': (None, '1'),
    },
    'gammaEhigh': {
        'type': float,
        'default': (None, '1'),
    },
    'masInitTimeStep': {
        'type': float,
        'default': (None, 'day'),
    },
    'masStartTime': {
        'type': float,
        'default': (None, 'day'),
    },
    'epEquilibriumCalcDuration': {
        'type': float,
        'default': (None, 'day'),
    },
    'preEruptionDuration': {
        'type': float,
        'default': (None, 'day'),
    },
}
"""Physics-based parameter values that control simulation execution."""

_options = {
    'pointObserverOutput': {
        'type': bool,
        'default': None,
    },
    'enlilCouple': {
        'type': bool,
        'default': None,
    },
    'outputFloat': {
        'type': bool,
        'default': None,
    },
    'numRowsPerFace': {
        'type': int,
        'default': None,
    },
    'numColumnsPerFace': {
        'type': int,
        'default': None,
    },
    'numNodesPerStream': {
        'type': int,
        'default': None,
    },
    'numEnergySteps': {
        'type': int,
        'default': None,
    },
    'numMuSteps': {
        'type': int,
        'default': None,
    },
    'useDrift': {
        'type': bool,
        'default': None,
    },
    'useShellDiffusion': {
        'type': bool,
        'default': None,
    },
    'unifiedOutput': {
        'type': bool,
        'default': None,
    },
    'streamFluxOutput': {
        'type': bool,
        'default': None,
    },
    'epremDomain': {
        'type': bool,
        'default': None,
    },
    'dumpFreq': {
        'type': bool,
        'default': None,
    },
    'idealShock': {
        'type': bool,
        'default': None,
    },
    'shockSolver': {
        'type': bool,
        'default': None,
    },
    'fluxLimiter': {
        'type': bool,
        'default': None,
    },
    'numEpSteps': {
        'type': int,
        'default': None,
    },
    'useParallelDiffusion': {
        'type': bool,
        'default': None,
    },
    'useAdiabaticChange': {
        'type': bool,
        'default': None,
    },
    'useAdiabaticFocus': {
        'type': bool,
        'default': None,
    },
    'numSpecies': {
        'type': int,
        'default': None,
    },
    'boundaryFunctionInitDomain': {
        'type': bool,
        'default': None,
    },
    'checkSeedPopulation': {
        'type': bool,
        'default': None,
    },
    'subTimeCouple': {
        'type': bool,
        'default': None,
    },
    'FailModeDump': {
        'type': bool,
        'default': None,
    },
    'masCouple': {
        'type': bool,
        'default': None,
    },
    'masDirectory': {
        'type': str,
        'default': None,
    },
    'masInitFromOuterBoundary': {
        'type': bool,
        'default': None,
    },
    'masRotateSolution': {
        'type': bool,
        'default': None,
    },
    'useMasSteadyStateDt': {
        'type': bool,
        'default': None,
    },
    'masDigits': {
        'type': int,
        'default': None,
    },
}
"""Non-physical simulation runtime options."""


_metadata = {
    'assumptions': _assumptions,
    'constraints': _constraints,
    'options': _options,
}

metadata = {k: iterables.AliasedMapping.of(v) for k, v in _metadata.items()}


_BASETYPES_H = {
    'T': {
        'info': 'True',
        'type': int,
    },
    'F': {
        'info': 'False',
        'type': int,
    },
    'PI': {
        'info': 'The value of π.',
        'type': float,
    },
    'TWO_PI': {
        'info': 'The value of 2π.',
        'type': float,
    },
    'VERYSMALL': {
        'info': 'A very small value.',
        'type': float,
    },
    'BADVALUE': {
        'info': 'A bad (invalid) float value.',
        'type': float,
    },
    'BADINT': {
        'info': 'A bad (invalid) integer value.',
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
        'formula': 'MP * C^2 * GEV^-1',
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
        'formula': 'RSUN * AU^-1',
        'type': float,
    },
    'TAU': {
        'info': 'The canonical EPREM time scale.',
        'unit': 's',
        'formula': 'AU * C^-1',
        'type': float,
    },
    'DAY': {
        'info': 'The conversion from EPREM time steps to Julian days.',
        'unit': 'day',
        'formula': 'TAU * 1.1574074074074072e-05', 
        'type': float,
    },
    'MHD_DENSITY_NORM': {
        'info': 'The normalization factor for density.',
        'type': float,
    },
    'MHD_B_NORM': {
        'info': 'The normalization for magnetic fields.',
        'formula': 'MP^0.5 * MHD_DENSITY_NORM^0.5 * C',
        'type': float,
    },
    'OM': {
        'info': 'The normalization for ion gyrofrequency.',
        'formula': 'MHD_B_NORM * Q * AU * MP^-1 * C^-2',
        'type': float,
    },
    'FCONVERT': {
        'info': 'The conversion from distribution to flux.',
        'formula': 'C^-4 * 1e+30',
        'type': float,
    },
    'VOLT': {
        'info': 'The conversion from volts to statvolts.',
        'type': float,
    },
    'THRESH': {
        'info': 'The threshold for perpendicular diffusion.',
        'type': float,
    },
    'MAS_TIME_NORM': {
        'info': 'The MAS time normalization factor.',
        'type': float,
    },
    'MAS_LENGTH_NORM': {
        'info': 'The MAS length normalization factor.',
        'type': float,
    },
    'MAS_RHO_NORM': {
        'info': 'The MAS plasma-density normalization.',
        'type': float,
    },
    'MAS_TIME_CONVERT': {
        'info': 'The time conversion from MAS units.',
        'formula': 'MAS_TIME_NORM * 1.1574074074074072e-05',
        'type': float,
    },
    'MAS_V_CONVERT': {
        'info': 'The velocity conversion from MAS units.',
        'formula': 'MAS_LENGTH_NORM * MAS_TIME_NORM^-1 * C^-1',
        'type': float,
    },
    'MAS_RHO_CONVERT': {
        'info': 'The density conversion from MAS units.',
        'formula': 'MAS_RHO_NORM * MP^-1 * MHD_DENSITY_NORM^-1',
        'type': float,
    },
    'MAS_B_CONVERT': {
        'info': 'The magnetic field conversion from MAS units.',
        'formula': (
            'PI^0.5 * MAS_RHO_NORM^0.5 * MAS_LENGTH_NORM * MAS_TIME_NORM^-1 '
            '* MHD_B_NORM^-1 * 2.0'
        ),
        'type': float,
    },
    'MAX_STRING_SIZE': {
        'type': int,
    },
    'MHD_DEFAULT': {
        'info': 'Use the default MHD solver.',
        'type': int,
    },
    'MHD_ENLIL': {
        'info': 'Use ENLIL for MHD values.',
        'type': int,
    },
    'MHD_LFMH': {
        'info': 'Use LFM for MHD values.',
        'type': int,
    },
    'MHD_BATSRUS': {
        'info': 'Use BATS-R-US for MHD values.',
        'type': int,
    },
    'MHD_MAS': {
        'info': 'Use MAS for MHD values.',
        'type': int,
    },
    'NUM_MPI_BOUNDARY_FLDS': {
        'info': 'Number of MPI psuedo-fields for use in creating MPI typedefs.',
        'type': int,
    },
}


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
                        if 'formula' in metadata:
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

    def print(self, tabsize: int=4, stream: typing.TextIO=None):
        """Print formatted reference information."""
        indent = ' ' * tabsize
        print("{", file=stream)
        for key, definition in self.definitions.items():
            formula = (
                definition.format(separator=' * ')
                if isinstance(definition, algebra.Expression)
                else None
            )
            print(
                f"{indent}{key!r}: {'{'}\n"
                f"{indent}{indent}'value': {self[key]},\n"
                f"{indent}{indent}'formula': {formula!r},\n"
                f"{indent}{'},'}",
                file=stream,
            )
        print("}", file=stream)


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


class ConfigurationC(iterables.MappingBase):
    """A representation of EPREM `configuration.c`."""

    def __init__(self, src: typing.Union[str, pathlib.Path]=None) -> None:
        path = pathlib.Path(src or '.') / 'configuration.c'
        self.file = iotools.TextFile(path)
        readline = FunctionCall()
        self._defined = self.file.extract(readline.match, readline.parse)
        super().__init__(tuple(self._defined))
        scalar_t = VariableDefinition()
        self._defaults = self.file.extract(scalar_t.match, scalar_t.parse)
        self._path = path

    @property
    def path(self):
        """The path to the relevant EPREM distribution."""
        return iotools.ReadOnlyPath(self._path)

    def __getitem__(self, key: str):
        """"""
        if key in self:
            return self._prepare(self._defined[key])
        raise KeyError(f"No default for {key!r}")

    def _prepare(self, defined: str):
        """"""
        mode = defined['mode']
        args = self._normalize(defined['args'])
        if mode == 'readInt':
            keys = ('default', 'minimum', 'maximum')
            return {'type': int, **dict(zip(keys, args))}
        if mode == 'readDouble':
            keys = ('default', 'minimum', 'maximum')
            return {'type': float, **dict(zip(keys, args))}
        if mode == 'readString':
            return {'type': str, 'default': args[0].strip('"')}
        if mode == 'readDoubleArray':
            return {'type': list, 'default': args[1]}
        raise ValueError(f"Unknown configuration mode {mode!r}")

    def _normalize(self, args: typing.Iterable[str]) -> typing.List[str]:
        """"""
        result = []
        for arg in args:
            if arg in self._defaults:
                result.append(self._defaults[arg]['value'])
            else:
                result.append(arg)
        return result

    def print(self, tabsize: int=4, stream: typing.TextIO=None):
        """Print formatted reference information."""
        indent = ' ' * tabsize
        print("{", file=stream)
        for key, defined in self.items():
            args = ''.join(
                f"{indent}{indent}{k!r}: {v!r},\n"
                for k, v in defined.items()
            )
            print(
                f"{indent}{key!r}: {'{'}\n"
                f"{args}",
                f"{indent}{'},'}",
                file=stream,
            )
        print("}", file=stream)


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


class ConfigManager(iterables.MappingBase):
    """Interface to EPREM runtime configuration."""

    def __init__(
        self,
        source_path: iotools.PathLike,
        config_path: iotools.PathLike=None,
        **kwargs
    ) -> None:
        self._paths = {
            'source': source_path,
            'config': config_path,
        }
        super().__init__(tuple(self.defaults))
        self._kwargs = kwargs

    def path_to(self, *current: str, **new: iotools.PathLike):
        """Get or set paths."""
        if current and new:
            raise TypeError("Can't simultaneously get an set paths")
        if current:
            if len(current) == 1:
                return self._paths[current[0]]
            return [self._paths[name] for name in current]
        if new:
            self._paths.update(new)
            return self
        raise TypeError("No paths to get or set")

    # TODO: Keep track of updates to prevent creating new instances with every
    # call to `runtime`, `defaults`, and `basetypes`.

    @property
    def runtime(self) -> typing.Mapping[str, str]:
        """The runtime parameter values."""
        if path := self._paths['config']:
            return ConfigFile(path, **self._kwargs)
        return {}

    @property
    def defaults(self):
        """Default parameter definitions in `configuration.c`."""
        return ConfigurationC(self._paths['source'])

    @property
    def basetypes(self):
        """Values of constants defined in `baseTypes.h`."""
        return BaseTypesH(self._paths['source'])

    def __getitem__(self, key: str):
        """"""
        if key in self:
            return self._get_value(key)
        raise KeyError(f"Unknown configuration parameter {key!r}")

    # TODO: Either cache requested values or pre-evaluate all values.

    def _get_value(self, key: str):
        """"""
        parameter = self.defaults[key]
        realtype = parameter['type']
        if key in self.runtime:
            value = self.runtime[key]
            convert = (
                iterables.string_to_list if realtype == list
                else realtype
            )
            return convert(value)
        return self._evaluate(parameter['default'], realtype)

    _special_cases = {
        'N_PROCS': None,
        'third': 1/3,
    }

    _struct_member = re.compile(r'\Aconfig\.\w*\Z')

    # TODO: Consider running through defaults once and attempting to convert as
    # many values as possible to their respective real type.

    def _evaluate(
        self,
        arg: typing.Union[str, numbers.Real],
        realtype: typing.Type,
    ) -> numbers.Real:
        """Compute the default numerical value from a definition."""
        if isinstance(arg, numbers.Real):
            return arg
        if arg in self.basetypes:
            return self.basetypes[arg]
        if arg in self._special_cases:
            return self._special_cases[arg]
        if self._struct_member.match(arg):
            return self._evaluate(
                arg.replace('config.', ''),
                realtype,
            )
        if result := self._compute_sum(arg, realtype):
            return result
        if any(c in arg for c in {'*', '/'}):
            expression = algebra.Expression(arg)
            evaluated = [
                term.coefficient * self._evaluate(
                    term.base, realtype
                ) ** term.exponent
                for term in expression
            ]
            return functools.reduce(lambda x,y: x*y, evaluated)
        if arg in self.defaults:
            return self._convert(self.defaults[arg]['default'], realtype)
        return self._convert(arg, realtype)

    def _compute_sum(
        self,
        arg: str,
        realtype: typing.Type,
    ) -> numbers.Real:
        """"""
        # HACK: This is only designed to handle strings that contain a single
        # additive operator joining two arguments that `_evaluate` already knows
        # how to handle. It first needs to catch strings with '+/-' in
        # exponential notation.
        for operator in ('+', '-'):
            if operator in arg:
                try:
                    asfloat = float(arg)
                except ValueError:
                    terms = [
                        self._evaluate(s.strip(), realtype)
                        for s in arg.split(operator)
                    ]
                    return terms[0] + float(f'{operator}1')*terms[1]
                else:
                    return asfloat

    def _convert(
        self,
        arg: str,
        realtype: typing.Type,
    ) -> numbers.Real:
        """"""
        if realtype == list:
            return [float(v) for v in iterables.Separable(arg)]
        return realtype(arg)

    def __str__(self) -> str:
        """A simplified representation of this object."""
        paths = ''.join(
            f"    {k}: {v},\n"
            for k, v in self._paths.items()
        )
        return '{\n'f"{paths}"'}'


class Reference(iterables.ReprStrMixin):
    """Reference metadata for an EPREM parameter."""

    def __init__(self, metadata: typing.Dict[str, typing.Any]) -> None:
        self._metadata = metadata or {}
        self._unit = None
        self._value = None

    def interpret(self, string: str):
        """Attempt to extract a value from the given string."""
        return (
            iterables.string_to_list(string)
            if self.realtype is list else self.realtype(string)
        )

    @property
    def realtype(self) -> type:
        """Type to which to cast a value parsed from a config file.

        This will first check for the type in the given metadata. If it does not
        find one, it will attempt to guess the type from the default value. If
        that also fails, it will fall back to `str`.
        """
        if 'type' in self._metadata:
            return self._metadata['type']
        if self.value is not None:
            return type(self.value)
        return str

    @property
    def value(self):
        """The default value."""
        if self._value is None:
            self._value = self._metadata.get('default')
        return self._value

    def __str__(self) -> str:
        """A simplified representation of this object."""
        strtype = self.realtype.__qualname__
        return f"unit='{self.unit}', type={strtype}, default={self.value}"


class Parameter(iterables.ReprStrMixin):
    """Base class for EPREM parameters."""

    def __init__(self, reference: Reference, value: typing.Any=None) -> None:
        self.reference = reference
        self._types = tuple({str, reference.realtype})
        self.value = self._normalize(value) if value else reference.value

    def update(self, value: typing.Any):
        """Update this parameter's value."""
        self.value = self._normalize(value)
        return self

    def _normalize(self, value):
        """Internal logic for self-consistently setting the parameter value."""
        if not isinstance(value, self._types):
            errmsg = f"Value may be one of {self._types} (got {type(value)})"
            raise TypeError(errmsg)
        return (
            self.reference.interpret(value) if isinstance(value, str)
            else value
        )

    def __str__(self) -> str:
        """A simplified version of this object."""
        return str(self.value)


class Assumption(quantities.Scalar):
    """A single EPREM physical assumption."""

    def __init__(self, parameter: Parameter) -> None:
        amount = parameter.value
        unit = parameter.reference.unit
        super().__init__(amount, unit=unit)
        self.parameter = parameter

    @property
    def asscalar(self):
        """A new `Scalar` object equivalent to this assumption."""
        return quantities.Scalar(self.amount, unit=self.unit)


class Option(iterables.ReprStrMixin):
    """A single EPREM runtime option."""

    def __init__(self, parameter: Parameter) -> None:
        self._value = parameter.value

    def __bool__(self) -> bool:
        """Called for bool(self)."""
        return bool(self._value)

    def __int__(self) -> int:
        """Called for int(self)."""
        return int(self._value)

    def __str__(self) -> str:
        """A simplified representation of this object."""
        return str(self._value)


class Constraint(Option):
    """A single EPREM simulation constraint."""

    def __init__(self, parameter: Parameter) -> None:
        self._value = parameter.value
        self._unit = parameter.reference.unit

    def __float__(self) -> float:
        """Called for float(self)."""
        return float(self._value)

    def __str__(self) -> str:
        return ', '.join((super().__str__(), f"[{self._unit}]"))


class Parameters(iterables.AliasedMapping):
    """An aliased mapping of EPREM runtime parameters."""

    def __init__(self, config: ConfigManager) -> None:
        _metadata = {
            **_options,
            **_constraints,
            **_assumptions,
        }
        mapping = {
            tuple([key, *info.get('aliases', ())]): {
                k: v for k, v in info.items() if k != 'aliases'
            }
            for key, info in _metadata.items()
        }
        super().__init__(mapping=mapping)
        self.config = config
        self._namemap = iterables.NameMap(_metadata.keys(), dict(_metadata))
        self._options = metadata['options']
        self._constraints = metadata['constraints']
        self._assumptions = metadata['assumptions']

    def __getitem__(self, key: str):
        """Aliased access to parameter values."""
        if key in self:
            reference = Reference(super().__getitem__(key))
            parameter = Parameter(reference, value=self._get_value(key))
            new = self._get_return_type(key)
            return new(parameter)
        raise KeyError(f"No parameter corresponding to '{key}'")

    def _get_value(self, key: str):
        """Get a value from the config file, if possible."""
        name = self._namemap[key]
        if name in self.config:
            return self.config[name]

    def _get_return_type(self, key: str):
        """Get the appropriate type in which to return a parameter."""
        if key in self._options:
            return Option
        if key in self._constraints:
            return Constraint
        if key in self._assumptions:
            return Assumption
        raise KeyError(f"No object associated with '{key}'")

