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

import collections.abc
from pathlib import Path
from typing import *

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
    'mass': {'type': list, 'unit': 'nucleon'},
    'charge': {'type': list, 'unit': 'e'},
    'minInjectionEnergy': {'type': float, 'unit': 'MeV'},
    'maxInjectionEnergy': {'type': float, 'unit': 'MeV'},
    'shockInjectionFactor': {'type': float, 'unit': '1'},
    'shockDetectPercent': {'type': float, 'unit': '1'},
    'rScale': {'type': float, 'unit': 'au'},
    'simStartTime': {'type': float, 'unit': 'day'},
    'simStopTime': {'type': float, 'unit': 'day'},
    'eMin': {'type': float, 'unit': 'MeV/nuc'},
    'eMax': {'type': float, 'unit': 'MeV/nuc'},
    'numObservers': {'type': int, 'unit': '1'},
    'obsR': {'type': list, 'unit': 'au'},
    'obsTheta': {'type': list, 'unit': 'rad'},
    'obsPhi': {'type': list, 'unit': 'rad'},
    'idw_p': {'type': float, 'unit': '1'},
    'idealShockSharpness': {'type': float, 'unit': '1'},
    'idealShockScaleLength': {'type': float, 'unit': 'au'},
    'idealShockJump': {'type': float, 'unit': '1'},
    'idealShockSpeed': {'type': float, 'unit': 'km/s'},
    'idealShockInitTime': {'type': float, 'unit': 'day'},
    'idealShockTheta': {'type': float, 'unit': 'rad'},
    'idealShockPhi': {'type': float, 'unit': 'rad'},
    'idealShockWidth': {'type': float, 'unit': 'rad'},
    'tDel': {'type': float, 'unit': 'day'},
    'gammaElow': {'type': float, 'unit': '1'},
    'gammaEhigh': {'type': float, 'unit': '1'},
    'masInitTimeStep': {'type': float, 'unit': 'day'},
    'masStartTime': {'type': float, 'unit': 'day'},
    'epEquilibriumCalcDuration': {'type': float, 'unit': 'day'},
    'preEruptionDuration': {'type': float, 'unit': 'day'},
}
"""Physics-based parameter values that control simulation execution."""

_options = {
    'pointObserverOutput': {'type': bool},
    'enlilCouple': {'type': bool},
    'outputFloat': {'type': bool},
    'numRowsPerFace': {'type': int},
    'numColumnsPerFace': {'type': int},
    'numNodesPerStream': {'type': int},
    'numEnergySteps': {'type': int},
    'numMuSteps': {'type': int},
    'useDrift': {'type': bool},
    'useShellDiffusion': {'type': bool},
    'unifiedOutput': {'type': bool},
    'streamFluxOutput': {'type': bool},
    'epremDomain': {'type': bool},
    'dumpFreq': {'type': bool},
    'idealShock': {'type': bool},
    'shockSolver': {'type': bool},
    'fluxLimiter': {'type': bool},
    'numEpSteps': {'type': int},
    'useParallelDiffusion': {'type': bool},
    'useAdiabaticChange': {'type': bool},
    'useAdiabaticFocus': {'type': bool},
    'numSpecies': {'type': int},
    'boundaryFunctionInitDomain': {'type': bool},
    'checkSeedPopulation': {'type': bool},
    'subTimeCouple': {'type': bool},
    'FailModeDump': {'type': bool},
    'masCouple': {'type': bool},
    'masDirectory': {'type': str},
    'masInitFromOuterBoundary': {'type': bool},
    'masRotateSolution': {'type': bool},
    'useMasSteadyStateDt': {'type': bool},
    'masDigits': {'type': int},
}
"""Non-physical simulation runtime options."""


_metadata = {
    'assumptions': _assumptions,
    'constraints': _constraints,
    'options': _options,
}

metadata = {k: iterables.AliasedMapping.of(v) for k, v in _metadata.items()}


class ConfigKeyError(KeyError):
    pass


class ConfigManager(collections.abc.Mapping):
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
      consistent with the convetions of EPREM (cf. configuration.h) and do not
      necessarily conform to a particular unit system. For example, many
      reference lengths or distances (e.g., reference mean free path or observer
      positions) are in au despite the fact that EPREM works with lengths in cm.
    """
    def __init__(
        self,
        filepath: Union[str, Path],
        comments: List[str]=None,
    ) -> None:
        self.filepath = iotools.ReadOnlyPath(filepath)
        self.comments = comments or ['#']
        self.KeyError = ConfigKeyError
        self._parsed = None

    def __iter__(self) -> Iterator:
        return iter(self.parsed)

    def __len__(self) -> int:
        return len(self.parsed)

    def __contains__(self, key: str) -> bool:
        return key in self.parsed

    @property
    def parsed(self) -> Dict[str, Any]:
        """Key-value pairs parsed from the configuration file."""
        if self._parsed is None:
            self._parsed = self._parse()
        return self._parsed

    def __getitem__(self, key: str):
        """Get a value and unit for a configuration option."""
        if key in self.parsed:
            return self.parsed[key]
        raise self.KeyError(key)

    def _parse(self) -> None:
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

    def __repr__(self) -> str:
        """An unambiguous representation of this object."""
        return f"{self.__class__.__qualname__}({self.parsed!r})"


class Reference(iterables.ReprStrMixin):
    """Reference metadata for an EPREM parameter."""

    def __init__(self, metadata: Dict[str, Any]) -> None:
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

    def __init__(self, reference: Reference, value: Any=None) -> None:
        self.reference = reference
        self._types = tuple({str, reference.realtype})
        self.value = self._normalize(value) if value else reference.value

    def update(self, value: Any):
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
        aliasmap = {name: info for name, info in _metadata.items()}
        self._namemap = iterables.NameMap(_metadata.keys(), aliasmap)
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

