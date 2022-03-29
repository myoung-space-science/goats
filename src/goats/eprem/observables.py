import typing

import numpy as np

from goats.core import base
from goats.core import aliased
from goats.core import quantities
from goats.core import algebra
from goats.core import iterables
from goats.core import physical
from goats.core import datasets
from goats.core import datatypes
from goats.eprem import functions
from goats.eprem import parameters
from goats.eprem import interpolation


MKS = quantities.MetricSystem('mks')


Observable = typing.TypeVar(
    'Observable',
    datatypes.Variable,
    functions.Function,
)
Observable = typing.Union[
    datatypes.Variable,
    functions.Function,
]


Assumption = typing.TypeVar(
    'Assumption',
    quantities.Scalar,
    typing.Iterable[quantities.Scalar],
)
Assumption = typing.Union[
    quantities.Scalar,
    typing.Iterable[quantities.Scalar],
]


Implementation = typing.TypeVar(
    'Implementation',
    Observable,
    algebra.Expression,
)
Implementation = typing.Union[
    Observable,
    algebra.Expression,
]


Dependency = typing.TypeVar(
    'Dependency',
    Observable,
    quantities.Scalar,
)
Dependency = typing.Union[
    Observable,
    quantities.Scalar,
]


Reference = typing.TypeVar(
    'Reference',
    datatypes.Variable,
    typing.Iterable,
)
Reference = typing.Union[
    datatypes.Variable,
    typing.Iterable,
]


class Compound(typing.NamedTuple):
    """An algebraic combination of primary or derived observables."""

    expression: algebra.Expression
    axes: typing.Tuple[str]
    name: str=None


class Application:
    """The object that handles evaluating all observables."""

    def __init__(
        self,
        indices: typing.Mapping[str, datatypes.Indices],
        assumptions: typing.Mapping[str, Assumption],
        observables: typing.Mapping[str, Observable],
        reference: typing.Mapping[str, Reference],
    ) -> None:
        self.indices = indices
        self.assumptions = assumptions
        self.observables = observables
        self.reference = reference

    def evaluate(self, implementation: Implementation):
        """Create a variable from the given implementation."""
        if isinstance(implementation, datatypes.Variable):
            return self._evaluate_variable(implementation)
        if isinstance(implementation, functions.Function):
            return self._evaluate_function(implementation)
        if isinstance(implementation, Compound):
            return self._evaluate_compound(implementation)
        raise TypeError(f"Unknown implementation: {type(implementation)}")

    def _evaluate_variable(self, variable: datatypes.Variable):
        """Apply relevant updates (e.g., indices) to this variable."""
        target_axes = [
            axis for axis in variable.axes if self._need_interp(axis)
        ]
        if 'radius' in self.assumptions:
            target_axes.append('shell')
        if self._is_reference(variable) or not target_axes:
            return self._standard(variable)
        return self._interpolated(variable, target_axes)

    def _standard(self, variable: datatypes.Variable):
        """Produce a new variable by subscripting the given variable."""
        indices = tuple(self.indices[axis] for axis in variable.axes)
        return variable[indices]

    def _interpolated(
        self,
        original: datatypes.Variable,
        axes: typing.Iterable[str],
    ) -> datatypes.Variable:
        """Produce a new variable by interpolating the given variable."""
        indexable = list(set(original.axes) - set(axes))
        variable = self._interpolate(original, axes)
        indices = tuple(
            self.indices[axis]
            if axis in indexable else slice(None)
            for axis in variable.axes
        )
        return variable[indices]

    def _is_reference(self, variable: datatypes.Variable):
        """True if this is an axis reference variable.

        This method attempts to determine if the given variable is one of the
        axis or axis-like reference objects used for interpolation, so calling
        code can avoid accidentally interpolating over it. NB: Two variables
        with different units will always compare false by triggering a
        `quantities.ComparisonError`.
        """
        return any(alias in self.reference for alias in variable.name)

    def _need_interp(self, axis: str):
        """True if we need to interpolate over this axis."""
        index = self.indices[axis]
        if not isinstance(index, datatypes.Coordinates):
            return False
        reference = self.reference[axis]
        targets = np.array(index.values)
        refarr = np.array(reference)
        available = [self._in_array(target, refarr) for target in targets]
        return not np.all(available)

    def _in_array(self, target: float, array: np.ndarray):
        """True if the target value is in the given array."""
        if target < np.min(array) or target > np.max(array):
            return False
        return np.any([np.isclose(target, array, atol=0.0)])

    def _interpolate(
        self,
        variable: datatypes.Variable,
        axes: typing.Iterable[str],
    ) -> datatypes.Variable:
        """Interpolate the variable over certain axes."""
        array = None
        coordinates = {
            axis: {
                'targets': np.array(indices.values),
                'reference': self.reference[axis],
            }
            for axis, indices in self.indices.items()
            if axis in axes and isinstance(indices, datatypes.Coordinates)
        }
        if 'radius' in self.assumptions:
            radii = iterables.whole(self.assumptions['radius'])
            coordinates.update(
                radius={
                    'targets': np.array([float(radius) for radius in radii]),
                    'reference': self.reference['radius'],
                }
            )
        for coordinate, current in coordinates.items():
            array = self._interpolate_coordinate(
                variable,
                current['targets'],
                current['reference'],
                coordinate=coordinate,
                workspace=array,
            )
        return datatypes.Variable(
            array,
            variable.unit,
            variable.axes,
            name=variable.name,
        )

    def _interpolate_coordinate(
        self,
        variable: datatypes.Variable,
        targets: np.ndarray,
        reference: datatypes.Variable,
        coordinate: str=None,
        workspace: np.ndarray=None,
    ) -> np.ndarray:
        """Interpolate a variable array based on a known coordinate."""
        array = np.array(variable) if workspace is None else workspace
        indices = (variable.axes.index(d) for d in reference.axes)
        dst, src = zip(*enumerate(indices))
        reordered = np.moveaxis(array, src, dst)
        interpolated = interpolation.apply(
            reordered,
            np.array(reference),
            targets,
            coordinate=coordinate,
        )
        return np.moveaxis(interpolated, dst, src)

    def _evaluate_function(self, function: functions.Function):
        """Gather dependencies and call this function."""
        deps = {p: self._get_observable(p) for p in function.parameters}
        unit = MKS.get_unit(quantity=function.quantity)
        return function(deps, unit)

    def _evaluate_compound(self, implementation: Compound):
        """Combine variables and functions based on this expression."""
        expression = implementation.expression
        variables = [
            self._get_observable(term.base) for term in expression
        ]
        exponents = [term.exponent for term in expression]
        result = variables[0] ** exponents[0]
        for variable, exponent in zip(variables[1:], exponents[1:]):
            result *= variable ** exponent
        return result

    def _get_observable(self, key: str):
        """Get an observable dependency by keyword."""
        if key in self.assumptions:
            return self.assumptions[key]
        if key in self.observables:
            observable = self.observables[key]
            if isinstance(observable, datatypes.Variable):
                return self._evaluate_variable(observable)
            if isinstance(observable, functions.Function):
                return self._evaluate_function(observable)
        raise KeyError(f"Can't find observable {key!r}")


class Interface(base.Interface):
    """A concrete EPREM observing interface."""

    def __init__(
        self,
        implementation: Implementation,
        dataset: datasets.Dataset,
        dependencies: typing.Mapping[str, Dependency]=None,
    ) -> None:
        self.implementation = implementation
        self.axes = dataset.axes
        self.dependencies = aliased.Mapping(dependencies or {})
        self._result = None
        self._context = None
        self.indices = aliased.MutableMapping.fromkeys(
            self.axes,
            value=(),
        )
        self.assumptions = aliased.MutableMapping(
            {
                k: v for k, v in self.dependencies.items(aliased=True)
                if isinstance(v, parameters.Assumption)
            }
        )
        self.observables = aliased.MutableMapping(
            {
                k: v for k, v in self.dependencies.items(aliased=True)
                if isinstance(v, (datatypes.Variable, functions.Function))
            }
        )
        axes_ref = {k: v.reference for k, v in self.axes.items(aliased=True)}
        variables = dataset.variables
        rtp_ref = {
            (k, *variables.alias(k, include=True)): variables[k]
            for k in {'radius', 'theta', 'phi'}
        }
        self.reference = aliased.Mapping({**axes_ref, **rtp_ref})

    def update_indices(self, constraints: typing.Mapping):
        """Update the instance indices based on user constraints."""
        current = self.indices.copy()
        new = {k: v for k, v in constraints.items() if k in self.axes}
        current.update(new)
        updates = {k: self._update_index(k, v) for k, v in current.items()}
        updated = self.indices.copy()
        updated.update(updates)
        return aliased.Mapping(updated)

    def _update_index(self, key: str, indices):
        """Update a single indexing object based on user input."""
        if not isinstance(indices, datatypes.Indices):
            axis = self.axes[key]
            indices = axis(*iterables.whole(indices))
        if isinstance(indices, datatypes.Coordinates):
            unit = MKS.get_unit(unit=indices.unit)
            return indices.with_unit(unit)
        return indices

    def update_assumptions(self, constraints: typing.Mapping):
        """Update the observing assumptions based on user constraints."""
        current = self.assumptions.copy()
        new = {k: v for k, v in constraints.items() if k not in self.axes}
        current.update(new)
        updates = {k: self._update_assumption(v) for k, v in current.items()}
        updated = self.assumptions.copy()
        updated.update(updates)
        return aliased.Mapping(updated)

    def _update_assumption(self, scalar):
        """Update a single assumption from user input."""
        if isinstance(scalar, quantities.Scalar):
            unit = MKS.get_unit(unit=scalar.unit())
            return scalar.unit(unit)
        measured = quantities.measure(scalar)
        assumption = [self._update_assumption(v) for v in measured[:]]
        return assumption[0] if len(assumption) == 1 else assumption

    def apply(self, constraints: typing.Mapping):
        """Construct the target variable within the given constraints."""
        indices = self.update_indices(constraints)
        assumptions = self.update_assumptions(constraints)
        application = Application(
            indices,
            assumptions,
            self.observables,
            self.reference,
        )
        self._result = application.evaluate(self.implementation)
        self._context = {
            'indices': aliased.Mapping(
                {
                    (k, *indices.alias(k, include=True)): v
                    for k, v in indices.items()
                    if k in self._result.axes
                }
            ),
            'assumptions': aliased.Mapping(
                {
                    (k, *assumptions.alias(k, include=True)): v
                    for k, v in assumptions.items()
                    if k in list(self.assumptions) + ['radius']
                }
            ),
        }

    @property
    def result(self):
        return self._result

    @property
    def context(self):
        return self._context

    def __str__(self) -> str:
        """A simplified representation of this object."""
        pairs = [
            f"implementation={self.implementation}",
            f"axes={self.axes}",
        ]
        return ', '.join(pairs)


class Observables(iterables.MappingBase):
    """An aliased mapping of observable quantities from an EPREM simulation."""

    def __init__(
        self,
        dataset: datasets.Dataset,
        arguments: parameters.Arguments,
    ) -> None:
        self.variables = dataset.variables
        self.functions = functions.Functions(dataset, arguments)
        vkeys = self.variables.keys
        fkeys = self.functions.keys
        self.primary = tuple(vkeys())
        self.derived = tuple(fkeys())
        self.names = self.primary + self.derived
        super().__init__(self.names)
        akeys = tuple(vkeys(aliased=True)) + tuple(fkeys(aliased=True))
        self.aliases = aliased.KeyMap(akeys)
        self.dataset = dataset
        self.arguments = arguments
        self._cache = {}

    def __getitem__(self, key: str):
        """Create the requested observable, if possible."""
        interface = self._implement(key)
        if interface is None:
            raise KeyError(f"No observable corresponding to {key!r}") from None
        return base.Observable(interface, self.aliases.get(key, key))

    def _implement(self, key: str):
        """Create an interface to this observable, if possible."""
        if key in self._cache:
            cached = self._cache[key]
            return Interface(
                cached['implementation'],
                self.dataset,
                dependencies=cached['dependencies'],
            )
        implementation = self.get_implementation(key)
        if implementation is not None:
            dependencies = self.get_dependencies(key)
            self._cache[key] = {
                'implementation': implementation,
                'dependencies': dependencies,
            }
            return Interface(
                implementation,
                self.dataset,
                dependencies=dependencies,
            )

    def get_implementation(self, key: str):
        """Get the implementation of the target observable."""
        if key in self.variables:
            return self.variables[key]
        if key in self.functions:
            return self.functions[key]
        if '/' in key or '*' in key:
            expression = algebra.Expression(key)
            return self._build_compound_observable(expression)

    def _build_compound_observable(self, expression: algebra.Expression):
        """Build a compound observable from an algebraic expression."""
        unique = list(
            {
                axis
                for term in expression
                for axis in self.get_implementation(term.base).axes
            }
        )
        canonical = ('time', 'shell', 'species', 'energy', 'mu')
        indices = (unique.index(axis) for axis in canonical if axis in unique)
        axes = tuple(unique[i] for i in indices)
        name = str(expression)
        return Compound(expression=expression, axes=axes, name=name)

    _RT = typing.TypeVar('_RT', bound=dict)
    _RT = typing.Dict[aliased.MappingKey, Dependency]

    def get_dependencies(self, key: str) -> _RT:
        """Get the dependencies for the given observable, if possible."""
        if key in self.variables:
            k = self.variables.alias(key, include=True)
            v = self.variables[key]
            return {k: v}
        if key in self.functions:
            function = self.functions[key]
            items = [
                self._get_dependency(name)
                for name in function.dependencies
            ]
            return {i[0]: i[1] for i in items}
        if '/' in key or '*' in key:
            terms = algebra.Expression(key).terms
            out = {}
            for term in terms:
                name = term.base
                if name in self.functions:
                    k = self.functions.alias(name, include=True)
                    out[k] = self.functions[name]
                out.update(self.get_dependencies(name))
            return out

    def _get_dependency(self, name: str):
        """Get an aliased dependency based on type."""
        if name in self.variables:
            k = self.variables.alias(name, include=True)
            v = self.variables[name]
            return k, v
        if name in self.functions:
            k = self.functions.alias(name, include=True)
            v = self.functions[name]
            return k, v
        if name in self.arguments:
            k = self.arguments.alias(name, include=True)
            v = self.arguments[name]
            return k, v

