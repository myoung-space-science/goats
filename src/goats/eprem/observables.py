from typing import *

import numpy as np

from goats.common import base
from goats.common import aliased
from goats.common import quantities
from goats.common import algebra
from goats.common import iterables
from goats.common import indexing
from goats.common import physical
from goats.eprem import functions
from goats.eprem import datasets
from goats.eprem import parameters
from goats.eprem import interpolation


ObservableType = TypeVar(
    'ObservableType',
    quantities.Variable,
    functions.Function,
)
ObservableType = Union[
    quantities.Variable,
    functions.Function,
]


Implementation = TypeVar(
    'Implementation',
    ObservableType,
    algebra.Expression,
)
Implementation = Union[
    ObservableType,
    algebra.Expression,
]


Dependency = TypeVar(
    'Dependency',
    ObservableType,
    quantities.Scalar,
)
Dependency = Union[
    ObservableType,
    quantities.Scalar,
]


class Compound(NamedTuple):
    """An algebraic combination of primary or derived observables."""

    expression: algebra.Expression
    axes: Tuple[str]
    name: str=None


class Application:
    """The object that handles evaluating all observables."""

    def __init__(
        self,
        indices: Mapping[str, indexing.Indices],
        assumptions: Mapping[str, quantities.Scalar],
        observables: Mapping[str, ObservableType],
        reference: Mapping[str, Union[quantities.Variable, Iterable[Any]]],
        system: quantities.MetricSystem,
    ) -> None:
        self.indices = indices
        self.assumptions = assumptions
        self.observables = observables
        self.reference = reference
        self.system = system

    def evaluate(self, implementation: Implementation):
        """Create a variable from the given implementation."""
        if isinstance(implementation, quantities.Variable):
            return self._evaluate_variable(implementation)
        if isinstance(implementation, functions.Function):
            return self._evaluate_function(implementation)
        if isinstance(implementation, Compound):
            return self._evaluate_compound(implementation)
        raise TypeError(f"Unknown implementation: {type(implementation)}")

    def _evaluate_variable(self, variable: quantities.Variable):
        """Apply relevant updates (e.g., indices) to this variable."""
        target_axes = [
            axis for axis in variable.axes if self._need_interp(axis)
        ]
        if 'radius' in self.assumptions:
            target_axes.append('shell')
        if self._is_reference(variable) or not target_axes:
            return self._standard(variable)
        return self._interpolated(variable, target_axes)

    def _standard(self, variable: quantities.Variable):
        """Produce a new variable by subscripting the given variable."""
        indices = tuple(self.indices[axis] for axis in variable.axes)
        return variable[indices]

    def _interpolated(
        self,
        original: quantities.Variable,
        axes: Iterable[str],
    ) -> quantities.Variable:
        """Produce a new variable by interpolating the given variable."""
        indexable = list(set(original.axes) - set(axes))
        variable = self._interpolate(original, axes)
        indices = tuple(
            self.indices[axis]
            if axis in indexable else slice(None)
            for axis in variable.axes
        )
        return variable[indices]

    def _is_reference(self, variable: quantities.Variable):
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
        if not isinstance(index, indexing.Coordinates):
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
        return np.any([np.isclose(target, array)])

    def _interpolate(
        self,
        variable: quantities.Variable,
        axes: Iterable[str],
    ) -> quantities.Variable:
        """Interpolate the variable over certain axes."""
        array = None
        coordinates = [
            {
                'targets': np.array(indices.values),
                'reference': self.reference[axis],
            }
            for axis, indices in self.indices.items()
            if axis in axes and isinstance(indices, indexing.Coordinates)
        ]
        for current in coordinates:
            array = self._interpolate_coordinate(
                variable,
                current['targets'],
                current['reference'],
                workspace=array,
            )
        if 'radius' in self.assumptions:
            radii = iterables.whole(self.assumptions['radius'])
            targets = np.array([float(radius) for radius in radii])
            array = self._interpolate_radius(
                variable,
                targets,
                self.reference['radius'],
                workspace=array,
            )
        return quantities.Variable(
            array,
            variable.unit(),
            variable.axes,
            name=variable.name,
        )

    def _interpolate_coordinate(
        self,
        variable: quantities.Variable,
        targets: np.ndarray,
        reference: quantities.Variable,
        workspace: np.ndarray=None,
    ) -> np.ndarray:
        """Interpolate a variable array based on a known coordinate."""
        array = np.array(variable) if workspace is None else workspace
        source = [variable.axes.index(d) for d in reference.axes]
        destination = list(range(len(source)))
        reordered = np.moveaxis(array, source, destination)
        interpolated = interpolation.standard(
            reordered,
            np.array(reference),
            targets,
        )
        return np.moveaxis(interpolated, destination, source)

    def _interpolate_radius(
        self,
        variable: quantities.Variable,
        targets: np.ndarray,
        reference: quantities.Variable,
        workspace: np.ndarray=None,
    ) -> np.ndarray:
        """Interpolate a variable to the given radius or radii."""
        array = np.array(variable) if workspace is None else workspace
        source = [variable.axes.index(d) for d in reference.axes]
        destination = list(range(len(source)))
        reordered = np.moveaxis(array, source, destination)
        interpolated = interpolation.radius(
            reordered,
            np.array(reference),
            targets,
        )
        return np.moveaxis(interpolated, destination, source)

    def _evaluate_function(self, function: functions.Function):
        """Gather dependencies and call this function."""
        deps = {p: self._get_observable(p) for p in function.parameters}
        unit = self.system.get_unit(quantity=function.quantity)
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
            if isinstance(observable, quantities.Variable):
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
        system: quantities.MetricSystem,
        dependencies: Mapping[str, Dependency]=None,
    ) -> None:
        self.implementation = implementation
        self.axes = dataset.axes
        self.system = system
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
                if isinstance(v, quantities.Scalar)
            }
        )
        self.observables = aliased.MutableMapping(
            {
                k: v for k, v in self.dependencies.items(aliased=True)
                if isinstance(v, (quantities.Variable, functions.Function))
            }
        )
        axes_ref = {k: v.reference for k, v in self.axes.items(aliased=True)}
        variables = dataset.variables
        rtp_ref = {
            tuple([k, *variables.alias(k, include=True)]): variables[k]
            for k in {'radius', 'theta', 'phi'}
        }
        self.reference = aliased.Mapping({**axes_ref, **rtp_ref})

    def update_indices(self, constraints: Mapping):
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
        if not isinstance(indices, indexing.Indices):
            axis = self.axes[key]
            indices = axis(*iterables.whole(indices))
        if isinstance(indices, indexing.Coordinates):
            unit = self.system.get_unit(unit=indices.unit)
            return indices.with_unit(unit)
        return indices

    def update_assumptions(self, constraints: Mapping):
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
        if not isinstance(scalar, quantities.Scalar):
            scalar = quantities.Scalar(*iterables.whole(scalar))
        unit = self.system.get_unit(unit=scalar.unit())
        return scalar.unit(unit)

    def apply(self, constraints: Mapping):
        """Construct the target variable within the given constraints."""
        indices = self.update_indices(constraints)
        assumptions = self.update_assumptions(constraints)
        application = Application(
            indices,
            assumptions,
            self.observables,
            self.reference,
            self.system,
        )
        self._result = application.evaluate(self.implementation)
        self._context = {
            'indices': aliased.Mapping({
                tuple([k, *indices.alias(k, include=True)]): v
                for k, v in indices.items()
                if k in self._result.axes
            }),
            'assumptions': aliased.Mapping({
                tuple([k, *assumptions.alias(k, include=True)]): v
                for k, v in assumptions.items()
                if k in self.assumptions
            }),
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
            f"system={self.system}",
        ]
        return ', '.join(pairs)


class Observables(iterables.MappingBase):
    """An aliased mapping of observable quantities from an EPREM simulation."""

    def __init__(
        self,
        dataset: datasets.Dataset,
        system: quantities.MetricSystem,
        arguments: parameters.Arguments,
    ) -> None:
        self.variables = dataset.variables
        constants = physical.Constants(system)
        self.functions = functions.Functions(dataset, arguments, constants)
        vkeys = self.variables.keys
        fkeys = self.functions.keys
        self.primary = tuple(vkeys())
        self.derived = tuple(fkeys())
        self.names = self.primary + self.derived
        super().__init__(self.names)
        akeys = tuple(vkeys(aliased=True)) + tuple(fkeys(aliased=True))
        self.aliases = aliased.AliasMap(akeys)
        self.dataset = dataset
        self.system = system
        self.arguments = arguments
        self.constants = constants
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
                self.system,
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
                self.system,
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

    _RT = TypeVar('_RT', bound=dict)
    _RT = Dict[aliased.MappingKey, Dependency]

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
        if name in self.constants:
            k = name
            v = self.constants[name]
            return k, v
        if name in self.arguments:
            k = self.arguments.alias(name, include=True)
            v = self.arguments[name]
            return k, v

