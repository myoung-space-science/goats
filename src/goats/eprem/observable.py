import abc
import typing

import numpy

from goats.core import algebraic
from goats.core import aliased
from goats.core import axis
from goats.core import functions
from goats.core import index
from goats.core import iterables
from goats.core import measurable
from goats.core import observed
from goats.core import observables
from goats.core import parameter
from goats.core import physical
from goats.core import variable
from goats.eprem import parameters
from goats.eprem import interpolation


class Interpolator:
    """An object that performs interpolation of variable quantities."""

    def __init__(
        self,
        __q: variable.Quantity,
        references: typing.Mapping[str, variable.Quantity],
        indices: typing.Mapping[str, index.Quantity],
        assumptions: typing.Mapping[str, parameter.Assumption],
    ) -> None:
        self.q = __q
        self.references = references
        self.indices = indices
        self.assumptions = assumptions
        self._coordinates = None
        self._axes = None

    @property
    def coordinates(self):
        """The measurable indices."""
        if self._coordinates is None:
            self._coordinates = {
                k: v for k, v in self.indices.items() if v.unit is not None
            }
        return self._coordinates

    @property
    def axes(self):
        """The axes over which to interpolate, if any."""
        if self._axes is None:
            coordinates = {
                a: self.coordinates[a].data
                for a in self.q.axes if a in self.coordinates
            }
            references = [
                self.references[a]
                for a in self.q.axes if a in self.coordinates
            ]
            self._axes = [
                a for (a, c), r in zip(coordinates.items(), references)
                if not numpy.all([target in r for target in c])
            ]
            if 'radius' in self.assumptions:
                self._axes.append('radius')
        return self._axes

    @property
    def required(self):
        """True if this variable quantity requires interpolation."""
        return (
            not any(alias in self.references for alias in self.q.name)
            and bool(self.axes)
        )

    @property
    def result(self) -> variable.Quantity:
        """The interpolated variable quantity."""
        q = self._interpolated
        indexable = list(set(self.q.axes) - set(self.axes))
        indices = tuple(
            self.indices[axis]
            if axis in indexable else slice(None)
            for axis in q.axes
        )
        return q[indices]

    @property
    def _interpolated(self) -> variable.Quantity:
        """Internal interpolation logic."""
        coordinates = {
            axis: {
                'targets': numpy.array(indices.data),
                'reference': self.references[axis],
            }
            for axis, indices in self.indices.items()
            if axis in self.axes and indices.unit is not None
        }
        if 'radius' in self.assumptions:
            radii = iterables.whole(self.assumptions['radius'])
            coordinates.update(
                radius={
                    'targets': numpy.array([float(radius) for radius in radii]),
                    'reference': self.references['radius'],
                }
            )
        array = None
        for coordinate, current in coordinates.items():
            array = self._interpolate_coordinate(
                current['targets'],
                current['reference'],
                coordinate=coordinate,
                workspace=array,
            )
        meta = {k: getattr(self.q, k, None) for k in {'unit', 'name', 'axes'}}
        return variable.Quantity(array, **meta)

    def _interpolate_coordinate(
        self,
        targets: numpy.ndarray,
        reference: variable.Quantity,
        coordinate: str=None,
        workspace: numpy.ndarray=None,
    ) -> numpy.ndarray:
        """Interpolate a variable array based on a known coordinate."""
        array = numpy.array(self.q) if workspace is None else workspace
        indices = (self.q.axes.index(d) for d in reference.axes)
        dst, src = zip(*enumerate(indices))
        reordered = numpy.moveaxis(array, src, dst)
        interpolated = interpolation.apply(
            reordered,
            numpy.array(reference),
            targets,
            coordinate=coordinate,
        )
        return numpy.moveaxis(interpolated, dst, src)


# Unused
class Application:
    """Constraints applied to an observing context."""

    def __init__(
        self,
        axes: axis.Interface,
        variables: variable.Interface,
        **constraints
    ) -> None:
        self.axes = axes
        self.variables = variables
        self.constraints = constraints
        self._indices = None
        self._assumptions = None

    @property
    def indices(self):
        """The relevant axis indices."""
        if self._indices is None:
            self._indices = {
                k: self._compute_index(k, v)
                for k, v in self.constraints.items()
                if k in self.axes
            }
        return self._indices

    @property
    def assumptions(self):
        """The relevant physical assumptions."""
        if self._assumptions is None:
            self._assumptions = {
                k: self._get_assumption(v)
                for k, v in self.constraints.items()
                if k not in self.axes
            }
        return self._assumptions

    def _compute_index(self, key: str, indices):
        """Compute a single indexing object."""
        if not isinstance(indices, index.Quantity):
            axis = self.axes[key]
            indices = axis(*iterables.whole(indices))
        if indices.unit is not None:
            unit = self.variables.system.get_unit(unit=indices.unit)
            return indices.convert(unit)
        return indices

    def _get_assumption(self, this):
        """Get a single assumption from user input."""
        scalar = self._force_scalar(this)
        unit = self.variables.system.get_unit(unit=scalar.unit)
        return scalar.convert(unit)

    def _force_scalar(self, this) -> measurable.Scalar:
        """Make sure `this` is a `~measurable.Scalar`."""
        if isinstance(this, measurable.Scalar):
            return this
        if isinstance(this, parameter.Assumption):
            return this[0]
        if isinstance(this, measurable.Measurement):
            return physical.Scalar(this.values[0], unit=this.unit)
        measured = measurable.measure(this)
        if len(measured) > 1:
            raise ValueError("Can't use a multi-valued assumption") from None
        return self._force_scalar(measured)



class Context:
    """A general observing context."""

    def __init__(
        self,
        axes: axis.Interface,
        variables: variable.Interface,
        assumptions: typing.Mapping[str, parameter.Assumption],
    ) -> None:
        self.axes = axes
        self.variables = variables
        self.assumptions = assumptions
        self._indices = None
        self._scalars = None
        self._references = None
        self._user = {}
        self._axes_cache = {}
        self._dependencies_cache = {}

    def apply(self, **constraints):
        """Apply the given constraints to this observing context."""
        self._user = constraints

    def evaluate_variable(self, name: str):
        """Apply user constraints to a variable quantity."""
        q = self.variables[name] # use variable.Interface KeyError
        interpolator = Interpolator(
            q,
            self.references,
            self.indices,
            self.scalars,
        )
        if interpolator.required:
            return interpolator.result
        return q[tuple(self.indices)]

    def evaluate_function(self, name: str):
        """"""
        interface = functions.REGISTRY[name]
        method = interface.pop('method')
        caller = variable.Caller(method, **interface)
        deps = {p: self.get_attribute(p) for p in caller.parameters}
        data = caller(**deps)
        return variable.Quantity(
            data,
            axes=self.get_axes(name),
            unit=self.get_unit(name),
            name=name,
        )

    def get_attribute(self, name: str):
        """"""
        if name in self.scalars:
            return self.scalars[name]
        if name in self.variables:
            return self.evaluate_variable(name)
        return self.evaluate_function(name)

    def get_unit(self, key: str):
        """Determine the metric unit corresponding to `key`."""
        quantity = observables.METADATA.get(key, {}).get('quantity', None)
        return self.variables.system.get_unit(quantity=quantity)

    def get_axes(self, key: str):
        """Retrieve or compute the axes corresponding to `key`."""
        if key in self._axes_cache:
            return self._axes_cache[key]
        method = functions.REGISTRY[self.name]
        self._removed = self._get_metadata(method, 'removed')
        self._added = self._get_metadata(method, 'added')
        self._accumulated = []
        axes = self._gather_axes(method)
        self._axes_cache[key] = axes
        return axes

    def _gather_axes(self, target: variable.Caller):
        """Recursively gather appropriate axes."""
        for parameter in target.parameters:
            if parameter in self.variables:
                axes = self.variables[parameter].axes
                self._accumulated.extend(axes)
            elif method := functions.REGISTRY[parameter]:
                self._removed.extend(self._get_metadata(method, 'removed'))
                self._added.extend(self._get_metadata(method, 'added'))
                self._accumulated.extend(self._gather_axes(method))
        unique = set(self._accumulated) - set(self._removed) | set(self._added)
        return self.axes.resolve(unique, mode='append')

    def _get_metadata(self, method: variable.Caller, key: str) -> list:
        """Helper for accessing a method's metadata dictionary."""
        if key not in method.meta:
            return [] # Don't go through the trouble if it's not there.
        value = method.meta[key]
        return list(iterables.whole(value))

    def get_dependencies(self, key: str):
        """Compute the names of all dependencies of `key`."""
        if key in self._dependencies_cache:
            return self._dependencies_cache[key]
        try:
            target = functions.REGISTRY[parameter]
            p = self._gather_dependencies(target)
        except KeyError:
            return set()
        else:
            self._dependencies_cache[key] = p
            return p

    def _gather_dependencies(self, target: variable.Caller):
        """Recursively gather the names of the target method's dependencies."""
        resolved = []
        for parameter in target.parameters:
            if parameter in self.variables:
                resolved.append(parameter)
            elif parameter in self:
                resolved.append(parameter)
                method = functions.REGISTRY[parameter]
                resolved.extend(self._gather_dependencies(method))
        return set(resolved)

    @property
    def indices(self):
        """The relevant axis indices."""
        if self._indices is None:
            self._indices = aliased.MutableMapping.fromkeys(
                self.axes,
                value=(),
            )
        updates = {
            k: self._compute_index(k, v)
            for k, v in self._user.items()
            if k in self.axes
        }
        self._indices.update(updates)
        return self._indices

    @property
    def scalars(self):
        """The relevant scalar assumptions."""
        if self._scalars is None:
            self._scalars = aliased.MutableMapping(self.assumptions)
        updates = {
            k: self._get_assumption(v)
            for k, v in self._user.items()
            if k not in self.axes
        }
        self._scalars.update(updates)
        return self._scalars

    def _compute_index(self, key: str, indices):
        """Compute a single indexing object."""
        if not isinstance(indices, index.Quantity):
            axis = self.axes[key]
            indices = axis(*iterables.whole(indices))
        if indices.unit is not None:
            unit = self.variables.system.get_unit(unit=indices.unit)
            return indices.convert(unit)
        return indices

    def _get_assumption(self, this):
        """Get a single assumption from user input."""
        scalar = self._force_scalar(this)
        unit = self.get_unit(unit=scalar.unit)
        return scalar.convert(unit)

    def _force_scalar(self, this) -> measurable.Scalar:
        """Make sure `this` is a `~measurable.Scalar`."""
        if isinstance(this, measurable.Scalar):
            return this
        if isinstance(this, parameter.Assumption):
            return this[0]
        if isinstance(this, measurable.Measurement):
            return physical.Scalar(this.values[0], unit=this.unit)
        measured = measurable.measure(this)
        if len(measured) > 1:
            raise ValueError("Can't use a multi-valued assumption") from None
        return self._force_scalar(measured)

    @property
    def references(self):
        """Reference quantities for indexing and interpolation."""
        if self._references is None:
            axes = {k: v.reference for k, v in self.axes.items(aliased=True)}
            rtp = {
                (k, *self.variables.alias(k, include=True)): self.variables[k]
                for k in {'radius', 'theta', 'phi'}
            }
            self._references = aliased.Mapping({**axes, **rtp})
        return self._references

    # --Development graveyard--

    # def get_axes(self, name: str):
    #     """Get the named axis quantity, if available."""
    #     if found := self.axes.get(name):
    #         return found

    # def get_variable(self, name: str):
    #     """Get the named variable quantity, if available."""
    #     if found := self.variables.get(name):
    #         return found

    # def get_assumption(self, name: str):
    #     """Get a numerical assumption, if available."""
    #     if found := self.assumptions.get(name):
    #         return found

    # def get_indices(self, **constraints):
    #     """Compute array indices based on constraints."""
    #     axes = {k: v for k, v in constraints.items() if k in self.axes}
    #     return {k: self._compute_index(k, v) for k, v in axes.items()}

    # def subscript(self, __q: variable.Quantity, **constraints):
    #     """Extract the appropriate subset of a variable quantity."""
    #     indices = [
    #         self.get_indices(axis, **constraints)
    #         for axis in __q.axes
    #     ]
    #     return __q[tuple(indices)]


class Implementation(abc.ABC):
    """ABC for implementations of observable quantities."""

    def __init__(self, name: str, context: Context) -> None:
        self.name = name
        self.context = context

    # If everything ends up involving a call to `self.context`, this may be one
    # level of abstraction too many.

    @abc.abstractmethod
    def apply(self, **constraints) -> observed.Quantity:
        """Create an observable quantity under the given constraints."""
        raise NotImplementedError


class Primary(Implementation):
    """Implementation of a primary observable quantity."""

    def apply(self, **constraints) -> observed.Quantity:
        self.context.apply(**constraints)
        return observed.Quantity(
            self.context.evaluate_variable(self.name),
            indices=self.context.indices,
            scalars=self.context.scalars,
        )


class Derived(Implementation):
    """Implementation of a derived observable quantity."""

    # def __init__(self, name: str, context: Context) -> None:
    #     super().__init__(name, context)
    #     self._axes_cache = {}
    #     self._dependencies_cache = {}

    def apply(self, **constraints) -> observed.Quantity:
        self.context.apply(**constraints)
        return observed.Quantity(
            self.context.evaluate_function(self.name),
            indices=self.context.indices,
            scalars=self.context.scalars,
        )
        # interface = functions.REGISTRY[self.name]
        # method = interface.pop('method')
        # caller = variable.Caller(method, **interface)
        # this = variable.Computer(
        #     functions.REGISTRY[self.name],
        #     axes=self.get_axes(self.name),
        #     unit=self.get_unit(self.name),
        #     name=self.name
        # )

    # def get_axes(self, key: str):
    #     """Retrieve or compute the axes corresponding to `key`."""
    #     if key in self._axes_cache:
    #         return self._axes_cache[key]
    #     method = functions.REGISTRY[self.name]
    #     self._removed = self._get_metadata(method, 'removed')
    #     self._added = self._get_metadata(method, 'added')
    #     self._accumulated = []
    #     axes = self._gather_axes(method)
    #     self._axes_cache[key] = axes
    #     return axes

    # def get_unit(self, key: str):
    #     """Determine the metric unit corresponding to `key`."""
    #     quantity = observables.METADATA.get(key, {}).get('quantity', None)
    #     return self.context.variables.system.get_unit(quantity=quantity)

    # def _gather_axes(self, target: variable.Caller):
    #     """Recursively gather appropriate axes."""
    #     for parameter in target.parameters:
    #         if parameter in self.context.variables:
    #             axes = self.context.variables[parameter].axes
    #             self._accumulated.extend(axes)
    #         elif method := functions.REGISTRY[parameter]:
    #             self._removed.extend(self._get_metadata(method, 'removed'))
    #             self._added.extend(self._get_metadata(method, 'added'))
    #             self._accumulated.extend(self._gather_axes(method))
    #     unique = set(self._accumulated) - set(self._removed) | set(self._added)
    #     return self.context.axes.resolve(unique, mode='append')

    # def _get_metadata(self, method: variable.Caller, key: str) -> list:
    #     """Helper for accessing a method's metadata dictionary."""
    #     if key not in method.meta:
    #         return [] # Don't go through the trouble if it's not there.
    #     value = method.meta[key]
    #     return list(iterables.whole(value))

    # def get_dependencies(self, key: str):
    #     """Compute the names of all dependencies of `key`."""
    #     if key in self._dependencies_cache:
    #         return self._dependencies_cache[key]
    #     try:
    #         target = functions.REGISTRY[parameter]
    #         p = self._gather_dependencies(target)
    #     except KeyError:
    #         return set()
    #     else:
    #         self._dependencies_cache[key] = p
    #         return p

    # def _gather_dependencies(self, target: variable.Caller):
    #     """Recursively gather the names of the target method's dependencies."""
    #     resolved = []
    #     for parameter in target.parameters:
    #         if parameter in self.context.variables:
    #             resolved.append(parameter)
    #         elif parameter in self:
    #             resolved.append(parameter)
    #             method = functions.REGISTRY[parameter]
    #             resolved.extend(self._gather_dependencies(method))
    #     return set(resolved)


class Composed(Implementation):
    """Implementation of a composed observable quantity."""

    def apply(self, **constraints) -> observed.Quantity:
        this = algebraic.Expression(self.name)


class Interface:
    """Interface to all EPREM observable quantities."""

    def __init__(
        self,
        axes: axis.Interface,
        variables: variable.Interface,
        arguments: parameters.Arguments,
    ) -> None:
        assumptions = {
            k: v for k, v in arguments.items()
            if isinstance(v, parameter.Assumption)
        } # -> aliased mapping
        self.context = Context(axes, variables, assumptions)
        # These could become input parameters that provide the caller with some
        # flexibility about which variables and functions are available to a
        # given observer. For example:
        # - [primary] The EPREM dataset contains scalar quantities, such as
        #   `preEruption`, that should not be formally observable.
        # - [derived] If a dataset doesn't contain flux or the particle
        #   distribution, we can't expect to compute integral flux.
        self.primary = list(variables)
        self.derived = list(functions.METHODS)

    def implement(self, name: str):
        """Create the implementation of an observable quantity."""
        if name in self.primary:
            return Primary(name, self.context)
        if name in self.derived:
            return Derived(name, self.context)
        if '/' in name or '*' in name:
            return Composed(name, self.context)

