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

    def interpolate(
        self,
        indices: typing.Mapping[str, index.Quantity],
    ) -> variable.Quantity:
        """Interpolated the variable quantity."""
        q = self._interpolate(indices)
        indexable = list(set(self.q.axes) - set(self.axes))
        idx = tuple(
            indices[axis]
            if axis in indexable else slice(None)
            for axis in q.axes
        )
        return q[idx]

    def _interpolate(
        self,
        indices: typing.Mapping[str, index.Quantity],
    ) -> variable.Quantity:
        """Internal interpolation logic."""
        coordinates = {
            axis: {
                'targets': numpy.array(idx.data),
                'reference': self.references[axis],
            }
            for axis, idx in indices.items()
            if axis in self.axes and idx.unit is not None
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


class _Context:
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
        """Create a variable quantity from a function."""
        interface = functions.REGISTRY[name]
        method = interface.pop('method')
        caller = variable.Caller(method, **interface)
        deps = {p: self.get_quantity(p) for p in caller.parameters}
        quantity = observables.METADATA.get(name, {}).get('quantity', None)
        data = caller(**deps)
        return variable.Quantity(
            data,
            axes=self.get_axes(name),
            unit=self.variables.system.get_unit(quantity=quantity),
            name=name,
        )

    def get_quantity(self, name: str):
        """Retrieve the named quantity from available attributes."""
        if name in self.scalars:
            return self.scalars[name]
        if name in self.variables:
            return self.evaluate_variable(name)
        return self.evaluate_function(name)

    def get_axes(self, key: str):
        """Retrieve or compute the axes corresponding to `key`."""
        if key in self._axes_cache:
            return self._axes_cache[key]
        method = functions.REGISTRY[key]
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
            elif parameter in functions.REGISTRY:
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

    def _compute_index(self, key: str, this):
        """Compute a single indexing object from input values."""
        target = (
            self.axes[key].at(*iterables.whole(this))
            if not isinstance(this, index.Quantity)
            else this
        )
        if target.unit is not None:
            unit = self.variables.system.get_unit(unit=target.unit)
            return target.convert(unit)
        return target

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


class _Implementation(abc.ABC):
    """ABC for implementations of observable quantities."""

    def __init__(self, name: str, context: _Context) -> None:
        self.name = name
        self.context = context

    @abc.abstractmethod
    def apply(self, **constraints) -> observed.Quantity:
        """Create an observable quantity under the given constraints."""
        raise NotImplementedError


class Primary(_Implementation):
    """Implementation of a primary observable quantity."""

    def apply(self, **constraints) -> observed.Quantity:
        self.context.apply(**constraints)
        return observed.Quantity(
            self.context.evaluate_variable(self.name),
            indices=self.context.indices,
            scalars=self.context.scalars,
        )


class Derived(_Implementation):
    """Implementation of a derived observable quantity."""

    def apply(self, **constraints) -> observed.Quantity:
        self.context.apply(**constraints)
        return observed.Quantity(
            self.context.evaluate_function(self.name),
            indices=self.context.indices,
            scalars=self.context.scalars,
        )


class Composed(_Implementation):
    """Implementation of a composed observable quantity."""

    def apply(self, **constraints) -> observed.Quantity:
        this = algebraic.Expression(self.name)


class Context:
    """A constrainable observing context."""

    def __init__(
        self,
        axes: axis.Interface,
        variables: variable.Interface,
        arguments: parameters.Arguments,
    ) -> None:
        self.axes = axes
        self.variables = variables
        self.arguments = arguments
        self._references is None
        self._indices = aliased.MutableMapping.fromkeys(self.axes, value=())
        self._scalars = aliased.MutableMapping(self.arguments)
        self._cache = {}

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

    def get_scalars(self, **user):
        """Gather the relevant scalar assumptions."""
        updates = {
            k: self._get_assumption(v)
            for k, v in user.items()
            if k not in self.axes
        }
        self._scalars.update(updates)
        return self._scalars

    def get_indices(self, **user):
        """Create the relevant observing indices."""
        updates = {
            k: self._compute_index(k, v)
            for k, v in user.items()
            if k in self.axes
        }
        self._indices.update(updates)
        return self._indices

    def _compute_index(self, key: str, this):
        """Compute a single indexing object from input values."""
        target = (
            self.axes[key].at(*iterables.whole(this))
            if not isinstance(this, index.Quantity)
            else this
        )
        if target.unit is not None:
            unit = self.variables.system.get_unit(unit=target.unit)
            return target.convert(unit)
        return target

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

    def get_axes(self, key: str):
        """Retrieve or compute the axes corresponding to `key`."""
        if key in self._cache:
            return self._cache[key]
        method = functions.REGISTRY[key]
        self._removed = self._get_metadata(method, 'removed')
        self._added = self._get_metadata(method, 'added')
        self._accumulated = []
        axes = self._gather_axes(method)
        self._cache[key] = axes
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


class Computer:
    """"""


class Application:
    """The result of applying user values to an observing implementation."""

    def __init__(
        self,
        context: Context,
        **user
    ) -> None:
        self.user = user
        self.indices = context.get_indices(**self.user)
        self.scalars = context.get_scalars(**self.user)
        self.axes = context.axes
        self.variables = context.variables

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
        """Create a variable quantity from a function."""
        interface = functions.REGISTRY[name]
        method = interface.pop('method')
        caller = variable.Caller(method, **interface)
        deps = {p: self.get_quantity(p) for p in caller.parameters}
        quantity = observables.METADATA.get(name, {}).get('quantity', None)
        data = caller(**deps)
        return variable.Quantity(
            data,
            axes=self.get_axes(name),
            unit=self.variables.system.get_unit(quantity=quantity),
            name=name,
        )

    def get_quantity(self, name: str):
        """Retrieve the named quantity from available attributes."""
        if name in self.scalars:
            return self.scalars[name]
        if name in self.variables:
            return self.evaluate_variable(name)
        return self.evaluate_function(name)

    def get_axes(self, key: str):
        """Retrieve or compute the axes corresponding to `key`."""
        if key in self._axes_cache:
            return self._axes_cache[key]
        method = functions.REGISTRY[key]
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


class Implementation:
    """"""

    def __init__(
        self,
        name: str,
        context: Context,
    ) -> None:
        self.name = name
        self.context = context

    def apply(self, **constraints):
        """"""
        application = Application(self.context, **constraints)
        data = self.interface.implement(self.name, indices, scalars)
        return variable.Quantity(
            data,
            axes=self.context.get_axes(self.name),
            unit=self.variables.system.get_unit(quantity=quantity),
            name=name,
        )


class Interface:
    """Interface to all EPREM observable quantities."""

    def __init__(
        self,
        axes: axis.Interface,
        variables: variable.Interface,
        arguments: parameters.Arguments,
    ) -> None:
        self.context = Context(axes, variables, arguments)
        # These could become input parameters that provide the caller with some
        # flexibility about which variables and functions are available to a
        # given observer. For example:
        # - [primary] The EPREM dataset contains scalar quantities, such as
        #   `preEruption`, that should not be formally observable.
        # - [derived] If a dataset doesn't contain flux or the particle
        #   distribution, we can't expect to compute integral flux.
        self.primary = list(variables)
        """The names of observable quantities in the dataset."""
        self.derived = list(functions.METHODS)
        """The names of observable quantities computed from variables."""

    def implement(self, name: str):
        """Create the implementation of an observable quantity."""
        return Implementation(name, self.context)
        # if name in self.primary:
        #     return Primary(name, self.context)
        # if name in self.derived:
        #     return Derived(name, self.context)
        # if '/' in name or '*' in name:
        #     return Composed(name, self.context)

