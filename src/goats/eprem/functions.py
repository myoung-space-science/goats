import typing

import numpy as np

from goats.core import aliased
from goats.core import variable
from goats.core import physical
from goats.core import functions
from goats.core import iterables
from goats.core import reference
from goats.core import measurable
from goats.core import metric
from goats.eprem import runtime


class Function(iterables.ReprStrMixin):
    """A function from variables and scalars to a single variable."""

    def __init__(
        self,
        method: functions.Method,
        quantity: str,
        axes: typing.Tuple[str],
        dependencies: typing.Iterable[str]=None,
        name: str=None,
    ) -> None:
        self.method = method
        self.quantity = quantity
        self.axes = axes
        self.parameters = tuple(self.method.parameters)
        self.dependencies = tuple(dependencies or ())
        self.name = name or '<anonymous>'

    Argument = typing.TypeVar(
        'Argument',
        variable.Quantity,
        measurable.Scalar,
        typing.Iterable[measurable.Scalar],
    )
    Argument = typing.Union[
        variable.Quantity,
        measurable.Scalar,
        typing.Iterable[measurable.Scalar],
    ]

    def __call__(
        self,
        arguments: typing.Mapping[str, Argument],
        unit: typing.Union[str, metric.Unit],
    ) -> variable.Quantity:
        """Build a variable by calling the instance method."""
        arrays = []
        floats = []
        known = [
            argument for key, argument in arguments.items()
            if key in self.parameters
        ]
        for arg in known:
            if isinstance(arg, physical.Array):
                arrays.append(np.array(arg))
            elif isinstance(arg, physical.Scalar):
                floats.append(float(arg))
            elif (
                isinstance(arg, typing.Iterable)
                and all(isinstance(a, physical.Scalar) for a in arg)
            ): floats.extend([float(a) for a in arg])
        data = self.method(*arrays, *floats)
        return variable.Quantity(
            data,
            unit=metric.Unit(unit),
            name=self.name,
            axes=self.axes,
        )

    def __str__(self) -> str:
        """A simplified representation of this object."""
        attrs = [
            f"method={self.method}",
            f"quantity='{self.quantity}'",
            f"axes={self.axes}",
        ]
        return ', '.join(attrs)


class Functions(aliased.Mapping):
    """Functions from variables and scalars to a single variable."""

    def __init__(
        self,
        data: variable.Interface,
        arguments: runtime.Arguments,
    ) -> None:
        super().__init__(mapping=functions.METHODS)
        self.dataset = data
        self._primary = (
            *tuple(self.dataset.variables.keys()),
            *tuple(arguments.keys()),
        )
        self._axes_cache = {}
        self._dependencies_cache = {}

    def __getitem__(self, key: str):
        """Construct the requested function object, if possible"""
        if method := self.get_method(key):
            axes = self.get_axes(key)
            quantity = reference.METADATA.get(key, {}).get('quantity', None)
            dependencies = self.get_dependencies(key)
            return Function(
                method,
                quantity,
                axes,
                dependencies=dependencies,
                name=key,
            )
        raise KeyError(f"No function corresponding to {key!r}")

    def get_method(self, key: str) -> functions.Method:
        """Attempt to retrieve a method by name based on `key`."""
        try:
            method = super().__getitem__(key)
        except KeyError:
            method = None
        return method

    def get_axes(self, key: str):
        """Retrieve or compute the axes corresponding to `key`."""
        if key in self._axes_cache:
            return self._axes_cache[key]
        method = self.get_method(key)
        self._removed = self._get_metadata(method, 'removed')
        self._added = self._get_metadata(method, 'added')
        self._accumulated = []
        axes = self._gather_axes(method)
        self._axes_cache[key] = axes
        return axes

    def _gather_axes(self, target: functions.Method):
        """Recursively gather appropriate axes."""
        for parameter in target.parameters:
            if parameter in self.dataset.variables:
                axes = self.dataset.variables[parameter].axes
                self._accumulated.extend(axes)
            elif method := self.get_method(parameter):
                self._removed.extend(self._get_metadata(method, 'removed'))
                self._added.extend(self._get_metadata(method, 'added'))
                self._accumulated.extend(self._gather_axes(method))
        unique = set(self._accumulated) - set(self._removed) | set(self._added)
        return self.dataset.resolve_axes(unique, mode='append')

    def _get_metadata(self, method: functions.Method, key: str) -> list:
        """Helper for accessing a method's metadata dictionary."""
        if key not in method.metadata:
            return [] # Don't go through the trouble if it's not there.
        value = method.metadata[key]
        return list(iterables.whole(value))

    def get_dependencies(self, key: str):
        """Compute the names of all dependencies of `key`."""
        if key in self._dependencies_cache:
            return self._dependencies_cache[key]
        try:
            target = self.get_method(key)
            p = self._gather_dependencies(target)
        except KeyError:
            return set()
        else:
            self._dependencies_cache[key] = p
            return p

    def _gather_dependencies(self, target: functions.Method):
        """Recursively gather the names of the target method's dependencies."""
        resolved = []
        for parameter in target.parameters:
            if parameter in self._primary:
                resolved.append(parameter)
            elif parameter in self:
                resolved.append(parameter)
                method = self.get_method(parameter)
                resolved.extend(self._gather_dependencies(method))
        return set(resolved)


