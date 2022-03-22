import typing

import numpy as np

from goats.core import aliased
from goats.core import datasets
from goats.core import datatypes
from goats.core import functions
from goats.core import iterables
from goats.core import observables
from goats.core import physical
from goats.core import quantities
from goats.eprem import parameters


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
        datatypes.Variable,
        quantities.Scalar,
        typing.Iterable[quantities.Scalar],
    )
    Argument = typing.Union[
        datatypes.Variable,
        quantities.Scalar,
        typing.Iterable[quantities.Scalar],
    ]

    def __call__(
        self,
        arguments: typing.Mapping[str, Argument],
        unit: typing.Union[str, quantities.Unit],
    ) -> datatypes.Variable:
        """Build a variable by calling the instance method."""
        arrays = []
        floats = []
        known = [
            argument for key, argument in arguments.items()
            if key in self.parameters
        ]
        for arg in known:
            if isinstance(arg, datatypes.Variable):
                arrays.append(np.array(arg))
            elif isinstance(arg, quantities.Scalar):
                floats.append(float(arg))
            elif (
                isinstance(arg, typing.Iterable)
                and all(isinstance(a, quantities.Scalar) for a in arg)
            ): floats.extend([float(a) for a in arg])
        data = self.method(*arrays, *floats)
        return datatypes.Variable(
            data,
            quantities.Unit(unit),
            self.axes,
            name=self.name,
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
        dataset: datasets.Dataset,
        arguments: parameters.Arguments,
    ) -> None:
        super().__init__(mapping=functions.METHODS)
        self.dataset = dataset
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
            quantity = observables.METADATA.get(key, {}).get('quantity', None)
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
        self._removed = list(method.metadata.get('removed', []))
        self._restored = list(method.metadata.get('restored', []))
        self._accumulated = []
        axes = self._gather_axes(method)
        self._axes_cache[key] = axes
        return axes

    def _gather_axes(self, target: functions.Method):
        """Recursively gather appropriate axes."""
        for parameter in target.parameters:
            if parameter in self.dataset.variables:
                axes = self.dataset.iter_axes(parameter)
                self._accumulated.extend(axes)
            elif method := self.get_method(parameter):
                self._removed.extend(method.metadata.get('removed', []))
                self._restored.extend(method.metadata.get('restored', []))
                self._accumulated.extend(self._gather_axes(method))
        unique = set(self._accumulated) - set(self._removed)
        return self.dataset.resolve_axes(unique)

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


