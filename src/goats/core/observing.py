import abc
import collections
import collections.abc
import contextlib
import numbers
import operator as standard
import typing

import numpy

from goats.core import aliased
from goats.core import axis
from goats.core import computed
from goats.core import constant
from goats.core import interpolation
from goats.core import iterables
from goats.core import metadata
from goats.core import metric
from goats.core import physical
from goats.core import reference
from goats.core import symbolic
from goats.core import variable


T = typing.TypeVar('T')


class Parameters(collections.abc.Sequence, iterables.ReprStrMixin):
    """The parameters of an `~observing.Quantity`."""

    def __init__(self, *names: str) -> None:
        self._names = self._init(*names)

    def _init(self, *args):
        names = iterables.unique(*iterables.unwrap(args, wrap=tuple))
        if not names or all(isinstance(name, str) for name in names):
            return names
        raise TypeError(
            f"Can't initialize instance of {type(self)}"
            f" with {names!r}"
        )

    __abs__ = metadata.identity(abs)
    """Called for abs(self)."""
    __pos__ = metadata.identity(standard.pos)
    """Called for +self."""
    __neg__ = metadata.identity(standard.neg)
    """Called for -self."""

    __add__ = metadata.identity(standard.add)
    """Called for self + other."""
    __sub__ = metadata.identity(standard.sub)
    """Called for self - other."""

    def merge(a, *others):
        """Return the unique axis names in order."""
        names = list(a._names)
        for b in others:
            if isinstance(b, Parameters):
                names.extend(b._names)
        return Parameters(*set(names))

    __mul__ = merge
    """Called for self * other."""
    __rmul__ = merge
    """Called for other * self."""
    __truediv__ = merge
    """Called for self / other."""

    def __pow__(self, other):
        """Called for self ** other."""
        if isinstance(other, numbers.Real):
            return self
        return NotImplemented

    def __eq__(self, other):
        """True if self and other represent the same axes."""
        return (
            isinstance(other, Parameters) and other._names == self._names
            or (
                isinstance(other, str)
                and len(self) == 1
                and other == self._names[0]
            )
            or (
                isinstance(other, typing.Iterable)
                and len(other) == len(self)
                and all(i in self for i in other)
            )
        )

    def __hash__(self):
        """Support use as a mapping key."""
        return hash(self._names)

    def __len__(self) -> int:
        """Called for len(self)."""
        return len(self._names)

    def __getitem__(self, __i: typing.SupportsIndex):
        """Called for index-based access."""
        return self._names[__i]

    def __str__(self) -> str:
        return f"[{', '.join(repr(name) for name in self._names)}]"


class Quantity(variable.Quantity):
    """A quantity with one or more name(s), a unit, and axes."""

    _parameters: Parameters=None

    def __init__(self, __data, **meta) -> None:
        super().__init__(__data, **meta)
        parsed = self.parse_attrs(__data, meta, parameters=())
        self._parameters = Parameters(parsed['parameters'])
        self.meta.register('parameters')
        self.display.register('parameters')
        self.display['__str__'].append("parameters={parameters}")
        self.display['__repr__'].append("parameters={parameters}")

    def parse_attrs(self, this, meta: dict, **targets):
        if (
            isinstance(this, variable.Quantity)
            and not isinstance(this, Quantity)
        ): # barf
            meta.update({k: getattr(this, k) for k in ('unit', 'name', 'axes')})
            this = this.data
        return super().parse_attrs(this, meta, **targets)

    @property
    def parameters(self):
        """The optional parameters that define this observing quantity."""
        return Parameters(self._parameters)


class Dataset(collections.abc.Mapping):
    """A collection of observing-related physical quantities."""

    def __init__(self, *mappings: typing.Mapping[str]) -> None:
        self._mappings = mappings

    def __len__(self) -> int:
        """Compute the number of items in all mappings."""
        return len(self._mappings)

    def __iter__(self) -> typing.Iterator[str]:
        return iter(self._mappings)

    def __getitem__(self, __key: str):
        return self._mappings[__key]

    def get_unit(self, key: str):
        """Compute or retrieve the metric unit of a physical quantity."""
        if found := self._lookup('unit', key):
            return found
        s = str(key)
        expression = symbolic.Expression(reference.NAMES.get(s, s))
        term = expression[0]
        this = self.get_unit(term.base) ** term.exponent
        if len(expression) > 1:
            for term in expression[1:]:
                this *= self.get_unit(term.base) ** term.exponent
        return metadata.Unit(this)

    def get_dimensions(self, key: str):
        """Compute or retrieve the array dimensions of a physical quantity."""
        if found := self._lookup('dimensions', key):
            return found
        s = str(key)
        expression = symbolic.Expression(reference.NAMES.get(s, s))
        term = expression[0]
        this = self.get_dimensions(term.base)
        if len(expression) > 1:
            for term in expression[1:]:
                this *= self.get_dimensions(term.base)
        return metadata.Axes(this)

    def get_parameters(self, key: str):
        """Compute or retrieve the parameters of a physical quantity."""
        if found := self._lookup('parameters', key):
            return found
        s = str(key)
        expression = symbolic.Expression(reference.NAMES.get(s, s))
        term = expression[0]
        this = self.get_parameters(term.base)
        if len(expression) > 1:
            for term in expression[1:]:
                this *= self.get_parameters(term.base)
        return Parameters(this)

    def _lookup(self, __name: str, target: str):
        """Search for an attribute among available quantities."""
        for mapping in self._mappings:
            if target in mapping:
                return getattr(mapping[target], __name, None)


class Interface(collections.abc.Collection):
    """Base class for observing-related interfaces."""

    def __init__(
        self,
        __quantities: Dataset,
        **constraints
    ) -> None:
        self._quantities = __quantities
        self._constraints = constraints
        self._observables = None

    def __contains__(self, __x: str) -> bool:
        """True if `__x` is an available quantity."""
        return __x in self.quantities

    def __len__(self) -> int:
        """Compute the number of available quantities."""
        return len(self.quantities)

    def __iter__(self) -> typing.Iterator[str]:
        """Iterate over names of available quantities."""
        return iter(self.quantities)

    @property
    def quantities(self):
        """The names of available physical quantities."""
        return self._quantities

    @property
    def observables(self):
        """The names of observable physical quantities."""
        return self._observables

    @abc.abstractmethod
    def get_quantity(self, key: str):
        """Compute or retrieve a physical quantity."""

    @abc.abstractmethod
    def get_observable(self, key: str):
        """Compute or retrieve an observable quantity."""


class Implementation(collections.abc.Mapping):
    """Base class for observation results.
    
    This class represents the observer-specific result of observing an
    observable quantity.
    """

    def __init__(
        self,
        __interface: Interface,
        name: str,
        constraints: typing.Mapping,
    ) -> None:
        """
        Initialize this instance.

        Parameters
        ----------
        interface : `~Interface`
            The interface to all observing-related quantities.

        name : string
            The name of the quantity to observe.

        constraints : mapping
            User-provided observing constraints.
        """
        self._interface = __interface
        self._name = name
        self._contraints = constraints
        self._data = None
        self._metadata = None
        self._context = None

    def __len__(self) -> int:
        """Compute the number of metadata elements."""
        return len(self._metadata)

    def __iter__(self) -> typing.Iterator[str]:
        """Iterate over metadata parameters."""
        return iter(self._metadata)

    def __getitem__(self, __k: str):
        """Retrieve a metadata attribute."""
        return self._metadata[__k]

    @property
    def data(self) -> variable.Quantity:
        """The observed variable quantity."""
        if self._data is None:
            raise NotImplementedError
        return self._data

    @property
    def context(self) -> typing.Mapping:
        """The metadata attributes relevant to this observation."""
        if self._context is None:
            raise NotImplementedError
        return self._context

