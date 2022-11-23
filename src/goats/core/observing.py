import abc
import collections
import collections.abc
import contextlib
import numbers
import operator as standard
import typing

from goats.core import iterables
from goats.core import metadata
from goats.core import observed
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


class Quantities(collections.abc.Mapping):
    """A collection of observing-related physical quantities."""

    def __init__(
        self,
        *observables: typing.Mapping[str],
        **others: typing.Mapping[str],
    ) -> None:
        self._observables = observables
        self._others = others

    def __len__(self) -> int:
        """Compute the number of available physical quantities."""
        return len(self.available)

    def __iter__(self) -> typing.Iterator[str]:
        """Iterate over available physical quantities."""
        return iter(self.available)

    @property
    def available(self):
        """The names of all available physical quantities."""
        others = tuple(k for m in self._others.values() for k in m)
        return self.observable + others

    @property
    def observable(self):
        """The names of observable physical quantities."""
        return tuple(k for m in self._observables for k in m)

    def __getitem__(self, __k: str):
        """Access physical quantities by key."""
        # Is it an observable quantity?
        for mapping in self._observables:
            if __k in mapping:
                return mapping[__k]
        # Is it an unobservable quantity?
        for mapping in self._others.values():
            if __k in mapping:
                return mapping[__k]
        # Is it a group of unobservable quantities?
        if __k in self._others:
            return self._others[__k]
        # We're out of options.
        raise KeyError(f"No known quantity for {__k!r}") from None

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
        for mapping in self._observables:
            if target in mapping:
                return getattr(mapping[target], __name, None)


class Interface(collections.abc.Collection):
    """Base class for observing-related interfaces."""

    def __init__(
        self,
        __quantities: Quantities,
        **constraints
    ) -> None:
        """Create a new instance.
        
        Parameters
        ----------
        __quantities
            An instance of `~observing.Dataset` or a subclass.

        constraints : mapping
            User-provided observing constraints.
        """
        self._quantities = __quantities
        self._constraints = constraints
        self._cache = {}
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

    def get_quantity(self, key: str):
        """Compute or retrieve a physical quantity."""
        return self._quantities.get(key)

    @abc.abstractmethod
    def get_result(self, key: str) -> Quantity:
        """Compute an observed quantity."""

    @abc.abstractmethod
    def get_context(self, key: str) -> typing.Mapping:
        """Define the observing context."""


class Implementation:
    """The implementation of an observable quantity."""

    def __init__(
        self,
        __type: typing.Type[Interface],
        name: str,
        dataset: Quantities,
    ) -> None:
        """Initialize this instance.

        Parameters
        ----------
        __type
            # TODO

        name : string
            The name of the quantity to observe.

        dataset
            # TODO
        """
        self._type = __type
        self._name = name
        self._dataset = dataset
        self._unit = None
        self._dimensions = None
        self._parameters = None

    def apply(self, **constraints):
        """Apply user constraints to this implementation."""
        interface = self._type(self.dataset, **constraints)
        context = interface.get_context(self.name)
        return observed.Quantity(
            interface.get_result(self.name),
            context['axes'],
            constants=context.get('constants'),
        )

    @property
    def dataset(self):
        """A copy of the underlying dataset."""
        return self._dataset

    @property
    def unit(self):
        """The metric unit of this observable quantity."""
        if self._unit is None:
            self._unit = self.dataset.get_unit(self.name)
        return self._unit

    @property
    def dimensions(self):
        """The array dimensions of this observable quantity."""
        if self._dimensions is None:
            self._dimensions = self.dataset.get_dimensions(self.name)
        return self._dimensions

    @property
    def parameters(self):
        """The physical parameters of this observable quantity."""
        if self._parameters is None:
            self._parameters = self.dataset.get_parameters(self.name)
        return self._parameters

    @property
    def name(self):
        """The name of the target observable quantity."""
        return self._name

