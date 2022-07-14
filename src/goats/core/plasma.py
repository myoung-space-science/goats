from goats.core import physical
from goats.core import iterables
from goats.core import fundamental


class Species(iterables.ReprStrMixin):
    """A single species in a plasma."""

    def __init__(
        self,
        symbol: str=None,
        mass: float=None,
        charge: float=None,
    ) -> None:
        self._symbol = symbol
        self._mass = mass
        self._charge = charge
        if self._symbol is None and self._mass is None and self._charge is None:
            raise ValueError("Element is undefined")

    @property
    def symbol(self) -> str:
        """The elemental symbol of this species."""
        if self._symbol is None:
            s = fundamental.elements([self._mass], [self._charge])
            self._symbol = s[0]
        return self._symbol

    @property
    def mass(self) -> physical.Scalar:
        """The mass of this species."""
        if self._mass is None:
            base = self._symbol.rstrip('+-')
            element = fundamental.ELEMENTS.find(base, unique=True)
            value = element['mass']
            unit = 'nucleon'
            aliases = ['m', 'mass']
            self._mass = physical.Scalar(value, unit=unit, name=aliases)
        return self._mass

    @property
    def m(self):
        """Alias for mass."""
        return self.mass

    @property
    def charge(self) -> physical.Scalar:
        """The charge of this species."""
        if self._charge is None:
            base = self._symbol.rstrip('+-')
            sign = self._symbol.lstrip(base)
            value = sum(float(f"{s}1.0") for s in sign)
            unit = 'e'
            aliases = ['q', 'charge']
            self._charge = physical.Scalar(value, unit=unit, name=aliases)
        return self._charge

    @property
    def q(self):
        """Alias for charge."""
        return self.charge

    def __str__(self) -> str:
        """A simplified representation of this object."""
        return f"{self.symbol!r}, mass={self.m}, charge={self.q}"


