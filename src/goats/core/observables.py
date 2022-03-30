import typing

import numpy

from goats.core import aliased
from goats.core import datatypes
from goats.core import iterables
from goats.core import quantities


__all__ = [
    'ALIASES',
    'METADATA',
]


class Observation(iterables.ReprStrMixin):
    """The result of observing an observable object."""

    def __init__(
        self,
        data: datatypes.Variable,
        indices: typing.Mapping[str, datatypes.Indices],
        assumptions: typing.Mapping[str, datatypes.Assumption]=None,
    ) -> None:
        self._data = data
        self.name = data.name
        self._indices = indices
        self._assumptions = assumptions or {}
        self._axes = None
        self._parameters = None

    def __array__(self, *args, **kwargs) -> numpy.ndarray:
        """Support automatic conversion to a `numpy.ndarray`."""
        return numpy.array(self._data, *args, **kwargs)

    def __getitem__(self, item):
        """Get an assumption, an array axis, or array values."""
        if isinstance(item, str):
            if item in self._indices:
                return self._indices[item]
            if item in self._assumptions:
                return self._assumptions[item]
            raise KeyError(item) from None
        return self._data[item]

    @property
    def axes(self):
        """The indexable axes of this observation's array."""
        if self._axes is None:
            if isinstance(self._indices, aliased.Mapping):
                self._axes = self._indices.keys(aliased=True)
            else:
                self._axes = self._indices.keys()
        return self._axes

    @property
    def parameters(self):
        """The names of assumptions relevant to this observation."""
        if self._parameters is None:
            if isinstance(self._assumptions, aliased.Mapping):
                self._parameters = self._assumptions.keys(aliased=True)
            else:
                self._parameters = self._assumptions.keys()
        return self._parameters

    def unit(self, new: typing.Union[str, quantities.Unit]=None):
        """Get or set the unit of this observation's data values."""
        if not new:
            return self._data.unit
        self._data = self._data.convert_to(new)
        return self

    def __eq__(self, other) -> bool:
        """True if two instances have equivalent attributes."""
        if not isinstance(other, Observation):
            return NotImplemented
        if not self._equal_attrs(other):
            return False
        return super().__eq__(other)

    def _equal_attrs(self, other):
        """True if two instances have the same attributes."""
        return all(
            getattr(other, attr) == getattr(self, attr)
            for attr in {'indices', 'assumptions'}
        )

    def __str__(self) -> str:
        """A simplified representation of this object."""
        axes = [str(axis) for axis in self.axes]
        parameters = [str(parameter) for parameter in self.parameters]
        attrs = [
            f"'{self.name}'",
            f"unit='{self.unit()}'",
            f"axes={axes}",
            f"parameters={parameters}",
        ]
        return ', '.join(attrs)


_METADATA = {
    ('time', 't', 'times'): {
        'quantity': 'time',
    },
    ('shell', 'shells'): {
        'quantity': 'number',
    },
    (
        'mu', 'mus',
        'pitch angle', 'pitch-angle cosine',
        'pitch angles', 'pitch-angle cosines',
    ): {
        'quantity': 'ratio',
    },
    ('mass', 'm'): {
        'quantity': 'mass',
    },
    ('charge', 'q'): {
        'quantity': 'charge',
    },
    ('egrid', 'energy', 'energies', 'E'): {
        'quantity': 'energy',
    },
    ('vgrid', 'speed', 'v', 'vparticle'): {
        'quantity': 'velocity',
    },
    ('R', 'r', 'radius'): {
        'quantity': 'length',
    },
    ('T', 'theta'): {
        'quantity': 'plane angle',
    },
    ('P', 'phi'): {
        'quantity': 'plane angle',
    },
    ('Br', 'br'): {
        'quantity': 'magnetic field',
    },
    ('Bt', 'bt', 'Btheta', 'btheta'): {
        'quantity': 'magnetic field',
    },
    ('Bp', 'bp', 'Bphi', 'bphi'): {
        'quantity': 'magnetic field',
    },
    ('Vr', 'vr'): {
        'quantity': 'velocity',
    },
    ('Vt', 'vt', 'Vtheta', 'vtheta'): {
        'quantity': 'velocity',
    },
    ('Vp', 'vp', 'Vphi', 'vphi'): {
        'quantity': 'velocity',
    },
    ('Rho', 'rho'): {
        'quantity': 'number density',
    },
    ('Dist', 'dist', 'f'): {
        'quantity': 'particle distribution',
    },
    ('flux', 'Flux', 'J', 'J(E)', 'j', 'j(E)'): {
        'quantity': (
            'number / (area * solid_angle * time * energy / mass_number)'
        ),
    },
    ('x', 'X'): {
        'quantity': 'length',
    },
    ('y', 'Y'): {
        'quantity': 'length',
    },
    ('z', 'Z'): {
        'quantity': 'length',
    },
    ('b_mag', '|B|', 'B', 'bmag', 'b mag'): {
        'quantity': 'magnetic field',
    },
    ('v_mag', '|V|', 'V', 'vmag', 'v mag'): {
        'quantity': 'velocity',
    },
    ('bv_mag', 'bv', '|bv|', 'BV', '|BV|'): {
        'quantity': 'velocity * magnetic field',
    },
    ('v_para', 'vpara', 'Vpara'): {
        'quantity': 'velocity',
    },
    ('v_perp', 'vperp', 'Vperp'): {
        'quantity': 'velocity',
    },
    ('flow_angle', 'flow angle', 'angle'): {
        'quantity': 'plane angle',
    },
    ('div_v', 'divV', 'divv', 'div V', 'div v', 'div(V)', 'div(v)'): {
        'quantity': '1 / time',
    },
    ('density_ratio', 'density ratio' ,'n2/n1', 'n_2/n_1'): {
        'quantity': 'number',
    },
    ('rigidity', 'Rg', 'R_g'): {
        'quantity': 'momentum / charge',
    },
    ('mean_free_path', 'mean free path', 'mfp'): {
        'quantity': 'length',
    },
    ('acceleration_rate', 'acceleration rate'): {
        'quantity': '1 / time',
    },
    ('energy_density', 'energy density'): {
        'quantity': 'energy / volume',
    },
    ('average_energy', 'average energy'): {
        'quantity': 'energy',
    },
    ('isotropic_distribution', 'isotropic distribution', 'isodist', 'f'): {
        'removed axes': ['mu'],
        'quantity': 'particle distribution',
    },
    'fluence': {
        'removed axes': ['time'],
        'quantity': 'number / (area * solid_angle * energy / mass_number)',
    },
    ('integral_flux', 'integral flux'): {
        'removed axes': ['energy'],
        'quantity': 'number / (area * solid_angle * time)',
    },
}


ALIASES = aliased.KeyMap(_METADATA.keys())
METADATA = aliased.Mapping(_METADATA)

