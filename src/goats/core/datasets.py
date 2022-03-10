"""Tools for managing datasets."""

import abc
import collections.abc
import typing

import netCDF4
import numpy.typing

from goats.core import aliased
from goats.core import iotools
from goats.core import iterables
from goats.core import quantities


class DataViewer(collections.abc.Mapping):
    """An abstract base class for data-viewing objects."""

    def __init__(self, path: iotools.ReadOnlyPath) -> None:
        self.members = self.get_members(path)

    def __iter__(self) -> typing.Iterator:
        return iter(self.members)

    def __len__(self) -> int:
        return len(self.members)

    @abc.abstractmethod
    def get_members(self, path: iotools.ReadOnlyPath) -> typing.Mapping:
        """Get the appropriate members for this viewer."""
        pass

    def __str__(self) -> str:
        """A simplified representation of this object."""
        return '\n\n'.join(f"{k}:\n{v!r}" for k, v in self.items())

    def __repr__(self) -> str:
        """An unambiguous representation of this object."""
        return f"\n::{self.__class__.__qualname__}::\n\n{self}"


class DatasetVariable(typing.NamedTuple):
    """A dataset variable."""

    data: numpy.typing.ArrayLike
    unit: str
    axes: typing.Tuple[str]
    name: str

    def __str__(self) -> str:
        """A simplified representation of this object."""
        return self._display(sep='\n')

    def __repr__(self) -> str:
        """An unambiguous representation of this object."""
        display = self._display(sep='\n', tab=4)
        return f"{self.__class__.__qualname__}(\n{display}\n)"

    def _display(
        self,
        sep: str=', ',
        tab: int=0,
    ) -> str:
        """Helper for `__str__` and `__repr__`."""
        attrs = [
            f"data={type(self.data)}",
            f"unit={self.unit!r}",
            f"axes={self.axes}",
            f"name={self.name!r}",
        ]
        indent = ' ' * tab
        return sep.join(f"{indent}{attr}" for attr in attrs)


class NetCDFVariables(DataViewer):
    """An object for viewing variables in a NetCDF dataset."""

    def get_members(self, path: iotools.ReadOnlyPath) -> typing.Mapping:
        dataset = netCDF4.Dataset(path, 'r')
        return dataset.variables

    def __getitem__(self, name: str) -> DatasetVariable:
        if name in self.members:
            data = self.members[name]
            unit = self._get_unit_from_data(data)
            axes = self._get_axes_from_data(data)
            return DatasetVariable(data, unit, axes, name)
        raise KeyError(f"No variable called '{name}'")

    def _get_axes_from_data(self, data):
        """Compute appropriate variable axes from a dataset object."""
        return tuple(getattr(data, 'dimensions', ()))

    def _get_unit_from_data(self, data):
        """Compute appropriate variable units from a dataset object."""
        available = (
            getattr(data, attr) for attr in ('unit', 'units')
            if hasattr(data, attr)
        )
        return next(available, None)


class DatasetAxis(typing.NamedTuple):
    """A dataset axis."""

    size: int
    name: str

    def __str__(self) -> str:
        """A simplified representation of this object."""
        return f"size={self.size}, name={self.name!r}"

    def __repr__(self) -> str:
        """An unambiguous representation of this object."""
        return f"{self.__class__.__qualname__}({self})"


class NetCDFAxes(DataViewer):
    """An object for viewing axes in a NetCDF dataset."""

    def get_members(self, path: iotools.ReadOnlyPath) -> typing.Mapping:
        dataset = netCDF4.Dataset(path, 'r')
        return dataset.dimensions

    def __getitem__(self, name: str) -> DatasetAxis:
        if name in self.members:
            data = self.members[name]
            size = getattr(data, 'size', None)
            return DatasetAxis(size, name)
        raise KeyError(f"No axis corresponding to {name!r}")


class ViewerFactory(collections.abc.MutableMapping):
    """A class that creates appropriate viewers for a dataset."""

    _viewer_map = {
        '.nc': {
            'variables': NetCDFVariables,
            'axes': NetCDFAxes,
        }
    }

    def __init__(self, path: iotools.ReadOnlyPath) -> None:
        self._viewers = self._get_viewers(path)
        self.path = path

    def __iter__(self) -> typing.Iterator:
        return iter(self._viewers)

    def __len__(self) -> int:
        return len(self._viewers)

    def _get_viewers(
        self,
        path: iotools.ReadOnlyPath,
    ) -> typing.Dict[str, typing.Type[DataViewer]]:
        """Get the viewers for this file format.

        This may expand to accommodate additional file formats or alternate
        methods.
        """
        try:
            viewers = self._viewer_map[path.suffix]
        except KeyError:
            TypeError(f"Unrecognized file type for {path}")
        else:
            return viewers

    def __getitem__(self, group: str):
        """Get the appropriate viewer for this dataset group."""
        if group in self._viewers:
            viewer = self._viewers[group]
            return viewer(self.path)
        raise KeyError(f"No viewer for group '{group}' in {self.path}")

    def __setitem__(self, group: str, viewer: typing.Type[DataViewer]):
        """Associate a new viewer with this dataset group."""
        self._viewers[group] = viewer

    def __delitem__(self, group: str):
        """Delete the viewer associated with this dataset group."""
        del self._viewers[group]


class SubsetKeys(typing.NamedTuple):
    """A subset of names of attributes."""

    full: typing.Tuple[str]
    aliased: typing.Tuple[aliased.MappingKey]
    canonical: typing.Tuple[str]


class DatasetView(iterables.ReprStrMixin, metaclass=iotools.PathSet):
    """A format-agnostic view of a dataset.
    
    An instance of this class provides aliased access to variables and axes
    defined in a specific dataset, given a path to that dataset. It is designed
    to provide a single interface, regardless of file type, with as little
    overhead as possible. Therefore, it does not attempt to modify attributes
    (e.g., converting variable units), since doing so could result in reading a
    potentially large array from disk.
    """

    def __init__(self, path: iotools.PathLike) -> None:
        self.path = iotools.ReadOnlyPath(path)
        self.viewers = ViewerFactory(self.path)
        self._variables = None
        self._axes = None
        self._units = None

    @property
    def variables(self):
        """The variables in this dataset."""
        if self._variables is None:
            variables = {
                ALIASES.get(name, name): variable
                for name, variable in self.viewers['variables'].items()
            }
            self._variables = aliased.Mapping(variables)
        return self._variables

    @property
    def units(self):
        """The unit of each variable, if available."""
        if self._units is None:
            units = {
                name: variable.unit
                for name, variable in self.variables.items(aliased=True)
            }
            self._units = aliased.Mapping(units)
        return self._units

    @property
    def axes(self):
        """The axes in this dataset."""
        if self._axes is None:
            axes = {
                ALIASES.get(name, name): axis
                for name, axis in self.viewers['axes'].items()
            }
            self._axes = aliased.Mapping(axes)
        return self._axes

    def available(self, key: str):
        """Provide the names of available attributes."""
        if key in {'variable', 'variables'}:
            return SubsetKeys(
                full=tuple(self.variables),
                aliased=tuple(self.variables.keys(aliased=True)),
                canonical=tuple(self.viewers['variables'].keys()),
            )
        if key in {'axis', 'axes'}:
            return SubsetKeys(
                full=tuple(self.axes),
                aliased=tuple(self.axes.keys(aliased=True)),
                canonical=tuple(self.viewers['axes'].keys()),
            )

    def iter_axes(self, name: str):
        """Iterate over the axes for the named variable."""
        this = self.variables[name].axes if name in self.variables else ()
        return iter(this)

    def resolve_axes(self, names: typing.Iterable[str]):
        """Compute and order the available axes in `names`."""
        axes = self.available('axes').canonical
        return tuple(name for name in axes if name in names)

    def use(self, **viewers) -> 'DatasetView':
        """Update the viewers for this instance."""
        self.viewers.update(viewers)
        return self

    def __str__(self) -> str:
        return str(self.path)


S = typing.TypeVar('S', bound='Variables')


class Variables(aliased.Mapping):
    """An interface to dataset variables with an optional metric system."""

    def __init__(
        self,
        path: iotools.PathLike,
        system: str=None,
    ) -> None:
        self.dataset = DatasetView(path)
        variables = self.dataset.variables
        super().__init__(variables)
        self._system = system
        self._units = None

    @typing.overload
    def system(self: S) -> quantities.MetricSystem:
        """Get the metric system in use for unit conversions.
        
        Parameters
        ----------
        None

        Returns
        -------
        `~quantities.MetricSystem`
            This instance's current metric system.
        """

    @typing.overload
    def system(self: S, new: str) -> S:
        """Set the metric system for unit conversions.
        
        Parameters
        ----------
        new : string or `~quantities.MetricSystem`
            A new metric system to use for unit conversions.

        Returns
        -------
        `~datasets.Variables`
            The updated instance.
        """

    def system(self, new=None):
        """Concrete implementation."""
        if new:
            self._system = new
            self._units = None
            return self
        if isinstance(self._system, str):
            return quantities.MetricSystem(self._system)
        if isinstance(self._system, quantities.MetricSystem):
            return self._system

    @property
    def units(self):
        """The current unit of each variable."""
        if self._units is None:
            units = {
                name: self._get_unit(unit)
                for name, unit in self.dataset.units.items(aliased=True)
            }
            self._units = aliased.Mapping(units)
        return self._units

    def _get_unit(self, key: str):
        """Get a valid unit based on `key`."""
        unit = standardize(key)
        return (
            self.system().get_unit(unit=unit) if self.system()
            else quantities.Unit(unit)
        )

    def __getitem__(self, key: str):
        """Create the named variable, if possible."""
        variable = super().__getitem__(key)
        unit = self.units[key]
        axes = variable.axes
        name = ALIASES[key]
        data = (unit // variable.unit) * variable.data[:]
        return quantities.Variable(data, unit, axes, name=name)


substitutions = {
    'julian date': 'day',
    'shell': '1',
    'cos(mu)': '1',
    'e-': 'e',
    '# / cm^2 s sr MeV': '# / cm^2 s sr MeV/nuc',
}
"""Conversions from non-standard units."""

T = typing.TypeVar('T')

def standardize(unit: T):
    """Replace this unit string with a standard unit string, if possible.

    This function looks for `unit` in the known conversions and returns the
    standard unit string if it exists. If this doesn't find a standard unit
    string, it just returns the input.
    """
    unit = substitutions.get(str(unit), unit)
    if '/' in unit:
        num, den = str(unit).split('/', maxsplit=1)
        unit = ' / '.join((num.strip(), f"({den.strip()})"))
    return unit


_OBSERVABLES = {
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


ALIASES = aliased.KeyMap(_OBSERVABLES.keys())
METADATA = aliased.Mapping(_OBSERVABLES)

