import sys
import inspect
import typing

import numpy as np
from scipy import integrate

from goats.common import aliased
from goats.common import numerical
from goats.common import quantities
from goats.common import physical
from goats.common import iterables
from goats.eprem import datasets
from goats.eprem import parameters


_metadata = {
    'x': {
        'aliases': ('X',),
        'quantity': 'length',
    },
    'y': {
        'aliases': ('Y',),
        'quantity': 'length',
    },
    'z': {
        'aliases': ('Z',),
        'quantity': 'length',
    },
    'b_mag': {
        'aliases': ('|B|', 'B', 'bmag', 'b mag'),
        'quantity': 'magnetic field',
    },
    'v_mag': {
        'aliases': ('|V|', 'V', 'vmag', 'v mag'),
        'quantity': 'velocity',
    },
    'bv_mag': {
        'aliases': ('bv', '|bv|', 'BV', '|BV|'),
        'quantity': 'velocity * magnetic field',
    },
    'v_para': {
        'aliases': ('vpara', 'Vpara'),
        'quantity': 'velocity',
    },
    'v_perp': {
        'aliases': ('vperp', 'Vperp'),
        'quantity': 'velocity',
    },
    'flow_angle': {
        'aliases': ('flow angle', 'angle'),
        'quantity': 'plane angle',
    },
    'div_v': {
        'aliases': ('divV', 'divv', 'div V', 'div v', 'div(V)', 'div(v)'),
        'quantity': '1 / time',
    },
    'density_ratio': {
        'aliases': ('density ratio' ,'n2/n1', 'n_2/n_1'),
        'quantity': 'number',
    },
    'rigidity': {
        'aliases': ('Rg', 'R_g'),
        'quantity': 'momentum / charge',
    },
    'mean_free_path': {
        'aliases': ('mean free path', 'mfp'),
        'quantity': 'length',
    },
    'acceleration_rate': {
        'aliases': ('acceleration rate',),
        'quantity': '1 / time',
    },
    'energy_density': {
        'aliases': ('energy density',),
        'quantity': 'energy / volume',
    },
    'average_energy': {
        'aliases': ('average energy',),
        'quantity': 'energy',
    },
    'isotropic_distribution': {
        'aliases': ('isotropic distribution', 'isodist', 'f'),
        'removed axes': ['mu'],
        'quantity': 'particle distribution',
    },
    'flux': datasets._VARIABLES['flux'].copy(),
    'fluence': {
        'aliases': (),
        'removed axes': ['time'],
        'quantity': 'number / (area * solid_angle * energy / mass_number)',
    },
    'integral_flux': {
        'aliases': ('integral flux',),
        'removed axes': ['energy'],
        'quantity': 'number / (area * solid_angle * time)',
    },
}

metadata = aliased.Mapping(_metadata)


class Method(iterables.ReprStrMixin):
    """A method that acts on EPREM variables."""

    def __init__(
        self,
        __callable: typing.Callable[..., np.ndarray],
        __metadata: typing.Mapping[str, typing.Any]=None,
    ) -> None:
        self.callable = __callable
        self.metadata = __metadata or {}
        self.signature = inspect.signature(self.callable)
        self.parameters = tuple(self.signature.parameters)

    def __call__(self, *args, **kwargs):
        """Produce the results of this method."""
        return self.callable(*args, **kwargs)

    def __str__(self) -> str:
        """A simplified representation of this object."""
        prms = ', '.join(str(p) for p in self.parameters)
        return f"{self.callable.__qualname__}({prms})"


class Methods(iterables.MappingBase):
    """A mapping of methods that act on EPREM variables."""

    def __init__(self, constants: physical.Constants) -> None:
        self.names = tuple(_metadata.keys())
        super().__init__(self.names)
        self.constants = constants

    def __getitem__(self, name: str) -> Method:
        """Look up a method on this instance."""
        try:
            method = getattr(self, name)
        except AttributeError as exc:
            raise KeyError(f"No method for {name!r}.") from exc
        else:
            return Method(method, metadata[name])

    def C(self, name: str) -> float:
        """Get the numerical value of a single constant."""
        if name in self.constants:
            return float(self.constants[name])
        raise KeyError(f"No constant named {name!r}")

    def x(
        self,
        r: np.ndarray,
        theta: np.ndarray,
        phi: np.ndarray,
    ) -> np.ndarray:
        """The x-axis component given by x = r·sin(θ)cos(φ)"""
        return r * np.sin(theta) * np.cos(phi)

    def y(
        self,
        r: np.ndarray,
        theta: np.ndarray,
        phi: np.ndarray,
    ) -> np.ndarray:
        """The y-axis component given by y = r·sin(θ)sin(φ)"""
        return r * np.sin(theta) * np.sin(phi)

    def z(
        self,
        r: np.ndarray,
        theta: np.ndarray,
    ) -> np.ndarray:
        """The z-axis component given by z = r·cos(θ)"""
        return r * np.cos(theta)

    def b_mag(
        self,
        br: np.ndarray,
        bt: np.ndarray,
        bp: np.ndarray,
    ) -> np.ndarray:
        """The magnetic-field magnitude."""
        return np.sqrt(br**2 + bt**2 + bp**2)

    def v_mag(
        self,
        vr: np.ndarray,
        vt: np.ndarray,
        vp: np.ndarray,
    ) -> np.ndarray:
        """The velocity-field magnitude."""
        return np.sqrt(vr**2 + vt**2 + vp**2)

    def bv_mag(
        self,
        br: np.ndarray,
        bt: np.ndarray,
        bp: np.ndarray,
        vr: np.ndarray,
        vt: np.ndarray,
        vp: np.ndarray,
    ) -> np.ndarray:
        """|B·V|"""
        return self.b_mag(br, bt, bp) * self.v_mag(vr, vt, vp)

    def flow_angle(
        self,
        br: np.ndarray,
        bt: np.ndarray,
        bp: np.ndarray,
        vr: np.ndarray,
        vt: np.ndarray,
        vp: np.ndarray,
    ) -> np.ndarray:
        """The angle between the magnetic- and velocity-field vectors."""
        b_dot_v = self.v_para(br, bt, bp, vr, vt, vp)
        bv_mag = self.bv_mag(br, bt, bp, vr, vt, vp)
        arg = np.array(b_dot_v / bv_mag)
        arg[arg < -1.0] = -1.0
        arg[arg > +1.0] = +1.0
        return np.arccos(arg)

    def v_para(
        self,
        vr: np.ndarray,
        vt: np.ndarray,
        vp: np.ndarray,
        br: np.ndarray,
        bt: np.ndarray,
        bp: np.ndarray,
    ) -> np.ndarray:
        """The velocity-field component parallel to the magnetic field."""
        return (br*vr + bt*vt + bp*vp) / self.b_mag(br, bt, bp)

    def v_perp(
        self,
        vr: np.ndarray,
        vt: np.ndarray,
        vp: np.ndarray,
        br: np.ndarray,
        bt: np.ndarray,
        bp: np.ndarray,
    ) -> np.ndarray:
        """The velocity-field component parallel to the magnetic field."""
        v_mag = self.v_mag(vr, vt, vp)
        v_para = self.v_para(vr, vt, vp, br, bt, bp)
        return np.sqrt(v_mag**2 - v_para**2)

    def div_v(
        self,
        rho: np.ndarray,
        time: np.ndarray,
    ) -> np.ndarray:
        """Divergence of the velocity field.

        The method computes ∇·V from the continuity equation. The continuity
        equation in the co-moving frame can be written as

            1  dn
            --- -- = - div(V)
            n  dt

        where n represents density and dn/dt is the convective derivative. This
        method rewrites the time derivative as a logarithmic derivative and
        computes

                        d(ln n)
            div(V) = - --------
                        dt
        """
        return -1.0*np.gradient(np.log(rho), time, axis=0)

    def density_ratio(
        self,
        rho: np.ndarray,
    ) -> np.ndarray:
        """Compute the ratio of the local density to the initial density."""
        return rho / rho[0, ...]

    def rigidity(
        self,
        energy: typing.Union[float, typing.Iterable[float]],
        mass: typing.Union[np.ndarray, float],
        charge: typing.Union[np.ndarray, float],
    ) -> np.ndarray:
        """The magnetic rigidity.

        This function assumes that `energy` represents the kinetic energy of a
        particle. It then computes the particle momentum based on the
        relativistic relation E² = (`energy` + mc²)² = p²c² + (mc²)². Finally,
        it divides the momentum by the particle charge to get rigidity.
        """
        c = self.C('c')
        total_energy = np.array(energy, ndmin=1) + mass*c**2
        rest_energy = mass*c**2
        p = np.sqrt(total_energy**2 - rest_energy**2) / c
        return p / charge

    def mean_free_path(
        self,
        r: np.ndarray,
        energy: np.ndarray,
        mass: typing.Union[np.ndarray, float],
        charge: typing.Union[np.ndarray, float],
        mfp_radial_power: float,
        rigidity_power: float,
        lambda0: float,
    ) -> np.ndarray:
        """The scattering mean free path."""
        rg = self.rigidity(energy, mass, charge)
        rg0 = self.rigidity(1e6 * self.C('eV'), mass, charge)
        mfp = np.tensordot(
            pow(r / float(self.C('au')), float(mfp_radial_power)),
            pow(rg / rg0, float(rigidity_power)),
            axes=0,
        )
        return mfp * lambda0

    def acceleration_rate(
        self,
        vr: np.ndarray,
        vtheta: np.ndarray,
        vphi: np.ndarray,
        vparticle: np.ndarray,
        angle: np.ndarray,
        mfp: np.ndarray,
        kper_kpar: float,
    ) -> np.ndarray:
        """Compute the acceleration rate for this dataset."""
        v1_dot_v2 = (
            self._ahead(vr)*self._behind(vr)
            + self._ahead(vtheta)*self._behind(vtheta)
            + self._ahead(vphi)*self._behind(vphi)
        )
        v = self.v_mag(vr, vtheta, vphi)
        v1 = self._ahead(v)
        delta_v = v1 * np.abs(1 - v1_dot_v2 / (v1*v1))
        kxx_kpar = np.cos(angle)**2 + kper_kpar*np.sin(angle)**2
        vpara = np.tensordot(kxx_kpar, vparticle, axes=0)
        kappa = mfp * (vpara / 3.0)
        forward = inverse = (2, 3, 0, 1)
        tmp = kappa.transpose(forward) / v
        kxx_vx = tmp.transpose(inverse)
        delta_x = 3.0 * (self._ahead(kxx_vx) + self._behind(kxx_vx))
        tmp = delta_v / delta_x.transpose(forward)
        return tmp.transpose(inverse)

    def _ahead(self, arr: np.ndarray) -> np.ndarray:
        """Extract array values at the shell immediately ahead."""
        nshells = arr.shape[1]
        idx = [*list(range(1, nshells)), nshells-1]
        return arr[:, idx, ...]

    def _behind(self, arr: np.ndarray) -> np.ndarray:
        """Extract array values at the shell immediately behind."""
        nshells = arr.shape[1]
        idx = [0, *list(range(nshells-1))]
        return arr[:, idx, ...]

    def energy_density(
        self,
        speed: np.ndarray,
        isodist: np.ndarray,
    ) -> np.ndarray:
        """Compute the distribution energy density for this dataset.

        This function converts the result to energy density in ergs / cm³ before
        returning. It first approximates the integral of v⁴f(v) with respect to
        v, where v is the particle velocity and f(v) is the isotropic
        distribution. The particle velocities have units of cm/s and the
        isotropic distribution has units of s³/cm⁶, giving the intermediate
        value units of (cm/s)⁵s³/cm⁶, or 1/cm/s². Multiplying by the proton mass
        in grams yields an expression with units of g/cm/s², or erg/cm³.
        """
        v = np.array(speed, ndmin=1)
        dv = np.gradient(v, axis=-1)
        return self.C('mp') * np.sum(v**4 * isodist * dv, axis=-1)

    def average_energy(
        self,
        speed: np.ndarray,
        isodist: np.ndarray,
    ) -> np.ndarray:
        """Compute the average distribution energy for this dataset.

        This function converts the result to average energy in ergs before
        returning. It first approximates the integral of v²f(v) with respect to
        v, where v is the particle velocity and f(v) is the isotropic
        distribution. The particle velocities have units of cm/s and the
        isotropic distribution has units of s³/cm⁶, giving the intermediate
        value units of 1/cm³. This function then computes the energy density via
        `energy_density()`, divides that result by the intermediate result
        computed here, and further divides by 2 to represent the kinetic energy.
        The final result has units of ergs.
        """
        v = np.array(speed, ndmin=1)
        dv = np.gradient(v, axis=-1)
        normalizer = np.sum(v**2 * isodist * dv, axis=-1)
        epsilon = self.energy_density(v, isodist)
        return 0.5 * epsilon / normalizer

    def isotropic_distribution(
        self,
        dist: np.ndarray,
    ) -> np.ndarray:
        """Compute the isotropic distribution.

        This function computes the isotropic form of a given particle
        distribution by averaging over all pitch angles. It returns an array
        with dimensions of (times, shells, species, energies).
        """
        return np.mean(dist, axis=-1)

    def flux(
        self,
        energy: typing.Union[float, typing.Iterable[float]],
        isodist: np.ndarray,
    ) -> np.ndarray:
        """Compute the differential energy flux of a distribution."""
        dist_to_flux = 2 * np.array(energy, ndmin=1) / self.C('mp')**2
        return dist_to_flux * isodist

    def fluence(
        self,
        flux: np.ndarray,
        time: np.ndarray,
    ) -> np.ndarray:
        """Compute the distribution fluence."""
        return integrate.simps(flux, time, axis=0)

    def alt_intflux(
        self,
        energies: np.ndarray,
        flux: np.ndarray,
        minimum_energy: float,
    ) -> np.ndarray:
        """Compute the integral flux by the old algorithm.
        
        Currently exists only for testing.
        """
        x0 = energies[0, :-1]
        x1 = energies[0, 1:]
        y0 = np.squeeze(flux[..., :-1])
        y1 = np.squeeze(flux[..., 1:])
        dlnx = np.log10(x1) - np.log10(x0)
        kernel = 0.5 * dlnx * (x0*y0 + x1*y1)
        kernel[..., x1 <= minimum_energy] = 0.0
        return np.sum(
            kernel,
            axis=-1,
            dtype=np.float64,
        )

    def integral_flux(
        self,
        energies: np.ndarray,
        flux: np.ndarray,
        *minimum_energy: float,
    ) -> np.ndarray:
        """Compute the integral flux of a distribution above a given energy."""
        if energies.ndim == 1:
            energies = energies[None, :]
            if flux.ndim == 3:
                # This is under the energies.ndim check because I assume that if
                # the species dimension was squeezed from energies, it might
                # have also been squeezed from flux. In other words, I am
                # assuming that the only reason flux would have 3 (instead of 4)
                # dimensions is if the species dimensions was squeezed. Is this
                # a good assumption?
                flux = flux[:, :, None, :]
        int_flux = np.zeros((*flux.shape[:3], len(minimum_energy)))
        for i, bound in enumerate(minimum_energy):
            m = max(bound, sys.float_info.min)
            use_all = m < np.min(energies)
            for s, species_energy in enumerate(energies):
                f = flux[..., s, :]
                e = species_energy
                y, x = (f, e) if use_all else self._interpolate(f, e, m)
                int_flux[..., s, i] = integrate.simps(y, x)
        return int_flux

    def _interpolate(
        self,
        f: np.ndarray,
        x: np.ndarray,
        m: float,
    ) -> np.ndarray:
        """Interpolate `f` at `x = m` if necessary.

        This function uses an algorithm that interpolates `f` at `x = m` via a
        power law. If `m` is already in `x`, the interpolation leaves `f`
        unchanged. It was designed to support computations of integral flux and
        therefore currently only interpolates to a lower bound.

        This function makes copies of `x` and `f` before applying the algorithm.
        """
        xc = x.copy()
        fc = f.copy()
        i0, x0 = numerical.find_nearest(xc, m, bound='upper')
        # TODO: Make sure i0+1 is not out of range.
        beta = np.log(fc[..., i0+1] / fc[..., i0]) / np.log(xc[i0+1] / x0)
        base = np.full_like(beta, m / x0)
        xc[i0] = m
        fc[..., i0] *= np.power(base, beta)
        return fc[..., i0:], xc[i0:]


class Function(iterables.ReprStrMixin):
    """A function from variables and scalars to a single variable."""

    def __init__(
        self,
        method: Method,
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
        quantities.Variable,
        quantities.Scalar,
        typing.Iterable[quantities.Scalar],
    )
    Argument = typing.Union[
        quantities.Variable,
        quantities.Scalar,
        typing.Iterable[quantities.Scalar],
    ]

    def __call__(
        self,
        arguments: typing.Mapping[str, Argument],
        unit: typing.Union[str, quantities.Unit],
    ) -> quantities.Variable:
        """Build a variable by calling the instance method."""
        arrays = []
        floats = []
        known = [
            argument for key, argument in arguments.items()
            if key in self.parameters
        ]
        for arg in known:
            if isinstance(arg, quantities.Variable):
                arrays.append(np.array(arg))
            elif isinstance(arg, quantities.Scalar):
                floats.append(float(arg))
            elif (
                isinstance(arg, typing.Iterable)
                and all(isinstance(a, quantities.Scalar) for a in arg)
            ): floats.extend([float(a) for a in arg])
        data = self.method(*arrays, *floats)
        return quantities.Variable(
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
        constants: physical.Constants,
    ) -> None:
        methods = Methods(constants)
        mapping = {
            tuple([k, *v.get('aliases', ())]): methods[k]
            for k, v in _metadata.items()
        }
        super().__init__(mapping=mapping)
        self.dataset = dataset
        self._primary = (
            *tuple(self.dataset.variables.keys()),
            *tuple(arguments.keys()),
            *tuple(constants.keys()),
        )
        self._axes_cache = {}
        self._dependencies_cache = {}

    def __getitem__(self, key: str):
        """Construct the requested function object, if possible"""
        if method := self.get_method(key):
            axes = self.get_axes(key)
            quantity = metadata.get(key, {}).get('quantity', None)
            dependencies = self.get_dependencies(key)
            return Function(
                method,
                quantity,
                axes,
                dependencies=dependencies,
                name=key,
            )
        raise KeyError(f"No function corresponding to {key!r}")

    def get_method(self, key: str) -> Method:
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
        self._removed = list(method.metadata.get('removed axes', []))
        self._restored = list(method.metadata.get('restored axes', []))
        self._accumulated = []
        axes = self._gather_axes(method)
        self._axes_cache[key] = axes
        return axes

    def _gather_axes(self, target: Method):
        """Recursively gather appropriate axes."""
        for parameter in target.parameters:
            if parameter in self.dataset.variables:
                axes = self.dataset.iter_axes(parameter)
                self._accumulated.extend(axes)
            elif method := self.get_method(parameter):
                self._removed.extend(method.metadata.get('removed axes', []))
                self._restored.extend(method.metadata.get('restored axes', []))
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

    def _gather_dependencies(self, target: Method):
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


