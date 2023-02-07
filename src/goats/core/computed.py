"""
Globally defined functions and corresponding interface objects.
"""

import collections.abc
import inspect
import typing
import sys

import numpy
import numpy.typing
from scipy import integrate

from goats.core import aliased
from goats.core import axis
from goats.core import iterables
from goats.core import metadata
from goats.core import numerical
from goats.core import physical
from goats.core import reference
from goats.core import variable


registry = iterables.ObjectRegistry(object_key='method')


MKS = physical.Constants('mks')


@registry.register
def x(
    r: numpy.ndarray,
    theta: numpy.ndarray,
    phi: numpy.ndarray,
) -> numpy.ndarray:
    """The x-axis component given by x = r·sin(θ)cos(φ)"""
    return r * numpy.sin(theta) * numpy.cos(phi)


@registry.register
def y(
    r: numpy.ndarray,
    theta: numpy.ndarray,
    phi: numpy.ndarray,
) -> numpy.ndarray:
    """The y-axis component given by y = r·sin(θ)sin(φ)"""
    return r * numpy.sin(theta) * numpy.sin(phi)


@registry.register
def z(
    r: numpy.ndarray,
    theta: numpy.ndarray,
) -> numpy.ndarray:
    """The z-axis component given by z = r·cos(θ)"""
    return r * numpy.cos(theta)


@registry.register
def b_mag(
    br: numpy.ndarray,
    bt: numpy.ndarray,
    bp: numpy.ndarray,
) -> numpy.ndarray:
    """The magnetic-field magnitude."""
    return numpy.sqrt(br**2 + bt**2 + bp**2)


@registry.register
def v_mag(
    vr: numpy.ndarray,
    vt: numpy.ndarray,
    vp: numpy.ndarray,
) -> numpy.ndarray:
    """The velocity-field magnitude."""
    return numpy.sqrt(vr**2 + vt**2 + vp**2)


@registry.register
def bv_mag(
    br: numpy.ndarray,
    bt: numpy.ndarray,
    bp: numpy.ndarray,
    vr: numpy.ndarray,
    vt: numpy.ndarray,
    vp: numpy.ndarray,
) -> numpy.ndarray:
    """|B·V|"""
    return b_mag(br, bt, bp) * v_mag(vr, vt, vp)


@registry.register
def flow_angle(
    br: numpy.ndarray,
    bt: numpy.ndarray,
    bp: numpy.ndarray,
    vr: numpy.ndarray,
    vt: numpy.ndarray,
    vp: numpy.ndarray,
) -> numpy.ndarray:
    """The angle between the magnetic- and velocity-field vectors."""
    b_dot_v = v_para(br, bt, bp, vr, vt, vp)
    bv = bv_mag(br, bt, bp, vr, vt, vp)
    arg = numpy.array(b_dot_v / bv)
    arg[arg < -1.0] = -1.0
    arg[arg > +1.0] = +1.0
    return numpy.arccos(arg)


@registry.register
def v_para(
    vr: numpy.ndarray,
    vt: numpy.ndarray,
    vp: numpy.ndarray,
    br: numpy.ndarray,
    bt: numpy.ndarray,
    bp: numpy.ndarray,
) -> numpy.ndarray:
    """The velocity-field component parallel to the magnetic field."""
    return (br*vr + bt*vt + bp*vp) / b_mag(br, bt, bp)


@registry.register
def v_perp(
    vr: numpy.ndarray,
    vt: numpy.ndarray,
    vp: numpy.ndarray,
    br: numpy.ndarray,
    bt: numpy.ndarray,
    bp: numpy.ndarray,
) -> numpy.ndarray:
    """The velocity-field component parallel to the magnetic field."""
    mag = v_mag(vr, vt, vp)
    para = v_para(vr, vt, vp, br, bt, bp)
    return numpy.sqrt(mag**2 - para**2)


@registry.register
def div_v(
    rho: numpy.ndarray,
    time: numpy.ndarray,
) -> numpy.ndarray:
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
    return -1.0*numpy.gradient(numpy.log(rho), time, axis=0)


@registry.register
def density_ratio(
    rho: numpy.ndarray,
) -> numpy.ndarray:
    """Compute the ratio of the local density to the initial density."""
    return rho / rho[0, ...]


@registry.register
def rigidity(
    energy: typing.Union[float, typing.Iterable[float]],
    mass: typing.Union[numpy.ndarray, float],
    charge: typing.Union[numpy.ndarray, float],
) -> numpy.ndarray:
    """The magnetic rigidity.

    This function assumes that `energy` represents the kinetic energy of a
    particle. It then computes the particle momentum based on the
    relativistic relation E² = (`energy` + mc²)² = p²c² + (mc²)². Finally,
    it divides the momentum by the particle charge to get rigidity.
    """
    c = float(MKS['c'])
    total_energy = numpy.array(energy, ndmin=1) + mass*c**2
    rest_energy = mass*c**2
    p = numpy.sqrt(total_energy**2 - rest_energy**2) / c
    return p / charge


@registry.register
def mean_free_path(
    r: numpy.ndarray,
    energy: numpy.ndarray,
    mass: typing.Union[numpy.ndarray, float],
    charge: typing.Union[numpy.ndarray, float],
    mfp_radial_power: float,
    rigidity_power: float,
    lambda0: float,
) -> numpy.ndarray:
    """The scattering mean free path."""
    rg = rigidity(energy, mass, charge)
    rg0 = rigidity(1e6 * float(MKS['eV']), mass, charge)
    mfp = numpy.tensordot(
        pow(r / float(MKS['au']), mfp_radial_power),
        pow(rg / rg0, rigidity_power),
        axes=0,
    )
    return mfp * lambda0


@registry.register
def acceleration_rate(
    vr: numpy.ndarray,
    vtheta: numpy.ndarray,
    vphi: numpy.ndarray,
    vparticle: numpy.ndarray,
    angle: numpy.ndarray,
    mfp: numpy.ndarray,
    kper_kpar: float,
) -> numpy.ndarray:
    """Compute the acceleration rate for this dataset."""
    v1_dot_v2 = (
        _ahead(vr)*_behind(vr)
        + _ahead(vtheta)*_behind(vtheta)
        + _ahead(vphi)*_behind(vphi)
    )
    v = v_mag(vr, vtheta, vphi)
    v1 = _ahead(v)
    delta_v = v1 * numpy.abs(1 - v1_dot_v2 / (v1*v1))
    kxx_kpar = numpy.cos(angle)**2 + kper_kpar*numpy.sin(angle)**2
    vpara = numpy.tensordot(kxx_kpar, vparticle, axes=0)
    kappa = mfp * (vpara / 3.0)
    forward = inverse = (2, 3, 0, 1)
    tmp = kappa.transpose(forward) / v
    kxx_vx = tmp.transpose(inverse)
    delta_x = 3.0 * (_ahead(kxx_vx) + _behind(kxx_vx))
    tmp = delta_v / delta_x.transpose(forward)
    return tmp.transpose(inverse)

def _ahead(arr: numpy.ndarray) -> numpy.ndarray:
    """Extract array values at the shell immediately ahead."""
    nshells = arr.shape[1]
    idx = [*list(range(1, nshells)), nshells-1]
    return arr[:, idx, ...]

def _behind(arr: numpy.ndarray) -> numpy.ndarray:
    """Extract array values at the shell immediately behind."""
    nshells = arr.shape[1]
    idx = [0, *list(range(nshells-1))]
    return arr[:, idx, ...]


@registry.register(removed='energy')
def energy_density(
    speed: numpy.ndarray,
    isodist: numpy.ndarray,
) -> numpy.ndarray:
    """Compute the distribution energy density for this dataset.

    This function converts the result to energy density in J/cm³ before
    returning. It first approximates the integral of v⁴f(v) with respect to v,
    where v is the particle velocity and f(v) is the isotropic distribution. The
    particle velocities have units of m/s and the isotropic distribution has
    units of s³/m⁶, giving the intermediate value units of (m/s)⁵s³/m⁶, or
    1/m/s². Multiplying by the proton mass yields an expression with units of
    kg/m/s², or J/cm³.
    """
    v = numpy.array(speed, ndmin=1)
    dv = numpy.gradient(v, axis=-1)
    return float(MKS['mp']) * numpy.sum(v**4 * isodist * dv, axis=-1)


@registry.register(removed='energy')
def average_energy(
    speed: numpy.ndarray,
    isodist: numpy.ndarray,
) -> numpy.ndarray:
    """Compute the average distribution energy for this dataset.

    This function converts the result to average energy in J before returning.
    It first approximates the integral of v²f(v) with respect to v, where v is
    the particle velocity and f(v) is the isotropic distribution. The particle
    velocities have units of m/s and the isotropic distribution has units of
    s³/m⁶, giving the intermediate value units of 1/m³. This function then
    computes the energy density via `energy_density()`, divides that result by
    the intermediate result computed here, and further divides by 2 to represent
    the kinetic energy. The final result has units of J.
    """
    v = numpy.array(speed, ndmin=1)
    dv = numpy.gradient(v, axis=-1)
    normalizer = numpy.sum(v**2 * isodist * dv, axis=-1)
    epsilon = energy_density(v, isodist)
    return 0.5 * epsilon / normalizer


@registry.register(removed='mu')
def isotropic_distribution(
    dist: numpy.ndarray,
) -> numpy.ndarray:
    """Compute the isotropic distribution.

    This function computes the isotropic form of a given particle
    distribution by averaging over all pitch angles. It returns an array
    with the final dimension removed.
    """
    return numpy.mean(dist, axis=-1)


@registry.register
def flux(
    energy: typing.Union[float, typing.Iterable[float]],
    isodist: numpy.ndarray,
) -> numpy.ndarray:
    """Compute the differential energy flux of a distribution."""
    dist_to_flux = 2 * numpy.array(energy, ndmin=1) / float(MKS['mp'])**2
    return dist_to_flux * isodist


@registry.register(removed='time')
def fluence(
    flux: numpy.ndarray,
    time: numpy.ndarray,
) -> numpy.ndarray:
    """Compute the distribution fluence."""
    return integrate.simps(flux, time, axis=0)


def alt_intflux(
    energies: numpy.ndarray,
    flux: numpy.ndarray,
    minimum_energy: float,
) -> numpy.ndarray:
    """Compute the integral flux by the old algorithm.
    
    Currently exists only for testing.
    """
    x0 = energies[0, :-1]
    x1 = energies[0, 1:]
    y0 = numpy.squeeze(flux[..., :-1])
    y1 = numpy.squeeze(flux[..., 1:])
    dlnx = numpy.log10(x1) - numpy.log10(x0)
    kernel = 0.5 * dlnx * (x0*y0 + x1*y1)
    kernel[..., x1 <= minimum_energy] = 0.0
    return numpy.sum(
        kernel,
        axis=-1,
        dtype=numpy.float64,
    )


@registry.register(removed='energy', added='minimum energy')
def integral_flux(
    energies: numpy.ndarray,
    flux: numpy.ndarray,
    *minimum_energy: float,
) -> numpy.ndarray:
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
    int_flux = numpy.zeros((*flux.shape[:3], len(minimum_energy)))
    for i, bound in enumerate(minimum_energy):
        m = max(bound, sys.float_info.min)
        use_all = m < numpy.min(energies)
        for s, species_energy in enumerate(energies):
            f = flux[..., s, :]
            e = species_energy
            y, x = (f, e) if use_all else _interpolate(f, e, m)
            int_flux[..., s, i] = integrate.simpson(y, x)
    return int_flux

def _interpolate(
    f: numpy.ndarray,
    x: numpy.ndarray,
    m: float,
) -> numpy.ndarray:
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
    beta = numpy.log(fc[..., i0+1] / fc[..., i0]) / numpy.log(xc[i0+1] / x0)
    base = numpy.full_like(beta, m / x0)
    xc[i0] = m
    fc[..., i0] *= numpy.power(base, beta)
    return fc[..., i0:], xc[i0:]


REGISTRY = aliased.Mapping(registry, aliases=reference.ALIASES)


class Method(collections.abc.Mapping, iterables.ReprStrMixin):
    """A callable object with associated information."""

    def __init__(
        self,
        __callable: typing.Callable[..., numpy.ndarray],
        **info,
    ) -> None:
        self.callable = __callable
        self.info = info
        self.signature = inspect.signature(self.callable)
        self._name = None
        self._parameters = None

    @property
    def parameters(self):
        """The names of arguments to this method."""
        if self._parameters is None:
            self._parameters = tuple(self.signature.parameters)
        return self._parameters

    @property
    def name(self):
        """The name of this method's callable object."""
        if self._name is None:
            self._name = str(self.callable)
        return self._name

    def compute(self, **dependencies):
        """Produce the results of this method."""
        return self.callable(*self.parse(dependencies))

    def parse(self, dependencies: typing.Mapping):
        """Extract arrays and floats from `dependencies`."""
        arrays = []
        floats = []
        for name in self.parameters:
            arg = dependencies[name]
            if isinstance(arg, variable.Quantity):
                arrays.append(numpy.array(arg))
            elif isinstance(arg, physical.Scalar):
                floats.append(float(arg))
            elif (
                isinstance(arg, typing.Iterable)
                and all(isinstance(a, physical.Scalar) for a in arg)
            ): floats.extend([float(a) for a in arg])
        return *arrays, *floats

    def __len__(self):
        """The number of available metadata attributes."""
        return len(self.info)

    def __iter__(self):
        """Iterate over metadata attributes."""
        return iter(self.info)

    def __getitem__(self, __k: str):
        """Retrieve a metadata attribute, if possible."""
        if __k in self.info:
            return self.info[__k]
        raise KeyError(f"Unknown metadata attribute {__k!r}")

    def __bool__(self) -> bool:
        """Same as the truth value of this method's callable object."""
        return bool(self.callable)

    def __str__(self) -> str:
        """A simplified representation of this object."""
        return f"{self.name}{self.signature}"


class Quantity(iterables.ReprStrMixin):
    """A callable quantity that produces a variable quantity."""

    def __init__(
        self,
        __method: Method,
        dimensions: typing.Iterable[str]=None,
        unit: metadata.UnitLike=None,
    ) -> None:
        self.method = __method
        self._dimensions = dimensions
        self._unit = unit
        self._parameters = None

    @property
    def dimensions(self):
        """This quantity's indexable axes."""
        return metadata.Dimensions(self._dimensions)

    @property
    def unit(self):
        """This quantity's metric unit."""
        return metadata.Unit(self._unit)

    @property
    def parameters(self):
        """The parameters to this quantity's method."""
        if self._parameters is None:
            self._parameters = tuple(self.method.parameters)
        return self._parameters

    def __getitem__(self, unit: metadata.UnitLike):
        """Set the unit of the resultant values.
        
        Notes
        -----
        See note at `~measurable.Quantity.__getitem__`.
        """
        if unit != self._unit:
            self._unit = unit
        return self

    def __call__(self, **quantities):
        """Create a variable quantity from input quantities."""
        return variable.Quantity(
            self.method.compute(**quantities),
            dimensions=self.dimensions,
            unit=self.unit,
        )

    def __str__(self) -> str:
        attrs = [
            self.method.name,
            f"dimensions={self.dimensions}",
            f"unit={str(self.unit)!r}",
        ]
        return ', '.join(attrs)

