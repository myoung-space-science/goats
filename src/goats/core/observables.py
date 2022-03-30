from goats.core import aliased


__all__ = [
    'ALIASES',
    'METADATA',
]


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

