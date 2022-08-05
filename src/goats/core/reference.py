from goats.core import aliased


__all__ = [
    'ALIASES',
    'METADATA',
    'NAMES',
]


_BASE = {
    ('time', 't', 'times'): {
        'quantity': 'time',
    },
    ('shell', 'shells'): {
        'quantity': 'number',
    },
    (
        'mu', 'mus',
        'pitch angle', 'pitch-angle', 'pitch-angle cosine',
        'pitch angles', 'pitch-angles', 'pitch-angle cosines',
    ): {
        'quantity': 'ratio',
    },
    ('mass', 'm'): {
        'quantity': 'mass',
    },
    ('charge', 'q'): {
        'quantity': 'charge',
    },
    ('energy', 'egrid', 'energies', 'E'): {
        'quantity': 'energy',
    },
    ('speed', 'vgrid', 'vparticle'): {
        'quantity': 'velocity',
    },
    ('radius', 'R', 'r'): {
        'quantity': 'length',
    },
    ('theta', 'T'): {
        'quantity': 'plane angle',
    },
    ('phi', 'P'): {
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
    ('rho', 'Rho'): {
        'quantity': 'number density',
    },
    ('dist', 'Dist', 'f'): {
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
    ('B', 'b_mag', '|B|', 'bmag', 'b mag'): {
        'quantity': 'magnetic field',
    },
    ('V', 'v_mag', '|V|', 'vmag', 'v mag', 'v', '|v|'): {
        'quantity': 'velocity',
    },
    ('BV', 'bv_mag', 'bv', '|bv|', '|BV|'): {
        'quantity': 'velocity * magnetic field',
    },
    ('Vpara', 'v_para', 'vpara'): {
        'quantity': 'velocity',
    },
    ('Vperp', 'v_perp', 'vperp'): {
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
    ('isotropic_distribution', 'isotropic distribution', 'isodist'): {
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

_NAMES = {
    v.get('name', k if isinstance(k, str) else k[0]): k
    for k, v in _BASE.items()
}

NAMES = aliased.NameMap(_NAMES)

_METADATA = {
    k: {i: j for i, j in v.items() if i != 'name'}
    for k, v in _BASE.items()
}

ALIASES = aliased.KeyMap(*_METADATA.keys())
METADATA = aliased.Mapping(_METADATA)

