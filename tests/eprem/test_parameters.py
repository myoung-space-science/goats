from goats.eprem import parameters


reference = {
    'T': 1,
    'F': 0,
    'PI': 3.141592653589793,
    'TWO_PI': 6.283185307179586,
    'VERYSMALL': 1e-33,
    'BADVALUE': 1e+33,
    'BADINT': 2147483647,
    'MP': 1.6726e-24,
    'EV': 1.6022e-12,
    'MEV': 1.6022e-06,
    'GEV': 0.0016022,
    'Q': 4.80320425e-10,
    'C': 29979245800.0,
    'MZERO': 0.9382461065754594,
    'AU': 14959787070000.0,
    'RSUN': 69600000000.0,
    'RSAU': 0.0046524726370988385,
    'TAU': 499.0047838361564,
    'DAY': 0.005775518331436994,
    'MHD_DENSITY_NORM': 1.0,
    'MHD_B_NORM': 0.038771870111656996,
    'OM': 185327.43617160583,
    'FCONVERT': 1.2379901472361203e-12,
    'VOLT': 0.0033333,
    'THRESH': 0.025,
    'MAS_TIME_NORM': 1445.87003080685,
    'MAS_LENGTH_NORM': 69600000000.0,
    'MAS_RHO_NORM': 1.6726e-16,
    'MAS_TIME_CONVERT': 0.016734606838042242,
    'MAS_V_CONVERT': 0.0016056810454004597,
    'MAS_RHO_CONVERT': 100000000.00000001,
    'MAS_B_CONVERT': 56.9199110449208,
    'MAX_STRING_SIZE': 240,
    'MHD_DEFAULT': 0,
    'MHD_ENLIL': 1,
    'MHD_LFMH': 2,
    'MHD_BATSRUS': 3,
    'MHD_MAS': 4,
    'NUM_MPI_BOUNDARY_FLDS': 2,
}

def test_basetypes():
    """Regression test for values defined in src/baseTypes.h."""
    b = parameters.BaseTypesH('~/emmrem/epicMas/source/eprem/src')
    for key, value in reference.items():
        assert b[key] == value
