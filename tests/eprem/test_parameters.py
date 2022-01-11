import pytest

from goats.eprem import parameters


@pytest.fixture
def source_path():
    """The path to the reference EPREM distribution."""
    return '~/emmrem/epicMas/source/eprem/src'


@pytest.fixture
def config_path(datadirs):
    """The path to the test configuration file"""
    return datadirs['cone']['obs'] / 'eprem_input_file'


def test_basetypes_h(source_path):
    """Regression test for values defined in src/baseTypes.h."""
    b = parameters.BaseTypesH(source_path)
    for key, value in _BASETYPES_H.items():
        assert b[key] == value


def test_configuration_c(source_path):
    """Make sure the object contains everything in src/configuration.c."""
    c = parameters.ConfigurationC(source_path)
    assert len(c) == len(_CONFIGURATION_C)
    assert all(key in c for key in _CONFIGURATION_C)


def test_default_values(source_path):
    """Compare the values of all parameters to reference values."""
    cfg = parameters.ConfigManager(source_path)
    for key, parameter in _CONFIGURATION_C.items():
        assert cfg[key] == parameter['default']


_BASETYPES_H = {
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

_CONFIGURATION_C = {
    'numNodesPerStream': {
        'type': int,
        'default': None,
        'minimum': None,
        'maximum': _BASETYPES_H['BADINT'],
     },
    'numRowsPerFace': {
        'type': int,
        'default': 2,
        'minimum': 1,
        'maximum': _BASETYPES_H['BADINT'],
     },
    'numColumnsPerFace': {
        'type': int,
        'default': 2,
        'minimum': 1,
        'maximum': _BASETYPES_H['BADINT'],
     },
    'numEnergySteps': {
        'type': int,
        'default': 20,
        'minimum': 2,
        'maximum': _BASETYPES_H['BADINT'],
     },
    'numMuSteps': {
        'type': int,
        'default': 20,
        'minimum': 2,
        'maximum': _BASETYPES_H['BADINT'],
     },
    'rScale': {
        'type': float,
        'default': 0.005,
        'minimum': _BASETYPES_H['VERYSMALL'],
        'maximum': _BASETYPES_H['BADVALUE'],
     },
    'flowMag': {
        'type': float,
        'default': 400.0e5,
        'minimum': _BASETYPES_H['VERYSMALL'],
        'maximum': _BASETYPES_H['BADVALUE'],
     },
    'mhdDensityAu': {
        'type': float,
        'default': 8.30,
        'minimum': _BASETYPES_H['VERYSMALL'],
        'maximum': _BASETYPES_H['BADVALUE'],
     },
    'mhdBAu': {
        'type': float,
        'default': 1.60e-5,
        'minimum': _BASETYPES_H['VERYSMALL'],
        'maximum': _BASETYPES_H['BADVALUE'],
     },
    'simStartTime': {
        'type': float,
        'default': 0.0,
        'minimum': 0.0,
        'maximum': _BASETYPES_H['BADVALUE'],
     },
    'tDel': {
        'type': float,
        'default': 0.01041666666667,
        'minimum': _BASETYPES_H['VERYSMALL'],
        'maximum': _BASETYPES_H['BADVALUE'],
     },
    'simStopTime': {
        'type': float,
        'default': 0.01041666666667,
        'minimum': 0.0,
        'maximum': _BASETYPES_H['BADVALUE'],
     },
    'numEpSteps': {
        'type': int,
        'default': 30,
        'minimum': 1,
        'maximum': _BASETYPES_H['BADINT'],
     },
    'aziSunStart': {
        'type': float,
        'default': 0.0,
        'minimum': 0.0,
        'maximum': _BASETYPES_H['BADVALUE'],
     },
    'omegaSun': {
        'type': float,
        'default': 0.004144*_BASETYPES_H['TAU']/_BASETYPES_H['MAS_TIME_NORM'],
        'minimum': 0.0,
        'maximum': _BASETYPES_H['BADVALUE'],
     },
    'lamo': {
        'type': float,
        'default': 1.0,
        'minimum': _BASETYPES_H['VERYSMALL'],
        'maximum': _BASETYPES_H['BADVALUE'],
     },
    'dsh_min': {
        'type': float,
        'default': 5.0e-5,
        'minimum': _BASETYPES_H['VERYSMALL'],
        'maximum': _BASETYPES_H['BADVALUE'],
     },
    'dsh_hel_min': {
        'type': float,
        'default': 2.5e-4,
        'minimum': _BASETYPES_H['VERYSMALL'],
        'maximum': _BASETYPES_H['BADVALUE'],
     },
    'kperxkpar': {
        'type': float,
        'default': 0.01,
        'minimum': _BASETYPES_H['VERYSMALL'],
        'maximum': _BASETYPES_H['BADVALUE'],
     },
    'mfpRadialPower': {
        'type': float,
        'default': 2.0,
        'minimum': -1.0 * _BASETYPES_H['BADVALUE'],
        'maximum': _BASETYPES_H['BADVALUE'],
     },
    'rigidityPower': {
        'type': float,
        'default': 1 / 3,
        'minimum': _BASETYPES_H['VERYSMALL'],
        'maximum': _BASETYPES_H['BADVALUE'],
     },
    'focusingLimit': {
        'type': float,
        'default': 1.0,
        'minimum': 0.0,
        'maximum': 1.0,
     },
    'eMin': {
        'type': float,
        'default': 1.0,
        'minimum': _BASETYPES_H['VERYSMALL'],
        'maximum': _BASETYPES_H['BADVALUE'],
     },
    'eMax': {
        'type': float,
        'default': 1000.0,
        'minimum': 1.0,
        'maximum': _BASETYPES_H['BADVALUE'],
     },
    'useStochastic': {
        'type': int,
        'default': 0,
        'minimum': 0,
        'maximum': 1,
     },
    'useEPBoundary': {
        'type': int,
        'default': 1,
        'minimum': 0,
        'maximum': 1,
     },
    'checkSeedPopulation': {
        'type': int,
        'default': 1,
        'minimum': 0,
        'maximum': 1,
     },
    'seedFunctionTest': {
        'type': int,
        'default': 0,
        'minimum': 0,
        'maximum': 1,
     },
    'fluxLimiter': {
        'type': int,
        'default': 1,
        'minimum': 0,
        'maximum': 1,
     },
    'gammaEhigh': {
        'type': float,
        'default': 0.0,
        'minimum': -1.0 * _BASETYPES_H['BADVALUE'],
        'maximum': _BASETYPES_H['BADVALUE'],
     },
    'gammaElow': {
        'type': float,
        'default': 0.0,
        'minimum': -1.0 * _BASETYPES_H['BADVALUE'],
        'maximum': _BASETYPES_H['BADVALUE'],
     },
    'FailModeDump': {
        'type': int,
        'default': 1,
        'minimum': 0,
        'maximum': 1,
     },
    'outputFloat': {
        'type': int,
        'default': 0,
        'minimum': 0,
        'maximum': 1,
     },
    'unifiedOutput': {
        'type': int,
        'default': 1,
        'minimum': 0,
        'maximum': 1,
     },
    'unifiedOutputTime': {
        'type': float,
        'default': 0.0,
        'minimum': 0.0,
        'maximum': _BASETYPES_H['BADVALUE'],
     },
    'pointObserverOutput': {
        'type': int,
        'default': 0,
        'minimum': 0,
        'maximum': 1,
     },
    'pointObserverOutputTime': {
        'type': float,
        'default': 0.0,
        'minimum': 0.0,
        'maximum': _BASETYPES_H['BADVALUE'],
     },
    'streamFluxOutput': {
        'type': int,
        'default': 0,
        'minimum': 0,
        'maximum': 1,
     },
    'streamFluxOutputTime': {
        'type': float,
        'default': 0.0,
        'minimum': 0.0,
        'maximum': _BASETYPES_H['BADVALUE'],
     },
    'subTimeCouple': {
        'type': int,
        'default': 0,
        'minimum': 0,
        'maximum': 1,
     },
    'epremDomain': {
        'type': int,
        'default': 0,
        'minimum': 0,
        'maximum': 1,
     },
    'epremDomainOutputTime': {
        'type': float,
        'default': 0.0,
        'minimum': 0.0,
        'maximum': _BASETYPES_H['BADVALUE'],
     },
    'unstructuredDomain': {
        'type': int,
        'default': 0,
        'minimum': 0,
        'maximum': 1,
     },
    'unstructuredDomainOutputTime': {
        'type': float,
        'default': 0.0,
        'minimum': 0.0,
        'maximum': _BASETYPES_H['BADVALUE'],
     },
    'useAdiabaticChange': {
        'type': int,
        'default': 1,
        'minimum': 0,
        'maximum': 1,
     },
    'useAdiabaticFocus': {
        'type': int,
        'default': 1,
        'minimum': 0,
        'maximum': 1,
     },
    'useShellDiffusion': {
        'type': int,
        'default': 0,
        'minimum': 0,
        'maximum': 1,
     },
    'useParallelDiffusion': {
        'type': int,
        'default': 1,
        'minimum': 0,
        'maximum': 1,
     },
    'useDrift': {
        'type': int,
        'default': 0,
        'minimum': 0,
        'maximum': 1,
     },
    'numSpecies': {
        'type': int,
        'default': 1,
        'minimum': 0,
        'maximum': 100,
     },
    'mass': {
        'type': list,
        'default': [1.0],
     },
    'charge': {
        'type': list,
        'default': [1.0],
     },
    'numObservers': {
        'type': int,
        'default': 0,
        'minimum': 0,
        'maximum': 1000,
     },
    'obsR': {
        'type': list,
        'default': [0],
     },
    'obsTheta': {
        'type': list,
        'default': [0],
     },
    'obsPhi': {
        'type': list,
        'default': [0],
     },
    'idw_p': {
        'type': float,
        'default': 3.0,
        'minimum': _BASETYPES_H['VERYSMALL'],
        'maximum': _BASETYPES_H['BADVALUE'],
     },
    'masTriLinear': {
        'type': int,
        'default': 1,
        'minimum': 0,
        'maximum': 1,
     },
    'masCouple': {
        'type': int,
        'default': 0,
        'minimum': 0,
        'maximum': 1,
     },
    'masCorRotateFake': {
        'type': int,
        'default': 0,
        'minimum': 0,
        'maximum': 1,
     },
    'masHelCouple': {
        'type': int,
        'default': 0,
        'minimum': 0,
        'maximum': 1,
     },
    'masNumFiles': {
        'type': int,
        'default': 0,
        'minimum': 0,
        'maximum': 32767,
     },
    'masHelNumFiles': {
        'type': int,
        'default': 0,
        'minimum': 0,
        'maximum': 32767,
     },
    'useMasSteadyStateDt': {
        'type': int,
        'default': 1,
        'minimum': 0,
        'maximum': 1,
     },
    'masSteadyState': {
        'type': int,
        'default': 1,
        'minimum': 0,
        'maximum': 1,
     },
    'masDirectory': {
        'type': str,
        'default': ' ',
     },
    'masHelDirectory': {
        'type': str,
        'default': ' ',
     },
    'masDigits': {
        'type': int,
        'default': 3,
        'minimum': 0,
        'maximum': 32767,
     },
    'masHelDigits': {
        'type': int,
        'default': 3,
        'minimum': 0,
        'maximum': 32767,
     },
    'masCoupledTime': {
        'type': int,
        'default': 1,
        'minimum': 0,
        'maximum': 1,
     },
    'masStartTime': {
        'type': float,
        'default': 0.0,
        'minimum': 0.0,
        'maximum': _BASETYPES_H['BADVALUE'],
     },
    'epEquilibriumCalcDuration': {
        'type': float,
        'default': 0.0,
        'minimum': 0.0,
        'maximum': _BASETYPES_H['BADVALUE'],
     },
    'preEruptionDuration': {
        'type': float,
        'default': 0.0,
        'minimum': 0.0,
        'maximum': _BASETYPES_H['BADVALUE'],
     },
    'masRadialMin': {
        'type': float,
        'default': 0.0,
        'minimum': 0.0,
        'maximum': _BASETYPES_H['BADVALUE'],
     },
    'masRadialMax': {
        'type': float,
        'default': 0.0,
        'minimum': 0.0,
        'maximum': _BASETYPES_H['BADVALUE'],
     },
    'masHelRadialMin': {
        'type': float,
        'default': 0.0,
        'minimum': 0.0,
        'maximum': _BASETYPES_H['BADVALUE'],
     },
    'masHelRadialMax': {
        'type': float,
        'default': 0.0,
        'minimum': 0.0,
        'maximum': _BASETYPES_H['BADVALUE'],
     },
    'masVmin': {
        'type': float,
        'default': 50.0e5,
        'minimum': 0.0,
        'maximum': _BASETYPES_H['BADVALUE'],
     },
    'masInitFromOuterBoundary': {
        'type': int,
        'default': 2,
        'minimum': 0,
        'maximum': 2,
     },
    'masInitMonteCarlo': {
        'type': int,
        'default': 0,
        'minimum': 0,
        'maximum': 1,
     },
    'masInitRadius': {
        'type': float,
        'default': 0.0,
        'minimum': 0.0,
        'maximum': _BASETYPES_H['BADVALUE'],
     },
    'masInitTimeStep': {
        'type': float,
        'default': 0.000011574074074,
        'minimum': 0.0,
        'maximum': _BASETYPES_H['BADVALUE'],
     },
    'parallelFlow': {
        'type': float,
        'default': 0.0,
        'minimum': 0.0,
        'maximum': _BASETYPES_H['BADVALUE'],
     },
    'fieldAligned': {
        'type': int,
        'default': 0,
        'minimum': 0,
        'maximum': 1,
     },
    'clusterNodes': {
        'type': int,
        'default': 0,
        'minimum': 0,
        'maximum': 1,
     },
    'nodeClusterTheta': {
        'type': float,
        'default': 1.570796,
        'minimum': 0.0,
        'maximum': _BASETYPES_H['PI'],
     },
    'nodeClusterPhi': {
        'type': float,
        'default': 0.0,
        'minimum': 0.0,
        'maximum': 2.0 * _BASETYPES_H['PI'],
     },
    'nodeClusterWidth': {
        'type': float,
        'default': 0.0,
        'minimum': 0.0,
        'maximum': _BASETYPES_H['PI'],
     },
    'epCalcStartTime': {
        'type': float,
        'default': 0.0,
        'minimum': 0.0,
        'maximum': _BASETYPES_H['BADVALUE'],
     },
    'masRotateSolution': {
        'type': int,
        'default': 1,
        'minimum': 0,
        'maximum': 1,
     },
    'useBoundaryFunction': {
        'type': int,
        'default': 1,
        'minimum': 0,
        'maximum': 1,
     },
    'boundaryFunctionInitDomain': {
        'type': int,
        'default': 1,
        'minimum': 0,
        'maximum': 1,
     },
    'boundaryFunctAmplitude': {
        'type': float,
        'default': 1.0,
        'minimum': _BASETYPES_H['VERYSMALL'],
        'maximum': _BASETYPES_H['BADVALUE'],
     },
    'boundaryFunctXi': {
        'type': float,
        'default': 1.0,
        'minimum': 0.0,
        'maximum': _BASETYPES_H['BADVALUE'],
     },
    'boundaryFunctGamma': {
        'type': float,
        'default': 2.0,
        'minimum': 0.0,
        'maximum': _BASETYPES_H['BADVALUE'],
     },
    'boundaryFunctBeta': {
        'type': float,
        'default': 1.7,
        'minimum': 0.0,
        'maximum': _BASETYPES_H['BADVALUE'],
     },
    'boundaryFunctEcutoff': {
        'type': float,
        'default': 1.0,
        'minimum': 0.0,
        'maximum': _BASETYPES_H['BADVALUE'],
     },
    'shockSolver': {
        'type': int,
        'default': 0,
        'minimum': 0,
        'maximum': 1,
     },
    'shockDetectPercent': {
        'type': float,
        'default': 1.0,
        'minimum': 0.0,
        'maximum': _BASETYPES_H['BADVALUE'],
     },
    'minInjectionEnergy': {
        'type': float,
        'default': 0.01,
        'minimum': _BASETYPES_H['VERYSMALL'],
        'maximum': _BASETYPES_H['BADVALUE'],
     },
    'shockInjectionFactor': {
        'type': float,
        'default': 1.0,
        'minimum': 0.0,
        'maximum': _BASETYPES_H['BADVALUE'],
     },
    'idealShock': {
        'type': int,
        'default': 0,
        'minimum': 0,
        'maximum': 1,
     },
    'idealShockSharpness': {
        'type': float,
        'default': 1.0,
        'minimum': _BASETYPES_H['VERYSMALL'],
        'maximum': _BASETYPES_H['BADVALUE'],
     },
    'idealShockScaleLength': {
        'type': float,
        'default': 0.0046491,
        'minimum': _BASETYPES_H['VERYSMALL'],
        'maximum': _BASETYPES_H['BADVALUE'],
     },
    'idealShockJump': {
        'type': float,
        'default': 4.0,
        'minimum': _BASETYPES_H['VERYSMALL'],
        'maximum': _BASETYPES_H['BADVALUE'],
     },
    'idealShockSpeed': {
        'type': float,
        'default': 1500e5,
        'minimum': _BASETYPES_H['VERYSMALL'],
        'maximum': _BASETYPES_H['BADVALUE'],
     },
    'idealShockInitTime': {
        'type': float,
        'default': 0.0,
        'minimum': 0.0,
        'maximum': _BASETYPES_H['BADVALUE'],
     },
    'idealShockTheta': {
        'type': float,
        'default': 1.570796,
        'minimum': 0.0,
        'maximum': _BASETYPES_H['PI'],
     },
    'idealShockPhi': {
        'type': float,
        'default': 0.0,
        'minimum': 0.0,
        'maximum': 2.0 * _BASETYPES_H['PI'],
     },
    'idealShockWidth': {
        'type': float,
        'default': 0.0,
        'minimum': 0.0,
        'maximum': _BASETYPES_H['PI'],
     },
    'dumpFreq': {
        'type': int,
        'default': 1,
        'minimum': 0,
        'maximum': 1000000,
     },
    'outputRestart': {
        'type': int,
        'default': 0,
        'minimum': 0,
        'maximum': 1000000,
     },
    'dumpOnAbort': {
        'type': int,
        'default': 0,
        'minimum': 0,
        'maximum': 1,
     },
    'saveRestartFile': {
        'type': int,
        'default': 0,
        'minimum': 0,
        'maximum': 1,
     },
    'warningsFile': {
        'type': str,
        'default': 'warningsXXX.txt',
     },
}