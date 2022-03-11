"""
This script creates a netCDF file for testing `core.datasets`. It originated from the tutorial at https://unidata.github.io/netcdf4-python/
"""

import argparse
import typing

import netCDF4


DIMENSIONS = {
    'eprem': {
        'time': 20,
        'shell': 100,
        'species': 2,
        'energy': 10,
        'mu': 5,
    },
}


VARIABLES = {
    'eprem': {
        'time': {
            'type': 'f4',
            'dimensions': ('time',),
            'units': 's',
        },
        'Vr': {
            'type': 'f8',
            'dimensions': ('time', 'shell'),
            'units': 'km / s',
        },
        'flux': {
            'type': 'f8',
            'dimensions': ('time', 'shell', 'species', 'energy'),
            'units': '# / cm^2 s sr MeV',
        },
        'Dist': {
            'type': 'f8',
            'dimensions': ('time', 'shell', 'species', 'energy', 'mu'),
            'units': 's^3 / km^6',
        },
    },
}


DATASETS = {
    'eprem-flux': {
        'dimensions': DIMENSIONS['eprem'].copy(),
        'variables': {
            name: attrs for name, attrs in VARIABLES['eprem'].items()
            if name != 'Dist'
        },
    },
    'eprem-obs': {
        'dimensions': DIMENSIONS['eprem'].copy(),
        'variables': {
            name: attrs for name, attrs in VARIABLES['eprem'].items()
            if name != 'flux'
        },
    },
    'basic': {
        'dimensions': {
            'level': None,
            'time': None,
            'lat': 73,
            'lon': 144,
        },
        'variables': {
            'level': {
                'type': 'i4',
                'dimensions': ('level',),
            },
            'time': {
                'type': 'f8',
                'dimensions': ('time',),
            },
            'lat': {
                'type': 'f4',
                'dimensions': ('lat',),
            },
            'lon': {
                'type': 'f4',
                'dimensions': ('lon',),
            },
            'temp': {
                'type': 'f4',
                'dimensions': ('time', 'level', 'lat', 'lon'),
                'units': 'K',
            },
        }
    }
}


def main(
    names: typing.Iterable[str]=None,
    verbose: int=0,
) -> None:
    """Create test datasets for the GOATS package."""
    datasets = {
        f"{name}.nc": DATASETS[name]
        for name in names or DATASETS.keys()
    }
    for filename, dataset in datasets.items():
        create_dataset(filename, dataset, verbose=verbose)


def create_dataset(
    filename: str,
    dataset: typing.Dict[str, typing.Any],
    verbose: int=0,
) -> None:
    """Create the named dataset, if possible."""
    if verbose:
        print(f"\n=== Creating {filename} dataset ===")
    rootgrp = netCDF4.Dataset(filename, 'w', format='NETCDF4')
    if verbose > 1:
        print(rootgrp.data_model)
    for name, size in dataset['dimensions'].items():
        rootgrp.createDimension(name, size)
    if verbose > 1:
        print(rootgrp.dimensions)
    for name, attrs in dataset['variables'].items():
        args = [attrs[key] for key in ('type', 'dimensions')]
        var = rootgrp.createVariable(name, *args)
        if 'units' in attrs:
            var.units = attrs['units']
    if verbose > 1:
        print(rootgrp.variables)
    rootgrp.close()
    if verbose:
        print("=== Done ===", end='\n\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description=main.__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        '-n',
        '--name',
        dest='names',
        help=(
            "The name(s) of test datasets to create."
            "\nThe default behavior is to create all datasets."
        ),
        nargs='*',
        choices=('basic', 'eprem-flux', 'eprem-obs'),
    )
    p.add_argument(
        '-v',
        '--verbose',
        help=(
            "Print runtime messages."
            "\nPass multiple times to increase verbosity."
        ),
        action='count',
        default=0,
    )
    args = p.parse_args()
    main(**vars(args))
