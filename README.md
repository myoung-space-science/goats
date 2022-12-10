# GOATS

A set of tools for analyzing heliophysical datasets

The Generalized Observing Application Tool Suite (GOATS) is a collection of objects that support interactive and scripted analysis of simulated and observed data in heliophysics.

## Installation

```bash
$ pip install goats
```

## Usage Example: EPREM

The [Energetic Particle Radiation Environment Module](https://github.com/myoung-space-science/eprem) (EPREM) simulates acceleration and transport of energetic particles throughout the heliosphere by modeling the focused transport equation on a Lagragian grid in the frame co-moving with the solar wind. It was the primary motivation for developing this package.

The first thing we'll do is import the packages we need.

* `pathlib` is the built-in package for working with system paths  
* `matplotlib` is a popular third-party package for creating figures  
* `eprem` is the GOATS subpackage for working with EPREM output  


```python
import pathlib

import matplotlib.pyplot as plt

from goats import eprem
```

Next, we'll create a stream observer, which is the type of observer that corresponds to an EPREM output file with a name like `obs######.nc`. This invokation assumes that the data file, as well as an EPREM runtime parameter file called `eprem_input_file`, are in a local subdirectory called `data`.

Note that if the data file is in the current directory, you can omit `source=<path/to/data>`.


```python
stream = eprem.Stream(350, source='data/example', config='eprem_input_file')
```

We can request the value of simulation runtime parameters by aliased keyword. For example, let's check the assumed mean free path at 1 au.


```python
print(stream['lambda0'])
```

    'lam0 | lambda0 | lamo': [1.] [au]


The text tells us that this simulation run used a value of 1.0 au (astronomical unit) for this parameter. It also suggests that we could have requested this value by the keywords 'lamo' or 'lam0'.


```python
print(stream['lamo'])
print(stream['lam0'])
```

    'lam0 | lambda0 | lamo': [1.] [au]
    'lam0 | lambda0 | lamo': [1.] [au]


We can also request observable quantities by aliased keyword. Here is the radial velocity.


```python
vr = stream['Vr']
print(vr)
```

    Observable('Vr', unit='m s^-1')


The text tells us that the radial velocity output array has a time axis and a shell axis. EPREM shells are logical surface of nodes in the Lagrangian grid. Each shell index along a given stream represents one node. We can observe radial velocity at a single time (e.g., 1 hour of real time since simulation start) on a single node as follows:


```python
t0 = 1.0, 'hour'
vr.observe(time=t0, shell=1000)
```




    Observation(unit='m s^-1', dimensions=['time', 'shell'])



In the case of a constant isotropic solar wind, the stream nodes would extend radially outward from the Sun; with some trial-and-error, we could figure out which shell is closest to a particular radius (e.g., 1 au).

Instead, we often want to interpolate an observation to the radius of interest.


```python
observed = vr.observe(radius=[0.1, 'au'])
```

Now that we have an observation of the radial velocity at 0.1 au as a function of time, we can plot it. First, we'll define intermediate variables to hold the time in hours and the radial velocity in kilometers per second.


```python
time = observed['time']
```

Next, we'll make sure there's a `figures` directory (to avoid cluttering the current directory) and load the plotting library.


```python
figpath = pathlib.Path('figures').resolve()
figpath.mkdir(exist_ok=True)
```

Finally, we'll create and save the plot.


```python
plt.plot(time['hour'], observed['km / s'].array)
plt.xlabel('Time [hours]')
plt.ylabel('Vr [km/s]')
plt.savefig(figpath / 'vr-hours.png')
```


    
![png](readme_files/readme_25_0.png)
    


There are many other observable quantities available to an observer, and they are not limited to those in the observer's source data.


```python
print('flux' in stream.observables)
print('mean free path' in stream.observables)
```

    True
    True



```python
stream['flux']
```




    Observable('flux', unit='J^-1 s^-1 sr^-1 m^-2')




```python
stream['mean free path']
```




    Observable('mean free path', unit='m')



We can even create observable quantities by symbolically composing existing observable quantities


```python
stream['mfp / Vr']
```




    Observable('mfp / Vr', unit='s')




```python
stream['rho * energy']
```




    Observable('rho * energy', unit='kg m^-1 s^-2')



Note that the unit is consistent with the composed quantity and that the axes of the composed quantity represent the union of the axes of the component quantities.

To illustrate full use of a composed quantity, consider observing the ratio of the mean free path of protons with 1 and 5 MeV to the radial velocity of the solar wind.


```python
observed = stream['mfp / Vr'].observe(radius=[0.1, 'au'], energy=[1, 5, 'MeV'])
lines = plt.plot(observed['time']['hour'], observed.array)
lines[0].set_label('1 MeV')
lines[1].set_label('5 MeV')
plt.xlabel('Time [hours]')
plt.ylabel('mfp / Vr [s]')
plt.legend()
plt.savefig(figpath / 'mfp_vr-hours.png')
```


    
![png](readme_files/readme_35_0.png)
    


## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`goats` was created by Matt Young. It is licensed under the terms of the GNU General Public License v3.0 license.

## Credits

`goats` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).

