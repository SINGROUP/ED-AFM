# ED-AFM
Electrostatic Discovery Atomic Force Microscopy

![Schematic](/figures/data_flow.png)

[Reference](https://ed-afm.readthedocs.io/en/latest/index.html)

## Setup

The code here is written for Python 3.8 or higher. All the main code is in the `edafm` directory. Make sure it is on the Python PATH when running any of the scripts. With Anaconda, you can install the required environment using the provided `environment.yml` file:
```sh
conda env create -f environment.yml
conda activate edafm
```
Otherwise take a look inside the `environment.yml` file to see which packages are required.

For AFM simulations we need the [ProbeParticleModel](https://github.com/ProkopHapala/ProbeParticleModel) AFM simulation code. Unfortunately, the GPU version of the code that we use does not have a stable release, so we instead clone a specific commit of the github repository:
```sh
git clone https://github.com/ProkopHapala/ProbeParticleModel.git
cd ProbeParticleModel
git checkout 64154be5ef059a8dc6549f330bea9bd77b0758c9
```
If you clone the ProbeParticleModel repository to the root of this repository, then all the scripts here should work as they are. Otherwise, make sure the ProbeParticleModel directory is on Python PATH when running the scripts.

