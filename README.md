# ED-AFM
Electrostatic Discovery Atomic Force Microscopy

[Link to paper](https://arxiv.org/abs/2108.04333) (Paper still pending peer review.)

Abstract:
_While offering unprecedented resolution of atomic and electronic structure, Scanning Probe Microscopy techniques have found greater challenges in providing reliable electrostatic characterization at the same scale. In this work, we introduce Electrostatic Discovery Atomic Force Microscopy, a machine learning based method which provides immediate quantitative maps of the electrostatic potential directly from Atomic Force Microscopy images with functionalized tips. We apply this to characterize the electrostatic properties of a variety of molecular systems and compare directly to reference simulations, demonstrating good agreement. This approach opens the door to reliable atomic scale electrostatic maps on any system with minimal computational overhead._

![Method schematic](/figures/method_schem.png)

This repository serves as an archive of the code and data used to achieve the results in the above paper, and it is not subject to any further development. However, if there is a broken link etc., do raise an issue to let us know.

## Setup

The code here is written for Python 3.8 or higher. All the main code is in the `edafm` directory. Make sure it is on the Python PATH when running any of the scripts. With Anaconda/Miniconda, you can install the required environment using the provided `environment.yml` file:
```sh
conda env create -f environment.yml
conda activate edafm
```
Otherwise take a look inside the `environment.yml` file to see which packages are required. Note that we specify version 1.7.0 of pytorch because we had some problems in data loading with newer versions on our systems, but other versions could work for you. We also provide the file `environment_exact.yml` that specifies the exact python environment that we used in fitting the model.

For AFM simulations we need the [ProbeParticleModel](https://github.com/ProkopHapala/ProbeParticleModel) AFM simulation code. Unfortunately, the GPU version of the code that we use does not have a stable release, so we instead clone a specific commit of the github repository:
```sh
git clone https://github.com/ProkopHapala/ProbeParticleModel.git
cd ProbeParticleModel
git checkout 363709ad3ed191422bfdcafea9ad6135e7906972
```
If you clone the ProbeParticleModel repository to the root of this repository, then all the scripts here should work as they are. Otherwise, make sure the ProbeParticleModel directory is on Python PATH when running the scripts.

## ML model

We use a U-net type convolutional neural network with attention gates in the skip connections. Similar model was used previously by Oktay et al. for segmenting medical images (https://arxiv.org/abs/1804.03999v2).

![Model schematic](/figures/model_schem.png)
![AG schematic](/figures/AG_schem.png)

Our implementation of the model in PyTorch can be found in this repository: `edafm/models.py` contains the model definition and `edafm/layers.py` contains definitions of layers and layer blocks used by the model. In `edafm/models.py`, two modules can be found: `AttentionUNet`, which is the generic version of the model, and `EDAFMNet`, which is a subclass of the former specifying the exact hyperparameters for the model that we used.

In `EDAFMNet` one can also specify pretrained weights of several types to download to the model using the `trained_weights` argument:

 - 'base': The base model used for all predictions in the main ED-AFM paper and used for comparison in the various test in the supplementary information of the paper.
 - 'single-channel': Model trained on only a single CO-tip AFM input.
 - 'CO-Cl': Model trained on alternative tip combination of CO and Cl.
 - 'Xe-Cl': Model trained on alternative tip combination of Xe and Cl.
 - 'constant-noise': Model trained using constant noise amplitude instead of normally distributed amplitude.
 - 'uniform-noise': Model trained using uniform random noise amplitude instead of normally distributed amplitude.
 - 'no-gradient': Model trained without background-gradient augmentation.
 - 'matched-tips': Model trained on data with matched tip distance between CO and Xe, instead of independently randomized distances.

The model weights can also be downloaded directly from https://www.dropbox.com/sh/1jd4h7ovpafg3uk/AACQX6C6Nks6RxGbvqb6vIo-a?dl=0. The weights are saved in the state_dict format of PyTorch.

## Data and model training

We don't provide the full training/validation/test sets for download because they are very large (~1TiB in total). Instead, we provide the database of molecular geometries that can be used to generate the full dataset using ProbeParticleModel. The provided script `generate_data.py` does the data generation and will download the molecule database automatically. Alternatively, the molecule database can be downloaded directly from https://www.dropbox.com/s/z4113upq82puzht/Molecules_rebias_210611.tar.gz?dl=0. 

The model training can be done using the provided script `train.py`. Note that performing the training using all the same settings as we used requires a significant amount of time and also a significant amount VRAM on the GPU, likely more than can be found on a single GPU. In our case the model training took ~5 days using 4 x Nvidia Tesla V100 (32GB) GPUs. However, inference on the trained model can be done even on a single lower-end GPU or on CPU.

All the data used for the predictions in the paper can be found under the `data` directory.

## Figures

The scripts used to generate most of the figures in the paper are provided under the directory `figures`. The scripts correspond to the figures as follows:

 - Fig. 2: sims.py
 - Fig. 3: ptcda.py
 - Fig. 4: bcb.py
 - Fig. 5: bcb_surface_sim.py
 - Fig. S1: model_schem.tex
 - Fig. S3: stats.py\*
 - Fig. S4: esmap_sample.py and then esmap_schem.tex
 - Fig. S5: afm_stacks.py
 - Fig. S6: sims_hartree.py
 - Fig. S7: ptcda_surface_sim.py
 - Fig. S8: single_tip.py
 - Fig. S9: sims_Cl.py
 - Fig. S10: height_dependence.py
 - Fig. S11: extra_electron.py
 - Fig. S12: background_gradient.py

\* Precalculated MSE values used by the plotting script are provided under `figures/stats`. The scripts used to calculate these values are also under `figures/stats`.

You can also use `run_all.sh`-script to run all of the scripts in one go. Note that compiling the .tex files additionally requires a working LaTex installation on your system.
