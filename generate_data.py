

import os
import sys
import time
import numpy as np

sys.path.append('./ProbeParticleModel') # Make sure ProbeParticleModel is on path
from pyProbeParticle import oclUtils     as oclu
from pyProbeParticle import fieldOCL     as FFcl
from pyProbeParticle import RelaxOpenCL  as oclr
from pyProbeParticle import common       as PPU
from pyProbeParticle import basUtils
from pyProbeParticle import AuxMap       as aux
from pyProbeParticle.AFMulatorOCL_Simple    import AFMulator
from pyProbeParticle.GeneratorOCL_Simple2   import InverseAFMtrainer

from edafm.common_utils import download_molecules

class Trainer(InverseAFMtrainer):

    def on_sample_start(self):
        self.randomize_distance(delta=0.25)
        self.randomize_tip(max_tilt=0.5)

# Options
molecules_dir   = './molecules'     # Where to save molecule database
# save_dir        = './data'          # Where to save training data
save_dir        = '/media/storage/ES_data'          # Where to save training data

# Initialize OpenCL environment on GPU
env = oclu.OCLEnvironment( i_platform = 0 )
FFcl.init(env)
oclr.init(env)

afmulator_args = {
    'pixPerAngstrome'   : 20,
    'lvec'              : np.array([
                            [ 0.0,  0.0, 0.0],
                            [20.0,  0.0, 0.0],
                            [ 0.0, 20.0, 0.0],
                            [ 0.0,  0.0, 5.0]
                            ]),
    'scan_dim'          : (128, 128, 20),
    'scan_window'       : ((2.0, 2.0, 6.0), (18.0, 18.0, 8.0)),
    'amplitude'         : 1.0,
    'df_steps'          : 10,
    'initFF'            : True
}

generator_kwargs = {
    'batch_size'    : 30,
    'distAbove'     : 5.3,

     # Xe, CO
    'iZPPs'         : [54, 8],
    'Qs'            : [[  30, -60,   30, 0 ], [ -10, 20,  -10, 0 ]],
    'QZs'           : [[ 0.1,   0, -0.1, 0 ], [ 0.1,  0, -0.1, 0 ]]

    # Cl, CO
    # 'iZPPs'         : [17, 8],
    # 'Qs'            : [[ -0.3, 0, 0, 0 ], [ -10, 20,  -10, 0 ]],
    # 'QZs'           : [[    0, 0, 0, 0 ], [ 0.1,  0, -0.1, 0 ]]

    # Xe, Cl
    # 'iZPPs'         : [54, 17],
    # 'Qs'            : [[  30, -60,   30, 0 ], [ -0.3, 0, 0, 0 ]],
    # 'QZs'           : [[ 0.1,   0, -0.1, 0 ], [    0, 0, 0, 0 ]]
    
}

# Define AFMulator
afmulator = AFMulator(**afmulator_args)
afmulator.npbc = (0,0,0)

# Define AuxMaps
aux_maps = [
    aux.ESMap(
        scanner = afmulator.scanner,
        zmin    = -2.0,
        iso     = 0.1
    )
]

# Rotations
rotations = PPU.sphereTangentSpace(n=100)
n_best_rotations = 30

# Number of molecules in each dataset
# N_train = 4728   # Number of training molecules
# N_val   = 600    # Number of validation molecules
# N_test  = 1000   # Number of test molecules
N_train = 47   # Number of training molecules
N_val   = 6    # Number of validation molecules
N_test  = 10   # Number of test molecules

# Paths to molecule xyz files
database_dir = os.path.join(molecules_dir, 'heavy')
train_paths  = [os.path.join(database_dir, f'{n}.xyz') for n in range(N_train)]
val_paths    = [os.path.join(database_dir, f'{n}.xyz') for n in range(N_train,N_train+N_val)]
test_paths   = [os.path.join(database_dir, f'{n}.xyz') for n in range(N_train+N_val,N_train+N_val+N_test)]

# Download molecules if not already there
download_molecules(molecules_dir, verbose=1)

# Make sure save directory exists
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

start_time = time.time()
counter = 1
total_len = np.floor((len(train_paths)+len(val_paths)+len(test_paths))*n_best_rotations/generator_kwargs['batch_size'])
for mode, paths in zip(['train', 'val', 'test'], [train_paths, val_paths, test_paths]):

    # Define generator
    trainer = Trainer(afmulator, aux_maps, paths, **generator_kwargs)

    # Augment with rotations and shuffle
    trainer.augment_with_rotations_entropy(rotations, n_best_rotations=n_best_rotations)

    # Make sure target directory exists
    target_dir = os.path.join(save_dir, mode)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Generate batches
    for i, batch in enumerate(trainer):
        np.savez(os.path.join(target_dir, f'batch_{i}.npz'), batch)
        eta = (time.time() - start_time)/counter * (total_len - counter)
        print(f'Generated {mode} batch {i+1}/{len(trainer)} - ETA: {eta:.1f}s')
        counter += 1

print(f'Total time taken: {time.time() - start_time:.1f}s')
