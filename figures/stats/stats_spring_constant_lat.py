
import os
import sys
import glob
import time
import random
import numpy as np
import matplotlib.pyplot as plt

import torch

sys.path.append('../../ProbeParticleModel')
from pyProbeParticle import oclUtils     as oclu
from pyProbeParticle import fieldOCL     as FFcl
from pyProbeParticle import RelaxOpenCL  as oclr
from pyProbeParticle import AuxMap       as aux
from pyProbeParticle.AFMulatorOCL_Simple    import AFMulator
from pyProbeParticle.GeneratorOCL_Simple2   import InverseAFMtrainer

sys.path.append('../..')
import edafm.preprocessing as pp
from edafm.models import EDAFMNet
from edafm.utils import download_molecules
from edafm.visualization import plot_input

# Set random seed for reproducibility
random.seed(0)

model_type      = 'base'                            # Type of pretrained weights to use
save_file       = 'mse_spring_constants_lat.csv'    # File to save MSE values into
device          = 'cuda'                            # Device to run inference on
molecules_dir   = '../../molecules'                 # Path to molecule database
sim_save_dir    = './test_sim_lat'                  # Directory where to save test simulations for inspection
test_constants  = np.linspace(0.2, 0.3, 21)         # Test lateral spring constants to run
n_samples       = 3000                              # Number of samples to run

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
    'iZPPs'         : [8, 54],
    'Qs'            : [[ -10, 20,  -10, 0 ], [  30, -60,   30, 0 ]],
    'QZs'           : [[ 0.1,  0, -0.1, 0 ], [ 0.1,   0, -0.1, 0 ]]
}

def apply_preprocessing(batch):
    Xs, Ys, xyzs = batch
    Xs = [x[...,2:8] for x in Xs]
    pp.add_norm(Xs)
    pp.add_noise(Xs, c=0.08, randomize_amplitude=False)
    return Xs, Ys

# Initialize OpenCL environment on GPU
env = oclu.OCLEnvironment( i_platform = 0 )
FFcl.init(env)
oclr.init(env)

# Define AFMulator
afmulator = AFMulator(**afmulator_args)
afmulator.npbc = (0,0,0)

# Define AuxMaps
aux_maps = [
    aux.ESMapConstant(
        scan_dim    = afmulator.scan_dim[:2],
        scan_window = [afmulator.scan_window[0][:2], afmulator.scan_window[1][:2]],
        height      = 4.0,
        vdW_cutoff  = -2.0,
        Rpp         = 1.0
    )
]

# Download molecules if not already there
download_molecules(molecules_dir, verbose=1)

# Define generator
xyz_paths = glob.glob(os.path.join(molecules_dir, 'test/*.xyz'))
trainer = InverseAFMtrainer(afmulator, aux_maps, xyz_paths, **generator_kwargs)

# Pick samples
random.shuffle(trainer.molecules)
trainer.molecules = trainer.molecules[:n_samples]

# Make model
model = EDAFMNet(device=device, trained_weights=model_type, weights_dir='../weights')

# Initialize save file
if os.path.exists(save_file):
    raise RuntimeError('Save file already exists')
with open(save_file, 'w') as f:
    pass

if not os.path.exists(sim_save_dir):
    os.makedirs(sim_save_dir)

# Calculate MSE at every height for every batch
start_time = time.time()
total_len = len(test_constants)*len(trainer)
for ih, k_lat in enumerate(test_constants):

    print(f'Lateral spring constant = {k_lat:.3f}')
    afmulator.scanner.stiffness = np.array([k_lat, k_lat, 0.0, 30.0], dtype=np.float32) / -16.0217662

    mses = []
    for ib, batch in enumerate(trainer):

        if ib == 0:
            print('Saving example simulations...')
            for s in range(10):
                fig = plot_input(batch[0][0][s])
                plt.savefig(os.path.join(sim_save_dir, f'sim{s}_klat_{k_lat:.3f}.png'))
                plt.close()

        X, ref = apply_preprocessing(batch)
        X = [torch.from_numpy(x).unsqueeze(1).to(device) for x in X]
        ref = [torch.from_numpy(r).to(device) for r in ref]

        with torch.no_grad():
            pred = model(X)[0]

        diff = pred - ref[0]
        for d in diff:
            mses.append((d**2).mean().cpu().numpy())
        
        eta = (time.time() - start_time) * (total_len / (ih*len(trainer)+ib+1) - 1)
        print(f'Batch {ib+1}/{len(trainer)} - ETA: {eta:.1f}s')

    with open(save_file, 'a') as f:
        f.write(f'{k_lat:.3f},')
        f.write(','.join([str(v) for v in mses]))
        f.write('\n')
