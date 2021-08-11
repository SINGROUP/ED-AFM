
import os
import sys
import glob
import time
import random
import numpy as np

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

# Set random seed for reproducibility
random.seed(0)

# # Normal noise model
# model_type      = 'base'                    # Type of pretrained weights to use
# save_file       = 'mse_normal.csv'          # File to save MSE values into

# # Uniform noise model
# model_type      = 'uniform-noise'           # Type of pretrained weights to use
# save_file       = 'mse_uniform.csv'         # File to save MSE values into

# Constant noise model
model_type      = 'constant-noise'          # Type of pretrained weights to use
save_file       = 'mse_constant.csv'        # File to save MSE values into

device          = 'cuda'                    # Device to run inference on
molecules_dir   = '../../molecules'         # Path to molecule database
test_amplitudes = np.linspace(0, 0.2, 21)   # Test amplitudes to run
n_samples       = 3000                      # Number of samples to run

if os.path.exists(save_file):
    raise RuntimeError('Save file already exists')

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

# Calculate MSE at every height for every batch
start_time = time.time()
total_len = len(test_amplitudes)*len(trainer)
mses = [[] for _ in range(len(test_amplitudes))]
for ib, batch in enumerate(trainer):

    X, ref, _ = batch
    X = [x[...,2:8] for x in X]
    pp.add_norm(X)
    ref = [torch.from_numpy(r).to(device) for r in ref]

    for ia, noise_amp in enumerate(test_amplitudes):

        X_ = [x.copy() for x in X]
        pp.add_noise(X_, c=noise_amp, randomize_amplitude=False)
        X_ = [torch.from_numpy(x).unsqueeze(1).to(device) for x in X_]

        with torch.no_grad():
            pred = model(X_)[0]

        diff = pred - ref[0]
        for d in diff:
            mses[ia].append((d**2).mean().cpu().numpy())

    eta = (time.time() - start_time) * (len(trainer) / (ib+1) - 1)
    print(f'Batch {ib+1}/{len(trainer)} - ETA: {eta:.1f}s')

with open(save_file, 'w') as f:
    for noise_amp, mse_amp in zip(test_amplitudes, mses):
        f.write(f'{noise_amp:.2f},')
        f.write(','.join([str(v) for v in mse_amp]))
        f.write('\n')
