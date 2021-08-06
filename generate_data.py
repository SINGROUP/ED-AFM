

import os
import sys
import time
import glob
import numpy as np

sys.path.append('./ProbeParticleModel') # Make sure ProbeParticleModel is on PATH
from pyProbeParticle import oclUtils     as oclu
from pyProbeParticle import fieldOCL     as FFcl
from pyProbeParticle import RelaxOpenCL  as oclr
from pyProbeParticle import AuxMap       as aux
from pyProbeParticle.AFMulatorOCL_Simple    import AFMulator
from pyProbeParticle.GeneratorOCL_Simple2   import InverseAFMtrainer

from edafm.utils import download_molecules

class Trainer(InverseAFMtrainer):

    def on_afm_start(self):
        # Use different lateral stiffness for Cl than CO and Xe
        if self.afmulator.iZPP in [8, 54]:
            afmulator.scanner.stiffness = np.array([0.25, 0.25, 0.0, 30.0], dtype=np.float32) / -16.0217662
        elif self.afmulator.iZPP == 17:
            afmulator.scanner.stiffness = np.array([0.5, 0.5, 0.0, 30.0], dtype=np.float32) / -16.0217662
        else:
            raise RuntimeError(f'Unknown tip {self.afmulator.iZPP}')

    def handle_distance(self):
        '''
        Set correct distance from scan region for the current molecule.
        '''
        self.randomize_distance(delta=0.25)
        self.randomize_tip(max_tilt=0.5)
        RvdwPP = self.afmulator.typeParams[self.afmulator.iZPP-1][0]
        Rvdw = self.REAs[:,0] - RvdwPP
        zs = self.xyzs[:,2]
        imax = np.argmax(zs + Rvdw)
        total_distance = self.distAboveActive + Rvdw[imax] + RvdwPP - (zs.max() - zs[imax])
        self.xyzs[:,2] += (self.afmulator.scan_window[1][2] - total_distance) - zs.max()

# Options
molecules_dir   = './molecules'     # Where to save molecule database
save_dir        = './data'          # Where to save training data (Note: takes ~1TiB of disk space)

# Initialize OpenCL environment on GPU
env = oclu.OCLEnvironment( i_platform = 0 )
FFcl.init(env)
oclr.init(env)

afmulator_args = {
    'pixPerAngstrome'   : 20,
    'lvec'              : np.array([
                            [ 0.0,  0.0, 0.0],
                            [28.0,  0.0, 0.0],
                            [ 0.0, 28.0, 0.0],
                            [ 0.0,  0.0, 5.0]
                            ]),
    'scan_dim'          : (192, 192, 20),
    'scan_window'       : ((2.0, 2.0, 6.0), (26.0, 26.0, 8.0)),
    'amplitude'         : 1.0,
    'df_steps'          : 10,
    'initFF'            : True
}

generator_kwargs = {
    'batch_size'    : 30,
    'distAbove'     : 5.3,
    'iZPPs'         : [8, 54, 17], # CO, Xe, Cl
    'Qs'            : [[ -10,  20,  -10, 0 ], [  30, -60,  30, 0 ], [ -0.3, 0, 0, 0 ]],
    'QZs'           : [[ 0.1,   0, -0.1, 0 ], [ 0.1,  0, -0.1, 0 ], [    0, 0, 0, 0 ]]
}

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

# Paths to molecule xyz files
train_paths  = glob.glob(os.path.join(molecules_dir, 'train/*.xyz'))
val_paths    = glob.glob(os.path.join(molecules_dir, 'validation/*.xyz'))
test_paths   = glob.glob(os.path.join(molecules_dir, 'test/*.xyz'))

# Make sure save directory exists
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

start_time = time.time()
counter = 1
total_len = np.floor((len(train_paths)+len(val_paths)+len(test_paths))/generator_kwargs['batch_size'])
for mode, paths in zip(['train', 'val', 'test'], [train_paths, val_paths, test_paths]):

    # Define generator
    trainer = Trainer(afmulator, aux_maps, paths, **generator_kwargs)

    # Shuffle
    trainer.shuffle_molecules()

    # Make sure target directory exists
    target_dir = os.path.join(save_dir, mode)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Generate batches
    for i, batch in enumerate(trainer):
        np.savez(os.path.join(target_dir, f'batch_{i}.npz'), np.array(batch, dtype=object))
        eta = (time.time() - start_time)/counter * (total_len - counter)
        print(f'Generated {mode} batch {i+1}/{len(trainer)} - ETA: {eta:.1f}s')
        counter += 1

print(f'Total time taken: {time.time() - start_time:.1f}s')
