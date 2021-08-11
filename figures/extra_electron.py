
import os
import sys
import imageio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.ndimage import rotate, shift

import torch

sys.path.append('../ProbeParticleModel')
from pyProbeParticle import oclUtils     as oclu
from pyProbeParticle import fieldOCL     as FFcl
from pyProbeParticle import RelaxOpenCL  as oclr
from pyProbeParticle import AuxMap       as aux
from pyProbeParticle.AFMulatorOCL_Simple    import AFMulator
from pyProbeParticle.GeneratorOCL_Simple2   import InverseAFMtrainer

sys.path.append('..')
import edafm.preprocessing as pp

# # Set matplotlib font
# from matplotlib import rc
# rc('font', family = 'serif', serif = 'cmr10')

class Trainer(InverseAFMtrainer):

    # Override this method to set the Xe tip closer
    def handle_distance(self):
        if self.afmulator.iZPP == 54:
            self.distAboveActive -= 0.4
        super().handle_distance()
    
    # Override position handling to center on the non-Cu atoms
    def handle_positions(self):
        sw = self.afmulator.scan_window
        scan_center = np.array([sw[1][0] + sw[0][0], sw[1][1] + sw[0][1]]) / 2
        self.xyzs[:,:2] += scan_center - self.xyzs[self.Zs != 29,:2].mean(axis=0)

def apply_preprocessing(batch):

    X, Y, xyzs = batch

    X = [x[..., 2:8] for x in X]

    pp.add_norm(X)
    np.random.seed(0)
    pp.add_noise(X, c=0.05)

    return X, Y, xyzs

data_dir    = '../data'         # Path to data
X_slices    = [0, 3, 5]         # Which AFM slices to plot
tip_names   = ['CO', 'Xe']      # AFM tip types
fig_width   = 100               # Figure width in mm
fontsize    = 8
dpi         = 300

# Initialize OpenCL environment on GPU
env = oclu.OCLEnvironment( i_platform = 0 )
FFcl.init(env)
oclr.init(env)

afmulator_args = {
    'pixPerAngstrome'   : 20,
    'lvec'              : np.array([
                            [ 0.0,  0.0, 0.0],
                            [22.0,  0.0, 0.0],
                            [ 0.0, 22.0, 0.0],
                            [ 0.0,  0.0, 7.0]
                            ]),
    'scan_dim'          : (144, 104, 20),
    'scan_window'       : ((2.0, 2.0, 7.0), (20.0, 15.0, 9.0)),
    'amplitude'         : 1.0,
    'df_steps'          : 10,
    'initFF'            : True
}

generator_kwargs = {
    'batch_size'    : 1,
    'distAbove'     : 4.8,
    'iZPPs'         : [8, 54],
    'Qs'            : [[ -10, 20,  -10, 0 ], [  30, -60,   30, 0 ]],
    'QZs'           : [[ 0.1,  0, -0.1, 0 ], [ 0.1,   0, -0.1, 0 ]]
}

# Paths to molecule xyz files
molecules = [os.path.join(data_dir, 'PTCDA/mol-1.xyz')]

# Define AFMulator
afmulator = AFMulator(**afmulator_args)
afmulator.npbc = (0,0,0)

# Define generator
trainer = Trainer(afmulator, [], molecules, **generator_kwargs)

# Get simulation data
X, _, _ = apply_preprocessing(next(iter(trainer)))

# Create figure grid
fig_width = 0.1/2.54*fig_width
fig = plt.figure(figsize=(fig_width, 0.49*fig_width))
axes = fig.add_gridspec(len(X), len(X_slices), wspace=0.02, hspace=0.02).subplots(squeeze=False)

# Plot AFM
for i, x in enumerate(X):
    for j, s in enumerate(X_slices):
        
        # Plot AFM slice
        im = axes[i, j].imshow(x[0,:,:,s].T, origin='lower', cmap='afmhot')
        axes[i, j].set_axis_off()
    
    # Put tip names to the left of the AFM image rows
    axes[i, 0].text(-0.1, 0.5, tip_names[i], horizontalalignment='center',
        verticalalignment='center', transform=axes[i, 0].transAxes,
        rotation='vertical', fontsize=fontsize)

plt.savefig('extra_electron.pdf', bbox_inches='tight', dpi=dpi)
