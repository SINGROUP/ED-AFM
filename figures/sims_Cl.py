
import os
import sys
import time
import string
import shutil
import imageio
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('../ProbeParticleModel') # Make sure ProbeParticleModel is on path
from pyProbeParticle import oclUtils     as oclu
from pyProbeParticle import fieldOCL     as FFcl
from pyProbeParticle import RelaxOpenCL  as oclr
from pyProbeParticle import common       as PPU
from pyProbeParticle import basUtils
from pyProbeParticle import AuxMap       as aux
from pyProbeParticle.AFMulatorOCL_Simple    import AFMulator
from pyProbeParticle.GeneratorOCL_Simple2   import InverseAFMtrainer

sys.path.append('../') # Make sure ED-AFM is on path
import edafm.preprocessing as pp
import edafm.visualization as vis
from edafm.models import ESUNet, load_pretrained_weights

# Set tensorflow memory growth
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

# # Set matplotlib font
# from matplotlib import rc
# rc('font', family = 'serif', serif = 'cmr10')

class Trainer(InverseAFMtrainer):

    def on_afm_start(self):
        if self.afmulator.iZPP in [8, 54]:
            afmulator.scanner.stiffness = np.array([0.25, 0.25, 0.0, 30.0], dtype=np.float32) / -16.0217662
        elif self.afmulator.iZPP == 17:
            afmulator.scanner.stiffness = np.array([0.75, 0.75, 0.0, 30.0], dtype=np.float32) / -16.0217662
        else:
            raise RuntimeError(f'Unknown tip {self.afmulator.iZPP}')

def apply_preprocessing(batch):

    X, Y, xyzs = batch

    pp.add_norm(X)
    np.random.seed(0)
    pp.add_noise(X, c=0.05)
    
    Y = [Y[-1][...,i] for i in range(2)]

    return X, Y, xyzs

X_slices            = [0, 5, 9]                 # Which AFM slices to plot
tip_names           = ['CO', 'Xe', 'Cl']        # AFM tip types
fontsize            = 24

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
                            [ 0.0,  0.0, 6.0]
                            ]),
    'scan_dim'          : (128, 128, 20),
    'scan_window'       : ((2.0, 2.0, 7.0), (20.0, 20.0, 9.0)),
    'amplitude'         : 1.0,
    'df_steps'          : 10,
    'initFF'            : True
}

# Paths to molecule xyz files
molecules = ['data/mol1.xyz', 'data/mol2.xyz', 'data/ttf-tdz.xyz']

# Define AFMulator
afmulator = AFMulator(**afmulator_args)
afmulator.npbc = (0,0,0)

# Define AuxMaps
scan_window = [afmulator.scan_window[0][:2], afmulator.scan_window[1][:2]]
aux_maps = [
    aux.ESMap(
        scanner     = afmulator.scanner,
        zmin        = -2.0,
        iso         = 0.1
    )
]

# Define generator for Cl-CO
trainer_Cl_CO = Trainer(afmulator, aux_maps, molecules,
    batch_size  = 1,
    distAbove   = 5.2,
    iZPPs       = [17, 8], # Cl, CO
    Qs          = [[ -0.3, 0, 0, 0 ], [ -10, 20,  -10, 0 ]],
    QZs         = [[    0, 0, 0, 0 ], [ 0.1,  0, -0.1, 0 ]]
)

# Define generator for Xe-Cl
trainer_Xe_Cl = Trainer(afmulator, aux_maps, molecules,
    batch_size  = 1,
    distAbove   = 5.2,
    iZPPs       = [54, 17], # Xe, Cl
    Qs          = [[  30, -60,   30, 0 ], [ -0.3, 0, 0, 0 ]],
    QZs         = [[ 0.1,   0, -0.1, 0 ], [    0, 0, 0, 0 ]]
)

# Load model for Cl-CO
input_shape = afmulator.scan_dim[:2] + (afmulator.scan_dim[2]-afmulator.df_steps,)
model_Cl_CO = ESUNet(n_in=2, n_out=2, input_shape=input_shape, last_relu=[False, True])
load_pretrained_weights(model_Cl_CO, tip_type='Cl-CO')

# Load model for Xe-Cl
input_shape = afmulator.scan_dim[:2] + (afmulator.scan_dim[2]-afmulator.df_steps,)
model_Xe_Cl = ESUNet(n_in=2, n_out=2, input_shape=input_shape, last_relu=[False, False])
load_pretrained_weights(model_Xe_Cl, tip_type='Xe-Cl')

# Loop over molecules and plot
width_ratios = [6, 12]
fig = plt.figure(figsize=(sum(width_ratios), 6*len(molecules)))
fig_grid = fig.add_gridspec(len(molecules), 1, wspace=0, hspace=0.03)
for ib, (batch_Cl_CO, batch_Xe_Cl) in enumerate(zip(trainer_Cl_CO, trainer_Xe_Cl)):

    # Get batch and predict
    X_Cl_CO, Y_Cl_CO, _ = apply_preprocessing(batch_Cl_CO)
    X_Xe_Cl, Y_Xe_Cl, _ = apply_preprocessing(batch_Xe_Cl)
    X = [X_Cl_CO[1], X_Xe_Cl[0], X_Xe_Cl[1]] # CO, Xe, Cl
    pred_Cl_CO = model_Cl_CO.predict(X_Cl_CO)
    pred_Xe_Cl = model_Xe_Cl.predict(X_Xe_Cl)

    # Create plot grid
    sample_grid = fig_grid[ib, 0].subgridspec(1, 2, wspace=0.01, hspace=0, width_ratios=width_ratios)
    input_axes = sample_grid[0, 0].subgridspec(len(X), len(X_slices), wspace=0.01, hspace=0.02).subplots(squeeze=False)
    pred_Cl_CO_ax, pred_Xe_Cl_ax = sample_grid[0, 1].subgridspec(1, 2, wspace=0.01, hspace=0).subplots(squeeze=True)

    # Set subfigure reference letters
    grid_pos = sample_grid.get_grid_positions(fig)
    x, y = grid_pos[2][0]-0.03, (grid_pos[1][0] + grid_pos[0][0]) / 2 + 0.3/len(molecules) + 0.01
    fig.text(x, y, string.ascii_uppercase[ib], fontsize=fontsize)

    # Plot AFM inputs
    ims = []
    for i, x in enumerate(X):
        for j, s in enumerate(X_slices):
            ims.append(input_axes[i, j].imshow(x[0,:,:,s].T, origin='lower', cmap='afmhot'))
            input_axes[i, j].set_axis_off()
        input_axes[i, 0].text(-0.1, 0.5, tip_names[i], horizontalalignment='center',
            verticalalignment='center', transform=input_axes[i, 0].transAxes,
            rotation='vertical', fontsize=fontsize)

    # Overlay molecule geometry onto first AFM image
    xyz_img = np.flipud(imageio.imread(f'images/{molecules[ib][5:-4]}.png'))
    input_axes[0, 0].imshow(xyz_img, origin='lower', extent=ims[0].get_extent())

    # Plot predictions
    vis.plot_ES_contour(pred_Cl_CO_ax, pred_Cl_CO[0][0], pred_Cl_CO[1][0], 
        es_colorbar=False, hm_colorbar=False, axis_off=True)
    vis.plot_ES_contour(pred_Xe_Cl_ax, pred_Xe_Cl[0][0], pred_Xe_Cl[1][0],
        es_colorbar=False, hm_colorbar=False, axis_off=True)

    # Set titles for first row of images
    if ib == 0:
        input_axes[0, len(X_slices)//2].set_title('AFM simulation', fontsize=fontsize)
        pred_Cl_CO_ax.set_title('Prediction (Cl-CO)', fontsize=fontsize)
        pred_Xe_Cl_ax.set_title('Prediction (Cl-Xe)', fontsize=fontsize)

plt.savefig('sims_Cl.pdf', bbox_inches='tight')
