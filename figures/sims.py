
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

def apply_preprocessing(batch):

    X, Y, xyzs = batch

    pp.add_norm(X)
    np.random.seed(0)
    pp.add_noise(X, c=0.05)

    X = list(reversed(X))
    
    Y = [Y[-1][...,i] for i in range(2)]

    return X, Y, xyzs

X_slices        = [0, 5, 9]     # Which AFM slices to plot
tip_names       = ['CO', 'Xe']  # AFM tip types
fontsize        = 20

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

generator_kwargs = {
    'batch_size'    : 1,
    'distAbove'     : 5.2,
    'iZPPs'         : [54, 8],
    'Qs'            : [[  30, -60,   30, 0 ], [ -10, 20,  -10, 0 ]],
    'QZs'           : [[ 0.1,  0, -0.1, 0 ], [ 0.1,   0, -0.1, 0 ]]
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

# Define generator
trainer = InverseAFMtrainer(afmulator, aux_maps, molecules, **generator_kwargs)

# Load model
input_shape = afmulator.scan_dim[:2] + (afmulator.scan_dim[2]-afmulator.df_steps,)
model = ESUNet(n_in=2, n_out=2, input_shape=input_shape, last_relu=[False, True])
load_pretrained_weights(model, tip_type='CO-Xe')

# Loop over molecules and plot
width_ratios = [3, 6, 8]
fig = plt.figure(figsize=(sum(width_ratios), 4.05*len(molecules)))
fig_grid = fig.add_gridspec(len(molecules), 1, wspace=0, hspace=0.03)
for ib, batch in enumerate(trainer):

    # Get batch and predict
    X, Y, xyzs = apply_preprocessing(batch)
    pred = model.predict(X)

    # Create plot grid
    sample_grid = fig_grid[ib, 0].subgridspec(1, 3, wspace=0.01, hspace=0, width_ratios=width_ratios)
    xyz_ax = sample_grid[0, 0].subgridspec(1, 1, wspace=0, hspace=0).subplots(squeeze=True)
    input_axes = sample_grid[0, 1].subgridspec(len(X), len(X_slices), wspace=0.01, hspace=0.02).subplots(squeeze=False)
    pred_ax, ref_ax = sample_grid[0, 2].subgridspec(1, 2, wspace=0.01, hspace=0).subplots(squeeze=True)

    # Set subfigure reference letters
    grid_pos = sample_grid.get_grid_positions(fig)
    x, y = grid_pos[2][0], (grid_pos[1][0] + grid_pos[0][0]) / 2 + 0.3/len(molecules)
    fig.text(x, y, string.ascii_uppercase[ib], fontsize=fontsize)

    # Plot molecule geometries
    xyz_ax.imshow(imageio.imread(f'images/{molecules[ib][5:-4]}_side.png'))
    xyz_ax.set_axis_off()

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

    # Figure out data limits
    vmax = max(abs(pred[0].min()), abs(pred[0].max()), abs(Y[0].min()), abs(Y[0].max()))
    data_lims_es = (-vmax, vmax)
    data_lims_hm = (
        min(pred[1].min(), Y[1].min()),
        max(pred[1].max(), Y[1].max())
    )

    # Plot prediction
    vis.plot_ES_contour(pred_ax, pred[0][0], pred[1][0], data_lims_es=data_lims_es,
        data_lims_hm=data_lims_hm, es_colorbar=False, hm_colorbar=False, axis_off=True)

    # Plot reference
    vis.plot_ES_contour(ref_ax, Y[0][0], Y[1][0], data_lims_es=data_lims_es,
        data_lims_hm=data_lims_hm, es_colorbar=False, hm_colorbar=False, axis_off=True)

    # Set titles for first row of images
    if ib == 0:
        input_axes[0, len(X_slices)//2].set_title('AFM simulation', fontsize=fontsize)
        pred_ax.set_title('Prediction', fontsize=fontsize)
        ref_ax.set_title('Reference', fontsize=fontsize)

    # Calculate relative error metric
    rel_abs_err_es = np.mean(np.abs(pred[0] - Y[0])) / np.ptp(Y[0])
    rel_abs_err_hm = np.mean(np.abs(pred[1] - Y[1])) / np.ptp(Y[1])
    print(f'Relative error (ES Map/Height Map): {rel_abs_err_es*100:.2f}%/{rel_abs_err_hm*100:.2f}%')

plt.savefig('sims.pdf', bbox_inches='tight')
