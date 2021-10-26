
import os
import sys
import string
import imageio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

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
from edafm.models import EDAFMNet

# # Set matplotlib font rendering to use LaTex
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "font.serif": ["Computer Modern Roman"]
# })

data_dir    = '../data'                     # Path to data directory
X_slices    = [0, 3, 5]                     # Which AFM slices to plot
tip_names   = ['CO', 'Xe']                  # AFM tip types
device      = 'cuda'                        # Device to run inference on
molecules   = ['NCM', 'PTH', 'TTF-TDZ']     # Molecules to run
fig_width   = 140                           # Figure width in mm
fontsize    = 8
dpi         = 300

def apply_preprocessing(batch):

    X, Y, xyzs = batch

    X = [x[..., 2:8] for x in X]

    pp.add_norm(X)
    np.random.seed(0)
    pp.add_noise(X, c=0.08)

    return X, Y, xyzs

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
    'scan_dim'          : (144, 144, 20),
    'scan_window'       : ((2.0, 2.0, 7.0), (20.0, 20.0, 9.0)),
    'amplitude'         : 1.0,
    'df_steps'          : 10,
    'initFF'            : True
}

generator_kwargs = {
    'batch_size'    : 1,
    'distAbove'     : 5.3,
    'iZPPs'         : [8, 54],
    'Qs'            : [[ -10, 20,  -10, 0 ], [  30,  -60,  30, 0 ]],
    'QZs'           : [[ 0.1,  0, -0.1, 0 ], [ 0.1,   0, -0.1, 0 ]]
}

# Paths to molecule xyz files
molecules = [os.path.join(data_dir, m) for m in molecules]
xyz_paths = [os.path.join(m, 'mol.xyz') for m in molecules]

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

# Define generator
trainer = InverseAFMtrainer(afmulator, aux_maps, xyz_paths, **generator_kwargs)

# Load model
model = EDAFMNet(device=device, trained_weights='base')

# Set figure
fig_width = 0.1/2.54*fig_width
width_ratios = [3, 6, 8, 0.3]
fig = plt.figure(figsize=(fig_width, 4.05*len(molecules)*fig_width/sum(width_ratios)))
fig_grid = fig.add_gridspec(len(molecules), 1, wspace=0, hspace=0.03)

# Define ticks for colorbars
ticks = [
    [-0.10, -0.05, 0.00, 0.05, 0.10],
    [-0.04, -0.02, 0.00, 0.02, 0.04],
    [-0.08, -0.04, 0.00, 0.04, 0.08]
]

# Loop over molecules and plot
for ib, batch in enumerate(trainer):

    # Get batch and predict
    X, Y_pc, xyzs = apply_preprocessing(batch)
    with torch.no_grad():
        X = [torch.from_numpy(x).unsqueeze(1).to(device) for x in X]
        pred, attentions = model(X, return_attention=True)
        pred = [p.cpu().numpy() for p in pred]
        attentions = [a.cpu().numpy() for a in attentions]
        X = [x.squeeze(1).cpu().numpy() for x in X]

    # Load Hartree reference
    Y_hartree = np.load(os.path.join(molecules[ib], 'ESMapHartree.npy'))

    # Create plot grid
    sample_grid = fig_grid[ib, 0].subgridspec(1, len(width_ratios), wspace=0.02, hspace=0, width_ratios=width_ratios)
    xyz_ax = sample_grid[0, 0].subgridspec(1, 1, wspace=0, hspace=0).subplots(squeeze=True)
    input_axes = sample_grid[0, 1].subgridspec(len(X), len(X_slices), wspace=0.01, hspace=0.02).subplots(squeeze=False)
    pred_ax, ref_ax = sample_grid[0, 2].subgridspec(1, 2, wspace=0.01, hspace=0).subplots(squeeze=True)
    cbar_ax = sample_grid[0, 3].subgridspec(1, 1, wspace=0, hspace=0).subplots(squeeze=True)

    # Set subfigure reference letters
    grid_pos = sample_grid.get_grid_positions(fig)
    x, y = grid_pos[2][0], (grid_pos[1][0] + grid_pos[0][0]) / 2 + 0.3/len(molecules)
    fig.text(x, y, string.ascii_uppercase[ib], fontsize=fontsize)

    # Plot molecule geometry
    xyz_ax.imshow(imageio.imread(os.path.join(molecules[ib], 'mol.png')))

    # Plot AFM inputs
    ims = []
    for i, x in enumerate(X):
        for j, s in enumerate(X_slices):
            ims.append(input_axes[i, j].imshow(x[0,:,:,s].T, origin='lower', cmap='afmhot'))
            input_axes[i, j].set_axis_off()
        input_axes[i, 0].text(-0.1, 0.5, tip_names[i], horizontalalignment='center',
            verticalalignment='center', transform=input_axes[i, 0].transAxes,
            rotation='vertical', fontsize=fontsize)

    # Figure out data limits
    vmax = max(
        abs(pred[0].min()), abs(pred[0].max()),
        abs(Y_pc[0].min()), abs(Y_pc[0].max()),
    )
    vmin = -vmax

    # Plot prediction and references
    pred_ax.imshow(pred[0][0].T, origin='lower', cmap='coolwarm', vmin=vmin, vmax=vmax)
    ref_ax.imshow(Y_pc[0][0].T, origin='lower', cmap='coolwarm', vmin=vmin, vmax=vmax)
    
    # Plot ES Map colorbar
    m_es = cm.ScalarMappable(cmap=cm.coolwarm)
    m_es.set_array((vmin, vmax))
    cbar = plt.colorbar(m_es, cax=cbar_ax)
    cbar.set_ticks(ticks[ib])
    cbar_ax.tick_params(labelsize=fontsize-1)
    cbar.set_label('V/Å', fontsize=fontsize)

    # Turn off axes ticks
    xyz_ax.set_axis_off()
    pred_ax.set_axis_off()
    ref_ax.set_axis_off()

    # Set titles for first row of images
    if ib == 0:
        input_axes[0, len(X_slices)//2].set_title('AFM simulation', fontsize=fontsize, y=0.90)
        pred_ax.set_title('Prediction', fontsize=fontsize, y=0.95)
        ref_ax.set_title('Reference', fontsize=fontsize, y=0.95)

    # Calculate relative error metric
    rel_abs_err_es = np.mean(np.abs(pred[0] - Y_pc[0])) / np.ptp(Y_pc[0])
    print(f'Relative error: {rel_abs_err_es*100:.2f}%')

plt.savefig('sims.pdf', bbox_inches='tight', dpi=dpi)
