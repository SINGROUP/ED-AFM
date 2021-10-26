
import os
import sys
import string
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.ndimage import rotate, shift

import torch

sys.path.append('..')
import edafm.preprocessing as pp
from edafm.models import EDAFMNet

# # Set matplotlib font rendering to use LaTex
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "font.serif": ["Computer Modern Roman"]
# })

def apply_preprocessing_bcb(X, real_dim):

    # Pick slices
    X[0] = np.concatenate([X[0][..., i:i+6] for i in [5, 4, 3]], axis=0)
    X[1] = np.concatenate([X[1][..., i:i+6] for i in [10, 9, 8, 4]], axis=0)

    X = pp.interpolate_and_crop(X, real_dim)
    pp.add_norm(X)

    # Flip, rotate and shift Xe data
    X[1] = X[1][:,::-1]
    X[1] = rotate(X[1], angle=-12, axes=(2,1), reshape=False, mode='reflect')
    X[1] = shift(X[1], shift=(0,-5,1,0), mode='reflect')
    X = [x[:,0:96] for x in X]

    return X

def apply_preprocessing_ptcda(X, real_dim):

    # Pick slices
    X[0] = np.concatenate([X[0][..., i:i+6] for i in [3, 2, 1]], axis=0)
    X[1] = np.concatenate([X[1][..., i:i+6] for i in [6, 2, 1, 0]], axis=0)
    
    X = pp.interpolate_and_crop(X, real_dim)
    pp.add_norm(X)
    X = [x[:,:,6:78] for x in X]

    return X

# Options
data_dir    = '../data'             # Path to data directory
device      = 'cuda'                # Device to run inference on
fig_width   = 150                   # Figure width in mm
fontsize    = 8
dpi         = 300

# Load model
model = EDAFMNet(device=device, trained_weights='base')

# Load BCB data and preprocess
data_bcb = np.load(os.path.join(data_dir, 'BCB/data_CO_exp.npz'))
afm_dim_bcb = (data_bcb['lengthX'], data_bcb['lengthY'])
X_bcb_CO = data_bcb['data']
X_bcb_Xe = np.load(os.path.join(data_dir, 'BCB/data_Xe_exp.npz'))['data']
X_bcb = apply_preprocessing_bcb([X_bcb_CO[None], X_bcb_Xe[None]], afm_dim_bcb)

# Load PTCDA data and preprocess
data_ptcda = np.load(os.path.join(data_dir, 'PTCDA/data_CO_exp.npz'))
afm_dim_ptcda = (data_ptcda['lengthX'], data_ptcda['lengthY'])
X_ptcda_CO = data_ptcda['data']
X_ptcda_Xe = np.load(os.path.join(data_dir, 'PTCDA/data_Xe_exp.npz'))['data']
X_ptcda = apply_preprocessing_ptcda([X_ptcda_CO[None], X_ptcda_Xe[None]], afm_dim_ptcda)

# Create figure grid
fig_width = 0.1/2.54*fig_width
height_ratios = [1, 0.525]
width_ratios = [1, 0.03]
fig = plt.figure(figsize=(fig_width, 0.86*sum(height_ratios)*fig_width/sum(width_ratios)))
fig_grid = fig.add_gridspec(2, 2, wspace=0.05, hspace=0.15, height_ratios=height_ratios, width_ratios=width_ratios)

ticks = [
    [-0.03, -0.02, -0.01, 0.00, 0.01, 0.02, 0.03],
    [-0.08, -0.04, 0.00, 0.04, 0.08]
]

offsets_labels = [
    [
        ['-0.1Å', '+0.0Å', '+0.1Å', '+0.5Å'],
        ['-0.1Å', '+0.0Å', '+0.1Å']
    ],
    [
        ['-0.6Å', '-0.2Å', '-0.1Å', '+0.0Å'],
        ['-0.1Å', '+0.0Å', '+0.1Å']
    ]
]

# Do for both BCB and PTCDA
for k, X in enumerate([X_bcb, X_ptcda]):

    # Create subgrid for predictions and colorbar
    pred_grid = fig_grid[k, 0].subgridspec(3, 4, wspace=0.01, hspace=0)
    pred_axes = pred_grid.subplots(squeeze=False)
    cbar_ax = fig_grid[k, 1].subgridspec(1, 1, wspace=0, hspace=0).subplots(squeeze=True)

    preds = np.zeros([3, 4, X[0].shape[1], X[0].shape[2]])
    for i in range(3):
        for j in range(4):

            # Pick a subset of slices
            X_ = [x.copy() for x in X]
            X_[0] = X_[0][i:i+1]
            X_[1] = X_[1][j:j+1]
            X_cuda = [torch.from_numpy(x.astype(np.float32)).unsqueeze(1).to(device) for x in X_]

            # Make prediction
            with torch.no_grad():
                pred = model(X_cuda)
                preds[i, j] = pred[0][0].cpu().numpy()

    # Figure out data limits
    vmax = max(abs(preds.min()), abs(preds.max()))
    vmin = -vmax

    # Plot predictions
    for i in range(3):
        for j in range(4):
            pred_axes[i, j].imshow(preds[i, j].T, origin='lower', cmap='coolwarm', vmin=vmin, vmax=vmax)
            pred_axes[i, j].set_axis_off()
            if i == 0:
                pred_axes[i, j].text(0.5, 1.06+k*0.03, offsets_labels[k][0][j], horizontalalignment='center',
                verticalalignment='center', transform=pred_axes[i, j].transAxes,
                fontsize=fontsize-2)
            if j == 0:
                pred_axes[i, j].text(-0.06, 0.5, offsets_labels[k][1][i], horizontalalignment='center',
                verticalalignment='center', transform=pred_axes[i, j].transAxes,
                rotation='vertical', fontsize=fontsize-2)
    
    # Plot ES Map colorbar
    m_es = cm.ScalarMappable(cmap=cm.coolwarm)
    m_es.set_array((vmin, vmax))
    cbar = plt.colorbar(m_es, cax=cbar_ax)
    cbar.set_ticks(ticks[k])
    cbar_ax.tick_params(labelsize=fontsize-1)

    # Set Xe-shift title
    ((x0, _), ( _, y)) = pred_axes[0,  0].get_position().get_points()
    (( _, _), (x1, _)) = pred_axes[0, -1].get_position().get_points()
    plt.text((x0 + x1)/2, y+0.03, 'Xe-shift', fontsize=fontsize,
        transform=fig.transFigure, horizontalalignment='center', verticalalignment='center')

    # Set CO-shift title
    (( x,  _), (_, y1)) = pred_axes[ 0, 0].get_position().get_points()
    (( _, y0), (_,  _)) = pred_axes[-1, 0].get_position().get_points()
    plt.text(x0-0.04, (y0 + y1)/2, 'CO-shift', fontsize=fontsize,
        transform=fig.transFigure, horizontalalignment='center', verticalalignment='center',
        rotation='vertical')

    # Set subfigure reference letters
    grid_pos = pred_grid.get_grid_positions(fig)
    x, y = grid_pos[2][0]-0.03, grid_pos[1][0]+0.01
    fig.text(x, y, string.ascii_uppercase[k], fontsize=fontsize,
        horizontalalignment='center', verticalalignment='center')

plt.savefig('height_dependence.pdf', bbox_inches='tight', dpi=dpi)
