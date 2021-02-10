
import os
import sys
import time
import string
import shutil
import imageio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.ndimage import rotate, shift

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

def apply_preprocessing_exp(X, real_dim):

    # Pick slices
    X[0] = X[0][..., 0:10] # CO
    X[1] = X[1][..., 4:14] # Xe

    X = pp.interpolate_and_crop(X, real_dim)
    pp.add_norm(X)

    # Flip, rotate and shift Xe data
    X[1] = X[1][:,::-1]
    X[1] = rotate(X[1], angle=-12, axes=(2,1), reshape=False, mode='reflect')
    X[1] = shift(X[1], shift=(0,-5,1,0), mode='reflect')
    X = [x[:,2:90] for x in X]
    
    return X

descriptors     = ['ES', 'Height Map']      # Labels used when outputting info during prediction ('ES' is special with different colormap)
model_path      = './model.h5'              # Path to trained model weights
X_slices        = [0, 5, 9]                 # Which AFM slices to plot
tip_names       = ['CO', 'Xe']              # AFM tip types
fontsize        = 20

# Load experimental data and preprocess
data1 = np.load('./data/BCB_CO.npz')
X1 = data1['data']
afm_dim1 = (data1['lengthX'], data1['lengthY'])

data2 = np.load('./data/BCB_Xe.npz')
X2 = data2['data']
afm_dim2 = (data2['lengthX'], data2['lengthY'])

assert afm_dim1 == afm_dim2
afm_dim = afm_dim1
X = apply_preprocessing_exp([X1[None], X2[None]], afm_dim)

# Load model
input_shape = X[0].shape[1:]
model = ESUNet(n_in=2, n_out=2, input_shape=input_shape, last_relu=[False, True])
load_pretrained_weights(model, tip_type='CO-Xe')

# Get predictions
pred = model.predict(X)

# Create figure grid
width_ratios = [6, 4, 2]
gap = 0.15
fig = plt.figure(figsize=(sum(width_ratios), 4.68))
fig_grid = fig.add_gridspec(1, len(width_ratios), wspace=0.02, hspace=0, width_ratios=width_ratios)
afm_grid = fig_grid[0, 0].subgridspec(1, 1, wspace=0, hspace=gap)
pred_grid = fig_grid[0, 1].subgridspec(1, 1, wspace=0, hspace=gap)
cbar_grid = fig_grid[0, 2].subgridspec(1, 2, wspace=4, hspace=0)

# Get axes from grid
afm_axes = afm_grid[0, 0].subgridspec(len(X), len(X_slices), wspace=0.01, hspace=0.01).subplots(squeeze=False)
pred_ax = pred_grid.subplots(squeeze=True)
cbar_es_ax, cbar_hm_ax = cbar_grid.subplots(squeeze=True)

# Plot AFM
for i, x in enumerate(X):
    for j, s in enumerate(X_slices):
        
        # Plot AFM slice
        im = afm_axes[i, j].imshow(x[0,:,:,s].T, origin='lower', cmap='afmhot')
        afm_axes[i, j].set_axis_off()
    
    # Put tip names to the left of the AFM image rows
    afm_axes[i, 0].text(-0.1, 0.5, tip_names[i], horizontalalignment='center',
        verticalalignment='center', transform=afm_axes[i, 0].transAxes,
        rotation='vertical', fontsize=fontsize)

# Plot prediction
vis.plot_ES_contour(pred_ax, pred[0][0], pred[1][0], es_colorbar=False, hm_colorbar=False, axis_off=True)

# Plot ES Map colorbar
plt.rcParams["font.serif"] = "cmr10"
vmax = max(abs(pred[0][0].min()), abs(pred[0][0].max()))
vmin = -vmax
m_es = cm.ScalarMappable(cmap=cm.coolwarm)
m_es.set_array([vmin, vmax])
cbar = plt.colorbar(m_es, cax=cbar_es_ax)
cbar.set_ticks([-0.05, 0.0, 0.05])
# cbar.set_ticklabels([f'{i:.2f}'.replace('-', '$-$') for i in cbar.get_ticks()])
cbar_es_ax.tick_params(labelsize=fontsize-4)
cbar.set_label('V/Å', fontsize=fontsize) # Seems to be broken in matplotlib 3.3.3 with cmr10 font

# Plot Height Map colorbar
m_hm = cm.ScalarMappable(cmap=cm.viridis)
m_hm.set_array([pred[1][0].min(), pred[1][0].max()])
cbar = plt.colorbar(m_hm, cax=cbar_hm_ax)
cbar.set_ticks([0, 0.5, 1.0, 1.5, 2.0])
cbar_hm_ax.tick_params(labelsize=fontsize-4)
cbar.set_label('Å', fontsize=fontsize) # Seems to be broken in matplotlib 3.3.3 with cmr10 font

# Set titles
afm_axes[0, len(X_slices)//2].set_title('AFM experiment', fontsize=fontsize)
pred_ax.set_title('Exp. prediction', fontsize=fontsize)

plt.savefig('bcb_mirrored.pdf', bbox_inches='tight')
