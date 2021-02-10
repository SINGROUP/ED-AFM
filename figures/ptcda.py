
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

class Trainer(InverseAFMtrainer):

    # Override position handling to offset the molecule to the side
    def handle_positions(self):
        sw = self.afmulator.scan_window
        scan_center = np.array([sw[1][0] + sw[0][0], sw[1][1] + sw[0][1]]) / 2
        self.xyzs[:,:2] += scan_center - self.xyzs[:,:2].mean(axis=0) + np.array([3.5, 0.0])

def apply_preprocessing_sim(batch):

    X, Y, xyzs = batch

    pp.add_norm(X)
    np.random.seed(0)
    pp.add_noise(X, c=0.05)

    X = list(reversed(X))
    
    Y = [Y[-1][...,i] for i in range(2)]

    return X, Y, xyzs

def apply_preprocessing_exp(X, real_dim):

    # Pick slices
    X[0] = X[0][..., :10] # CO
    X[1] = X[1][..., :10] # Xe

    X = pp.interpolate_and_crop(X, real_dim)
    pp.add_norm(X)
    X = [x[:,:80,8:80] for x in X]
    
    return X

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
                            [ 0.0,  0.0, 7.0]
                            ]),
    'scan_dim'          : (104, 96, 20),
    'scan_window'       : ((2.0, 2.0, 7.0), (15.0, 14.0, 9.0)),
    'amplitude'         : 1.0,
    'df_steps'          : 10,
    'initFF'            : True
}

generator_kwargs = {
    'batch_size'    : 1,
    'distAbove'     : 5.4,
    'iZPPs'         : [54, 8],
    'Qs'            : [[  30, -60,   30, 0 ], [ -10, 20,  -10, 0 ]],
    'QZs'           : [[ 0.1,  0, -0.1, 0 ], [ 0.1,   0, -0.1, 0 ]]
}

# Paths to molecule xyz files
molecules = ['data/ptcda.xyz']

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
trainer = Trainer(afmulator, aux_maps, molecules, **generator_kwargs)

# Get simulation data
X_sim, ref, xyzs = apply_preprocessing_sim(next(iter(trainer)))

# Load experimental data and preprocess
data1 = np.load('./data/PTCDA_CO.npz')
X1 = data1['data']
afm_dim1 = (data1['lengthX'], data1['lengthY'])

data2 = np.load('./data/PTCDA_Xe.npz')
X2 = data2['data']
afm_dim2 = (data2['lengthX'], data2['lengthY'])

assert afm_dim1 == afm_dim2
afm_dim = afm_dim1
X_exp = apply_preprocessing_exp([X1[None], X2[None]], afm_dim)

# Load model for sim
input_shape = afmulator.scan_dim[:2] + (10,)
model_sim = ESUNet(n_in=2, n_out=2, input_shape=input_shape, last_relu=[False, True])
load_pretrained_weights(model_sim, tip_type='CO-Xe')

# Load model for exp (need two models because of different input sizes)
input_shape = X_exp[0].shape[1:]
model_exp = ESUNet(n_in=2, n_out=2, input_shape=input_shape, last_relu=[False, True])
load_pretrained_weights(model_exp, tip_type='CO-Xe')

# Get predictions
pred_sim = model_sim.predict(X_sim)
pred_exp = model_exp.predict(X_exp)

# Create figure grid
width_ratios = [6, 4, 2]
height_ratios = [1, 0.975]
gap = 0.15
fig = plt.figure(figsize=(sum(width_ratios), 7.8))
fig_grid = fig.add_gridspec(1, len(width_ratios), wspace=0.02, hspace=0, width_ratios=width_ratios)
afm_grid = fig_grid[0, 0].subgridspec(2, 1, wspace=0, hspace=gap, height_ratios=height_ratios)
pred_grid = fig_grid[0, 1].subgridspec(2, 1, wspace=0, hspace=gap, height_ratios=height_ratios)
cbar_grid = fig_grid[0, 2].subgridspec(1, 2, wspace=4, hspace=0)

# Get axes from grid
afm_sim_axes = afm_grid[0, 0].subgridspec(len(X_sim), len(X_slices), wspace=0.01, hspace=0.01).subplots(squeeze=False)
afm_exp_axes = afm_grid[1, 0].subgridspec(len(X_exp), len(X_slices), wspace=0.01, hspace=0.01).subplots(squeeze=False)
pred_sim_ax, pred_exp_ax = pred_grid.subplots(squeeze=True)
cbar_es_ax, cbar_hm_ax = cbar_grid.subplots(squeeze=True)

# Plot AFM
for k, (axes, X) in enumerate(zip([afm_sim_axes, afm_exp_axes], [X_sim, X_exp])):
    for i, x in enumerate(X):
        for j, s in enumerate(X_slices):
            
            # Plot AFM slice
            im = axes[i, j].imshow(x[0,:,:,s].T, origin='lower', cmap='afmhot')
            axes[i, j].set_axis_off()

            # Overlay molecule geometry onto first simulated AFM image
            if i == j == k == 0:
                xyz_img = np.flipud(imageio.imread(f'images/{molecules[0][5:-4]}.png'))
                axes[0, 0].imshow(xyz_img, origin='lower', extent=im.get_extent())
        
        # Put tip names to the left of the AFM image rows
        axes[i, 0].text(-0.1, 0.5, tip_names[i], horizontalalignment='center',
            verticalalignment='center', transform=axes[i, 0].transAxes,
            rotation='vertical', fontsize=fontsize)


# Figure out data limits
vmax_es = max(
    abs(pred_sim[0].min()), abs(pred_sim[0].max()),
    abs(pred_exp[0].min()), abs(pred_exp[0].max())
)
data_lims_es = (-vmax_es, vmax_es)
data_lims_hm = (
    min(pred_sim[1].min(), pred_exp[1].min()),
    max(pred_sim[1].max(), pred_exp[1].max())-0.15
)

# Set contour line levels to plot
levels_sim = [0.3, 0.6, 0.9, 1.2, 1.4, 1.53, 1.63, 1.75]
levels_exp = [0.3, 0.6, 0.9, 1.2, 1.4, 1.53, 1.63, 1.71]

# Plot simulation prediction
vis.plot_ES_contour(pred_sim_ax, pred_sim[0][0], pred_sim[1][0], data_lims_es=data_lims_es,
    data_lims_hm=data_lims_hm, levels=levels_sim, es_colorbar=False, hm_colorbar=False, axis_off=True)

# Plot experimental prediction
vis.plot_ES_contour(pred_exp_ax, pred_exp[0][0], pred_exp[1][0], data_lims_es=data_lims_es,
    data_lims_hm=data_lims_hm, levels=levels_exp, es_colorbar=False, hm_colorbar=False, axis_off=True)

# Plot ES Map colorbar
m_es = cm.ScalarMappable(cmap=cm.coolwarm)
m_es.set_array(data_lims_es)
cbar = plt.colorbar(m_es, cax=cbar_es_ax)
cbar.set_ticks([-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3])
# cbar.set_ticklabels([f'{i:.1f}'.replace('-', '$-$') for i in cbar.get_ticks()])
cbar_es_ax.tick_params(labelsize=fontsize-4)
cbar.set_label('V/Å', fontsize=fontsize) # Seems to be broken in matplotlib 3.3.3 with cmr10 font

# Plot Height Map colorbar
m_hm = cm.ScalarMappable(cmap=cm.viridis)
m_hm.set_array(data_lims_hm)
cbar = plt.colorbar(m_hm, cax=cbar_hm_ax)
cbar.set_ticks([0, 0.5, 1.0, 1.5, 2.0])
cbar_hm_ax.tick_params(labelsize=fontsize-4)
cbar.set_label('Å', fontsize=fontsize) # Seems to be broken in matplotlib 3.3.3 with cmr10 font

# Set titles
afm_sim_axes[0, len(X_slices)//2].set_title('AFM simulation', fontsize=fontsize)
afm_exp_axes[0, len(X_slices)//2].set_title('AFM experiment', fontsize=fontsize)
pred_sim_ax.set_title('Sim. prediction', fontsize=fontsize)
pred_exp_ax.set_title('Exp. prediction', fontsize=fontsize)

plt.savefig('ptcda.pdf', bbox_inches='tight')
