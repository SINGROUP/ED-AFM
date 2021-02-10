
import sys
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

# # Set matplotlib font
# from matplotlib import rc
# rc('font', family = 'serif', serif = 'cmr10')

def apply_preprocessing_sim(batch):

    X, Y, xyzs = batch

    pp.add_norm(X)
    np.random.seed(0)
    pp.add_noise(X, c=0.05)

    # Add background gradient
    c = 0.3
    angle = -np.pi / 2
    x, y = np.meshgrid(np.arange(0, X[0].shape[1]), np.arange(0, X[0].shape[2]), indexing='ij')
    n = [np.cos(angle), np.sin(angle), 1]
    z = -(n[0]*x + n[1]*y)
    z -= z.mean()
    z /= np.ptp(z)
    for x in X:
        x += z[None, :, :, None]*c*np.ptp(x)

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
                            [23.0,  0.0, 0.0],
                            [ 0.0, 23.0, 0.0],
                            [ 0.0,  0.0, 6.0]
                            ]),
    'scan_dim'          : (152, 128, 20),
    'scan_window'       : ((2.0, 2.0, 7.0), (21.0, 18.0, 9.0)),
    'amplitude'         : 1.0,
    'df_steps'          : 10,
    'initFF'            : True
}

generator_kwargs = {
    'batch_size'    : 1,
    'distAbove'     : 5.3,
    'iZPPs'         : [54, 8],
    'Qs'            : [[  30, -60,   30, 0 ], [ -10, 20,  -10, 0 ]],
    'QZs'           : [[ 0.1,  0, -0.1, 0 ], [ 0.1,   0, -0.1, 0 ]]
}

# Paths to molecule xyz files
molecules = ['./data/ptcda.xyz']

# Define AFMulator
afmulator = AFMulator(**afmulator_args)
afmulator.npbc = (0,0,0)

# Define AuxMaps
aux_maps = [
    aux.ESMap(
        scanner = afmulator.scanner,
        zmin    = -2.0,
        iso     = 0.1
    )
]

# Define generator
trainer = InverseAFMtrainer(afmulator, aux_maps, molecules, **generator_kwargs)

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
load_pretrained_weights(model_sim, tip_type='CO-Xe-nograd')
pred_sim = model_sim.predict(X_sim)

# Load different weights with gradient augmentation
load_pretrained_weights(model_sim, tip_type='CO-Xe')
pred_sim_grad = model_sim.predict(X_sim)

# Load model for exp (need several models because of different input sizes)
input_shape = X_exp[0].shape[1:]
model_exp = ESUNet(n_in=2, n_out=2, input_shape=input_shape, last_relu=[False, True])
load_pretrained_weights(model_exp, tip_type='CO-Xe-nograd')
pred_exp = model_exp.predict(X_exp)

# Create figure grid
fig = plt.figure(figsize=(10, 6.75))
fig_grid = fig.add_gridspec(1, 2, wspace=0.1, hspace=0, width_ratios=[6, 4])
left_grid = fig_grid[0, 0].subgridspec(2, 1, wspace=0, hspace=0.1)

pred_sim_ax, pred_sim_grad_ax = fig_grid[0, 1].subgridspec(2, 1, wspace=0, hspace=0.1).subplots()
pred_exp_ax = left_grid[0, 0].subgridspec(1, 1).subplots()
afm_axes = left_grid[1, 0].subgridspec(len(X_sim), len(X_slices), wspace=0.01, hspace=0.01).subplots(squeeze=False)

# Plot AFM
for i, x in enumerate(X_sim):
    for j, s in enumerate(X_slices):
        
        # Plot AFM slice
        im = afm_axes[i, j].imshow(x[0,:,:,s].T, origin='lower', cmap='afmhot')
        afm_axes[i, j].set_axis_off()
    
    # Put tip names to the left of the AFM image rows
    afm_axes[i, 0].text(-0.1, 0.5, tip_names[i], horizontalalignment='center',
        verticalalignment='center', transform=afm_axes[i, 0].transAxes,
        rotation='vertical', fontsize=fontsize)

# Figure out ES data limits
vmax_sim = max(abs(pred_sim[0].min()), abs(pred_sim[0].max()))
vmax_sim_grad = max(abs(pred_sim_grad[0].min()), abs(pred_sim_grad[0].max()))
vmax_exp = max(abs(pred_exp[0].min()), abs(pred_exp[0].max()))
vmin_sim = -vmax_sim
vmin_sim_grad = -vmax_sim_grad
vmin_exp = -vmax_exp

# Plot ES predictions
pred_sim_ax.imshow(pred_sim[0][0].T, origin='lower', cmap='coolwarm', vmin=vmin_sim, vmax=vmax_sim)
pred_sim_grad_ax.imshow(pred_sim_grad[0][0].T, origin='lower', cmap='coolwarm', vmin=vmin_sim_grad, vmax=vmax_sim_grad)
pred_exp_ax.imshow(pred_exp[0][0].T, origin='lower', cmap='coolwarm', vmin=vmin_exp, vmax=vmax_exp)

pred_sim_ax.set_axis_off()
pred_sim_grad_ax.set_axis_off()
pred_exp_ax.set_axis_off()

# Set labels
pred_exp_ax.text(-0.1, 0.98, 'A', horizontalalignment='center',
    verticalalignment='center', transform=pred_exp_ax.transAxes, fontsize=fontsize)
afm_axes[0, 0].text(-0.2, 1.1, 'B', horizontalalignment='center',
    verticalalignment='center', transform=afm_axes[0, 0].transAxes, fontsize=fontsize)
pred_sim_ax.text(-0.06, 0.98, 'C', horizontalalignment='center',
    verticalalignment='center', transform=pred_sim_ax.transAxes, fontsize=fontsize)
pred_sim_grad_ax.text(-0.06, 0.98, 'D', horizontalalignment='center',
    verticalalignment='center', transform=pred_sim_grad_ax.transAxes, fontsize=fontsize)

plt.savefig('background_gradient.pdf', bbox_inches='tight')