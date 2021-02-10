
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

    X = list(reversed(X))
    Y = [Y[-1][...,i] for i in range(2)]

    return X, Y, xyzs

def apply_preprocessing_bcb(X, real_dim):
    X[0] = X[0][..., :10] # Pick slices
    X = pp.interpolate_and_crop(X, real_dim)
    pp.add_norm(X)
    return X

def apply_preprocessing_ptcda(X, real_dim):
    X[0] = X[0][..., :10] # Pick slices
    X = pp.interpolate_and_crop(X, real_dim)
    pp.add_norm(X)
    X = [x[:,:80,8:80] for x in X]
    return X

# Options
fontsize = 20

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
molecules = ['./data/ttf-tdz.xyz']

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

# Get simulation data
batch = next(iter(trainer))
X_sim, Y_sim, xyz = apply_preprocessing_sim(batch)

# Load BCB data and preprocess
data_bcb = np.load('./data/BCB_CO.npz')
X_bcb = data_bcb['data']
afm_dim_bcb = (data_bcb['lengthX'], data_bcb['lengthY'])
X_bcb = apply_preprocessing_bcb([X_bcb[None]], afm_dim_bcb)

# Load PTCDA data and preprocess
data_ptcda = np.load('./data/PTCDA_CO.npz')
X_ptcda = data_ptcda['data']
afm_dim_ptcda = (data_ptcda['lengthX'], data_ptcda['lengthY'])
X_ptcda = apply_preprocessing_ptcda([X_ptcda[None]], afm_dim_ptcda)

# Load model for simulations
input_shape = afmulator.scan_dim[:2] + (afmulator.scan_dim[2]-afmulator.df_steps,)
model_sim = ESUNet(n_in=1, n_out=2, input_shape=input_shape, last_relu=[False, True])
load_pretrained_weights(model_sim, tip_type='CO')

# Load model for BCB
model_bcb = ESUNet(n_in=1, n_out=2, input_shape=X_bcb[0].shape[1:], last_relu=[False, True])
load_pretrained_weights(model_bcb, tip_type='CO')

# Load model for PTCDA
model_ptcda = ESUNet(n_in=1, n_out=2, input_shape=X_ptcda[0].shape[1:], last_relu=[False, True])
load_pretrained_weights(model_ptcda, tip_type='CO')

# Make predictions
pred_sim = model_sim.predict(X_sim)
pred_bcb = model_bcb.predict(X_bcb)
pred_ptcda = model_ptcda.predict(X_ptcda)

# Make figure
width_ratios = [4, 4, 4.5]
gridspec_kw = {'width_ratios': width_ratios, 'wspace': 0.3}
fig, (ax_sim, ax_bcb, ax_ptcda) = plt.subplots(1, 3, figsize=(sum(width_ratios), 2.6), gridspec_kw=gridspec_kw)

# Plot simulation
plt.axes(ax_sim)
cbar, _ = vis.plot_ES_contour(ax_sim, pred_sim[0][0], pred_sim[1][0], es_colorbar=True, hm_colorbar=False, axis_off=True)
cbar.set_ticks([-0.2, -0.1, 0.0, 0.1, 0.2])
# cbar.set_ticklabels([f'{i:.2f}'.replace('-', '$-$') for i in cbar.get_ticks()])
cbar.ax.tick_params(labelsize=fontsize-4)
cbar.set_label('V/Å', fontsize=fontsize) # Seems to be broken in matplotlib 3.3.3 with cmr10 font

# Plot BCB
plt.axes(ax_bcb)
cbar, _ = vis.plot_ES_contour(ax_bcb, pred_bcb[0][0], pred_bcb[1][0], es_colorbar=True, hm_colorbar=False, axis_off=True)
cbar.set_ticks([-0.1, -0.05, 0.0, 0.05, 0.1])
# cbar.set_ticklabels([f'{i:.2f}'.replace('-', '$-$') for i in cbar.get_ticks()])
cbar.ax.tick_params(labelsize=fontsize-4)
cbar.set_label('V/Å', fontsize=fontsize) # Seems to be broken in matplotlib 3.3.3 with cmr10 font

# Plot PTCDA
plt.axes(ax_ptcda)
cbar, _ = vis.plot_ES_contour(ax_ptcda, pred_ptcda[0][0], pred_ptcda[1][0], es_colorbar=True, hm_colorbar=False, axis_off=True)
cbar.set_ticks([-0.2, -0.1, 0.0, 0.1, 0.2])
# cbar.set_ticklabels([f'{i:.2f}'.replace('-', '$-$') for i in cbar.get_ticks()])
cbar.ax.tick_params(labelsize=fontsize-4)
cbar.set_label('V/Å', fontsize=fontsize) # Seems to be broken in matplotlib 3.3.3 with cmr10 font

# Add labels
for ax, label in zip([ax_sim, ax_bcb, ax_ptcda], ['A', 'B', 'C']):
    ax.text(-0.1, 0.95, label, horizontalalignment='center',
        verticalalignment='center', transform=ax.transAxes, fontsize=fontsize)

plt.savefig('single_tip.pdf', bbox_inches='tight')
