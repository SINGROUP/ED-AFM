
import os
import sys
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

# # Set matplotlib font
# from matplotlib import rc
# rc('font', family = 'serif', serif = 'cmr10')
# plt.rcParams["font.serif"] = "cmr10"

def apply_preprocessing_sim(batch):

    X, Y, xyzs = batch

    X = [x[..., 2:8] for x in X]

    pp.add_norm(X)
    np.random.seed(0)
    pp.add_noise(X, c=0.08)

    return X, Y, xyzs

def apply_preprocessing_bcb(X, real_dim):
    x0_start = 4
    X[0] = X[0][..., x0_start:x0_start+6] # CO
    X = pp.interpolate_and_crop(X, real_dim)
    pp.add_norm(X)
    return X

def apply_preprocessing_ptcda(X, real_dim):
    x0_start = 2
    X[0] = X[0][..., x0_start:x0_start+6] # CO
    X = pp.interpolate_and_crop(X, real_dim)
    pp.add_norm(X)
    X = [x[:,:,6:78] for x in X]
    return X

data_dir    = '../data'     # Path to data directory
device      = 'cuda'        # Device to run inference on
fig_width   = 160           # Figure width in mm
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
    'distAbove'     : 5.3,
    'iZPPs'         : [8],
    'Qs'            : [[ -10, 20,  -10, 0 ]],
    'QZs'           : [[ 0.1,  0, -0.1, 0 ]]
}

# Paths to molecule xyz files
molecules = [os.path.join(data_dir, 'TTF-TDZ/mol.xyz')]

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
trainer = InverseAFMtrainer(afmulator, aux_maps, molecules, **generator_kwargs)

# Get simulation data
batch = next(iter(trainer))
X_sim, Y_sim, xyz = apply_preprocessing_sim(batch)
X_sim_cuda = [torch.from_numpy(x).unsqueeze(1).to(device) for x in X_sim]

# Load BCB data and preprocess
data_bcb = np.load(os.path.join(data_dir, 'BCB/data_CO_exp.npz'))
X_bcb = data_bcb['data']
afm_dim_bcb = (data_bcb['lengthX'], data_bcb['lengthY'])
X_bcb = apply_preprocessing_bcb([X_bcb[None]], afm_dim_bcb)
X_bcb_cuda = [torch.from_numpy(x.astype(np.float32)).unsqueeze(1).to(device) for x in X_bcb]

# Load PTCDA data and preprocess
data_ptcda = np.load(os.path.join(data_dir, 'PTCDA/data_CO_exp.npz'))
X_ptcda = data_ptcda['data']
afm_dim_ptcda = (data_ptcda['lengthX'], data_ptcda['lengthY'])
X_ptcda = apply_preprocessing_ptcda([X_ptcda[None]], afm_dim_ptcda)
X_ptcda_cuda = [torch.from_numpy(x.astype(np.float32)).unsqueeze(1).to(device) for x in X_ptcda]

# Load model
model = EDAFMNet(device=device, trained_weights='single-channel')

# Make predictions
with torch.no_grad():
    pred_sim, attentions_sim = model(X_sim_cuda, return_attention=True)
    pred_bcb, attentions_bcb = model(X_bcb_cuda, return_attention=True)
    pred_ptcda, attentions_ptcda = model(X_ptcda_cuda, return_attention=True)
    pred_sim = [p.cpu().numpy() for p in pred_sim]
    pred_bcb = [p.cpu().numpy() for p in pred_bcb]
    pred_ptcda = [p.cpu().numpy() for p in pred_ptcda]
    attentions_sim = [a.cpu().numpy() for a in attentions_sim]
    attentions_bcb = [a.cpu().numpy() for a in attentions_bcb]
    attentions_ptcda = [a.cpu().numpy() for a in attentions_ptcda]

# Make figure
fig_width = 0.1/2.54*fig_width
width_ratios = [4, 4, 6.9]
fig, axes = plt.subplots(1, 3, figsize=(fig_width, 2.6*fig_width/sum(width_ratios)),
    gridspec_kw={'width_ratios': width_ratios, 'wspace': 0.3})

tick_arrays = [
    [-0.03, 0.0, 0.03],
    [-0.05, 0.0, 0.05],
    [-0.1, 0.0, 0.1]
]

# Plot all predictions
for ax, ticks, pred, label in zip(axes, tick_arrays, [pred_sim, pred_bcb, pred_ptcda], ['A', 'B', 'C']):
    vmax = max(abs(pred[0][0].max()), abs(pred[0][0].min())); vmin = -vmax
    ax.imshow(pred[0][0].T, vmin=vmin, vmax=vmax, cmap='coolwarm', origin='lower')
    plt.axes(ax)
    m = cm.ScalarMappable(cmap=cm.coolwarm)
    m.set_array([vmin, vmax])
    cbar = plt.colorbar(m)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f'{i:.2f}'.replace('-', '$-$') for i in cbar.get_ticks()])
    cbar.ax.tick_params(labelsize=fontsize-1)
    ax.text(-0.1, 0.95, label, horizontalalignment='center',
        verticalalignment='center', transform=ax.transAxes, fontsize=fontsize)
    ax.set_axis_off()

# Calculate relative error metric for simulation
rel_abs_err_es = np.mean(np.abs(pred_sim[0] - Y_sim[0])) / np.ptp(Y_sim[0])
print(f'Relative error: {rel_abs_err_es*100:.2f}%')

plt.savefig('single_tip_predictions.pdf', bbox_inches='tight', dpi=dpi)
