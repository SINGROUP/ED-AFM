
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

# Set matplotlib font rendering to use LaTex
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "font.serif": ["Computer Modern Roman"]
# })

np.random.seed(0)

class Trainer(InverseAFMtrainer):
    
    # Override this method to set the Xe tip further
    def handle_distance(self):
        if self.afmulator.iZPP == 54:
            self.distAboveActive += 1.0
        super().handle_distance()

    # Override position handling to center on the non-Cu atoms
    def handle_positions(self):
        sw = self.afmulator.scan_window
        scan_center = np.array([sw[1][0] + sw[0][0], sw[1][1] + sw[0][1]]) / 2
        self.xyzs[:,:2] += scan_center - self.xyzs[self.Zs != 29,:2].mean(axis=0)

def apply_preprocessing_sim(batch):

    X, Y, xyzs = batch

    X = [x[..., 2:8] for x in X]

    pp.add_norm(X)
    np.random.seed(0)
    pp.add_noise(X, c=0.08)

    return X, Y, xyzs

def apply_preprocessing_exp(X, real_dim):

    # Pick slices
    x0_start, x1_start = 12, 12
    X[0] = X[0][..., x0_start:x0_start+6] # CO
    X[1] = X[1][..., x1_start:x1_start+6] # Xe

    X = pp.interpolate_and_crop(X, real_dim)
    pp.add_norm(X)

    # Crop
    X = [x[:, 36:-20, 26:-30] for x in X]
    
    print(X[0].shape)
    
    return X

data_dir    = '../data'             # Path to data directory
X_slices    = [0, 3, 5]             # Which AFM slices to plot
tip_names   = ['CO', 'Xe']          # AFM tip types
device      = 'cuda'                # Device to run inference on
fig_width   = 160                   # Figure width in mm
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
    'scan_dim'          : (144, 144, 20),
    'scan_window'       : ((2.0, 2.0, 7.0), (20.0, 20.0, 9.0)),
    'amplitude'         : 1.0,
    'df_steps'          : 10,
    'initFF'            : True
}

generator_kwargs = {
    'batch_size'    : 1,
    'distAbove'     : 5.35,
    'iZPPs'         : [8, 54],
    'Qs'            : [[ -10, 20,  -10, 0 ], [  30, -60,   30, 0 ]],
    'QZs'           : [[ 0.1,  0, -0.1, 0 ], [ 0.1,   0, -0.1, 0 ]]
}

# Paths to molecule xyz files
molecules = [os.path.join(data_dir, 'Water/mol.xyz')]

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
trainer = Trainer(afmulator, aux_maps, molecules, **generator_kwargs)

# Get simulation data
X_sim, ref, xyzs = apply_preprocessing_sim(next(iter(trainer)))
X_sim_cuda = [torch.from_numpy(x).unsqueeze(1).to(device) for x in X_sim]

# Load experimental data and preprocess
data1 = np.load(os.path.join(data_dir, 'Water/data_CO_exp.npz'))
X1 = data1['data']
afm_dim1 = (data1['lengthX'], data1['lengthY'])

data2 = np.load(os.path.join(data_dir, 'Water/data_Xe_exp.npz'))
X2 = data2['data']
afm_dim2 = (data2['lengthX'], data2['lengthY'])

print(X1.shape, X2.shape)
assert afm_dim1 == afm_dim2
afm_dim = afm_dim1
X_exp = apply_preprocessing_exp([X1[None], X2[None]], afm_dim)
X_exp_cuda = [torch.from_numpy(x.astype(np.float32)).unsqueeze(1).to(device) for x in X_exp]

# Load model
model = EDAFMNet(device=device, trained_weights='base')

# Get predictions
with torch.no_grad():
    pred_sim, attentions_sim = model(X_sim_cuda, return_attention=True)
    pred_exp, attentions_exp = model(X_exp_cuda, return_attention=True)
    pred_sim = [p.cpu().numpy() for p in pred_sim]
    pred_exp = [p.cpu().numpy() for p in pred_exp]
    attentions_sim = [a.cpu().numpy() for a in attentions_sim]
    attentions_exp = [a.cpu().numpy() for a in attentions_exp]

# Create figure grid
fig_width = 0.1/2.54*fig_width
width_ratios = [6, 8, 0.3]
height_ratios = [1, 1]
gap = 0.20
fig = plt.figure(figsize=(fig_width, 8.7*fig_width/sum(width_ratios)))
fig_grid = fig.add_gridspec(1, len(width_ratios), wspace=0.02, hspace=0, width_ratios=width_ratios)
afm_grid = fig_grid[0, 0].subgridspec(2, 1, wspace=0, hspace=gap, height_ratios=height_ratios)
pred_grid = fig_grid[0, 1].subgridspec(2, 2, wspace=0.02, hspace=gap, height_ratios=height_ratios)
cbar_grid = fig_grid[0, 2].subgridspec(1, 1, wspace=0, hspace=0)

# Get axes from grid
afm_sim_axes = afm_grid[0, 0].subgridspec(len(X_sim), len(X_slices), wspace=0.01, hspace=0.01).subplots(squeeze=False)
afm_exp_axes = afm_grid[1, 0].subgridspec(len(X_exp), len(X_slices), wspace=0.01, hspace=0.01).subplots(squeeze=False)
pred_sim_ax, ref_pc_ax, pred_exp_ax, geom_ax = pred_grid.subplots(squeeze=True).flatten()
cbar_ax = cbar_grid.subplots(squeeze=True)

# Plot AFM
for k, (axes, X) in enumerate(zip([afm_sim_axes, afm_exp_axes], [X_sim, X_exp])):
    for i, x in enumerate(X):
        for j, s in enumerate(X_slices):
            
            # Plot AFM slice
            im = axes[i, j].imshow(x[0,:,:,s].T, origin='lower', cmap='afmhot')
            axes[i, j].set_axis_off()
        
        # Put tip names to the left of the AFM image rows
        axes[i, 0].text(-0.1, 0.5, tip_names[i], horizontalalignment='center',
            verticalalignment='center', transform=axes[i, 0].transAxes,
            rotation='vertical', fontsize=fontsize)


# Figure out data limits
vmax = max(
    abs(pred_sim[0].min()), abs(pred_sim[0].max()),
    abs(pred_exp[0].min()), abs(pred_exp[0].max()),
    abs(ref[0].min()), abs(ref[0].max())
)
vmin = -vmax
print(ref[0].min())

# Plot predictions and references
pred_sim_ax.imshow(pred_sim[0][0].T, origin='lower', cmap='coolwarm', vmin=vmin, vmax=vmax)
pred_exp_ax.imshow(pred_exp[0][0].T, origin='lower', cmap='coolwarm', vmin=vmin, vmax=vmax)
ref_pc_ax.imshow(ref[0][0].T, origin='lower', cmap='coolwarm', vmin=vmin, vmax=vmax)

# Plot molecule geometry
xyz_img = np.flipud(imageio.imread(os.path.join(data_dir, 'Water/mol.png')))
geom_ax.imshow(xyz_img, origin='lower')

# Plot ES Map colorbar
m_es = cm.ScalarMappable(cmap=cm.coolwarm)
m_es.set_array((vmin, vmax))
cbar = plt.colorbar(m_es, cax=cbar_ax)
cbar.set_ticks([-0.1, -0.05, 0.0, 0.05, 0.1])
cbar.set_ticklabels([f'{i:.2f}'.replace('-', '$-$') for i in cbar.get_ticks()])
cbar_ax.tick_params(labelsize=fontsize-1)
cbar.set_label('V/Å', fontsize=fontsize) # Seems to be broken in matplotlib 3.3.3 with cmr10 font

 # Turn off axes ticks
pred_sim_ax.set_axis_off()
pred_exp_ax.set_axis_off()
ref_pc_ax.set_axis_off()
geom_ax.set_axis_off()

# Set titles
afm_sim_axes[0, len(X_slices)//2].set_title('AFM simulation', fontsize=fontsize, y=0.91)
afm_exp_axes[0, len(X_slices)//2].set_title('AFM experiment', fontsize=fontsize, y=0.91)
pred_sim_ax.set_title('Sim. prediction', fontsize=fontsize, y=0.96)
pred_exp_ax.set_title('Exp. prediction', fontsize=fontsize, y=0.96)
ref_pc_ax.set_title('Reference', fontsize=fontsize, y=0.96)

plt.savefig('water.pdf', bbox_inches='tight', dpi=dpi)
