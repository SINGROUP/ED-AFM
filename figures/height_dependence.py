
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('../') # Make sure ED-AFM is on path
import edafm.preprocessing as pp
import edafm.visualization as vis
from edafm.models import ESUNet, load_pretrained_weights

# # Set matplotlib font
# from matplotlib import rc
# rc('font', family = 'serif', serif = 'cmr10')

def apply_preprocessing_bcb(X, real_dim):
    X[0] = X[0][..., 0:12] # CO
    X[1] = X[1][..., 4:16] # Xe
    X = pp.interpolate_and_crop(X, real_dim)
    X_ = []
    for z in range(X[0].shape[-1]-10+1):
        X_.append([x[..., z:z+10] for x in X])
    X = [np.concatenate([x[i] for x in X_], axis=0) for i in range(len(X))]
    pp.add_norm(X)
    return X

def apply_preprocessing_ptcda(X, real_dim):
    X[0] = X[0][..., :12] # CO
    X[1] = X[1][..., :12] # Xe
    X = pp.interpolate_and_crop(X, real_dim)
    X_ = []
    for z in range(X[0].shape[-1]-10+1):
        X_.append([x[..., z:z+10] for x in X])
    X = [np.concatenate([x[i] for x in X_], axis=0) for i in range(len(X))]
    pp.add_norm(X)
    X = [x[:,:80,8:80] for x in X]
    return X

# Options
X_slices    = [0, 9]                 # Which AFM slices to plot
tip_names   = ['CO-AFM', 'Xe-AFM']   # AFM tip types
fontsize    = 24

# Load BCB data and preprocess
data_bcb = np.load('./data/BCB_CO.npz')
afm_dim_bcb = (data_bcb['lengthX'], data_bcb['lengthY'])
X_bcb_CO = data_bcb['data']
X_bcb_Xe = np.load('./data/BCB_Xe.npz')['data']
X_bcb = apply_preprocessing_bcb([X_bcb_CO[None], X_bcb_Xe[None]], afm_dim_bcb)
l_bcb = len(X_bcb[0])

# Load PTCDA data and preprocess
data_ptcda = np.load('./data/PTCDA_CO.npz')
afm_dim_ptcda = (data_ptcda['lengthX'], data_ptcda['lengthY'])
X_ptcda_CO = data_ptcda['data']
X_ptcda_Xe = np.load('./data/PTCDA_Xe.npz')['data']
X_ptcda = apply_preprocessing_ptcda([X_ptcda_CO[None], X_ptcda_Xe[None]], afm_dim_ptcda)
l_ptcda = len(X_ptcda[0])

# Load model for BCB
model_bcb = ESUNet(n_in=2, n_out=2, input_shape=X_bcb[0].shape[1:], last_relu=[False, True])
load_pretrained_weights(model_bcb, tip_type='CO-Xe')

# Load model for PTCDA
model_ptcda = ESUNet(n_in=2, n_out=2, input_shape=X_ptcda[0].shape[1:], last_relu=[False, True])
load_pretrained_weights(model_ptcda, tip_type='CO-Xe')

# Make predictions
pred_bcb = model_bcb.predict(X_bcb)
pred_ptcda = model_ptcda.predict(X_ptcda)

# Create figure grid
fig = plt.figure(figsize=(20, 23))
fig_axes = fig.add_gridspec(l_bcb+l_ptcda, 2*len(X_slices)+1, wspace=0.01, hspace=0.01,
    height_ratios=[1]*l_bcb+[0.9]*l_ptcda).subplots(squeeze=False)
bcb_axes = fig_axes[:l_bcb, :]
ptcda_axes = fig_axes[l_bcb:, :]

# Do for both BCB and PTCDA
for axes, X, pred, title in [(bcb_axes, X_bcb, pred_bcb, 'BCB'), (ptcda_axes, X_ptcda, pred_ptcda, 'PTCDA')]:

    # Loop over distances
    for i in range(len(X[0])):

        # Plot AFM
        for j, x in enumerate(X):
            for k, s in enumerate(X_slices):
                ax = axes[i, j*len(X_slices)+k]
                im = ax.imshow(x[i,:,:,s].T, origin='lower', cmap='afmhot')
                ax.set_axis_off()

        # Plot predictions
        vis.plot_ES_contour(axes[i, -1], pred[0][i], pred[1][i],
            es_colorbar=False, hm_colorbar=False, axis_off=True)

    # Set titles
    p0 = axes[0, 0].get_position().get_points()
    p1 = axes[-1, 0].get_position().get_points()
    plt.text(p1[0][0]-0.016, (p1[0][1]+p0[1][1])/2, title, fontsize=fontsize, transform=fig.transFigure,
        horizontalalignment='center', verticalalignment='center', rotation='vertical')
    px = [p0[0][0]-0.005, p1[0][0]-0.005]
    py = [p0[1][1]-0.01, p1[0][1]+0.01]
    fig_axes[0, 0].plot(px, py, transform=fig.transFigure, clip_on=False, color='k', linewidth=2)

# Set titles
for i, tip in enumerate(tip_names):
    p0 = fig_axes[0, len(X_slices)*i].get_position().get_points()
    p1 = fig_axes[0, len(X_slices)*(i+1)-1].get_position().get_points()
    plt.text((p1[1][0]+p0[0][0])/2, p1[1][1]+0.013, tip, fontsize=fontsize,
        transform=fig.transFigure, horizontalalignment='center')
    px = [p0[0][0]+0.01, p1[1][0]-0.01]
    py = [p0[1][1]+0.005, p1[1][1]+0.005]
    fig_axes[0, len(X_slices)*i].plot(px, py, transform=fig.transFigure, clip_on=False, color='k', linewidth=2)

p = fig_axes[0, -1].get_position().get_points()
plt.text((p[1][0]+p[0][0])/2, p[1][1]+0.013, 'Prediction', fontsize=fontsize,
    transform=fig.transFigure, horizontalalignment='center')
px = [p[0][0]+0.01, p[1][0]-0.01]
py = [p[1][1]+0.005, p[1][1]+0.005]
fig_axes[0, -1].plot(px, py, transform=fig.transFigure, clip_on=False, color='k', linewidth=2)

plt.savefig('height_dependence.pdf', bbox_inches='tight')