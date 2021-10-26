
import os
import numpy as np
import matplotlib.pyplot as plt

# # Set matplotlib font rendering to use LaTex
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "font.serif": ["Computer Modern Roman"]
# })

data_dir    = '../data'     # Path to data directory
fig_width   = 160
fontsize    = 8
dpi         = 300

# Load data
water_CO = np.load(os.path.join(data_dir, 'Water/data_CO_exp.npz'))
water_Xe = np.load(os.path.join(data_dir, 'Water/data_Xe_exp.npz'))

fig = plt.figure(figsize=(0.1/2.54*fig_width, 5.0))
fig_grid = fig.add_gridspec(2, 1, wspace=0, hspace=0.1)

# Water plots
for i, (sample, label) in enumerate(zip([water_CO, water_Xe], ['E', 'F'])):
    d = sample['data']
    l = sample['lengthX']
    axes = fig_grid[i, 0].subgridspec(3, 8, wspace=0.02, hspace=0.02).subplots().flatten()
    for j, ax in enumerate(axes):
        if j < d.shape[-1]:
            ax.imshow(d[:,:,j].T, origin='lower', cmap='afmhot')
        ax.axis('off')
    axes[0].text(-0.3, 0.8, label, horizontalalignment='center',
        verticalalignment='center', transform=axes[0].transAxes, fontsize=fontsize)
    axes[0].plot([50, 50+5/l*d.shape[0]], [470, 470], color='k')

plt.savefig('afm_stacks2.pdf', bbox_inches='tight', dpi=dpi)
