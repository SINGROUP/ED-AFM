
import numpy as np
import matplotlib.pyplot as plt

# Load data
bcb_CO = np.load('./data/BCB_CO.npz')
bcb_Xe = np.load('./data/BCB_Xe.npz')
ptcda_CO = np.load('./data/PTCDA_CO.npz')
ptcda_Xe = np.load('./data/PTCDA_Xe.npz')

fontsize = 16
height_ratios = [2, 2, 2.45, 2.45]
fig = plt.figure(figsize=(7.5, sum(height_ratios)))
fig_grid = fig.add_gridspec(4, 1, wspace=0, hspace=0.1, height_ratios=height_ratios)

# BCB plots
for i, (sample, label) in enumerate(zip([bcb_CO, bcb_Xe], ['A', 'B'])):
    d = sample['data']
    l = sample['lengthX']
    print(l)
    axes = fig_grid[i, 0].subgridspec(2, 8, wspace=0.02, hspace=0.02).subplots().flatten()
    for j, ax in enumerate(axes):
        if j < d.shape[-1]:
            ax.imshow(d[:,:,j].T, origin='lower', cmap='afmhot')
        ax.axis('off')
    axes[0].text(-0.3, 0.8, label, horizontalalignment='center',
            verticalalignment='center', transform=axes[0].transAxes, fontsize=fontsize)
    axes[0].plot([50, 50+5/l*d.shape[0]], [470, 470], color='k')

# PTCDA plots
for i, (sample, label) in enumerate(zip([ptcda_CO, ptcda_Xe], ['C', 'D'])):
    d = sample['data']
    l = sample['lengthX']
    print(l)
    axes = fig_grid[i+2, 0].subgridspec(3, 6, wspace=0.02, hspace=0.02).subplots().flatten()
    for j, ax in enumerate(axes):
        if j < d.shape[-1]:
            ax.imshow(d[:,:,j].T, origin='lower', cmap='afmhot')
        ax.axis('off')
    axes[0].text(-0.22, 0.7, label, horizontalalignment='center',
            verticalalignment='center', transform=axes[0].transAxes, fontsize=fontsize)
    axes[0].plot([20, 20+5/l*d.shape[0]], [135, 135], color='k')

plt.savefig('afm_full.pdf', bbox_inches='tight')
