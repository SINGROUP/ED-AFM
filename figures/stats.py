
import os
import numpy as np
import matplotlib.pyplot as plt

# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "font.serif": ["Computer Modern Roman"]
# })

stats_dir = './stats'   # Path to directory where csv files for losses are stored
fig_width = 160         # Figure width in mm
fontsize  = 8
dpi       = 300
linewidth = 1

data_independent = np.loadtxt(os.path.join(stats_dir, 'mse_independent.csv'), delimiter=',')
data_matched = np.loadtxt(os.path.join(stats_dir, 'mse_matched.csv'), delimiter=',')
data_independent_Xe = np.loadtxt(os.path.join(stats_dir, 'mse_independent_Xe.csv'), delimiter=',')
data_matched_Xe = np.loadtxt(os.path.join(stats_dir, 'mse_matched_Xe.csv'), delimiter=',')

data_constant = np.loadtxt(os.path.join(stats_dir, 'mse_constant.csv'), delimiter=',')
data_uniform = np.loadtxt(os.path.join(stats_dir, 'mse_uniform.csv'), delimiter=',')
data_normal = np.loadtxt(os.path.join(stats_dir, 'mse_normal.csv'), delimiter=',')

h = data_independent[:, 0]
data_independent = data_independent[:, 1:]
data_matched = data_matched[:, 1:]
data_independent_Xe = data_independent_Xe[:, 1:]
data_matched_Xe = data_matched_Xe[:, 1:]

amp = data_constant[:, 0]
data_constant = data_constant[:, 1:]
data_uniform = data_uniform[:, 1:]
data_normal = data_normal[:, 1:]

dh = h-5.3
m1 = data_independent.mean(axis=1)
m2 = data_matched.mean(axis=1)
m3 = data_independent_Xe.mean(axis=1)
m4 = data_matched_Xe.mean(axis=1)

m5 = data_constant.mean(axis=1)
m6 = data_uniform.mean(axis=1)
m7 = data_normal.mean(axis=1)

p1_05, p1_95 = np.percentile(data_independent, [5, 95], axis=1)
p2_05, p2_95 = np.percentile(data_matched, [5, 95], axis=1)
p3_05, p3_95 = np.percentile(data_independent_Xe, [5, 95], axis=1)
p4_05, p4_95 = np.percentile(data_matched_Xe, [5, 95], axis=1)

p5_05, p5_95 = np.percentile(data_constant, [5, 95], axis=1)
p6_05, p6_95 = np.percentile(data_uniform, [5, 95], axis=1)
p7_05, p7_95 = np.percentile(data_normal, [5, 95], axis=1)

r = 0.35
w = 0.25
h = w/r
b = 0.18
fig_width = 0.1/2.54*fig_width
fig = plt.figure(figsize=(fig_width, r*fig_width), dpi=dpi)
ax1 = plt.axes((0.070, b, w, h))
ax2 = plt.axes((0.415, b, w, h))
ax3 = plt.axes((0.732, b, w, h))

ax1.plot(dh, m1, 'b', linewidth=linewidth)
ax1.plot(dh, m2, 'r', linewidth=linewidth)
ax1.plot(dh, p1_05, '--b', linewidth=linewidth)
ax1.plot(dh, p1_95, '--b', linewidth=linewidth)
ax1.plot(dh, p2_05, '--r', linewidth=linewidth)
ax1.plot(dh, p2_95, '--r', linewidth=linewidth)
ax1.set_xlabel('$dh$(\AA)', fontsize=fontsize)
ax1.set_ylabel('MSE', fontsize=fontsize)
ax1.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3))
ax1.set_title('Both tips shifted', fontsize=fontsize+1)
ax1.legend(['Independent tips', 'Matched tips'], fontsize=fontsize-1)
ax1.text(-0.26, 1.08, 'A', transform=ax1.transAxes)

ax2.semilogy(dh, m3, 'b', linewidth=linewidth)
ax2.plot(dh, m4, 'r', linewidth=linewidth)
ax2.plot(dh, p3_05, '--b', linewidth=linewidth)
ax2.plot(dh, p3_95, '--b', linewidth=linewidth)
ax2.plot(dh, p4_05, '--r', linewidth=linewidth)
ax2.plot(dh, p4_95, '--r', linewidth=linewidth)
ax2.set_xlabel('$dh$(\AA)', fontsize=fontsize)
ax2.set_ylabel('MSE', fontsize=fontsize)
ax2.set_title('Only Xe shifted', fontsize=fontsize+1)
ax2.legend(['Independent tips', 'Matched tips'], fontsize=fontsize-1)
ax2.set_ylim((1e-6, 1e-2))
ax2.text(-0.32, 1.08, 'B', transform=ax2.transAxes)

ax3.plot(amp, m5, 'b', linewidth=linewidth)
ax3.plot(amp, m6, 'r', linewidth=linewidth)
ax3.plot(amp, m7, 'k', linewidth=linewidth)
ax3.plot(amp, p5_05, '--b', linewidth=linewidth)
ax3.plot(amp, p5_95, '--b', linewidth=linewidth)
ax3.plot(amp, p6_05, '--r', linewidth=linewidth)
ax3.plot(amp, p6_95, '--r', linewidth=linewidth)
ax3.plot(amp, p7_05, '--k', linewidth=linewidth)
ax3.plot(amp, p7_95, '--k', linewidth=linewidth)
ax3.set_xlabel('Noise amplitude', fontsize=fontsize)
ax3.set_ylabel('MSE', fontsize=fontsize)
ax3.set_title('Noise amp. distributions', fontsize=fontsize+1)
ax3.legend(['$\mathcal{C}(0.08)$', '$\mathcal{U}(0.16)$', '$\mathcal{N}(0.1)$'],
    fontsize=fontsize-1)
ax3.text(-0.22, 1.08, 'C', transform=ax3.transAxes)

for ax in [ax1, ax2, ax3]:
    ax.tick_params(axis='both', labelsize=fontsize-1)
    tx = ax.yaxis.get_offset_text()
    tx.set_fontsize(fontsize-1)
    tx.set_position((-0.15, 0))

plt.savefig('stats.pdf')
