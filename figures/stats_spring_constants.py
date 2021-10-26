
import os
import numpy as np
import matplotlib.pyplot as plt

# # Set matplotlib font rendering to use LaTex
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "font.serif": ["Computer Modern Roman"]
# })

stats_dir = './stats'   # Path to directory where csv files for losses are stored
fig_width = 130         # Figure width in mm
fontsize  = 8
dpi       = 300
linewidth = 1

data_lat = np.loadtxt(os.path.join(stats_dir, 'mse_spring_constants_lat.csv'), delimiter=',')
data_rad = np.loadtxt(os.path.join(stats_dir, 'mse_spring_constants_rad.csv'), delimiter=',')

k_lat = data_lat[:, 0]
data_lat = data_lat[:, 1:]

k_rad = data_rad[:, 0]
data_rad = data_rad[:, 1:]

m1 = data_lat.mean(axis=1)
m2 = data_rad.mean(axis=1)

print(m1.max()/m1.min())

p1_05, p1_95 = np.percentile(data_lat, [5, 95], axis=1)
p2_05, p2_95 = np.percentile(data_rad, [5, 95], axis=1)

r = 0.55
w = 0.38
h = w/r
b = 0.16
fig_width = 0.1/2.54*fig_width
fig = plt.figure(figsize=(fig_width, r*fig_width), dpi=dpi)
ax1 = plt.axes((0.100, b, w, h))
ax2 = plt.axes((0.600, b, w, h))

ax1.plot(k_lat, m1, 'b', linewidth=linewidth)
ax1.plot(k_lat, p1_05, '--b', linewidth=linewidth)
ax1.plot(k_lat, p1_95, '--b', linewidth=linewidth)
ax1.set_xlabel('$k_{\mathrm{lat}}$(N/m)', fontsize=fontsize)
ax1.set_ylabel('MSE', fontsize=fontsize)
ax1.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3))
ax1.set_title('Lateral spring constant', fontsize=fontsize+1)
ax1.text(-0.23, 1.08, 'A', transform=ax1.transAxes)

ax2.plot(k_rad, m2, 'b', linewidth=linewidth)
ax2.plot(k_rad, p2_05, '--b', linewidth=linewidth)
ax2.plot(k_rad, p2_95, '--b', linewidth=linewidth)
ax2.set_xlabel('$k_{\mathrm{rad}}$(N/m)', fontsize=fontsize)
ax2.set_ylabel('MSE', fontsize=fontsize)
ax2.set_title('Radial spring constant', fontsize=fontsize+1)
ax2.text(-0.23, 1.08, 'B', transform=ax2.transAxes)

for ax in [ax1, ax2]:
    ax.tick_params(axis='both', labelsize=fontsize-1)
    tx = ax.yaxis.get_offset_text()
    tx.set_fontsize(fontsize-1)
    tx.set_position((-0.15, 0))

plt.savefig('stats_spring_constants.pdf')
