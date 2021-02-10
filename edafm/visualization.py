
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def _calc_plot_dim(n, f=0.3):
    rows = max(int(np.sqrt(n) - f), 1)
    cols = 1
    while rows*cols < n:
        cols += 1
    return rows, cols

def plot_input(X, constant_range=False, cmap='afmhot'):
    '''
    Plot single stack of AFM images.

    Arguments:
        X: np.ndarray of shape (x, y, z). AFM image to plot.
        constant_range: Boolean. Whether the different slices should use the same value range or not.
        cmap: str or matplotlib colormap. Colormap to use for plotting.

    Returns:
        matplotlib.pyplot.figure. Figure on which the image was plotted.
    '''
    rows, cols = _calc_plot_dim(X.shape[-1])
    fig = plt.figure(figsize=(3.2*cols,2.5*rows))
    vmax = X.max()
    vmin = X.min()
    for k in range(X.shape[-1]):
        fig.add_subplot(rows,cols,k+1)
        if constant_range:
            plt.imshow(X[:,:,k].T, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
        else:
            plt.imshow(X[:,:,k].T, cmap=cmap, origin="lower")
        plt.colorbar()
    plt.tight_layout()
    return fig
        
def make_input_plots(Xs, outdir='./predictions/', start_ind=0, constant_range=False, cmap='afmhot', verbose=1):
    '''
    Plot multiple AFM images to files 0_input.png, 1_input.png, ... etc.

    Arguments:
        Xs: list of np.ndarray of shape (batch, x, y, z). Input AFM images to plot.
        outdir: str. Directory where images are saved.
        start_ind: int. Save index increments by one for each image. The first index is start_ind.
        constant_range: Boolean. Whether the different slices should use the same value range or not.
        cmap: str or matplotlib colormap. Colormap to use for plotting.
        verbose: int 0 or 1. Whether to print output information.
    '''

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    img_ind = start_ind
    for i in range(Xs[0].shape[0]):
        
        for j in range(len(Xs)):
            
            plot_input(Xs[j][i], constant_range, cmap=cmap)
            
            save_name = f'{img_ind}_input'
            if len(Xs) > 1:
                save_name += str(j+1)
            save_name = os.path.join(outdir, save_name)
            save_name += '.png'
            plt.savefig(save_name)
            plt.close()

            if verbose > 0: print(f'Input image saved to {save_name}')

        img_ind += 1

def make_prediction_plots(preds=None, true=None, losses=None, descriptors=None, outdir='./predictions/', start_ind=0, verbose=1):
    '''
    Plot predictions/references for image descriptors.

    Arguments:
        preds: list of np.ndarray of shape (batch_size, x_dim, y_dim). Predicted maps.
            Each list element corresponds to one descriptor.
        true: list of np.ndarray of shape (batch_size, x_dim, y_dim). Reference maps.
            Each list element corresponds to one descriptor.
        losses: np.ndarray of shape (len(preds), batch_size). Losses for each predictions.
        descriptors: list of str. Names of descriptors. The name "ES" causes the coolwarm colormap to be used.
        outdir: str. Directory where images are saved.
        start_ind: int. Starting index for saved images.
        verbose: int 0 or 1. Whether to print output information.
    '''
    
    rows = (preds is not None) + (true is not None)
    if rows == 0:
        raise ValueError('preds and true cannot both be None.')
    elif rows == 1:
        data = preds if preds is not None else true
    else:
        assert len(preds) == len(true)

    cols = len(preds) if preds is not None else len(true)
    if descriptors is not None:
        assert len(descriptors) == cols
    
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    img_ind = start_ind
    batch_size = len(preds[0]) if preds is not None else len(true[0])

    for j in range(batch_size):
        
        fig, axes = plt.subplots(rows, cols)
        fig.set_size_inches(6*cols, 5*rows)

        if rows == 1:
            axes = np.expand_dims(axes, axis=0)
        if cols == 1:
            axes = np.expand_dims(axes, axis=1)

        for i in range(cols):
            
            top_ax = axes[0,i]
            bottom_ax = axes[-1,i]

            if rows == 2:
                p = preds[i][j]
                t = true[i][j]
                vmax = np.concatenate([p,t]).max()
                vmin = np.concatenate([p,t]).min()
            else:
                d = data[i][j]
                vmax = d.max()
                vmin = d.min()
                
            title1 = ''
            title2 = ''
            cmap = cm.viridis
            if descriptors is not None:
                descriptor = descriptors[i]
                title1 += f'{descriptor} Prediction'
                title2 += f'{descriptor} Reference'
                if descriptor == 'ES':
                    vmax = max(abs(vmax), abs(vmin))
                    vmin = -vmax
                    cmap = cm.coolwarm
            if losses is not None:
                title1 += f'\nMSE = {losses[i,j]:.2E}'

            if rows == 2:
                im1 = top_ax.imshow(p.T, vmax=vmax, vmin=vmin, cmap=cmap, origin='lower')
                im2 = bottom_ax.imshow(t.T, vmax=vmax, vmin=vmin, cmap=cmap, origin='lower')
                if title1:
                    top_ax.set_title(title1)
                    bottom_ax.set_title(title2)
            else:
                im1 = top_ax.imshow(d.T, vmax=vmax, vmin=vmin, cmap=cmap, origin='lower')
                if title1:
                    title = title1 if preds is not None else title2
                    top_ax.set_title(title)

            for axi in axes[:,i]:
                pos = axi.get_position()
                pos_new = [pos.x0, pos.y0, 0.8*(pos.x1-pos.x0), pos.y1-pos.y0]
                axi.set_position(pos_new)
            
            pos1 = top_ax.get_position()
            pos2 = bottom_ax.get_position()
            c_pos = [pos1.x1+0.1*(pos1.x1-pos1.x0), pos2.y0, 0.08*(pos1.x1-pos1.x0), pos1.y1-pos2.y0]
            cbar_ax = fig.add_axes(c_pos)
            fig.colorbar(im1, cax=cbar_ax)

        save_name = os.path.join(outdir, f'{img_ind}_pred.png')
        plt.savefig(save_name, bbox_inches="tight")
        plt.close()
        
        if verbose > 0: print(f'Prediction saved to {save_name}')
        img_ind += 1

def plot_ES_3D(ax, es, height_map, box_lims=((2, 18), (2, 18)), data_lims=None, colorbar=True, axis_off=False):
    '''
    Plot ES Map descriptor onto the 3D surface defined by corresponding Height Map.

    Arguments:
        ax: mpl_toolkits.mplot3d.axes3d.Axes3D. Axes onto which the descriptor is plotted.
        es: np.ndarray of shape (x, y). ES Map descriptor.
        height_map: np.ndarray of shape (x, y). Height Map descriptor.
        box_lims: tuple ((x_min, x_max), (y_min, y_max)). Plot region limits in angstroms.
        data_lims: None or tuple (vmin, vmax). Minimum and maximum value of data range.
            If None, calculated from ES Map.
        colorbar: bool. Plot colorbar for ES Map.
        axis_off: bool. Turn axis off.
    '''

    assert es.shape == height_map.shape

    x_dim, y_dim = es.shape[0], es.shape[1]
    X = np.linspace(box_lims[0][0], box_lims[0][1], x_dim)
    Y = np.linspace(box_lims[1][0], box_lims[1][1], y_dim)
    X, Y = np.meshgrid(X, Y, indexing='ij')

    if data_lims is None:
        vmax = max(abs(es.max()), abs(es.min()))
        vmin = -vmax
    else:
        vmin, vmax = data_lims

    es_norm = (es - vmin) / (vmax - vmin)
    col = cm.coolwarm(es_norm)

    surf = ax.plot_surface(X, Y, height_map, facecolors=col,
        linewidth=0, antialiased=False)

    if colorbar:
        m = cm.ScalarMappable(cmap=cm.coolwarm)
        m.set_array([vmin, vmax])
        cbar = plt.colorbar(m)

    x_aspect = box_lims[0][1] - box_lims[0][0]
    y_aspect = box_lims[1][1] - box_lims[1][0]
    z_aspect = height_map.max() - height_map.min()
    ax.set_box_aspect((x_aspect, y_aspect, z_aspect))
    ax.view_init(50, -70)
    if axis_off:
        ax.set_axis_off()
    else:
        ax.set_zticks([height_map.min(), height_map.max()])
        ax.set_xlabel('$x(\AA)$')
        ax.set_ylabel('$y(\AA)$')
        ax.set_zlabel('$z(\AA)$')

def make_prediction_plots_ES(preds, true, box_lims=((2, 18), (2, 18)), outdir='./predictions/', start_ind=0, verbose=1):
    '''
    Plot 3D representation of predictions/references for ES and Height Maps.

    Arguments:
        preds: list of two np.ndarrays of shape (batch_size, x_dim, y_dim). Predicted ES and Height Maps.
        true: list of two np.ndarrays of shape (batch_size, x_dim, y_dim). Reference ES and Height Maps.
        box_lims: tuple ((x_min, x_max), (y_min, y_max)). Plot region limits in angstroms.
        outdir: str. Directory where images are saved.
        start_ind: int. Starting index for saved images.
        verbose: int 0 or 1. Whether to print output information.
    '''

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    img_ind = start_ind
    for i in range(preds[0].shape[0]):

        p_es = preds[0][i]
        p_hm = preds[1][i]
        t_es = true[0][i]
        t_hm = true[1][i]

        fig = plt.figure(figsize=(15,8))

        vmax = max(abs(p_es.max()), abs(p_es.min()), abs(t_es.max()), abs(t_es.min()))
        vmin = -vmax
        data_lims = (vmin, vmax)

        ax1 = fig.add_subplot(121, projection='3d')
        plot_ES_3D(ax1, p_es, p_hm, box_lims=box_lims, data_lims=data_lims, colorbar=False, axis_off=False)
        ax1.set_title('Prediction', y=1.0, fontsize=16)

        ax2 = fig.add_subplot(122, projection='3d')
        plot_ES_3D(ax2, t_es, t_hm, box_lims=box_lims, data_lims=data_lims, colorbar=False, axis_off=False)
        ax2.set_title('Reference', y=1.0, fontsize=16)

        fig.tight_layout()

        # Colorbar
        fig.subplots_adjust(bottom=0.15)
        pos1 = ax1.get_position()
        pos2 = ax2.get_position()
        c_pos = [pos1.x0, pos1.y0-0.06, pos2.x1-pos1.x0, 0.04]
        cbar_ax = fig.add_axes(c_pos)
        m = cm.ScalarMappable(cmap=cm.coolwarm)
        m.set_array([vmin, vmax])
        cbar = plt.colorbar(m, cax=cbar_ax, orientation='horizontal')
        cbar.set_label('$V/\AA$', fontsize=14)

        save_name = os.path.join(outdir, f'{img_ind}_pred_ES.png')
        plt.savefig(save_name, bbox_inches="tight")
        plt.close()
        
        if verbose > 0: print(f'Prediction saved to {save_name}')
        img_ind += 1

def plot_ES_contour(ax, es, height_map, data_lims_es=None, data_lims_hm=None, levels=None,
        es_colorbar=True, hm_colorbar=True, axis_off=False):
    '''
    Plot ES Map descriptor and overlay it with a contour plot of the Height Map.

    Arguments:
        ax: mpl_toolkits.mplot3d.axes3d.Axes3D. Axes onto which the descriptor is plotted.
        es: np.ndarray of shape (x, y). ES Map descriptor.
        height_map: np.ndarray of shape (x, y). Height Map descriptor.
        data_lims_es: None or tuple (vmin, vmax). Minimum and maximum value of ES data range.
            If None, calculated from ES Map.
        data_lims_hm: None or tuple (vmin, vmax). Minimum and maximum value of Height data range.
            If None, calculated from Height Map.
        levels: list of float or None. Contour line levels to plot. If None, calculated automatically from Height Map.
        es_colorbar: bool. Plot colorbar for ES Map.
        hm_colorbar: bool. Plot colorbar for Height Map contour.
        axis_off: bool. Turn axis off.

    Returns:
        : tuple (es_cbar, hm_cbar). es_cbar: matplotlib.colorbar.Colorbar if es_colorbar==True or None otherwise.
        ES map colorbar instance. hm_cbar: matplotlib.colorbar.Colorbar if hm_colorbar==True or None otherwise.
        Height Map colorbar instance.
    '''

    if data_lims_es is None:
        vmax = max(abs(es.max()), abs(es.min()))
        vmin = -vmax
    else:
        vmin, vmax = data_lims_es

    if data_lims_hm is None:
        hm_min, hm_max = height_map.min(), height_map.max()
    else:
        hm_min, hm_max = data_lims_hm

    if levels is None:
        hm_range = hm_max - hm_min
        levels = np.linspace(hm_min+0.05*hm_range, hm_max-0.08*hm_range, 7)

    ax.contour(height_map.T, levels=levels, vmin=hm_min, vmax=hm_max)
    ax.imshow(es.T, vmin=vmin, vmax=vmax, cmap='coolwarm', origin='lower')

    if es_colorbar:
        m = cm.ScalarMappable(cmap=cm.coolwarm)
        m.set_array([vmin, vmax])
        es_cbar = plt.colorbar(m)
    else:
        es_cbar = None

    if hm_colorbar:
        m = cm.ScalarMappable(cmap=cm.viridis)
        m.set_array([height_map.min(), height_map.max()])
        hm_cbar = plt.colorbar(m)
    else:
        hm_cbar = None
    
    if axis_off:
        ax.set_axis_off()

    return es_cbar, hm_cbar
