
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
    Plot multiple AFM image stacks to files 0_input.png, 1_input.png, ... etc.

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
