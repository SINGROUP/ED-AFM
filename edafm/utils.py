
import os
import shutil
import tarfile
import numpy as np
import matplotlib.pyplot as plt
from urllib.request import urlretrieve

import torch

from .visualization import _calc_plot_dim

def download_molecules(save_path='./Molecules', verbose=1):
    '''
    Download database of molecules.
    
    Arguments:
        save_path: str. Path where the molecule xyz files will be saved.
        verbose: int 0 or 1. Whether to print progress information.
    '''
    if not os.path.exists(save_path):
        download_url = 'https://www.dropbox.com/s/z4113upq82puzht/Molecules_rebias_210611.tar.gz?dl=1'
        temp_file = '.temp_molecule.tar'
        if verbose: print('Downloading molecule tar archive...')
        temp_file, info = urlretrieve(download_url, temp_file)
        if verbose: print('Extracting tar archive...')
        with tarfile.open(temp_file, 'r') as f:
            base_dir = os.path.normpath(f.getmembers()[0].name).split(os.sep)[0]
            f.extractall()
        if verbose: print('Done extracting.')
        shutil.move(base_dir, save_path)
        os.remove(temp_file)
        if verbose: print(f'Moved files to {save_path}.')
    else:
        print(f'Target folder {save_path} already exists. Skipping downloading molecules.')

def count_parameters(module):
    '''
    Count pytorch module parameters.
    
    Arguments:
        module: torch.nn.Module.
    '''
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

class LossLogPlot:
    '''
    Log and plot model training loss history.

    Arguments:
        log_path: str. Path where loss log is saved.
        plot_path: str. Path where plot of loss history is saved.
        loss_labels: list of str. Labels for different loss components.
        loss_weights: list of int or str. Weights for different loss components.
            Empty string for no weight (e.g. Total loss).
    '''
    def __init__(self, log_path, plot_path, loss_labels, loss_weights=None):
        self.log_path = log_path
        self.plot_path = plot_path
        self.loss_labels = loss_labels
        if not loss_weights:
            self.loss_weights = [''] * len(self.loss_labels)
        else:
            assert len(loss_weights) == len(loss_labels)
            self.loss_weights = loss_weights
        self.train_losses = np.empty((0, len(loss_labels)))
        self.val_losses = np.empty((0, len(loss_labels)))
        self.epoch = 0
        self._init_log()

    def _init_log(self):
        if not(os.path.isfile(self.log_path)):
            with open(self.log_path, 'w') as f:
                f.write('epoch')
                for i, label in enumerate(self.loss_labels):
                    label = f';train_{label}'
                    if self.loss_weights[i]:
                        label += f' (x {self.loss_weights[i]})'
                    f.write(label)
                for i, label in enumerate(self.loss_labels):
                    label = f';val_{label}'
                    if self.loss_weights[i]:
                        label += f' (x {self.loss_weights[i]})'
                    f.write(label)
                f.write('\n')
            print(f'Created log at {self.log_path}')
        else:
            with open(self.log_path, 'r') as f:
                header = f.readline().rstrip('\r\n').split(';') 
                hl = (len(header)-1) // 2
                if len(self.loss_labels) != hl:
                    raise ValueError(f'The length of the given list of loss names and the length of the header of the existing log at {self.log_path} do not match.')
                for line in f:
                    line = line.rstrip('\n').split(';')
                    self.train_losses = np.append(self.train_losses, [[float(s) for s in line[1:hl+1]]], axis=0)
                    self.val_losses = np.append(self.val_losses, [[float(s) for s in line[hl+1:]]], axis=0)
                    self.epoch += 1
            print(f'Using existing log at {self.log_path}')
    
    def add_losses(self, train_loss, val_loss):
        '''
        Add losses to log.

        Arguments:
            train_loss: list of floats of length len(self.loss_labels). Training losses for the epoch.
            val_loss: list of floats of length len(self.loss_labels). Validation losses for the epoch.
        '''
        self.epoch += 1
        self.train_losses = np.append(self.train_losses, [train_loss], axis=0)
        self.val_losses = np.append(self.val_losses, [val_loss], axis=0)
        with open(self.log_path, 'a') as f:
            f.write(str(self.epoch))
            for l in train_loss:
                f.write(f';{l}')
            for l in val_loss:
                f.write(f';{l}')
            f.write('\n')
        
    def plot_history(self, show=False, verbose=1):
        '''
        Plot and save history of current losses into plot_path.

        Arguments:
            show: Bool. Whether to show the plot on screen.
            verbose: int 0 or 1. Whether to print output information.
        '''
        x = range(1, self.epoch+1)
        n_rows, n_cols = _calc_plot_dim(len(self.loss_labels), f=0)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(7*n_cols, 6*n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.expand_dims(axes, axis=0)
        for i, (label, ax) in enumerate(zip(self.loss_labels, axes.flatten())):
            ax.semilogy(x, self.train_losses[:,i],'-bx')
            ax.semilogy(x, self.val_losses[:,i],'-gx')
            ax.legend(['Training', 'Validation'])
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Loss')
            if self.loss_weights[i]:
                label = f'{label} (x {self.loss_weights[i]})'
            ax.set_title(label)
        fig.tight_layout()
        plt.savefig(self.plot_path)
        if verbose: print(f'Loss history plot saved to {self.plot_path}')
        if show:
            plt.show()
        else:
            plt.close()

def save_checkpoint(model, optimizer, epoch, save_dir, lr_scheduler=None, verbose=1):
    '''
    Save pytorch checkpoint.

    Arguments:
        model: torch.nn.Module. Model whose state to save.
        optimizer: torch.optim.Optimizer. Optimizer whose state to save
        epoch: int. Training epoch.
        save_dir: str. Directory to save in.
        lr_scheduler: torch.optim.lr_scheduler or None. If not None, save state of this scheduler.
        verbose: int 0 or 1. Whether to print information.
    '''

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if hasattr(model, 'module'):
        model = model.module

    state = {
        'model_params': model.state_dict(),
        'optim_params': optimizer.state_dict(),
    }
    if lr_scheduler is not None:
        state['scheduler_params'] = lr_scheduler.state_dict()
        
    torch.save(state, os.path.join(save_dir, f'model_{epoch}.pth'))
    if verbose: print(f'Model, optimizer weights on epoch {epoch} saved to {save_dir}')
        
def load_checkpoint(model, optimizer=None, file_name='./model.pth', lr_scheduler=None, verbose=1):
    '''
    Load pytorch checkpoint.

    Arguments:
        model: torch.nn.Module. Model where parameters are loaded to.
        optimizer: torch.optim.Optimizer or None. If not None, load state to this optimizer.
        file_name: str. Checkpoint file to load from.
        lr_scheduler: torch.optim.lr_scheduler or None. If not None, try loading state to this scheduler.
        verbose: int 0 or 1. Whether to print information.
    '''

    state = torch.load(file_name)
    model.load_state_dict(state['model_params'])

    if optimizer:
        optimizer.load_state_dict(state['optim_params'])
        msg = f'Model, optimizer weights loaded from {file_name}'
    else:
        msg = f'Model weights loaded from {file_name}'

    if lr_scheduler is not None:
        try:
        	lr_scheduler.load_state_dict(state['scheduler_params'])
        except:
            print('Learning rate scheduler parameters could not be loaded.')
            
    if verbose: print(msg)
