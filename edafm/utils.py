
import os
import numpy as np
import matplotlib.pyplot as plt

from .visualization import _calc_plot_dim

elements = ['H' , 'He', 
            'Li', 'Be',  'B',  'C',  'N',  'O',  'F', 'Ne', 
            'Na', 'Mg', 'Al', 'Si',  'P',  'S', 'Cl', 'Ar',
             'K', 'Ca', 
            'Sc', 'Ti',  'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
                        'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
            'Rb', 'Sr',
             'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
                        'In', 'Sn', 'Sb', 'Te',  'I', 'Xe'
]

def read_xyzs(file_paths):
    '''
    Read molecule xyz files.
    Arguments:
        file_paths: list of str. Paths to xyz files
    Returns: list of np.array of shape (num_atoms, 4) or (num_atoms, 5). Each row
             corresponds to one atom with [x, y, z, element] or [x, y, z, charge, element].
    '''
    mols = []
    for file_path in file_paths:
        with open(file_path, 'r') as f:
            N = int(f.readline().strip())
            f.readline()
            atoms = []
            for line in f:
                line = line.strip().split()
                try:
                    elem = int(line[0])
                except ValueError:
                    elem = elements.index(line[0]) + 1
                posc = [float(p) for p in line[1:]]
                atoms.append(posc + [elem])
        mols.append(np.array(atoms))
    return mols

def write_to_xyz(molecule, outfile='./pos.xyz', verbose=1):
    '''
    Write molecule into xyz file.
    Arguments:
        molecule: np.array of shape (num_atoms, 4) or (num_atoms, 5). Molecule to write.
                  Each row corresponds to one atom with [x, y, z, element] or [x, y, z, charge, element].
        outfile: str. Path where xyz file will be saved.
        verbose: int 0 or 1. Whether to print output information.
    '''
    molecule = molecule[molecule[:,-1] > 0]
    with open(outfile, 'w') as f:
        f.write(f'{len(molecule)}\n\n')
        for atom in molecule:
            f.write(f'{int(atom[-1])}\t')
            for i in range(len(atom)-1):
                f.write(f'{atom[i]:10.8f}\t')
            f.write('\n')
    if verbose > 0: print(f'Molecule xyz file saved to {outfile}')

def batch_write_xyzs(xyzs, outdir='./', start_ind=0, verbose=1):
    '''
    Write a batch of xyz files 0_mol.xyz, 1_mol.xyz, ...
    Arguments:
        xyzs: list of np.array of shape (num_atoms, 4) or (num_atoms, 5). Molecules to write.
        outdir: str. Directory where files are saved.
        start_ind: int. Index where file numbering starts.
        verbose: int 0 or 1. Whether to print output information.
    '''
    if outdir and not os.path.exists(outdir):
        os.makedirs(outdir)
    ind = start_ind
    for xyz in xyzs:
        write_to_xyz(xyz, os.path.join(outdir, f'{ind}_mol.xyz'), verbose=verbose)
        ind += 1

def save_optimizer_state(model, save_path):
    '''
    Save keras optimizer state.
    Arguments:
        model: tensorflow.keras.Model.
        save_path: str. Path where optimizer state file is saved to.
    '''
    weights = model.optimizer.get_weights()
    np.savez(save_path, weights)
    print(f'Optimizer weights saved to {save_path}')

def load_optimizer_state(model, load_path):
    '''
    Load keras optimizer state.
    Arguments:
        model: tensorflow.keras.Model.
        save_path: str. Path where optimizer state file is loaded from.
    '''

    if not os.path.exists(load_path):
        print('No optimizer weights found')
        return 0

    weights = list(np.load(load_path, allow_pickle=True)['arr_0'])
    model.optimizer._create_all_weights(model.trainable_variables)

    try:
        model.optimizer.set_weights(weights)
    except:
        print('Optimizer weights found, but could not load them. Probably incompatible model.')
        return 0
    print(f'Optimizer weights loaded from {load_path}')

    return 1

class LossLogPlot:
    '''
    Log and plot model training loss history. Add epoch losses with add_losses and plot with plot_history.
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
        
    def plot_history(self, show=False):
        '''
        Plot and save history of current losses into self.plot_path.
        Arguments:
            show: Bool. Whether to show the plot on screen.
        '''
        x = range(1, self.epoch+1)
        n_rows, n_cols = _calc_plot_dim(len(self.loss_labels), f=0)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(7*n_cols, 6*n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.expand_dims(axes, axis=0)
        for i, (label, ax) in enumerate(zip(self.loss_labels, axes.flatten())):
            ax.semilogy(x, self.train_losses[:,i])
            ax.semilogy(x, self.val_losses[:,i])
            ax.legend(['Training', 'Validation'])
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Loss')
            if self.loss_weights[i]:
                label = f'{label} (x {self.loss_weights[i]})'
            ax.set_title(label)
        fig.tight_layout()
        plt.savefig(self.plot_path)
        print(f'Loss history plot saved to {self.plot_path}')
        if show:
            plt.show()
        else:
            plt.close()