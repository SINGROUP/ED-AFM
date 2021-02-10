
import os
import re
import glob
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import Callback

from .common_utils import load_optimizer_state, save_optimizer_state
from .visualization import _calc_plot_dim

class DataLoader(Sequence):
    '''
    Data loader for keras model training.

    Arguments:
        data_path: str. Path to directory where batch data is saved.
    '''
    def __init__(self, data_path):
        batch_ids = glob.glob(os.path.join(data_path, 'batch_*.npz'))
        self.batch_ids = sorted(batch_ids, key=lambda s: int(re.search('batch_\d+', s)[0][6:]))

    def apply_preprocessing(self, batch):
        '''
        Override this method to preprocess batch before handing it to the model.
        '''
        Xs, Ys, mols = batch
        return Xs, Ys

    def load_batch(self, index):
        '''
        Load batch from disk.
        
        Arguments:
            index: int. Index of batch to load.
        
        Returns:
            list of np.ndarray.
        '''
        path = os.path.join(self.batch_ids[index])
        batch = np.load(path, allow_pickle=True)
        return batch['arr_0']

    def __len__(self):
        return len(self.batch_ids)

    def __getitem__(self, index):
        batch = self.load_batch(index)
        return self.apply_preprocessing(batch)

    def shuffle(self):
        '''Shuffle batch order.'''
        random.shuffle(self.batch_ids)

class OptimizerResume(Callback):
    '''
    Save keras optimizer state at the end of epoch and load when resuming training.

    Arguments:
        model: tf.keras.Model. Model whose optimizer state to save and load.
        save_path: str. Path where to save/load the optimizer state.
    '''
    def __init__(self, model, save_path='./optimizer_state.npz'):
        super().__init__()
        self.model = model
        self.save_path = save_path
        load_optimizer_state(self.model, self.save_path)

    def on_epoch_end(self, epoch, log):
        '''Save current optimizer state.'''
        print('')
        save_optimizer_state(self.model, self.save_path)

class HistoryPlotter(Callback):
    '''
    Keras callback for plotting history in the middle of training.

    Arguments:
        log_path: str. Path where loss log is saved.
        plot_path: str. Path where plot of loss history is saved.
        loss_labels: list of str. Labels for different loss components.
    '''
    def __init__(self, log_path, plot_path, loss_labels):
        super().__init__()
        self.log_path = log_path
        self.plot_path = plot_path
        self.loss_labels = ['Total_weighted'] + loss_labels
        self._read_log()

    @property
    def losses(self):
        '''np.ndarray. Training losses.'''
        return np.array(self._losses)

    @property
    def val_losses(self):
        '''np.ndarray. Validation losses.'''
        return np.array(self._val_losses)

    def _read_log(self):
        self._losses = []
        self._val_losses = []
        if os.path.exists(self.log_path):
            with open(self.log_path, 'r') as f:
                f.readline()
                for line in f:
                    line = line.split(',')
                    lt = []
                    lv = []
                    for i in range(len(self.loss_labels)):
                        lt.append(float(line[1+i]))
                        lv.append(float(line[1+len(self.loss_labels)+i]))
                    self._losses.append([lt[-1]] + lt[:-1])
                    self._val_losses.append([lv[-1]] + lv[:-1])

    def on_epoch_end(self, epoch, logs):
        '''Add losses of current epoch'''
        lt = [logs['loss']]
        lv = [logs['val_loss']]
        for label in self.loss_labels[1:]:
            lt.append(logs[label+'_loss'])
            lv.append(logs['val_'+label+'_loss'])
        self._losses.append(lt)
        self._val_losses.append(lv)
        self.plot()

    def plot(self, show=False):
        '''
        Plot history of losses.

        Arguments:
            show: bool. Whether to show the plot on screen.
        '''
        x = range(1, len(self._losses)+1)
        n_rows, n_cols = _calc_plot_dim(len(self.loss_labels), f=0)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5*n_cols, 4*n_rows))
        for i, (label, ax) in enumerate(zip(self.loss_labels, axes.flatten())):
            ax.semilogy(x, self.losses[:,i])
            ax.semilogy(x, self.val_losses[:,i])
            ax.legend(['Training', 'Validation'])
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Loss')
            ax.set_title(label)
        fig.tight_layout()
        plt.savefig(self.plot_path)
        if show:
            plt.show()
        else:
            plt.close()