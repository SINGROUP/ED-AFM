
import os
import shutil
import tarfile
import numpy as np
import matplotlib.pyplot as plt
from urllib.request import urlretrieve

from .visualization import _calc_plot_dim

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

def calculate_losses(model, true, preds=None, X=None):
    '''
    Calculate losses on each item of a batch for a keras model.
    Arguments:
        model: tensorflow.keras.Model.
        true: list of np.ndarray. Reference outputs.
        preds: list of np.ndarray. Predicted outputs.
        X: list of np.ndarray. Inputs used to make predictions in case preds==None.
    Returns: np.ndarray of shape (len(true), batch_size).
    Note: At least one of preds or X has to be provided.
    '''
    import tensorflow.keras.backend as K

    if preds is None and X is None:
        raise ValueError('preds and X cannot both be None')
    
    if preds is None:
        preds = model.predict_on_batch(X)
    
    losses = np.zeros((len(true), true[0].shape[0]))
    for i, (t, p) in enumerate(zip(true, preds)):
        for j in range(t.shape[0]):
            tj = K.variable(t[j])
            pj = K.variable(p[j])
            loss = model.compiled_loss._losses[i](tj, pj) # A private attribute, probably should not use this
            losses[i,j] = K.eval(loss)
    
    return losses

def download_molecules(save_path='./Molecules', verbose=1):
    '''
    Download database of molecules.
    Arguments:
        save_path: str. Path where the molecule xyz files will be saved.
        verbose: int 0 or 1. Whether to print output information.
    '''
    if not os.path.exists(save_path):
        download_url = 'https://www.dropbox.com/s/g6ngxz2qsju94db/Molecules_xyz3.tar?dl=1'
        temp_file = '.temp_molecule.tar'
        if verbose: print('Downloading molecule tar archive...')
        temp_file, info = urlretrieve(download_url, temp_file)
        if verbose: print('Extracting tar archive...')
        with tarfile.open(temp_file, 'r') as f:
            base_dir = os.path.normpath(f.getmembers()[0].name).split(os.sep)[0]
            f.extractall()
        shutil.move(base_dir, save_path)
        os.remove(temp_file)
    else:
        print(f'Target folder {save_path} already exists. Skipping downloading molecules.')
