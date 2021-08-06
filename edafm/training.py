
import os
import re
import glob
import time
import random
import numpy as np

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

class ImgDataset(Dataset):
    '''
    Pytorch dataset for loading AFM data.

    Arguments:
        data_dir: str. Path to directory where data is saved.
        preproc_fn: Python function. Preprocessing function which is applied to every batch.
        print_timings: Bool. Whether to print timings for each loaded batch.
    '''
    def __init__(self, data_dir, preproc_fn=None, print_timings=False):
        paths = glob.glob(os.path.join(data_dir, 'batch_*.npz'))
        self.batch_paths = sorted(paths, key=lambda s: int(re.search('batch_\d+', s)[0][6:]))
        if preproc_fn:
            self._preproc_fn = preproc_fn
        self.print_timings = print_timings

    def _preproc_fn(batch):
        Xs, Ys, mols = batch
        return Xs, Ys

    def _load_batch(self, file_path):
        batch = np.load(file_path, allow_pickle=True)
        return batch['arr_0']
        
    def __len__(self):
        return len(self.batch_paths)
        
    def __getitem__(self, idx):
        if self.print_timings: t0 = time.time()
        batch = self._load_batch(self.batch_paths[idx])
        if self.print_timings: t1 = time.time()
        batch = self._preproc_fn(batch)
        if self.print_timings:
            t2 = time.time()
            if wi := torch.utils.data.get_worker_info():
                msg = f'(Gen {wi.id}) '
            else:
                msg = '(Gen)'
            print(msg+f't0/load/preproc: {t0}/{t1-t0}/{t2-t1}')
        return batch
        
    def shuffle(self):
        '''Shuffle batch order'''
        random.shuffle(self.batch_paths)

class ImgLoss(nn.Module):
    '''
    Weighted mean squared loss for images.

    Arguments:
        loss_factors: list of int. Loss weights.
    '''

    def __init__(self, loss_factors):
        super().__init__()
        self.loss_factors = loss_factors

    def forward(self, pred, ref, separate_batch_items=False):
        
        assert len(pred) == len(ref) == len(self.loss_factors)

        losses = []
        total_loss = 0.0
        for p, r, f in zip(pred, ref, self.loss_factors):
            loss = torch.mean((p - r) ** 2, dim=(1,2))
            if not separate_batch_items:
                loss = loss.mean()
            losses.append(loss)
            total_loss += f*loss

        loss = [total_loss] + losses

        return loss

def _collate_fn(batch):
    X, Y = batch
    X = [torch.from_numpy(x).unsqueeze(1).float() for x in X]
    Y = [torch.from_numpy(y).float() for y in Y]
    return X, Y
    

def _worker_init_fn(worker_id):
    np.random.seed(int((time.time() % 1e5)*1000) + worker_id)
    
def make_dataloader(datadir, preproc_fn, print_timings=False, num_workers=8): #TODO memory pinning?
    '''
    Produce a dataset and dataloader from data directory.

    Arguments:
        datadir: str. Path to directory with data.
        preproc_fn: Python function. Preprocessing function to apply to each batch.
        print_timings: Boolean. Whether to print timings for each batch.
        num_workers: int. Number of parallel processes for data loading.

    Returns:
        dataset: ImgDataset.
        dataloader: torch.DataLoader.
    '''
    dataset = ImgDataset(datadir, preproc_fn, print_timings=print_timings)
    dataloader = DataLoader(
        dataset, 
        batch_size=None,
        shuffle=False,
        num_workers=num_workers,
        collate_fn = _collate_fn,
        worker_init_fn=_worker_init_fn,
        pin_memory=True
    )
    return dataset, dataloader
