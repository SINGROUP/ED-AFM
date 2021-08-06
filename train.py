
import os
import sys
import time
import pickle
import numpy as np

import torch
from torch import nn, optim

import edafm.preprocessing  as pp
from edafm.visualization    import make_input_plots, make_prediction_plots
from edafm.utils            import count_parameters, LossLogPlot, save_checkpoint, load_checkpoint
from edafm.training         import ImgLoss, make_dataloader
from edafm.models           import EDAFMNet

# Training options
epochs            = 50                                              # How many epochs to train
pred_batches      = 2                                               # How many batches to do example predictions on
data_dir          = './data'                                        # Directory where data is loaded from
model_dir         = './model'                                       # Directory where all output files are saved to
pred_dir          = os.path.join(model_dir, 'predictions/')         # Where to save predictions
checkpoint_dir    = os.path.join(model_dir, 'checkpoints/')         # Where to save model checkpoints
log_path          = os.path.join(model_dir, 'training.log')         # Where to save loss history during training
history_plot_path = os.path.join(model_dir, 'loss_history.png')     # Where to plot loss history during training
optimizer_path    = os.path.join(model_dir, 'optimizer_state.npz')  # Where to save optimizer state
descriptors       = ['ES']                                          # Labels for outputting information
num_workers       = 8                                               # Number of parallel workers to use for data loading
timings           = False                                           # Print timings for each batch
print_interval    = 10                                              # How often to print losses
device            = 'cuda'                                          # Device which model will be loaded onto

def apply_preprocessing(batch):

    X, Y, mols = batch

    X = [X[0], X[1]] # Pick CO and Xe
    X = [x[:, :, :, 2:8] for x in X ]
    pp.rand_shift_xy_trend(X, shift_step_max=0.02, max_shift_total=0.04)

    X, Y = pp.add_rotation_reflection(X, Y, reflections=True, multiple=3, crop=(128, 128))
    X, Y = pp.random_crop(X, Y, min_crop=0.75, max_aspect=1.25)
   
    pp.add_norm(X, per_layer=True)
    pp.add_gradient(X, c=0.3)
    pp.add_noise(X, c=0.1, randomize_amplitude=True, normal_amplitude=True)
    pp.add_cutout(X, n_holes=5)

    return X, Y

def batch_to_device(batch):
    X, Y = batch
    X = [x.to(device) for x in X]
    Y = [y.to(device) for y in Y] 
    return X, Y

def batch_to_host(X, Y, preds, losses):
    X = [x.cpu().numpy().squeeze() for x in X]
    Y = [y.cpu().numpy() for y in Y]
    preds = [p.cpu().numpy() for p in preds]
    losses = np.stack([l.cpu().numpy() for l in losses], axis=0)
    return X, Y, preds, losses

def loss_str(losses):
    s = f'{losses[0]:.6f}'
    return s

if __name__ == '__main__':

    start_time = time.time()

    # Create model directory
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Define model
    print(f'CUDA is AVAILABLE = {torch.cuda.is_available()}')
    model = EDAFMNet(device) 
    if device == 'cuda' and (n_gpus := torch.cuda.device_count()) > 1:
        print(f'Using {n_gpus} GPUs')
        model = nn.DataParallel(model)
    print(f'Model total parameters: {count_parameters(model)}')

    # Define optimizer
    lr_decay = 1e-5
    optimizer = optim.Adam(model.parameters(), lr = 1e-4)
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda b: 1.0/(1.0+lr_decay*b))

    # Define loss
    criterion = ImgLoss(loss_factors=[1])

    # Create datasets and dataloaders
    train_set, train_loader = make_dataloader(os.path.join(data_dir, 'train/'),apply_preprocessing,
        print_timings=timings, num_workers=num_workers)
    val_set, val_loader = make_dataloader(os.path.join(data_dir, 'val/'), apply_preprocessing,
        print_timings=timings, num_workers=num_workers)
    test_set, test_loader = make_dataloader(os.path.join(data_dir, 'test/'), apply_preprocessing,
        print_timings=timings, num_workers=num_workers)

    # Create a folder for model checkpoints
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        
    # Load checkpoint if available
    for init_epoch in reversed(range(1, epochs+1)):
        if os.path.exists( state_file := os.path.join(checkpoint_dir, f'model_{init_epoch}.pth') ):
            load_checkpoint(model, optimizer, state_file, lr_scheduler)
            init_epoch += 1
            break
    
    if init_epoch <= epochs:
        print(f'\n ========= Starting training from epoch {init_epoch}')
    else:
        print('Model already trained')
        
    # Setup logging
    log_path = os.path.join(model_dir, 'loss_log.csv')
    plot_path = os.path.join(model_dir, 'loss_history.png')
    logger = LossLogPlot(log_path, history_plot_path, descriptors)
    
    for epoch in range(init_epoch, epochs+1):

        print(f'\n === Epoch {epoch}')

        # Train
        train_losses = []
        epoch_start = time.time()
        if timings: t0 = epoch_start

        model.train()
        for ib, batch in enumerate(train_loader):
            
            # Transfer batch to device
            X, Y = batch_to_device(batch)

            if timings:
                if device == 'cuda': torch.cuda.synchronize()
                t1 = time.time()
            
            # Forward
            pred = model(X)
            losses = criterion(pred, Y)
            loss = losses[0]
            
            if timings: 
                if device == 'cuda': torch.cuda.synchronize()
                t2 = time.time()
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            train_losses.append([losses[0].item()])

            if ib == len(train_loader) or (ib+1) % print_interval == 0:
                eta = (time.time() - epoch_start) / (ib + 1) * ((len(train_loader)+len(val_loader)) - (ib + 1))
                mean_loss = np.mean(train_losses[-print_interval:], axis=0)
                loss_msg = f'Loss: {loss_str(mean_loss)}'
                print(f'Epoch {epoch}, Train Batch {ib+1}/{len(train_loader)} - {loss_msg} - ETA: {eta:.2f}s')
            
            if timings:
                if device == 'cuda': torch.cuda.synchronize()
                t3 = time.time()
                print(f'(Train) t0/Load Batch/Forward/Backward: {t0}/{t1-t0}/{t2-t1}/{t3-t2}')
                t0 = t3

        # Validate
        val_losses = []
        val_start = time.time()
        if timings: t0 = val_start
        
        model.eval()
        with torch.no_grad():
            
            for ib, batch in enumerate(val_loader):
                
                # Transfer batch to device
                X, Y = batch_to_device(batch)
                
                if timings: 
                    if device == 'cuda': torch.cuda.synchronize()
                    t1 = time.time()
                
                # Forward
                pred = model(X)
                losses = criterion(pred, Y)
                
                val_losses.append([losses[0].item()])
                
                if (ib+1) % print_interval == 0:
                    eta = (time.time() - epoch_start) / (len(train_loader) + ib + 1) * (len(val_loader) - (ib + 1))
                    print(f'Epoch {epoch}, Val Batch {ib+1}/{len(val_loader)} - ETA: {eta:.2f}s')
                
                if timings:
                    if device == 'cuda': torch.cuda.synchronize()
                    t2 = time.time()
                    print(f'(Val) t0/Load Batch/Forward: {t0}/{t1-t0}/{t2-t1}')
                    t0 = t2
        
        train_loss = np.mean(train_losses, axis=0)
        val_loss = np.mean(val_losses, axis=0)
        print(f'End of epoch {epoch}')
        print(f'Train loss: {loss_str(train_loss)}')
        print(f'Val loss: {loss_str(val_loss)}')

        epoch_end = time.time()
        train_step = (val_start - epoch_start) / len(train_loader)
        val_step = (epoch_end - val_start) / len(val_loader)
        print(f'Epoch time: {epoch_end - epoch_start:.2f}s - Train step: {train_step:.5f}s - Val step: {val_step:.5f}s')
        
        # Add losses to log
        logger.add_losses(train_loss, val_loss)
        logger.plot_history()
        
        # Save checkpoint and shuffle training set batches
        save_checkpoint(model, optimizer, epoch, checkpoint_dir, lr_scheduler)
        train_set.shuffle()
        
    # Save final model
    torch.save(model.module if isinstance(model, nn.DataParallel) else model,
        save_path := os.path.join(model_dir, 'model.pth'))
    print(f'\nModel saved to {save_path}')

    # Test
    print('\n ========= Evaluating on test set')
    
    eval_losses = []
    eval_start = time.time()
    if timings: t0 = eval_start
    
    model.eval()
    with torch.no_grad():
        
        for ib, batch in enumerate(test_loader):
                
            # Transfer batch to device
            X, Y = batch_to_device(batch)
            
            if timings: 
                if device == 'cuda': torch.cuda.synchronize()
                t1 = time.time()
            
            # Forward
            pred = model(X)
            losses = criterion(pred, Y)
            eval_losses.append([losses[0].item()])
            
            if timings: 
                if device == 'cuda': torch.cuda.synchronize()
                t2 = time.time()
              
            if (ib+1) % print_interval == 0:
                eta = (time.time() - eval_start) / (ib + 1) * (len(test_loader) - (ib + 1))
                print(f'Test Batch {ib+1}/{len(test_loader)} - ETA: {eta:.2f}s')
            
            if timings:
                if device == 'cuda': torch.cuda.synchronize()
                t3 = time.time()
                print(f'(Test) t0/Load Batch/Forward/Stats: {t0}/{t1-t0}/{t2-t1}/{t3-t2}')
                t0 = t3
    
    eval_loss = np.mean(eval_losses, axis=0)
    print(f'Test set loss: {loss_str(eval_loss)}')

    # Save test set loss to file
    with open(os.path.join(model_dir, 'test_loss.txt'),'w') as f:
        f.write(';'.join([str(l) for l in eval_loss]))
  
    # Make predictions
    print(f'\n ========= Predict on {pred_batches} batches from the test set')

    counter = 0
    pred_dir = os.path.join(model_dir, 'predictions/')
    with torch.no_grad():
        
        for ib, batch in enumerate(test_loader):
        
            if ib >= pred_batches: break
            
            # Transfer batch to device
            X, Y = batch_to_device(batch)
            
            # Forward
            preds = model(X)
            losses = criterion(preds, Y, separate_batch_items=True)

            # Back to host
            X, Y, preds, losses = batch_to_host(X, Y, preds, losses)

            # Visualize predictions
            make_prediction_plots(preds, Y, losses, descriptors, pred_dir, counter)
            make_input_plots(X, outdir=pred_dir, start_ind=counter, cmap = 'afmhot')

            # Save prediction data
            pickle.dump((X, Y, preds), open(os.path.join(pred_dir, f'pred_batch_{ib}.pickle'), 'wb'))
            
            counter += losses.shape[1]
    
    print(f'Done. Total time: {time.time() - start_time:.0f}s')
