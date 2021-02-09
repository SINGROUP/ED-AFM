
import os
import numpy as np
import matplotlib as mpl; mpl.use('agg')

import tensorflow as tf
from tensorflow.keras           import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.utils     import plot_model

from edafm.training import DataLoader, HistoryPlotter, OptimizerResume
from edafm.models import ESUNet
import edafm.visualization as vis
import edafm.common_utils  as cu
import edafm.preprocessing as pp

# Make TF not allocate all memory on GPU right away
# https://www.tensorflow.org/guide/gpu
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

class Loader(DataLoader):

    def apply_preprocessing(self, batch):

        Xs, Ys, xyz = batch

        Xs = [X[:10] for X in Xs]
        Ys = [Y[:10] for Y in Ys]
        
        pp.add_norm(Xs, per_layer=True)
        pp.add_gradient(Xs, c=0.3)
        pp.add_noise(Xs, c=0.1, randomize_amplitude=False)
        pp.rand_shift_xy(Xs, c=0.02)
        pp.add_cutout(Xs, n_holes=5)
        
        Ys = Ys[0]
        Ys = [Ys[:,:,:,i] for i in range(Ys.shape[-1])]

        Xs, Ys = pp.add_rotation_reflection(Xs, Ys, rotations=True, reflections=True,
            multiple=2, crop=(128, 128))

        Xs = list(reversed(Xs)) # Only for CO-Xe tip combination

        return Xs, Ys

# Training options
loss_weights      = [1.0, 0.1]                                      # Weights for balancing the loss
epochs            = 50                                              # How many epochs to train
pred_batches      = 2                                               # How many batches to do predictions on
input_shape       = (128,128,10)                                    # Input size of model
# data_dir          = './data'                                        # Directory where data is loaded from
data_dir          = '/media/storage/ES_data'                   
model_dir         = './model'                                       # Directory where all output files are saved to
pred_dir          = os.path.join(model_dir, 'predictions/')         # Where to save predictions
checkpoint_dir    = os.path.join(model_dir, 'checkpoints/')         # Where to save model checkpoints
log_path          = os.path.join(model_dir, 'training.log')         # Where to save loss history during training
history_plot_path = os.path.join(model_dir, 'loss_history.png')     # Where to plot loss history during training
optimizer_path    = os.path.join(model_dir, 'optimizer_state.npz')  # Where to save optimizer state
descriptors       = ['ES', 'Height_Map']                            # Labels for outputting information

# Create output folder
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# Define model
model = ESUNet(n_in=2, n_out=2, input_shape=input_shape, last_relu=[False, True], labels=descriptors) # CO-Xe, Cl-CO
# model = ESUNet(n_in=2, n_out=2, input_shape=input_shape, last_relu=[False, False], labels=descriptors) # Cl-Xe
# model = ESUNet(n_in=1, n_out=2, input_shape=input_shape, last_relu=[False, True], labels=descriptors) # CO
optimizer = optimizers.Adam(lr=0.001, decay=1e-5)
model.compile(optimizer, 'mse', loss_weights=loss_weights)
model.summary()
    
# Setup data loading
train_loader = Loader(os.path.join(data_dir, 'train/'))
val_loader   = Loader(os.path.join(data_dir, 'val/'))
test_loader  = Loader(os.path.join(data_dir, 'test/'))

# Setup callbacks
checkpointer = ModelCheckpoint(checkpoint_dir+'weights_{epoch:d}.h5', save_weights_only=True)
logger = CSVLogger(log_path, append=True)
plotter = HistoryPlotter(log_path, history_plot_path, descriptors)
optim_resume = OptimizerResume(model, optimizer_path)
callbacks = [checkpointer, logger, plotter, optim_resume]

# Resume previous epoch if exists
init_epoch = 0
model_file = None
for i in range(1, epochs+1):
    cp_file = os.path.join(checkpoint_dir, 'weights_%d.h5' % i)
    if os.path.exists(cp_file):
        init_epoch += 1
        model_file = cp_file
    else:
        break
if init_epoch > 0:
    model.load_weights(model_file)
    print('Model weights loaded from '+model_file)
    
# Fit model
model.fit(
    train_loader,
    validation_data=val_loader,
    epochs=epochs,
    initial_epoch=init_epoch,
    callbacks=callbacks,
    shuffle=True,
    verbose=1
)

# Save final weights
model.save_weights(os.path.join(model_dir, 'model.h5'))

# Test model
test_loss = model.evaluate(test_loader, verbose=1)
print(f'Losses on training set: {plotter.losses[-1]}')
print(f'Losses on validation set: {plotter.val_losses[-1]}')
print(f'Losses on test set: {test_loss}')

# Save losses to file
with open(os.path.join(model_dir, 'losses.csv'), 'w') as f:
    f.write('dataset,' + ','.join(plotter.loss_labels) + '\n')
    f.write('train,' + ','.join([str(v) for v in plotter.losses[-1]]) + '\n')
    f.write('validation,' + ','.join([str(v) for v in plotter.val_losses[-1]]) + '\n')
    f.write('test,' + ','.join([str(v) for v in test_loss]))

# Make example predictions on test set
counter = 0
for i in range(pred_batches):

    X, true = test_loader[i]
    preds = model.predict_on_batch(X)
    losses = cu.calculate_losses(model, true, preds)

    vis.make_prediction_plots(preds, true, losses, descriptors, pred_dir, start_ind=counter)
    vis.make_input_plots(X, pred_dir, start_ind=counter, constant_range=False)

    np.save(os.path.join(pred_dir, f'X_{i}.npy'), X)
    np.save(os.path.join(pred_dir, f'preds_{i}.npy'), preds)
    np.save(os.path.join(pred_dir, f'true_{i}.npy'), true)

    counter += len(X[0])
