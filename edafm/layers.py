
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Conv3D, Lambda, Input
import tensorflow.keras.backend as K

n_upsample = 0
n_conv2d_p = 0
n_conv2d_r = 0
n_conv3d_r = 0
n_conv3d_p = 0

def NNUpsampling(scale=2):
    '''
    Keras nearest-neighbour upsampling layer.

    Arguments:
        scale: int. Factor by which to upsample.
    Returns: 
        tensorflow.keras.layers.Lambda.
    '''
    global n_upsample
    n_upsample += 1
    def out_shape(input_shape, scale=scale):
        return (input_shape[0], input_shape[1]*scale, input_shape[2]*scale, input_shape[3])
    def NNresize(x, scale=scale):
        return K.resize_images(x, scale, scale, 'channels_last')
    return Lambda(NNresize, name='nn_upsampling_'+str(n_upsample), output_shape=out_shape)

def conv2D_with_bc(input_shape, filters, kernel_size=(3,3), boundary_condition='reflective'):
    '''
    Keras Conv2D layer with boundary conditions for padding.

    Arguments:
        input_shape: tuple of ints. Input shape of layer.
        filters: int. Number of filters/channels.
        kernel_size: int or (int, int). Convolution kernel size.
        boundary_condition: 'reflective' or 'periodic'. Type of boundary condition.

    Returns:
        tensorflow.keras.models.Model.
    '''
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    kx1 = int(np.floor((kernel_size[0]-1)/2.0))
    kx2 = int(np.ceil((kernel_size[0]-1)/2.0))
    ky1 = int(np.floor((kernel_size[1]-1)/2.0))
    ky2 = int(np.ceil((kernel_size[1]-1)/2.0))
    paddings = tf.constant([[0, 0], [kx1, kx2], [ky1, ky2], [0, 0]])

    def periodic_padding_2D(X):
        m = K.concatenate([X[:,-kx1:,:], X, X[:,:kx2,:]], axis=1)
        t = m[:,:,-ky1:]
        b = m[:,:,:ky2]
        return K.concatenate([t,m,b], axis=2)

    inp = Input(shape=input_shape)
    if boundary_condition == 'periodic':
        global n_conv2d_p
        n_conv2d_p += 1
        name = 'conv2d_periodic_%d' % n_conv2d_p
        padded = Lambda(periodic_padding_2D)(inp)
    elif boundary_condition == 'reflective':
        global n_conv2d_r
        n_conv2d_r += 1
        name = 'conv2d_reflective_%d' % n_conv2d_r
        padded = Lambda(lambda x: tf.pad(x, paddings, mode='SYMMETRIC'))(inp)
    else:
        print('Invalid boundary condition')
        return None
    conv = Conv2D(filters=filters, kernel_size=kernel_size, padding='valid')(padded)
    return Model(inputs=inp, outputs=conv, name=name)

def conv3D_with_bc(input_shape, filters, kernel_size=(3,3,3), boundary_condition='reflective'):
    '''
    Keras Conv3D layer with boundary conditions for padding.

    Arguments:
        input_shape: tuple of ints. Input shape of layer.
        filters: int. Number of filters/channels.
        kernel_size: int or (int, int, int). Convolution kernel size.
        boundary_condition: 'reflective' or 'periodic'. Type of boundary condition.

    Returns:
        tensorflow.keras.models.Model.
    '''
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size, kernel_size)
    kx1 = int(np.floor((kernel_size[0]-1)/2.0))
    kx2 = int(np.ceil((kernel_size[0]-1)/2.0))
    ky1 = int(np.floor((kernel_size[1]-1)/2.0))
    ky2 = int(np.ceil((kernel_size[1]-1)/2.0))
    kz1 = int(np.floor((kernel_size[2]-1)/2.0))
    kz2 = int(np.ceil((kernel_size[2]-1)/2.0))
    paddings = tf.constant([[0, 0], [kx1, kx2], [ky1, ky2], [kz1, kz2], [0, 0]])

    def periodic_padding_3D(X):
        m = K.concatenate([X[:,-kx1:], X, X[:,:kx2]], axis=1)
        m = K.concatenate([m[:,:,-ky1:], m, m[:,:,:ky2]], axis=2)
        m = K.concatenate([m[:,:,:,-kz1:], m, m[:,:,:,:kz2]], axis=3)
        return m

    inp = Input(shape=input_shape)
    if boundary_condition == 'periodic':
        global n_conv3d_p
        n_conv3d_p += 1
        name = 'conv3d_periodic_%d' % n_conv3d_p
        padded = Lambda(periodic_padding_3D)(inp)
    elif boundary_condition == 'reflective':
        global n_conv3d_r
        n_conv3d_r += 1
        name = 'conv3d_reflective_%d' % n_conv3d_r
        padded = Lambda(lambda x: tf.pad(x, paddings, mode='SYMMETRIC'))(inp)
    else:
        print('Invalid boundary condition')
        return None
    conv = Conv3D(filters=filters, kernel_size=kernel_size, padding='valid')(padded)
    return Model(inputs=inp, outputs=conv, name=name)
