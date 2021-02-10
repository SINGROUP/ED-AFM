
import os

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Input
from tensorflow.keras.layers import Reshape, Concatenate, AveragePooling3D, MaxPooling3D, Dropout
from tensorflow.keras.layers import Activation, LeakyReLU

from .layers import NNUpsampling, conv2D_with_bc, conv3D_with_bc

WEIGHTS_DIR = os.path.join(os.path.realpath(os.path.dirname(os.path.abspath(__file__))), 'pretrained_weights')

def ESUNet(n_in=2, n_out=2, 
    input_shape=(128,128,10),
    lrelu_factor=0.1,
    boundary_condition='reflective',
    last_relu=[False, True],
    labels=None
    ):
    '''
    Create ED-AFM Unet Keras model.
    Arguments:
        n_in: int. Number of inputs
        n_out: int. Number of outputs.
        input_shape: tuple of int. Shape of input without channels.
        lrelu_factor: float. Negative slope of LeakyReLU.
        boundary_condition: 'reflective' or 'periodic'. Type of boundary condition.
        last_relu: bool or list of bool of length n_out. Whether to use relu after last layer
            for each output.
        labels: list of str of length n_out. Labels on output layers.
    Returns: tensorflow.keras.models.Model.
    '''
    
    def activation():
        return LeakyReLU(alpha=lrelu_factor)
    
    if labels is None:
        labels = [f'out_{i}' for i in range(n_out)]
    else:
        assert len(labels) == n_out

    if not isinstance(last_relu, list):
        last_relu = [last_relu] * n_out

    # ==== Input branches
    inputs = []
    inp_branches = []
    for i in range(n_in):

        inp = Input(shape=input_shape)
        x = Reshape(input_shape+(1,))(inp)
        
        x = conv3D_with_bc(input_shape=x.shape[1:], filters=8, kernel_size=(3,3,3),
            boundary_condition=boundary_condition)(x)
        x = activation()(x)
        x = conv3D_with_bc(input_shape=x.shape[1:], filters=8, kernel_size=(3,3,3),
            boundary_condition=boundary_condition)(x)
        x = activation()(x)
        x = conv3D_with_bc(input_shape=x.shape[1:], filters=8, kernel_size=(3,3,3),
            boundary_condition=boundary_condition)(x)
        x = activation()(x)
        
        inputs.append(inp)
        inp_branches.append(x)

    if n_in > 1:
        x = Concatenate()(inp_branches)
    else:
        x = inp_branches[0]
    
    # ==== Encoder
    conv1  = conv3D_with_bc(input_shape=x.shape[1:], filters=8, kernel_size=(3,3,3),
        boundary_condition=boundary_condition)(x)
    conv1 = activation()(conv1)
    conv1 = conv3D_with_bc(input_shape=conv1.shape[1:], filters=8, kernel_size=(3,3,3),
        boundary_condition=boundary_condition)(conv1)
    conv1 = activation()(conv1)
    shape1 = conv1.shape.as_list()
    pool1 = MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2))(conv1)

    conv2 = conv3D_with_bc(input_shape=pool1.shape[1:], filters=16, kernel_size=(3,3,3),
        boundary_condition=boundary_condition)(pool1)
    conv2 = activation()(conv2)
    conv2 = conv3D_with_bc(input_shape=conv2.shape[1:], filters=16, kernel_size=(3,3,3),
        boundary_condition=boundary_condition)(conv2)
    conv2 = activation()(conv2)
    shape2 = conv2.shape.as_list()
    pool2 = MaxPooling3D(pool_size=(2,2,2), strides=(2,2,1))(conv2)

    conv3 = conv3D_with_bc(input_shape=pool2.shape[1:], filters=32, kernel_size=(3,3,3),
        boundary_condition=boundary_condition)(pool2)
    conv3 = activation()(conv3)
    conv3 = conv3D_with_bc(input_shape=conv3.shape[1:], filters=32, kernel_size=(3,3,3),
        boundary_condition=boundary_condition)(conv3)
    conv3 = activation()(conv3)
    shape3 = conv3.shape.as_list()
    pool3 = MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2))(conv3)

    shape4 = pool3.shape.as_list()
    reshape4 = Reshape(tuple(shape4[1:3])+(shape4[3]*shape4[4],))(pool3)
    conv4 = conv2D_with_bc(input_shape=reshape4.shape[1:], filters=64, kernel_size=(3,3),
        boundary_condition=boundary_condition)(reshape4)
    conv4 = activation()(conv4)
    conv4 = conv2D_with_bc(input_shape=conv4.shape[1:], filters=64, kernel_size=(3,3),
        boundary_condition=boundary_condition)(conv4)
    conv4 = activation()(conv4)
    drop4 = Dropout(0.1)(conv4)

    # ==== Decoder
    up5 = NNUpsampling(scale=2)(drop4)
    conv5 = conv2D_with_bc(up5.shape[1:], filters=16, kernel_size=3,
        boundary_condition=boundary_condition)(up5)
    conv5 = activation()(conv5)
    merge5 = Concatenate()([Reshape(tuple(shape3[1:3])+(shape3[3]*shape3[4],))(conv3), conv5])
    conv5 = conv2D_with_bc(merge5.shape[1:], filters=16, kernel_size=3,
        boundary_condition=boundary_condition)(merge5)
    conv5 = activation()(conv5)
    conv5 = conv2D_with_bc(conv5.shape[1:], filters=16, kernel_size=3,
        boundary_condition=boundary_condition)(conv5)
    conv5 = activation()(conv5)

    up6 = NNUpsampling(scale=2)(conv5)
    conv6 = conv2D_with_bc(up6.shape[1:], filters=16, kernel_size=3,
        boundary_condition=boundary_condition)(up6)
    conv6 = activation()(conv6)
    merge6 = Concatenate()([Reshape(tuple(shape2[1:3])+(shape2[3]*shape2[4],))(conv2), conv6])
    conv6 = conv2D_with_bc(merge6.shape[1:], filters=16, kernel_size=3,
        boundary_condition=boundary_condition)(merge6)
    conv6 = activation()(conv6)
    conv6 = conv2D_with_bc(conv6.shape[1:], filters=16, kernel_size=3,
        boundary_condition=boundary_condition)(conv6)
    conv6 = activation()(conv6)

    up7 = NNUpsampling(scale=2)(conv6)
    conv7 = conv2D_with_bc(up7.shape[1:], filters=16, kernel_size=3,
        boundary_condition=boundary_condition)(up7)
    conv7 = activation()(conv7)
    merge7 = Concatenate()([Reshape(tuple(shape1[1:3])+(shape1[3]*shape1[4],))(conv1), conv7])
    conv7 = conv2D_with_bc(merge7.shape[1:], filters=16, kernel_size=3,
        boundary_condition=boundary_condition)(merge7)
    conv7 = activation()(conv7)
    conv7 = conv2D_with_bc(conv7.shape[1:], filters=16, kernel_size=3,
        boundary_condition=boundary_condition)(conv7)
    conv7 = activation()(conv7)

    # ==== Split branches
    outputs = []
    for i in range(n_out):

        conv8 = conv2D_with_bc(conv7.shape[1:], filters=16, kernel_size=3,
            boundary_condition=boundary_condition)(conv7)
        conv8 = activation()(conv8)
        conv8 = conv2D_with_bc(conv8.shape[1:], filters=16, kernel_size=3,
            boundary_condition=boundary_condition)(conv8)
        conv8 = activation()(conv8)
        conv8 = conv2D_with_bc(conv8.shape[1:], filters=16, kernel_size=3,
            boundary_condition=boundary_condition)(conv8)
        conv8 = activation()(conv8)
        conv8 = conv2D_with_bc(conv8.shape[1:], filters=1, kernel_size=3,
            boundary_condition=boundary_condition)(conv8)
        if last_relu[i]:
            conv8 = Activation('relu')(conv8)
        out = Reshape(input_shape[:2], name=labels[i])(conv8)
        outputs.append(out)

    model = Model(inputs=inputs, outputs=outputs)

    return model

def load_pretrained_weights(model, tip_type='CO-Xe'):
    '''
    Load pretrained weights for ED-AFM model.
    Arguments:
        model: tensorflow.keras.models.Model. Model to load weights to.
        tip_type: 'CO-Xe', 'Cl-CO', 'Xe-Cl', 'CO', or 'CO-Xe-nograd'. Which tip combination of
            trained weights to load.
    '''
    if tip_type not in ['CO-Xe', 'Cl-CO', 'Xe-Cl', 'CO', 'CO-Xe-nograd']:
        raise ValueError(f'Unknown tip type "{tip_type}".')
    weights_path = os.path.join(WEIGHTS_DIR, f'model_{tip_type}.h5')
    model.load_weights(weights_path)