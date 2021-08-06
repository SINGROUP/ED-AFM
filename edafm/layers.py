
import torch
import torch.nn as nn
import torch.nn.functional as F

def _get_padding(kernel_size, nd):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, )*nd
    padding = []
    for i in range(nd):
        padding += [(kernel_size[i]-1) // 2]
    return tuple(padding)

class _ConvNdBlock(nn.Module):

    def __init__(self,
        in_channels,
        out_channels,
        nd,
        kernel_size=3,
        depth=2,
        padding_mode='zeros',
        res_connection=True,
        activation=None,
        last_activation=True
    ):

        assert depth >= 1

        if nd == 2:
            conv = nn.Conv2d
        elif nd == 3:
            conv = nn.Conv3d
        else:
            raise ValueError(f'Invalid convolution dimensionality {nd}.')
        
        super().__init__()
        
        self.res_connection = res_connection
        if not activation:
            self.act = nn.ReLU()
        else:
            self.act = activation

        if last_activation:
            self.acts = [self.act] * depth
        else:
            self.acts = [self.act] * (depth-1) + [self._identity]
        
        padding = _get_padding(kernel_size, nd)
        self.convs = nn.ModuleList([conv(in_channels, out_channels, kernel_size=kernel_size, padding=padding, padding_mode=padding_mode)])
        for i in range(depth-1):
            self.convs.append(conv(out_channels, out_channels, kernel_size=kernel_size, padding=padding, padding_mode=padding_mode))
        if res_connection and in_channels != out_channels:
            self.res_conv = conv(in_channels, out_channels, kernel_size=1)
        else:
            self.res_conv = None

    def _identity(self, x):
        return x
            
    def forward(self, x_in):
        x = x_in
        for conv, act in zip(self.convs, self.acts):
            x = act(conv(x))
        if self.res_connection:
            if self.res_conv:
                x = x + self.res_conv(x_in)
            else:
                x = x + x_in
        return x

class Conv3dBlock(_ConvNdBlock):
    '''
    Pytorch 3D convolution block module.

    Arguments:
        in_channels: int. Number of channels entering the first convolution layer.
        out_channels: int. Number of output channels in each layer of the block.
        kernel_size: int or tuple. Size of convolution kernel.
        depth: int >= 1. Number of convolution layers in the block.
        padding_mode: str. Type of padding in each convolution layer. 'zeros', 'reflect', 'replicate' or 'circular'.
        res_connection: Boolean. Whether to use residual connection over the block (f(x) = h(x) + x).
            If in_channels != out_channels, a 1x1x1 convolution is applied to the res connection
            to make the channel numbers match.
        activation: torch.nn.Module. Activation function to use after every layer in block. If None,
            defaults to ReLU.
        last_activation: Bool. Whether to apply the activation after the last conv layer (before res connection).
    '''
    def __init__(self,
        in_channels,
        out_channels,
        kernel_size=3,
        depth=2,
        padding_mode='zeros',
        res_connection=True,
        activation=None,
        last_activation=True
    ):
        super().__init__(in_channels, out_channels, 3, kernel_size, depth, padding_mode, res_connection, activation, last_activation)
        
class Conv2dBlock(_ConvNdBlock):
    '''
    Pytorch 2D convolution block module.

    Arguments:
        in_channels: int. Number of channels entering the first convolution layer.
        out_channels: int. Number of output channels in each layer of the block.
        kernel_size: int or tuple. Size of convolution kernel.
        depth: int >= 1. Number of convolution layers in the block.
        padding_mode: str. Type of padding in each convolution layer. 'zeros', 'reflect', 'replicate' or 'circular'.
        res_connection: Boolean. Whether to use residual connection over the block (f(x) = h(x) + x).
            If in_channels != out_channels, a 1x1 convolution is applied to the res connection
            to make the channel numbers match.
        activation: torch.nn.Module. Activation function to use after every layer in block. If None,
            defaults to ReLU.
        last_activation: Bool. Whether to apply the activation after the last conv layer (before res connection).
    '''
    def __init__(self,
        in_channels,
        out_channels,
        kernel_size=3,
        depth=2,
        padding_mode='zeros',
        res_connection=True,
        activation=None,
        last_activation=True
    ):
        super().__init__(in_channels, out_channels, 2, kernel_size, depth, padding_mode, res_connection, activation, last_activation)

class UNetAttentionConv(nn.Module):
    '''
    Pytorch attention layer for Attention U-net model upsampling stage.

    Arguments:
        in_channels: int. Number of channels in the attended feature map.
        query_channels: int. Number of channels in query feature map.
        query_channels: int. Number of channels in hidden convolution layer before computing attention.
        kernel_size: int. Size of convolution kernel.
        padding_mode: str. Type of padding in each convolution layer. 'zeros', 'reflect', 'replicate' or 'circular'.
        conv_activation: nn.Module. Activation function to use after convolution layers
        attention_activation: str. Type of activation to use for attention map. 'sigmoid' or 'softmax'.
        upsample_mode: str. Algorithm for upsampling query feature map to the attended
            feature map size. For options see torch.nn.functional.interpolate.
    
    References:
        https://arxiv.org/abs/1804.03999
    '''
    def __init__(self,
        in_channels,
        query_channels,
        attention_channels,
        kernel_size,
        padding_mode='zeros',
        conv_activation=nn.ReLU(),
        attention_activation='softmax',
        upsample_mode='bilinear'
    ):
        super().__init__()

        if attention_activation == 'softmax':
            self.attention_activation = self._softmax
        elif attention_activation == 'sigmoid':
            self.attention_activation = self._sigmoid
        else:
            raise ValueError(f'Unrecognized attention map activation {attention_activation}.')

        padding = _get_padding(kernel_size, 2)
        self.x_conv = nn.Conv2d(in_channels, attention_channels, kernel_size=kernel_size,
            padding=padding, padding_mode=padding_mode)
        self.q_conv = nn.Conv2d(query_channels, attention_channels, kernel_size=kernel_size,
            padding=padding, padding_mode=padding_mode)
        self.a_conv = nn.Conv2d(attention_channels, 1, kernel_size=kernel_size,
            padding=padding, padding_mode=padding_mode)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.upsample_mode = upsample_mode
        self.conv_activation = conv_activation

    def _softmax(self, a):
        shape = a.shape
        return self.softmax(a.reshape(shape[0], -1)).reshape(shape)
    
    def _sigmoid(self, a):
        return self.sigmoid(a)

    def forward(self, x, q):

        # Upsample query q to the size of input x and convolve
        q = F.interpolate(q, size=x.size()[2:], mode=self.upsample_mode, align_corners=False)
        q = self.conv_activation(self.q_conv(q))

        # Convolve input x and sum with q
        a = self.conv_activation(self.x_conv(x))
        a = self.conv_activation(a + q)

        # Get attention map and mix it with x
        a = self.attention_activation(self.a_conv(a))
        x = a * x

        return x, a.squeeze(dim=1)
