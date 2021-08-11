
import os

import torch
import torch.nn as nn
from  torch.nn.modules.upsampling import Upsample

from .layers import Conv3dBlock, Conv2dBlock, _get_padding, UNetAttentionConv
from .utils import download_weights

class AttentionUNet(nn.Module):
    '''
    Pytorch 3D-to-2D U-net model with attention.

    3D conv -> concatenate -> 3D conv/pool/dropout -> 2D conv/dropout -> 2D upsampling/conv with skip connections
    and attention. For multiple inputs, the inputs are first processed through separate 3D conv blocks before merging
    by concatenating along channel axis.

    Arguments:
        conv3d_in_channels: int. Number of channels in input.
        conv2d_in_channels: int. Number of channels in first 2D conv layer after flattening 3D to 2D.
        conv3d_out_channels: list of int of same length as conv3d_block_channels. Number of channels after
            3D-to-2D flattening after each 3D conv block. Depends on input z size.
        n_in: int. Number of input 3D images.
        n_out: int. Number of output 2D maps.
        merge_block_channels: list of int. Number of channels in input merging 3D conv blocks.
        merge_block_depth: int. Number of layers in each merge conv block.
        conv3d_block_channels: list of ints. Number channels in 3D conv blocks.
        conv3d_block_depth: int. Number of layers in each 3D conv block.
        conv3d_dropouts: list of int of same lenght as conv3d_block_channels. Dropout rates after each conv3d block.
        conv2d_block_channels: list of ints. Number channels in 2D conv blocks.
        conv2d_block_depth: int. Number of layers in each 2D conv block.
        conv2d_dropouts: list of int of same lenght as conv2d_block_channels. Dropout rates after each conv2d block.
        attention_channels: list of int of same lenght as conv3d_block_channels. Number of channels in conv layer
            within each attention block.
        upscale2d_block_channels: list of int of same length as conv3d_block_channels. Number of channels in
            each 2D conv block after upscale before skip connection.
        upscale2d_block_depth: int. Number of layers in each 2D conv block after upscale before skip connection.
        upscale2d_block_channels2: list of int of same length as conv3d_block_channels. Number of channels in
            each 2D conv block after skip connection.
        upscale2d_block_depth2: int. Number of layers in each 2D conv block after skip connection.
        split_conv_block_channels: list of int. Number of channels in 2d conv blocks after splitting outputs.
        split_conv_block_depth: int. Number of layers in each 2d conv block after splitting outputs.
        res_connections: Boolean. Whether to use residual connections in conv blocks.
        out_convs_channels: int or list of int. Number of channels in splitted outputs.
        out_relus: Bool or list of Bool of length n_out. Whether to apply relu activation to the output 2D maps.
        pool_type: str ('max' or 'avg'). Type of pooling to use.
        pool_z_strides: list of int of same length as conv3d_block_channels. Stride of pool layers in z direction.
        padding_mode: str. Type of padding in each convolution layer. 'zeros', 'reflect', 'replicate' or 'circular'.
        activation: str ('relu', 'lrelu', or 'elu') or nn.Module. Activation to use after every layer except last one.
        attention_activation: str. Type of activation to use for attention map. 'sigmoid' or 'softmax'.
        device: str. Device to load model onto.
    '''

    def __init__(self,
        conv3d_in_channels,
        conv2d_in_channels,
        conv3d_out_channels,
        n_in=1,
        n_out=3,
        merge_block_channels=[8],
        merge_block_depth=2,
        conv3d_block_channels=[8, 16, 32],
        conv3d_block_depth=2,
        conv3d_dropouts=[0.0, 0.0, 0.0],
        conv2d_block_channels=[128],
        conv2d_block_depth=3,
        conv2d_dropouts=[0.1],
        attention_channels=[32, 32, 32],
        upscale2d_block_channels=[16, 16, 16],
        upscale2d_block_depth=1,
        upscale2d_block_channels2=[16, 16, 16],
        upscale2d_block_depth2=2,
        split_conv_block_channels=[16],
        split_conv_block_depth=[3],
        res_connections=True,
        out_convs_channels = 1,
        out_relus=True,
        pool_type='avg',
        pool_z_strides=[2, 1, 2],
        padding_mode='zeros',
        activation='lrelu',
        attention_activation='softmax',
        device='cuda'
    ):
        super().__init__()

        assert (
            len(conv3d_block_channels)
            == len(conv3d_out_channels)
            == len(conv3d_dropouts)
            == len(upscale2d_block_channels)
            == len(upscale2d_block_channels2)
            == len(attention_channels)
        )
        
        if isinstance(activation, nn.Module):
            self.act = activation
        elif activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'lrelu':
            self.act = nn.LeakyReLU()
        elif activation == 'elu':
            self.act = nn.ELU()
        else:
            raise ValueError(f'Unknown activation function {activation}')

        if not isinstance(out_relus, list):
            out_relus = [out_relus] * n_out
        else:
            assert len(out_relus) == n_out

        if not isinstance(out_convs_channels, list):
            out_convs_channels = [out_convs_channels] * n_out
        else:
            assert len(out_convs_channels) == n_out

        self.out_relus = out_relus
        self.relu_act = nn.ReLU()

        # -- Input merge conv blocks --
        self.merge_convs = nn.ModuleList([None]*n_in)
        for i in range(n_in):
            self.merge_convs[i] = nn.ModuleList([
                Conv3dBlock(conv3d_in_channels, merge_block_channels[0], 3,
                    merge_block_depth, padding_mode, res_connections, self.act, False)
            ])
            for j in range(len(merge_block_channels)-1):
                self.merge_convs[i].append(
                    Conv3dBlock(merge_block_channels[j], merge_block_channels[j+1], 3,
                        merge_block_depth, padding_mode, res_connections, self.act, False)
                )

        # -- Encoder conv blocks --
        self.conv3d_blocks = nn.ModuleList([
            Conv3dBlock(n_in*merge_block_channels[-1], conv3d_block_channels[0], 3,
                conv3d_block_depth, padding_mode, res_connections, self.act, False)
        ])
        self.conv3d_dropouts = nn.ModuleList([nn.Dropout(conv3d_dropouts[0])])
        for i in range(len(conv3d_block_channels)-1):
            self.conv3d_blocks.append(
                Conv3dBlock(conv3d_block_channels[i], conv3d_block_channels[i+1], 3,
                    conv3d_block_depth, padding_mode, res_connections, self.act, False)
            )
            self.conv3d_dropouts.append(nn.Dropout(conv3d_dropouts[i+1]))
            
        # -- Middle conv blocks --
        self.conv2d_blocks = nn.ModuleList([
            Conv2dBlock(conv2d_in_channels, conv2d_block_channels[0], 3,
                conv2d_block_depth, padding_mode, res_connections, self.act, False)
        ])
        self.conv2d_dropouts = nn.ModuleList([nn.Dropout(conv2d_dropouts[0])])
        for i in range(len(conv2d_block_channels)-1):
            self.conv2d_blocks.append(
                Conv2dBlock(conv2d_block_channels[i], conv2d_block_channels[i+1], 3,
                    conv2d_block_depth, padding_mode, res_connections, self.act, False)
            )
            self.conv2d_dropouts.append(nn.Dropout(conv2d_dropouts[i+1]))
        
        # -- Decoder conv blocks --
        self.attentions = nn.ModuleList([])
        for c_att, c_conv in zip(attention_channels, reversed(conv3d_out_channels)):
            self.attentions.append(
                UNetAttentionConv(c_conv, conv2d_block_channels[-1], c_att, 3,
                    padding_mode, self.act, attention_activation, upsample_mode='bilinear')
            )

        self.upscale2d_blocks = nn.ModuleList([
            Conv2dBlock(conv2d_block_channels[-1], upscale2d_block_channels[0], 3,
                upscale2d_block_depth, padding_mode, res_connections, self.act, False)
        ])
        for i in range(len(upscale2d_block_channels)-1):
            self.upscale2d_blocks.append(
                Conv2dBlock(upscale2d_block_channels2[i], upscale2d_block_channels[i+1], 3,
                    upscale2d_block_depth, padding_mode, res_connections, self.act, False)
            )
    
        self.upscale2d_blocks2 = nn.ModuleList([])
        for i in range(len(upscale2d_block_channels2)):
            self.upscale2d_blocks2.append(
                Conv2dBlock(upscale2d_block_channels[i]+conv3d_out_channels[-(i+1)], upscale2d_block_channels2[i],
                    3, upscale2d_block_depth2, padding_mode, res_connections, self.act, False)
            )
        
        # -- Output split conv blocks --
        padding = _get_padding(3, 2)
        self.out_convs = nn.ModuleList([])
        self.split_convs = nn.ModuleList([None]*n_out)
        for i_out in range(n_out):
            
            self.split_convs[i_out] = nn.ModuleList([
                Conv2dBlock(upscale2d_block_channels2[-1], split_conv_block_channels[0], 3,
                    split_conv_block_depth, padding_mode, res_connections, self.act, False)
            ])
            for i in range(len(split_conv_block_channels)-1):
                self.split_convs.append(
                    Conv2dBlock(split_conv_block_channels[i], split_conv_block_channels[i+1], 3,
                        split_conv_block_depth, padding_mode, res_connections, self.act, False)
                )
            
            self.out_convs.append(nn.Conv2d(split_conv_block_channels[-1], out_convs_channels[i_out], kernel_size=3,
                padding=padding, padding_mode=padding_mode))

        if pool_type == 'avg':
            pool = nn.AvgPool3d
        elif pool_type == 'max':
            pool = nn.MaxPool3d
        self.pools = nn.ModuleList([pool(2, stride=(2, 2, pz)) for pz in pool_z_strides])

        self.upsample2d = Upsample(scale_factor=2, mode='nearest')
        self.device = device
        self.n_out = n_out
        self.n_in = n_in

        self.to(device)

    def _flatten(self, x):
        return x.permute(0,1,4,2,3).reshape(x.size(0), -1, x.size(2), x.size(3))

    def forward(self, x, return_attention=False):

        assert len(x) == self.n_in

        # Do 3D convolutions for each input
        in_branches = []
        for xi, convs in zip(x, self.merge_convs):
            for conv in convs:
                xi = self.act(conv(xi))
            in_branches.append(xi)

        # Merge input branches
        x = torch.cat(in_branches, dim=1)

        # Encode
        x_3ds = []
        for conv, dropout, pool in zip(self.conv3d_blocks, self.conv3d_dropouts, self.pools):
            x = self.act(conv(x))
            x = dropout(x)
            x_3ds.append(x)
            x = pool(x)

        # Middle 2d convs
        x = self._flatten(x)
        for conv, dropout in zip(self.conv2d_blocks, self.conv2d_dropouts):
            x = self.act(conv(x))
            x = dropout(x)

        # Compute attention maps
        attention_maps = []
        x_gated = []
        for attention, x_3d in zip(self.attentions, reversed(x_3ds)):
            g, a = attention(self._flatten(x_3d), x)
            x_gated.append(g)
            attention_maps.append(a)

        # Decode
        for i, (conv1, conv2, xg) in enumerate(zip(self.upscale2d_blocks, self.upscale2d_blocks2, x_gated)):
            x = self.upsample2d(x)
            x = self.act(conv1(x))
            x = torch.cat([x, xg], dim=1) # Attention-gated skip connection
            x = self.act(conv2(x))

        # Split into different outputs
        outputs = []
        for i, (split_convs, out_conv) in enumerate(zip(self.split_convs, self.out_convs)):

            h = x
            for conv in split_convs:
                h = self.act(conv(h))
            h = out_conv(h)
            
            if self.out_relus[i]:
                h = self.relu_act(h)
            outputs.append(h.squeeze(1))

        if return_attention:
            outputs = (outputs, attention_maps)
        
        return outputs

class EDAFMNet(AttentionUNet):
    '''
    ED-AFM Attention U-net.

    This is the model used in the ED-AFM paper for task of predicting electrostatics from AFM images.
    It is a subclass of the AttentionUnet class with specific hyperparameters.

    Arguments:
        device: str. Device to load model onto.
        trained_weights: str or None. If not None, load pretrained weights to the model. One of 'base',
            'single-channel', 'CO-Cl', 'Xe-Cl', 'constant-noise', 'uniform-noise', 'no-gradient', or
            'matched-tips'. See README at https://github.com/SINGROUP/ED-AFM for explanations for the
            different options.
        weights_dir: str. If weights_type is not None, directory where the weights will be downloaded into.
    '''

    def __init__(self, device='cuda', trained_weights=None, weights_dir='./weights'):

        if trained_weights == 'single-channel':
            n_in = 1
        else:
            n_in = 2

        super().__init__(
            conv3d_in_channels          = 1,
            conv2d_in_channels          = 192,
            merge_block_channels        = [32],
            merge_block_depth           = 2,
            conv3d_out_channels         = [288, 288, 384],
            conv3d_dropouts             = [0.0, 0.0, 0.0],
            conv3d_block_channels       = [48, 96, 192],
            conv3d_block_depth          = 3,
            conv2d_block_channels       = [512],
            conv2d_block_depth          = 3,
            conv2d_dropouts             = [0.0],
            n_in                        = n_in,
            n_out                       = 1,
            upscale2d_block_channels    = [256, 128, 64],
            upscale2d_block_depth       = 2,
            upscale2d_block_channels2   = [256, 128, 64],
            upscale2d_block_depth2      = 2,
            split_conv_block_channels   = [64],
            split_conv_block_depth      = 3,
            out_relus                   = [False],
            pool_z_strides              = [2, 1, 2], 
            activation                  = nn.LeakyReLU(negative_slope = 0.1, inplace = True),
            padding_mode                = 'replicate',
            device                      = device
        )
        
        if trained_weights:
            weights_path = download_weights(trained_weights, weights_dir)
            self.load_state_dict(torch.load(weights_path))
