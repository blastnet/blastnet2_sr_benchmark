from torch import nn
import torch 
import math
import torch.nn.functional as F
import functools
from . import tools


class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv3d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv3d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv3d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv3d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv3d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        #initialization
        tools.initialize_weights(self.conv1,activation='leaky_relu',a=0.2,scale=0.1)
        tools.initialize_weights(self.conv2,activation='leaky_relu',a=0.2,scale=0.1)
        tools.initialize_weights(self.conv3,activation='leaky_relu',a=0.2,scale=0.1)
        tools.initialize_weights(self.conv4,activation='leaky_relu',a=0.2,scale=0.1)
        tools.initialize_weights(self.conv5,activation='linear',a=0,scale=0.1)
        # mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x
        
class InterpolateLayer(nn.Module):
    def __init__(self, scale_factor=2, mode='nearest'):
        super(InterpolateLayer, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x

class RRDBNet(nn.Module):
    #modified from https://github.com/xinntao/ESRGAN/blob/master/RRDBNet_arch.py
    def __init__(self, in_nc, out_nc, nf, nb, gc=32, upscale=4):
        super(RRDBNet, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)

        self.conv_first = nn.Conv3d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = self.make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv3d(nf, nf, 3, 1, 1, bias=True)
        #### upsampling
        self.HRconv = nn.Conv3d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv3d(nf, out_nc, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.upscale = upscale
        upscale_list = []
        for _ in range(int(math.log(upscale,2))):
            upscale_list.append(InterpolateLayer(scale_factor=2, mode='nearest'))
            upscale_list.append(nn.Conv3d(nf, nf, 3, 1, 1, bias=True))
            upscale_list.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.upscale_x = nn.Sequential(*upscale_list)

        #initialization taken from supplementary material in ESRGAN paper
        tools.initialize_weights(self.conv_first,activation='linear',a=0,scale=0.1)
        tools.initialize_weights(self.trunk_conv,activation='linear',a=0,scale=0.1)
        tools.initialize_weights(self.HRconv,activation='leaky_relu',a=0.2,scale=0.1)
        tools.initialize_weights(self.conv_last,activation='linear',a=0,scale=0.1)
        for layer in self.upscale_x:
            if isinstance(layer,nn.Conv3d):
                tools.initialize_weights(layer,activation='leaky_relu',a=0.2,scale=0.1)

    def make_layer(self, block, n_layers):
        layers = []
        for _ in range(n_layers):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = torch.add(fea, trunk)
        fea = self.upscale_x(fea)
        out = self.conv_last(self.lrelu(self.HRconv(fea)))

        return out
    
def init_rrdb(approx_param,upscale):
    #1b,firstc 16  = 0.888M params, 1 group, 1 block,34 feats
    #1b,firstc 32  = 1.4M params, 1 group, 1 block,44 feats
    #1b = 2.7M params, 1 group, 1 block, 60 feats
    #5b = 11M params 2 group, 20 block,64 feats
    #8b = 16M params 3 group, 20 block
    #16b = 35.1M params 7 group, 20 block
    #23b = 48M params 10 group, 20 block
    first_channel = 64
    if approx_param == '0.5M':
        first_channel = 4
        model_blocks = 1
    elif approx_param == '0.8M':
        first_channel = 16
        model_blocks = 1
    elif approx_param == '1.4M':
        first_channel = 32
        model_blocks = 1
    elif approx_param == '2.7M':
        model_blocks = 1
    elif approx_param == '5M':
        model_blocks = 2
    elif approx_param == '11M':
        model_blocks = 5
    elif approx_param == '17M':
        model_blocks = 8
    elif approx_param == '50M':
        model_blocks = 23
    else:
        raise ValueError('Please provide the correct model_blocks and first_channel')
    return RRDBNet(4, 4, first_channel, model_blocks, gc=32,upscale=upscale)
