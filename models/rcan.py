
#modified from https://github.com/yulunzhang/RCAN/tree/master/RCAN_TrainCode/code/model
import torch.nn as nn
import math
from . import tools

class PixelShuffle3d(nn.Module):
    '''
    This class is a 3d version of pixelshuffle.
    '''
    def __init__(self, scale):
        '''
        :param scale: upsample scale
        '''
        super().__init__()
        self.scale = scale

    def forward(self, input):
        batch_size, channels, in_depth, in_height, in_width = input.size()
        nOut = channels // self.scale ** 3

        out_depth = in_depth * self.scale
        out_height = in_height * self.scale
        out_width = in_width * self.scale

        input_view = input.contiguous().view(batch_size, nOut, self.scale, self.scale, self.scale, in_depth, in_height, in_width)

        output = input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()

        return output.view(batch_size, nOut, out_depth, out_height, out_width)

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv3d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feat, 8 * n_feat, 3, bias))
                tools.initialize_weights(m[-1], 'relu', None, 0.1)
                m.append(PixelShuffle3d(2))
                if bn: m.append(nn.BatchNorm3d(n_feat))
                if act: m.append(act())
        elif scale == 3:
            m.append(conv(n_feat, 27 * n_feat, 3, bias))
            tools.initialize_weights(m[-1], 'relu', None, 0.1)
            m.append(PixelShuffle3d(3))
            if bn: m.append(nn.BatchNorm3d(n_feat))
            if act: m.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        # feature channel downscale and upscale --> channel weight
        conv1 = nn.Conv3d(channel, channel // reduction, 1, padding=0, bias=True)
        conv2 = nn.Conv3d(channel // reduction, channel, 1, padding=0, bias=True)
        tools.initialize_weights(conv1, 'relu', None, 0.1)
        tools.initialize_weights(conv2, 'linear', None, 0.1)
        self.conv_du = nn.Sequential(
                conv1,
                nn.ReLU(inplace=True),
                conv2,
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if i ==0: 
                tools.initialize_weights(modules_body[-1], 'relu', None, 0.1)
            else:
                tools.initialize_weights(modules_body[-1], 'linear', None, 0.1)
            if bn: modules_body.append(nn.BatchNorm3d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x
        return res

## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
            for _ in range(n_resblocks)]
        conv1 = conv(n_feat, n_feat, kernel_size)
        tools.initialize_weights(conv1, 'linear', None, 0.1)
        modules_body.append(conv1)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

## Residual Channel Attention Network (RCAN)
class RCAN(nn.Module):
    def __init__(self, n_resgroups, n_resblocks, n_feats, reduction, scale, n_colors, res_scale, conv=default_conv):
        super(RCAN, self).__init__()
        
        kernel_size = 3
        act = nn.ReLU(True)
        self.upscale=scale
        # define head module
        conv1 = conv(n_colors, n_feats, kernel_size)
        tools.initialize_weights(conv1, 'linear', None, 0.1)
        modules_head = [conv1]

        # define body module
        #already initialized in 
        modules_body = [
            ResidualGroup(
                conv, n_feats, kernel_size, reduction, act=act, res_scale=res_scale, n_resblocks=n_resblocks) \
            for _ in range(n_resgroups)]
        conv2 = conv(n_feats, n_feats, kernel_size)
        tools.initialize_weights(conv2, 'linear', None, 0.1)
        modules_body.append(conv2)

        # define tail module
        conv3 = conv(n_feats, n_colors, kernel_size)
        tools.initialize_weights(conv3, 'linear', None, 0.1)
        modules_tail = [
            Upsampler(conv, scale, n_feats, act=False),
            conv3]

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)

        return x 

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))
            

def init_rcan(approx_param,upscale):
    #1b,firstc 16  = 0.888M params, 1 group, 1 block,34 feats
    #1b,firstc 32  = 1.4M params, 1 group, 1 block,44 feats
    #1b = 2.7M params, 1 group, 1 block, 60 feats
    #5b = 11M params 2 group, 20 block,64 feats
    #8b = 16M params 3 group, 20 block
    #16b = 35.1M params 7 group, 20 block
    #23b = 48M params 10 group, 20 block
    if approx_param == '0.5M':
        n_feats = 26
        n_resblocks = 1
        n_resgroups = 1
    elif approx_param == '0.8M':
        n_feats = 34
        n_resblocks = 1
        n_resgroups = 1
    elif approx_param == '1.4M':
        n_feats = 44
        n_resblocks = 1
        n_resgroups = 1
    elif approx_param == '2.7M':
        n_feats = 60
        n_resblocks = 1
        n_resgroups = 1
    elif approx_param == '5M':
        n_feats = 64
        n_resblocks = 10
        n_resgroups = 1
    elif approx_param == '11M':
        n_feats = 64
        n_resblocks = 20
        n_resgroups = 2
    elif approx_param == '17M':
        n_feats = 64
        n_resblocks = 20
        n_resgroups = 3
    elif approx_param == '50M':
        n_feats = 64
        n_resblocks = 20
        n_resgroups = 10
    else:
        raise ValueError('Please provide the correct model_blocks and first_channel')
    return RCAN(n_feats=n_feats,n_colors=4, n_resgroups=n_resgroups,n_resblocks=n_resblocks,reduction=16,scale=upscale,res_scale=1)
