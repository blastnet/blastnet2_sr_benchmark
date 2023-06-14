#written by W.T. Chung and B. Akoush
#modified from https://github.com/sanghyun-son/EDSR-PyTorch
from torch import nn
from .rcan import Upsampler,default_conv
from . import tools
class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if i ==0: 
                tools.initialize_weights(m[-1], 'relu', None, 0.1)
            else:
                tools.initialize_weights(m[-1], 'linear', None, 0.1)
            if bn:
                m.append(nn.BatchNorm3d(n_feats))
            if i == 0:
                m.append(act)
            

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class EDSR(nn.Module):
    def __init__(self, in_channels,n_feats,kernel_size,res_scale,n_resblocks,scale, conv=default_conv):
        super(EDSR, self).__init__()

        
        # define head module
        m_head = [conv(in_channels, n_feats, kernel_size)]
        tools.initialize_weights(m_head[-1], 'linear', None, 0.1)

        # define body module
        m_body = [
            ResBlock(
                conv, n_feats, kernel_size, 
                act=nn.ReLU(True), res_scale=res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, in_channels, kernel_size)
        ]
        tools.initialize_weights(m_tail[-1], 'linear', None, 0.1)

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)
        self.upscale = scale

    def forward(self, x):
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)

        return x 
    
def init_edsr(approx_param,upscale):
    if approx_param == '0.5M':
        n_feats = 14
    elif approx_param == '0.8M':
        n_feats = 20
    elif approx_param == '1.4M':
        n_feats = 24
    elif approx_param == '2.7M':
        n_feats = 34
    elif approx_param == '5M':
        n_feats = 46
    elif approx_param == '11M':
        n_feats = 68
    elif approx_param == '17M':
        n_feats = 86
    elif approx_param == '50M': 
        #this is actually 35M since 50M runs out of memory
        n_feats = 120
    return EDSR(in_channels=4,n_feats=n_feats,kernel_size=3,res_scale=0.1,n_resblocks=32,scale=upscale)
