#written by W.T. Chung
#modfied from Conv-FNO and U-FNO: https://github.com/gegewen/ufno
import torch
from torch import nn
from torch.nn import functional as F
from .rcan import Upsampler,default_conv
from . import tools
import numpy as np
from .edsr import ResBlock


class UpsamplerDecoder(nn.Module):
    def __init__(self,n_feats,out_channel,kernel_size,scale, conv=default_conv):
        super(UpsamplerDecoder, self).__init__()

        
        # define tail module
        m_tail = [
            Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, out_channel, kernel_size)
        ]
        tools.initialize_weights(m_tail[-1], 'linear', None, 0.1)

        self.tail = nn.Sequential(*m_tail)
        self.upscale = scale

    def forward(self, x):
        x = self.tail(x)
        return x 

class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        weights1 = self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat)
        self.weights1 = nn.Parameter(torch.view_as_real(weights1))
        weights2 = self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat)
        self.weights2 = nn.Parameter(torch.view_as_real(weights2))
        weights3 = self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat)
        self.weights3 = nn.Parameter(torch.view_as_real(weights3))
        weights4 = self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat)
        self.weights4 = nn.Parameter(torch.view_as_real(weights4))
        self.einsum_op = "bixyz,ioxyz->boxyz"

    # Complex multiplication
    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        return torch.einsum(self.einsum_op, input, weights)

    #todo take mixed precision treatment from updated repo
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-3,-2,-1])

        # Multiply relevant Fourier modes
        #removed //2 + 1 since all input dims are spatial
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1), dtype=torch.cfloat, device=x.device)
        # print("out ft",out_ft.shape)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], torch.view_as_complex(self.weights1))
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], torch.view_as_complex(self.weights2))
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], torch.view_as_complex(self.weights3))
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], torch.view_as_complex(self.weights4))

        #Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x

class FNOResLayer(nn.Module):
    def __init__(self, modes1, modes2, modes3, width):
        super(FNOResLayer, self).__init__()
        self.width = width
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3

        self.conv0 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.w0 = nn.Conv3d(self.width, self.width, 1)
        self.u0 = ResBlock(default_conv,self.width,3,res_scale=0.1)
      
        tools.initialize_weights(self.w0, 'relu', None, 0.1)

    def forward(self, input):
        x1 = self.conv0(input)
        x2 = self.w0(input)
        x3 = self.u0(input)
        return F.relu(x1 + x2 + x3)
       

class FNO3dEncoder(nn.Module):
    def __init__(self, modes1, modes2, modes3, width,in_channel,out_channel =128,n_resblocks=32):
        super(FNO3dEncoder, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t). It's a constant function in time, except for the last index.
        input shape: (batchsize, x=64, y=64, t=40, c=13)
        output: the solution of the next 40 timesteps
        output shape: (batchsize, x=64, y=64, t=40, c=1)
        """

        self.padding = 6 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(in_channel+3, width)
        tools.initialize_weights(self.fc0, 'linear', None, 0.1)

        m_body = [
            FNOResLayer(
                modes1, modes2, modes3, width
            ) for _ in range(n_resblocks)
        ]
        self.body = nn.Sequential(*m_body)

        self.fc1 = nn.Linear(width, out_channel)
        tools.initialize_weights(self.fc1, 'relu', None, 0.1)

    def forward(self, x):
        x = x.permute(0, 2, 3, 4, 1)
        grid = self.get_grid(x)  
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = F.pad(x, [0,self.padding,0,self.padding,0,self.padding]) # pad the domain if input is non-periodic for all spatial domains

        x = self.body(x)

        x = x[..., :-self.padding, :-self.padding, :-self.padding]
        x = x.permute(0, 2, 3, 4, 1) # pad the domain if input is non-periodic
        x = self.fc1(x)
        x = F.relu(x)
        x = x.permute(0, 4, 1, 2, 3)
        return x

    def get_grid(self, x):
        batchsize, size_x, size_y, size_z = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float,device=x.device)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float,device=x.device)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float,device=x.device)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1)

class ConvFNO(nn.Module):
    def __init__(self, modes, bottleneck,kernel_size, fno_width=20,in_channel=4,out_channel=4,upscale =8,n_resblocks=32):
        super(ConvFNO, self).__init__()
        self.upscale = upscale
        self.FNOEncoder = FNO3dEncoder(modes,modes,modes, fno_width,in_channel,bottleneck,n_resblocks=n_resblocks)
        self.UpsamplerDecoder = UpsamplerDecoder(bottleneck,out_channel,kernel_size,upscale)
    def forward(self, x):
        x = self.FNOEncoder(x)
        x = self.UpsamplerDecoder(x)
        return x



def init_convfno(approx_param,upscale):
    modes = 2
    if approx_param == '0.5M':
        n_feats = 12
    elif approx_param == '1.4M':
        n_feats = 20
    elif approx_param == '2.7M':
        n_feats = 24
    elif approx_param == '5M':
        n_feats = 34
    elif approx_param == '11M':
        n_feats = 46
    elif approx_param == '17M':
        n_feats = 68
    elif approx_param == '50M': 
        #this is actually ~35M since 50M runs out of memory
        n_feats =86
    return  ConvFNO(modes,n_feats,3,fno_width=n_feats,in_channel=4,out_channel=4,upscale=upscale)

