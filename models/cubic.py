import torch
from torch import nn
import numpy as np
from .metrics import SSIM3D, GradLoss
import pytorch_lightning as pl
import torch.nn.functional as F


import sys
sys.path.append('../')
from common.functions import sgs,remove_edges,divergence,divergence_sgs_separate

def get_cubicfile(idx,train_dict,cubic_path,data_path,mode,upscale):
    hash_id = train_dict['hash_id'][idx]
    scalars = ['RHO_kgm-3_id','UX_ms-1_id','UY_ms-1_id','UZ_ms-1_id']
    #return a 4channel numpy array of the 4 scalars
    Yhat = []
    for scalar in scalars:
        yhatpath = cubic_path+'LR_'+str(upscale)+'x/'+mode+'/'+scalar+hash_id+'.dat'
        Yhat.append(np.memmap(yhatpath,dtype=np.float32).reshape(128,128,128))
    Yhat = np.stack(Yhat,axis=0)
    Y = []
    for scalar in scalars:
        ypath = data_path+'HR/'+mode+'/'+scalar+hash_id+'.dat'
        Y.append(np.memmap(ypath,dtype=np.float32).reshape(128,128,128))
    Y = np.stack(Y,axis=0)
    dx = torch.tensor(np.float32(train_dict['dx'][idx]))
    if train_dict['dy'][idx] != '':
        dy = torch.tensor(np.float32(train_dict['dy'][idx]))
    else:
        dy = dx
    if train_dict['dz'][idx] != '':
        dz = torch.tensor(np.float32(train_dict['dz'][idx]))
    else:
        dz = dx

    return torch.from_numpy(Yhat),torch.from_numpy(Y),dx,dy,dz

class CubicDataset(torch.utils.data.Dataset):
    def __init__(self, my_dict,cubic_path,path, mode,upscale,dx_min):
        self.train_dict = my_dict

        self.mode = mode
        self.path = path
        self.upscale = upscale
        self.dx_min = dx_min
        self.cubic_path = cubic_path
    def __len__(self):
        return len(self.train_dict['hash_id'])

    def __getitem__(self, idx):
        Yhat,Y,dx,dy,dz = get_cubicfile(idx,self.train_dict,self.cubic_path,self.path,self.mode,self.upscale)
        dx = dx/self.dx_min
        dy = dy/self.dx_min
        dz = dz/self.dx_min
        
        return Yhat, Y,dx,dy,dz

#dummy model for testing
class CubicModel(nn.Module):
    def __init__(self,upscale):
        super().__init__()
        self.upscale = upscale
        
    def forward(self, x):
        return x
    
class CubicLitModel(pl.LightningModule):
    def __init__(self, model,learning_rate=1e-4,loss_type='mse',dx_min= 3.906250185536919e-06,lr_sched_factor = 1,
                 grad_loss_weight=0.5):
        super().__init__()
        self.model = model
        self.upscale = model.upscale
        self.learning_rate = learning_rate
        self.loss_type = loss_type
        if self.loss_type == 'mse_grad':
            self.gradloss = GradLoss()
        self.SSIM = SSIM3D()
        self.dx_min = dx_min
        self.grad_loss_weight = grad_loss_weight
        self.lr_sched_factor = lr_sched_factor

  

    def test_step(self, batch, batch_idx, dataloader_idx):
        # training_step defines the train loop.
        # it is independent of forward
        y_hat, y,dx,dy,dz = batch

        #no need to rescale since we interpolate from the features without scaling
        rescaled_y_hat = y_hat
        rescaled_y = y
        #still need to rescale dx due to train script
        rescaled_dx = dx*self.dx_min
        rescaled_dy = dy*self.dx_min
        rescaled_dz = dz*self.dx_min

        #stresses
        stressp = sgs(rescaled_y_hat,self.upscale)
        stresst = sgs(rescaled_y,self.upscale)        
        divp1,divp2,divp3 = divergence_sgs_separate(stressp,dx=rescaled_dx*self.upscale,dy=rescaled_dy*self.upscale,dz=rescaled_dz*self.upscale)
        divt1,divt2,divt3 = divergence_sgs_separate(stresst,dx=rescaled_dx*self.upscale,dy=rescaled_dy*self.upscale,dz=rescaled_dz*self.upscale)
        divt1 = remove_edges(divt1)
        divt2 = remove_edges(divt2)
        divt3 = remove_edges(divt3)
        divp1 = remove_edges(divp1)
        divp2 = remove_edges(divp2)
        divp3 = remove_edges(divp3)

        stressp = remove_edges(stressp)
        stresst = remove_edges(stresst)


        ssim_divsgs1 = self.SSIM(divp1,divt1)
        mse_divsgs1 = F.mse_loss(divp1,divt1)
        ssim_divsgs2 = self.SSIM(divp2,divt2)
        mse_divsgs2 = F.mse_loss(divp2,divt2)
        ssim_divsgs3 = self.SSIM(divp3,divt3)
        mse_divsgs3 = F.mse_loss(divp3,divt3)

        ssim_sgs = self.SSIM(stressp,stresst)
        mse_sgs = F.mse_loss(stressp,stresst)

       
        rescaled_y_hat = remove_edges(rescaled_y_hat)
        rescaled_y = remove_edges(rescaled_y)
        mserho = F.mse_loss(rescaled_y_hat[:,0:1,:,:,:], rescaled_y[:,0:1,:,:,:])
        mseux = F.mse_loss(rescaled_y_hat[:,1:2,:,:,:], rescaled_y[:,1:2,:,:,:])
        mseuy = F.mse_loss(rescaled_y_hat[:,2:3,:,:,:], rescaled_y[:,2:3,:,:,:])
        mseuz = F.mse_loss(rescaled_y_hat[:,3:4,:,:,:], rescaled_y[:,3:4,:,:,:])
        ssimrho = self.SSIM(rescaled_y_hat[:,0:1,:,:,:], rescaled_y[:,0:1,:,:,:])
        ssimux = self.SSIM(rescaled_y_hat[:,1:2,:,:,:], rescaled_y[:,1:2,:,:,:])
        ssimuy = self.SSIM(rescaled_y_hat[:,2:3,:,:,:], rescaled_y[:,2:3,:,:,:])
        ssimuz = self.SSIM(rescaled_y_hat[:,3:4,:,:,:], rescaled_y[:,3:4,:,:,:])

        
        self.log('test_mse_rho', mserho)
        self.log('test_ssim_rho', ssimrho)
        self.log('test_mse_ux', mseux)
        self.log('test_ssim_ux', ssimux)
        self.log('test_mse_uy', mseuy)
        self.log('test_ssim_uy', ssimuy)
        self.log('test_mse_uz', mseuz)
        self.log('test_ssim_uz', ssimuz)

        self.log('test_mse_sgs', mse_sgs)
        self.log('test_ssim_sgs', ssim_sgs)

        self.log('test_mse_divsgs1', mse_divsgs1)
        self.log('test_ssim_divsgs1', ssim_divsgs1)
        self.log('test_mse_divsgs2', mse_divsgs2)
        self.log('test_ssim_divsgs2', ssim_divsgs2)
        self.log('test_mse_divsgs3', mse_divsgs3)
        self.log('test_ssim_divsgs3', ssim_divsgs3)

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)