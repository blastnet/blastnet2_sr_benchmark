import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from .metrics import SSIM3D, GradLoss

import sys
sys.path.append('../')
from common.functions import sgs,remove_edges,divergence_sgs_separate



class LitModel(pl.LightningModule):
    def __init__(self, model,mean,std,learning_rate=1e-4,loss_type='mse',dx_min= 3.906250185536919e-06,lr_sched_factor = 0.5,
                 grad_loss_weight=0.5):
        super().__init__()
        self.model = model
        self.upscale = model.upscale
        self.learning_rate = learning_rate
        self.loss_type = loss_type
        if self.loss_type == 'mse_grad':
            self.gradloss = GradLoss()
        self.SSIM = SSIM3D()
        self.mean = mean[None,:,None,None,None] #broadcast to batch size
        self.std = std[None,:,None,None,None] #broadcast to batch size
        self.dx_min = dx_min
        self.grad_loss_weight = grad_loss_weight
        self.lr_sched_factor = lr_sched_factor

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y,_,_,_ = batch
        y_hat = self.model(x)
        if self.loss_type == 'mse':
            loss = F.mse_loss(y_hat, y)
        elif self.loss_type == 'l1':
            loss = F.l1_loss(y_hat, y)
        elif self.loss_type == 'mse_grad':
            gradloss  = self.grad_loss_weight*self.gradloss(y_hat,y)
            mseloss = (1.0-self.grad_loss_weight)*F.mse_loss(y_hat, y)
            loss = mseloss + gradloss
            self.log('train_weightedgradloss', gradloss, on_step=False, on_epoch=True,sync_dist=True)
            self.log('train_weightedmseloss', mseloss, on_step=False, on_epoch=True,sync_dist=True)
        else:
            raise ValueError('loss_type must be specified')
        self.log('train_'+self.loss_type+'loss', loss, on_step=False, on_epoch=True,sync_dist=True)

        return loss
    
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def validation_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y,_,_,_ = batch
        y_hat = self.model(x)

        mseloss = F.mse_loss(y_hat, y)

        self.log('val_mseloss', mseloss, on_step=False, on_epoch=True,sync_dist=True)

    def test_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y,dx,dy,dz = batch
        y_hat = self.model(x)

        #leave dx as scaled to avoid overflow
        mean = self.mean.type_as(dx)
        std = self.std.type_as(dx)
        rescaled_y_hat = y_hat*std + mean
        rescaled_y = y*std + mean
        rescaled_dx = dx*self.dx_min
        rescaled_dy = dy*self.dx_min
        rescaled_dz = dz*self.dx_min

        #stresses
        # print('calculating stresses')
        stressp = sgs(rescaled_y_hat,self.upscale)
        stresst = sgs(rescaled_y,self.upscale)        
        div1p,div2p,div3p = divergence_sgs_separate(stressp,dx=rescaled_dx*self.upscale,dy=rescaled_dy*self.upscale,dz=rescaled_dz*self.upscale)
        div1t,div2t,div3t = divergence_sgs_separate(stresst,dx=rescaled_dx*self.upscale,dy=rescaled_dy*self.upscale,dz=rescaled_dz*self.upscale)
        div1t = remove_edges(div1t)
        div2t = remove_edges(div2t)
        div3t = remove_edges(div3t)
        div1p = remove_edges(div1p)
        div2p = remove_edges(div2p)
        div3p = remove_edges(div3p)

        stressp = remove_edges(stressp)
        stresst = remove_edges(stresst)


        ssim_divsgs1 = self.SSIM(div1p,div1t)
        ssim_divsgs2 = self.SSIM(div2p,div2t)
        ssim_divsgs3 = self.SSIM(div3p,div3t)
        mse_divsgs1 = F.mse_loss(div1p,div1t)
        mse_divsgs2 = F.mse_loss(div2p,div2t)
        mse_divsgs3 = F.mse_loss(div3p,div3t)


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

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=300*self.lr_sched_factor,gamma=0.5)
        return [optimizer], [scheduler]
    
    def training_epoch_end(self, outputs):
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('learning_rate', lr, on_step=False, on_epoch=True,sync_dist=False)
    
    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)
