# Written by W.T. Chung
import os
import torch
import pytorch_lightning as pl
import time
from pytorch_lightning.strategies import DDPStrategy
import numpy as np
# import matplotlib.pyplot as plt
import multiprocessing
print("CPU count:", multiprocessing.cpu_count())


# %%
from models.litmodel import LitModel
from models.rrdb import init_rrdb
from models.rcan import init_rcan
from models.edsr import init_edsr
from models.convfno import init_convfno

from common import data

# %%
from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--num_workers", type=int, default=6)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--nepochs", type=int, default=600)
parser.add_argument("--ckpt_name", type=str, default="last.ckpt")
parser.add_argument("--num_nodes", type=int, default=1)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--upscale", type=int, default=8)
parser.add_argument("--train_meta", type=str, default='./metadata/train_data_summary.csv')
parser.add_argument("--val_meta", type=str, default='./metadata/val_data_summary.csv')
parser.add_argument("--data_path", type=str, default='./data/sample/')
parser.add_argument("--port", type=int, default=55001)
parser.add_argument("--tune",action='store_true')
parser.add_argument("--fast_dev_run", type=int, default=0)
parser.add_argument("--profiler",type=str,default=None)
parser.add_argument("--accumulate_grad_batches",type=int,default=1)
parser.add_argument("--approx_param",type=str,default='0.5M')
parser.add_argument("--log_gpu_memory",action='store_true')
parser.add_argument("--timeit", action='store_true')
parser.add_argument("--gpu",type=int,default=4)
parser.add_argument("--save_period",type=int,default=1)
parser.add_argument("--max_time",type=str,default="00:11:58:00")
parser.add_argument("--precision",type=int,default=16)
parser.add_argument("--rank_file",type=str,default='../rank_file.txt')
parser.add_argument("--mode",type=str,default='pbatch')
parser.add_argument("--case_name",type=str,default='tmp')
parser.add_argument("--loss_type", type=str, default='mse', help='mse | mse_grad')
parser.add_argument("--model_type", type=str, default='rrdb', help='rrdb | edsr | rcan ')
parser.add_argument("--grad_loss_weight", type=float, default=0.99)
parser.add_argument("--lr_sched_factor",type=int,default=0.5,help='lr scheduler multiplier')
parser.set_defaults(tune=False)
parser.set_defaults(timeit=True)
parser.set_defaults(log_gpu_memory=False)

args = parser.parse_args()

learning_rate = args.learning_rate
num_workers = args.num_workers
nepochs = args.nepochs
batch_size = args.batch_size
ckpt_name = args.ckpt_name
num_nodes = args.num_nodes
seed = args.seed
upscale = args.upscale
train_meta = args.train_meta
val_meta = args.val_meta
data_path = args.data_path
port = args.port
tune = args.tune
num_nodes = args.num_nodes
fast_dev_run = args.fast_dev_run
profiler = args.profiler
accumulate_grad_batches = args.accumulate_grad_batches
approx_param = args.approx_param
log_gpu_memory = args.log_gpu_memory
timeit = args.timeit
gpu = args.gpu
save_period = args.save_period
max_time = args.max_time
precision = args.precision
rank_file = args.rank_file
mode = args.mode
case_name = args.case_name
loss_type = args.loss_type
model_type = args.model_type
grad_loss_weight = args.grad_loss_weight
lr_sched_factor=args.lr_sched_factor
#  must be set before main block


### this block is specfic for IBM LSF based multi-node multi-gpu training on LLNL Lassen ###
os.environ["WORLD_SIZE"] = str(num_nodes*gpu)
world_rank = int(os.environ.get('OMPI_COMM_WORLD_RANK'))
info = np.loadtxt(rank_file,delimiter=',',dtype=str)
master_addr = info[0]
port = info[1]
os.environ['MASTER_PORT'] = str(port)
os.environ["NODE_RANK"] = str(world_rank//gpu)
os.environ["LOCAL_RANK"] = str(world_rank%gpu)
os.environ['MASTER_ADDR'] = master_addr
print("WORLD_RANK: ",world_rank,", NODE_RANK: ",os.environ["NODE_RANK"],", LOCAL_RANK: ",os.environ["LOCAL_RANK"], ", WORLD_SIZE: ",os.environ["WORLD_SIZE"],
      ", MASTER_PORT: ",os.environ['MASTER_PORT'],", MASTER_ADDR: ",os.environ['MASTER_ADDR'])
####### replace this block with your own multi-node multi-gpu env setup ####################


detect_anomaly = False

#debug settings 
if mode == 'pdebug':
    nepochs = save_period*2
    log_gpu_memory = True
    profiler = 'simple'
    detect_anomaly = True

#define save dirs
log_dir = './logs/'+mode+'/'+case_name
checkpoint_dir = '../ckpt/'+mode+'/'+case_name+'/'

if __name__ == '__main__':
    # %%
    pl.seed_everything(seed)

    my_mean,my_std = data.get_mean_std()
    assert my_mean is not None and my_std is not None, 'Please provide mean and std for the dataset'

    # %%
    if './metadata/train_data_summary' in train_meta:
        dx_min = 3.906250185536919e-06
    else:
        raise ValueError('Please provide the correct train_meta file')
    

    #DataLoader setup
    train_dict = data.my_read_csv(train_meta)
    val_dict = data.my_read_csv(val_meta)
    scale_transform = data.ScaleTransform(my_mean,my_std)

    train_ds = data.MyDataset(train_dict,data_path,'train',upscale,scale_transform,scale_transform,dx_min)
    train_loader = torch.utils.data.DataLoader(train_ds,batch_size=batch_size,
                            shuffle=True,num_workers=num_workers,pin_memory=True)
    

    val_ds = data.MyDataset(val_dict,data_path,'val',upscale,scale_transform,scale_transform,dx_min)
    val_loader = torch.utils.data.DataLoader(val_ds,batch_size=batch_size,
                            shuffle=False,num_workers=num_workers,pin_memory=True)

    #get output from train_loader
    X0,Y0,_,_,_ = next(iter(train_loader))
    train_ds_len = len(train_ds)
    if world_rank == 0:
        print("Train size: ",train_ds_len)
        print("Val size: ",len(val_ds))
        print("num_workers: ",num_workers)
        print("X shape: ",X0.shape)
        print("Y shape: ",Y0.shape)
        print("Effective Batch size: ",batch_size*accumulate_grad_batches*num_nodes*gpu)

    # Model setup 
    if model_type == 'rrdb':
        model = init_rrdb(approx_param=approx_param,upscale=upscale)
        find_unused_parameters = False 
    elif model_type == 'rcan':
        model = init_rcan(approx_param=approx_param,upscale=upscale)
    elif model_type == 'edsr':
        model = init_edsr(approx_param=approx_param,upscale=upscale)
    elif model_type == 'convfno':
        model = init_convfno(approx_param=approx_param,upscale=upscale)
    else:
        raise ValueError('Please provide the correct model_type')
    
    model = LitModel(model=model,mean=my_mean,std=my_std,learning_rate=learning_rate,loss_type=loss_type,lr_sched_factor = lr_sched_factor,
                                 grad_loss_weight=grad_loss_weight)
    find_unused_parameters = False

    #generic cluster environment for multi-node multi-gpu training
    plugin_list = [pl.plugins.environments.LightningEnvironment()]     

    #call backs for saving best mse and save every 10*5 epochs
    loss_callback = pl.callbacks.ModelCheckpoint(dirpath=checkpoint_dir, save_top_k=3, every_n_epochs = save_period, 
                                                        monitor="val_mseloss", mode="min",save_last=True, 
                                                        save_on_train_epoch_end=False,
                                                        filename='model-{epoch:02d}-{val_mseloss:.4e}')
    epoch_callback = pl.callbacks.ModelCheckpoint(dirpath=checkpoint_dir, save_top_k=-1, every_n_epochs = 10*save_period,
                                                        monitor="epoch", mode="max",save_last=False, 
                                                        save_on_train_epoch_end=False,
                                                        filename='model-{epoch:02d}')

    callback_list = [loss_callback,epoch_callback]

    #for profiling 
    if log_gpu_memory == True:
        gpu_call_back = pl.callbacks.DeviceStatsMonitor()
        if callback_list is not None:
            callback_list.append(gpu_call_back)
        else:
            callback_list = [gpu_call_back]

    #set up csv logger
    csv_logger = pl.loggers.CSVLogger(save_dir=log_dir, name="csv_logs",
                                      flush_logs_every_n_steps=save_period*train_ds_len//(batch_size*num_nodes*gpu))
    my_logger = [csv_logger]

    #Train!
    trainer = pl.Trainer(logger=my_logger,max_epochs=nepochs,
                             max_time=max_time,
                            accelerator="gpu", devices=gpu,strategy=DDPStrategy(find_unused_parameters=find_unused_parameters),
                            num_nodes=num_nodes, sync_batchnorm=True, precision=precision,
                            log_every_n_steps = 1, deterministic=True,detect_anomaly =detect_anomaly, 
                            accumulate_grad_batches=accumulate_grad_batches,
                            plugins=plugin_list,callbacks=callback_list,
                            check_val_every_n_epoch=save_period,
                            fast_dev_run = fast_dev_run, profiler=profiler)
    if world_rank == 0:
        print('Training...')

    if world_rank == 0:
        if timeit:
            start_time = time.time()
    if os.path.exists(checkpoint_dir+ckpt_name):
        trainer.fit(model=model, train_dataloaders=train_loader,val_dataloaders=val_loader,ckpt_path=checkpoint_dir+ckpt_name)
    else:
        trainer.fit(model=model, train_dataloaders=train_loader,val_dataloaders=val_loader)

    if timeit:
        end_time = time.time()
        if world_rank == 0:    
            print('Training time: ' + str((end_time-start_time)/60))



