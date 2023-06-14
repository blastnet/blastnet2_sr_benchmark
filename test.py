# Written by W.T. Chung
import os
import torch
import pytorch_lightning as pl
import time
import multiprocessing
import  json
from common import tools
print("CPU count:", multiprocessing.cpu_count())


# %%
from models.litmodel import LitModel
from models.rrdb import init_rrdb
from models.rcan import init_rcan
from models.edsr import init_edsr
from common import data

# %%
from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("--num_workers", type=int, default=6)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--ckpt_name", type=str, default="last.ckpt")
parser.add_argument("--num_nodes", type=int, default=1)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--upscale", type=int, default=8)
parser.add_argument("--train_meta", type=str, default='./metadata/train_data_summary.csv')
parser.add_argument("--val_meta", type=str, default='./metadata/val_data_summary.csv')
parser.add_argument("--test_meta", type=str, default='./metadata/test_data_summary.csv')
parser.add_argument("--forcedhit_meta", type=str, default='./metadata/forcedhit_data_summary.csv')
parser.add_argument("--paramvar_meta", type=str, default='./metadata/paramvar_data_summary.csv')
parser.add_argument("--data_path", type=str, default='./data/sample/')
parser.add_argument("--port", type=int, default=55001)
parser.add_argument("--approx_param", type=str, default='0.5M')
parser.add_argument("--timeit", action='store_true')
parser.add_argument("--gpu",type=int,default=1)
parser.add_argument("--precision",type=int,default=32)
parser.add_argument("--rank_file",type=str,default='../rank_file.txt')
parser.add_argument("--case_name",type=str,default='tmp')
parser.add_argument("--model_type", type=str, default='srresnet', help='srresnet | edsr | rcan')
parser.add_argument("--outpath", type=str, default='./eval/', help='output path')
parser.add_argument("--test_data", type=str, default='all', help='all | forcedhit | paramvar | test')

parser.set_defaults(timeit=True)

# Parse the user inputs and defaults (returns a argparse.Namespace)
args = parser.parse_args()

num_workers = args.num_workers
batch_size = args.batch_size
ckpt_name = args.ckpt_name
num_nodes = args.num_nodes
seed = args.seed
upscale = args.upscale
train_meta = args.train_meta
val_meta = args.val_meta
test_meta = args.test_meta
forcedhit_meta = args.forcedhit_meta
paramvar_meta = args.paramvar_meta
data_path = args.data_path
port = args.port
num_nodes = args.num_nodes
approx_param = args.approx_param
timeit = args.timeit
gpu = args.gpu
precision = args.precision
rank_file = args.rank_file
case_name = args.case_name
model_type = args.model_type
outpath = args.outpath
test_data = args.test_data
#  must be set before main block

log_dir = './logs/pbatch/'+case_name+'/csv_logs/'

if __name__ == '__main__':
    # %%
    pl.seed_everything(seed)

    # %%
    if train_meta == './metadata/train_data_summary.csv':
        dx_min = 3.906250185536919e-06
    else:
        raise ValueError('Please provide the correct train_meta file')
    
    test_mean, test_std = data.get_mean_std_test()
    extrapRe_mean, extrapRe_std = data.get_mean_std_extrapRe()
    extrapffcm_mean, extrapffcm_std = data.get_mean_std_extrapffcm()

    test_dict = data.my_read_csv(test_meta)
    forcedhit_dict = data.my_read_csv(forcedhit_meta)
    paramvar_dict = data.my_read_csv(paramvar_meta)

    test_scale_transform = data.ScaleTransform(test_mean,test_std)
    extrapRe_scale_transform = data.ScaleTransform(extrapRe_mean,extrapRe_std)
    extrapffcm_scale_transform = data.ScaleTransform(extrapffcm_mean,extrapffcm_std)
    

    test_ds = data.MyDataset(test_dict,data_path,'test',upscale,test_scale_transform,test_scale_transform,dx_min)
    test_loader = torch.utils.data.DataLoader(test_ds,batch_size=batch_size,
                            shuffle=False,num_workers=num_workers,pin_memory=True)
    
    forcedhit_ds = data.MyDataset(forcedhit_dict,data_path,'forcedhit',upscale,extrapffcm_scale_transform,extrapffcm_scale_transform,dx_min)
    forcedhit_loader = torch.utils.data.DataLoader(forcedhit_ds,batch_size=batch_size,
                            shuffle=False,num_workers=num_workers,pin_memory=True)
    paramvar_ds = data.MyDataset(paramvar_dict,data_path,'paramvar',upscale,extrapRe_scale_transform,extrapRe_scale_transform,dx_min)
    paramvar_loader = torch.utils.data.DataLoader(paramvar_ds,batch_size=batch_size,
                            shuffle=False,num_workers=num_workers,pin_memory=True)
    

    #get output from train_loader
    X0,Y0,_,_,_ = next(iter(test_loader))
    print("Test size: ",len(test_ds))
    print("Forced HIT size: ",len(forcedhit_ds))
    print("Param Var. size: ",len(paramvar_ds))
    print("num_workers: ",num_workers)
    print("X shape: ",X0.shape)
    print("Y shape: ",Y0.shape)
    print("Effective Batch size: ",batch_size*num_nodes*gpu)


    if model_type == 'rrdb':
        model = init_rrdb(approx_param=approx_param,upscale=upscale)
        find_unused_parameters = False 
    elif model_type == 'rcan':
        model = init_rcan(approx_param=approx_param,upscale=upscale)
    elif model_type == 'edsr':
        model = init_edsr(approx_param=approx_param,upscale=upscale)
    else:
        raise ValueError('Please provide the correct model_type')
    
    find_unused_parameters = False
    plugin_list = None   
    callback_list = None
    my_logger = None

    trainer = pl.Trainer(logger=my_logger,max_epochs=-1,
                            accelerator="gpu", devices=gpu,
                            num_nodes=num_nodes, precision=precision,
                            log_every_n_steps = 1, deterministic=True, 
                            plugins=plugin_list,callbacks=callback_list)
    print('Testing...')

    if timeit:
        start_time = time.time()
    
    if test_data == 'all':
        loader_names = ['test','forcedhit','paramvar']
        dataloaders = [test_loader,forcedhit_loader,paramvar_loader]
        mean_list = [test_mean,extrapffcm_mean,extrapRe_mean]
        std_list = [test_std,extrapffcm_std,extrapRe_std]
    else:
        loader_names = [test_data]
        if test_data == 'test':
            dataloaders = [test_loader]
        elif test_data == 'paramvar':
            dataloaders = [paramvar_loader]
        elif test_data == 'forcedhit': 
            dataloader = [forcedhit_loader]     
    
    for i in range(len(loader_names)):
        #todo correct this for pt files
        model.load_state_dict(torch.load(case_name))
        litmodel = LitModel(model=model,mean=mean_list[i],std=std_list[i],learning_rate=None,loss_type=None)
        results = trainer.test(model=litmodel, dataloaders=dataloaders[i],ckpt_path=None)
        filename = outpath + case_name+ckpt+'.'+loader_names[i]+'.json'
        with open(filename, 'w') as f:
            json.dump(results, f)
        
    if timeit:
        end_time = time.time()
        print('Test time: ' + str((end_time-start_time)/60))

    


