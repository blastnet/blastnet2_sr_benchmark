# Written by W.T. Chung
import torch
import pytorch_lightning as pl
import time
# import matplotlib.pyplot as plt
import multiprocessing
import  json
print("CPU count:", multiprocessing.cpu_count())


# %%
from models.cubic import CubicLitModel,CubicModel,CubicDataset
from common import data
from argparse import ArgumentParser

parser = ArgumentParser()
# %%
parser.add_argument("--num_workers", type=int, default=6)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--num_nodes", type=int, default=1)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--upscale", type=int, default=8)
parser.add_argument("--test_meta", type=str, default='./metadata/test_data_summary.csv')
parser.add_argument("--forcedhit_meta", type=str, default='./metadata/forcedhit_data_summary.csv')
parser.add_argument("--paramvar_meta", type=str, default='./metadata/paramvar_data_summary.csv')
parser.add_argument("--data_path", type=str, default='./data/sample/')
parser.add_argument("--cubic_path", type=str, default='../cubic/')
parser.add_argument("--port", type=int, default=55001)
parser.add_argument("--timeit", action='store_true')
parser.add_argument("--gpu",type=int,default=1)
parser.add_argument("--precision",type=int,default=32)
parser.add_argument("--outpath", type=str, default='./eval/', help='output path')

args = parser.parse_args()

num_workers = args.num_workers
batch_size = args.batch_size
num_nodes = args.num_nodes
seed = args.seed
upscale = args.upscale
test_meta = args.test_meta
forcedhit_meta = args.forcedhit_meta
paramvar_meta = args.paramvar_meta
data_path = args.data_path
cubic_path = args.cubic_path
port = args.port
timeit = args.timeit
gpu = args.gpu
precision = args.precision
outpath = args.outpath

# %%
if __name__ == '__main__':

    pl.seed_everything(seed)


    dx_min = 3.906250185536919e-06


    test_dict = data.my_read_csv(test_meta)
    forcedhit_dict = data.my_read_csv(forcedhit_meta)
    paramvar_dict = data.my_read_csv(paramvar_meta)

    test_ds = CubicDataset(test_dict,cubic_path,data_path,'test',upscale,dx_min)
    test_loader = torch.utils.data.DataLoader(test_ds,batch_size=batch_size,
                            shuffle=False,num_workers=num_workers,pin_memory=True)

    forcedhit_ds = CubicDataset(forcedhit_dict,cubic_path,data_path,'forcedhit',upscale,dx_min)
    forcedhit_loader = torch.utils.data.DataLoader(forcedhit_ds,batch_size=batch_size,
                            shuffle=False,num_workers=num_workers,pin_memory=True)
    paramvar_ds = CubicDataset(paramvar_dict,cubic_path,data_path,'paramvar',upscale,dx_min)
    paramvar_loader = torch.utils.data.DataLoader(paramvar_ds,batch_size=batch_size,
                            shuffle=False,num_workers=num_workers,pin_memory=True)

    #get output from train_loader
    X0,Y0,_,_,_ = next(iter(test_loader))
    print("Test size: ",len(test_ds))
    print("ForcedHIT size: ",len(forcedhit_ds))
    print("ParamVar size: ",len(paramvar_ds))
    print("num_workers: ",num_workers)
    print("X shape: ",X0.shape)
    print("Y shape: ",Y0.shape)
    print("Effective Batch size: ",batch_size*num_nodes*gpu)

    model = CubicModel(upscale=upscale)
    model = CubicLitModel(model=model,learning_rate=None,loss_type=None)
    find_unused_parameters = False 


    plugin_list = None   


    callback_list = None


    #set up logger
    my_logger = None

    trainer = pl.Trainer(logger=my_logger,max_epochs=-1,
                            #accelerator="gpu", devices=gpu,
                            num_nodes=num_nodes, precision=precision,
                            log_every_n_steps = 1, deterministic=True, 
                            plugins=plugin_list,callbacks=callback_list)
    print('Testing...')

    if timeit:
        start_time = time.time()

    dataloaders = [test_loader,forcedhit_loader,paramvar_loader]
    loader_names = ['test','forcedhit','paramvar']

    results = trainer.test(model=model, dataloaders=dataloaders)
    for i, d in enumerate(results):
        filename = outpath +'cubic_'+str(upscale)+'xmodel.'+loader_names[i]+'.json'
        with open(filename, 'w') as f:
            json.dump(d, f)

    if timeit:
        end_time = time.time()
        print('Test time: ' + str((end_time-start_time)/60))

# %%




