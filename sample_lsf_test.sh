#!/bin/bash
#BSUB -alloc_flags ipisolate
#BSUB -B
#BSUB -J rcan_0.5M
#BSUB -N
#BSUB -nnodes 1
#BSUB -o ./joboutputs/joboutput.test_rcan_0.5M.%J
#BSUB -q pdebug
#BSUB -W 0:30 


source /usr/workspace/chung34/anaconda39/bin/activate
conda activate opence-1.7.2-cuda-11.4
module unload cuda
export LD_LIBRARY_PATH=/path/to/anaconda/envs/opence-1.7.2-cuda-11.4/lib:$LD_LIBRARY_PATH

# set master addr to hostname of first compute node in allocation

###WARNING change --datapath to your own path #########################
python test.py --rank_file=../$LSB_JOBID.address_port.csv \
--data_path=../diverse_2K_with_extrap/ \
--upscale=8 --timeit \
--approx_param=0.5M --case_name=./weights/seed42/rcan_approx0.5M_8xSR.pt \
--precision=32 --num_nodes=1 --model_type=rcan


