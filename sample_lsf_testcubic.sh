#!/bin/bash
#BSUB -alloc_flags ipisolate
#BSUB -B
#BSUB -J  cubic_8x 
#BSUB -N
#BSUB -nnodes 1
#BSUB -o ./joboutputs/joboutput.test_cubic_8x.%J
#BSUB -q pdebug
#BSUB -W 0:30 


source /usr/workspace/chung34/anaconda39/bin/activate
conda activate opence-1.7.2-cuda-11.4
module unload cuda
export LD_LIBRARY_PATH=/path/to/anaconda/envs/opence-1.7.2-cuda-11.4/lib:$LD_LIBRARY_PATH

#### Cubic files can be precomputed create_cubic_files.ipynb ########################
##WARNING change --datapath and --cubicpath to your own path #########################
python testcubic.py \
--data_path=../diverse_2K_with_extrap/ \
--cubic_path=../cubic/ \
--upscale=8 --timeit \
--batch_size=2  \
--precision=32 --num_nodes=1 
