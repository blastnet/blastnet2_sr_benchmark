#!/bin/bash
#BSUB -alloc_flags ipisolate
#BSUB -B
#BSUB -J rcan_0.5M_seed44
#BSUB -N
#BSUB -nnodes 4
#BSUB -o ./joboutputs/joboutput.rcan_0.5M.%J
#BSUB -q pbatch
#BSUB -W 0:30 


source /usr/workspace/chung34/anaconda39/bin/activate
conda activate opence-1.7.2-cuda-11.4
module unload cuda
export LD_LIBRARY_PATH=/path/to/anaconda/envs/opence-1.7.2-cuda-11.4/lib:$LD_LIBRARY_PATH

# set master addr to hostname of first compute node in allocation
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2

echo $LSB_QUEUE

rank_file=rank_info.$LSB_JOBID.txt
#SMT = 4 so OMP_NUM_THREADS
# -a 1 means 1 task per node
# -r 1 means 1 rank per task
# -g 4 means 4 gpus per rank
# -c 40 means 40 cores per rank
# -E OMP_NUM_THREADS=4 means 4 threads per core for SMT=4
jsrun -n ALL_HOSTS -a 4 -r 1 -g 4 -c 40 -E OMP_NUM_THREADS=10 -b none \
-d packed -l gpu-gpu,gpu-mem,cpu-mem,gpu-cpu \
python find_addport.py

jsrun -n ALL_HOSTS -a 4 -r 1 -g 4 -c 40 -E OMP_NUM_THREADS=10 -b none \
-d packed -l gpu-gpu,gpu-mem,cpu-mem,gpu-cpu js_task_info

###WARNING change --datapath to your own path #########################
jsrun -n ALL_HOSTS -a 4 -r 1 -g 4 -c 40 -E OMP_NUM_THREADS=10 -b none \
-d packed -l gpu-gpu,gpu-mem,cpu-mem,gpu-cpu \
python train.py --rank_file=../$LSB_JOBID.address_port.csv \
--data_path=../diverse_2K_with_extrap/ \
--upscale=8 --timeit  --nepochs=1500 \
--approx_param=0.5M --batch_size=4 \
--mode=$LSB_QUEUE --save_period=5 --case_name=$LSB_JOBNAME \
--precision=16 --num_nodes=4 --model_type=rcan --seed=44   --max_time=00:11:58:00


rm ../$LSB_JOBID.address_port.csv 
