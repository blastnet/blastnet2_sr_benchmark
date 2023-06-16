# BLASTNet 2.0 Super-Resolution Benchmark
Code for Super-Resolution (SR) Benchmark Study involving BLASTNet 2.0 Data
Written by W.T Chung and B. Akoush with citations for open-source code in individual files.

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

This code works with Python 3.9.  

## Data

Data for this benchmark can be found at [Kaggle](https://www.kaggle.com/datasets/waitongchung/blastnet-momentum-3d-sr-dataset).

## Summary

Brief rundown on code:
1. **common** contains code for math, data loading, and general utilities.
2. **metadata** contains csv files with file ids that can be fed into the dataloaders for 5 different splits (train/test/val + 2 OOD sets)
3. **create_cubic_files.ipynb** precomputes cubic interpolation on files for a baseline comparison.
4. **find_addport.py** finds master port for multinode training.
5. **sample_lsf_{train,test,testcubic}.sh** are batch submission scripts for IBM LSF (not Slurm).
6. **{train,test,testcubic}.py** perform multinode training, and single-node evaluations for ML and cubic interpolation, respectively.
7. **requirements.txt** provides recommended packages to run this code

## Training

To train the models, we provide a **sample_lsf_train.sh** that provides multi-node training via a batch submission on IBM LSF (not SLURM).

## Evaluation

To eval the models, we provide a **sample_lsf_test.sh** that provides single-gpu evaluation via a batch submission on IBM LSF (not SLURM). Essentially it does::

```eval

python test.py \
--data_path=../diverse_2K_with_extrap/ \
--upscale=8 --timeit \
--approx_param=0.5M --case_name=./weights/seed42/rcan_approx0.5M_8xSR.pt \
--precision=32 --num_nodes=1 --model_type=rcan

```
If you want to compare the ML models with cubic interpolation. You can use  **create_cubic_files.ipynb** to obtain interpolated data. And use **sample_lsf_testcubic.sh**, which does the same as: 

```eval cubic interp.

python testcubic.py \
--data_path=../diverse_2K_with_extrap/ \
--cubic_path=../cubic/ \
--upscale=8 --timeit \
--batch_size=2  \
--precision=32 --num_nodes=1 

```
## Pre-trained Models

We also provided pre-trained weights in the same [Kaggle](https://www.kaggle.com/datasets/waitongchung/blastnet-momentum-3d-sr-dataset) repo. A sample is provided in the **weights** folder.

