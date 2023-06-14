# blastnet2_sr_benchmark
Code for Super-Resolution Benchmark Study involving BLASTNet 2.0 Data
Written by W.T Chung with citations for open-source code in individual files.

Data for this super-resolution can be found at [Kaggle](https://www.kaggle.com/datasets/waitongchung/blastnet-momentum-3d-sr-dataset).

We also provided pre-trained weights in the same [Kaggle](https://www.kaggle.com/datasets/waitongchung/blastnet-momentum-3d-sr-dataset) repo. A sample is provided in the **weights** folder

Brief rundown on code:
1. **common** contains code for math, data loading, and general utilities.
2. **metadata** contains csv files with file ids that can be fed into the dataloaders for 5 different splits (train/test/val + 2 OOD sets)
3. **create_cubic_files.ipynb** precomputes cubic interpolation on files for a baseline comparison.
4. **find_addport.py** finds master port for multinode training.
5. **sample_lsf_{train,test,testcubic}.sh** are batch submission scripts for IBM LSF (not Slurm).
6. **{train,test,testcubic.py}** perform multinode training, and single-node evaluations for ML and cubic interpolation, respectively.
7. **requirements.txt** provides recommended packages to run this code