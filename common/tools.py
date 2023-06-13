import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
import glob

    

def get_metrics(log_path,metric):
    df = pd.read_csv(log_path)
    return df[['epoch',metric]].dropna()

def plot_metrics(df_list,scale='linear',normalize=False):
    plt.figure(figsize=(4,4))
    if normalize:
        for df in df_list:
            plt.plot(df['epoch'],df[df.columns[1]]/df[df.columns[1]].max(),label=df.columns[1])
    else:
        for df in df_list:
            plt.plot(df['epoch'],df[df.columns[1]],label=df.columns[1])
    plt.yscale(scale)
    #legend outside
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.grid()
    plt.show()

def get_best_epoch(path,metric):
    #folders in path
    folders = os.listdir(path)
    #get all csv files in folders
    csv_files = []
    for folder in folders:
        csv_files += glob.glob(os.path.join(path,folder,'*.csv'))
    #read csv files
    df_list = []
    for i,csv_file in enumerate(csv_files):
        df = pd.read_csv(csv_file)
        #check if df[metric] exists
        if metric not in df.columns:
            print(f'{metric} not in {csv_file}')
            continue
        df = df[['epoch',metric]].dropna()
        df_list.append(df)
    #concat dataframe
    df = pd.concat(df_list)
    #get best epoch
    best_epoch = df[df[metric] == df[metric].min()]['epoch'].values[0]
    return best_epoch

def my_estimate_batch_size(params,forward,memory):
    opt_batch_size = (memory - params)/forward
    return opt_batch_size
    
def calculate_global_mean_and_std(directory_path, scalar):
    # initialize variables for accumulating sum and count
    sum_vals = 0
    sum_squared_diffs = 0
    count_vals = 0

    # loop through directory to find binary files
    for file_name in os.listdir(directory_path):
        if file_name.startswith(scalar):
            # load binary file as numpy array
            file_path = os.path.join(directory_path, file_name)
            arr = np.fromfile(file_path, dtype=np.float32)

            # accumulate sum and count
            sum_vals += np.sum(arr)
            count_vals += arr.size

    # calculate global mean
    global_mean = sum_vals / count_vals

    # iterate over files again to calculate sum of squared differences from mean
    for file_name in os.listdir(directory_path):
        if file_name.startswith(scalar):
            # load binary file as numpy array
            file_path = os.path.join(directory_path, file_name)
            arr = np.fromfile(file_path, dtype=np.float32)

            # calculate squared differences from mean and accumulate
            squared_diffs = np.square(arr - global_mean)
            sum_squared_diffs += np.sum(squared_diffs)

    # calculate global standard deviation
    global_std = np.sqrt(sum_squared_diffs / count_vals)

    return global_mean, global_std

def calculate_global_max(directory_path, scalar):
    # initialize variables for accumulating sum and count
    max = 0 

    # loop through directory to find binary files
    for file_name in os.listdir(directory_path):
        if file_name.startswith(scalar):
            # load binary file as numpy array
            file_path = os.path.join(directory_path, file_name)
            arr = np.fromfile(file_path, dtype=np.float32)

            # accumulate sum and count
            arr = np.abs(arr)
            if np.max(arr) > max:
                max = np.max(arr)

    return max


class TestCalculateGlobalMeanAndStd():
    def setUp(self):
        # create a temporary directory and binary files with known values
        self.directory_path = "./tempdir"
        if not os.path.exists(self.directory_path):
            os.makedirs(self.directory_path)

        # create binary files with known values
        self.file1_path = os.path.join(self.directory_path, "file1.bin")
        file1 = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        file1.tofile(self.file1_path)

        self.file2_path = os.path.join(self.directory_path, "file2.bin")
        file2 = np.array([6, 7, 8, 9, 10], dtype=np.float32)
        file2.tofile(self.file2_path)

        # expected results for known values
        self.expected_mean = np.mean(np.hstack((file1, file2)))
        self.expected_std = np.std(np.hstack((file1, file2)))

    def test_calculate_global_mean_and_std(self):
        # calculate global mean and standard deviation
        global_mean, global_std = calculate_global_mean_and_std(self.directory_path, "file")

        # assert that results match expected values
        np.testing.assert_almost_equal(global_mean, self.expected_mean, decimal=5)
        np.testing.assert_almost_equal(global_std, self.expected_std, decimal=5)
        return global_mean, global_std

    def tearDown(self):
        # delete temporary directory and files
        os.remove(self.file1_path)
        os.remove(self.file2_path)
        os.rmdir(self.directory_path)
        
    def run(self):
        self.setUp()
        self.test_calculate_global_mean_and_std()
        self.tearDown()


if __name__ == "__main__":
    testmeanstd = TestCalculateGlobalMeanAndStd()
    testmeanstd.run()
