import pandas as pd
import numpy as np
import matplotsoccer
import matplotlib.pyplot as plt
import scipy
from scipy.ndimage import gaussian_filter1d
import json
import os
import pdb  


def plot_heat_map(inference_data_path, min_max_dict_path, save_path, row_num):
    inference_data = pd.read_csv(inference_data_path)

    with open(min_max_dict_path, 'r') as f:
        min_max_dict = json.load(f)
    min_dict= min_max_dict['min_dict']
    max_dict= min_max_dict['max_dict']

    required_col_y = [f'start_y_{i}_prob' for i in range(101)]
    required_col_x = [f'start_x_{i}_prob' for i in range(101)]

    y_bins = np.linspace(0, 100, 101)/100
    x_bins = np.linspace(0, 100, 101)/100

    #scale the bins with the min_max_dict
    y_bins = y_bins*(max_dict["start_y"]-min_dict["start_y"]) + min_dict["start_y"]
    x_bins = x_bins*(max_dict["start_x"]-min_dict["start_x"]) + min_dict["start_x"]

    #check if the required columns are present in the data
    if not set(required_col_y+required_col_x).issubset(inference_data.columns):
        raise ValueError("Missing columns in the data")

    row_sums_x = inference_data[required_col_x].sum(axis=1)
    row_sums_y = inference_data[required_col_y].sum(axis=1)
    if not np.allclose(row_sums_x, 1):
        #scale the probabilities to sum to 1
        inference_data[required_col_x] = inference_data[required_col_x].div(row_sums_x, axis=0)
    if not np.allclose(row_sums_y, 1):
        #scale the probabilities to sum to 1
        inference_data[required_col_y] = inference_data[required_col_y].div(row_sums_y, axis=0)

    
    #fill NaN values with 0
    inference_data.fillna(0, inplace=True)

    y_prob = inference_data[required_col_y].values[row_num]
    x_prob = inference_data[required_col_x].values[row_num]

    # apply gaussian filter to smooth the heatmap
    y_prob = gaussian_filter1d(y_prob, 1)
    x_prob = gaussian_filter1d(x_prob, 1)

    # pdb.set_trace()

    #based on the probabilities, sample the x and y coordinates proportionally
    x_data = np.random.choice(x_bins, 100000, p=x_prob)
    y_data = np.random.choice(y_bins, 100000, p=y_prob)

    hm = matplotsoccer.count( pd.Series(x_data), pd.Series(y_data),n=101,m=101) # Construct a 25x25 heatmap from x,y-coordinates
    hm = scipy.ndimage.gaussian_filter(hm,1) # blur the heatmap
    fig, ax = plt.subplots()
    matplotsoccer.heatmap(hm)
    plt.savefig(save_path+"heatmap.png")
    plt.close(fig)

if __name__ == "__main__":
    min_max_dict_path = os.getcwd() + "/test/model/LEM/out/optuna/20241009_010549/min_max_dict.json"
    inference_data_path = os.getcwd() + "/test/inference/LEM/inference.csv"
    save_path = os.getcwd()+"/test/application/"

    row_num = 100
    plot_heat_map(inference_data_path, min_max_dict_path, save_path, row_num)