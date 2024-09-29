import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.init as init
import math
import numpy as np
import pandas as pd
import json
import os
import time

from ..dataloaders.UEID_loader import UEID_loader
from ..trainers.MAJ_train import MAJ_train
from ..utils.UEID_preprocessing import UEID_preprocessing

import pdb

class MAJ(nn.Module):
    # the model is a simple majority class classifier
    def __init__(self, bin_list_action, bin_list_delta_T, bin_list_start_x, bin_list_start_y):
        super(MAJ, self).__init__()
        # Initialize layers based on input dimensions
        self.action_dist = bin_list_action
        self.delta_T_dist = bin_list_delta_T
        self.start_x_dist = bin_list_start_x
        self.start_y_dist = bin_list_start_y

    
    def forward(self, X):
        batch_size = X.size(0) 
        # return the majority class batch size times
        action = torch.tensor([self.action_dist]*batch_size)
        delta_T = torch.tensor([self.delta_T_dist]*batch_size)
        start_x = torch.tensor([self.start_x_dist]*batch_size)
        start_y = torch.tensor([self.start_y_dist]*batch_size)
        #combine the tensors into one tensor
        out = torch.cat((action, delta_T, start_x, start_y), dim=1)

        return out

         

def get_prob_dist(df,feature, bin=9):
    """
    Get the probability distribution of the features
    """
    bin_list = [0]*bin
    if feature == 'action':
        for i in range(bin):
            bin_list[i] = len(df[df['action']==i])/len(df)
    else:
        df[feature] = (df[feature]*100).astype(int)
        for i in range(bin):
            bin_list[i] = len(df[df[feature]==i])/len(df)

    return bin_list
  
def train_MAJ(current_time, config, train_df, valid_df, seq_len, features, epochs=1, batch_size=256, 
            num_actions=9, device=None, print_freq=10, patience=5, num_workers=4):

    # check if the train_df and valid_df are pandas dataframe
    if not isinstance(train_df, pd.DataFrame):
        #check if the train_df is a path
        if isinstance(train_df, str):
            train_df = pd.read_csv(train_df) if not config['test'] else pd.read_csv(train_df).head(1000)
        else:
            raise ValueError("train_df must be a pandas dataframe or a path to a csv file")
    if not isinstance(valid_df, pd.DataFrame):
        #check if the valid_df is a path
        if isinstance(valid_df, str):
            valid_df = pd.read_csv(valid_df) if not config['test'] else pd.read_csv(valid_df).head(1000)
        else:
            raise ValueError("valid_df must be a pandas dataframe or a path to a csv file")
        
    # Preprocess the data
    train_df, min_dict, max_dict = UEID_preprocessing(train_df)
    valid_df, _, _ = UEID_preprocessing(valid_df, min_dict, max_dict)

    # Get the probability distribution of the features
    bin_list_action = get_prob_dist(train_df, 'action', num_actions)
    bin_list_delta_T = get_prob_dist(train_df, 'delta_T', config['delta_T_bin'])
    bin_list_start_x = get_prob_dist(train_df, 'start_x', config['start_x_bin'])
    bin_list_start_y = get_prob_dist(train_df, 'start_y', config['start_y_bin'])

    #if not device is given, check if cuda is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None or device == 'None' else device

    # Initialize model layers based on data
    model = MAJ(bin_list_action, bin_list_delta_T, bin_list_start_x, bin_list_start_y).to(device)
    
    # Create data loaders
    train_loader = UEID_loader(train_df, seq_len, features)
    valid_loader = UEID_loader(valid_df, seq_len, features)

    # Convert data loaders to torch DataLoader
    train_loader = DataLoader(train_loader, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_loader, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
    

    # Training logic here
    model_state_dict, best_epoch, train_losses, valid_losses, train_loss_components, valid_loss_components = MAJ_train(model, train_loader, valid_loader, min_dict, max_dict, device, epochs, print_freq, patience,config=config)

    #create one csv file for the training and validation losses
    columns = ['train_loss', 'BCEL_action', 'BCEL_deltaT', 'BCEL_start_x', 'BCEL_start_y', 'ACC_action', 'F1_action', 'AE_deltaT', 'AE_start_x', 'AE_start_y',
                'valid_loss', 'BCEL_action_v', 'BCEL_deltaT_v', 'BCEL_start_x_v', 'BCEL_start_y_v', 'ACC_action_v', 'F1_action_v', 'AE_deltaT_v', 'AE_start_x_v', 'AE_start_y_v']
    data = [train_losses, np.array(train_loss_components)[:,0], np.array(train_loss_components)[:,1], np.array(train_loss_components)[:,2], np.array(train_loss_components)[:,3], np.array(train_loss_components)[:,4], np.array(train_loss_components)[:,5], np.array(train_loss_components)[:,6], np.array(train_loss_components)[:,7], np.array(train_loss_components)[:,8],
            valid_losses, np.array(valid_loss_components)[:,0], np.array(valid_loss_components)[:,1], np.array(valid_loss_components)[:,2], np.array(valid_loss_components)[:,3], np.array(valid_loss_components)[:,4], np.array(valid_loss_components)[:,5], np.array(valid_loss_components)[:,6], np.array(valid_loss_components)[:,7], np.array(valid_loss_components)[:,8]]
    data = np.array(data).T
    loss_df = pd.DataFrame(data, columns=columns)
    #round the loss_df to 4 decimal places
    loss_df = loss_df.round(4)

    #save the model and the loss_df
    method = "train"

    i=1
    save_path = config['save_path']+f"/out/{method}/{current_time}/run_{i}/"
    while os.path.exists(save_path):
        i+=1
        save_path = config['save_path']+f"/out/{method}/{current_time}/run_{i}/"
    os.makedirs(save_path)

    model_save_path = save_path + f"_model_{best_epoch}.pth"
    loss_save_path = save_path + "_loss.csv"
    hyperparameters_save_path = save_path + "hyperparameters.json"

    #save the min_dict and max_dict
    min_max_dict_path = config['save_path']+f"/out/{method}/{current_time}/min_max_dict.json"
    with open(min_max_dict_path, 'w') as f:
        json.dump({'min_dict':min_dict, 'max_dict':max_dict}, f, indent=4)
    
    #save all the hyperparameters values
    hyperparameters = {'current_time': current_time,
                       'train_path': config['train_path'], 'valid_path': config['valid_path'], 'save_path': config['save_path'],
                        'test': config['test'], 'num_epochs': epochs, 'print_freq': print_freq, 
                        'dataloader_num_worker': config['dataloader_num_worker'], 'batch_size': batch_size, 
                        'device': config['device'], 'basic_features': config['basic_features'], 'features': features,
                        'other_features': [],
                        'num_actions': num_actions, 
                        'seq_len': config['seq_len'],
                        'delta_T_bin': config['delta_T_bin'], 'start_x_bin': config['start_x_bin'], 'start_y_bin': config['start_y_bin'],
                        'bin_list_action': bin_list_action, 'bin_list_delta_T': bin_list_delta_T, 'bin_list_start_x': bin_list_start_x, 'bin_list_start_y': bin_list_start_y,
                        'model':model.__class__.__name__,
                        'best_epoch':best_epoch}
                
    # torch.save(model_state_dict, model_save_path)
    loss_df.to_csv(loss_save_path, index=False)
    with open(hyperparameters_save_path, 'w') as f:
        json.dump(hyperparameters, f, indent=4)
        
    print("Model and loss saved at", save_path)

def main(config):
    train_path = config['train_path']
    valid_path = config['valid_path']

    # Extract input dimensions from train_df
    if config['test']:
        seq_len = 3
        epochs = 1
        features = config['basic_features']
    else:
        seq_len = config['seq_len']
        epochs = config['num_epoch']
        features = config['basic_features']
    
    batch_size = config['batch_size']
    num_actions = config['num_actions']
    print_freq = config['print_freq']
    device = config['device']
    num_workers = config['dataloader_num_worker']

    current_time = time.strftime("%Y%m%d_%H%M%S")
    train_MAJ(current_time, config, train_path, valid_path, seq_len, features, epochs, batch_size, 
            num_actions, device, print_freq, num_workers)
    
if __name__ == "__main__":
    import argparse
    import yaml

    prase = argparse.ArgumentParser()
    prase.add_argument('--config_path', '-c', type=str, default=os.getcwd()+'/event/sports/soccer/models/model_yaml_test/train_MAJ.yaml')
    args = prase.parse_args()
    
    config_path = args.config_path

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    main(config)