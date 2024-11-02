import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import functional as F
import math
import numpy as np
import pandas as pd
import json
import os
import time

from ..dataloaders.UEID_loader import UEID_loader
from ..trainers.UEID_train import train
from ..utils.UEID_preprocessing import UEID_preprocessing

import pdb

class Seq2Event(nn.Module):
    def __init__(self, action_embedding_input_len, action_embedding_out_len, scale_grad_by_freq, 
                 continuous_embedding_input_len, continuous_embedding_output_len,
                 multihead_attention, hidden_dim,
                 transformer_num_layers,transformer_finaldenselayer_dim,
                 deltaT_output_len=1, location_output_len=2, action_output_len=9):
        super(Seq2Event, self).__init__()
        
        # Initialize layers based on input dimensions
        self.action_embedding = nn.Embedding(action_embedding_input_len, action_embedding_out_len, scale_grad_by_freq=scale_grad_by_freq)
        self.continuous_embedding = nn.Linear(continuous_embedding_input_len, continuous_embedding_output_len, bias=True)
        self.d_model=action_embedding_out_len+continuous_embedding_output_len
        self.positional_encoding = PositionalEncoding(d_model=self.d_model)

        # Transformer encoder layer
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=multihead_attention, batch_first=True, dim_feedforward=hidden_dim)
        self.seqnet = nn.TransformerEncoder(self.encoder_layer, num_layers=transformer_num_layers)
        
        # Prediction NN layers
        self.lin1 = nn.Linear(self.d_model,transformer_finaldenselayer_dim)
        self.lin2 = nn.Linear(transformer_finaldenselayer_dim, deltaT_output_len+location_output_len+action_output_len,bias=True)
    
    def forward(self, X):
        #input features 
        X_action=X[:,:,0]
        X_continuous=X[:,:,1:]

        #embedding
        X_action = self.action_embedding(X_action.int())
        X_continuous= self.continuous_embedding(X_continuous.float())
        X_concatenate = torch.cat((X_action,X_continuous),2).float()

        #positional encoding
        X_positional = X_concatenate+ self.positional_encoding(X_concatenate)
        X_positional=X_positional.float()
        
        #encoder layer
        X_encode= self.seqnet(X_positional)

        #prediction layers
        out = self.lin1(X_encode[:,-1,:])
        out = F.relu(out)
        out = self.lin2(out)

        return out
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Ensure div_term has the correct size
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Fill in the positional encoding
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:  # Handle odd d_model by adding a column of zeros
            pe[:, 1::2] = torch.cos(position * div_term[:d_model//2])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

def train_Seq2Event(current_time, config, train_df, valid_df, seq_len, features, epochs, batch_size,
            action_embedding_out_len, scale_grad_by_freq, 
            continuous_embedding_output_len,
            multihead_attention, hidden_dim, transformer_num_layers,transformer_finaldenselayer_dim,
            deltaT_output_len=1, location_output_len=2, action_output_len=9,
            num_actions=9, device=None, print_freq=10, patience=5, num_workers=4, optuna=False):

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
    
    # Define fixed hyperparameters 
    num_continous_features = len(features) - 1
    action_embedding_input_len=num_actions
    continuous_embedding_input_len=num_continous_features

    deltaT_output_len=1
    location_output_len=2
    action_output_len=num_actions

    #if not device is given, check if cuda is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None or device == 'None' else device

    # Initialize model layers based on data
    model = Seq2Event(action_embedding_input_len, action_embedding_out_len, scale_grad_by_freq, 
                continuous_embedding_input_len, continuous_embedding_output_len,
                multihead_attention, hidden_dim,
                transformer_num_layers,transformer_finaldenselayer_dim,
                deltaT_output_len, location_output_len, action_output_len).to(device)

    # Create data loaders
    train_loader = UEID_loader(train_df, seq_len, features)
    valid_loader = UEID_loader(valid_df, seq_len, features)

    # Convert data loaders to torch DataLoader
    train_loader = DataLoader(train_loader, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_loader, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
    
    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(),lr=0.01,eps=1e-16)

    # Training logic here
    model_state_dict, best_epoch, train_losses, valid_losses, train_loss_components, valid_loss_components, flops = train(model, train_loader, valid_loader, min_dict, max_dict, optimizer, device, epochs, print_freq, patience,config=config)

    #create one csv file for the training and validation losses
    columns = ['train_loss', 'CEL_action', 'RMSE_deltaT', 'RMSE_location', 'ACC_action', 'F1_action', 'MAE_deltaT', 'MAE_x', 'MAE_y',
                'valid_loss', 'CEL_action_v', 'RMSE_deltaT_v', 'RMSE_location_v', 'ACC_action_v', 'F1_action_v', 'MAE_deltaT_v', 'MAE_x_v', 'MAE_y_v']
    data = [train_losses, np.array(train_loss_components)[:,0], np.array(train_loss_components)[:,1], np.array(train_loss_components)[:,2], np.array(train_loss_components)[:,3], np.array(train_loss_components)[:,4], np.array(train_loss_components)[:,5], np.array(train_loss_components)[:,6], np.array(train_loss_components)[:,7],
            valid_losses, np.array(valid_loss_components)[:,0], np.array(valid_loss_components)[:,1], np.array(valid_loss_components)[:,2], np.array(valid_loss_components)[:,3], np.array(valid_loss_components)[:,4], np.array(valid_loss_components)[:,5], np.array(valid_loss_components)[:,6], np.array(valid_loss_components)[:,7]]
    data = np.array(data).T
    loss_df = pd.DataFrame(data, columns=columns)
    #round the loss_df to 4 decimal places
    loss_df = loss_df.round(4)

    
    #save the model and the loss_df
    if config['optuna']:
        method = "optuna"
    else:
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
    flops_save_path = save_path + "_model_stats.txt"

    #save the min_dict and max_dict
    min_max_dict_path = config['save_path']+f"/out/{method}/{current_time}/min_max_dict.json"
    with open(min_max_dict_path, 'w') as f:
        json.dump({'min_dict':min_dict, 'max_dict':max_dict}, f, indent=4)
    
    #save all the hyperparameters values
    hyperparameters = {'current_time': current_time,
                       'train_path': config['train_path'], 'valid_path': config['valid_path'], 'save_path': config['save_path'],
                        'test': config['test'], 'batch_size': batch_size, 'num_epoch': config['num_epoch'],
                        'print_freq': print_freq, 'early_stop_patience': config['early_stop_patience'],
                        'dataloader_num_worker': config['dataloader_num_worker'], 'device': config['device'], 
                        'basic_features': config['basic_features'], 'other_features': config['other_features'],
                        'use_other_features': config['use_other_features'],
                        'features':features, 'num_actions': num_actions, 'seq_len': seq_len,
                        'optuna': config['optuna'], 'optuna_n_trials': config['optuna_n_trials'],
                        'num_continous_features': num_continous_features, 'action_embedding_input_len': action_embedding_input_len,
                        'action_embedding_out_len': action_embedding_out_len, 'scale_grad_by_freq': scale_grad_by_freq,
                        'continuous_embedding_input_len': continuous_embedding_input_len,
                        'continuous_embedding_output_len': continuous_embedding_output_len,
                        'multihead_attention': multihead_attention, 'hidden_dim': hidden_dim,
                        'transformer_num_layers': transformer_num_layers, 
                        'transformer_finaldenselayer_dim': transformer_finaldenselayer_dim,
                        'deltaT_output_len':deltaT_output_len, 
                        'location_output_len':location_output_len, 
                        'action_output_len':action_output_len,
                        'model':model.__class__.__name__,
                        'best_epoch':best_epoch}
                
    torch.save(model_state_dict, model_save_path)
    loss_df.to_csv(loss_save_path, index=False)
    with open(hyperparameters_save_path, 'w') as f:
        json.dump(hyperparameters, f, indent=4)
    with open(flops_save_path, "w") as f:
        f.write(str(flops))
        

    print("Model and loss saved at", save_path)

    if optuna:
        return valid_losses[best_epoch]

def main(config):
    train_path = config['train_path']
    valid_path = config['valid_path']

    # Extract input dimensions from train_df
    if config['test']:
        seq_len = 10
        epochs = 2
        features = config['basic_features']+config['other_features']
    else:
        seq_len = config['seq_len']
        epochs = config['num_epoch']
        if config['use_other_features']:
            features = config['basic_features']+config['other_features']
        else:
            features = config['basic_features']
    
    batch_size = config['batch_size']
    num_actions = config['num_actions']
    print_freq = config['print_freq']
    early_stop_patience = config['early_stop_patience']
    dataloader_num_worker = config['dataloader_num_worker']
    device = config['device']

    #tunable hyperparameters
    action_embedding_out_len = config['action_embedding_out_len']
    scale_grad_by_freq = config['scale_grad_by_freq']
    continuous_embedding_output_len = config['continuous_embedding_output_len']
    
    multihead_attention = config['multihead_attention'] #fix to 1 given the previous papers
    hidden_dim = config['hidden_dim']
    transformer_num_layers = config['transformer_num_layers']
    transformer_finaldenselayer_dim = config['transformer_finaldenselayer_dim']

    current_time = time.strftime("%Y%m%d_%H%M%S")
    train_Seq2Event(current_time, config, train_path, valid_path, seq_len, features, epochs, batch_size,
                action_embedding_out_len, scale_grad_by_freq, 
                continuous_embedding_output_len,
                multihead_attention, hidden_dim,transformer_num_layers,transformer_finaldenselayer_dim,
                num_actions=num_actions,device=device,print_freq=print_freq, patience=early_stop_patience,num_workers=dataloader_num_worker)

def main_optuna(config):
    import optuna
    train_path = config['train_path']
    valid_path = config['valid_path']

    # Extract input dimensions from train_df
    if config['test']:
        seq_len = 10
        epochs = 2
        features = config['basic_features']+config['other_features']
    else:
        seq_len = config['seq_len']
        epochs = config['num_epoch']
        if config['use_other_features']:
            features = config['basic_features']+config['other_features']
        else:
            features = config['basic_features']
    
    batch_size = config['batch_size']
    num_actions = config['num_actions']
    print_freq = config['print_freq']
    early_stop_patience = config['early_stop_patience']
    dataloader_num_worker = config['dataloader_num_worker']
    device = config['device']

    def objective(trial):
        #tunable hyperparameters
        try:
            action_embedding_out_len=trial.suggest_categorical('action_embedding_out_len', config['action_embedding_out_len'])
            scale_grad_by_freq=trial.suggest_categorical('scale_grad_by_freq', config['scale_grad_by_freq'])
            continuous_embedding_output_len=trial.suggest_categorical('continuous_embedding_output_len', config['continuous_embedding_output_len'])
            
            multihead_attention=1 #fix to 1 given the previous papers
            hidden_dim=trial.suggest_categorical('hidden_dim', config['hidden_dim'])
            transformer_num_layers=trial.suggest_categorical('transformer_num_layers', config['transformer_num_layers'])
            transformer_finaldenselayer_dim=trial.suggest_categorical('transformer_finaldenselayer_dim', config['transformer_finaldenselayer_dim'])
        except:
            print("Error in the yaml file")
            
        # Call your train function with these parameters
        valid_loss = train_Seq2Event(current_time, config, train_path, valid_path, seq_len, features, epochs, batch_size,
                action_embedding_out_len, scale_grad_by_freq, 
                continuous_embedding_output_len,
                multihead_attention, hidden_dim,transformer_num_layers,transformer_finaldenselayer_dim, optuna=True,
                num_actions=num_actions,device=device,print_freq=print_freq, patience=early_stop_patience,num_workers=dataloader_num_worker)
        
        return valid_loss


    n_trials = config['optuna_n_trials'] if not config['test'] else 2

    # Create an Optuna study object
    study = optuna.create_study(direction='minimize')
    
    # Optimize the objective function
    current_time = time.strftime("%Y%m%d_%H%M%S")
    study.optimize(objective, n_trials=n_trials)
    
    # Print the best trial results
    print("Best trial:")
    trial = study.best_trial
    best_trial_number = trial.number+1
    best_params = trial.params
    print(f"  Trial Number: {best_trial_number}")
    print(f"  Best Valid Loss: {trial.value:.4f}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    # Add the trial number to the parameters dictionary
    best_params_with_trial = {
        'best_trial_number': best_trial_number,
        'best_valid_loss': trial.value,
        **best_params
    }

    # Save the best hyperparameters with the trial number to a JSON file
    with open(config['save_path']+f'out/optuna/{current_time}/optuna_best_hyperparameters.json', 'w') as f:
        json.dump(best_params_with_trial, f, indent=4)

if __name__ == "__main__":
    import pandas as pd
    import argparse
    import yaml

    prase = argparse.ArgumentParser()
    prase.add_argument('--config_path', '-c', type=str, default=os.getcwd()+'/event/sports/soccer/models/model_yaml_test/train_Seq2Event.yaml')
    args = prase.parse_args()
    
    config_path = args.config_path

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    if config['optuna']:
        main_optuna(config)
    else:
        main(config)

    '''
    all_features = ['match_id','poss_id','team','home_team','action','success','goal','home_score',
                'away_score','goal_diff','Period','Minute','Second','seconds','delta_T',
                'start_x','start_y','deltaX','deltaY','distance','dist2goal','angle2goal']
    basic_features = ['action', 'delta_T', 'start_x','start_y'] 
    derivable_features = ['team','home_team','action','success','seconds','delta_T','start_x','start_y','deltaX','deltaY','distance','dist2goal','angle2goal']
    other_features = ['team','home_team','success','seconds','deltaX','deltaY','distance','dist2goal','angle2goal']
    '''
    
