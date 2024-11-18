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
import yaml

from ..dataloaders.UEID_loader import UEID_loader
from ..trainers.LEM_train import LEM_train
from ..utils.UEID_preprocessing import UEID_preprocessing
from .LEM_action import LEM_action

import pdb

class LEM(nn.Module): 
    #ref https://github.com/nvsclub/LargeEventsModel/blob/main/lib/model_utils.py#L94
    def __init__(self, hidden_size, output_size, input_size_action, hidden_size_action, output_size_action,pth_action,
        activation='relu',activation_action='relu'):
        super(LEM, self).__init__()

        input_size = output_size_action + input_size_action

        activation_dict = {
            'relu': nn.ReLU,
            'sigmoid': nn.Sigmoid,
            'tanh': nn.Tanh,
            'leaky_relu': nn.LeakyReLU,
        }
        layers = [
            nn.Linear(input_size, hidden_size[0]),
            activation_dict[activation]()
        ] + flatten([
            [nn.Linear(hidden_size[i], hidden_size[i+1]),
            activation_dict[activation]()] for i in range(len(hidden_size) - 1)
        ]) + [
            nn.Linear(hidden_size[-1], output_size),
            nn.Sigmoid()
        ]

        self.model = nn.Sequential(*layers)
        
        # Initialize the linear layers
        self.init_weights()
        self.action_model=LEM_action(input_size_action, hidden_size_action, output_size_action, activation_action)
        #load the weights of the action model
        self.action_model.load_state_dict(torch.load(pth_action, weights_only=True))
        #forzen the weights of the action model
        for param in self.action_model.parameters():
            param.requires_grad = False

    def init_weights(self):
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                init.zeros_(m.bias)
    
    def forward(self, x):
        x_action = self.action_model(x)
        #concatenate the output of the action model to the input of the event model
        x = torch.cat((x, x_action), 1) if len(x.shape) == 2 else torch.cat((x, x_action), 2)
        x = self.model(x)
        x = torch.cat((x_action, x), 1) if len(x.shape) == 2 else torch.cat((x_action, x), 2)
        return x

def flatten(l):
    return [item for sublist in l for item in sublist]

def train_LEM(current_time, config, train_df, valid_df, seq_len, features, epochs, batch_size,
            hidden_size, activation,
            num_actions=9, device=None, print_freq=10, patience=5, num_workers=4, optuna=False,
            learning_rate=0.01,eps=1e-16):

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
    input_size=(num_actions+len(features)-1)*seq_len
    output_size = config['delta_T_bin']+config['start_x_bin']+config['start_y_bin']                        #61 for delta_T, 101 for start_x, 101 for start_y
    LEM_action_json_path=config['LEM_action_json_path']
    #load the hyperparameters of the action model
    with open(LEM_action_json_path, 'r') as f:
        action_hyperparameters = json.load(f)
    input_size_action = action_hyperparameters['input_size']
    hidden_size_action = action_hyperparameters['hidden_size']
    output_size_action = action_hyperparameters['output_size']
    activation_action = action_hyperparameters['activation']
    pth_action = config['LEM_aciton_pth_path']

    #if not device is given, check if cuda is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None or device == 'None' else device

    # Initialize model layers based on data
    model = LEM(hidden_size, output_size, input_size_action, hidden_size_action, output_size_action,pth_action,
        activation,activation_action).to(device)
    
    # Create data loaders
    train_loader = UEID_loader(train_df, seq_len, features)
    valid_loader = UEID_loader(valid_df, seq_len, features)

    # Convert data loaders to torch DataLoader
    train_loader = DataLoader(train_loader, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_loader, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
    
    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate,eps=eps)

    # Training logic here
    model_state_dict, best_epoch, train_losses, valid_losses, train_loss_components, valid_loss_components, model_stats = LEM_train(model, train_loader, valid_loader, min_dict, max_dict, optimizer, device, epochs, print_freq, patience,config=config)

    #create one csv file for the training and validation losses
    columns = ['train_loss', 'BCEL_continuous', 'ACC_action','F1_action','MAE_deltaT', 'MAE_start_x', 'MAE_start_y',
                'valid_loss', 'BCEL_continuous_v', 'ACC_action_v','F1_action_v', 'AE_deltaT_v', 'MAE_start_x_v', 'MAE_start_y_v']
    data = [train_losses, np.array(train_loss_components)[:,0], np.array(train_loss_components)[:,1], np.array(train_loss_components)[:,2], np.array(train_loss_components)[:,3], np.array(train_loss_components)[:,4], np.array(train_loss_components)[:,5],
            valid_losses, np.array(valid_loss_components)[:,0], np.array(valid_loss_components)[:,1], np.array(valid_loss_components)[:,2], np.array(valid_loss_components)[:,3], np.array(valid_loss_components)[:,4], np.array(valid_loss_components)[:,5]]
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
    model_stats_save_path = save_path + "_model_stats.txt"

    #save the min_dict and max_dict
    min_max_dict_path = config['save_path']+f"/out/{method}/{current_time}/min_max_dict.json"
    with open(min_max_dict_path, 'w') as f:
        json.dump({'min_dict':min_dict, 'max_dict':max_dict}, f, indent=4)
    
    #save all the hyperparameters values
    hyperparameters = {'current_time': current_time,
                       'train_path': config['train_path'], 'valid_path': config['valid_path'], 'save_path': config['save_path'],
                       'LEM_action_json_path': LEM_action_json_path, 'LEM_aciton_pth_path': pth_action,
                        'test': config['test'], 'batch_size': batch_size, 'num_epoch': config['num_epoch'],
                        'print_freq': print_freq, 'early_stop_patience': config['early_stop_patience'],
                        'dataloader_num_worker': config['dataloader_num_worker'], 'device': config['device'], 
                        'basic_features': config['basic_features'], 'other_features': config['other_features'],
                        'use_other_features': config['use_other_features'],
                        'features':features, 'num_actions': num_actions, 'seq_len': seq_len,
                        'delta_T_bin': config['delta_T_bin'], 'start_x_bin': config['start_x_bin'], 'start_y_bin': config['start_y_bin'],
                        'optuna': config['optuna'], 'optuna_n_trials': config['optuna_n_trials'],
                        'learning_rate': learning_rate, 'eps': eps,
                        'hidden_size': hidden_size, 'output_size': output_size,
                        'activation': activation,
                        'input_size_action': input_size_action, 'hidden_size_action': hidden_size_action,
                        'output_size_action': output_size_action, 'activation_action': activation_action,
                        'model':model.__class__.__name__,
                        'best_epoch':best_epoch}
                
    torch.save(model_state_dict, model_save_path)
    loss_df.to_csv(loss_save_path, index=False)
    with open(hyperparameters_save_path, 'w') as f:
        json.dump(hyperparameters, f, indent=4)
    with open(model_stats_save_path, "w") as f:
        json.dump(model_stats, f, indent=4)
        
    print("Model and loss saved at", save_path)

    if optuna:
        return valid_losses[best_epoch]

def main(config):
    train_path = config['train_path']
    valid_path = config['valid_path']

    # Extract input dimensions from train_df
    if config['test']:
        seq_len = config['seq_len']
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
    learning_rate = config['learning_rate']
    eps = config['eps']
    eps = float(eps) if not isinstance(eps, float) else eps
    device = config['device']

    #tunable hyperparameters
    hidden_size = config['hidden_size']
    activation = config['activation']

    current_time = time.strftime("%Y%m%d_%H%M%S")
    train_LEM(current_time, config, train_path, valid_path, seq_len, features, epochs, batch_size,
                hidden_size,activation,
                num_actions=num_actions,device=device,print_freq=print_freq, patience=early_stop_patience,num_workers=dataloader_num_worker,
                learning_rate=learning_rate,eps=eps)

def main_optuna(config):
    import optuna
    train_path = config['train_path']
    valid_path = config['valid_path']

    # Extract input dimensions from train_df
    if config['test']:
        seq_len = config['seq_len']
        epochs = 2
        features = config['basic_features']+config['other_features']
    else:
        seq_len = config['seq_len']
        epochs = config['num_epoch']
        if config['use_other_features']:
            features = config['basic_features']+config['other_features']
        else:
            features = config['basic_features']
    
    num_actions = config['num_actions']
    print_freq = config['print_freq']
    early_stop_patience = config['early_stop_patience']
    dataloader_num_worker = config['dataloader_num_worker']
    device = config['device']

    def objective(trial):
        # tunable hyperparameters
        try:
            learning_rate=trial.suggest_float('learning_rate', float(config['learning_rate'][0]), float(config['learning_rate'][1])) if len(config['learning_rate'])>1 else config['learning_rate'][0]
            config['eps'] = [float(i) for i in config['eps']]
            eps=trial.suggest_float('eps',  float(config['eps'][1]), float(config['eps'][1])) if len(config['eps'])>1 else config['eps'][0]
            batch_size=trial.suggest_categorical('batch_size', config['batch_size']) 

            hidden_layers = trial.suggest_categorical('hidden_layers', config['hidden_layers'])
            hidden_size = []
            for i in range(hidden_layers):
                hidden_size.append(trial.suggest_categorical(f'hidden_size_{i}', config['hidden_size']))
            activation = trial.suggest_categorical('activation', config['activation'])
        except:
            print("Error in the yaml file")
            
        # Call your train function with these parameters
        valid_loss = train_LEM(current_time, config, train_path, valid_path, seq_len, features, epochs, batch_size,
                hidden_size, activation, optuna=True,
                num_actions=num_actions,device=device,print_freq=print_freq, patience=early_stop_patience,num_workers=dataloader_num_worker,
                learning_rate=learning_rate,eps=eps)
        
        return valid_loss


    n_trials = config['optuna_n_trials'] if not config['test'] else 2

    # Create an Optuna study object
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(),direction='minimize')
    
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

    import argparse
    import yaml

    prase = argparse.ArgumentParser()
    prase.add_argument('--config_path', '-c', type=str, default=os.getcwd()+'/event/sports/soccer/models/model_yaml_test/train_LEM.yaml')
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
    
