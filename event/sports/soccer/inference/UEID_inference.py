import torch
from ..utils.UEID_preprocessing import UEID_preprocessing
from ..dataloaders.UEID_loader import UEID_loader
from ..dataloaders.UEID_loader import UEID_Simulation_loader
from ..utils.UEID_cost_function import cost_function
from torch.utils.data import DataLoader
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import pdb


def UEID_inference(train_df, data, model_name, model_path, model_config, num_workers=4, batch_size=64, device=None, min_max_dict_path=None):
    # Load the training configuration 
    with open(model_config, "r") as f:
        config = json.load(f)
    
    #check if train_df and data are paths or dataframes
    if train_df is not None and train_df != "None":
        if not isinstance(train_df, pd.DataFrame):
            #check if the train_df is a path
            if isinstance(train_df, str):
                train_df = pd.read_csv(train_df) if not config['test'] else pd.read_csv(train_df).head(1000)
            else:
                raise ValueError("train_df must be a pandas dataframe or a path to a csv file")
    if isinstance(data, pd.DataFrame):
        data_raw = data
    elif isinstance(data, str):
        data_raw = pd.read_csv(data) if not config['test'] else pd.read_csv(data).head(1000)
    else:
        raise ValueError("data must be a pandas dataframe or a path to a csv file")

    data = data_raw.copy()

    #set the hyperparameters
    seq_len = config["seq_len"]
    features = config["features"]
    num_workers = config["dataloader_num_worker"]
    batch_size = config["batch_size"]
    device = config["device"]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None or device == 'None' else device
    
    # Preprocess the data
    if min_max_dict_path != "None" and min_max_dict_path is not None:
        try:
            with open(min_max_dict_path, "r") as f:
                min_max_dict = json.load(f)
            min_dict = min_max_dict['min_dict']
            max_dict = min_max_dict['max_dict']
        except:
            train_df, min_dict, max_dict = UEID_preprocessing(train_df)
    else:
        train_df, min_dict, max_dict = UEID_preprocessing(train_df)
    data, _, _ = UEID_preprocessing(data, min_dict, max_dict)

    # Create the data loader
    data_loader = UEID_loader(data, seq_len, features)
    data_loader = DataLoader(data_loader, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False)

    # Load the model
    model = load_model(model_name, model_path, model_config)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.to(device)
    model.eval()
    
    # Run inference
    total_loss = 0
    total_loss_components = [0]*8
    with torch.no_grad():
        predicted = []
        end_idx_list = []
        for i, (input_seq, gt, end_idx) in tqdm(enumerate(data_loader), total=len(data_loader)):
            input_seq, gt = input_seq.to(device), gt.to(device)
            output = model(input_seq)
            #check if output is in device
            if output.device != device:
                output = output.to(device)
            loss, loss_components= cost_function(gt, output,  min_dict=min_dict, max_dict=max_dict, device=device,config=config)
            total_loss += loss.item()
            loss_components = [i.cpu().detach().numpy() for i in loss_components] if device != "cpu" else [i.detach().numpy() for i in loss_components]
            total_loss_components = [total_loss_components[j] + loss_components[j] for j in range(len(loss_components))]
            if device != "cpu":
                predicted.append(output.cpu().numpy())
                end_idx_list.append(end_idx.cpu().numpy())
            else:
                predicted.append(output.numpy())
                end_idx_list.append(end_idx.numpy())
        # Concatenate the results
        predicted = np.concatenate(predicted, axis=0)
        end_idx_list = np.concatenate(end_idx_list, axis=0)
        average_loss = total_loss / len(data_loader)
        average_loss_components = [total_loss_components[j] / len(data_loader) for j in range(len(loss_components))]
        
    #save the loss_df
    columns= ['train_loss', 'CEL_action', 'RMSE_deltaT', 'RMSE_location', 'ACC_action', 'F1_action', 'MAE_deltaT', 'MAE_x', 'MAE_y']
    loss_df = pd.DataFrame([[average_loss]+average_loss_components], columns=columns)
    loss_df = loss_df.round(4)
    
    for i, idx in enumerate(end_idx_list):
        data_raw.loc[idx, 'action_pred'] = np.argmax(predicted[i, :config['num_actions']])
        data_raw.loc[idx, 'delta_T_pred'] = predicted[i, config['num_actions']]
        data_raw.loc[idx, 'start_x_pred'] = predicted[i, config['num_actions']+1]
        data_raw.loc[idx, 'start_y_pred'] = predicted[i, config['num_actions']+2]

        for action_i in range(config['num_actions']):
            data_raw.loc[idx, f'action_{action_i}_prob'] = predicted[i, action_i]
        data_raw.loc[idx, 'delta_T_pred_unscaled'] = np.exp(predicted[i, config['num_actions']]*(max_dict["delta_T"]-min_dict["delta_T"]) + min_dict["delta_T"]) - 1e-6
        data_raw.loc[idx, 'start_x_pred_unscaled'] = predicted[i, config['num_actions']+1]*(max_dict["start_x"]-min_dict["start_x"]) + min_dict["start_x"]
        data_raw.loc[idx, 'start_y_pred_unscaled'] = predicted[i, config['num_actions']+2]*(max_dict["start_y"]-min_dict["start_y"]) + min_dict["start_y"]
    
    return data_raw, loss_df

def UEID_simulation_possession(train_df, data, model_name, model_path, model_config, num_workers=4, batch_size=64, device=None, random_selection=False, max_iter=200, min_max_dict_path=None):
        # Load the training configuration 
    with open(model_config, "r") as f:
        config = json.load(f)
    
    #check if train_df and data are paths or dataframes
    if train_df is not None and train_df != "None":
        if not isinstance(train_df, pd.DataFrame):
            #check if the train_df is a path
            if isinstance(train_df, str):
                train_df = pd.read_csv(train_df) if not config['test'] else pd.read_csv(train_df).head(1000)
            else:
                raise ValueError("train_df must be a pandas dataframe or a path to a csv file")
    if isinstance(data, pd.DataFrame):
        data_raw = data
    elif isinstance(data, str):
        data_raw = pd.read_csv(data) if not config['test'] else pd.read_csv(data).head(1000)
    else:
        raise ValueError("data must be a pandas dataframe or a path to a csv file")
    data = data_raw.copy()

    #set the hyperparameters
    seq_len = config["seq_len"]
    features = config["features"]
    num_workers = config["dataloader_num_worker"]
    batch_size = config["batch_size"]
    device = config["device"]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None or device == 'None' else device
    
    # Preprocess the data
    if min_max_dict_path != "None" and min_max_dict_path is not None:
        try:
            with open(min_max_dict_path, "r") as f:
                min_max_dict = json.load(f)
            min_dict = min_max_dict['min_dict']
            max_dict = min_max_dict['max_dict']
        except:
            train_df, min_dict, max_dict = UEID_preprocessing(train_df)
    else:
        train_df, min_dict, max_dict = UEID_preprocessing(train_df)

    data, _, _ = UEID_preprocessing(data, min_dict, max_dict)

    # Create the data loader
    data_loader = UEID_Simulation_loader(data, seq_len, features)
    data_loader = DataLoader(data_loader, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False)

    # Load the model
    model = load_model(model_name, model_path, model_config)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.to(device)
    model.eval()

    # Run inference
    with torch.no_grad():
        simulation = {}
        for i, (input_seq, end_idx) in tqdm(enumerate(data_loader)):
            #convert the end_idx to a list
            end_idx = end_idx.cpu().numpy().tolist() if device != "cpu" else end_idx.numpy().tolist()

            for idx in end_idx:
                simulation[idx] = []

            input_seq = input_seq.to(device)
            iteration = 1

            # Run the simulation
            while end_idx:
                # Forward pass through the model
                output = model(input_seq)

                # Convert the first config['num_actions'] to action probabilities
                action_probs = torch.nn.functional.softmax(output[:, :config['num_actions']], dim=1)

                # Select actions based on the probabilities
                if random_selection:
                    action = torch.multinomial(action_probs, 1)
                    action_prob = torch.gather(action_probs, 1, action)
                else:
                    action = torch.argmax(action_probs, dim=1)
                    action = action.unsqueeze(1)
                    action_prob = torch.gather(action_probs, 1, action)
                    
                # Recombine the action with the delta_T, start_x, start_y
                output = torch.cat((action.float(), action_prob.float(), output[:, config['num_actions']:]), dim=1)

                # Update the simulation dictionary
                mask = torch.ones(len(end_idx), dtype=bool)  # Initialize a mask to keep all entries

                for j, idx in enumerate(end_idx):
                    output_j = output[j].cpu().numpy().tolist() if device != "cpu" else output[j].numpy().tolist()
                    action_j = int(output_j[0])
                    action_prob_j = output_j[1]
                    delta_T_j = np.exp(output_j[2]*(max_dict["delta_T"]-min_dict["delta_T"]) + min_dict["delta_T"]) - 1e-6
                    start_x_j = output_j[3]*(max_dict["start_x"]-min_dict["start_x"]) + min_dict["start_x"]
                    start_y_j = output_j[4]*(max_dict["start_y"]-min_dict["start_y"]) + min_dict["start_y"]
                    
                    simulation[idx].append([iteration, action_j, action_prob_j, delta_T_j, start_x_j, start_y_j])

                    # Check if the current action indicates termination
                    if output[j, 0] == 3:
                        mask[j] = False  # Mark as False to remove this index

                #drop the action_prob
                output =torch.cat((output[:, 0].unsqueeze(1), output[:, 2:]), dim=1)

                # Apply the mask to remove finished sequences
                if not mask.any():
                    break  # Exit the loop if all sequences are finished
                
                print(f"Iteration: {iteration}") if config['test'] and __name__ == "__main__" else None 
                if iteration >= max_iter:
                    break
                else:
                    iteration += 1
                
                # Update end_idx and tensors based on the mask
                end_idx = [end_idx[j] for j in range(len(end_idx)) if mask[j]]
                input_seq = input_seq[mask]
                output = output[mask]
                
                # Create the required input sequence for the next step
                if len(config['other_features'])>0:
                    # other_features = ['team','home_team','success','seconds','deltaX','deltaY','distance','dist2goal','angle2goal']
                    #get the previous timestep
                    prev_timestep = input_seq[:,-1,:]
                    #create a tensor with the same shape as the prev_timestep
                    append_tensor = torch.zeros((prev_timestep.shape[0],len(config['features'])))
                    #convert the prev_timestep to a numpy array
                    prev_timestep = prev_timestep.cpu().numpy() if device != "cpu" else prev_timestep.numpy()
                    #get the previous timestep's action
                    prev_action = prev_timestep[:,0]
                    #create a position dict for the position of the config['other_features'] in config['features']
                    position_dict = {feature:config['features'].index(feature) for feature in config['other_features']}
                    #if action is 3, then update the 'team','home_team','success','seconds'
                    for i, output_i in enumerate(output):
                        if output_i[0] == 3:
                            action = output_i[0]
                            delta_T = 0
                            x = output_i[2]
                            y = output_i[3]
                            team_idx= position_dict.get('team')
                            team = prev_timestep[i,team_idx] if team_idx is not None else None
                            home_team_idx = position_dict.get('home_team')
                            home_team = prev_timestep[i,home_team_idx] if home_team_idx is not None else None
                            success_idx = position_dict.get('success')
                            success = 0 if success_idx is not None else None
                            seconds_idx = position_dict.get('seconds')
                            if seconds_idx is not None:
                                seconds = prev_timestep[i,position_dict['seconds']]
                                seconds = seconds*(max_dict["seconds"]-min_dict["seconds"]) + min_dict["seconds"]
                                seconds = seconds + np.exp(prev_timestep[i,1]*(max_dict["delta_T"]-min_dict["delta_T"]) + min_dict["delta_T"]) - 1e-6
                            deltaX_idx = position_dict.get('deltaX')
                            deltaX = 0 if deltaX_idx is not None else None
                            deltaY_idx = position_dict.get('deltaY')
                            deltaY = 0 if deltaY_idx is not None else None
                            distance_idx = position_dict.get('distance')
                            distance = 0 if distance_idx is not None else None
                            dist2goal_idx = position_dict.get('dist2goal')
                            dist2goal = 0 if dist2goal_idx is not None else None
                            angle2goal_idx = position_dict.get('angle2goal')
                            angle2goal = 0.5 if angle2goal_idx is not None else None
                            home_score_idx = position_dict.get('home_score')
                            home_score = prev_timestep[i,position_dict['home_score']] if home_score_idx is not None else None
                            away_score_idx = position_dict.get('away_score')
                            away_score = prev_timestep[i,position_dict['away_score']] if away_score_idx is not None else None
                        else:
                            action = output_i[0]
                            delta_T = output_i[1]
                            x = output_i[2]
                            y = output_i[3]
                            team_idx= position_dict.get('team')
                            if team_idx is not None:
                                previous_team = prev_timestep[i,team_idx]
                                match_id = data.loc[end_idx[i]]['match_id']
                                other_team = data[data['match_id'] == match_id].team.unique().tolist()
                                other_team.remove(previous_team)
                                other_team = other_team[0]
                                team = previous_team if prev_action[i] !=3 else other_team
                                if not isinstance(team, int):
                                    team = team.astype(int)
                            home_team_idx = position_dict.get('home_team')
                            if home_team_idx is not None:
                                previous_home_team = prev_timestep[i,home_team_idx]
                                home_team = previous_home_team if prev_action[i] !=3 else abs(1-previous_home_team)
                                if not isinstance(home_team, int):
                                    home_team = home_team.astype(int)
                            success_idx = position_dict.get('success')
                            if success_idx is not None:
                                success = 1 if action not in [3,6] else 0
                                if not isinstance(success, int):
                                    success = success.astype(int)
                            seconds_idx = position_dict.get('seconds')
                            if seconds_idx is not None:
                                seconds = prev_timestep[i,position_dict['seconds']]
                                seconds = seconds*(max_dict["seconds"]-min_dict["seconds"]) + min_dict["seconds"]
                                seconds = seconds + np.exp(prev_timestep[i,1]*(max_dict["delta_T"]-min_dict["delta_T"]) + min_dict["delta_T"]) - 1e-6
                                if not isinstance(seconds, float):
                                    seconds = seconds.astype(float)
                            deltaX_idx = position_dict.get('deltaX')
                            if deltaX_idx is not None:
                                deltaX = x*(max_dict["start_x"]-min_dict["start_x"]) + min_dict["start_x"]
                                deltaX = x - prev_timestep[i,2]*(max_dict["start_x"]-min_dict["start_x"]) + min_dict["start_x"]
                            deltaY_idx = position_dict.get('deltaY')
                            if deltaY_idx is not None:
                                deltaY = y*(max_dict["start_y"]-min_dict["start_y"]) + min_dict["start_y"]
                                deltaY = y - prev_timestep[i,3]*(max_dict["start_y"]-min_dict["start_y"]) + min_dict["start_y"]
                            distance_idx = position_dict.get('distance')
                            if distance_idx is not None:
                                distance = torch.sqrt((deltaX)**2 + (deltaY)**2)
                            dist2goal_idx = position_dict.get('dist2goal')
                            if dist2goal_idx is not None:
                                dist2goal = torch.sqrt((x*(max_dict["start_x"]-min_dict["start_x"]) + min_dict["start_x"] - 105)**2 + (y*(max_dict["start_y"]-min_dict["start_y"]) + min_dict["start_y"] - 34)**2)
                            angle2goal_idx = position_dict.get('angle2goal')
                            if angle2goal_idx is not None:
                                angle2goal = torch.abs(torch.atan2((y*(max_dict["start_y"]-min_dict["start_y"]) + min_dict["start_y"] - 34), (x*(max_dict["start_x"]-min_dict["start_x"]) + min_dict["start_x"] - 105)))
                            home_score_idx = position_dict.get('home_score')
                            if home_score_idx is not None:
                                if action == 8 and home_team == 1:
                                    home_score = prev_timestep[i,position_dict['home_score']]+1
                                else:
                                    home_score = prev_timestep[i,position_dict['home_score']]
                            away_score_idx = position_dict.get('away_score')
                            if away_score_idx is not None:
                                if action == 8 and home_team == 0:
                                    away_score = prev_timestep[i,position_dict['away_score']]+1
                                else:
                                    away_score = prev_timestep[i,position_dict['away_score']]  
                        #preprocess the features
                        for feature in config['other_features']:
                            if feature == 'seconds':
                                seconds = (seconds - min_dict['seconds'])/(max_dict['seconds'] - min_dict['seconds'])
                            elif feature == 'deltaX':
                                deltaX = (deltaX - min_dict['deltaX'])/(max_dict['deltaX'] - min_dict['deltaX'])
                            elif feature == 'deltaY':
                                deltaY = (deltaY - min_dict['deltaY'])/(max_dict['deltaY'] - min_dict['deltaY'])
                            elif feature == 'distance':
                                distance = (distance - min_dict['distance'])/(max_dict['distance'] - min_dict['distance'])
                            elif feature == 'dist2goal':
                                dist2goal = (dist2goal - min_dict['dist2goal'])/(max_dict['dist2goal'] - min_dict['dist2goal'])
                            elif feature == 'angle2goal':
                                angle2goal = (angle2goal - min_dict['angle2goal'])/(max_dict['angle2goal'] - min_dict['angle2goal'])
                            elif feature == 'home_score':
                                if max_dict['home_score'] - min_dict['home_score'] != 0:
                                    home_score = (home_score - min_dict['home_score'])/(max_dict['home_score'] - min_dict['home_score'])
                                else:
                                    home_score = 0
                            elif feature == 'away_score':
                                if max_dict['away_score'] - min_dict['away_score'] != 0:
                                    away_score = (away_score - min_dict['away_score'])/(max_dict['away_score'] - min_dict['away_score'])
                                else:
                                    away_score = 0
                        #update the append_tensor
                        append_tensor[i,0] = action
                        append_tensor[i,1] = delta_T
                        append_tensor[i,2] = x
                        append_tensor[i,3] = y
                        if team_idx is not None:
                            append_tensor[i,team_idx] = team
                        if home_team_idx is not None:
                            append_tensor[i,home_team_idx] = home_team.astype(int)
                        if success_idx is not None: #TODO: adjust this if success is added as a target feature
                            append_tensor[i,success_idx] = success
                        if seconds_idx is not None:
                            append_tensor[i,seconds_idx] = seconds.astype(float)
                        if deltaX_idx is not None:
                            append_tensor[i,deltaX_idx] = deltaX
                        if deltaY_idx is not None:
                            append_tensor[i,deltaY_idx] = deltaY
                        if distance_idx is not None:
                            append_tensor[i,distance_idx] = distance
                        if dist2goal_idx is not None:
                            append_tensor[i,dist2goal_idx] = dist2goal
                        if angle2goal_idx is not None:
                            append_tensor[i,angle2goal_idx] = angle2goal
                        if home_score_idx is not None:
                            append_tensor[i,home_score_idx] = home_score
                        if away_score_idx is not None:
                            append_tensor[i,away_score_idx] = away_score
                    append_tensor = append_tensor.to(device).unsqueeze(1)
                    input_seq = torch.cat((input_seq[:, 1:, :], append_tensor), dim=1)     
                else:
                    # Update the input sequence for the next step
                    input_seq = torch.cat((input_seq[:, 1:, :], output.unsqueeze(1)), dim=1)

        rows = []
        for key, value_list in simulation.items():
            for value in value_list:
                rows.append([key] + value)

        # Convert the list of rows into a DataFrame
        df = pd.DataFrame(rows, columns=['index', 'iteration', 'action', 'action_prob', 'delta_T', 'x', 'y'])
        #set to 4 dp
        df = df.round(4)

        # Ensure for each index the last action is action=3
        for idx in df['index'].unique():
            # Get the last row for the given index
            last_row_idx = df[df['index'] == idx].index[-1]
            
            # Check if the last action is not 3
            if df.loc[last_row_idx, 'action'] != 3:
                # Change the last row's action to 3
                df.loc[last_row_idx, 'action'] = 3

    return df

def UEID_simulation_match(train_df, data, model_name, model_path, model_config, num_workers=4, batch_size=64, device=None, random_selection=False, max_iter=200, min_max_dict_path=None, simulation_time=45,match_id=None): 
        # Load the training configuration 
    with open(model_config, "r") as f:
        config = json.load(f)
    
    #check if train_df and data are paths or dataframes
    if train_df is not None and train_df != "None":
        if not isinstance(train_df, pd.DataFrame):
            #check if the train_df is a path
            if isinstance(train_df, str):
                train_df = pd.read_csv(train_df) if not config['test'] else pd.read_csv(train_df).head(1000)
            else:
                raise ValueError("train_df must be a pandas dataframe or a path to a csv file")
    if isinstance(data, pd.DataFrame):
        data_raw = data
    elif isinstance(data, str):
        data_raw = pd.read_csv(data) if not config['test'] else pd.read_csv(data).head(1000)
    else:
        raise ValueError("data must be a pandas dataframe or a path to a csv file")
    data = data_raw.copy()

    #set the hyperparameters
    seq_len = config["seq_len"]
    features = config["features"]
    num_workers = config["dataloader_num_worker"]
    batch_size = config["batch_size"]
    device = config["device"]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None or device == 'None' else device
    
    # Preprocess the data
    if min_max_dict_path != "None" and min_max_dict_path is not None:
        try:
            with open(min_max_dict_path, "r") as f:
                min_max_dict = json.load(f)
            min_dict = min_max_dict['min_dict']
            max_dict = min_max_dict['max_dict']
        except:
            train_df, min_dict, max_dict = UEID_preprocessing(train_df)
    else:
        train_df, min_dict, max_dict = UEID_preprocessing(train_df)

    data, _, _ = UEID_preprocessing(data, min_dict, max_dict)

    #for each match, run the simulation for simulation_time and only initialize the simulation for the first row with the data
    if match_id is not None:
        match_ids = [match_id]
    else:
        match_ids = [data['match_id'].unique()[0]]
    match_simulation_data = []
    simulation_period = 1 if simulation_time >= 45 else 2
    simulation_minute = 90 - simulation_time 
    # pdb.set_trace()
    for match_id in match_ids:
        match_data = data[data['match_id'] == match_id]
        team_list = match_data['team'].unique().tolist()
        #find the first row where the column 'Period' is equal to simulation_period and 'Minute' is equal to simulation_minute
        try:
            start_idx = match_data[(match_data['Period'] == simulation_period) & (match_data['Minute'] == simulation_minute)].index[0]
        except:
            start_idx = match_data[(match_data['Period'] == simulation_period) & (match_data['Minute'] == simulation_minute+1)].index[0]
        #append the first and the following seq_len-1 rows to the match_simulation_data
        match_simulation_data.append(match_data.iloc[start_idx:start_idx+seq_len+1])

    #convert the match_simulation_data to a dataframe
    match_simulation_data = pd.concat(match_simulation_data)
    # Create the data loader
    data_loader = UEID_Simulation_loader(match_simulation_data, seq_len, features)
    data_loader = DataLoader(data_loader, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False)

    # Load the model
    model = load_model(model_name, model_path, model_config)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.to(device)
    model.eval()

    # Run inference
    with torch.no_grad():
        simulation = {}
        for i, (input_seq, end_idx) in tqdm(enumerate(data_loader)):
            #convert the end_idx to a list
            end_idx = end_idx.cpu().numpy().tolist() if device != "cpu" else end_idx.numpy().tolist()
            for idx in end_idx:
                simulation[idx] = []

            input_seq = input_seq.to(device)
            iteration = 1
            time = [0]*len(end_idx)
            # Run the simulation
            while end_idx:
                # Forward pass through the model
                output = model(input_seq)

                # Convert the first config['num_actions'] to action probabilities
                action_probs = torch.nn.functional.softmax(output[:, :config['num_actions']], dim=1)

                # Select actions based on the probabilities
                if random_selection:
                    action = torch.multinomial(action_probs, 1)
                    action_prob = torch.gather(action_probs, 1, action)
                else:
                    action = torch.argmax(action_probs, dim=1)
                    action = action.unsqueeze(1)
                    action_prob = torch.gather(action_probs, 1, action)
                    
                # Recombine the action with the delta_T, start_x, start_y
                output = torch.cat((action.float(), action_prob.float(), output[:, config['num_actions']:]), dim=1)

                # Update the simulation dictionary
                mask = torch.ones(len(end_idx), dtype=bool)  # Initialize a mask to keep all entries

                for j, idx in enumerate(end_idx):
                    output_j = output[j].cpu().numpy().tolist() if device != "cpu" else output[j].numpy().tolist()
                    action_j = int(output_j[0])
                    action_prob_j = output_j[1]
                    delta_T_j = np.exp(output_j[2]*(max_dict["delta_T"]-min_dict["delta_T"]) + min_dict["delta_T"]) - 1e-6
                    start_x_j = output_j[3]*(max_dict["start_x"]-min_dict["start_x"]) + min_dict["start_x"]
                    start_y_j = output_j[4]*(max_dict["start_y"]-min_dict["start_y"]) + min_dict["start_y"]
                    home_score_j = input_seq[j, -1,config['features'].index('home_score')] if 'home_score' in config['features'] else None
                    away_score_j = input_seq[j, -1, config['features'].index('away_score')] if 'away_score' in config['features'] else None
                    team_j = input_seq[j, -1, config['features'].index('team')] if 'team' in config['features'] else None
                    action_last_j = input_seq[j, -1, 0]
                    if team_j is not None:
                        if action_last_j == 3:
                            temp_team_j = team_list.copy()
                            temp_team_j.remove(team_j)
                            team_j = temp_team_j[0]
                        else:
                            team_j = team_j.cpu().numpy() if device != "cpu" else team_j.numpy()
                            team_j = int(team_j)
                    if home_score_j is not None: #detached the tensor
                        home_score_j = home_score_j.cpu().numpy() if device != "cpu" else home_score_j.numpy()
                        home_score_j = int(home_score_j)
                        #unnormalize the home_score and away_score
                        if max_dict['home_score'] - min_dict['home_score'] != 0:
                            home_score_j = home_score_j*(max_dict["home_score"]-min_dict["home_score"]) + min_dict["home_score"]
                    if away_score_j is not None: #detached the tensor
                        away_score_j = away_score_j.cpu().numpy() if device != "cpu" else away_score_j.numpy()
                        away_score_j = int(away_score_j)
                        #unnormalize the home_score and away_score
                        if max_dict['home_score'] - min_dict['home_score'] != 0:
                            away_score_j = away_score_j*(max_dict["away_score"]-min_dict["away_score"]) + min_dict["away_score"]
                    #check if idx is a key in the simulation dictionary
                    if idx not in simulation.keys():
                        simulation[idx] = []
                    simulation[idx].append([action_j, action_prob_j, delta_T_j, start_x_j, start_y_j]+[team_j,home_score_j, away_score_j])
                    time[j] += delta_T_j
                    print(f"Time: {time[j]}") if config['test'] and __name__ == "__main__" else None 
                    # # Check if the current action indicates termination
                    # if output[j, 0] == 3:
                    if time[j] >= 10 and config['test']:
                        mask[j] = False

                    if time[j] >= simulation_time*60:
                        mask[j] = False  # Mark as False to remove this index

                #drop the action_prob
                output =torch.cat((output[:, 0].unsqueeze(1), output[:, 2:]), dim=1)

                # Apply the mask to remove finished sequences
                if not mask.any():
                    break  # Exit the loop if all sequences are finished
                
                if iteration >= max_iter:
                    output_i[0]=3
                iteration += 1
                print(f"Iteration: {iteration}") if config['test'] and __name__ == "__main__" else None

                # # Update end_idx and tensors based on the mask
                # end_idx = [end_idx[j] for j in range(len(end_idx)) if mask[j]]
                # input_seq = input_seq[mask]
                # output = output[mask]
                
                # Create the required input sequence for the next step
                if len(config['other_features'])>0:
                    # other_features = ['team','home_team','success','seconds','deltaX','deltaY','distance','dist2goal','angle2goal']
                    #get the previous timestep
                    prev_timestep = input_seq[:,-1,:]
                    #create a tensor with the same shape as the prev_timestep
                    append_tensor = torch.zeros((prev_timestep.shape[0],len(config['features'])))
                    #convert the prev_timestep to a numpy array
                    prev_timestep = prev_timestep.cpu().numpy() if device != "cpu" else prev_timestep.numpy()
                    #get the previous timestep's action
                    prev_action = prev_timestep[:,0]
                    #create a position dict for the position of the config['other_features'] in config['features']
                    position_dict = {feature:config['features'].index(feature) for feature in config['other_features']}
                    #if action is 3, then update the 'team','home_team','success','seconds'
                    for i, output_i in enumerate(output):
                        if output_i[0] == 3:
                            end_idx[i] = end_idx[i] + 1
                            action = output_i[0]
                            delta_T = 0
                            x = output_i[2]
                            y = output_i[3]
                            team_idx= position_dict.get('team')
                            team = prev_timestep[i,team_idx] if team_idx is not None else None
                            home_team_idx = position_dict.get('home_team')
                            home_team = prev_timestep[i,home_team_idx] if home_team_idx is not None else None
                            success_idx = position_dict.get('success')
                            success = 0 if success_idx is not None else None
                            seconds_idx = position_dict.get('seconds')
                            if seconds_idx is not None:
                                seconds = prev_timestep[i,position_dict['seconds']]
                                seconds = seconds*(max_dict["seconds"]-min_dict["seconds"]) + min_dict["seconds"]
                                seconds = seconds + np.exp(prev_timestep[i,1]*(max_dict["delta_T"]-min_dict["delta_T"]) + min_dict["delta_T"]) - 1e-6
                            deltaX_idx = position_dict.get('deltaX')
                            deltaX = 0 if deltaX_idx is not None else None
                            deltaY_idx = position_dict.get('deltaY')
                            deltaY = 0 if deltaY_idx is not None else None
                            distance_idx = position_dict.get('distance')
                            distance = 0 if distance_idx is not None else None
                            dist2goal_idx = position_dict.get('dist2goal')
                            dist2goal = 0 if dist2goal_idx is not None else None
                            angle2goal_idx = position_dict.get('angle2goal')
                            angle2goal = 0.5 if angle2goal_idx is not None else None
                            home_score_idx = position_dict.get('home_score')
                            home_score = prev_timestep[i,position_dict['home_score']] if home_score_idx is not None else None
                            away_score_idx = position_dict.get('away_score')
                            away_score = prev_timestep[i,position_dict['away_score']] if away_score_idx is not None else None                            
                        else:
                            action = output_i[0]
                            delta_T = output_i[1]
                            x = output_i[2]
                            y = output_i[3]
                            team_idx= position_dict.get('team')
                            if team_idx is not None:
                                previous_team = prev_timestep[i,team_idx]
                                match_id = data.loc[end_idx[i]]['match_id']
                                other_team = team_list.copy()
                                try:
                                    other_team.remove(previous_team)
                                except:
                                    pdb.set_trace()
                                other_team = other_team[0]
                                team = previous_team if prev_action[i] !=3 else other_team
                                if not isinstance(team, int):
                                    team = team.astype(int)
                            home_team_idx = position_dict.get('home_team')
                            if home_team_idx is not None:
                                previous_home_team = prev_timestep[i,home_team_idx]
                                home_team = previous_home_team if prev_action[i] !=3 else abs(1-previous_home_team)
                                if not isinstance(home_team, int):
                                    home_team = home_team.astype(int)
                            success_idx = position_dict.get('success')
                            if success_idx is not None:
                                success = 1 if action not in [3,6] else 0
                                if not isinstance(success, int):
                                    success = success.astype(int)
                            seconds_idx = position_dict.get('seconds')
                            if seconds_idx is not None:
                                seconds = prev_timestep[i,position_dict['seconds']]
                                seconds = seconds*(max_dict["seconds"]-min_dict["seconds"]) + min_dict["seconds"]
                                seconds = seconds + np.exp(prev_timestep[i,1]*(max_dict["delta_T"]-min_dict["delta_T"]) + min_dict["delta_T"]) - 1e-6
                                if not isinstance(seconds, float):
                                    seconds = seconds.astype(float)
                            deltaX_idx = position_dict.get('deltaX')
                            if deltaX_idx is not None:
                                deltaX = x*(max_dict["start_x"]-min_dict["start_x"]) + min_dict["start_x"]
                                deltaX = x - prev_timestep[i,2]*(max_dict["start_x"]-min_dict["start_x"]) + min_dict["start_x"]
                            deltaY_idx = position_dict.get('deltaY')
                            if deltaY_idx is not None:
                                deltaY = y*(max_dict["start_y"]-min_dict["start_y"]) + min_dict["start_y"]
                                deltaY = y - prev_timestep[i,3]*(max_dict["start_y"]-min_dict["start_y"]) + min_dict["start_y"]
                            distance_idx = position_dict.get('distance')
                            if distance_idx is not None:
                                distance = torch.sqrt((deltaX)**2 + (deltaY)**2)
                            dist2goal_idx = position_dict.get('dist2goal')
                            if dist2goal_idx is not None:
                                dist2goal = torch.sqrt((x*(max_dict["start_x"]-min_dict["start_x"]) + min_dict["start_x"] - 105)**2 + (y*(max_dict["start_y"]-min_dict["start_y"]) + min_dict["start_y"] - 34)**2)
                            angle2goal_idx = position_dict.get('angle2goal')
                            if angle2goal_idx is not None:
                                angle2goal = torch.abs(torch.atan2((y*(max_dict["start_y"]-min_dict["start_y"]) + min_dict["start_y"] - 34), (x*(max_dict["start_x"]-min_dict["start_x"]) + min_dict["start_x"] - 105)))
                            home_score_idx = position_dict.get('home_score')
                            if home_score_idx is not None:
                                if action == 8 and home_team == 1:
                                    home_score = prev_timestep[i,position_dict['home_score']]+1
                                else:
                                    home_score = prev_timestep[i,position_dict['home_score']]
                            away_score_idx = position_dict.get('away_score')
                            if away_score_idx is not None:
                                if action == 8 and home_team == 0:
                                    away_score = prev_timestep[i,position_dict['away_score']]+1
                                else:
                                    away_score = prev_timestep[i,position_dict['away_score']]                      
                        #preprocess the features
                        for feature in config['other_features']:
                            if feature == 'seconds':
                                seconds = (seconds - min_dict['seconds'])/(max_dict['seconds'] - min_dict['seconds'])
                            elif feature == 'deltaX':
                                deltaX = (deltaX - min_dict['deltaX'])/(max_dict['deltaX'] - min_dict['deltaX'])
                            elif feature == 'deltaY':
                                deltaY = (deltaY - min_dict['deltaY'])/(max_dict['deltaY'] - min_dict['deltaY'])
                            elif feature == 'distance':
                                distance = (distance - min_dict['distance'])/(max_dict['distance'] - min_dict['distance'])
                            elif feature == 'dist2goal':
                                dist2goal = (dist2goal - min_dict['dist2goal'])/(max_dict['dist2goal'] - min_dict['dist2goal'])
                            elif feature == 'angle2goal':
                                angle2goal = (angle2goal - min_dict['angle2goal'])/(max_dict['angle2goal'] - min_dict['angle2goal'])
                            elif feature == 'home_score':
                                if max_dict['home_score'] - min_dict['home_score'] != 0:
                                    home_score = (home_score - min_dict['home_score'])/(max_dict['home_score'] - min_dict['home_score'])
                                else:
                                    home_score = 0
                            elif feature == 'away_score':
                                if max_dict['away_score'] - min_dict['away_score'] != 0:
                                    away_score = (away_score - min_dict['away_score'])/(max_dict['away_score'] - min_dict['away_score'])
                                else:
                                    away_score = 0                                
                        #update the append_tensor
                        append_tensor[i,0] = action
                        append_tensor[i,1] = delta_T
                        append_tensor[i,2] = x
                        append_tensor[i,3] = y
                        if team_idx is not None:
                            append_tensor[i,team_idx] = int(team)
                        if home_team_idx is not None:
                            append_tensor[i,home_team_idx] = home_team.astype(int)
                        if success_idx is not None: #TODO: adjust this if success is added as a target feature
                            append_tensor[i,success_idx] = success
                        if seconds_idx is not None:
                            append_tensor[i,seconds_idx] = seconds.astype(float)
                        if deltaX_idx is not None:
                            append_tensor[i,deltaX_idx] = deltaX
                        if deltaY_idx is not None:
                            append_tensor[i,deltaY_idx] = deltaY
                        if distance_idx is not None:
                            append_tensor[i,distance_idx] = distance
                        if dist2goal_idx is not None:
                            append_tensor[i,dist2goal_idx] = dist2goal
                        if angle2goal_idx is not None:
                            append_tensor[i,angle2goal_idx] = angle2goal
                        if home_score_idx is not None:
                            append_tensor[i,home_score_idx] = home_score
                        if away_score_idx is not None:
                            append_tensor[i,away_score_idx] = away_score
                    append_tensor = append_tensor.to(device).unsqueeze(1)
                    input_seq = torch.cat((input_seq[:, 1:, :], append_tensor), dim=1)     
                else:
                    # Update the input sequence for the next step
                    input_seq = torch.cat((input_seq[:, 1:, :], output.unsqueeze(1)), dim=1)

        rows = []
        for key, value_list in simulation.items():
            for value in value_list:
                rows.append([key] + value)

        # Convert the list of rows into a DataFrame
        columns = ['index', 'action', 'action_prob', 'delta_T', 'x', 'y']
        columns += ['team','home_score', 'away_score'] if 'home_score' in config['features'] and 'away_score' in config['features'] else []
        df = pd.DataFrame(rows, columns=columns)
        #set to 4 dp
        df = df.round(4)

        # Ensure for each index the last action is action=3
        for idx in df['index'].unique():
            # Get the last row for the given index
            last_row_idx = df[df['index'] == idx].index[-1]
            
            # Check if the last action is not 3
            if df.loc[last_row_idx, 'action'] != 3:
                # Change the last row's action to 3
                df.loc[last_row_idx, 'action'] = 3

    return df

def load_model(model_name:str,model_path:str,model_config:str):
    if model_name == "NMSTPP":
        from ..models.NMSTPP import NMSTPP
        with open(model_config, "r") as f:
            config = json.load(f)
        action_embedding_input_len = config["action_embedding_input_len"]
        action_embedding_out_len = config["action_embedding_out_len"]
        scale_grad_by_freq = config["scale_grad_by_freq"]
        continuous_embedding_input_len = config["continuous_embedding_input_len"]
        continuous_embedding_output_len = config["continuous_embedding_output_len"]
        multihead_attention = config["multihead_attention"]
        hidden_dim = config["hidden_dim"]
        feature_embedding_output_len = config["feature_embedding_output_len"]
        NN_deltaT_num_layers = config["NN_deltaT_num_layers"]
        NN_location_num_layers = config["NN_location_num_layers"]
        NN_action_num_layers = config["NN_action_num_layers"]
        deltaT_output_len = config["deltaT_output_len"]
        location_output_len = config["location_output_len"]
        action_output_len = config["action_output_len"]
        model = NMSTPP(action_embedding_input_len, action_embedding_out_len, scale_grad_by_freq, 
                 continuous_embedding_input_len, continuous_embedding_output_len,
                 multihead_attention, hidden_dim,feature_embedding_output_len,
                 NN_deltaT_num_layers, NN_location_num_layers, NN_action_num_layers,
                 deltaT_output_len, location_output_len, action_output_len)
        model.load_state_dict(torch.load(model_path, weights_only=True))
        return model
    elif model_name == "Seq2Event":
        from ..models.Seq2Event import Seq2Event
        with open(model_config, "r") as f:
            config = json.load(f)
        action_embedding_input_len = config["action_embedding_input_len"]
        action_embedding_out_len = config["action_embedding_out_len"]
        scale_grad_by_freq = config["scale_grad_by_freq"]
        continuous_embedding_input_len = config["continuous_embedding_input_len"]
        continuous_embedding_output_len = config["continuous_embedding_output_len"]
        multihead_attention = config["multihead_attention"]
        hidden_dim = config["hidden_dim"]
        transformer_num_layers = config["transformer_num_layers"]
        transformer_finaldenselayer_dim = config["transformer_finaldenselayer_dim"]
        deltaT_output_len = config["deltaT_output_len"]
        location_output_len = config["location_output_len"]
        action_output_len = config["action_output_len"]
        model = Seq2Event(action_embedding_input_len, action_embedding_out_len, scale_grad_by_freq, 
                 continuous_embedding_input_len, continuous_embedding_output_len,
                 multihead_attention, hidden_dim,transformer_num_layers, transformer_finaldenselayer_dim,
                 deltaT_output_len, location_output_len, action_output_len)
        model.load_state_dict(torch.load(model_path, weights_only=True))
        return model
    else:
        raise ValueError(f"Model {model_name} not supported")

def simulation_evaluation(simulation_df, ground_truth_df):
    #check if simulation_df and ground_truth_df are paths or dataframes
    if not isinstance(simulation_df, pd.DataFrame):
        #check if the simulation_df is a path
        if isinstance(simulation_df, str):
            simulation_df = pd.read_csv(simulation_df)
        else:
            raise ValueError("simulation_df must be a pandas dataframe or a path to a csv file")
    if not isinstance(ground_truth_df, pd.DataFrame):
        #check if the ground_truth_df is a path
        if isinstance(ground_truth_df, str):
            ground_truth_df = pd.read_csv(ground_truth_df)
        else:
            raise ValueError("ground_truth_df must be a pandas dataframe or a path to a csv file")
    
    #drop the game_over and period_over rows
    ground_truth_df = ground_truth_df[ground_truth_df["action"] != "game_over"]
    ground_truth_df = ground_truth_df[ground_truth_df["action"] != "period_over"]

    #get the unique indexes
    unique_indexes = simulation_df['index'].unique()
    #evaluating per timestep
    error_row = []
    timestep_eval = {}
    es_hota_list= []
    action_dict = {'short_pass': 0, 'carry': 1, 'high_pass': 2, '_': 3, 'cross': 4, 
                   'long_pass': 5, 'shot': 6, 'dribble': 7}
    #map the action to the action_dict
    ground_truth_df['action'] = ground_truth_df['action'].map(action_dict)
    ground_truth_df.loc[(ground_truth_df["action"] == 6) & (ground_truth_df["goal"] == 1), "action"] = 8

    for idx in tqdm(unique_indexes):
        try:
            sim_poss = simulation_df[simulation_df['index'] == idx]
            #get the ground truth (the index and the following action with the same poss_id)
            ground_truth = ground_truth_df.loc[idx]
            poss_id = ground_truth['poss_id']
            match_id = ground_truth['match_id']
            ground_truth_possession = ground_truth_df[(ground_truth_df['poss_id'] == poss_id) & (ground_truth_df['match_id'] == match_id)]
            #drop the row with index less than the indx
            ground_truth_possession = ground_truth_possession[ground_truth_possession.index >= idx]
            
            #evaluate the simulation per timestep
            for step_i, (_, row) in enumerate(sim_poss.iterrows()):
                #check if the simulation timestep exceed the ground truth timestep
                if step_i >= len(ground_truth_possession):
                    #next idx
                    break
                #initialize the row_eval if the timestep is not in the row_eval
                if step_i not in timestep_eval:
                    timestep_eval[step_i] = {"ACC_action":[],"MAE_delta_T":[],"MAE_start_x":[],"MAE_start_y":[]} 
                #get the ground truth for the current timestep
                ACC_action = 1 if int(row['action']) == int(ground_truth_possession.iloc[step_i]['action']) else 0
                MAE_delta_T = abs(row['delta_T'] - ground_truth_possession.iloc[step_i]['delta_T'])
                MAE_start_x = abs(row['x'] - ground_truth_possession.iloc[step_i]['start_x'])
                MAE_start_y = abs(row['y'] - ground_truth_possession.iloc[step_i]['start_y'])
                timestep_eval[step_i]["ACC_action"].append(ACC_action)
                timestep_eval[step_i]["MAE_delta_T"].append(MAE_delta_T)
                timestep_eval[step_i]["MAE_start_x"].append(MAE_start_x)
                timestep_eval[step_i]["MAE_start_y"].append(MAE_start_y)
                    
            #ES-HOTA
            es_hota_list.append([idx]+ES_HOTA_cal(sim_poss,ground_truth_possession))

        except:
            error_row.append([idx])
            continue

    #convert the timestep_eval to a dataframe with key as the column time_step and the loss as other columns
    time_step_data = []
    for time_step, metrics in timestep_eval.items():
        mean_entry = {
            'time_step': time_step,
            'count': len(metrics['ACC_action']),
            'ACC_action': sum(metrics['ACC_action']) / len(metrics['ACC_action']),
            'MAE_delta_T': sum(metrics['MAE_delta_T']) / len(metrics['MAE_delta_T']),
            'MAE_start_x': sum(metrics['MAE_start_x']) / len(metrics['MAE_start_x']),
            'MAE_start_y': sum(metrics['MAE_start_y']) / len(metrics['MAE_start_y'])
        }
        time_step_data.append(mean_entry)

    # Create DataFrame
    timestep_eval_df = pd.DataFrame(time_step_data)
    #add a row for the average of the timestep_eval
    timestep_eval_df.loc[len(timestep_eval_df)] = ['Overall']+ timestep_eval_df.iloc[:,1:].mean().tolist()

    #round the values to 4 decimal places
    timestep_eval_df = timestep_eval_df.round(4)
    es_hota_df = pd.DataFrame(es_hota_list, columns=['idx','TP','FN','FP']+[f"ES_HOTA_{i:.2f}" for i in np.arange(0.05,1,0.05)]+['ES_HOTA'])
    #add a row for the average of the ES_HOTA
    es_hota_df.loc[len(es_hota_df)] = ['Overall']+ es_hota_df.iloc[:,1:4].sum().tolist()+ es_hota_df.iloc[:,4:-1].mean().tolist() + [es_hota_df.iloc[:,-1].mean()]
    es_hota_df = es_hota_df.round(4)

    #print the error rows
    if error_row:
        print(f"Error idx: {error_row}")
    #print the es_hota mean value and es_hota 0.5 value
    print('Results for the Simulation Evaluation:')
    print(f"ES-HOTA: {es_hota_df.iloc[-1,-1]:.4f} | ES-HOTA_0.5: {es_hota_df.iloc[-1,14]:.4f} | TP: {es_hota_df.iloc[-1,1]} | FN: {es_hota_df.iloc[-1,2]} | FP: {es_hota_df.iloc[-1,3]}")

    return timestep_eval_df, es_hota_df

def ES_HOTA_cal(sim_poss,ground_truth_possession,tau_t=5,tau_l=5):
    sim_list = []
    ES_HOTA_list = []
    for i in range(max(len(sim_poss),len(ground_truth_possession))):
        if i < len(ground_truth_possession) and i < len(sim_poss):
            IT_sim = np.exp(np.log(0.05)*(sim_poss.iloc[i]['delta_T']-ground_truth_possession.iloc[i]['delta_T'])**2/tau_t**2)
            dist = (sim_poss.iloc[i]['x']-ground_truth_possession.iloc[i]['start_x'])**2 + (sim_poss.iloc[i]['y']-ground_truth_possession.iloc[i]['start_y'])**2
            LOC_sim = np.exp(np.log(0.05)*dist/tau_l**2)
            ACT_sim = sim_poss.iloc[i]['action_prob'] if int(sim_poss.iloc[i]['action']) == int(ground_truth_possession.iloc[i]['action']) else 0
            sim_score=IT_sim*LOC_sim*ACT_sim
            sim_list.append(sim_score)
        else:
            sim_list.append(0)

    for sim_i in np.arange(0.05,1,0.05):
        #convert the sim_list to binary values where the value is 1 if the value is greater than sim_i
        sim_list_binary = [1 if sim >= sim_i else 0 for sim in sim_list]
        len_sim = len(sim_list)
        len_gt = len(ground_truth_possession)
        TP_count = sum(sim_list_binary)
        FP_count = len_sim - len_gt if len_sim > len_gt else 0
        #sum the gt_list_binary for 0 values
        FN_count = len_gt - TP_count

        A_count = 0
        for j in range(TP_count):
            # count sim_list_binary[:TP_count+1] for the element equal to sim_list_binary(j)
            TPA_count = TP_count 
            FNA_count = len_gt - TP_count
            FPA_count = len_sim - len_gt if len_sim > len_gt else 0
            A_count += TPA_count/(TPA_count+FNA_count+FPA_count)
        
        ES_HOTA_list.append(np.sqrt(A_count/(TP_count+FN_count+FP_count)))
    
    #get the average of the ES_HOTA_list
    ES_HOTA_value = sum(ES_HOTA_list)/len(ES_HOTA_list)
    ES_HOTA_list.append(ES_HOTA_value)

    #add the TP, FN, FP at the beginning of the list
    ES_HOTA_list = [TP_count,FN_count,FP_count] + ES_HOTA_list

    return ES_HOTA_list

if __name__ == "__main__":
    model_path_nmstpp = os.getcwd()+"/test/model/NMSTPP/out/optuna/20241009_011920/run_1/_model_1.pth"
    model_config_nmstpp = os.getcwd()+"/test/model/NMSTPP/out/optuna/20241009_011920/run_1/hyperparameters.json"
    min_max_dict_path = os.getcwd()+"/test/model/NMSTPP/out/optuna/20241009_011920/min_max_dict.json"
    save_path_nmstpp = os.getcwd()+"/test/inference/nmstpp/"

    model_path_seq2event = os.getcwd()+"/test/model/Seq2Event/out/optuna/20241009_011944/run_1/_model_1.pth"
    model_config_seq2event = os.getcwd()+"/test/model/Seq2Event/out/optuna/20241009_011944/run_1/hyperparameters.json"
    save_path_seq2event = os.getcwd()+"/test/inference/seq2event/"

    train_path = os.getcwd()+"/test/dataset/csv/train.csv"
    valid_path = os.getcwd()+"/test/dataset/csv/valid.csv"
    
    os.makedirs(save_path_seq2event, exist_ok=True)
    os.makedirs(save_path_nmstpp, exist_ok=True)

    # testing inference for NMSTPP
    # inferenced_data,loss_df = UEID_inference(train_path, valid_path, "NMSTPP", model_path_nmstpp, model_config_nmstpp)
    # inferenced_data,loss_df = UEID_inference(None, valid_path, "NMSTPP", model_path_nmstpp, model_config_nmstpp, min_max_dict_path=min_max_dict_path)
    # inferenced_data.to_csv(save_path_nmstpp+"inference.csv",index=False)
    # loss_df.to_csv(save_path_nmstpp+"loss.csv",index=False)
    # df = UEID_simulation_possession(train_path, valid_path, "NMSTPP", model_path_nmstpp, model_config_nmstpp, random_selection=True, max_iter=20)
    # df.to_csv(save_path_nmstpp+"simulation.csv",index=False)
    df_90 = UEID_simulation_match(train_path, valid_path, "NMSTPP", model_path_nmstpp, model_config_nmstpp, random_selection=True, max_iter=20)
    df_90.to_csv(save_path_nmstpp+"simulation_90.csv",index=False)
    
    

    # # testing evaluation
    # simulation_df_path = os.getcwd()+"/test/inference/nmstpp/simulation.csv"
    # ground_truth_df_path = os.getcwd()+"/test/dataset/csv/valid.csv"
    # timestep_eval_df,es_hota_df = simulation_evaluation(simulation_df_path, ground_truth_df_path)
    # timestep_eval_df.to_csv(save_path_nmstpp+"timestep_eval.csv",index=False)
    # es_hota_df.to_csv(save_path_nmstpp+"ES_HOTA.csv",index=False)


    #testing inference for Seq2Event
    # inferenced_data,loss_df = UEID_inference(train_path, valid_path, "Seq2Event", model_path_seq2event, model_config_seq2event)
    # inferenced_data,loss_df = UEID_inference(None, valid_path, "Seq2Event", model_path_seq2event, model_config_seq2event, min_max_dict_path=min_max_dict_path)
    # inferenced_data.to_csv(save_path_seq2event+"inference.csv",index=False)
    # loss_df.to_csv(save_path_seq2event+"loss.csv",index=False)
    # df = UEID_simulation_possession(train_path, valid_path, "Seq2Event", model_path_seq2event, model_config_seq2event, random_selection=True, max_iter=20)
    # df.to_csv(save_path_seq2event+"simulation.csv",index=False)

    # #testing evaluation
    # simulation_df_path = os.getcwd()+"/test/inference/seq2event/simulation.csv"
    # ground_truth_df_path = os.getcwd()+"/test/dataset/csv/valid.csv"
    # timestep_eval_df,es_hota_df = simulation_evaluation(simulation_df_path, ground_truth_df_path)
    # timestep_eval_df.to_csv(save_path_seq2event+"timestep_eval.csv",index=False)
    # es_hota_df.to_csv(save_path_seq2event+"ES_HOTA.csv",index=False)
    # pdb.set_trace()
    print('___________done______________')