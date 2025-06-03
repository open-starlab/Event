import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pdb

def calculate_HPUS(data_raw, shot_num=[6,8], cross_num=[4], num_actions=9):
   # Check if the input is a DataFrame or CSV file path
   if not isinstance(data_raw, pd.DataFrame):
      data = pd.read_csv(data_raw)
   else:
      data = data_raw.copy()
       

   # Map actions to integers and handle goals (action = 8 for goal)
   action_dict = {'short_pass': 0, 'carry': 1, 'high_pass': 2, '_': 3, 'cross': 4, 
                  'long_pass': 5, 'shot': 6, 'dribble': 7}
   data["action"] = data["action"].map(action_dict)
   data.loc[(data["action"] == 6) & (data["goal"] == 1), "action"] = 8

   # Remove rows with 'period_over' (action = 3)
   data = data[data["action"] != 3]

   # Fill NaN values with 0 for all columns
   data.fillna(0, inplace=True)

   # Define zone score based on `start_x_pred_unscaled`
   conditions = [
      (data["start_x_pred_unscaled"] > 83 * 1.05),
      (data["start_x_pred_unscaled"] < 50 * 1.05),
      ((50 * 1.05 <= data["start_x_pred_unscaled"]) & (data["start_x_pred_unscaled"] <= 83 * 1.05))
   ]
   zone_scores = [10, 0, 5]
   data["zone_score"] = np.select(conditions, zone_scores, default=0)

   # Normalize action probabilities if needed
   action_columns = [f'action_{i}_prob' for i in range(num_actions)]
   row_sums = data[action_columns].sum(axis=1)
   if not np.allclose(row_sums, 1):
      exp_values = np.exp(data[action_columns])
      data[action_columns] = exp_values.div(exp_values.sum(axis=1), axis=0)

   # Calculate action score based on weighted action probabilities
   weights = [5, 5, 5, 0, 10, 5, 10, 5, 10]  # Weights corresponding to each action
   data['action_score'] = sum(weights[i] * data[f'action_{i}_prob'] for i in range(num_actions))

   # Compute HPUS_t and HPUS
   data['HPUS_t'] = data['HPUS_t'] = np.clip(data['delta_T_pred_unscaled'], 1, None)
   data['HPUS'] = np.sqrt(data['action_score'] * data['zone_score']) / data['HPUS_t']

   # Compute attack_flag
   relevant_actions = set(shot_num + cross_num)
   data['attack_flag'] = data['action'].isin(relevant_actions).astype(int)

   # Group by match_id and poss_id to compute HPUS values
   HPUS_value = data.groupby(['match_id', 'poss_id']).apply(
    lambda group: pd.Series({
        'HPUS': np.sum(group['HPUS'] * np.exp(-0.3 * (len(group) - np.arange(len(group))))),
        'team': group['team'].iloc[0],
        'seconds': group['seconds'].iloc[-1],
        'attack_flag': int(group['attack_flag'].sum() > 0)
    })
   ).reset_index()

   return HPUS_value

def plot_HPUS(data_raw, hpus_data, save_path, match_id=None, plus=False, time_period=5, swap_home_away=False):
    # Check if the input is a DataFrame or CSV file path
    if not isinstance(data_raw, pd.DataFrame):
        data = pd.read_csv(data_raw)
    else:
        data = data_raw.copy()
    
    if match_id is None:
        match_id = hpus_data['match_id'].unique()[0]

    data = data[data['match_id'] == match_id]
    team = data.team.unique().tolist()
    team_1 = data.iloc[0]['team']
    team_1_home = data.iloc[0]['home_team']
    team.remove(team_1)
    team_dict = {team_1: team_1_home, team[0]: abs(team_1_home - 1)}
    #fill nan values with 0
    data.fillna(0, inplace=True)

    # Filter the data for the given match_id
    match_data = hpus_data[hpus_data['match_id'] == match_id]

    if plus:
        # Filter the data for attack_flag = 1
        match_data = match_data[match_data['attack_flag'] == 1]
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Iterate over each unique team in the match
    for team in match_data['team'].unique():
        # Filter data for the specific team
        team_data = match_data[match_data['team'] == team].copy()

        # Create 5-minute bins for the `seconds` column (intervals: 0-5, 5-10, ..., up to 90)
        bins = list(range(0, 90 * 60 + 1, time_period * 60))
        time_bins_labels = bins[:-1]
        team_data['time_bin'] = pd.cut(team_data['seconds'], bins=bins, labels=bins[:-1]).to_list()

        # Aggregate the HPUS values per 5-minute interval
        y = team_data.groupby('time_bin')['HPUS'].sum()

        # Check for NaN values and replace them with 0 if necessary
        y = y.reindex(time_bins_labels, fill_value=0)
        y = y.shift(1).fillna(0)

        # Prepare x values as time in minutes
        x = [bin_val / 60 for bin_val in y.index.astype(int)]

        # Plot for each team
        home_away = 'Home' if team_dict[team] == 1 else 'Away'
        color = '#ff7f0e' if team_dict[team] == 1 else '#1f77b4'
        if swap_home_away:
            home_away = 'Away' if home_away == 'Home' else 'Home'
            color = '#1f77b4' if team_dict[team] == 1 else '#ff7f0e'
        plt.plot(x, y, marker='o', linestyle='-', label=f'Team {team} ({home_away})', color=color)

        #plot the line where the score changes
        if home_away == 'Home':
            # data['value_change'] = data['home_score']!=data['home_score'].shift()
            data['value_change'] = (data['home_score'] != data['home_score'].shift()) & (data['home_score'].shift().notna())
        elif home_away == 'Away':
            data['value_change'] = (data['away_score'] != data['away_score'].shift()) & (data['away_score'].shift().notna())
        changed = data[data['value_change']==True]
        for i in range(len(changed)):
            plt.axvline(x=changed.iloc[i].Minute, color=color, linestyle='--', linewidth=1)


    # Add titles and labels
    if not plus:
        plt.title(f'HPUS Plot for Match {match_id}', fontsize=18) 
        plt.ylabel('HPUS', fontsize=18) 
    else:
        plt.title(f'HPUS+ Plot for Match {match_id}', fontsize=18) 
        plt.ylabel('HPUS+', fontsize=18) 

    plt.xlabel('Match Time (minutes)', fontsize=18)

    # Increase font size for tick labels
    plt.tick_params(axis='both', labelsize=16)

    # Add grid and legend
    plt.grid(True)
    plt.legend(fontsize=16)

    # Save the plot
    if not plus:
        plt.savefig(f"{save_path}/HPUS.png")
    else:
        plt.savefig(f"{save_path}/HPUS_plus.png")
    
    plt.close()  # Close the figure to free memory



if __name__ == "__main__":
   import os
   inference_data = os.getcwd()+"/test/inference/nmstpp/inference.csv"
   save_path = os.getcwd()+"/test/application/"
   hpus=calculate_HPUS(inference_data)
   hpus.to_csv(save_path+"HPUS.csv", index=False)
   plot_HPUS(inference_data,hpus,save_path)
   plot_HPUS(inference_data,hpus,save_path,plus=True)
   print("HPUS and HPUS+ plots saved successfully.")
   pdb.set_trace()

    