import pandas as pd
import numpy as np
import pdb

def UEID_preprocessing(df, min_dict_input=None, max_dict_input=None):
    # Drop the game_over and period_over rows:
    df = df[df["action"] != "game_over"]
    df = df[df["action"] != "period_over"]

    # Mapping for team and action columns
    team_dict = {'Granada': 0, 'Atlético Madrid': 1, 'Celta Vigo': 2, 'Osasuna': 3, 
                 'Valencia': 4, 'Sevilla': 5, 'Mallorca': 6, 'Las Palmas': 7, 'Cádiz': 8, 
                 'Deportivo Alavés': 9, 'Real Sociedad': 10, 'Girona': 11, 'Villarreal': 12, 
                 'Real Betis': 13, 'Barcelona': 14, 'Getafe': 15, 'Rayo Vallecano': 16, 
                 'Almería': 17, 'Athletic Club': 18, 'Real Madrid': 19}
    action_dict = {'short_pass': 0, 'carry': 1, 'high_pass': 2, '_': 3, 'cross': 4, 
                   'long_pass': 5, 'shot': 6, 'dribble': 7}

    # Map the team and action columns to integers
    df["team"] = df["team"].map(team_dict)
    df["action"] = df["action"].map(action_dict)

    #if action is 6 and success is 1, then change action to 8
    df.loc[(df["action"] == 6) & (df["success"] == 1), "action"] = 8

    # Initialize min and max dictionaries
    if min_dict_input is None or max_dict_input is None:
        min_dict = {}
        max_dict = {}
        
        # Calculate minimums and maximums for min-max normalization
        features_to_normalize = ["seconds", "start_x", "start_y", "deltaX", "deltaY", "distance", "dist2goal", "angle2goal"]
        for feature in features_to_normalize:
            min_dict[feature] = df[feature].min()
            max_dict[feature] = df[feature].max()
    else:
        min_dict = min_dict_input
        max_dict = max_dict_input

    # Apply min-max normalization
    for feature in ["seconds", "start_x", "start_y", "deltaX", "deltaY", "distance", "dist2goal", "angle2goal"]:
        df[feature] = (df[feature] - min_dict[feature]) / (max_dict[feature] - min_dict[feature])

    # Apply logarithmic transformation and min-max normalization to delta_T
    df["delta_T"] = df["delta_T"].apply(lambda x: np.log(x + 1e-6))
    if min_dict_input is None and max_dict_input is None:
        min_dict["delta_T"] = df["delta_T"].min()
        max_dict["delta_T"] = df["delta_T"].max()
    else:
        min_dict["delta_T"] = min_dict_input["delta_T"]
        max_dict["delta_T"] = max_dict_input["delta_T"]
        
    df["delta_T"] = (df["delta_T"] - min_dict["delta_T"]) / (max_dict["delta_T"] - min_dict["delta_T"])
    
    return df, min_dict, max_dict

