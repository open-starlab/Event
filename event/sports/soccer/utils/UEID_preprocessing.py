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
    df["team_map"] = df["team"].copy().map(team_dict) 
    if df["team_map"].isnull().sum() == 0:
        df["team"] = df["team_map"]
    df.drop(columns=["team_map"], inplace=True)
    df["action"] = df["action"].map(action_dict)

    #if action is 6 and success is 1, then change action to 8
    df.loc[(df["action"] == 6) & (df["goal"] == 1), "action"] = 8

    # Initialize min and max dictionaries
    if min_dict_input is None or max_dict_input is None:
        min_dict = {}
        max_dict = {}
        
        # Calculate minimums and maximums for min-max normalization
        features_to_normalize = ["seconds", "start_x", "start_y", "deltaX", "deltaY", "distance", "dist2goal", "angle2goal",'home_score','away_score']
        for feature in features_to_normalize:
            min_dict[feature] = float(df[feature].min())
            max_dict[feature] = float(df[feature].max())
    else:
        min_dict = min_dict_input
        max_dict = max_dict_input

    

    # Apply min-max normalization
    for feature in ["seconds", "start_x", "start_y", "deltaX", "deltaY", "distance", "dist2goal", "angle2goal",'home_score','away_score']:
        if max_dict[feature] - min_dict[feature] != 0:
            df[feature] = (df[feature] - min_dict[feature]) / (max_dict[feature] - min_dict[feature])
        else:
            df[feature] = 0
        #ensure the values are between 0 and 1
        df[feature] = df[feature].apply(lambda x: 0 if x < 0 else 1 if x > 1 else x)


    # Apply logarithmic transformation and min-max normalization to delta_T
    df["delta_T"] = df["delta_T"].apply(lambda x: np.log(x + 1e-6))
    if min_dict_input is None and max_dict_input is None:
        min_dict["delta_T"] = df["delta_T"].min()
        max_dict["delta_T"] = df["delta_T"].max()
    else:
        min_dict["delta_T"] = min_dict_input["delta_T"]
        max_dict["delta_T"] = max_dict_input["delta_T"]
        
    df["delta_T"] = (df["delta_T"] - min_dict["delta_T"]) / (max_dict["delta_T"] - min_dict["delta_T"])
    #ensure the values are between 0 and 1
    df["delta_T"] = df["delta_T"].apply(lambda x: 0 if x < 0 else 1 if x > 1 else x)
    #reset index
    df.reset_index(drop=True, inplace=True)
    
    
    return df, min_dict, max_dict

