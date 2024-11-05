import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pdb
import os
from ..inference.UEID_inference import UEID_simulation_match
from ..inference.LEM_inference import LEM_simulation_match
from ..inference.FMS_inference import FMS_simulation_match


def result_sim(valid_path,model_path,model_config,min_max_dict_path,match_id=None,model="NMSTPP",n_sim=2):

    if model in ["NMSTPP","Seq2Event"]:
        simulation_function = UEID_simulation_match
    elif model in ["LEM"]:
        simulation_function = LEM_simulation_match
    elif model in ["FMS"]:
        simulation_function = FMS_simulation_match
    else:
        raise ValueError("Model not supported")
    home_win_prob = []
    away_win_prob = []
    time_bin = []
    for simulation_time in range(0,90,5):
        time_bin.append(simulation_time) 
        temp_home_win_prob = []
        temp_away_win_prob = []
        for i in range(n_sim):
            try:
                df = simulation_function(None, valid_path, model, model_path, model_config,min_max_dict_path=min_max_dict_path, random_selection=True, max_iter=26,simulation_time=90-simulation_time,match_id=match_id)
                home_score = df["home_score"].iloc[-1]
                away_score = df["away_score"].iloc[-1]
            except:
                pdb.set_trace()
                print("Simulation failed for time bin: ",simulation_time)
                home_score = 0
                away_score = 0
            if home_score > away_score:
                temp_home_win_prob.append(1)
                temp_away_win_prob.append(0)
            elif home_score < away_score:
                temp_home_win_prob.append(0)
                temp_away_win_prob.append(1)
            else:
                temp_home_win_prob.append(0.5)
                temp_away_win_prob.append(0.5)
        home_win_prob.append(np.mean(temp_home_win_prob))
        away_win_prob.append(np.mean(temp_away_win_prob))
    return home_win_prob,away_win_prob,time_bin

def plot_result(home_win_prob,away_win_prob,time_bin,save_path=None):
    plt.plot(time_bin,home_win_prob,label="Home Win")
    plt.plot(time_bin,away_win_prob,label="Away Win")
    plt.xlabel("Simulation Time",fontsize=18)
    plt.ylabel("Probability",fontsize=18)
    plt.legend(fontsize=18)
    plt.savefig(save_path+"/result_sim.png")
    plt.close()
if __name__ == "__main__":
    model_path_nmstpp = os.getcwd()+"/test/model/NMSTPP/out/optuna/20241009_011920/run_1/_model_1.pth"
    model_config_nmstpp = os.getcwd()+"/test/model/NMSTPP/out/optuna/20241009_011920/run_1/hyperparameters.json"
    min_max_dict_path = os.getcwd()+"/test/model/NMSTPP/out/optuna/20241009_011920/min_max_dict.json"
    valid_path = os.getcwd()+"/test/dataset/csv/valid.csv"
    save_path = os.getcwd()+"/test/application/"
    home_win_prob,away_win_prob,time_bin = result_sim(valid_path,model_path_nmstpp,model_config_nmstpp,min_max_dict_path,model="NMSTPP",n_sim=2)
    plot_result(home_win_prob,away_win_prob,time_bin,save_path)
    pdb.set_trace()