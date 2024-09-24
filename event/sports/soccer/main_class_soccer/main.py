import yaml
import json
import os

from ..models.FMS import main as FMS_main
from ..models.FMS import main_optuna  as FMS_main_optuna 
from ..models.LEM_action import main as LEM_action_main
from ..models.LEM_action import main_optuna as LEM_action_main_optuna
from ..models.LEM import main as LEM_main
from ..models.LEM import main_optuna as LEM_main_optuna
from ..models.MAJ import main as MAJ_main
from ..models.NMSTPP import main as NMSTPP_main
from ..models.NMSTPP import main_optuna as NMSTPP_main_optuna
from ..models.Seq2Event import main as Seq2Event_main
from ..models.Seq2Event import main_optuna as Seq2Event_main_optuna

from ..inference import FMS_inference as FMS_inference
from ..inference import LEM_inference as LEM_inference
from ..inference import UEID_inference as UEID_inference

import pdb

class event_model_soccer:
    def __init__(self, model_name, config=None):
        self.model_name = model_name   
        self.config = config
        with open(self.config, 'r') as file:
            config = yaml.safe_load(file)
        self.config = config
        if self.model_name != 'MAJ':
            self.optuna = config['optuna']
        else:
            self.optuna = False


    def train(self):
        if self.optuna:
            if self.model_name == 'FMS':
                FMS_main_optuna(self.config)
            elif self.model_name == 'LEM_action':
                LEM_action_main_optuna(self.config)
            elif self.model_name == 'LEM':
                LEM_main_optuna(self.config)
            elif self.model_name == 'NMSTPP':
                NMSTPP_main_optuna(self.config)
            elif self.model_name == 'Seq2Event':
                Seq2Event_main_optuna(self.config)
            else:
                raise ValueError(f'Unknown model name: {self.model_name}')
        else:
            if self.model_name == 'FMS':
                FMS_main(self.config)
            elif self.model_name == 'LEM_action':
                LEM_action_main(self.config)
            elif self.model_name == 'LEM':
                LEM_main(self.config)
            elif self.model_name == 'MAJ':
                MAJ_main(self.config)
            elif self.model_name == 'NMSTPP':
                NMSTPP_main(self.config)
            elif self.model_name == 'Seq2Event':
                Seq2Event_main(self.config)
            else:
                raise ValueError(f'Unknown model name: {self.model_name}')

    def inference(self, model_path, model_config, train_path=None, valid_path=None, save_path=None, simulation=False, random_selection=True, max_iter=20):
        #read the json file
        with open(model_config, 'r') as file:
            config_json = json.load(file)

        #check if the paths are provided
        if train_path is None:
            train_path = config_json['train_path']
        if valid_path is None:
            valid_path = config_json['valid_path']
        if save_path is None:
            save_path = config_json['save_path']+"/inference/"
        os.makedirs(save_path, exist_ok=True)

        #inference function
        if simulation==False:
            if self.model_name in ['FMS', 'MAJ']:
                inferenced_data = FMS_inference.FMS_inference(train_path, valid_path, self.model_name, model_path, model_config)
            elif self.model_name=='LEM':
                inferenced_data = LEM_inference.LEM_inference(train_path, valid_path, self.model_name, model_path, model_config)
            elif self.model_name in ['NMSTPP', 'Seq2Event']:
                inferenced_data = UEID_inference.UEID_inference(train_path, valid_path, self.model_name, model_path, model_config)
            else:
                raise ValueError(f'Unknown model name: {self.model_name}')
            #save the inferenced data
            inferenced_data.to_csv(save_path+"inference.csv",index=False)
        else:
            if self.model_name in ['FMS', 'MAJ']:
                df = FMS_inference.FMS_simulation_possession(train_path, valid_path, self.model_name, 
                                                             model_path, model_config, random_selection=random_selection, max_iter=max_iter)
                timestep_eval_df,es_hota_df = FMS_inference.simulation_evaluation(df, valid_path)
            elif self.model_name == 'LEM':
                df = LEM_inference.LEM_simulation_possession(train_path, valid_path, self.model_name, 
                                                             model_path, model_config, random_selection=random_selection, max_iter=max_iter)
                timestep_eval_df,es_hota_df = LEM_inference.simulation_evaluation(df, valid_path)
            elif self.model_name in ['NMSTPP', 'Seq2Event']:
                df = UEID_inference.UEID_simulation_possession(train_path, valid_path, self.model_name, 
                                                             model_path, model_config, random_selection=random_selection, max_iter=max_iter)
                timestep_eval_df,es_hota_df = UEID_inference.simulation_evaluation(df, valid_path)
            else:
                raise ValueError(f'Unknown model name: {self.model_name}')
            #save the inferenced data
            df.to_csv(save_path+"simulation.csv",index=False)
            timestep_eval_df.to_csv(save_path+"timestep_eval.csv",index=False)
            es_hota_df.to_csv(save_path+"ES_HOTA.csv",index=False)

if __name__ == '__main__':
    # #Test FMS
    # model = event_model_soccer('FMS', os.getcwd()+'/event/sports/soccer/models/train_FMS.yaml')
    # model.train()
    # #Example only, run the inference function after training
    # model_path = os.getcwd()+'/test/model/FMS/out/train/20240922-181450/run_1/_model_1.pth'
    # model_config = os.getcwd()+'/test/model/FMS/out/train/20240922-181450/run_1/hyperparameters.json'
    # model.inference(model_path, model_config) #simple inference
    # model.inference(model_path, model_config, simulation=True, random_selection=True, max_iter=20) #simulation with evaluation
    # model = event_model_soccer('FMS', os.getcwd()+'/event/sports/soccer/models/train_FMS_optuna.yaml')
    # model.train()
    # print('FMS Done')

    # #Test LEM_action
    # model = event_model_soccer('LEM_action', os.getcwd()+'/event/sports/soccer/models/train_LEM_action.yaml')
    # model.train()
    # model = event_model_soccer('LEM_action', os.getcwd()+'/event/sports/soccer/models/train_LEM_action_optuna.yaml')
    # model.train()
    # print('LEM_action Done')

    #Test LEM
    # model = event_model_soccer('LEM', os.getcwd()+'/event/sports/soccer/models/train_LEM.yaml')
    # model.train()
    # #Example only, run the inference function after training
    # model_path = os.getcwd()+'/test/model/LEM/out/train/20240922-230404/run_1/_model_1.pth'
    # model_config = os.getcwd()+'/test/model/LEM/out/train/20240922-230404/run_1/hyperparameters.json'
    # model.inference(model_path, model_config) #simple inference
    # model.inference(model_path, model_config, simulation=True, random_selection=True, max_iter=20) #simulation with evaluation
    # model = event_model_soccer('LEM', os.getcwd()+'/event/sports/soccer/models/train_LEM_optuna.yaml')
    # model.train()
    # print('LEM Done')
    
    # #Test MAJ
    # model = event_model_soccer('MAJ', os.getcwd()+'/event/sports/soccer/models/train_MAJ.yaml')
    # model.train()
    # model_path = os.getcwd()+'/test/model/MAJ/out/train/20240923_061427/run_1/_model_1.pth'
    # model_config = os.getcwd()+'/test/model/MAJ/out/train/20240923_061427/run_1/hyperparameters.json'
    # model.inference(model_path, model_config) #simple inference
    # model.inference(model_path, model_config, simulation=True, random_selection=True, max_iter=20) #simulation with evaluation
    # print('MAJ Done')

    # #Test NMSTPP
    # model = event_model_soccer('NMSTPP', os.getcwd()+'/event/sports/soccer/models/train_NMSTPP.yaml')
    # model.train()
    # model_path = os.getcwd()+'/test/model/NMSTPP/out/train/20240922-134432/run_1/_model_1.pth'
    # model_config = os.getcwd()+'/test/model/NMSTPP/out/train/20240922-134432/run_1/hyperparameters.json'
    # model.inference(model_path, model_config) #simple inference
    # model.inference(model_path, model_config, simulation=True, random_selection=True, max_iter=20) #simulation with evaluation
    # model = event_model_soccer('NMSTPP', os.getcwd()+'/event/sports/soccer/models/train_NMSTPP_optuna.yaml')
    # model.train()
    # print('NMSTPP Done')

    # #Test Seq2Event
    # model = event_model_soccer('Seq2Event', os.getcwd()+'/event/sports/soccer/models/train_Seq2Event.yaml')
    # model.train()
    # model_path = os.getcwd()+'/test/model/Seq2Event/out/train/20240922-150633/run_1/_model_1.pth'
    # model_config = os.getcwd()+'/test/model/Seq2Event/out/train/20240922-150633/run_1/hyperparameters.json'
    # model.inference(model_path, model_config) #simple inference
    # model.inference(model_path, model_config, simulation=True, random_selection=True, max_iter=20) #simulation with evaluation
    # model = event_model_soccer('Seq2Event', os.getcwd()+'/event/sports/soccer/models/train_Seq2Event_optuna.yaml')
    # model.train()
    # print('Seq2Event Done')
    print('Done')