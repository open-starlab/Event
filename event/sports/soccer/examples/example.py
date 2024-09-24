from event import Event_Model

import os

# To train NMSTPP model
model = Event_Model('NMSTPP', 'path/to/train_NMSTPP.yaml')
# #or to use optuna for hyperparameter optimization
# model = Event_Model('NMSTPP', 'path/to/train_NMSTPP_optuna.yaml')
model.train()

# Inference
# Example only, run the inference function after training
model_path = 'path/to/_model_1.pth'
model_config = 'path/to/hyperparameters.json'
# Simple inference
model.inference(model_path, model_config) 
# Simulation with evaluation
model.inference(model_path, model_config, simulation=True, random_selection=True, max_iter=20) 

