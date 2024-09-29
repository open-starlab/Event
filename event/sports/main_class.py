# from /home/c_yeung/workspace6/python/openstarlab/Event/event/sports/soccer/main_class_soccer/main.py
from .soccer.main_class_soccer.main import event_model_soccer

class Event_Model:
    soccer_event_model = ['FMS','LEM_action','LEM','MAJ','NMSTPP','Seq2Event']
    other_model = []

    def __new__(cls, event_model, *args, **kwargs):
        if event_model in cls.soccer_event_model:
            return event_model_soccer(event_model, *args, **kwargs)
        elif event_model in cls.other_model:
            raise NotImplementedError('other model not implemented yet')
        else:
            raise ValueError(f'Unknown event model: {event_model}')


if __name__ == '__main__':
    import os
    #Test FMS
    # model = 'Seq2Event'
    # model = Event_Model(model, f'/home/c_yeung/workspace6/python/openstarlab/Event_Pretrain/model_yaml_test/train_{model}_optuna.yaml')
    # model.train()
    #Example only, run the inference function after training
    # model_path = os.getcwd()+'/test/model/FMS/out/train/20240922-181450/run_1/_model_1.pth'
    # model_config = os.getcwd()+'/test/model/FMS/out/train/20240922-181450/run_1/hyperparameters.json'
    # model.inference(model_path, model_config) #simple inference
    # model.inference(model_path, model_config, simulation=True, random_selection=True, max_iter=20) #simulation with evaluation
    # model = Event_Model('FMS', os.getcwd()+'/event/sports/soccer/models/train_FMS_optuna.yaml')
    # model.train()
    print('Done')