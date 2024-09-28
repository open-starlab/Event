import torch
import torch.nn as nn
import pdb

def cost_function(gt, pred, weight_action_class=None, action_weight=1, deltaT_weight=5, location_weight=10, min_dict=None, max_dict=None, device="None",config=None):
    if config is None and config =="None":
        num_actions = 9
    else:
        num_actions = config["num_actions"]

    action = gt[:,0].long()
    deltaT = gt[:,1].float()
    start_x = gt[:,2].float()
    start_y = gt[:,3].float()

    action_pred = pred[:,0:num_actions].float()
    deltaT_pred = pred[:,num_actions].float()
    start_x_pred = pred[:,num_actions+1].float()
    start_y_pred = pred[:,num_actions+2].float()

    CEL_action_func = nn.CrossEntropyLoss(weight=weight_action_class,reduction ="none")
    CEL_action  = torch.mean(CEL_action_func(action_pred,action)) 

    RMSE_deltaT = torch.mean((deltaT_pred-deltaT)**2)**0.5
    RMSE_location = torch.mean((start_x_pred-start_x)**2 + (start_y_pred-start_y)**2)**0.5

    #unprocess the loss components
    if max_dict is not None and min_dict is not None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None or device == 'None' else device
        deltaT_pred = torch.exp( deltaT_pred * ( torch.tensor(max_dict["delta_T"], device=device) - torch.tensor(min_dict["delta_T"], device=device)) + torch.tensor(min_dict["delta_T"], device=device)) - torch.tensor(1e-6, device=device)
        deltaT = torch.exp( deltaT * ( torch.tensor(max_dict["delta_T"], device=device) - torch.tensor(min_dict["delta_T"], device=device)) + torch.tensor(min_dict["delta_T"], device=device)) - torch.tensor(1e-6, device=device)
        start_x_pred = start_x_pred * ( torch.tensor(max_dict["start_x"], device=device) - torch.tensor(min_dict["start_x"], device=device)) + torch.tensor(min_dict["start_x"], device=device)
        start_y_pred = start_y_pred * ( torch.tensor(max_dict["start_y"], device=device) - torch.tensor(min_dict["start_y"], device=device)) + torch.tensor(min_dict["start_y"], device=device)
        start_x = start_x * ( torch.tensor(max_dict["start_x"], device=device) - torch.tensor(min_dict["start_x"], device=device)) + torch.tensor(min_dict["start_x"], device=device)
        start_y = start_y * ( torch.tensor(max_dict["start_y"], device=device) - torch.tensor(min_dict["start_y"], device=device)) + torch.tensor(min_dict["start_y"], device=device)

        #ignore in backpropagation
        ACC_action = torch.mean((torch.argmax(action_pred,1)==action).float()).detach()
        AE_deltaT = torch.mean(torch.abs(deltaT_pred-deltaT)).detach()
        AE_start_x = torch.mean(torch.abs(start_x_pred-start_x)).detach()
        AE_start_y = torch.mean(torch.abs(start_y_pred-start_y)).detach()
    else:
        ACC_action = torch.tensor(-1, device=device)
        AE_deltaT = torch.tensor(-1, device=device)
        AE_start_x = torch.tensor(-1, device=device)
        AE_start_y = torch.tensor(-1, device=device)
    
    #f1 score
    pred_classes = torch.argmax(action_pred, dim=1)
    f1_action = calculate_f1_score(action, pred_classes, num_classes=num_actions)
        
    #cost function
    train_loss = CEL_action*action_weight + RMSE_deltaT*deltaT_weight + RMSE_location*location_weight

    return train_loss,[CEL_action,RMSE_deltaT,RMSE_location,ACC_action,f1_action,AE_deltaT,AE_start_x,AE_start_y]

def calculate_f1_score(gt_classes, pred_classes, num_classes):
    f1_scores = []
    
    for i in range(num_classes):
        # True Positives (TP), False Positives (FP), False Negatives (FN)
        tp = torch.sum((pred_classes == i) & (gt_classes == i)).float()
        fp = torch.sum((pred_classes == i) & (gt_classes != i)).float()
        fn = torch.sum((pred_classes != i) & (gt_classes == i)).float()
        
        # Precision and Recall
        precision = tp / (tp + fp + 1e-8)  # Add small constant to avoid division by zero
        recall = tp / (tp + fn + 1e-8)     # Add small constant to avoid division by zero
        
        # F1 Score
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)  # Add small constant to avoid division by zero
        f1_scores.append(f1)
    
    # Average F1 Score across all classes
    avg_f1_score = torch.mean(torch.stack(f1_scores))
    
    return avg_f1_score