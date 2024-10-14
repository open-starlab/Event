import torch
import numpy as np
from ..utils.MAJ_cost_function import MAJ_cost_function
import torchprofile
import warnings
import pdb

def MAJ_epoch_function(model, dataloader, min_dict, max_dict, device, train=True, print_freq=10,config=None):
    model.eval()

    total_loss = 0
    total_loss_components = [0]*9
    total_batches = len(dataloader)
    
    for i, (inputs, gt, _) in enumerate(dataloader):
        inputs, gt = inputs.to(device), gt.to(device)
        num_action=config["num_actions"] if config is not None or config !="None" else 9
        
        gt_action_one_hot = torch.nn.functional.one_hot(gt[:,0].long(), num_classes=num_action).float()
        gt_deltaT_one_hot = one_hot_encode_from_continuous(gt[:,1], num_bins=config["delta_T_bin"]).float()
        gt_start_x_one_hot = one_hot_encode_from_continuous(gt[:,2], num_bins=config["start_x_bin"]).float()
        gt_start_y_one_hot = one_hot_encode_from_continuous(gt[:,3], num_bins=config["start_y_bin"]).float()
        gt_binary = torch.cat((gt_action_one_hot, gt_deltaT_one_hot, gt_start_x_one_hot, gt_start_y_one_hot), dim=1)

        # Forward pass
        outputs = model(inputs)
        loss, loss_components= MAJ_cost_function(gt_binary, outputs, min_dict, max_dict, device=device,config=config)
        
        total_loss += loss.item()
        #detach the loss components
        loss_components = [i.cpu().detach().numpy() for i in loss_components] if device != "cpu" else [i.detach().numpy() for i in loss_components]
        total_loss_components = [total_loss_components[j] + loss_components[j] for j in range(len(loss_components))]

        if (i + 1) % print_freq == 0 or (i + 1) == total_batches:

            average_loss = total_loss / (i + 1)
            average_loss_components = [total_loss_components[j] / (i + 1) for j in range(len(loss_components))]
            
            print(f"Batch [{i + 1}/{total_batches}] | Loss: {average_loss:.4f} | "
                    f"BCEL_action: {average_loss_components[0]:.4f} | BCEL_deltaT: {average_loss_components[1]:.4f} | "
                    f"BCEL_start_x: {average_loss_components[2]:.4f} | BCEL_start_y: {average_loss_components[3]:.4f} | " 
                    f"ACC_action: {average_loss_components[4]:.4f} | F1_action: {average_loss_components[5]:.4f} | "
                    f"AE_deltaT: {average_loss_components[6]:.4f} | AE_start_x: {average_loss_components[7]:.4f} | AE_start_y: {average_loss_components[8]:.4f}"
                    )

    average_loss = total_loss / total_batches

    if train:
        return average_loss, average_loss_components, model.state_dict()
    else:
        return average_loss, average_loss_components

def one_hot_encode_from_continuous(gt, num_bins):
    # Clamp the continuous values to stay between 0 and 1
    gt = torch.clamp(gt, 0, 1)
    
    # Map the continuous values to bin indices
    bin_idx = (gt * (num_bins - 1)).long()

    # One-hot encode the bin indices
    one_hot_encoded = torch.nn.functional.one_hot(bin_idx, num_classes=num_bins).float()

    return one_hot_encoded

def MAJ_train(model, train_loader, valid_loader, min_dict, max_dict,  device, epochs=10, print_freq=10, patience=5,config=None):
    torch.cuda.empty_cache(); import gc; gc.collect()
    train_losses = []
    valid_losses = []
    train_loss_components = []
    valid_loss_components = []

    best_valid_loss = float('inf')
    epochs_no_improve = 0
    best_epoch = 0

    for epoch in range(epochs):
        torch.cuda.empty_cache(); import gc; gc.collect()
        print(f"Epoch [{epoch + 1}/{epochs}]")

        with torch.no_grad():
            # Training phase
            train_loss, average_loss_components, model_state_dict = MAJ_epoch_function(model, train_loader, min_dict, max_dict, device, train=True, print_freq=print_freq,config=config)
            train_losses.append(train_loss)
            train_loss_components.append(average_loss_components)
            print(f"Training Loss: {train_loss:.4f}")

            # Validation phase
            torch.cuda.empty_cache(); import gc; gc.collect()
            valid_loss, average_loss_components = MAJ_epoch_function(model, valid_loader, min_dict, max_dict, device=device, train=False, print_freq=print_freq,config=config)
        valid_losses.append(valid_loss)
        valid_loss_components.append(average_loss_components)
        print(f"Validation Loss: {valid_loss:.4f}\n")

        # Check if validation loss improved
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_model = model_state_dict
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Early stopping
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            model_state_dict = best_model if best_model else model.state_dict()
            epoch = best_epoch
            break

    #Print the best epoch, the best validation loss and the corresponding training loss
    print(f"Best epoch: {best_epoch + 1} | Best validation loss: {best_valid_loss:.4f} | Corresponding training loss: {train_losses[best_epoch]:.4f}")


    return model_state_dict, epoch, train_losses, valid_losses, train_loss_components, valid_loss_components, 

