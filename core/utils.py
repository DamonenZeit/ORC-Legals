"""
Contains various utility functions for PyTorch model training and saving.
"""
import torch
import torchvision
import torch.nn as nn
from pathlib import Path

def save_model(model: torch.nn.Module,
            target_dir: str,
            model_name: str,
            epoch: int,
            optimizer: torch.optim,
            loss_fn: torch.nn,
            log_state: dict,
            class_names: list
            ):
    """Saves a PyTorch model to a target directory.

    Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
        either ".pth" or ".pt" as the file extension.
    epoch: Number of epoches are used to train model.
    optimizer: The optimizer that is used to train the model.
    loss_fn: The loss function that is used in training procedure
    log_state: The logging state when training the model, should be at dict type
        For example: {'train_loss': [ 0.17012777348281816, 0.06052514116334565],
        'train_acc': [ 0.9375, 0.9816176470588235],
        'test_loss': [0.06218957348028198, 0.06115782563574612],
        'test_acc': [1.0, 0.9583333333333333]}
    class_names: names of  class in data
    Example usage:
    save_model(model=model_0,
                target_dir="models",
                model_name="05_going_modular_tingvgg_model.pth")
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                        exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss_fn': loss_fn,
        'log_state': log_state,
        'class_names': class_names
        }, model_save_path)

def load_model(target_dir: str):
    """Loading a model checkpoint and then load state dict to the provided model

    Args:
        model: the model to load state dict(they must have the same architecture such as resnet152,...)
        target_dir: The path to the modelcheckpoint
            For example: /content/resnet152.pth
    Return:
        The model have been loaded state dict
        A dict contains epoch, optimizer_statedict, loss_fn, log_state(dict type)
    """
    checkpoint = torch.load(target_dir)
    model_state_dict = checkpoint["model_state_dict"]
    epoch = checkpoint["epoch"]
    optimizer_state_dict = checkpoint["optimizer_state_dict"]
    loss_fn = checkpoint["loss_fn"]
    log_state = checkpoint["log_state"]
    class_names = checkpoint["class_names"]
    return {
        "epoch": epoch,
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": optimizer_state_dict,
        "loss_fn": loss_fn,
        "log_state": log_state,
        "class_names": class_names
    }

def init_model(model_name: str, out_features: int = None):
    if model_name=="resnet152":
        weights = torchvision.models.ResNet152_Weights.DEFAULT # NEW in torchvision 0.13, "DEFAULT" means "best weights available"
        model = torchvision.models.resnet152(weights=weights)
        # Change the number of out feature to your own number of class
        if(out_features!=None):
            ## Show the default out features
            default_out_features = model.fc.out_features
            print(f"The default out features: {default_out_features}")
            ## Change the default out features to number of class we have
            default_in_features = model.fc.in_features
            model.fc = torch.nn.Linear(in_features=default_in_features, out_features=6)
            adjusted_out_features = model.fc.out_features
            print(f"The  out features after being adjusted: {adjusted_out_features}")
    return model
