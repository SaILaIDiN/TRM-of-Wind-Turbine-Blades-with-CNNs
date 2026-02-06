import os
import yaml
import mlflow
import torch
from utils.env_config import path_check_and_join
from utils.models import WindTurbineModel
from model_optim.optimize import optimize_model, help_evaluate_dataset, help_mlflow_logging


def execute_optimization(experiment_id, experiment_name, run_id, run_name, dir_yaml, partition_loaders, model=None):
    """ Loads the parameters relevant for optimization from yaml-configuration and runs the optimization. """
    # Load configuration from YAML file
    with open(dir_yaml, 'r') as config_file:
        config = yaml.safe_load(config_file)

    torch_manual_seed = config.get("torch_manual_seed", 42)
    n_metadata = config["n_metadata"]
    n_classes = config["n_classes"]
    epochs = config["epochs"]
    checkpointing = config["checkpointing"]
    name_optim = config["name_optim"]
    lr = config["lr"]
    momentum = config["momentum"]
    eval_val = config["eval_val"]
    eval_train = config["eval_train"]
    eval_test = config["eval_test"]

    # Initialize model
    torch.manual_seed(torch_manual_seed)
    if model is None:
        model = WindTurbineModel(n_metadata=n_metadata, n_classes=n_classes)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # Provide path to save final model
    if "RESULTS_PATH" in os.environ and "MODELS_PATH" in os.environ:
        results_path = os.environ.get("RESULTS_PATH")
        model_path = os.environ.get("MODELS_PATH")
    else:
        print("Environmental variables not set!")
        return

    # Optimize
    train_loader, val_loader, test_loader = partition_loaders
    optimize_model(train_loader, val_loader, test_loader, model, n_metadata, n_classes, epochs, name_optim,
                   lr, momentum, device, experiment_id, experiment_name, run_id, run_name=run_name,
                   save_model_path=model_path, checkpointing=checkpointing, results_path=results_path,
                   eval_val=eval_val, eval_train=eval_train, eval_test=eval_test)


def execute_inference(experiment_id, experiment_name, run_id, run_name, dir_yaml, dataloader, model=None):
    """ Loads the pretrained weights into a model and evaluates the performance on a dataset. """
    # Load configuration data from YAML file
    with open(dir_yaml, 'r') as config_file:
        config = yaml.safe_load(config_file)

    # Configuration data for model loading
    run_name_opt = config["run_name_opt"]
    model_file = config["model_pth"]

    # Configuration data for inference
    n_metadata = config["n_metadata"]
    n_classes = config["n_classes"]
    epoch = config["best_epoch"]
    mode = config["mode"]
    log_via_mlflow = config["log_via_mlflow"]

    if "RESULTS_PATH" in os.environ and "MODELS_PATH" in os.environ:
        results_path = os.environ.get("RESULTS_PATH")
        model_path = os.environ.get("MODELS_PATH")
    else:
        print("Environmental variables not set!")
        return
    model_path = path_check_and_join(model_path, [experiment_name, run_name_opt, model_file], end_is_file=True)

    with mlflow.start_run(experiment_id=experiment_id, run_id=run_id):

        # Load and setup pretrained model
        if model is None:
            model = WindTurbineModel(n_metadata=n_metadata, n_classes=n_classes)
        model.load_state_dict(torch.load(model_path))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Inference
        eval_dict = help_evaluate_dataset(dataloader, mode, model, n_metadata, n_classes, epoch, device, results_path,
                                          experiment_name, run_name)
        if log_via_mlflow == "True":
            help_mlflow_logging(eval_dict=eval_dict, epoch=epoch)


def execute_finetuning(experiment_id, experiment_name, run_id, run_name, dir_yaml, partition_loaders, model=None):
    """ Loads the pretrained weights into a model and then finetunes it on another dataset. """
    # Load configuration from YAML file
    with open(dir_yaml, 'r') as config_file:
        config = yaml.safe_load(config_file)

    # Configuration data for model loading
    run_name_opt = config["run_name_opt"]
    model_file = config["model_pth"]

    # Configuration data for optimization (finetuning)
    n_metadata = config["n_metadata_fine"]
    n_classes = config["n_classes"]
    epochs = config["epochs_fine"]
    checkpointing = config["checkpointing_fine"]
    name_optim = config["name_optim_fine"]
    lr = config["lr_fine"]
    momentum = config["momentum_fine"]
    eval_val = config["eval_val_fine"]
    eval_train = config["eval_train_fine"]
    eval_test = config["eval_test_fine"]

    if "RESULTS_PATH" in os.environ and "MODELS_PATH" in os.environ:
        results_path = os.environ.get("RESULTS_PATH")
        model_path = os.environ.get("MODELS_PATH")
    else:
        print("Environmental variables not set!")
        return
    model_path_pretrained = path_check_and_join(
        model_path, [experiment_name, run_name_opt, model_file], end_is_file=True)

    # Load pretrained model
    if model is None:
        model = WindTurbineModel(n_metadata=n_metadata, n_classes=n_classes)
    model.load_state_dict(torch.load(model_path_pretrained))

    # Freeze layers of choice
    for name, child in model.named_children():
        if isinstance(child, (torch.nn.Conv2d, torch.nn.BatchNorm2d)):
            for param in child.parameters():
                param.requires_grad = False
        else:
            for param in child.parameters():
                param.requires_grad = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Optimize
    train_loader, val_loader, test_loader = partition_loaders
    optimize_model(train_loader, val_loader, test_loader, model, n_metadata, n_classes, epochs, name_optim,
                   lr, momentum, device, experiment_id, experiment_name, run_id, run_name=run_name,
                   save_model_path=model_path, checkpointing=checkpointing, results_path=results_path,
                   eval_val=eval_val, eval_train=eval_train, eval_test=eval_test)
