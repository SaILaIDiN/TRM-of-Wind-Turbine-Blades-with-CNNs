""" This script investigates the ability of the model to detect rotor blades of a wind turbine when previously trained
    with another turbine's data.
    The CNN will be trained on data from location X and its performance will be compared on the test data for location X
    and location Y. The same will be performed in the opposite direction.
    For a fair comparison, the compared data between location X and Y should be within the same time interval.
"""
import os
import argparse
from utils.env_config import setup_environment
from data.dataloader import prepare_dataloaders
from model_analysis.analysis import execute_optimization, execute_inference, execute_finetuning
from utils.mlflow_custom_functions import get_mlflow_experiment, get_mlflow_run


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--system_mode", type=str, default="local-H-Drive")
    args = parser.parse_args()
    setup_environment(args.system_mode)

    EXPERIMENT_NAME = "transfer_location"
    EXPERIMENT_ID = get_mlflow_experiment(EXPERIMENT_NAME)

    dir_python_script = os.path.join(os.environ.get("CODE_PATH_ANALYSIS"), "analysis_transfer_location.py")

    dir_yaml_data_vb06 = os.path.join(os.environ.get("YAML_PATH_ANALYSIS"), "transfer_location_prep_data_vb06.yaml")
    dir_yaml_train_vb06 = os.path.join(os.environ.get("YAML_PATH_ANALYSIS"), "transfer_location_train_vb06.yaml")
    dir_yaml_infer_vb07 = os.path.join(os.environ.get("YAML_PATH_ANALYSIS"), "transfer_location_infer_vb07.yaml")
    dir_yaml_finetune_vb07 = os.path.join(os.environ.get("YAML_PATH_ANALYSIS"), "transfer_location_finetune_vb07.yaml")

    dir_yaml_data_vb07 = os.path.join(os.environ.get("YAML_PATH_ANALYSIS"), "transfer_location_prep_data_vb07.yaml")
    dir_yaml_train_vb07 = os.path.join(os.environ.get("YAML_PATH_ANALYSIS"), "transfer_location_train_vb07.yaml")
    dir_yaml_infer_vb06 = os.path.join(os.environ.get("YAML_PATH_ANALYSIS"), "transfer_location_infer_vb06.yaml")
    dir_yaml_finetune_vb06 = os.path.join(os.environ.get("YAML_PATH_ANALYSIS"), "transfer_location_finetune_vb06.yaml")

    # Run 0: log python script of experiment as artifact
    RUN_NAME = "log_python_script_of_experiment"
    get_mlflow_run(EXPERIMENT_ID, RUN_NAME, config_path_save=dir_python_script)

    # Run 1: initialize vb06 DataLoaders
    RUN_NAME = "transfer_location_prep_data_vb06"
    RUN_ID = get_mlflow_run(EXPERIMENT_ID, RUN_NAME, config_path_save=dir_yaml_data_vb06)
    partition_loaders_vb06 = prepare_dataloaders(dir_yaml_data_vb06, EXPERIMENT_ID, EXPERIMENT_NAME, RUN_ID, RUN_NAME)

    # Run 2: initialize vb07 DataLoaders
    RUN_NAME = "transfer_location_prep_data_vb07"
    RUN_ID = get_mlflow_run(EXPERIMENT_ID, RUN_NAME, config_path_save=dir_yaml_data_vb07)
    partition_loaders_vb07 = prepare_dataloaders(dir_yaml_data_vb07, EXPERIMENT_ID, EXPERIMENT_NAME, RUN_ID, RUN_NAME)

    # Run 3: vb06 (optimize)
    RUN_NAME = "transfer_location_train_vb06"
    RUN_ID = get_mlflow_run(EXPERIMENT_ID, RUN_NAME, config_path_save=dir_yaml_train_vb06)
    execute_optimization(EXPERIMENT_ID, EXPERIMENT_NAME, RUN_ID, RUN_NAME, dir_yaml_train_vb06, partition_loaders_vb06)

    # Run 4: vb07 (optimize)
    RUN_NAME = "transfer_location_train_vb07"
    RUN_ID = get_mlflow_run(EXPERIMENT_ID, RUN_NAME, config_path_save=dir_yaml_train_vb07)
    execute_optimization(EXPERIMENT_ID, EXPERIMENT_NAME, RUN_ID, RUN_NAME, dir_yaml_train_vb07, partition_loaders_vb07)

    # Run 5: test vb06 model on vb07 data
    RUN_NAME = "transfer_location_vb06_to_vb07_inference"
    RUN_ID = get_mlflow_run(EXPERIMENT_ID, RUN_NAME, config_path_save=dir_yaml_infer_vb07)
    test_loader_vb07 = partition_loaders_vb07[0]  # switch placement of partitions (this was the original training set)
    execute_inference(EXPERIMENT_ID, EXPERIMENT_NAME, RUN_ID, RUN_NAME, dir_yaml_infer_vb07, test_loader_vb07)

    # Run 6: test vb07 model on vb06 data
    RUN_NAME = "transfer_location_vb07_to_vb06_inference"
    RUN_ID = get_mlflow_run(EXPERIMENT_ID, RUN_NAME, config_path_save=dir_yaml_infer_vb06)
    test_loader_vb06 = partition_loaders_vb06[0]  # switch placement of partitions (this was the original training set)
    execute_inference(EXPERIMENT_ID, EXPERIMENT_NAME, RUN_ID, RUN_NAME, dir_yaml_infer_vb06, test_loader_vb06)

    # Run 7: fine-tune vb06 model with vb07 and evaluate on vb07
    RUN_NAME = "transfer_location_vb06_to_vb07_finetuning"
    RUN_ID = get_mlflow_run(EXPERIMENT_ID, RUN_NAME, config_path_save=dir_yaml_finetune_vb07)
    loader_0, loader_1, loader_2 = partition_loaders_vb07
    partition_loaders_vb07 = loader_2, loader_1, loader_0  # swap placement of loaders to use test part to finetune
    # mental check: finetuning dataset should be 10% of total data, see *prep_data*.yaml file
    execute_finetuning(EXPERIMENT_ID, EXPERIMENT_NAME, RUN_ID, RUN_NAME, dir_yaml_finetune_vb07, partition_loaders_vb07)

    # Run 8: fine-tune vb07 model with vb06 and evaluate on vb06
    RUN_NAME = "transfer_location_vb07_to_vb06_finetuning"
    RUN_ID = get_mlflow_run(EXPERIMENT_ID, RUN_NAME, config_path_save=dir_yaml_finetune_vb06)
    loader_0, loader_1, loader_2 = partition_loaders_vb06
    partition_loaders_vb06 = loader_2, loader_1, loader_0  # swap placement of loaders to use test part to finetune
    # mental check: finetuning dataset should be 10% of total data, see *prep_data*.yaml file
    execute_finetuning(EXPERIMENT_ID, EXPERIMENT_NAME, RUN_ID, RUN_NAME, dir_yaml_finetune_vb06, partition_loaders_vb06)
