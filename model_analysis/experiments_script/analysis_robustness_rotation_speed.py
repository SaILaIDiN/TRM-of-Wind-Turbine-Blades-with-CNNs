""" This script investigates the capability of the CNN to correctly classify samples of rotor blades from previously
    unseen values of a certain meta parameter.
"""
import os
import argparse
from utils.env_config import setup_environment
from data.dataloader import prepare_dataloaders, prepare_dataset_partitions, concat_dataset_partitions, \
    create_dataloaders
from model_analysis.analysis import execute_optimization, execute_inference
from utils.mlflow_custom_functions import get_mlflow_experiment, get_mlflow_run


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--system_mode", type=str, default="local-H-Drive")
    args = parser.parse_args()
    setup_environment(args.system_mode)

    EXPERIMENT_NAME = "robustness_rotation_speed"
    EXPERIMENT_ID = get_mlflow_experiment(EXPERIMENT_NAME)

    dir_python_script = os.path.join(
        os.environ.get("CODE_PATH_ANALYSIS"), "analysis_robustness_rotation_speed.py")

    dir_yaml_data_low = os.path.join(
        os.environ.get("YAML_PATH_ANALYSIS"), "robustness_rotation_speed_low_prep_data_vb06.yaml")
    dir_yaml_data_high = os.path.join(
        os.environ.get("YAML_PATH_ANALYSIS"), "robustness_rotation_speed_high_prep_data_vb06.yaml")
    dir_yaml_train_low = os.path.join(
        os.environ.get("YAML_PATH_ANALYSIS"), "robustness_rotation_speed_low_train_vb06.yaml")
    dir_yaml_train_high = os.path.join(
        os.environ.get("YAML_PATH_ANALYSIS"), "robustness_rotation_speed_high_train_vb06.yaml")
    dir_yaml_infer_high = os.path.join(
        os.environ.get("YAML_PATH_ANALYSIS"), "robustness_rotation_speed_low_train_high_infer_vb06.yaml")
    dir_yaml_infer_low = os.path.join(
        os.environ.get("YAML_PATH_ANALYSIS"), "robustness_rotation_speed_high_train_low_infer_vb06.yaml")

    dir_yaml_train_low_and_high = os.path.join(
        os.environ.get("YAML_PATH_ANALYSIS"), "robustness_rotation_speed_low_and_high_train_vb06.yaml")
    dir_yaml_data_mid = os.path.join(
        os.environ.get("YAML_PATH_ANALYSIS"), "robustness_rotation_speed_mid_prep_data_vb06.yaml")
    dir_yaml_infer_mid = os.path.join(
        os.environ.get("YAML_PATH_ANALYSIS"), "robustness_rotation_speed_low_and_high_train_mid_infer_vb06.yaml")

    # Run 0: log python script of experiment as artifact
    RUN_NAME = "log_python_script_of_experiment"
    get_mlflow_run(EXPERIMENT_ID, RUN_NAME, config_path_save=dir_python_script)

    # Run 1: initialize vb06 DataLoaders (lower end of parameter distribution)
    RUN_NAME = "robustness_rotation_speed_low_prep_data_vb06"
    RUN_ID = get_mlflow_run(EXPERIMENT_ID, RUN_NAME, config_path_save=dir_yaml_data_low)
    partition_loaders_low = prepare_dataloaders(dir_yaml_data_low, EXPERIMENT_ID, EXPERIMENT_NAME, RUN_ID, RUN_NAME)

    # Run 2: initialize vb06 DataLoaders (higher end of parameter distribution)
    RUN_NAME = "robustness_rotation_speed_high_prep_data_vb06"
    RUN_ID = get_mlflow_run(EXPERIMENT_ID, RUN_NAME, config_path_save=dir_yaml_data_high)
    partition_loaders_high = prepare_dataloaders(dir_yaml_data_high, EXPERIMENT_ID, EXPERIMENT_NAME, RUN_ID, RUN_NAME)

    # Run 3: optimize model with dataset (lower end of parameter distribution)
    RUN_NAME = "robustness_rotation_speed_low_train_vb06"
    RUN_ID = get_mlflow_run(EXPERIMENT_ID, RUN_NAME, config_path_save=dir_yaml_train_low)
    execute_optimization(EXPERIMENT_ID, EXPERIMENT_NAME, RUN_ID, RUN_NAME, dir_yaml_train_low, partition_loaders_low)

    # Run 4: optimize model with dataset (higher end of parameter distribution)
    RUN_NAME = "robustness_rotation_speed_high_train_vb06"
    RUN_ID = get_mlflow_run(EXPERIMENT_ID, RUN_NAME, config_path_save=dir_yaml_train_high)
    execute_optimization(EXPERIMENT_ID, EXPERIMENT_NAME, RUN_ID, RUN_NAME, dir_yaml_train_high, partition_loaders_high)

    # Run 5: infer higher end of parameter distribution on model from lower end of parameter distribution
    RUN_NAME = "robustness_rotation_speed_low_train_high_infer_vb06"
    RUN_ID = get_mlflow_run(EXPERIMENT_ID, RUN_NAME, config_path_save=dir_yaml_infer_high)
    test_loader_high = partition_loaders_high[0]
    execute_inference(EXPERIMENT_ID, EXPERIMENT_NAME, RUN_ID, RUN_NAME, dir_yaml_infer_high, test_loader_high)

    # Run 6: infer lower end of parameter distribution on model from higher end of parameter distribution
    RUN_NAME = "robustness_rotation_speed_high_train_low_infer_vb06"
    RUN_ID = get_mlflow_run(EXPERIMENT_ID, RUN_NAME, config_path_save=dir_yaml_infer_low)
    test_loader_low = partition_loaders_low[0]
    execute_inference(EXPERIMENT_ID, EXPERIMENT_NAME, RUN_ID, RUN_NAME, dir_yaml_infer_low, test_loader_low)

    # Run 7: optimize model with dataset (lower and higher end of parameter distribution)
    RUN_NAME = "robustness_rotation_speed_low_and_high_train_vb06"
    RUN_ID = get_mlflow_run(EXPERIMENT_ID, RUN_NAME, config_path_save=dir_yaml_train_low_and_high)
    partition_datasets_low = prepare_dataset_partitions(dir_yaml_data_low, EXPERIMENT_ID, EXPERIMENT_NAME, RUN_ID, RUN_NAME)
    partition_datasets_high = prepare_dataset_partitions(dir_yaml_data_high, EXPERIMENT_ID, EXPERIMENT_NAME, RUN_ID, RUN_NAME)
    new_partitions = concat_dataset_partitions(partition_datasets_low, partition_datasets_high)
    partition_loaders_low_and_high = create_dataloaders(new_partitions[0], new_partitions[1], new_partitions[2], batch_size=16, num_workers=2, torch_seed=42)
    execute_optimization(EXPERIMENT_ID, EXPERIMENT_NAME, RUN_ID, RUN_NAME, dir_yaml_train_low_and_high, partition_loaders_low_and_high)

    # Run 8: infer middle of parameter distribution on model from lower and higher end of parameter distribution
    RUN_NAME = "robustness_rotation_speed_low_and_high_train_mid_infer_vb06"
    RUN_ID = get_mlflow_run(EXPERIMENT_ID, RUN_NAME, config_path_save=dir_yaml_infer_mid)
    partition_loaders_mid = prepare_dataloaders(dir_yaml_data_mid, EXPERIMENT_ID, EXPERIMENT_NAME, RUN_ID, RUN_NAME)
    execute_inference(EXPERIMENT_ID, EXPERIMENT_NAME, RUN_ID, RUN_NAME, dir_yaml_infer_mid, partition_loaders_mid[0])
