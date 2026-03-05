""" This script investigates the features within the components of the trained neural network. """

import os
import argparse
from utils.env_config import setup_environment
from data.dataloader import prepare_dataloaders
from model_analysis.interpret_model import execute_class_activation_mapping
from utils.mlflow_custom_functions import get_mlflow_experiment, get_mlflow_run


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--system_mode", type=str, default="local-H-Drive")
    args = parser.parse_args()
    setup_environment(args.system_mode)

    EXPERIMENT_NAME = "robustness_rotation_speed"
    EXPERIMENT_ID = get_mlflow_experiment(EXPERIMENT_NAME)

    dir_python_script = os.path.join(os.environ.get("CODE_PATH_ANALYSIS"), "interpret_robustness_rotation_speed.py")

    dir_yaml_interpret_low = os.path.join(os.environ.get("YAML_PATH_ANALYSIS"), "robustness_rotation_speed_low_interpret_vb06.yaml")
    dir_yaml_interpret_high = os.path.join(os.environ.get("YAML_PATH_ANALYSIS"), "robustness_rotation_speed_high_interpret_vb06.yaml")
    dir_yaml_interpret_mid = os.path.join(os.environ.get("YAML_PATH_ANALYSIS"), "robustness_rotation_speed_mid_interpret_vb06.yaml")

    # NOTE: the yaml files contain both the dataset preparation config and the model inference config of the labeled
    # parameter range i.e. low or high. If it does not contain model inference data, then it has the data preparation!

    # Run 0: log python script of experiment as artifact
    RUN_NAME = "log_python_script_of_experiment"
    get_mlflow_run(EXPERIMENT_ID, RUN_NAME, config_path_save=dir_python_script)

    # Run 1: initialize vb06 DataLoaders (lower end of parameter distribution)
    partition_loaders_low = prepare_dataloaders(dir_yaml_interpret_low, EXPERIMENT_ID, EXPERIMENT_NAME, None, None)
    # run_id and run_name are ignored because this preparation of data is not logged in mlflow

    # Run 2: initialize vb06 DataLoaders (higher end of parameter distribution)
    partition_loaders_high = prepare_dataloaders(dir_yaml_interpret_high, EXPERIMENT_ID, EXPERIMENT_NAME, None, None)
    # run_id and run_name are ignored because this preparation of data is not logged in mlflow

    # Run 3: initialize vb06 DataLoaders (higher end of parameter distribution)
    partition_loaders_mid = prepare_dataloaders(dir_yaml_interpret_mid, EXPERIMENT_ID, EXPERIMENT_NAME, None, None)
    # run_id and run_name are ignored because this preparation of data is not logged in mlflow

    # Run 4: visualize vb06 samples with pretrained model (lower end of parameter distribution)
    RUN_NAME = "interpret_robustness_rotation_speed_low_vb06"
    RUN_ID = get_mlflow_run(EXPERIMENT_ID, RUN_NAME, config_path_save=dir_yaml_interpret_low)
    test_loader_low = partition_loaders_low[0]
    execute_class_activation_mapping(EXPERIMENT_ID, EXPERIMENT_NAME, RUN_ID, RUN_NAME, dir_yaml_interpret_low,
                                     test_loader_low)

    # Run 5: visualize vb06 samples with pretrained model (higher end of parameter distribution)
    RUN_NAME = "interpret_robustness_rotation_speed_high_vb06"
    RUN_ID = get_mlflow_run(EXPERIMENT_ID, RUN_NAME, config_path_save=dir_yaml_interpret_high)
    test_loader_high = partition_loaders_high[0]
    execute_class_activation_mapping(EXPERIMENT_ID, EXPERIMENT_NAME, RUN_ID, RUN_NAME, dir_yaml_interpret_high,
                                     test_loader_high)

    # Run 6: visualize vb06 samples from lower end of parameter distribution on model trained with higher end
    RUN_NAME = "interpret_robustness_rotation_speed_low_train_high_infer_vb06"
    RUN_ID = get_mlflow_run(EXPERIMENT_ID, RUN_NAME, config_path_save=[dir_yaml_interpret_low, dir_yaml_interpret_high])
    execute_class_activation_mapping(EXPERIMENT_ID, EXPERIMENT_NAME, RUN_ID, RUN_NAME, dir_yaml_interpret_low,
                                     test_loader_high)

    # Run 7: visualize vb06 samples from higher end of parameter distribution on model trained with lower end
    RUN_NAME = "interpret_robustness_rotation_speed_high_train_low_infer_vb06"
    RUN_ID = get_mlflow_run(EXPERIMENT_ID, RUN_NAME, config_path_save=[dir_yaml_interpret_high, dir_yaml_interpret_low])
    execute_class_activation_mapping(EXPERIMENT_ID, EXPERIMENT_NAME, RUN_ID, RUN_NAME, dir_yaml_interpret_high,
                                     test_loader_low)

    # Run 8: visualize vb06 samples from middle of parameter distribution on model trained with low and high ends
    RUN_NAME = "interpret_robustness_rotation_speed_low_and_high_train_mid_infer_vb06"
    RUN_ID = get_mlflow_run(EXPERIMENT_ID, RUN_NAME, config_path_save=dir_yaml_interpret_mid)
    test_loader_mid = partition_loaders_mid[0]
    execute_class_activation_mapping(EXPERIMENT_ID,EXPERIMENT_NAME, RUN_ID, RUN_NAME, dir_yaml_interpret_mid,
                                     test_loader_mid)
