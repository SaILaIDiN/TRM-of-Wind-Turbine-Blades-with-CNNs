""" This script investigates the features within the components of the trained neural network.
    This version's cropping tries to avoid the radar echo to be included.
"""

import os
import argparse
from utils.env_config import setup_environment
import timm
from data.dataloader import prepare_dataloaders
from model_analysis.interpret_model import execute_class_activation_mapping
from utils.mlflow_custom_functions import get_mlflow_experiment, get_mlflow_run
import albumentations as A


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--system_mode", type=str, default="local-H-Drive")
    args = parser.parse_args()
    setup_environment(args.system_mode)

    net_type = "resnet_18_tv"
    model_name = "resnet18.tv_in1k"

    EXPERIMENT_NAME = "robustness_rotation_speed"
    EXPERIMENT_ID = get_mlflow_experiment(EXPERIMENT_NAME)

    dir_python_script = os.path.join(os.environ.get("CODE_PATH_ANALYSIS"), "interpret_robustness_rotation_speed_resnet.py")

    dir_yaml_interpret_low = os.path.join(os.environ.get("YAML_PATH_ANALYSIS"), "robustness_rotation_speed_low_interpret_vb06_resnet.yaml")
    dir_yaml_interpret_high = os.path.join(os.environ.get("YAML_PATH_ANALYSIS"), "robustness_rotation_speed_high_interpret_vb06_resnet.yaml")
    dir_yaml_interpret_mid = os.path.join(os.environ.get("YAML_PATH_ANALYSIS"), "robustness_rotation_speed_mid_interpret_vb06_resnet.yaml")

    dir_yaml_interpret_low_pretrained = os.path.join(os.environ.get("YAML_PATH_ANALYSIS"), "robustness_rotation_speed_low_interpret_vb06_resnet_pretrained.yaml")
    dir_yaml_interpret_high_pretrained = os.path.join(os.environ.get("YAML_PATH_ANALYSIS"), "robustness_rotation_speed_high_interpret_vb06_resnet_pretrained.yaml")
    dir_yaml_interpret_mid_pretrained = os.path.join(os.environ.get("YAML_PATH_ANALYSIS"), "robustness_rotation_speed_mid_interpret_vb06_resnet_pretrained.yaml")

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
    RUN_NAME = f"interpret_robustness_rotation_speed_low_vb06_{net_type}"
    RUN_ID = get_mlflow_run(EXPERIMENT_ID, RUN_NAME, config_path_save=dir_yaml_interpret_low)
    test_loader_low = partition_loaders_low[0]
    model = timm.create_model(model_name, pretrained=False, num_classes=3,
                              in_chans=1)  # Must change structure outside of source code!
    execute_class_activation_mapping(EXPERIMENT_ID, EXPERIMENT_NAME, RUN_ID, RUN_NAME, dir_yaml_interpret_low,
                                     test_loader_low, model, model.layer4[1].conv2)

    # Run 5: visualize vb06 samples with pretrained model (higher end of parameter distribution)
    RUN_NAME = f"interpret_robustness_rotation_speed_high_vb06_{net_type}"
    RUN_ID = get_mlflow_run(EXPERIMENT_ID, RUN_NAME, config_path_save=dir_yaml_interpret_high)
    test_loader_high = partition_loaders_high[0]
    model = timm.create_model(model_name, pretrained=False, num_classes=3,
                              in_chans=1)  # Must change structure outside of source code!
    execute_class_activation_mapping(EXPERIMENT_ID, EXPERIMENT_NAME, RUN_ID, RUN_NAME, dir_yaml_interpret_high,
                                     test_loader_high, model, model.layer4[1].conv2)

    # Run 6: visualize vb06 samples from lower end of parameter distribution on model trained with higher end
    RUN_NAME = f"interpret_robustness_rotation_speed_low_train_high_infer_vb06_{net_type}"
    RUN_ID = get_mlflow_run(EXPERIMENT_ID, RUN_NAME, config_path_save=[dir_yaml_interpret_low, dir_yaml_interpret_high])
    model = timm.create_model(model_name, pretrained=False, num_classes=3,
                              in_chans=1)  # Must change structure outside of source code!
    execute_class_activation_mapping(EXPERIMENT_ID, EXPERIMENT_NAME, RUN_ID, RUN_NAME, dir_yaml_interpret_low,
                                     test_loader_high, model, model.layer4[1].conv2)

    # Run 7: visualize vb06 samples from higher end of parameter distribution on model trained with lower end
    RUN_NAME = f"interpret_robustness_rotation_speed_high_train_low_infer_vb06_{net_type}"
    RUN_ID = get_mlflow_run(EXPERIMENT_ID, RUN_NAME, config_path_save=[dir_yaml_interpret_high, dir_yaml_interpret_low])
    model = timm.create_model(model_name, pretrained=False, num_classes=3,
                              in_chans=1)  # Must change structure outside of source code!
    execute_class_activation_mapping(EXPERIMENT_ID, EXPERIMENT_NAME, RUN_ID, RUN_NAME, dir_yaml_interpret_high,
                                     test_loader_low, model, model.layer4[1].conv2)

    # Run 8: visualize vb06 samples from middle of parameter distribution on model trained with low and high ends
    RUN_NAME = f"interpret_robustness_rotation_speed_low_and_high_train_mid_infer_vb06_{net_type}"
    RUN_ID = get_mlflow_run(EXPERIMENT_ID, RUN_NAME, config_path_save=dir_yaml_interpret_mid)
    test_loader_mid = partition_loaders_mid[0]
    model = timm.create_model(model_name, pretrained=False, num_classes=3,
                              in_chans=1)  # Must change structure outside of source code!
    execute_class_activation_mapping(EXPERIMENT_ID, EXPERIMENT_NAME, RUN_ID, RUN_NAME, dir_yaml_interpret_mid,
                                     test_loader_mid, model, model.layer4[1].conv2)

    # Run 9: visualize vb06 samples from lower end of parameter distribution on model trained with low and high ends (pretrained with ImageNet weights)
    RUN_NAME = f"interpret_robustness_rotation_speed_low_and_high_train_low_infer_vb06_{net_type}_pretrained"
    RUN_ID = get_mlflow_run(EXPERIMENT_ID, RUN_NAME, config_path_save=dir_yaml_interpret_low_pretrained)
    test_loader_low = partition_loaders_low[0]
    model = timm.create_model(model_name, pretrained=False, num_classes=3,
                              in_chans=1)  # Must change structure outside of source code!
    execute_class_activation_mapping(EXPERIMENT_ID, EXPERIMENT_NAME, RUN_ID, RUN_NAME, dir_yaml_interpret_low_pretrained,
                                     test_loader_low, model, model.layer4[1].conv2)

    # Run 10: visualize vb06 samples from higher end of parameter distribution on model trained with low and high ends (pretrained with ImageNet weights)
    RUN_NAME = f"interpret_robustness_rotation_speed_low_and_high_train_high_infer_vb06_{net_type}_pretrained"
    RUN_ID = get_mlflow_run(EXPERIMENT_ID, RUN_NAME, config_path_save=dir_yaml_interpret_high_pretrained)
    test_loader_high = partition_loaders_high[0]
    model = timm.create_model(model_name, pretrained=False, num_classes=3,
                              in_chans=1)  # Must change structure outside of source code!
    execute_class_activation_mapping(EXPERIMENT_ID, EXPERIMENT_NAME, RUN_ID, RUN_NAME, dir_yaml_interpret_high_pretrained,
                                     test_loader_high, model, model.layer4[1].conv2)

    # Run 11: visualize vb06 samples from middle of parameter distribution on model trained with low and high ends (pretrained with ImageNet weights)
    RUN_NAME = f"interpret_robustness_rotation_speed_low_and_high_train_mid_infer_vb06_{net_type}_pretrained"
    RUN_ID = get_mlflow_run(EXPERIMENT_ID, RUN_NAME, config_path_save=dir_yaml_interpret_mid_pretrained)
    test_loader_mid = partition_loaders_mid[0]
    model = timm.create_model(model_name, pretrained=False, num_classes=3,
                              in_chans=1)  # Must change structure outside of source code!
    execute_class_activation_mapping(EXPERIMENT_ID, EXPERIMENT_NAME, RUN_ID, RUN_NAME, dir_yaml_interpret_mid_pretrained,
                                     test_loader_mid, model, model.layer4[1].conv2)
