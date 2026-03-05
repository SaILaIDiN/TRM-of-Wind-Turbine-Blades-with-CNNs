""" This script investigates the capability of the CNN to correctly classify samples of rotor blades from previously
    unseen values of a certain meta parameter. This version's cropping tries to avoid the radar echo to be included.
"""
import os
import argparse
from utils.env_config import setup_environment
import timm
from data.dataloader import prepare_dataloaders, prepare_dataset_partitions, concat_dataset_partitions, \
    create_dataloaders
from model_analysis.analysis import execute_optimization, execute_inference
from utils.mlflow_custom_functions import get_mlflow_experiment, get_mlflow_run
import albumentations as A


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--system_mode", type=str, default="local-H-Drive")
    args = parser.parse_args()
    setup_environment(args.system_mode)

    net_type = "resnet_18_tv_crop"
    model_name = "resnet18.tv_in1k"

    EXPERIMENT_NAME = "robustness_rotation_speed"
    EXPERIMENT_ID = get_mlflow_experiment(EXPERIMENT_NAME)

    dir_python_script = os.path.join(
        os.environ.get("CODE_PATH_ANALYSIS"), "analysis_robustness_rotation_speed_resnet.py")

    dir_yaml_data_low = os.path.join(
        os.environ.get("YAML_PATH_ANALYSIS"), "robustness_rotation_speed_low_prep_data_vb06_resnet.yaml")
    dir_yaml_data_high = os.path.join(
        os.environ.get("YAML_PATH_ANALYSIS"), "robustness_rotation_speed_high_prep_data_vb06_resnet.yaml")
    dir_yaml_train_low = os.path.join(
        os.environ.get("YAML_PATH_ANALYSIS"), "robustness_rotation_speed_low_train_vb06_resnet.yaml")
    dir_yaml_train_high = os.path.join(
        os.environ.get("YAML_PATH_ANALYSIS"), "robustness_rotation_speed_high_train_vb06_resnet.yaml")
    dir_yaml_infer_high = os.path.join(
        os.environ.get("YAML_PATH_ANALYSIS"), "robustness_rotation_speed_low_train_high_infer_vb06_resnet_crop.yaml")
    dir_yaml_infer_low = os.path.join(
        os.environ.get("YAML_PATH_ANALYSIS"), "robustness_rotation_speed_high_train_low_infer_vb06_resnet_crop.yaml")

    dir_yaml_train_low_and_high = os.path.join(
        os.environ.get("YAML_PATH_ANALYSIS"), "robustness_rotation_speed_low_and_high_train_vb06_resnet.yaml")
    dir_yaml_data_mid = os.path.join(
        os.environ.get("YAML_PATH_ANALYSIS"), "robustness_rotation_speed_mid_prep_data_vb06_resnet.yaml")
    dir_yaml_infer_mid = os.path.join(
        os.environ.get("YAML_PATH_ANALYSIS"), "robustness_rotation_speed_low_and_high_train_mid_infer_vb06_resnet_crop.yaml")

    dir_yaml_train_low_and_high_pretrained = os.path.join(
        os.environ.get("YAML_PATH_ANALYSIS"), "robustness_rotation_speed_low_and_high_train_vb06_resnet_pretrained.yaml")
    dir_yaml_infer_mid_pretrained = os.path.join(
        os.environ.get("YAML_PATH_ANALYSIS"), "robustness_rotation_speed_low_and_high_train_mid_infer_vb06_resnet_pretrained_crop.yaml")

    # Run 0: log python script of experiment as artifact
    RUN_NAME = "log_python_script_of_experiment"
    get_mlflow_run(EXPERIMENT_ID, RUN_NAME, config_path_save=dir_python_script)

    # Run 1: initialize vb06 DataLoaders (lower end of parameter distribution)
    RUN_NAME = f"robustness_rotation_speed_low_prep_data_vb06_{net_type}"
    RUN_ID = get_mlflow_run(EXPERIMENT_ID, RUN_NAME, config_path_save=dir_yaml_data_low)
    transform_cropping = A.Compose([A.Crop(y_min=150, x_min=0, y_max=600, x_max=250), A.RandomCrop(height=224, width=224)])
    partition_loaders_low = prepare_dataloaders(dir_yaml_data_low, EXPERIMENT_ID, EXPERIMENT_NAME, RUN_ID, RUN_NAME, transform=transform_cropping)

    # Run 2: initialize vb06 DataLoaders (higher end of parameter distribution)
    RUN_NAME = f"robustness_rotation_speed_high_prep_data_vb06_{net_type}"
    RUN_ID = get_mlflow_run(EXPERIMENT_ID, RUN_NAME, config_path_save=dir_yaml_data_high)
    transform_cropping = A.Compose([A.Crop(y_min=150, x_min=0, y_max=600, x_max=250), A.RandomCrop(height=224, width=224)])
    partition_loaders_high = prepare_dataloaders(dir_yaml_data_high, EXPERIMENT_ID, EXPERIMENT_NAME, RUN_ID, RUN_NAME, transform=transform_cropping)

    # Run 3: optimize model with dataset (lower end of parameter distribution)
    RUN_NAME = f"robustness_rotation_speed_low_train_vb06_{net_type}"
    RUN_ID = get_mlflow_run(EXPERIMENT_ID, RUN_NAME, config_path_save=dir_yaml_train_low)
    model = timm.create_model(model_name, pretrained=False, num_classes=3,
                              in_chans=1)  # Must change structure outside of source code!
    execute_optimization(EXPERIMENT_ID, EXPERIMENT_NAME, RUN_ID, RUN_NAME, dir_yaml_train_low, partition_loaders_low, model)

    # Run 4: optimize model with dataset (higher end of parameter distribution)
    RUN_NAME = f"robustness_rotation_speed_high_train_vb06_{net_type}"
    RUN_ID = get_mlflow_run(EXPERIMENT_ID, RUN_NAME, config_path_save=dir_yaml_train_high)
    model = timm.create_model(model_name, pretrained=False, num_classes=3,
                              in_chans=1)  # Must change structure outside of source code!
    execute_optimization(EXPERIMENT_ID, EXPERIMENT_NAME, RUN_ID, RUN_NAME, dir_yaml_train_high, partition_loaders_high, model)

    # Run 5: infer higher end of parameter distribution on model from lower end of parameter distribution
    RUN_NAME = f"robustness_rotation_speed_low_train_high_infer_vb06_{net_type}"
    RUN_ID = get_mlflow_run(EXPERIMENT_ID, RUN_NAME, config_path_save=dir_yaml_infer_high)
    test_loader_high = partition_loaders_high[0]
    model = timm.create_model(model_name, pretrained=False, num_classes=3,
                              in_chans=1)  # Must change structure outside of source code!
    execute_inference(EXPERIMENT_ID, EXPERIMENT_NAME, RUN_ID, RUN_NAME, dir_yaml_infer_high, test_loader_high, model)

    # Run 6: infer lower end of parameter distribution on model from higher end of parameter distribution
    RUN_NAME = f"robustness_rotation_speed_high_train_low_infer_vb06_{net_type}"
    RUN_ID = get_mlflow_run(EXPERIMENT_ID, RUN_NAME, config_path_save=dir_yaml_infer_low)
    test_loader_low = partition_loaders_low[0]
    model = timm.create_model(model_name, pretrained=False, num_classes=3,
                              in_chans=1)  # Must change structure outside of source code!
    execute_inference(EXPERIMENT_ID, EXPERIMENT_NAME, RUN_ID, RUN_NAME, dir_yaml_infer_low, test_loader_low, model)

    # Run 7: optimize model with dataset (lower and higher end of parameter distribution)
    RUN_NAME = f"robustness_rotation_speed_low_and_high_train_vb06_{net_type}"
    RUN_ID = get_mlflow_run(EXPERIMENT_ID, RUN_NAME, config_path_save=dir_yaml_train_low_and_high)
    transform_cropping = A.Compose([A.Crop(y_min=150, x_min=0, y_max=600, x_max=250), A.RandomCrop(height=224, width=224)])
    partition_datasets_low = prepare_dataset_partitions(dir_yaml_data_low, EXPERIMENT_ID, EXPERIMENT_NAME, RUN_ID, RUN_NAME, transform=transform_cropping)
    partition_datasets_high = prepare_dataset_partitions(dir_yaml_data_high, EXPERIMENT_ID, EXPERIMENT_NAME, RUN_ID, RUN_NAME, transform=transform_cropping)
    new_partitions = concat_dataset_partitions(partition_datasets_low, partition_datasets_high)
    partition_loaders_low_and_high = create_dataloaders(
        new_partitions[0], new_partitions[1], new_partitions[2], batch_size=16, num_workers=2, torch_seed=42)
    model = timm.create_model(model_name, pretrained=False, num_classes=3,
                              in_chans=1)  # Must change structure outside of source code!
    execute_optimization(
        EXPERIMENT_ID, EXPERIMENT_NAME, RUN_ID, RUN_NAME, dir_yaml_train_low_and_high, partition_loaders_low_and_high, model)

    # Run 8: infer middle of parameter distribution on model from lower and higher end of parameter distribution
    RUN_NAME = f"robustness_rotation_speed_low_and_high_train_mid_infer_vb06_{net_type}"
    RUN_ID = get_mlflow_run(EXPERIMENT_ID, RUN_NAME, config_path_save=dir_yaml_infer_mid)
    transform_cropping = A.Compose([A.Crop(y_min=150, x_min=0, y_max=600, x_max=250), A.RandomCrop(height=224, width=224)])
    partition_loaders_mid = prepare_dataloaders(dir_yaml_data_mid, EXPERIMENT_ID, EXPERIMENT_NAME, RUN_ID, RUN_NAME, transform=transform_cropping)
    model = timm.create_model(model_name, pretrained=False, num_classes=3,
                              in_chans=1)  # Must change structure outside of source code!
    execute_inference(EXPERIMENT_ID, EXPERIMENT_NAME, RUN_ID, RUN_NAME, dir_yaml_infer_mid, partition_loaders_mid[0], model)

    # Run 9: optimize model with dataset (lower and higher end of parameter distribution) (with ImageNet weights)
    RUN_NAME = f"robustness_rotation_speed_low_and_high_train_vb06_{net_type}_pretrained"
    RUN_ID = get_mlflow_run(EXPERIMENT_ID, RUN_NAME, config_path_save=dir_yaml_train_low_and_high_pretrained)
    transform_cropping = A.Compose([A.Crop(y_min=150, x_min=0, y_max=600, x_max=250), A.RandomCrop(height=224, width=224)])
    partition_datasets_low = prepare_dataset_partitions(dir_yaml_data_low, EXPERIMENT_ID, EXPERIMENT_NAME, RUN_ID, RUN_NAME, transform=transform_cropping)
    partition_datasets_high = prepare_dataset_partitions(dir_yaml_data_high, EXPERIMENT_ID, EXPERIMENT_NAME, RUN_ID, RUN_NAME, transform=transform_cropping)
    new_partitions = concat_dataset_partitions(partition_datasets_low, partition_datasets_high)
    partition_loaders_low_and_high = create_dataloaders(
        new_partitions[0], new_partitions[1], new_partitions[2], batch_size=16, num_workers=2, torch_seed=42)
    model = timm.create_model(model_name, pretrained=True, num_classes=3,
                              in_chans=1)  # Must change structure outside of source code for fine-tuning!
    execute_optimization(
        EXPERIMENT_ID, EXPERIMENT_NAME, RUN_ID, RUN_NAME, dir_yaml_train_low_and_high_pretrained, partition_loaders_low_and_high, model)

    # Run 10: infer middle of parameter distribution on model from lower and higher end of parameter distribution (with ImageNet weights)
    RUN_NAME = f"robustness_rotation_speed_low_and_high_train_mid_infer_vb06_{net_type}_pretrained"
    RUN_ID = get_mlflow_run(EXPERIMENT_ID, RUN_NAME, config_path_save=dir_yaml_infer_mid_pretrained)
    transform_cropping = A.Compose([A.Crop(y_min=150, x_min=0, y_max=600, x_max=250), A.RandomCrop(height=224, width=224)])
    partition_loaders_mid = prepare_dataloaders(dir_yaml_data_mid, EXPERIMENT_ID, EXPERIMENT_NAME, RUN_ID, RUN_NAME, transform=transform_cropping)
    model = timm.create_model(model_name, pretrained=False, num_classes=3,
                              in_chans=1)  # Must change structure outside of source code for fine-tuning!
    execute_inference(
        EXPERIMENT_ID, EXPERIMENT_NAME, RUN_ID, RUN_NAME, dir_yaml_infer_mid_pretrained, partition_loaders_mid[0], model)
