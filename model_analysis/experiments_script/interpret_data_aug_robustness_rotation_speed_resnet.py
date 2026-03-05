""" This script investigates the features within the components of the trained neural network.
    The focus is on the models sensitivity with respect to pixel-wise data augmentation of unseen samples. """

import os
import argparse
from utils.env_config import setup_environment
import timm
from data.dataloader import prepare_dataloaders
from model_analysis.interpret_model import execute_class_activation_mapping
from model_analysis.analysis import execute_inference
from utils.mlflow_custom_functions import get_mlflow_experiment, get_mlflow_run
import albumentations as A


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--system_mode", type=str, default="local-H-Drive")
    args = parser.parse_args()
    setup_environment(args.system_mode)

    model_name = "resnet18.tv_in1k"
    experimental_properties = "resnet_18_tv_pixelwise_data_aug"

    EXPERIMENT_NAME = "robustness_rotation_speed"
    EXPERIMENT_ID = get_mlflow_experiment(EXPERIMENT_NAME)

    dir_python_script = os.path.join(os.environ.get("CODE_PATH_ANALYSIS"), "interpret_data_aug_robustness_rotation_speed_resnet.py")

    dir_yaml_interpret_low = os.path.join(os.environ.get("YAML_PATH_ANALYSIS"), "robustness_rotation_speed_low_interpret_vb06_data_aug_resnet.yaml")

    dir_yaml_infer_low = os.path.join(os.environ.get("YAML_PATH_ANALYSIS"), "robustness_rotation_speed_low_train_low_infer_vb06_resnet.yaml")


    # NOTE: the yaml files contain both the dataset preparation config and the model inference config of the labeled
    # parameter range i.e. low or high. If it does not contain model inference data, then it has the data preparation!

    # Run 0: log python script of experiment as artifact
    RUN_NAME = "log_python_script_of_experiment"
    get_mlflow_run(EXPERIMENT_ID, RUN_NAME, config_path_save=dir_python_script)

    # Run 0.1: initialize vb06 DataLoaders (lower end of parameter distribution)
    transform = None  # This run is for comparison with non-augmented data
    partition_loaders_no_data_aug = prepare_dataloaders(dir_yaml_interpret_low, EXPERIMENT_ID, EXPERIMENT_NAME, None, None, transform=transform)
    # run_id and run_name are ignored because this preparation of data is not logged in mlflow

    # Run 1: initialize vb06 DataLoaders (lower end of parameter distribution)
    transform_gauss_noise_1 = A.Compose([A.GaussNoise(var_limit=(0.01, 0.01), p=1)])
    partition_loaders_low_gauss_noise_1 = prepare_dataloaders(dir_yaml_interpret_low, EXPERIMENT_ID, EXPERIMENT_NAME, None, None, transform=transform_gauss_noise_1)
    # run_id and run_name are ignored because this preparation of data is not logged in mlflow

    # Run 2: initialize vb06 DataLoaders (lower end of parameter distribution)
    transform_gauss_noise_2 = A.Compose([A.GaussNoise(var_limit=(0.5, 0.5), p=1)])
    partition_loaders_low_gauss_noise_2 = prepare_dataloaders(dir_yaml_interpret_low, EXPERIMENT_ID, EXPERIMENT_NAME, None, None, transform=transform_gauss_noise_2)
    # run_id and run_name are ignored because this preparation of data is not logged in mlflow

    # Run 1: initialize vb06 DataLoaders (lower end of parameter distribution)
    transform_gauss_blur_1 = A.Compose([A.GaussianBlur(blur_limit=(5, 5), p=1)])
    partition_loaders_low_gauss_blur_1 = prepare_dataloaders(dir_yaml_interpret_low, EXPERIMENT_ID, EXPERIMENT_NAME, None, None, transform=transform_gauss_blur_1)
    # run_id and run_name are ignored because this preparation of data is not logged in mlflow

    # Run 2: initialize vb06 DataLoaders (lower end of parameter distribution)
    transform_gauss_blur_2 = A.Compose([A.GaussianBlur(blur_limit=(13, 13), p=1)])
    partition_loaders_low_gauss_blur_2 = prepare_dataloaders(dir_yaml_interpret_low, EXPERIMENT_ID, EXPERIMENT_NAME, None, None, transform=transform_gauss_blur_2)
    # run_id and run_name are ignored because this preparation of data is not logged in mlflow

    # Run 1: initialize vb06 DataLoaders (lower end of parameter distribution)
    transform_defocus_1 = A.Compose([A.Defocus(radius=(3, 3), alias_blur=(0.1, 0.9), p=1)])
    partition_loaders_low_defocus_1 = prepare_dataloaders(dir_yaml_interpret_low, EXPERIMENT_ID, EXPERIMENT_NAME, None, None, transform=transform_defocus_1)
    # run_id and run_name are ignored because this preparation of data is not logged in mlflow

    # Run 2: initialize vb06 DataLoaders (lower end of parameter distribution)
    transform_defocus_2 = A.Compose([A.Defocus(radius=(11, 11), alias_blur=(0.1, 0.9), p=1)])
    partition_loaders_low_defocus_2 = prepare_dataloaders(dir_yaml_interpret_low, EXPERIMENT_ID, EXPERIMENT_NAME, None, None, transform=transform_defocus_2)
    # run_id and run_name are ignored because this preparation of data is not logged in mlflow

    # Run 1: initialize vb06 DataLoaders (lower end of parameter distribution)
    transform_sharpen_1 = A.Compose([A.Sharpen(alpha=(0.1, 0.1), lightness=(1, 1), p=1)])
    partition_loaders_low_sharpen_1 = prepare_dataloaders(dir_yaml_interpret_low, EXPERIMENT_ID, EXPERIMENT_NAME, None, None, transform=transform_sharpen_1)
    # run_id and run_name are ignored because this preparation of data is not logged in mlflow

    # Run 2: initialize vb06 DataLoaders (lower end of parameter distribution)
    transform_sharpen_2 = A.Compose([A.Sharpen(alpha=(0.7, 0.7), lightness=(1, 1), p=1)])
    partition_loaders_low_sharpen_2 = prepare_dataloaders(dir_yaml_interpret_low, EXPERIMENT_ID, EXPERIMENT_NAME, None, None, transform=transform_sharpen_2)
    # run_id and run_name are ignored because this preparation of data is not logged in mlflow

    # Run 1: initialize vb06 DataLoaders (lower end of parameter distribution)
    transform_downscale_1 = A.Compose([A.Downscale(scale_range=(0.2, 0.2), interpolation_pair={"upscale":0, "downscale":0}, p=1)])
    partition_loaders_low_downscale_1 = prepare_dataloaders(dir_yaml_interpret_low, EXPERIMENT_ID, EXPERIMENT_NAME, None, None, transform=transform_downscale_1)
    # run_id and run_name are ignored because this preparation of data is not logged in mlflow

    # Run 2: initialize vb06 DataLoaders (lower end of parameter distribution)
    transform_downscale_2 = A.Compose([A.Downscale(scale_range=(0.5, 0.5), interpolation_pair={"upscale":0, "downscale":0}, p=1)])
    partition_loaders_low_downscale_2 = prepare_dataloaders(dir_yaml_interpret_low, EXPERIMENT_ID, EXPERIMENT_NAME, None, None, transform=transform_downscale_2)
    # run_id and run_name are ignored because this preparation of data is not logged in mlflow


    model = timm.create_model(model_name, pretrained=False, num_classes=3,
                              in_chans=1)  # Must change structure outside of source code!

    ### NO DATA AUG ###
    # Run Baseline: visualize vb06 samples with pretrained model (lower end of parameter distribution)
    RUN_NAME = f"interpret_robustness_rotation_speed_low_vb06_{experimental_properties}_no_data_aug_baseline"
    RUN_ID = get_mlflow_run(EXPERIMENT_ID, RUN_NAME, config_path_save=dir_yaml_interpret_low)
    test_loader_low = partition_loaders_no_data_aug[2]
    execute_class_activation_mapping(EXPERIMENT_ID, EXPERIMENT_NAME, RUN_ID, RUN_NAME, dir_yaml_interpret_low,
                                     test_loader_low, model, model.layer4[1].conv2)

    # Run Baseline: infer lower end of parameter distribution on model from lower end of parameter distribution
    RUN_NAME = f"robustness_rotation_speed_low_train_low_infer_vb06_{experimental_properties}_no_data_aug_baseline"
    RUN_ID = get_mlflow_run(EXPERIMENT_ID, RUN_NAME, config_path_save=dir_yaml_infer_low)
    execute_inference(EXPERIMENT_ID, EXPERIMENT_NAME, RUN_ID, RUN_NAME, dir_yaml_infer_low, test_loader_low, model)


    ### GAUSS NOISE ###
    # Run Gauss Noise 1.1: visualize vb06 samples with pretrained model (lower end of parameter distribution)
    RUN_NAME = f"interpret_robustness_rotation_speed_low_vb06_{experimental_properties}_gauss_noise_1"
    RUN_ID = get_mlflow_run(EXPERIMENT_ID, RUN_NAME, config_path_save=dir_yaml_interpret_low)
    test_loader_low = partition_loaders_low_gauss_noise_1[2]
    execute_class_activation_mapping(EXPERIMENT_ID, EXPERIMENT_NAME, RUN_ID, RUN_NAME, dir_yaml_interpret_low,
                                     test_loader_low, model, model.layer4[1].conv2)

    # Run Gauss Noise 1.1: infer lower end of parameter distribution on model from lower end of parameter distribution
    RUN_NAME = f"robustness_rotation_speed_low_train_low_infer_vb06_{experimental_properties}_gauss_noise_1"
    RUN_ID = get_mlflow_run(EXPERIMENT_ID, RUN_NAME, config_path_save=dir_yaml_infer_low)
    execute_inference(EXPERIMENT_ID, EXPERIMENT_NAME, RUN_ID, RUN_NAME, dir_yaml_infer_low, test_loader_low, model)

    # Run Gauss Noise 1.2: visualize vb06 samples with pretrained model (lower end of parameter distribution)
    RUN_NAME = f"interpret_robustness_rotation_speed_low_vb06_{experimental_properties}_gauss_noise_2"
    RUN_ID = get_mlflow_run(EXPERIMENT_ID, RUN_NAME, config_path_save=dir_yaml_interpret_low)
    test_loader_low = partition_loaders_low_gauss_noise_2[2]
    execute_class_activation_mapping(EXPERIMENT_ID, EXPERIMENT_NAME, RUN_ID, RUN_NAME, dir_yaml_interpret_low,
                                     test_loader_low, model, model.layer4[1].conv2)

    # Run Gauss Noise 1.2: infer lower end of parameter distribution on model from lower end of parameter distribution
    RUN_NAME = f"robustness_rotation_speed_low_train_low_infer_vb06_{experimental_properties}_gauss_noise_2"
    RUN_ID = get_mlflow_run(EXPERIMENT_ID, RUN_NAME, config_path_save=dir_yaml_infer_low)
    execute_inference(EXPERIMENT_ID, EXPERIMENT_NAME, RUN_ID, RUN_NAME, dir_yaml_infer_low, test_loader_low, model)


    ### GAUSSIAN BLUR ###
    # Run Gaussian Blur 1.1: visualize vb06 samples with pretrained model (lower end of parameter distribution)
    RUN_NAME = f"interpret_robustness_rotation_speed_low_vb06_{experimental_properties}_gauss_blur_1"
    RUN_ID = get_mlflow_run(EXPERIMENT_ID, RUN_NAME, config_path_save=dir_yaml_interpret_low)
    test_loader_low = partition_loaders_low_gauss_blur_1[2]
    execute_class_activation_mapping(EXPERIMENT_ID, EXPERIMENT_NAME, RUN_ID, RUN_NAME, dir_yaml_interpret_low,
                                     test_loader_low, model, model.layer4[1].conv2)

    # Run Gaussian Blur 1.1: infer lower end of parameter distribution on model from lower end of parameter distribution
    RUN_NAME = f"robustness_rotation_speed_low_train_low_infer_vb06_{experimental_properties}_gauss_blur_1"
    RUN_ID = get_mlflow_run(EXPERIMENT_ID, RUN_NAME, config_path_save=dir_yaml_infer_low)
    execute_inference(EXPERIMENT_ID, EXPERIMENT_NAME, RUN_ID, RUN_NAME, dir_yaml_infer_low, test_loader_low, model)

    # Run Gaussian Blur 1.2: visualize vb06 samples with pretrained model (lower end of parameter distribution)
    RUN_NAME = f"interpret_robustness_rotation_speed_low_vb06_{experimental_properties}_gauss_blur_2"
    RUN_ID = get_mlflow_run(EXPERIMENT_ID, RUN_NAME, config_path_save=dir_yaml_interpret_low)
    test_loader_low = partition_loaders_low_gauss_blur_2[2]
    execute_class_activation_mapping(EXPERIMENT_ID, EXPERIMENT_NAME, RUN_ID, RUN_NAME, dir_yaml_interpret_low,
                                     test_loader_low, model, model.layer4[1].conv2)

    # Run Gaussian Blur 1.2: infer lower end of parameter distribution on model from lower end of parameter distribution
    RUN_NAME = f"robustness_rotation_speed_low_train_low_infer_vb06_{experimental_properties}_gauss_blur_2"
    RUN_ID = get_mlflow_run(EXPERIMENT_ID, RUN_NAME, config_path_save=dir_yaml_infer_low)
    execute_inference(EXPERIMENT_ID, EXPERIMENT_NAME, RUN_ID, RUN_NAME, dir_yaml_infer_low, test_loader_low, model)


    ### DEFOCUS ###
    # Run Defocus 1.1: visualize vb06 samples with pretrained model (lower end of parameter distribution)
    RUN_NAME = f"interpret_robustness_rotation_speed_low_vb06_{experimental_properties}_defocus_1"
    RUN_ID = get_mlflow_run(EXPERIMENT_ID, RUN_NAME, config_path_save=dir_yaml_interpret_low)
    test_loader_low = partition_loaders_low_defocus_1[2]
    execute_class_activation_mapping(EXPERIMENT_ID, EXPERIMENT_NAME, RUN_ID, RUN_NAME, dir_yaml_interpret_low,
                                     test_loader_low, model, model.layer4[1].conv2)

    # Run Defocus 1.1: infer lower end of parameter distribution on model from lower end of parameter distribution
    RUN_NAME = f"robustness_rotation_speed_low_train_low_infer_vb06_{experimental_properties}_defocus_1"
    RUN_ID = get_mlflow_run(EXPERIMENT_ID, RUN_NAME, config_path_save=dir_yaml_infer_low)
    execute_inference(EXPERIMENT_ID, EXPERIMENT_NAME, RUN_ID, RUN_NAME, dir_yaml_infer_low, test_loader_low, model)

    # Run Defocus 1.2: visualize vb06 samples with pretrained model (lower end of parameter distribution)
    RUN_NAME = f"interpret_robustness_rotation_speed_low_vb06_{experimental_properties}_defocus_2"
    RUN_ID = get_mlflow_run(EXPERIMENT_ID, RUN_NAME, config_path_save=dir_yaml_interpret_low)
    test_loader_low = partition_loaders_low_defocus_2[2]
    execute_class_activation_mapping(EXPERIMENT_ID, EXPERIMENT_NAME, RUN_ID, RUN_NAME, dir_yaml_interpret_low,
                                     test_loader_low, model, model.layer4[1].conv2)

    # Run Defocus 1.2: infer lower end of parameter distribution on model from lower end of parameter distribution
    RUN_NAME = f"robustness_rotation_speed_low_train_low_infer_vb06_{experimental_properties}_defocus_2"
    RUN_ID = get_mlflow_run(EXPERIMENT_ID, RUN_NAME, config_path_save=dir_yaml_infer_low)
    execute_inference(EXPERIMENT_ID, EXPERIMENT_NAME, RUN_ID, RUN_NAME, dir_yaml_infer_low, test_loader_low, model)


    ### SHARPEN ###
    # Run Sharpen 1.1: visualize vb06 samples with pretrained model (lower end of parameter distribution)
    RUN_NAME = f"interpret_robustness_rotation_speed_low_vb06_{experimental_properties}_sharpen_1"
    RUN_ID = get_mlflow_run(EXPERIMENT_ID, RUN_NAME, config_path_save=dir_yaml_interpret_low)
    test_loader_low = partition_loaders_low_sharpen_1[2]
    execute_class_activation_mapping(EXPERIMENT_ID, EXPERIMENT_NAME, RUN_ID, RUN_NAME, dir_yaml_interpret_low,
                                     test_loader_low, model, model.layer4[1].conv2)

    # Run Sharpen 1.1: infer lower end of parameter distribution on model from lower end of parameter distribution
    RUN_NAME = f"robustness_rotation_speed_low_train_low_infer_vb06_{experimental_properties}_sharpen_1"
    RUN_ID = get_mlflow_run(EXPERIMENT_ID, RUN_NAME, config_path_save=dir_yaml_infer_low)
    execute_inference(EXPERIMENT_ID, EXPERIMENT_NAME, RUN_ID, RUN_NAME, dir_yaml_infer_low, test_loader_low, model)

    # Run Sharpen 1.2: visualize vb06 samples with pretrained model (lower end of parameter distribution)
    RUN_NAME = f"interpret_robustness_rotation_speed_low_vb06_{experimental_properties}_sharpen_2"
    RUN_ID = get_mlflow_run(EXPERIMENT_ID, RUN_NAME, config_path_save=dir_yaml_interpret_low)
    test_loader_low = partition_loaders_low_sharpen_2[2]
    execute_class_activation_mapping(EXPERIMENT_ID, EXPERIMENT_NAME, RUN_ID, RUN_NAME, dir_yaml_interpret_low,
                                     test_loader_low, model, model.layer4[1].conv2)

    # Run Sharpen 1.2: infer lower end of parameter distribution on model from lower end of parameter distribution
    RUN_NAME = f"robustness_rotation_speed_low_train_low_infer_vb06_{experimental_properties}_sharpen_2"
    RUN_ID = get_mlflow_run(EXPERIMENT_ID, RUN_NAME, config_path_save=dir_yaml_infer_low)
    execute_inference(EXPERIMENT_ID, EXPERIMENT_NAME, RUN_ID, RUN_NAME, dir_yaml_infer_low, test_loader_low, model)


    ### DOWNSCALE ###
    # Run Downscale 1.1: visualize vb06 samples with pretrained model (lower end of parameter distribution)
    RUN_NAME = f"interpret_robustness_rotation_speed_low_vb06_{experimental_properties}_downscale_1"
    RUN_ID = get_mlflow_run(EXPERIMENT_ID, RUN_NAME, config_path_save=dir_yaml_interpret_low)
    test_loader_low = partition_loaders_low_downscale_1[2]
    execute_class_activation_mapping(EXPERIMENT_ID, EXPERIMENT_NAME, RUN_ID, RUN_NAME, dir_yaml_interpret_low,
                                     test_loader_low, model, model.layer4[1].conv2)

    # Run Downscale 1.1: infer lower end of parameter distribution on model from lower end of parameter distribution
    RUN_NAME = f"robustness_rotation_speed_low_train_low_infer_vb06_{experimental_properties}_downscale_1"
    RUN_ID = get_mlflow_run(EXPERIMENT_ID, RUN_NAME, config_path_save=dir_yaml_infer_low)
    execute_inference(EXPERIMENT_ID, EXPERIMENT_NAME, RUN_ID, RUN_NAME, dir_yaml_infer_low, test_loader_low, model)

    # Run Downscale 1.2: visualize vb06 samples with pretrained model (lower end of parameter distribution)
    RUN_NAME = f"interpret_robustness_rotation_speed_low_vb06_{experimental_properties}_downscale_2"
    RUN_ID = get_mlflow_run(EXPERIMENT_ID, RUN_NAME, config_path_save=dir_yaml_interpret_low)
    test_loader_low = partition_loaders_low_downscale_2[2]
    execute_class_activation_mapping(EXPERIMENT_ID, EXPERIMENT_NAME, RUN_ID, RUN_NAME, dir_yaml_interpret_low,
                                     test_loader_low, model, model.layer4[1].conv2)

    # Run Downscale 1.2: infer lower end of parameter distribution on model from lower end of parameter distribution
    RUN_NAME = f"robustness_rotation_speed_low_train_low_infer_vb06_{experimental_properties}_downscale_2"
    RUN_ID = get_mlflow_run(EXPERIMENT_ID, RUN_NAME, config_path_save=dir_yaml_infer_low)
    execute_inference(EXPERIMENT_ID, EXPERIMENT_NAME, RUN_ID, RUN_NAME, dir_yaml_infer_low, test_loader_low, model)
