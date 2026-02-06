import os
import time
import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from model_optim.validate import eval_model
from utils.metrics import standard_metrics, prediction_refactoring, label_refactoring, compute_confusion_matrix
from utils.env_config import path_check_and_join, path_dissect_and_join


def optimize_model(train_loader, val_loader, test_loader, model, n_metadata, n_classes, epochs, name_optim,
                   lr, momentum, device, experiment_id, experiment_name, run_id, run_name=None, save_model_path=None,
                   checkpointing=None, results_path=None, eval_val=None, eval_train=None, eval_test=None):

    with mlflow.start_run(experiment_id=experiment_id, run_id=run_id):

        # Define loss and optimizer
        criterion = nn.BCEWithLogitsLoss()
        if name_optim == "SGD":
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        else:
            optimizer = optim.Adam(model.parameters(), lr=lr)

        # Optimization loop
        for epoch in range(epochs):
            start_time = time.time()
            model.train()
            total_train_loss = 0.0
            iteration_counter = 0
            for i, batch in enumerate(train_loader):
                images = batch["image"].unsqueeze(1).to(device, dtype=torch.float32)
                labels = batch["label"].to(device, dtype=torch.float32)
                metadata = batch["metadata"].to(device, dtype=torch.float32) \
                    if isinstance(n_metadata, int) and n_metadata > 0 else None

                optimizer.zero_grad()

                if metadata is not None:
                    outputs = model(images, metadata)
                else:
                    outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()
                iteration_counter = i
            print("Training loss: ", total_train_loss / iteration_counter)

            end_time = time.time()
            # Calculate the execution time
            execution_time = end_time - start_time
            print(f"Execution time: {execution_time} seconds")

            # Evaluate Val
            if eval_val == "True":
                mode = "Val"
                eval_dict_val = help_evaluate_dataset(val_loader, mode, model, n_metadata, n_classes, epoch, device,
                                                      results_path, experiment_name, run_name)
                # MLFLOW logging
                help_mlflow_logging(eval_dict=eval_dict_val, epoch=epoch)

            # Evaluate Train
            if eval_train == "True":
                mode = "Train"
                eval_dict_train = help_evaluate_dataset(train_loader, mode, model, n_metadata, n_classes, epoch, device,
                                                        results_path, experiment_name, run_name)
                # MLFLOW logging
                help_mlflow_logging(eval_dict=eval_dict_train, epoch=epoch)

            # Evaluate Test
            if eval_test == "True":
                mode = "Test"
                eval_dict_test = help_evaluate_dataset(test_loader, mode, model, n_metadata, n_classes, epoch, device,
                                                       results_path, experiment_name, run_name)
                # MLFLOW logging
                help_mlflow_logging(eval_dict=eval_dict_test, epoch=epoch)

            if save_model_path and checkpointing == "True":
                save_model_path_tmp = path_check_and_join(save_model_path, [experiment_name, run_name,
                                                          f"epoch_{epoch}.pth"], end_is_file=True)
                torch.save(model.state_dict(), save_model_path_tmp)

        if save_model_path:
            save_model_path = path_check_and_join(save_model_path, [experiment_name, run_name, "final_weights.pth"],
                                                  end_is_file=True)
            torch.save(model.state_dict(), save_model_path)


def help_evaluate_dataset(data_loader, mode, model, n_metadata, n_classes, epoch, device, results_path,
                          experiment_name, run_name):
    """ This helper function performs the main evaluation of the models.
        It separately stores confusion matrices apart from the MLflow Logger.
        NOTE: Customized function for metrics of three-class rotor blade classification and binary anomaly detection
        of a single rotor blade.
    """
    eval_dict = {}
    # Evaluate dataset
    total_loss_avg, pred_collector, labels_collector = eval_model(data_loader, model, n_metadata, device)
    print(f"{mode} loss: ", total_loss_avg)

    # Evaluate with performance metrics
    pred_ref = prediction_refactoring(pred_collector, n_classes=n_classes)
    labels_ref = label_refactoring(labels_collector, n_classes=n_classes)

    if n_classes == 3:
        _, _, _, _, f1_score_class_based = standard_metrics(labels_ref, pred_ref, class_mode="multi")
        print(f"F1-Score (class-based, {mode} set): ", f1_score_class_based)
        f1_score_class_based_dict = {f"performance_metrics/{mode}/f1_score/ROT1": f1_score_class_based[0],
                                     f"performance_metrics/{mode}/f1_score/ROT2": f1_score_class_based[1],
                                     f"performance_metrics/{mode}/f1_score/ROT3": f1_score_class_based[2]}
        conf_mat_path = path_check_and_join(results_path, [experiment_name, run_name, "confusion_matrix", mode])
        conf_mat, output_file_conf_mat = compute_confusion_matrix(
            labels_ref, pred_ref, ["ROT1", "ROT2", "ROT3"], n_classes=n_classes, store="True", mode=mode, epoch=epoch,
            output_path=conf_mat_path)
    elif n_classes == 2:
        _, _, _, _, f1_score_class_based = standard_metrics(labels_ref, pred_ref, class_mode="multi")
        print(f"F1-Score (class-based, {mode} set): ", f1_score_class_based)
        f1_score_class_based_dict = {f"performance_metrics/{mode}/f1_score/no_anomaly": f1_score_class_based[0],
                                     f"performance_metrics/{mode}/f1_score/anomaly": f1_score_class_based[1]}
        conf_mat_path = path_check_and_join(results_path, [experiment_name, run_name, "confusion_matrix", mode])
        conf_mat, output_file_conf_mat = compute_confusion_matrix(
            labels_ref, pred_ref, ["no_anomaly", "anomaly"], n_classes=n_classes, store="True", mode=mode, epoch=epoch,
            output_path=conf_mat_path)

    eval_dict["total_loss_avg"] = total_loss_avg
    eval_dict["total_loss_avg_path"] = f"optimization_metrics/{mode}/loss"
    eval_dict["f1_score_class_based_dict"] = f1_score_class_based_dict
    eval_dict["conf_mat"] = conf_mat
    eval_dict["output_file_conf_mat"] = output_file_conf_mat

    return eval_dict


def help_mlflow_logging(eval_dict, epoch):
    # MLFLOW logging
    mlflow.log_metrics(eval_dict["f1_score_class_based_dict"], step=epoch)
    mlflow.log_metric(eval_dict["total_loss_avg_path"], eval_dict["total_loss_avg"], step=epoch)
    artifact_path = path_dissect_and_join(eval_dict["output_file_conf_mat"], n_folders=3)
    mlflow.log_artifact(eval_dict["output_file_conf_mat"], artifact_path=artifact_path)


if __name__ == '__main__':
    """ Example script to use defined functions """
    import argparse
    from utils.env_config import setup_environment
    from utils.mlflow_custom_functions import get_mlflow_experiment, get_mlflow_run
    from torch.utils.data import DataLoader, random_split
    from utils.models import WindTurbineModel
    from data.dataset import RotorBladeDatasetClean

    parser = argparse.ArgumentParser()
    parser.add_argument("--system_mode", type=str, default="local-H-Drive")
    args = parser.parse_args()
    setup_environment(args.system_mode)

    EXPERIMENT_NAME = "test_experiment"
    EXPERIMENT_ID = get_mlflow_experiment(EXPERIMENT_NAME)
    RUN_NAME = "test_run"
    RUN_ID = get_mlflow_run(EXPERIMENT_ID, RUN_NAME)

    # Define dataset origin
    csv_path_data = os.path.join(os.environ.get("COMPLETE_TRIPLE_PATH"), "complete_triples_vb06.csv")
    parameter_dir_names = \
        ["outside_temperature", "nacelle_orientation", "pitch_angle", "rotation_speed", "wind_speed", "wind_direction"]
    n_metadata = 8
    n_classes = 3

    # Initialize model
    torch.manual_seed(42)
    model = WindTurbineModel(n_metadata=n_metadata, n_classes=n_classes)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # Provide path to save final model
    model_path = os.environ.get("MODELS_PATH")
    results_path = os.environ.get("RESULTS_PATH")

    # Define optimization parameters
    name_optim = "Adam"
    epochs = 10
    batch_size = 16
    lr = 0.0001
    momentum = None

    # Define dataset instance and loader
    dataset = RotorBladeDatasetClean(csv_path_data, parameter_dir_names)
    train, val, test = random_split(dataset, [int(len(dataset) * 0.6), int(len(dataset) * 0.2),
                                              int(len(dataset) - int(len(dataset) * 0.6) - int(len(dataset) * 0.2))])
    print(len(train), len(val), len(test))
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=6, drop_last=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=True, num_workers=6, drop_last=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=True, num_workers=6, drop_last=True)

    # Optimize
    optimize_model(train_loader, val_loader, test_loader, model, n_metadata, n_classes, epochs, name_optim,
                   lr, momentum, device, EXPERIMENT_ID, EXPERIMENT_NAME, RUN_ID, run_name=RUN_NAME,
                   save_model_path=model_path, results_path=results_path,
                   eval_val="True", eval_train="True", eval_test="True")
