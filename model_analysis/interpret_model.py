import os
import warnings
import matplotlib.pyplot as plt
import mlflow
import yaml
import numpy as np
import torch
import torch.nn as nn
from utils.models import WindTurbineModel
from utils.env_config import path_check_and_join, path_dissect_and_join
from utils.metrics import prediction_refactoring
from captum.attr import GuidedGradCam
from captum.attr import visualization as viz


def execute_class_activation_mapping(experiment_id, experiment_name, run_id, run_name, dir_yaml, dataloader,
                                     model=None, vis_model_layer=None):
    """ Only works with DataLoader batch_size = 1 """

    # Load configuration data from YAML file
    with open(dir_yaml, 'r') as config_file:
        config = yaml.safe_load(config_file)

    # Configuration data for model loading
    run_name_opt = config["run_name_opt"]
    model_file = config["model_pth"]

    # Configuration data for inference
    n_metadata = config["n_metadata"]  # 0
    n_classes = config["n_classes"]  # 3
    n_batches = config["n_batches"]  # 5, not same as batch_size
    gt_or_all = config["ground_truth_or_all"]  # classes to check GuidedGradCAM for

    # Configuration data for radar and plot format
    radar_format = config.get("radar_format", None)
    plot_format = config.get("plot_format", None)
    assert (plot_format is None or radar_format is not None), "radar_format must be provided when plot_format != None"
    clean_spots = config.get("clean_spots", None)
    fig_size = config.get("fig_size", (8, 6))
    vis_types = config.get("vis_types", ["heat_map", "original_image", "blended_heat_map"])
    vis_signs = config.get("vis_signs", ["all", "all", "all"])  # "positive", "negative", "all", or "absolute_value"
    assert len(vis_types) == len(vis_signs), "Number of vis_types and vis_signs must match!"
    vis_titles = config.get("vis_titles", None)
    # positive attribution indicates that the presence of the area increases the prediction score
    # negative attribution indicates distract areas whose absence increases the score

    if "RESULTS_PATH" in os.environ and "MODELS_PATH" in os.environ:
        results_path = os.environ.get("RESULTS_PATH")
        model_path = os.environ.get("MODELS_PATH")
    else:
        print("Environmental variables not set!")
        return
    model_path = path_check_and_join(model_path, [experiment_name, run_name_opt, model_file], end_is_file=True)

    with mlflow.start_run(experiment_id=experiment_id, run_id=run_id):
        if model is None:
            model = WindTurbineModel(n_metadata=n_metadata, n_classes=n_classes)
            vis_model_layer = model.conv3
        model.load_state_dict(torch.load(model_path))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        # # # Define model interpretation
        guided_gc = GuidedGradCam(model, vis_model_layer)

        def apply_and_log_guided_gc(target, ground_truth, radar_format=None, plot_format=None, clean_spots=None, i=None):
            """ Inner function to apply and log a single run of GuidedGradCAM on an image with the class of interest.
                Allows to also check for wrong class activations. Helps with error analysis. For one-hot-encoded labels.
            """
            attribution = guided_gc.attribute(image, target=target, additional_forward_args=metadata)
            print(np.shape(attribution))
            print(np.shape(attribution.squeeze()))
            # Note: the current state of the code can throw an error when the model has a custom forward function like
            # we do. If the loaded model was trained with metadata, the forward function would pass "metadata" = None
            # because forward() is called inside the captum API.

            attribution = attribution.squeeze().cpu().detach().numpy()
            attribution = attribution[..., np.newaxis]  # Expand array from (H, W) to (H, W, C) for visualization

            # torch.Tensor of shape (1, 1, H, W) is squeezed to (H, W) and then unsqueezed to (H, W, 1)
            radargram = np.array(image.squeeze().unsqueeze(-1).cpu().detach().numpy())

            if radar_format == "linear":

                if clean_spots is not None:
                    # Removes all 0 values that result in log(x) i.e. undefined, causing white dots
                    radargram[(radargram == 0.0)] += 0.001

                if plot_format == "linear":
                    radargram = radargram

                elif plot_format == "log":
                    radargram = 20 * np.log10(radargram)

                elif plot_format == "log_norm":
                    max_radargram = max([max(i) for i in radargram])
                    radargram = 20 * np.log10(radargram / max_radargram)

            elif radar_format == "log":
                if plot_format == "log_norm":
                    # if Dataset instance passes calculated logs, you have to revert the log,
                    # then add the eps and apply the log again
                    radargram = 20 * np.log10((10^(radargram / 20) + 0.001))
                elif plot_format == "log_norm_thresh":
                    radargram = 20 * np.log10((10^(radargram / 20) + 0.001))
                    radargram[radargram < -30] = 20 * np.log10(0.001)

                radargram = radargram

            elif radar_format == "log_norm":
                if plot_format == "log_norm_thresh":
                    radargram[radargram < -30] = 20 * np.log10(0.001)

                radargram = radargram

            elif radar_format == "log_norm_thresh":
                radargram = radargram

            # # # Revert the negative values after normalized logarithm
            min_radargram = min([min(i) for i in radargram])
            radargram = np.add(radargram, -min_radargram)

            # # # # Intermediate check of radardata being plottable
            # radargram_regular = np.array(image.squeeze().cpu().detach().numpy())
            # fig = plt.figure()
            # ax1 = fig.add_subplot(111)
            # c = ax1.pcolormesh(radargram_regular)
            # ax1.set_xlabel("Distance [m]")
            # ax1.set_ylabel("Time [s]")
            # # Add a colorbar based on the pcolormesh plot
            # plt.colorbar(c, ax=ax1, label='Intensity [dB]')
            # plt.savefig(f"/user_path/projects/TRM-of-Wind-Turbine-Blades-with-CNNs/test_plots/plot_log_{i}.png")
            # # # # End check

            viz_figure, _ = viz.visualize_image_attr_multiple(attribution,  # must be shape (H, W, C)
                                                              radargram,  # must be shape (H, W, C)
                                                              vis_types,
                                                              vis_signs,
                                                              vis_titles,
                                                              fig_size,
                                                              show_colorbar=True,
                                                              use_pyplot=False
                                                              # outlier_perc=0
                                                              )

            # The logging procedure is placed here instead of inside the optimization because it is optional
            local_visualization_path = path_check_and_join(
                results_path, [experiment_name, run_name, "visuals", "GuidedGradCAM", sample_filename])
            local_visualization_file_path = os.path.join(
                local_visualization_path, f"Visualised_Class_{target +1}_Pred_{pred_ref +1}_GT_{ground_truth +1}.png")
            # plt.savefig(local_visualization_file_path, dpi=300)
            # upper line does not work right, because viz_figure is a Figure object and not a pyplot.figure
            viz_figure.savefig(local_visualization_file_path, dpi=300)
            plt.clf()
            plt.close()
            artifact_path = path_dissect_and_join(
                local_visualization_file_path, n_folders=4)  # n folders you take with you
            mlflow.log_artifact(local_path=local_visualization_file_path, artifact_path=artifact_path)

        def sub_eval_model():
            with torch.no_grad():
                if metadata is not None:
                    outputs = model(image, metadata)
                else:
                    outputs = model(image)
            m = nn.Softmax(dim=1)
            pred_detached = m(outputs).cpu().detach().numpy()
            pred_refactored = prediction_refactoring(pred_detached, n_classes=3)
            return pred_refactored

        for i, batch in enumerate(dataloader, 0):
            if i >= n_batches:
                break

            image = batch["image"].unsqueeze(1).to(device, dtype=torch.float32).requires_grad_(True)
            # For torch operations the batch must be of shape [B, C, H, W]
            label = batch["label"].to(device, dtype=torch.float32)  # Is a torch.tensor([i, j, k]) (one-hot-encoded)
            metadata = batch["metadata"].to(device, dtype=torch.float32) \
                if isinstance(n_metadata, int) and n_metadata > 0 else None
            sample_filename = batch["filename"][0]

            label = torch.nonzero(label[0] == 1).item()

            # Prediction -> passed to inner function apply_and_log_guided_gc()
            pred_ref = sub_eval_model()
            pred_ref = np.argmax(pred_ref[0])  # refactor to single integer label format

            if gt_or_all == "gt":
                try:
                    apply_and_log_guided_gc(target=label, ground_truth=label, radar_format=radar_format,
                                            plot_format=plot_format, clean_spots=clean_spots, i=i)
                except AssertionError:
                    warnings.warn("Most likely the code cannot normalize by scale factor = 0. "
                                  "Please check outlier_perc in visualize_image_attr_multiple() "
                                  "to verify the existence of attributions greater 0.")
                    # If outlier_perc = 0 and scale_factor is still 0, then there is no attribution value greater 0.
                    # I.e. in this specific case the model does not have activations favouring the respective class.
            elif gt_or_all == "all":
                for class_i in range(0, n_classes):
                    try:
                        apply_and_log_guided_gc(target=class_i, ground_truth=label, radar_format=radar_format,
                                                plot_format=plot_format, clean_spots=clean_spots, i=i)
                    except AssertionError:
                        warnings.warn("Most likely the code cannot normalize by scale factor = 0. "
                                      "Please check outlier_perc in visualize_image_attr_multiple() "
                                      "to verify the existence of attributions greater 0.")
                        # If outlier_perc = 0 and scale_factor is still 0, then there is no attribution value greater 0.
                        # I.e. in this specific case the model does not have activations favouring the respective class.
            else:
                print("Parameter gt_or_all was not correctly defined. Visualisation was not performed!")
