import torch
import torch.nn as nn


def eval_model(loader, model, n_metadata, device):
    model.eval()

    total_loss = 0.0
    iteration_counter = 0

    pred_collector = []   # for evaluation metrics
    labels_collector = []   # also for evaluation metrics

    for i, batch in enumerate(loader):
        images = batch["image"].unsqueeze(1).to(device, dtype=torch.float32)
        labels = batch["label"].to(device, dtype=torch.float32)
        metadata = batch["metadata"].to(device, dtype=torch.float32) \
            if isinstance(n_metadata, int) and n_metadata > 0 else None

        with torch.no_grad():
            if metadata is not None:
                outputs = model(images, metadata)
            else:
                outputs = model(images)

        m = nn.Softmax(dim=1)

        pred_detached = m(outputs).cpu().detach().numpy()
        labels_detached = labels.cpu().detach().numpy()

        pred_collector.append(pred_detached)
        labels_collector.append(labels_detached)

        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(outputs, labels)

        total_loss += loss.item()
        iteration_counter = i

    model.train()

    return total_loss/iteration_counter, pred_collector, labels_collector


if __name__ == '__main__':
    """ Example script to use defined functions """
    from utils.env_config import *
    from utils.models import WindTurbineModel
    from torch.utils.data import DataLoader
    from data.dataset import RotorBladeDatasetClean
    from utils.metrics import standard_metrics, label_refactoring, prediction_refactoring

    csv_path_data = os.path.join(os.environ.get("COMPLETE_TRIPLE_PATH"), "complete_triples_test.csv")
    parameter_dir_names = \
        ["outside_temperature", "nacelle_orientation", "pitch_angle", "rotation_speed", "wind_speed", "wind_direction"]
    n_metadata = 8

    model_path = os.path.join(os.environ.get("MODELS_PATH"), "vb06_test_model.pth")
    model = WindTurbineModel(n_metadata=n_metadata)
    model.load_state_dict(torch.load(model_path))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    batch_size = 1

    dataset = RotorBladeDatasetClean(csv_path_data, parameter_dir_names)
    dataset_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

    total_loss_avg, pred_collector, labels_collector = eval_model(dataset_loader, model, n_metadata, device)
    print("Pred collector: ", pred_collector)
    print("Labels collector: ", labels_collector)

    pred_ref = prediction_refactoring(pred_collector, n_classes=3)
    labels_ref = label_refactoring(labels_collector, n_classes=3)

    _, _, _, _, f1_score_class_based = standard_metrics(labels_ref, pred_ref, class_mode="multi")
    print(f1_score_class_based)
