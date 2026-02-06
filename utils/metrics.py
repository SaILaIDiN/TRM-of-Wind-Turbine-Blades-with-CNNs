""" This module contains both standard metrics from libraries and custom ones for evaluation of our models.
    The idea is to gather all of them in one module for better imports and referencing.
"""

import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def prediction_refactoring(pred, n_classes=2, threshold=None, majority=False):
    """ Takes the predictions and translates it to a format for the evaluation metrics.
        Args:
            pred (list): contains all predictions of the dataset in a single array
            mode (string):
            threshold (None): depends on mode and defines when a value belongs to a certain class
        Case 1:
            For binary refactor shape [1, 2] into [1] with labels [0, 1] -> [0] and [1, 0] -> [1]
        Case 2:
            For multiclass
        NOTE: only works correctly if prediction per sample sums up to 1
    """

    # pred is now a list of arrays with shape (1, n_classes)
    pred = np.asarray(pred).reshape([-1, n_classes])  # Removes intermediate dimension by batch sizes > 1
    pred_ref = np.zeros_like(pred)
    if majority:
        for i in range(0, len(pred)):
            if pred[i, np.argmax(pred[i])].item(0) == np.max(pred[i]):
                pred_ref[i, np.argmax(pred[i])] = 1
    else:
        threshold = 0.5 if threshold is None else threshold

        for i in range(0, len(pred)):
            if pred[i, np.argmax(pred[i])].item(0) >= threshold:
                pred_ref[i, np.argmax(pred[i])] = 1

    return pred_ref


def label_refactoring(labels, n_classes=2):
    """ Takes the labels and translates it to a format for the evaluation metrics.
        Args:
            labels (list): contains all labels of the dataset in a single array
            mode (string):
        Case 1:
            For binary, refactor shape [1, 2] into [1] with labels [0, 1] -> [0] and [1, 0] -> [1]
        Case 2:
            For multiclass
        NOTE: only works correctly if labels are one-hot-encoded
    """
    labels = np.asarray(labels).reshape([-1, n_classes])  # removes redundant brackets
    labels_ref = labels
    return labels_ref


def standard_metrics(labels, pred, class_mode="binary"):
    """ Function takes the predictions and corresponding labels once and returns all standard metrics from it.
        class_mode == binary can be removed and both conditions can be treated equally.
    """
    if class_mode == "binary":
        accuracy = accuracy_score(labels, pred)
        precision = precision_score(labels, pred)
        recall = recall_score(labels, pred)
        f1 = f1_score(labels, pred)
        f1_score_class_based = f1_score(labels, pred, average=None)
        print("Metrics computed!")
        return accuracy, precision, recall, f1, f1_score_class_based

    else:
        accuracy = accuracy_score(labels, pred)
        precision = np.mean(precision_score(labels, pred, average=None))
        recall = np.mean(recall_score(labels, pred, average=None))
        f1 = np.mean(f1_score(labels, pred, average=None))
        # precision_class_based = precision_score(labels, pred, average=None)
        # recall_class_based = recall_score(labels, pred, average=None)
        f1_score_class_based = f1_score(labels, pred, average=None)
        print("Metrics computed!")
        return accuracy, precision, recall, f1, f1_score_class_based


def compute_confusion_matrix(labels, preds, class_list, n_classes=2, store="False", mode="&Test&", epoch=0,
                             output_path=""):
    """ Computes the confusion matrix between predictions and labels.
        Expected shape is the full list of evaluated samples per validation step.
        Ideally, the intermediate batch dimension was previously removed.
        Multi-label information must be one-hot-encoded to work correctly.
    """
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    # Remove intermediate batch dimension if existent
    if labels.ndim >= 3:
        labels = np.asarray(labels).reshape([-1, n_classes])
    if preds.ndim >= 3:
        preds = np.asarray(preds).reshape([-1, n_classes])
    # Check for multi-label format and refactor to single integer label format
    try:
        if len(labels[0]) > 1:
            labels = np.argmax(labels, axis=1)
    except TypeError:
        print("Entry of labels array has no len().")
    try:
        if len(preds[0]) > 1:
            preds = np.argmax(preds, axis=1)
    except TypeError:
        print("Entry of preds array has no len().")

    conf_mat = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(conf_mat, display_labels=class_list)
    disp.plot(values_format='')
    output_file = "dummy_file"
    if store == "True":
        output_file = os.path.join(output_path, f"confusion_matrix_{mode}_Epoch_{epoch}.png")
        plt.savefig(output_file, dpi=300)
    plt.clf()
    plt.close()
    return conf_mat, output_file


if __name__ == '__main__':
    pred_test = [[0.77, 0.13, 0.1], [0.11, 0.11, 0.78], [0.2, 0.3, 0.5]]
    pred_ref = prediction_refactoring(pred_test, n_classes=3)
    print(pred_ref)

    labels_test = [[0., 1.], [0., 1.], [1., 0.]]
    labels_ref = label_refactoring(labels_test, n_classes=2)
    print(labels_ref)
