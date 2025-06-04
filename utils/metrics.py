import numpy as np
import math

def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> list:
    """Compute the confusion matrix (TP, FP, FN, TN) for binary classification."""
    true_positives = false_positives = false_negatives = true_negatives = 0
    for true, pred in zip(y_true, y_pred):
        if true == 0 and pred == 0:
            true_negatives += 1
        elif true == 0 and pred == 1:
            false_positives += 1
        elif true == 1 and pred == 0:
            false_negatives += 1
        elif true == 1 and pred == 1:
            true_positives += 1
    return [[true_positives, false_positives], [false_negatives, true_negatives]]

def compute_accuracy(conf_matrix: list) -> float:
    """Calculate accuracy from the confusion matrix."""
    tp, fp, fn, tn = conf_matrix[0][0], conf_matrix[0][1], conf_matrix[1][0], conf_matrix[1][1]
    total = tp + fp + fn + tn
    return (tp + tn) / total if total > 0 else 0.0

def compute_precision(conf_matrix: list) -> float:
    """Calculate precision from the confusion matrix."""
    tp, fp = conf_matrix[0][0], conf_matrix[0][1]
    return tp / (tp + fp) if tp + fp > 0 else 0.0

def compute_recall(conf_matrix: list) -> float:
    """Calculate recall from the confusion matrix."""
    tp, fn = conf_matrix[0][0], conf_matrix[1][0]
    return tp / (tp + fn) if tp + fn > 0 else 0.0

def compute_f1_score(conf_matrix: list) -> float:
    """Calculate F1-score from the confusion matrix."""
    precision = compute_precision(conf_matrix)
    recall = compute_recall(conf_matrix)
    return 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

def compute_mcc(conf_matrix: list) -> float:
    """Calculate Matthews correlation coefficient from the confusion matrix."""
    tp, fp, fn, tn = conf_matrix[0][0], conf_matrix[0][1], conf_matrix[1][0], conf_matrix[1][1]
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / denominator if denominator > 0 else 0.0

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
    """Evaluate model performance using multiple metrics."""
    conf_matrix = compute_confusion_matrix(y_true, y_pred)
    return (
        compute_precision(conf_matrix),
        compute_recall(conf_matrix),
        compute_f1_score(conf_matrix),
        compute_mcc(conf_matrix),
        compute_accuracy(conf_matrix)
    )