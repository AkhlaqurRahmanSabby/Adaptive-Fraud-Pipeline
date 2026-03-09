from sklearn.metrics import average_precision_score, f1_score, confusion_matrix, accuracy_score


def evaluate_model(y_true, y_pred_proba, threshold=0.5):
    """
    Evaluates model performance using a specific threshold.
    Returns a dictionary of metrics to be logged or printed by the caller.
    """

    y_pred = (y_pred_proba >= threshold).astype(int)
    
    cm = confusion_matrix(y_true, y_pred)
    
    metrics = {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "pr_auc": average_precision_score(y_true, y_pred_proba),
        "f1_score": f1_score(y_true, y_pred),
        "true_negatives": int(cm[0][0]),
        "false_positives": int(cm[0][1]),
        "false_negatives": int(cm[1][0]),
        "true_positives": int(cm[1][1])
    }
    
    return metrics