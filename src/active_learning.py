import numpy as np


def get_uncertain_predictions(y_pred_proba, lower_bound=0.40, upper_bound=0.60):
    """
    Identifies predictions where the model lacks confidence.
    Probabilities near the decision boundary (0.5) are mathematically uncertain.
    """

    # Create a boolean mask for probabilities falling within the uncertainty window
    uncertain_mask = (y_pred_proba >= lower_bound) & (y_pred_proba <= upper_bound)
    uncertain_indices = np.where(uncertain_mask)[0]
    
    return uncertain_indices


def get_hitl_queue_metrics(y_true, y_pred_proba, lower_bound=0.40, upper_bound=0.60):
    """
    Simulates the Human-in-the-Loop (HITL) review queue.
    Calculates how many actual fraud cases the model was 'unsure' about.
    """

    uncertain_indices = get_uncertain_predictions(y_pred_proba, lower_bound, upper_bound)
    
    # Extract the true labels for the uncertain predictions
    uncertain_true_labels = y_true.iloc[uncertain_indices] if hasattr(y_true, 'iloc') else y_true[uncertain_indices]
    
    total_flagged = len(uncertain_indices)
    actual_fraud_in_queue = sum(uncertain_true_labels == 1)
    
    metrics = {
        "queue_size": total_flagged,
        "actual_fraud_caught_by_hitl": actual_fraud_in_queue
    }
    
    return metrics