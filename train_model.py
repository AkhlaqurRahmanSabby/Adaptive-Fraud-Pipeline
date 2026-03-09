from src.preprocessing import load_and_preprocess_data
from src.models import get_baseline_models
from src.evaluation import evaluate_model
from src.active_learning import get_hitl_queue_metrics


def print_evaluation_report(model_name, metrics, hitl_metrics=None):
    """
    Prints a clean, objective evaluation report, including the HITL queue.
    """

    print(f"\n{'='*55}")
    print(f"MODEL: {model_name.upper()} (Threshold: {metrics['threshold']})")
    print(f"{'='*55}")
    
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"PR-AUC:   {metrics['pr_auc']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}\n")
    
    print("Confusion Matrix:")
    print(f"  True Negatives:  {metrics['true_negatives']}")
    print(f"  False Positives: {metrics['false_positives']}")
    print(f"  False Negatives: {metrics['false_negatives']}")
    print(f"  True Positives:  {metrics['true_positives']}\n")
    
    if hitl_metrics:
        print("Active Learning (Human-in-the-Loop Queue):")
        print(f"  Transactions Flagged for Review (Uncertainty 0.4-0.6): {hitl_metrics['queue_size']}")
        print(f"  Actual Fraud Rescued by HITL: {hitl_metrics['actual_fraud_caught_by_hitl']}")
    print(f"{'='*55}\n")


def main():
    print("Initializing pipeline...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_preprocess_data()
    models = get_baseline_models()
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        metrics = evaluate_model(y_val, y_pred_proba, threshold=0.5)
        
        # Calculate how many unsure predictions get routed to humans
        hitl_metrics = get_hitl_queue_metrics(y_val, y_pred_proba)
        
        print_evaluation_report(name, metrics, hitl_metrics)

if __name__ == "__main__":
    main()