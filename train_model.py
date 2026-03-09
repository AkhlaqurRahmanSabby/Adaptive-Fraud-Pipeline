from src.preprocessing import load_and_preprocess_data
from src.models import get_baseline_models
from src.evaluation import evaluate_model


def print_evaluation_report(model_name, metrics):
    """
    Prints a clean, objective evaluation report.
    """
    print(f"\n{'='*50}")
    print(f"MODEL: {model_name.upper()} (Threshold: {metrics['threshold']})")
    print(f"{'='*50}")
    
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"PR-AUC:   {metrics['pr_auc']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}\n")
    
    print("Confusion Matrix:")
    print(f"  True Negatives:  {metrics['true_negatives']}")
    print(f"  False Positives: {metrics['false_positives']}")
    print(f"  False Negatives: {metrics['false_negatives']}")
    print(f"  True Positives:  {metrics['true_positives']}")
    print(f"{'='*50}\n")


def main():
    print("Initializing pipeline...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_preprocess_data()
    models = get_baseline_models()
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        
        # Predict probabilities on the validation set
        # [:, 1] gets the probability of the positive class (Fraud)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        metrics = evaluate_model(y_val, y_pred_proba, threshold=0.5)
        print_evaluation_report(name, metrics)

if __name__ == "__main__":
    main()