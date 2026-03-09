from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def get_baseline_models():
    """
    Initializes baseline models with parameters explicitly configured for extreme class imbalance.
    """

    models = {
        # Logistic Regression serves as our fast, interpretable linear baseline.
        # class_weight='balanced' automatically adjusts weights inversely proportional to class frequencies.
        "LogisticRegression": LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        ),
        
        # Random Forest handles non-linear relationships and feature interactions well.
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            n_jobs=-1,
            random_state=42
        ),
        
        # scale_pos_weight acts as the class weight multiplier.
        # Calculation: ~284,315 normal / 492 fraud ≈ 578. We use 580 to heavily penalize false negatives.
        "XGBoost": XGBClassifier(
            scale_pos_weight=580,
            eval_metric='logloss',
            random_state=42
        )
    }

    return models