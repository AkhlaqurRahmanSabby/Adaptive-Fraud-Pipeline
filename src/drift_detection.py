import pandas as pd
from scipy.stats import ks_2samp


def detect_feature_drift(reference_data, current_data, p_value_threshold=0.05):
    """
    Uses the Kolmogorov-Smirnov (KS) test to detect statistical drift
    between a reference dataset (e.g., training) and current data (e.g., production).
    
    Returns a report dictionary and the total count of drifted features.
    """
    
    drift_report = {}
    features_drifted = 0
    
    for column in reference_data.columns:
        # ks_2samp computes the KS statistic and p-value for two samples
        statistic, p_value = ks_2samp(reference_data[column], current_data[column])
        
        # If p-value < threshold, we reject the null hypothesis that the distributions are the same
        is_drifting = bool(p_value < p_value_threshold)
        
        if is_drifting:
            features_drifted += 1
            
        drift_report[column] = {
            "statistic": statistic,
            "p_value": p_value,
            "is_drifting": is_drifting
        }
        
    return drift_report, features_drifted