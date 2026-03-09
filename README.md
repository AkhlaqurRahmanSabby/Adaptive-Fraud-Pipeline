# Adaptive Fraud Detection Pipeline: Active Learning & Concept Drift

This repository implements a production-minded machine learning architecture for credit card fraud detection. 

Moving beyond standard static classification, this system is designed to handle the realities of deployed fraud ML models: extreme class imbalance, shifting adversarial behavior (concept drift), and the need for Human-in-the-Loop (HITL) review pipelines.

## 🎯 System Objectives

1. **Robust Classification under Extreme Imbalance:** Baseline models configured to handle ~0.17% positive class distributions using class-weighting and advanced sampling techniques.
2. **Uncertainty Quantification & Active Learning:** A routing mechanism that flags low-confidence predictions (predictions near the decision boundary) and sends them to a HITL queue for manual review and subsequent model retraining.
3. **Concept Drift Detection:** Statistical monitoring of feature distributions over time to detect emerging fraud patterns and trigger alerts for model degradation.
4. **Business-Aligned Evaluation:** Focusing strictly on Precision-Recall AUC, F1-Score, and custom threshold tuning to balance high recall with manageable false-positive rates for fraud investigation teams.

## 📂 Project Structure

```text
adaptive-fraud-pipeline/
│
├── data/                                 # Raw and processed datasets
├── notebooks/
│   ├── 01_imbalance_baselines.ipynb      # EDA, SMOTE, and baseline model comparisons
│   └── 02_concept_drift_simulation.ipynb # Simulating adversarial drift and HITL recovery
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py                  # Feature scaling, time-based splits
│   ├── models.py                         # Model definitions (XGBoost, Random Forest, LR)
│   ├── active_learning.py                # Uncertainty quantification and HITL routing
│   ├── drift_detection.py                # Feature distribution monitoring (e.g., KS tests)
│   └── evaluation.py                     # PR-AUC, Confusion Matrices, Threshold tuning
│
├── train_model.py                        # Main training pipeline execution script
├── requirements.txt                      # Project dependencies
└── README.md
