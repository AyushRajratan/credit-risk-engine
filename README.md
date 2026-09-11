# Credit Risk Engine

An end-to-end machine learning pipeline for predicting credit default probabilities, built using the GiveMeSomeCredit dataset. This project demonstrates the transition from a legacy rules-based underwriting system to a production-ready, financially optimized predictive model.

## Key Features

*   **Domain-Informed Data Processing:** Robust handling of missing values and anomaly filtering (e.g., extreme unsecured line utilization and anomalous debt ratios) using strict train/test stratification to prevent data leakage.
*   **Model Progression:** Compares a baseline rules-based Decision Tree (depth of 3) against a highly tuned, regularized XGBoost classifier optimized for PR-AUC to handle class imbalances.
*   **Institutional Risk Evaluation:** Generates a standard Credit Bureau Risk Decile Table, calculating cumulative default rates and the Kolmogorov-Smirnov (KS) statistic (achieving 57.89% separation).
*   **Asymmetric Financial Loss Optimization:** Tunes the decision threshold based on real-world business constraints ($5,000 cost for a false negative vs. $500 opportunity cost for a false positive) to generate an optimal cut-off that minimizes portfolio loss.
*   **Regulatory Explainability (SHAP):** Utilizes TreeSHAP for global feature attribution and generates automated, FCRA-compliant Individual Adverse Action Notices extracting the top 3 rejection reasons for high-risk applicants.

## Performance Benchmark

| Metric | Decision Tree (Baseline) | XGBoost (Production) | Relative Lift |
| :--- | :--- | :--- | :--- |
| **ROC-AUC** | 0.821 | 0.867 | 5.57% |
| **PR-AUC (Avg Precision)** | 0.310 | 0.414 | 33.44% |

*Financial Impact:* Tuning the risk threshold from a naive 0.50 to the optimized 0.58 resulted in a net portfolio savings of **$89,500** (a 1.78% reduction in expected loss).

## Requirements

To run this notebook locally, ensure you have the following dependencies installed:

*   `pandas`
*   `numpy`
*   `scikit-learn`
*   `xgboost`
*   `shap`
*   `matplotlib`
*   `seaborn`
*   `scipy`
