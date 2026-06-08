# Loan Default Model: A Zindi Challenge on East African Credit Risk & Macroeconomic Default Prediction
## 1. Architectural Overview
This repository contains a highly optimized, memory-efficient XGBoost classification pipeline designed to predict loan defaults. The architecture integrates localized financial metrics with East African macroeconomic indicators.
Due to the strict physical memory constraints of the deployment environment (12 GiB RAM) and the extreme class imbalance of the target variable, standard data-engineering pipelines fail. This architecture bypasses these limitations through native categorical algorithm optimization and cost-sensitive gradient descent.
## 2. Data Pipeline & Type Enforcement
The integration of external macroeconomic variables with raw loan application data requires strict type safety to prevent matrix alignment failures during the merge process.
 String Normalization: All key columns (⁠country_id⁠, ⁠Indicator⁠, ⁠Year⁠) are explicitly stripped of trailing whitespaces and cast to base types prior to merging.
 Exclusion-Based Casting: To satisfy XGBoost's strict internal data mappers, the pipeline employs a robust exclusion check. Any feature matrix column that is not explicitly numeric or boolean is force-cast to a Pandas ⁠category⁠ type. This prevents ⁠ValueError⁠ crashes during the ⁠model.fit()⁠ execution.
## 3. Hardware Constraints & Memory Management
CRITICAL: Do not attempt to use ⁠pd.get_dummies()⁠ or any form of One-Hot Encoding on this dataset.
Mapping the high-cardinality macroeconomic indicators creates a dense 68653 \times 68654 matrix, instantly demanding 35.1 GiB of RAM and causing a fatal ⁠MemoryError⁠.
 The Solution: The pipeline strictly uses XGBoost's ⁠enable_categorical=True⁠ parameter. This allows the gradient boosting trees to handle string categories natively via dictionary-based splits, keeping the maximum memory footprint well under hardware limits.
 Garbage Collection: Explicit ⁠del⁠ and ⁠gc.collect()⁠ commands are embedded post-merge to flush raw dataframes from physical memory before pipeline initialization.
## 4. Algorithmic Imbalance Correction
The target variable exhibits severe class imbalance (the vast majority of loans are successfully repaid).
Rejection of SMOTE: Synthetic Minority Over-sampling Technique is explicitly banned in this architecture. SMOTE requires continuous vector spaces (incompatible with our native categorical optimization) and inflates the dataset size, which violates our RAM constraints.
Implementation of Cost-Sensitive Learning:
We address the imbalance at the algorithm level by modifying the binary cross-entropy loss function. We dynamically calculate the ratio of the majority class to the minority class and pass this scalar to the ⁠scale_pos_weight⁠ parameter in the XGBoost estimator. This strictly penalizes the model for misclassifying the minority class during gradient descent with O(1) memory overhead.
## 5. Evaluation Protocol
Due to the target imbalance, standard Accuracy is an invalid diagnostic metric for this pipeline. A naive model predicting "Repaid" for all instances would achieve artificially high accuracy while possessing zero predictive power.
The definitive evaluation metric for this architecture is ROC-AUC (Receiver Operating Characteristic - Area Under the Curve). The model is evaluated strictly on its mathematical ability to cleanly separate the probability distributions of the default and non-default classes. Feature importance is subsequently tracked to ensure macroeconomic variables are adequately contributing to the information gain against localized loan variables.
