# Credit Card Fraud Detection

An end-to-end machine learning project for detecting fraudulent credit card transactions in a highly imbalanced dataset.

## Project Overview

Credit card fraud detection is a binary classification problem where fraudulent transactions are extremely rare compared with legitimate transactions.

This project focuses on building a leakage-aware fraud detection workflow covering:

- Data quality and duplicate analysis
- Exploratory data analysis
- Temporal train-validation-test splitting
- Class imbalance handling using class-weighted learning
- Multiple machine learning models
- Precision-Recall based model comparison
- Decision-threshold optimization
- Error analysis
- Feature importance analysis
- Production-style inference using a saved model artifact

## Dataset

The project uses the publicly available Credit Card Fraud Detection dataset.

The original dataset contains:

- 284,807 transactions
- 30 input features
- 1 binary target variable (`Class`)
- `Class = 0` → legitimate transaction
- `Class = 1` → fraudulent transaction

The dataset is highly imbalanced, with fraudulent transactions representing a very small fraction of all transactions.

The raw dataset is **not included in this repository** because of its large file size.

## Data Quality & Duplicate Analysis

Exact duplicate transactions were investigated before modeling.

- Original rows: 284,807
- Duplicate rows removed: 1,081
- Rows after deduplication: 283,726

No conflicting labels were found across duplicate transaction groups.

## Exploratory Data Analysis

The analysis includes:

- Class distribution
- Transaction amount distribution
- Log-transformed transaction amount analysis
- Time-of-day analysis
- Feature correlation analysis

The anonymized `V1`–`V28` features were treated as model features without assigning business meaning that is not provided by the dataset documentation.

## Data Splitting Strategy

A chronological split was used instead of a random split:

- 70% → Training
- 15% → Validation
- 15% → Test

This approach helps simulate a more realistic fraud detection setting where future transactions should not influence model training.

The `Time` feature represents elapsed seconds from the first transaction in the dataset.

## Class Imbalance

The dataset contains a severe class imbalance.

Instead of applying synthetic oversampling, class-weighted learning was used for the classification models to give greater importance to fraudulent transactions.

## Models Evaluated

Three models were evaluated:

1. Logistic Regression
2. Random Forest
3. HistGradientBoosting

Because accuracy can be misleading for highly imbalanced fraud data, ROC-AUC and especially PR-AUC were used for model comparison.

### Validation Results

| Model | ROC-AUC | PR-AUC |
|---|---:|---:|
| Logistic Regression | 0.9824 | 0.8367 |
| Random Forest | 0.9794 | 0.8664 |
| HistGradientBoosting | 0.9784 | 0.8418 |

The Random Forest achieved the highest validation PR-AUC among the evaluated models.

## Threshold Optimization

The default classification threshold of 0.50 was not assumed to be optimal for fraud detection.

The Random Forest decision threshold was optimized using the validation set based on F1-score.

Selected threshold:

`0.2567`

At this threshold on the validation set:

- Precision: 1.0000
- Recall: 0.7818
- F1-score: 0.8776

The threshold was selected using validation data rather than the test set.

## Error Analysis

On the final test evaluation using the selected threshold:

- False Positives: 4
- False Negatives: 13
- True Positives: 39
- True Negatives: 42,503

This provides a more useful view of model behavior than accuracy alone, particularly because missing fraudulent transactions and incorrectly flagging legitimate transactions have different operational implications.

## Final Test Evaluation

| Metric | Result |
|---|---:|
| Accuracy | 0.9996 |
| ROC-AUC | 0.9372 |
| PR-AUC | 0.7657 |
| Fraud Precision | 0.91 |
| Fraud Recall | 0.75 |
| Fraud F1-score | 0.82 |

The test set was evaluated after model and threshold selection. It was not used to tune the model or threshold.

## Feature Importance

Random Forest feature importance was used to inspect which anonymized features contributed most to the model.

Top features included:

- `V14`
- `V10`
- `V4`
- `V12`
- `V17`
- `V11`
- `V16`

The `V1`–`V28` variables are anonymized/PCA-derived features, so their business interpretation is not inferred.

## Production-Style Inference

A reusable model artifact was created containing:

- Trained Random Forest pipeline
- Selected decision threshold
- Expected feature list

This allows inference to use the same preprocessing/model configuration and decision threshold consistently.

Example prediction workflow:

```python
import joblib

artifact = joblib.load("fraud_detection_artifact.pkl")

model = artifact["model"]
threshold = artifact["threshold"]
features = artifact["features"]

probability = model.predict_proba(transaction[features])[:, 1]
prediction = (probability >= threshold).astype(int)
