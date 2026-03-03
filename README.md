# Machine Learning Classification Experiment

This project demonstrates a supervised learning workflow for binary classification,
including model comparison and hyperparameter tuning.

---

## Project Overview

The goal of this experiment is to compare multiple classification models
and evaluate their performance using different metrics.

Models used:
- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest

---

## Techniques Applied

- Data preprocessing & standardization
- Train-test split
- 5-fold Cross Validation
- Model evaluation (Accuracy, F1 Score, ROC-AUC)
- Hyperparameter tuning using GridSearchCV

---

## Why Model Comparison?

Different models perform differently depending on:
- Linearity of data
- Feature distribution
- Noise sensitivity
- Overfitting risk

This experiment demonstrates how ensemble models (Random Forest)
can outperform linear models in non-linear scenarios.

---

## How to Run

```bash
pip install -r requirements.txt
python main.py
```

---

## Output Example

The script prints:
- Accuracy
- F1 Score
- ROC-AUC
- Best hyperparameters (for Random Forest)

---

## Key Learning Outcomes

- Understanding evaluation metrics beyond accuracy
- Practical use of cross-validation
- Hyperparameter tuning workflow
- Comparing linear vs ensemble models# ml-classification-experiment
Machine learning classification experiment with model comparison and hyperparameter tuning.
