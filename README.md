# decodelabs-_tasks02

# Credit Card Fraud Detection using Machine Learning

## Project Overview

This project focuses on detecting fraudulent credit card transactions using Machine Learning techniques. Due to the highly imbalanced nature of fraud datasets, techniques such as SMOTE (Synthetic Minority Oversampling Technique) are used to improve model performance.

The project compares the performance of Logistic Regression and Random Forest classifiers and evaluates them using multiple classification metrics.

---

## Problem Statement

Financial institutions face significant losses due to fraudulent transactions. The objective of this project is to build a machine learning system capable of accurately identifying fraudulent transactions while minimizing false negatives.

---

## Dataset

Dataset Used:
- fraudTrain.csv

Target Variable:
- `is_fraud`
  - 0 → Legitimate Transaction
  - 1 → Fraudulent Transaction

Dataset contains transaction details such as:
- Transaction amount
- Merchant information
- Category
- Gender
- Location details
- Customer job
- Transaction timestamp
- Customer DOB

---

## Project Workflow

### 1. Data Loading

- Import dataset using Pandas
- Sample 50,000 records for faster training

```python
data = pd.read_csv("fraudTrain.csv")
data = data.sample(n=50000, random_state=42)
