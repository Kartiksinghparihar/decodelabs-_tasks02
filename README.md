# decodelabs-_tasks02

# 💳 Credit Card Fraud Detection Using Machine Learning

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-green)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![SMOTE](https://img.shields.io/badge/SMOTE-Class%20Balancing-red)
![Status](https://img.shields.io/badge/Project-Completed-success)

---

## 📌 Project Overview

Credit card fraud is a growing concern in the financial industry, causing billions of dollars in losses every year. Fraudulent transactions are typically rare compared to legitimate transactions, making fraud detection a challenging machine learning problem due to severe class imbalance.

This project develops and evaluates machine learning models capable of detecting fraudulent transactions using historical transaction data. The workflow includes data cleaning, feature engineering, class balancing using SMOTE, model training, hyperparameter tuning, and performance evaluation.

---

## 🎯 Objectives

The primary goals of this project are:

- Analyze transaction data for fraud detection.
- Clean and preprocess raw transaction records.
- Engineer meaningful features from existing attributes.
- Handle class imbalance using SMOTE.
- Train multiple machine learning models.
- Evaluate models using fraud-specific metrics.
- Compare model performance and identify the best solution.

---

## 📊 Dataset Information

The dataset contains transaction-level information including customer details, merchant information, transaction amount, and fraud labels.

### Target Variable

| Value | Description |
|---------|-------------|
| 0 | Legitimate Transaction |
| 1 | Fraudulent Transaction |

### Important Features

| Feature | Description |
|----------|-------------|
| trans_date_trans_time | Transaction Timestamp |
| merchant | Merchant Name |
| category | Transaction Category |
| amt | Transaction Amount |
| gender | Customer Gender |
| city | Customer City |
| state | Customer State |
| job | Customer Occupation |
| dob | Customer Date of Birth |
| is_fraud | Fraud Indicator |

---

## ⚙️ Project Workflow

### 1️⃣ Data Collection

The dataset was loaded using Pandas and sampled to improve computational efficiency.

```python
data = pd.read_csv("fraudTrain.csv")
data = data.sample(n=50000, random_state=42)
```

---

### 2️⃣ Data Cleaning

Removed unnecessary columns that do not contribute significantly to fraud prediction:

```python
columns_to_drop = [
    'Unnamed: 0',
    'cc_num',
    'first',
    'last',
    'street',
    'trans_num'
]
```

Benefits:

- Reduces dimensionality
- Removes personally identifiable information
- Improves model efficiency

---

### 3️⃣ Missing Value Handling

Missing values were removed using:

```python
data.dropna(inplace=True)
```

This ensures consistent and reliable model training.

---

### 4️⃣ Feature Engineering

Additional features were extracted from timestamp and date information.

#### Transaction Hour

```python
data['transaction_hour']
```

Captures transaction timing patterns.

#### Transaction Day

```python
data['transaction_day']
```

Identifies daily transaction behavior.

#### Customer Age

```python
data['customer_age']
```

Derived from customer date of birth.

After feature creation:

```python
trans_date_trans_time
dob
```

were removed.

---

### 5️⃣ Categorical Encoding

Machine learning models require numerical inputs.

Label Encoding was applied to:

- Merchant
- Category
- Gender
- City
- State
- Job

```python
from sklearn.preprocessing import LabelEncoder
```

---

### 6️⃣ Train-Test Split

The dataset was split into:

- Training Set: 80%
- Testing Set: 20%

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)
```

Stratification preserves the original fraud distribution.

---

### 7️⃣ Handling Class Imbalance

#### Why SMOTE?

Fraudulent transactions are extremely rare.

Without balancing:

- Model becomes biased toward legitimate transactions.
- Fraud detection performance decreases.

#### Solution

Synthetic Minority Oversampling Technique (SMOTE)

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
```

Benefits:

✔ Better fraud detection

✔ Improved Recall

✔ Balanced class distribution

✔ Reduced bias

---

## 🤖 Machine Learning Models

### Logistic Regression

Used as a baseline classification model.

Pipeline:

```python
StandardScaler
SMOTE
LogisticRegression
```

Advantages:

- Fast training
- Easy interpretation
- Strong baseline performance

---

### Random Forest Classifier

An ensemble learning model used for improved prediction performance.

Advantages:

- Handles nonlinear relationships
- Robust against overfitting
- High predictive power
- Works well on structured datasets

---

## 🔧 Hyperparameter Tuning

Random Forest was optimized using:

```python
RandomizedSearchCV
```

Parameters tuned:

```python
n_estimators
max_depth
```

Optimization Metric:

```python
Recall
```

Recall is prioritized because missing a fraudulent transaction is more costly than flagging a legitimate transaction.

---

## 📈 Model Evaluation Metrics

### Precision

Measures how many predicted fraud transactions are actually fraudulent.

\[
Precision = \frac{TP}{TP + FP}
\]

---

### Recall

Measures how many actual fraud cases are detected.

\[
Recall = \frac{TP}{TP + FN}
\]

---

### F1 Score

Balances Precision and Recall.

\[
F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}
\]

---

### ROC-AUC Score

Measures the model's ability to distinguish between fraud and non-fraud transactions.

Higher values indicate better classification performance.

---

## 📊 Model Comparison

The following metrics were compared:

| Model | Precision | Recall | F1 Score | ROC-AUC |
|---------|-----------|---------|----------|----------|
| Logistic Regression | Evaluated | Evaluated | Evaluated | Evaluated |
| Random Forest | Evaluated | Evaluated | Evaluated | Evaluated |

The best-performing model was selected based on overall classification performance and ROC-AUC score.

---

## 📉 Exploratory Data Analysis (EDA)

The project includes analysis of:

- Fraud vs Non-Fraud Distribution
- Transaction Amount Distribution
- Category-Based Fraud Analysis
- Customer Demographics
- Correlation Analysis
- Feature Relationships

Visualizations help understand fraud patterns and customer behavior.

---

## 🛠️ Technologies Used

### Programming Language

- Python

### Libraries

- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- Imbalanced-Learn

---

## 📂 Project Structure

```text
Credit-Card-Fraud-Detection/
│
├── Project_2.ipynb
├── fraudTrain.csv
├── README.md
├── requirements.txt
└── Documentation.md
```

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/credit-card-fraud-detection.git
```

Move into project directory:

```bash
cd credit-card-fraud-detection
```

Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
```

---

## ▶️ Running the Project

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
Project_2.ipynb
```

Run all cells sequentially.

---

## 📌 Key Achievements

✅ Data Cleaning

✅ Feature Engineering

✅ Label Encoding

✅ Class Balancing Using SMOTE

✅ Logistic Regression Implementation

✅ Random Forest Implementation

✅ Hyperparameter Tuning

✅ Fraud Detection Optimization

✅ Performance Evaluation

✅ Model Comparison

---

## 🔮 Future Improvements

Potential enhancements include:

- XGBoost Implementation
- LightGBM Implementation
- Deep Learning Models
- Feature Selection Techniques
- Real-Time Fraud Detection API
- Streamlit Dashboard
- Flask/FastAPI Deployment
- Model Monitoring System

---

## 📚 Learning Outcomes

Through this project, the following concepts were applied:

- Data Cleaning
- Feature Engineering
- Handling Imbalanced Data
- SMOTE
- Logistic Regression
- Random Forest
- Hyperparameter Tuning
- Classification Metrics
- ROC-AUC Analysis
- End-to-End Machine Learning Workflow

---

## 👨‍💻 Author

### Kartik Singh Parihar

**Data Science Intern Project**

Credit Card Fraud Detection Using Machine Learning

---

## ⭐ If you found this project useful, consider giving it a star on GitHub!
