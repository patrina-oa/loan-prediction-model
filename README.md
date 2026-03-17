#  Loan Eligibility Predictor

A machine learning web application that predicts loan approval outcomes based on applicant details, built with a Support Vector Machine (SVM) classifier and deployed via a Streamlit interface.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Future Improvements](#future-improvements)

---

## Overview

This project automates the loan eligibility screening process by training a binary classification model on historical applicant data. Given a set of applicant features (income, credit history, education level, etc.), the model predicts whether a loan application will be **Approved** or **Rejected**.

The pipeline covers the full machine learning workflow:
- Data loading and cleaning
- Label encoding of categorical features
- Exploratory data analysis and visualization
- Model training with SVM
- Model evaluation
- Deployment as an interactive web app

---

## Dataset

- **Source:** [Kaggle – Loan Prediction Dataset](https://www.kaggle.com/datasets/ninzaami/loan-predication)
- **Size:** 614 rows × 13 columns
- **Target Variable:** `Loan_Status` (Y = Approved, N = Rejected)

### Features

| Feature | Type | Description |
|---|---|---|
| `Gender` | Categorical | Male / Female |
| `Married` | Categorical | Yes / No |
| `Dependents` | Categorical | 0, 1, 2, 3+ |
| `Education` | Categorical | Graduate / Not Graduate |
| `Self_Employed` | Categorical | Yes / No |
| `ApplicantIncome` | Numerical | Monthly income of the applicant |
| `CoapplicantIncome` | Numerical | Monthly income of the co-applicant |
| `LoanAmount` | Numerical | Loan amount requested (in thousands) |
| `Loan_Amount_Term` | Numerical | Term of loan in months |
| `Credit_History` | Binary | 1 = clear history, 0 = outstanding debts |
| `Property_Area` | Categorical | Rural / Semiurban / Urban |

---

## Project Structure
````
loan-prediction-model/
│
├── Loan_Predict_Model.ipynb   # Main notebook: EDA, training, and evaluation
├── app.py                     # Streamlit web application
├── loan_model.sav             # Saved trained SVM model (generated after training)
└── README.md
````

---

## Installation

### Prerequisites

- Python 3.8+
- pip

### Steps

1. **Clone the repository**
```bash
   git clone https://github.com/patrina-oa/loan-prediction-model.git
   cd loan-prediction-model
```

2. **Install dependencies**
```bash
   pip install numpy pandas scikit-learn seaborn streamlit
```

3. **Download the dataset** from [Kaggle](https://www.kaggle.com/datasets/ninzaami/loan-predication) and place the CSV in the project directory, or use `kagglehub`:
```bash
   pip install kagglehub
```

---

## Usage

### Training the Model

Run the Jupyter notebook to train the model and generate `loan_model.sav`:
```bash
jupyter notebook Loan_Predict_Model.ipynb
```

### Running the Web App
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

### Making a Prediction Programmatically
```python
import pickle
import pandas as pd

model = pickle.load(open('loan_model.sav', 'rb'))

feature_names = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
                 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
                 'Loan_Amount_Term', 'Credit_History', 'Property_Area']

applicant = [1, 1, 0, 1, 0, 4583, 1508, 128, 360, 1, 2]
input_df = pd.DataFrame([applicant], columns=feature_names)

prediction = model.predict(input_df)
print("Approved" if prediction[0] == 1 else "Rejected")
```

---

## Model Details

### Algorithm: Support Vector Machine (SVM)

- **Kernel:** Linear
- **Library:** `scikit-learn`
- **Why SVM?** SVM is well-suited for binary classification tasks. The linear kernel produces a clean decision boundary that generalises well on tabular data of this size.

### Preprocessing

- **Missing values:** Rows with null values were dropped (480 clean rows retained from 614).
- **Label encoding:** All categorical columns were manually mapped to integers.
- **Train/Test Split:** 90% training / 10% testing, stratified on `Loan_Status`, `random_state=4`.

### Encoding Reference

| Column | Mapping |
|---|---|
| `Loan_Status` | Y → 1, N → 0 |
| `Gender` | Male → 1, Female → 0 |
| `Married` | Yes → 1, No → 0 |
| `Education` | Graduate → 1, Not Graduate → 0 |
| `Self_Employed` | Yes → 1, No → 0 |
| `Property_Area` | Rural → 0, Semiurban → 1, Urban → 2 |
| `Dependents` | 3+ → 4 |

---

## Results

| Dataset | Accuracy |
|---|---|
| Training set | **78.01%** |
| Test set | **77.08%** |

The close gap between training and test accuracy indicates the model generalises well with no significant overfitting.

### Key Insights from EDA

- **Graduates** have a higher loan approval rate than non-graduates.
- **Married applicants** are more likely to receive loan approval.
- A clear **credit history** is one of the strongest predictors of approval.

---

## Technologies Used

| Tool | Purpose |
|---|---|
| Python | Core programming language |
| pandas | Data manipulation |
| NumPy | Numerical operations |
| scikit-learn | Model training and evaluation |
| seaborn | Data visualisation |
| Streamlit | Web app deployment |
| pickle | Model serialisation |
| Jupyter Notebook | Development environment |

---

## Future Improvements

- **Imputation over dropping** — use mean/mode strategies to retain all 614 samples instead of dropping rows with missing values.
- **Feature engineering** — e.g. total income (applicant + co-applicant), income-to-loan ratio.
- **Model comparison** — evaluate Random Forest, XGBoost, and Logistic Regression alongside SVM.
- **Cross-validation** for a more robust accuracy estimate.
- **Feature scaling** with `StandardScaler` for potentially improved SVM performance.
- **Input validation** in the Streamlit app to handle edge cases gracefully.


---

## Acknowledgements

- Dataset sourced from [Kaggle](https://www.kaggle.com/datasets/ninzaami/loan-predication).
- Built as a learning project to demonstrate an end-to-end machine learning pipeline.
