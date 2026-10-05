# Telecom Customer Churn Analytics

Analysis and prediction of customer churn for a telecom company. The goal is to understand **why customers leave** and to build a model that flags customers at high risk of churning, so retention efforts can be targeted.

## Problem Statement

Acquiring a new customer costs far more than keeping an existing one. This project explores customer demographics, service usage, and billing data to:

1. Identify the main drivers of churn
2. Build a classification model to predict whether a customer will churn
3. Suggest data-backed retention actions

## Dataset

- **Source:** TODO (e.g., IBM Telco Customer Churn dataset on Kaggle)
- **Size:** TODO (rows x columns)
- **Target variable:** `Churn` (Yes / No)

| Feature group | Examples |
|---|---|
| Demographics | gender, senior citizen, partner, dependents |
| Account info | tenure, contract type, payment method, paperless billing |
| Services | phone, internet service, online security, tech support, streaming |
| Billing | monthly charges, total charges |

> The raw data is not committed to this repo (see `.gitignore`). Download it from the source above and place it in a `data/` folder.

## Project Structure

```
Telecom_Customer_Churn/
├── telco-churn-analytics/
│   ├── data/              # raw and processed data (not tracked)
│   ├── notebooks/         # EDA and modeling notebooks   (TODO: adjust)
│   ├── src/               # reusable scripts             (TODO: adjust)
│   └── requirements.txt   # Python dependencies
└── README.md
```

## Approach

1. **Data cleaning**: fix data types (e.g., `TotalCharges`), handle missing values, remove duplicates
2. **Exploratory data analysis**: churn rate by contract, tenure, charges, and services
3. **Feature engineering**: encode categoricals, scale numeric features, create tenure groups
4. **Modeling**: train and compare classifiers (TODO: e.g., Logistic Regression, Random Forest, XGBoost)
5. **Evaluation**: accuracy, precision, recall, F1, ROC-AUC, with extra attention to recall on the churn class
6. **Interpretation**: feature importance to explain churn drivers

## Key Findings

> TODO: fill in once your analysis is complete. Example format:

- Customers on **month-to-month contracts** churn at a much higher rate than those on yearly contracts
- Churn is highest in the **first few months** of tenure
- TODO: add your own findings with numbers

## Model Performance

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| TODO | | | | | |

## Getting Started

### Prerequisites

- Python 3.9+
- Git

### Installation

```bash
git clone https://github.com/RohitR-26/Telecom_Customer_Churn.git
cd Telecom_Customer_Churn/telco-churn-analytics

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

pip install -r requirements.txt
```

### Run

```bash
jupyter notebook
```

Then open the notebooks in order. (TODO: list notebook names.)

## Tech Stack

- Python, Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- Jupyter Notebook

## Future Improvements

- Hyperparameter tuning and cross-validation
- Handle class imbalance (SMOTE or class weights)
- Deploy the model as a simple web app (Streamlit or Flask)
- Build a dashboard (Power BI or Tableau) for business users

## Author

**Rohit R**
GitHub: [@RohitR-26](https://github.com/RohitR-26)