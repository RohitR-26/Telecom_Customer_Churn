<p align="center">
  <img src="telco-churn-analytics/assests/banner.svg" alt="Telecom Customer Churn Analytics banner" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas"/>
  <img src="https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="scikit-learn"/>
  <img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white" alt="Jupyter"/>
  <img src="https://img.shields.io/badge/Status-In%20Progress-brightgreen?style=for-the-badge" alt="Status"/>
</p>

<p align="center">
  <b>📉 Understand churn &nbsp;•&nbsp; 🤖 Predict churn &nbsp;•&nbsp; 💡 Reduce churn</b>
</p>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Workflow](#-workflow)
- [Key Insights](#-key-insights)
- [Model Performance](#-model-performance)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Tech Stack](#-tech-stack)
- [Future Work](#-future-work)
- [Author](#-author)

---

## 🎯 Overview

Winning a new customer costs far more than keeping an existing one. This project analyses telecom customer data to:

| | Goal |
|---|---|
| 🔍 | Discover **why** customers churn |
| 🤖 | Build a model that **predicts** who is likely to leave |
| 💡 | Recommend **retention actions** backed by data |

---

## 🗂️ Dataset

- **Source:** TODO (e.g., IBM Telco Customer Churn on Kaggle)
- **Size:** TODO rows × TODO columns
- **Target:** `Churn` (Yes / No)

| 👤 Demographics | 📄 Account | 📡 Services | 💳 Billing |
|---|---|---|---|
| Gender | Tenure | Phone / Internet | Monthly charges |
| Senior citizen | Contract type | Online security | Total charges |
| Partner / Dependents | Payment method | Tech support, streaming | Paperless billing |

> 📝 Raw data is not committed to this repo. Download it from the source above and place it in a `data/` folder.

---

## 🔄 Workflow

```mermaid
flowchart LR
    A[📥 Raw Data] --> B[🧹 Cleaning]
    B --> C[📊 EDA]
    C --> D[🛠️ Feature Engineering]
    D --> E[🤖 Modeling]
    E --> F[📏 Evaluation]
    F --> G[💡 Insights]
```

---

## 📊 Key Insights

> 🚧 Replace the placeholders below with your real findings and charts.

<p align="center">
  <img src="assets/churn_distribution.png" width="45%" alt="Churn distribution"/>
  &nbsp;
  <img src="assets/churn_by_contract.png" width="45%" alt="Churn by contract type"/>
</p>

- 📆 **Contract type:** month-to-month customers tend to churn the most (TODO: add your %)
- ⏳ **Tenure:** churn is usually highest in the first months (TODO: add your numbers)
- 💸 **Charges:** TODO
- 🌐 **Services:** TODO

<details>
<summary><b>📸 How to add your charts</b></summary>

In your notebook, save each plot, then commit the `assets/` folder:

```python
import os
os.makedirs("assets", exist_ok=True)
plt.savefig("assets/churn_by_contract.png", dpi=150, bbox_inches="tight")
```

</details>

---

## 🏆 Model Performance

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| TODO (e.g., Logistic Regression) | – | – | – | – | – |
| TODO (e.g., Random Forest) | – | – | – | – | – |
| TODO (e.g., XGBoost) | – | – | – | – | – |

<p align="center">
  <img src="assets/confusion_matrix.png" width="40%" alt="Confusion matrix"/>
  &nbsp;
  <img src="assets/feature_importance.png" width="45%" alt="Feature importance"/>
</p>

---

## 📁 Project Structure

```
Telecom_Customer_Churn/
├── assets/                  # banner and chart images
├── telco-churn-analytics/
│   ├── data/                # datasets (not tracked)
│   ├── notebooks/           # EDA and modeling notebooks   (TODO: adjust)
│   └── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

```bash
# 1. Clone
git clone https://github.com/RohitR-26/Telecom_Customer_Churn.git
cd Telecom_Customer_Churn/telco-churn-analytics

# 2. Create a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch
jupyter notebook
```

---

## 🧰 Tech Stack

<p align="center">
  <img src="https://skillicons.dev/icons?i=python,pandas,numpy,sklearn,jupyter,git,github,vscode" alt="Tech stack icons"/>
</p>

---

## 🔮 Future Work

- [ ] Hyperparameter tuning and cross-validation
- [ ] Handle class imbalance (SMOTE or class weights)
- [ ] Deploy as a Streamlit web app
- [ ] Build a Power BI / Tableau dashboard

---

## 👨‍💻 Author

**Rohit R** — [@RohitR-26](https://github.com/RohitR-26)

<p align="center">
  ⭐ If you found this useful, consider giving the repo a star!
</p>