# Telco Customer Churn Analytics

**A team learning project by CDAC AI (PGCP-AI, CDAC, Kharghar)**
Built to apply Python, Data Analytics and Power BI concepts end-to-end on a real business problem, with concept-by-concept documentation for learning and interview prep.

---

## Business Problem

A telecom company is losing customers (churn). This project analyzes customer data to find **why** customers leave, tests which factors matter statistically, and builds a model to **predict** who is likely to leave next — so the business can act before losing them.

**Dataset:** Telco Customer Churn (IBM sample, via Kaggle) — ~7,043 customers, 21 features.

---

## Tech Stack

| Area | Tools |
|---|---|
| Data cleaning & analysis | Python, Pandas, NumPy |
| Statistics | SciPy, manual calculations |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | scikit-learn (Naive Bayes, Decision Tree) |
| Dashboard | Power BI (Power Query + DAX) |
| Environment | Google Colab |

---

## Project Structure

```
telco-churn-analytics/
├── data/
│   └── TelcoCustomerChurn.csv
├── notebooks/
│   ├── 01_load_and_understand.ipynb
│   ├── 02_cleaning.ipynb
│   ├── 03_eda_visualization.ipynb
│   ├── 04_descriptive_statistics.ipynb
│   ├── 05_probability_naive_bayes.ipynb
│   ├── 06_decision_tree.ipynb
│   └── 07_model_evaluation.ipynb
├── docs/
│   ├── 01_analytics_lifecycle.md
│   ├── 02_data_cleaning.md
│   ├── 03_eda_visualization.md
│   ├── 04_descriptive_statistics.md
│   ├── 05_probability_naive_bayes.md
│   ├── 06_decision_tree.md
│   └── 07_model_evaluation.md
├── dashboard/
│   └── telco_churn_dashboard.pbix   (in progress)
└── README.md
```

---

## Progress So Far

| Step | Topic | Syllabus Sessions | Status | Docs |
|---|---|---|---|---|
| 1 | Analytics Life Cycle & Data Loading | 1-3 | ✅ Done | [docs/01_analytics_lifecycle.md](docs/01_analytics_lifecycle.md) |
| 2 | Data Cleaning | 3-4 | ✅ Done | [docs/02_data_cleaning.md](docs/02_data_cleaning.md) |
| 3 | EDA & Visualization | 4 | ✅ Done | [docs/03_eda_visualization.md](docs/03_eda_visualization.md) |
| 4 | Descriptive Statistics | 5 | ✅ Done | [docs/04_descriptive_statistics.md](docs/04_descriptive_statistics.md) |
| 5 | Probability & Naive Bayes | 6-7 | ✅ Done | [docs/05_probability_naive_bayes.md](docs/05_probability_naive_bayes.md) |
| 6 | Predictive Modelling — Decision Tree | 13-16 | ✅ Done | [docs/06_decision_tree.md](docs/06_decision_tree.md) |
| 7 | Model Evaluation | 21-22 | ✅ Done | [docs/07_model_evaluation.md](docs/07_model_evaluation.md) |
| 8 | Power BI Dashboard | 25 | 🔄 In Progress | — |

Each doc in `docs/` follows the same structure: **Concept → Our Approach → Tools Used → Code with Explanations → Insights → Interview Q&A → Key Terms**, so any teammate can revise from it before an interview without re-reading the whole codebase.

---

## Key Results So Far

| Finding | Value |
|---|---|
| Overall churn rate | ~26.5% |
| Highest-churn contract type | Month-to-month (42.7% churn rate) |
| Highest-churn internet service | Fiber optic |
| Strongest correlation with churn | Tenure (-0.35) |
| Naive Bayes model accuracy | 77.2% |
| Decision Tree model accuracy | 79.3% |
| Decision Tree ROC-AUC | 0.826 |
| Decision Tree recall (Churn class) | 0.47 |
| Cross-validation mean / std | 0.787 / 0.010 (stable) |
| Most important feature (Decision Tree) | Contract (58.8% of model's decision power) |

**Headline insight:** Month-to-month, low-tenure, higher-paying customers are the highest churn-risk segment. Our Decision Tree model ranks churn risk well (ROC-AUC 0.826) but currently misses over half of actual churners (recall 0.47) at the default threshold — a real deployment would lower this threshold to catch more at-risk customers.

---

## How to Run

1. Open any notebook in `notebooks/` in Google Colab
2. Upload `data/WA_Fn-UseC_-Telco-Customer-Churn.csv` when prompted
3. Run cells top to bottom — each notebook is self-contained per step
4. Refer to the matching file in `docs/` for the concept explanation behind each code block

---

## Team

| Name | Role / Sessions Owned |
|---|---|
| ______________ | ______________ |
| ______________ | ______________ |
| ______________ | ______________ |

---

## Next Steps

- [ ] Finish Power BI dashboard (KPI cards, contract/internet service charts, tenure analysis)
- [ ] Add regression/forecasting and simulation/optimization modules (Sessions 17-20) as Phase 2
- [ ] Individual projects (each member builds a solo project using the same approach)
- [ ] LinkedIn post + GitHub polish