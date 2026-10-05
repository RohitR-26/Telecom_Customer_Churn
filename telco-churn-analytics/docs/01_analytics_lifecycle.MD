# 01. Data Analytics Life Cycle

**Project:** Telco Customer Churn Analytics
**Syllabus link:** Sessions 1-3 (Business Analytics, Life Cycle, Nature of Data)
**Written by:** ______________  **Reviewed by:** ______________

---

## Part 1: Understanding the Concept

**The Data Analytics Life Cycle** is a step-by-step path that every analytics or ML project follows — from understanding the business problem all the way to putting the final solution to real use.

Think of it like building a house: you don't start pouring concrete before you have a blueprint. Analytics works the same way — you don't start modelling before you know *what question you're answering* and *whether your data can even answer it*.

| Phase | Simple Meaning | Example in Our Project |
|---|---|---|
| **1. Discovery** | Understand the business problem and goal | "Why are telecom customers leaving?" |
| **2. Data Preparation** | Collect, clean and organize the data | Load the CSV, fix wrong data types, handle missing values |
| **3. Model Planning** | Decide which methods and tools to use | Choose stats tests, Naive Bayes, Decision Tree |
| **4. Model Building** | Build and test the models | Train a churn prediction model |
| **5. Quality Assurance and Documentation** | Check results and write things down | Cross-validation, this documentation |
| **6. Management Approval** | Present findings to decision makers | Power BI dashboard and summary |
| **7. Installation, Acceptance and Operation** | Put it into real use and monitor it | Company uses the model to find at-risk customers |

### Two supporting ideas from Session 3

**Analysis vs Reporting**

| | Reporting | Analysis |
|---|---|---|
| **Question** | "What happened?" | "Why did it happen, and what next?" |
| **Output** | Tables, dashboards, summaries | Insights, predictions, recommendations |
| **Example** | "27% of customers churned" | "Customers on month-to-month plans churn more, so offer them yearly deals" |

**Nature of Data** — knowing a column's type decides which chart, statistic, or test can be used on it later.

| Type | Meaning | Example in Our Data |
|---|---|---|
| **Numeric (continuous)** | Measurable numbers | `MonthlyCharges`, `tenure` |
| **Categorical** | Groups or labels | `Contract`, `gender`, `InternetService` |
| **Binary** | Only two values | `Churn` (Yes / No) |

---

## Part 2: Our Approach (What, Why, How)

| | Our Approach |
|---|---|
| **What** | Define the business problem first, set up our project structure, then load and take a first look at the raw data |
| **Why** | Jumping straight into code without a clear problem statement leads to analysis with no direction — we want every later chart, test, and model to trace back to one clear question |
| **How** | Write a one-line problem statement, create a structured GitHub repo, then use pandas to load the CSV and run basic inspection commands (`shape`, `info`, `describe`) |

---

## Part 3: What Are We Using? (The Tools)

| Tool | What It Is | Why We Use It Here |
|---|---|---|
| **Google Colab** | A free, cloud-based Jupyter notebook environment | No local setup needed, and it's the standard environment used in our labs |
| **pandas** | Python's core library for working with tabular data | Reads the CSV into a DataFrame — a table we can inspect, clean, and analyze |
| **GitHub repo structure** | Organized folders (`data/`, `notebooks/`, `docs/`, `dashboard/`) | Keeps the project navigable for teammates and recruiters, not just one giant notebook |

---

## Part 4: Code, Explanation, and Insight

### Problem Statement (written before any code)

> A telecom company is losing customers. We will analyze customer data to find why customers leave (churn), test which factors matter statistically, and build a model to predict who is likely to leave.

**Why we write this first:** It becomes our compass — every later step should help answer "why do customers leave" and "who is likely to leave next." If a chart or test doesn't help answer that, it's a distraction, not an insight.

### Setup

- GitHub repo: `telco-churn-analytics` with folders `data/`, `notebooks/`, `docs/`, `dashboard/`
- Dataset: Telco Customer Churn (Kaggle, IBM sample)

### Load and First Look at the Data

```python
import pandas as pd
import numpy as np

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

print(df.shape)                 # rows, columns
```

**Why we used this:** `.shape` instantly tells us the size of the dataset — how many customers (rows) and how many features (columns) we're working with, before doing anything else.

**Our result:** **7,043 rows x 21 columns.**

**Insight we take:** A dataset this size is fully explorable by hand — we don't need heavy sampling techniques, and it's small enough to run on Google Colab's free tier without performance issues.

```python
df.head()                       # first 5 rows
```

**Why we used this:** A quick visual sanity check — do the columns and values look like what we expect (real customer records), or is something obviously broken (wrong delimiter, shifted columns, garbage values)?

```python
df.info()                       # column names, data types, non-null counts
```

**Why we used this:** Shows every column's name, its data type (`int64`, `float64`, `object`), and how many non-null values it has — this is the single most useful command for spotting data quality issues early.

**Our result:** `TotalCharges` shows as `object` (text), even though it should clearly be a number.

**Insight we take:** This is exactly the kind of issue the Data Preparation phase exists to catch. If we had skipped straight to statistics or modelling, calculations on `TotalCharges` would have silently failed or errored out later — catching it here, in Discovery, saves debugging time downstream.

```python
df.describe()                   # summary of numeric columns
```

**Why we used this:** Gives count, mean, min, max and percentiles for every numeric column in one line — a first rough sense of the data's scale (though we cover this properly in Step 4: Descriptive Statistics).

```python
df["Churn"].value_counts()      # how many customers left vs stayed
```

**Why we used this:** Our target variable is `Churn` — before analyzing *why* customers churn, we need to know *how many* churn at all.

**Our result:** 5,174 "No" (stayed) vs 1,869 "Yes" (churned).

**Insight we take:** The split is roughly 73% vs 27% — this confirms right away that we have an **imbalanced classification problem**. This single observation shapes decisions all the way through the project: it's why we can't rely on accuracy alone later (Step 5 onward), and why we'll need metrics like recall and ROC-AUC (Step 7).

---

## Part 5: Interview Questions and Answers

**Q1: What are the phases of the data analytics life cycle?**
> Discovery, Data Preparation, Model Planning, Model Building, Quality Assurance and Documentation, Management Approval, and Installation/Acceptance/Operation. It's not strictly one-way — you often go back to an earlier phase when you find new issues.

**Q2: What is the difference between analysis and reporting?**
> Reporting answers "what happened," usually as tables or dashboards. Analysis answers "why it happened and what to do next" — it produces insights and recommendations, not just numbers.

**Q3: Why do you look at data types before analysis?**
> Because the wrong data type can silently break calculations or give wrong results. In our project, `TotalCharges` loaded as text — if we'd tried to average it without noticing, pandas would have either errored out or given a meaningless answer.

**Q4: What is an imbalanced dataset, and why does it matter?**
> It's when one class is much more common than the other — in our case, 73% "No churn" vs 27% "Churn." It matters because a model can score high accuracy just by always predicting the majority class, without learning anything useful, so we need different evaluation metrics.

**Q5: Why start with a business problem statement instead of jumping straight into the data?**
> Without a clear question, it's easy to produce charts and statistics that are technically correct but don't lead anywhere useful. Starting with the business question keeps every later step focused and gives us a way to judge whether an insight actually matters.

---

## Part 6: What We Can Learn More

| Topic | Why It's Worth Learning |
|---|---|
| **CRISP-DM** | A widely used industry variant of this same life cycle (Business Understanding → Data Understanding → Data Preparation → Modeling → Evaluation → Deployment) — worth knowing the name, since interviewers sometimes ask for it by this term |
| **Data dictionaries** | A formal document describing every column, its type, and its meaning — useful to create for larger real-world datasets with many columns |
| **Sampling techniques** | For datasets too large to explore in full, methods to work with a representative subset (not needed here, but common at scale) |

---

## What We Observed

| Check | Our Result |
|---|---|
| Rows x Columns | 7,043 rows x 21 columns |
| Churn = Yes / No | 5,174 "No" (stayed) vs 1,869 "Yes" (churned) |
| Any column with a wrong data type? | Yes — `TotalCharges` loaded as `object` (text) instead of a number |

**Insight sentence:** *"The dataset is a manageable 7,043 rows with a clearly imbalanced target (73% stayed vs 27% churned), and one data type issue (`TotalCharges` as text) that we need to fix before any numeric analysis — both facts directly shape our approach in every step that follows."*

---

## Key Terms

| Term | Meaning |
|---|---|
| **Churn** | A customer leaving the company |
| **Dataset** | A table of data (rows = customers, columns = features) |
| **Feature** | One column used to describe each row |
| **Target** | The column we want to predict (here: Churn) |
| **Imbalanced data** | One class is much larger than the other (more "No" than "Yes") |
| **DataFrame** | pandas' name for a table-like data structure (rows and columns) |

---

## Interview One-Liner

> "I start every project by defining the business problem, then I understand and prepare the data before choosing a model, following the analytics life cycle — and I check data types and class balance right at the start, since both shape every decision later in the project."