# 02. Data Cleaning

**Project:** Telco Customer Churn Analytics
**Syllabus link:** Sessions 3-4 (Data Preparation, Nature of Data)
**Written by:** ______________  **Reviewed by:** ______________

---

## What is it?

Data cleaning is the process of fixing problems in raw data before we analyze or model it — wrong data types, missing values, duplicate rows, and inconsistent labels.

---

## Why use it?

- Models and statistics give **wrong or misleading results** if the data is dirty.
- Example: if `TotalCharges` is stored as text, we cannot calculate its average — Python will throw an error or give a wrong answer.
- Cleaning usually takes the **most time** in any real project, so it is a core skill, not a small step.

---

## Where is it used?

Every real-world dataset needs cleaning — sensor data, sales data, survey data, customer records. No dataset from a company's live systems arrives ready to model.

---

## How it works (Simple Steps)

1. Find the problems: missing values, duplicates, wrong types, blank strings.
2. Decide the right fix for each problem — not always "delete" or "average," it depends on what the data means.
3. Apply the fix.
4. Re-check that the problem is gone.

---

## How we used it in our project

### Step A — Find the problems

```python
# 1. Missing values per column
# isnull() marks each cell True/False (True = missing)
# sum() then adds up how many True values are in each column
df.isnull().sum()
```

```python
# 2. Duplicate rows and duplicate customer IDs
# duplicated() flags a row as True if it is an exact repeat of an earlier row
# sum() counts how many such duplicate rows exist
print(df.duplicated().sum())

# Do the same check but only on the customerID column,
# because even if two full rows differ, the same customer
# should never appear twice
print(df["customerID"].duplicated().sum())
```

```python
# 3. TotalCharges is usually loaded as text (object), not a number.
# .dtype tells us what type Python thinks the column is.
df["TotalCharges"].dtype

# .str.strip() removes leading/trailing spaces from each text value.
# We then check how many values become an empty string "" —
# these are the hidden blanks causing the problem.
df[df["TotalCharges"].str.strip() == ""].shape
```

### Step B — Fix the problems

```python
# pd.to_numeric() converts a text column into numbers.
# errors="coerce" means: if a value cannot be converted (like a blank space),
# turn it into NaN (Not a Number / missing) instead of crashing the program.
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Now count how many NaN (missing) values were created by that conversion.
print(df["TotalCharges"].isnull().sum())
```

```python
# We checked separately that these blank rows all have tenure = 0,
# meaning the customer is brand new and has not been billed yet.
# So the correct fix here is to fill missing TotalCharges with 0 —
# NOT with the average, because these customers genuinely have not paid anything yet.
# fillna() replaces every NaN in that column with the value we give it.
df["TotalCharges"] = df["TotalCharges"].fillna(0)
```

```python
# Churn is currently text ("Yes"/"No"). Many calculations (correlation,
# statistical tests, model training) need numbers, not text.
# .map() replaces each text value with the number we specify in the dictionary.
# Now Churn_flag = 1 means the customer left, 0 means they stayed.
df["Churn_flag"] = df["Churn"].map({"Yes": 1, "No": 0})

# SeniorCitizen is stored as 0/1 (a number), but every other
# similar column in this dataset (like Partner, Dependents) is stored
# as "Yes"/"No" text. We convert it to match, so it's consistent
# and easier to use in charts and grouping later.
df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
```

### Step C — Confirm the fix worked

```python
# info() shows column names, data types and non-null counts again.
# TotalCharges should now show as float64 (a number), not object (text).
df.info()

# isnull().sum() gives missing values per column;
# adding a second .sum() adds all of those counts together
# into one single number. If it prints 0, the dataset has
# no missing values left anywhere.
df.isnull().sum().sum()   # should be 0
```

---

## Decisions We Made (and why)

| Problem | Fix Chosen | Reason |
|---|---|---|
| `TotalCharges` blank text | Filled with 0 | These customers have `tenure = 0` — genuinely no charges yet |
| `Churn` is Yes/No text | Added `Churn_flag` (1/0) | Statistics and models need numeric targets |
| `SeniorCitizen` is 0/1 | Converted to Yes/No | Keeps it consistent with similar columns, easier for charts |

> **Key lesson:** Cleaning is not "always fill with average" or "always drop missing rows." Look at *why* the value is missing before deciding.

---

## Key Terms

| Term | Meaning |
|---|---|
| **NaN** | "Not a Number" — pandas' way of marking a missing value |
| **dtype** | The data type of a column (e.g., int64, float64, object/text) |
| **Coerce** | Force a conversion; invalid values become NaN instead of crashing |
| **Duplicate row** | A row that is an exact copy of another row |
| **Flag column** | A new column that turns a category into a number (e.g., Yes/No → 1/0) |

---

## What We Observed (fill in after running)

| Check | Our Result |
|---|---|
| Missing `TotalCharges` before fix | ________ |
| Duplicate rows | ________ |
| Duplicate customer IDs | ________ |
| Missing values after cleaning (`isnull().sum().sum()`) | ________ (should be 0) |

---

## Interview One-Liner

> "I check for missing values, wrong data types and duplicates first, and I choose the fix based on what the data actually means — not by applying the same rule everywhere."

## Common Interview Questions

1. What is the difference between `NaN` and an empty string `""`?
2. Why did you fill `TotalCharges` with 0 instead of the mean?
3. What does `errors="coerce"` do in `pd.to_numeric()`?
4. Why convert `Churn` from Yes/No to 1/0?
5. How do you check for duplicate rows in pandas?