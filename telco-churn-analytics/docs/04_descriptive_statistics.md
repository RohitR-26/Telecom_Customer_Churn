# 04. Descriptive Statistics

**Project:** Telco Customer Churn Analytics
**Syllabus link:** Session 5 (Descriptive Statistical Measures, Central Tendency & Dispersion)
**Written by:** ______________  **Reviewed by:** ______________

---

## Part 1: Understanding the Concept

**Descriptive statistics** means summarizing a column of numbers into a few representative values, instead of looking at every single row one by one.

| Measure Type | What It Tells You | Examples |
|---|---|---|
| **Central Tendency** | Where the "middle" of the data is | Mean, Median, Mode |
| **Dispersion** | How spread out the data is | Range, IQR, Variance, Std Dev, Coefficient of Variation (CV) |

### Why does this matter?

- A dataset of 7,000 rows can't be read row by row. A handful of summary numbers give a fast, reliable snapshot.
- These numbers form the foundation for every later step: outlier detection, hypothesis testing, and model evaluation all build on central tendency and dispersion.
- Comparing statistics **between groups** (e.g., churned vs stayed) turns plain summary numbers into business insight.

---

## Part 2: Our Approach (What, Why, How)

| | Our Approach |
|---|---|
| **What** | Calculate central tendency and dispersion for `tenure`, `MonthlyCharges`, `TotalCharges`, both overall and split by `Churn` |
| **Why** | Single summary numbers reveal typical customer behavior, and splitting by churn status connects statistics directly to our business question |
| **How** | pandas' `describe()`, `.mode()`, `.quantile()` for the standard measures, plus a manual coefficient of variation and a `groupby()` comparison |

---

## Part 3: Code, Explanation, and Insight

```python
# describe() automatically calculates count, mean, std, min, 25%, 50%,
# 75%, and max for every numeric column in one line
df[["tenure", "MonthlyCharges", "TotalCharges"]].describe()
```

**Why we used this:** The fastest way to get most central tendency and dispersion measures together, instead of calling six separate functions.

**Insight we take:** A quick sense of how varied the customer base is, by comparing the "typical" customer (median) against the extremes (min/max).

```python
# Mode is not included in describe(), because it applies more naturally
# to categorical data. mode() returns the most frequently occurring value(s).
df["tenure"].mode()[0]
df["Contract"].mode()[0]
```

**Why:** Mean and median tell us the "center" for numbers, but mode tells us the single most common value — useful for categorical columns like Contract, where mean/median don't make sense.

**Our result:** Mode of `Contract` = **Month-to-month** — confirms this is the most common contract type in the whole dataset.

```python
# Quartiles split the data into four equal parts. Q1 (25th percentile) is
# the value below which 25% of the data falls; Q3 (75th percentile) is
# the value below which 75% of the data falls. IQR = Q3 - Q1 covers the
# middle 50% of the data, and is more robust to outliers than min/max.
Q1 = df["MonthlyCharges"].quantile(0.25)
Q3 = df["MonthlyCharges"].quantile(0.75)
IQR = Q3 - Q1
print(f"Q1: {Q1}, Q3: {Q3}, IQR: {IQR}")
```

**Why:** IQR is the standard way to define "normal range" before flagging outliers — the measure Session 10 will use formally.

**Our result:** Q1 = 35.5, Q3 = 89.85, IQR = 54.35. The middle 50% of customers pay between $35.5 and $89.85 per month.

```python
# Coefficient of Variation (CV) = (Std Dev / Mean) * 100
# It expresses spread as a PERCENTAGE of the mean, so we can compare
# variability across columns with very different scales
# (e.g. comparing spread of tenure in months vs charges in dollars).
cv_tenure = (df["tenure"].std() / df["tenure"].mean()) * 100
cv_charges = (df["MonthlyCharges"].std() / df["MonthlyCharges"].mean()) * 100
print(f"CV Tenure: {cv_tenure:.2f}%, CV MonthlyCharges: {cv_charges:.2f}%")
```

**Why:** Standard deviation alone can't be compared across columns with different units. CV solves this by making spread relative to the mean.

**Our result:** CV Tenure = 75.87%, CV MonthlyCharges = 46.46%. Tenure varies much more relative to its average than charges do — customer *lifespan* is far less predictable than what they pay.

```python
# Compare descriptive stats for churned vs non-churned customers separately —
# this connects statistics directly back to our business question.
df.groupby("Churn")[["tenure", "MonthlyCharges", "TotalCharges"]].mean()
```

**Why we used this:** The most useful version of descriptive stats here — not just "what's normal overall," but "what's different about customers who churn."

**Our result:**

| Churn | Avg tenure | Avg MonthlyCharges | Avg TotalCharges |
|---|---|---|---|
| No | 37.57 | 61.27 | 2549.91 |
| Yes | 17.98 | 74.44 | 1531.80 |

**Insight we take:** Churned customers have **less than half** the average tenure of retained customers, and pay about **$13 more per month** on average. This suggests new, higher-paying customers are the highest churn-risk group — a strong candidate for early-tenure retention offers.

---

## Part 4: Interview Questions and Answers

**Q1: What's the difference between variance and standard deviation?**
> Variance is the average squared distance from the mean. Standard deviation is the square root of variance, which brings it back to the same unit as the original data, making it easier to interpret directly.

**Q2: When would you use median instead of mean?**
> When the data is skewed or has outliers, since the mean gets pulled toward extreme values while the median stays representative of the typical value.

**Q3: What does a high coefficient of variation mean?**
> The data has high relative variability compared to its mean, meaning it's less consistent, even if the raw standard deviation looks small in absolute terms.

**Q4: Why calculate statistics separately for churned vs non-churned customers instead of just overall?**
> Overall statistics hide group differences. Splitting by the target variable shows whether a feature actually behaves differently for the group we care about, which is what makes the statistic useful for the business question, not just descriptive.

**Q5: What is IQR used for, beyond just describing spread?**
> It's the standard basis for outlier detection — values below Q1 - 1.5*IQR or above Q3 + 1.5*IQR are typically flagged as outliers, which we'll apply formally in the correlation and outliers step.

---

## Part 5: What We Can Learn More

| Topic | Why It's Worth Learning |
|---|---|
| **Skewness** | A precise number for how lopsided a distribution is, rather than just eyeballing a histogram (covered formally in Session 12) |
| **Trimmed mean** | A mean calculated after removing a small percentage of extreme values, useful when outliers distort the regular mean |
| **Percentiles beyond quartiles** | e.g., 90th or 95th percentile, useful for identifying "top-tier" customers by spend |

---

## What We Observed

| Statistic | Our Result |
|---|---|
| Mode of `Contract` | Month-to-month |
| MonthlyCharges: Q1 / Q3 / IQR | 35.5 / 89.85 / 54.35 |
| CV Tenure | 75.87% |
| CV MonthlyCharges | 46.46% |
| Avg tenure — stayed vs churned | 37.57 vs 17.98 |
| Avg MonthlyCharges — stayed vs churned | 61.27 vs 74.44 |

**Insight sentence:** *"Churned customers have less than half the average tenure and pay about $13/month more than retained customers, suggesting new, higher-paying customers are the highest churn risk — a candidate for early-tenure retention offers."*

---

## Key Terms

| Term | Meaning |
|---|---|
| **Mean** | The arithmetic average |
| **Median** | The middle value when data is sorted |
| **Mode** | The most frequently occurring value |
| **Quartile** | One of three points that split sorted data into four equal parts |
| **IQR** | Interquartile Range = Q3 - Q1, the spread of the middle 50% of data |
| **Variance** | Average squared distance from the mean |
| **Standard Deviation** | Square root of variance, same unit as the original data |
| **Coefficient of Variation (CV)** | Standard deviation expressed as a percentage of the mean |

---

## Interview One-Liner

> "I use descriptive statistics to summarize each feature, and I always compare them across the target groups, not just overall, because that's what reveals which features actually relate to the outcome we're predicting."