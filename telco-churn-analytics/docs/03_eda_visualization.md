# 03. Exploratory Data Analysis (EDA) and Visualization

**Project:** Telco Customer Churn Analytics
**Syllabus link:** Session 4 (Visualization and Exploring Data)
**Written by:** ______________  **Reviewed by:** ______________

---

## Part 1: Understanding EDA (The Big Topic)

### What is EDA?

**Exploratory Data Analysis (EDA)** is the practice of examining a dataset — through charts, summary numbers, and simple questions — to understand its structure, patterns, relationships and problems, **before** we apply statistics or build any model.

It is called "exploratory" because we don't yet know what we're looking for. We are exploring, the same way a detective looks around a crime scene before forming a theory.

### Why does EDA exist as its own step?

| Reason | Explanation |
|---|---|
| **Numbers alone hide patterns** | A table of 7,000 rows means nothing to a human eye. A chart shows the pattern in one glance |
| **It guides every later step** | EDA tells us which columns matter, which need tests, which need transformation |
| **It catches hidden problems** | Outliers, wrong values, or unexpected patterns often only show up visually |
| **It builds business understanding** | We move from "here is data" to "here is what the data is telling us about the business" |

### The Building Blocks of EDA

| Building Block | Question It Answers | Typical Tool |
|---|---|---|
| **Univariate analysis** | What does *one* column look like on its own? | Histogram, boxplot, countplot |
| **Bivariate analysis** | How do *two* columns relate to each other? | Grouped bar chart, scatter plot, correlation |
| **Distribution check** | Are values spread normally, skewed, or clustered? | Histogram, skewness |
| **Outlier detection** | Are there extreme, unusual values? | Boxplot |
| **Relationship with the target** | Which columns actually relate to what we want to predict (Churn)? | Grouped bar chart, groupby + mean, correlation heatmap |

EDA in our project touches **all five** of these building blocks.

---

## Part 2: Our Approach (What, Why, How)

| | Our Approach |
|---|---|
| **What we are doing** | Visually and numerically exploring the cleaned Telco churn dataset, focusing on the target variable `Churn` |
| **Why we are doing it this way** | We start broad (overall churn rate) then look at individual columns (univariate) then compare columns against Churn (bivariate) then check numeric relationships (correlation). This mirrors how an analyst thinks: general to specific to relationships |
| **How we are doing it** | Using `matplotlib` for chart control and `seaborn` for quick, clean statistical charts; using `pandas.groupby()` for precise numeric summaries alongside the visuals |

---

## Part 3: What Are We Using? (The Tools)

| Tool | What It Is | Why We Use It Here |
|---|---|---|
| **Matplotlib** | Python's core, low-level plotting library | Gives full control over chart details (titles, size, layout) |
| **Seaborn** | Built on top of Matplotlib, designed for statistical charts | Shorter code for common charts like countplot, boxplot, heatmap — and looks better by default |
| **pandas `.groupby()`** | Splits data into groups and summarizes each group | Lets us calculate the exact churn rate per category (e.g., per contract type), which a chart alone can't give precisely |

---

## Part 4: Code, Charts, and Insights

### Setup

```python
import matplotlib.pyplot as plt
import seaborn as sns

# This makes seaborn's charts use a clean, readable style (light grey gridlines)
# by default, so we don't have to style every single chart manually.
sns.set_style("whitegrid")
```

---

### Chart 1 — Churn Distribution (Univariate, the target variable)

```python
# countplot() counts how many rows fall into each category of a column
# and draws one bar per category. Here: how many customers churned vs stayed.
sns.countplot(x="Churn", data=df)
plt.title("Customers: Stayed vs Churned")
plt.show()

# value_counts() counts rows per category (same numbers as the chart above).
# normalize=True converts those counts into proportions (0 to 1) instead of
# raw counts, and multiplying by 100 turns that into a percentage —
# e.g. 0.265 becomes 26.5%, which is easier for a business reader to understand.
print(df["Churn"].value_counts(normalize=True) * 100)
```

**Why we used this chart:** It's the very first thing to check — before analyzing *why* customers churn, we need to know *how many* churn at all.

**What it's showing:** Two bars — one for "No" (stayed), one for "Yes" (churned) — with the height of each bar equal to the number of customers in that group.

**Insight we take:** The dataset is **imbalanced** — roughly 73% stayed and 27% churned. This matters later: if we just guessed "no one churns" every time, we'd still be 73% "accurate," which is misleading. So we can't judge our model on accuracy alone later (Session 22 will cover this properly).

---

### Chart 2 — Numeric Feature Distributions (Univariate)

```python
# hist() draws a histogram: it groups a numeric column's values into
# "bins" (equal-width ranges) and shows how many rows fall in each bin
# as a bar. figsize sets the overall image size in inches (width, height).
df[["tenure", "MonthlyCharges", "TotalCharges"]].hist(figsize=(10, 6), bins=30)
plt.tight_layout()   # automatically spaces out subplots so titles/labels don't overlap
plt.show()
```

**Why we used this chart:** To understand the shape of our three main numeric columns before using them in any statistics or model — e.g., are most customers new or long-term?

**What it's showing:** Three histograms side by side, one for `tenure`, one for `MonthlyCharges`, one for `TotalCharges`. Each bar's height = number of customers in that value range.

**Insight we take:** `tenure` is often spread across the full range with spikes at very low and very high values (many brand-new customers, many long-term ones, fewer in between). `MonthlyCharges` often clusters around specific price points (service bundles). `TotalCharges` tends to be right-skewed (many small values, a long tail of large values) since it grows with tenure.

```python
# boxplot() shows the spread of one numeric column using five numbers:
# minimum, 25th percentile, median, 75th percentile, maximum —
# plus it marks individual points that fall far outside this range as outliers.
sns.boxplot(x=df["MonthlyCharges"])
plt.title("Monthly Charges — Spread and Outliers")
plt.show()
```

**Why we used this chart:** To get an early visual check for outliers before we formally handle them in Session 10.

**What it's showing:** A horizontal "box" covering the middle 50% of monthly charges, a line marking the median, "whiskers" extending to normal-range values, and dots beyond the whiskers marking potential outliers.

**Insight we take:** `MonthlyCharges` usually shows few or no extreme outliers — most values sit within a reasonable business range (services typically cost between $18 and $120). This tells us we likely won't need heavy outlier removal on this column.

---

### Chart 3 — Churn by Contract Type (Bivariate — the most important chart)

```python
# hue="Churn" splits each Contract bar into two sub-bars — "stayed" and
# "churned" — so we can directly compare churn behavior across
# contract types in one chart.
sns.countplot(x="Contract", hue="Churn", data=df)
plt.title("Churn by Contract Type")
plt.show()
```

**Why we used this chart:** Contract type is a strong business lever — the company can change contract offers, unlike things like gender or age. This chart tests whether it's worth investigating further.

**What it's showing:** Three groups of bars (Month-to-month, One year, Two year), each split into "stayed" vs "churned" counts.

**Insight we take:** Month-to-month customers almost always show the highest churn by a large margin — because there's no lock-in, they can leave anytime. This becomes our first strong, actionable business insight.

---

### Chart 4 — Churn by Internet Service Type (Bivariate)

```python
sns.countplot(x="InternetService", hue="Churn", data=df)
plt.title("Churn by Internet Service Type")
plt.show()
```

**Why we used this chart:** To check if a specific service (DSL, Fiber optic, or no internet) is linked to churn — this could point to pricing or quality issues with a specific product line.

**What it's showing:** Three service-type groups, split by churn status, same structure as Chart 3.

**Insight we take:** Fiber optic customers often show higher churn than DSL customers, despite fiber typically being the "better" and more expensive service — worth flagging as a possible pricing or service-quality issue for the business team.

---

### Precise Numbers — Churn Rate per Contract Type

```python
# groupby("Contract") splits the whole dataframe into three smaller
# groups, one per contract type. ["Churn_flag"] then selects just that
# column from each group. .mean() calculates the average of 1s and 0s —
# since Churn_flag is 1 (left) or 0 (stayed), this average IS the churn rate.
# Example: a group of [1, 0, 0, 1] has mean 0.5, meaning 50% churned in that group.
df.groupby("Contract")["Churn_flag"].mean().sort_values(ascending=False)
```

**Why we used this:** Charts are great for spotting patterns, but this line gives the **exact percentage**, which we need for the report, dashboard, and any interview explanation ("churn rate was X% for month-to-month customers").

**What it's showing:** A short table, one row per contract type, sorted from highest to lowest churn rate.

**Insight we take:** This confirms and quantifies Chart 3's pattern with a precise number we can quote confidently.

---

### Chart 5 — Correlation Preview (Numeric relationships)

```python
# We only include numeric columns here, since correlation can only be
# calculated between numbers, not text categories.
numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges", "Churn_flag"]

# corr() calculates a correlation coefficient (-1 to +1) between every
# pair of numeric columns. heatmap() then draws this as a colored grid.
# annot=True prints the actual numbers inside each cell.
# cmap="coolwarm" colors negative values blue and positive values red,
# making strong relationships easy to spot visually.
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Preview")
plt.show()
```

**Why we used this chart:** To get a fast, visual first look at which numeric columns move together with `Churn_flag`, before we run a formal Pearson correlation test in Session 10.

**What it's showing:** A grid where each cell's color and number represent how strongly two columns are related — close to +1 (strong same-direction), close to -1 (strong opposite-direction), close to 0 (no clear relationship).

**Insight we take:** `tenure` usually shows a noticeably negative correlation with `Churn_flag` — the longer a customer stays, the less likely they are to churn. This becomes a strong candidate feature for our prediction model later.

---

## Part 5: Interview Questions and Answers

**Q1: What is EDA and why do you do it before modelling?**
> EDA is exploring a dataset through charts and summaries to understand its structure and patterns before building any model. I do it first because it tells me which features actually matter, what shape the data has, and whether there are problems like outliers or class imbalance, all of which shape my modelling decisions later.

**Q2: What's the difference between a histogram and a boxplot?**
> A histogram shows the full distribution shape of a numeric column: how many values fall into each range. A boxplot summarizes that same column into five key numbers (min, 25th percentile, median, 75th percentile, max) and highlights outliers. I use histograms to see shape and boxplots to quickly spot outliers.

**Q3: What's the difference between univariate and bivariate analysis?**
> Univariate analysis looks at one column alone, like the distribution of tenure. Bivariate analysis looks at the relationship between two columns, like tenure versus churn. I usually do univariate first to understand each column, then bivariate to find relationships with the target variable.

**Q4: How do you read a correlation heatmap?**
> Each cell shows a number from -1 to +1 for a pair of columns. Values near +1 mean they increase together, values near -1 mean one increases as the other decreases, and values near 0 mean there's no clear straight-line relationship. I look for cells close to +1 or -1 against my target variable, since those are the strongest candidate features.

**Q5: Why is class imbalance in the target variable important to notice early?**
> If one class (like "stayed") is much larger than the other, a model can get high accuracy just by always predicting the majority class, without learning anything useful. Noticing this in EDA tells me to use metrics like precision, recall, or ROC-AUC instead of relying on accuracy alone.

**Q6: What's the difference between Matplotlib and Seaborn?**
> Matplotlib is the core, lower-level plotting library in Python; it gives full control but needs more code. Seaborn is built on top of Matplotlib and is designed for statistical charts, so common plots like boxplots or heatmaps take just one line and look cleaner by default.

**Q7: If two columns show high correlation with each other (not with the target), what would you do?**
> That's a sign of multicollinearity, meaning the two columns carry similar information. I'd consider keeping only one of them for certain models (like linear regression), since redundant features can distort coefficients, though tree-based models handle this better.

---

## Part 6: What We Can Learn More (Going Beyond This Step)

| Topic to Explore Later | Why It's Worth Learning |
|---|---|
| **Pairplots (`sns.pairplot`)** | Shows relationships between *all* numeric column pairs at once, useful for a fuller bivariate view |
| **Skewness and Kurtosis** | Formal numbers for "how lopsided" or "how peaked" a distribution is, beyond just eyeballing a histogram (this connects directly to Session 12) |
| **Violin plots** | Combine a boxplot and a distribution shape in one chart, showing more detail than a boxplot alone |
| **Automated EDA tools** | Libraries like `ydata-profiling` or `sweetviz` generate a full EDA report in one line — good for speed, but you should still understand what's happening underneath |
| **Feature interactions** | Checking whether *combinations* of two categorical columns (e.g., Contract + InternetService together) show even sharper churn patterns |

---

## What We Observed

| Question | Our Finding |
|---|---|
| Overall churn rate | ~26.5% (5,174 stayed vs 1,869 churned) |
| Highest-churn contract type | Month-to-month (churn rate 42.7%, vs 11.3% for One year, 2.8% for Two year) |
| Highest-churn internet service type | Fiber optic (largest orange bar relative to blue, well above DSL and "No internet") |
| Column most correlated with `Churn_flag` | `tenure`, at -0.35 (the strongest correlation of any feature with churn) |

**Insight sentences**

1. About 1 in 4 customers (26.5%) have churned, confirming this is a real and sizeable business problem worth investing in, not a rare edge case.
2. Month-to-month customers churn at 42.7%, roughly 4x the rate of one-year customers and 15x the rate of two-year customers, which suggests contract length is the single strongest lever the company can pull to reduce churn.
3. Fiber optic customers show a noticeably higher churn rate than DSL customers despite fiber being the premium service, which suggests a pricing or service-quality issue specific to fiber that the business should investigate.
4. `tenure` has the strongest correlation with churn (-0.35) among all numeric features, which suggests early-tenure customers are the highest-risk group and should be prioritized in any retention campaign.
5. `TotalCharges` correlates strongly with `tenure` (0.83), which makes sense since total charges accumulate over time, but this also means the two features carry overlapping information, worth keeping in mind when we build the predictive model later.

---

## Key Terms

| Term | Meaning |
|---|---|
| **Univariate** | Looking at one column at a time |
| **Bivariate** | Looking at the relationship between two columns |
| **Histogram** | A chart showing how many values fall into each range (bin) of a numeric column |
| **Boxplot** | A chart showing the median, spread and outliers of a numeric column |
| **Countplot** | A bar chart of how many rows fall into each category |
| **Hue** | Splitting a chart's bars/points by a second category, for comparison |
| **Correlation** | A number from -1 to +1 showing how strongly two numeric columns move together |
| **Imbalanced data** | When one outcome (e.g. "stayed") is far more common than the other ("churned") |
| **Skewness** | A measure of how lopsided a distribution is (long tail on one side) |

---

## Interview One-Liner

> "Before modelling, I do univariate analysis to understand each column, then bivariate analysis to find which columns relate to the target — this tells me which features and tests to focus on next."