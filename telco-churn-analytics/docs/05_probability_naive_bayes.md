# 05. Probability and Naive Bayes

**Project:** Telco Customer Churn Analytics
**Syllabus link:** Sessions 6-7 (Sample & Population, Probability, Bayes' Theorem)
**Written by:** ______________  **Reviewed by:** ______________

---

## Part 1: Understanding the Concept

**Probability** is a number between 0 and 1 that measures how likely an event is. In data analytics, we use it to move from "what happened in our sample" to "how confident can we be about the pattern."

| Term | Meaning | Example in Our Project |
|---|---|---|
| **Population** | The entire group we care about | All Telco customers, ever |
| **Sample** | The subset of data we actually have | Our 7,043-row dataset |
| **Joint Probability** | Chance two events happen together | P(Contract=Month-to-month AND Churn=Yes) |
| **Conditional Probability** | Chance of one event *given* another already happened | P(Churn=Yes \| Contract=Month-to-month) |
| **Marginal Probability** | Chance of one event alone, ignoring everything else | P(Churn=Yes) overall |
| **Bayes' Theorem** | A rule for updating a probability when new evidence (a feature) is known | Used inside Naive Bayes to combine multiple features |
| **Naive Bayes** | A classifier built on Bayes' Theorem, assuming all features are independent of each other ("naive" assumption) | Our churn prediction model in this step |

### Why "naive"?

Naive Bayes assumes each feature affects the outcome **independently** of the others — e.g., it assumes `Contract` and `InternetService` don't interact. In reality, they probably do a little. That assumption is rarely 100% true, but the model is still fast, simple, and often surprisingly effective — which is why it's a common first model to try.

---

## Part 2: Our Approach (What, Why, How)

| | Our Approach |
|---|---|
| **What** | First calculate one conditional probability manually to build intuition, then train a full Naive Bayes classifier using multiple categorical features to predict churn |
| **Why** | Calculating one probability by hand shows exactly what the model is doing internally, before we let scikit-learn do it automatically at scale across all features |
| **How** | Manual conditional probability with pandas filtering and `.mean()`; then `CategoricalNB` from scikit-learn, after encoding text categories into numbers and splitting the data into train/test sets |

---

## Part 3: What Are We Using? (The Tools)

| Tool | What It Is | Why We Use It Here |
|---|---|---|
| **`LabelEncoder`** | Converts text categories into numbers | Naive Bayes needs numeric input, not text |
| **`train_test_split`** | Splits data into a training set and a test set | Lets us check how the model performs on data it has never seen |
| **`stratify=y`** | Keeps the same class ratio in train and test | Important here because churn is imbalanced (73%/27%) |
| **`CategoricalNB`** | Naive Bayes variant built for categorical (non-numeric-origin) features | Fits our features (Contract, InternetService, etc.), which are categories, not continuous numbers |

---

## Part 4: Code, Explanation, and Insight

### A. Manual Conditional Probability (building intuition)

```python
# Conditional probability: P(Churn=Yes | Contract=Month-to-month)
# We filter the dataframe to ONLY month-to-month customers, then take the
# mean of Churn_flag within that filtered group — same logic as our
# earlier groupby, but now explicitly framed as a conditional probability.
p_churn_given_mtm = df[df["Contract"] == "Month-to-month"]["Churn_flag"].mean()
print(f"P(Churn | Month-to-month) = {p_churn_given_mtm:.3f}")
```

**Why we used this:** Before letting a model calculate probabilities across many features, we calculate one by hand to be sure we understand exactly what "conditional probability" means in practice.

**Our result:** **P(Churn \| Month-to-month) = 0.427**

**Insight we take:** Given that a customer is on a month-to-month contract, there is a 42.7% chance they churn — matching what we saw visually in the EDA bar chart, but now expressed as a precise probability rather than just "the orange bar looks big."

### B. Preparing Data for the Model

```python
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Pick a few categorical features that showed strong patterns in EDA
features = ["Contract", "InternetService", "PaymentMethod", "SeniorCitizen"]
X = df[features].copy()
y = df["Churn_flag"]

# Naive Bayes needs numbers, not text, so we encode each category as an integer.
# LabelEncoder assigns each unique text value a number (e.g. "DSL"->0, "Fiber optic"->1)
encoders = {}
for col in features:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le   # save it, so we can decode predictions later if needed

# Split data: 80% to train the model, 20% to test it on unseen data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# stratify=y keeps the same churn ratio in both train and test sets,
# important because our data is imbalanced (73%/27%)
```

**Why we used this setup:** Models can't read text, so encoding is required. `train_test_split` with `stratify=y` ensures we test the model fairly on data it never saw, with the same churn balance as the full dataset.

### C. Training and Testing the Model

```python
# Train the Naive Bayes model on the training data
model = CategoricalNB()
model.fit(X_train, y_train)

# Predict on unseen test data
y_pred = model.predict(X_test)

# Quick accuracy check (we'll cover better metrics in Step 7)
from sklearn.metrics import accuracy_score
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
```

**Why we used this:** `.fit()` teaches the model the probability patterns in the training data. `.predict()` then applies what it learned to new, unseen customers. Accuracy gives a first rough sense of performance.

**Our result:** **Accuracy = 0.772** (77.2%)

**Insight we take:** Using just four categorical features, the model correctly classifies about 77% of customers. This is a reasonable first model, but remember from Step 3: since ~73% of customers don't churn anyway, a model that *always* predicts "No churn" would already score ~73% accuracy. So 77.2% is only a modest improvement over doing nothing — this is exactly why we'll use better metrics (precision, recall, ROC-AUC) in Step 7, not accuracy alone.

---

## Part 5: Interview Questions and Answers

**Q1: What is the difference between conditional probability and joint probability?**
> Conditional probability is the chance of one event given that another has already happened, like P(Churn | Month-to-month). Joint probability is the chance of both events happening together, like P(Churn AND Month-to-month), without assuming one is already known.

**Q2: Why is Naive Bayes called "naive"?**
> Because it assumes every feature is independent of every other feature when predicting the outcome. In real data, features often do interact, so this assumption isn't perfectly true, but the model still tends to work well and is very fast to train.

**Q3: Why did you compare your model's accuracy to the majority class baseline?**
> Because with imbalanced data, a model can score high accuracy just by always predicting the majority class, without learning anything useful. Comparing against that baseline shows whether the model is actually adding value.

**Q4: Why use `stratify=y` when splitting the data?**
> To make sure both the training and test sets keep the same proportion of churned vs non-churned customers as the full dataset. Without it, a random split could accidentally put too few churn cases in the test set, making evaluation unreliable.

**Q5: What is Bayes' Theorem, in simple terms?**
> It's a formula for updating the probability of something once you know some evidence. For example, updating "chance this customer churns" once you know their contract type, using the relationship between the evidence and the outcome learned from past data.

---

## Part 6: What We Can Learn More

| Topic | Why It's Worth Learning |
|---|---|
| **GaussianNB** | A Naive Bayes variant for continuous numeric features (like tenure), instead of only categorical ones |
| **Laplace smoothing** | Handles the case where a category combination never appeared in training data, avoiding a zero probability |
| **Feature independence testing** | Ways to check how much the "naive" assumption is actually being violated in your data |

---

## What We Observed

| Result | Value |
|---|---|
| P(Churn \| Month-to-month) | 0.427 |
| Model accuracy | 0.772 |
| Majority-class baseline (approx.) | ~0.73 (share of non-churners) |
| Real improvement over baseline | ~4 percentage points |

**Insight sentence:** *"Our Naive Bayes model reaches 77.2% accuracy using just four categorical features, only modestly above the 73% majority-class baseline, which tells us accuracy alone isn't a strong enough metric here and we need precision/recall/ROC-AUC to properly judge the model."*

---

## Key Terms

| Term | Meaning |
|---|---|
| **Population** | The entire group of interest |
| **Sample** | The subset of data we actually observe |
| **Conditional probability** | Probability of an event given another has occurred |
| **Bayes' Theorem** | Formula for updating probability based on evidence |
| **Naive Bayes** | Classifier using Bayes' Theorem with an independence assumption between features |
| **Label encoding** | Converting text categories into numbers |
| **Baseline (majority-class)** | The accuracy achieved by always predicting the most common class |

---

## Interview One-Liner

> "I calculate a manual conditional probability first to confirm I understand what the model is doing, then train Naive Bayes on categorical features — and I always compare model accuracy against the majority-class baseline, since a raw accuracy number can be misleading on imbalanced data."