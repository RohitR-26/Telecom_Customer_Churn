# 06. Predictive Modelling — Decision Tree

**Project:** Telco Customer Churn Analytics
**Syllabus link:** Sessions 13-16 (Predictive Modelling, Supervised Segmentation, Trees)
**Written by:** ______________  **Reviewed by:** ______________

---

## Part 1: Understanding the Concept

A **Decision Tree** is a model that predicts an outcome by asking a series of yes/no questions about the features, splitting the data step by step until it reaches a final prediction.

| Term | Meaning |
|---|---|
| **Node** | One question/split in the tree (e.g., "Contract <= 0.5?") |
| **Root node** | The very first, most important split |
| **Leaf node** | The final box at the bottom — the prediction |
| **Supervised Segmentation** | Splitting the data into groups using the target variable (Churn) to guide which splits matter |
| **Informative Attribute** | A feature that best separates churners from non-churners at a given split |
| **Gini** | A number (0 to 0.5) measuring how "mixed" a node is — 0 means a node is pure (all one class), higher means more mixed |
| **feature_importances_** | A score showing how much each feature contributed to the tree's decisions overall |

### Why a tree over Naive Bayes?

Naive Bayes assumes features act independently. A tree can capture **interactions** — for example, "Contract matters, but *among* month-to-month customers, tenure matters a lot more than it does for two-year customers." Trees naturally model that kind of "it depends" logic.

---

## Part 2: Our Approach (What, Why, How)

| | Our Approach |
|---|---|
| **What** | Train a Decision Tree using both categorical and numeric features to predict churn, then inspect the tree's structure and feature importances |
| **Why** | To compare against Naive Bayes, and to get a visual, explainable model — trees are easy to show a manager and say "here's exactly why we predict this" |
| **How** | scikit-learn's `DecisionTreeClassifier`, with `max_depth` limited to avoid an overly complex tree, then `plot_tree()` and `feature_importances_` to interpret it |

---

## Part 3: What Are We Using? (The Tools)

| Tool | What It Is | Why We Use It Here |
|---|---|---|
| **`DecisionTreeClassifier`** | scikit-learn's tree model | Handles both categorical (encoded) and numeric features together |
| **`max_depth`** | Limits how many splits deep the tree can go | Prevents the tree from growing too complex and memorizing the training data (a first defense against overfitting, covered fully in Step 7) |
| **`plot_tree()`** | Draws the tree structure visually | Makes the model's logic explainable, not a black box |
| **`feature_importances_`** | Scores each feature's contribution | Tells us which "informative attributes" mattered most, directly answering the Session 14 concept |

---

## Part 4: Code, Explanation, and Insight

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Use more features now, including the numeric ones — trees handle mixed types well
features = ["Contract", "InternetService", "PaymentMethod", "SeniorCitizen",
            "tenure", "MonthlyCharges"]
X = df[features].copy()
y = df["Churn_flag"]

# Encode remaining text columns
for col in ["Contract", "InternetService", "PaymentMethod", "SeniorCitizen"]:
    X[col] = LabelEncoder().fit_transform(X[col])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

**Why we used this setup:** Same encoding and split logic as Step 5, but now we include numeric features (`tenure`, `MonthlyCharges`) directly — unlike Naive Bayes, a tree doesn't need them binned or assumed independent.

```python
# max_depth limits how many splits deep the tree can go.
# Without a limit, trees tend to memorize the training data (overfitting) —
# we cover this formally in Step 7, but we cap it here as good practice.
tree_model = DecisionTreeClassifier(max_depth=4, random_state=42)
tree_model.fit(X_train, y_train)

y_pred_tree = tree_model.predict(X_test)
print(f"Decision Tree Accuracy: {accuracy_score(y_test, y_pred_tree):.3f}")
```

**Our result:** **Accuracy = 0.793** (79.3%), compared to Naive Bayes' 77.2%.

**Insight we take:** The tree outperforms Naive Bayes by about 2 percentage points. This gain likely comes from the tree's ability to combine features in sequence (e.g., first split on Contract, then on MonthlyCharges *within* that group), rather than treating every feature as independent.

```python
# plot_tree() visualizes the actual splits the model learned —
# great for explaining "why" the model predicts what it does.
plt.figure(figsize=(16, 8))
plot_tree(tree_model, feature_names=features, class_names=["No Churn", "Churn"],
          filled=True, max_depth=3, fontsize=8)
plt.show()
```

**What it's showing:** The root node splits on `Contract <= 0.5` (i.e., month-to-month vs. not), immediately confirming Contract is the single most decisive factor. Below that, the tree further splits on `MonthlyCharges` and `tenure`, showing how the model refines its prediction using combinations of features.

**Insight we take:** Reading the left branch (month-to-month customers): those with `tenure <= 13.5` and mid-to-high `MonthlyCharges` land in a leaf where the majority class flips to **"Churn"** — this is a very specific, actionable segment: *new, month-to-month customers paying more than a certain amount are the tree's highest-risk group.*

```python
# feature_importances_ tells us which features the tree relied on most —
# this is the tree's version of "informative attributes"
importances = pd.Series(tree_model.feature_importances_, index=features)
importances.sort_values(ascending=False)
```

**Our result:**

| Feature | Importance |
|---|---|
| Contract | 0.588 |
| tenure | 0.170 |
| MonthlyCharges | 0.164 |
| InternetService | 0.078 |
| SeniorCitizen | 0.0002 |
| PaymentMethod | 0.000 |

**Insight we take:** `Contract` alone accounts for nearly 59% of the tree's decision-making power — matching what EDA, the manual probability, and Naive Bayes all pointed to independently. `PaymentMethod` contributed nothing once other features were considered, meaning it's a candidate to drop from a simpler, cleaner model.

---

## Part 5: Interview Questions and Answers

**Q1: How does a Decision Tree decide where to split?**
> At each step, it tries every possible split on every feature and picks the one that best separates the classes (lowest Gini impurity, or highest information gain). It repeats this process at each resulting node until it reaches a stopping condition, like `max_depth`.

**Q2: What is Gini impurity?**
> A measure of how mixed a node is. A Gini of 0 means the node is pure — every sample belongs to one class. Higher Gini means the node has a more even mix of both classes, meaning the split hasn't fully separated them yet.

**Q3: Why limit `max_depth`?**
> An unlimited tree keeps splitting until nodes are pure, which usually means it starts memorizing quirks of the training data rather than learning general patterns — this is overfitting. Limiting depth forces the tree to generalize better.

**Q4: Why did the Decision Tree outperform Naive Bayes here?**
> Naive Bayes assumes features are independent, but in our data, Contract, tenure, and MonthlyCharges interact — a tree can model "if Contract is month-to-month AND tenure is low AND charges are high" as one combined rule, which Naive Bayes cannot represent directly.

**Q5: What does it mean when a feature has zero importance?**
> It means the tree never found that feature useful for splitting once the other features were already available — the information it carried was either irrelevant to churn or already captured by another feature.

---

## Part 6: What We Can Learn More

| Topic | Why It's Worth Learning |
|---|---|
| **Random Forest** | Combines many decision trees to reduce overfitting and improve accuracy — a natural next step after a single tree |
| **Pruning** | Techniques to trim a tree after it's built, instead of only limiting depth beforehand |
| **Entropy / Information Gain** | An alternative to Gini for measuring split quality, worth knowing since interviewers sometimes ask for the difference |
| **Rules extraction** | A trained tree can be converted into a simple set of if-then business rules, useful for non-technical stakeholders |

---

## What We Observed

| Metric | Value |
|---|---|
| Decision Tree Accuracy | 0.793 |
| Naive Bayes Accuracy (Step 5) | 0.772 |
| Root split | Contract <= 0.5 |
| Top feature (importance) | Contract (0.588) |
| Least useful feature | PaymentMethod (0.000) |

**Insight sentence:** *"The Decision Tree improved accuracy to 79.3% over Naive Bayes' 77.2% by capturing feature interactions, and confirmed across every method so far — EDA, manual probability, Naive Bayes, and now the tree — that Contract type is the dominant driver of churn, with tenure and MonthlyCharges as secondary factors."*

---

## Key Terms

| Term | Meaning |
|---|---|
| **Decision Tree** | A model that predicts using a sequence of yes/no splits on features |
| **Root node** | The first, most important split in the tree |
| **Leaf node** | A final prediction at the end of a branch |
| **Gini impurity** | A measure of how mixed the classes are in a node |
| **max_depth** | A limit on how many levels deep the tree can grow |
| **feature_importances_** | A score of how much each feature contributed to the model's decisions |

---

## Interview One-Liner

> "I moved from Naive Bayes to a Decision Tree to capture feature interactions, and the tree confirmed Contract type as the dominant churn driver — I also limited tree depth upfront as a first safeguard against overfitting."