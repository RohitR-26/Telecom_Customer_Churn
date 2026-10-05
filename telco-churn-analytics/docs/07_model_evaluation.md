# 07. Model Evaluation

**Project:** Telco Customer Churn Analytics
**Syllabus link:** Sessions 21-22 (Overfitting, Cross-Validation, Evaluating Classifiers)
**Written by:** ______________  **Reviewed by:** ______________

---

## Part 1: Understanding the Concept

**Accuracy alone is misleading**, especially with imbalanced data like ours (73% stayed, 27% churned). Model evaluation is about looking deeper than "how often was it right overall" to understand exactly **what kind of mistakes** the model makes, and whether it will generalize to new, unseen customers.

| Term | Meaning |
|---|---|
| **Confusion Matrix** | A 2x2 (or larger) grid comparing actual vs predicted classes |
| **True Positive (TP)** | Predicted Churn, actually Churn (correct catch) |
| **True Negative (TN)** | Predicted No Churn, actually No Churn (correct pass) |
| **False Positive (FP)** | Predicted Churn, actually No Churn (false alarm) |
| **False Negative (FN)** | Predicted No Churn, actually Churn (missed churner — usually the costliest mistake in churn problems) |
| **Precision** | Of everyone predicted to churn, how many actually did? (TP / (TP+FP)) |
| **Recall** | Of everyone who actually churned, how many did we catch? (TP / (TP+FN)) |
| **F1-score** | A single number balancing precision and recall |
| **ROC-AUC** | Measures how well the model ranks churners above non-churners, across all possible decision thresholds, not just one |
| **Overfitting** | When a model performs well on training data but poorly on new data, because it memorized noise instead of learning general patterns |
| **Cross-Validation** | Testing the model on several different train/test splits to get a more reliable, stable performance estimate |

### Why precision and recall matter more than accuracy here

In a churn problem, **missing a churner (False Negative)** usually costs more than a false alarm (False Positive) — a missed churner is lost revenue, while a false alarm just means an unnecessary retention offer. This is why **recall** on the Churn class deserves special attention, not just overall accuracy.

---

## Part 2: Our Approach (What, Why, How)

| | Our Approach |
|---|---|
| **What** | Evaluate the Decision Tree from Step 6 using a confusion matrix, precision/recall/F1, ROC-AUC, and 5-fold cross-validation |
| **Why** | To understand the *type* of errors the model makes (not just how many), and to confirm the model's performance is stable, not a lucky single split |
| **How** | scikit-learn's `confusion_matrix`, `classification_report`, `roc_auc_score`, and `cross_val_score` |

---

## Part 3: What Are We Using? (The Tools)

| Tool | What It Is | Why We Use It Here |
|---|---|---|
| **`confusion_matrix`** | Builds the TP/TN/FP/FN grid | Shows exactly which mistakes the model makes |
| **`classification_report`** | Prints precision, recall, F1 per class | Quick, complete summary in one call |
| **`predict_proba` + `roc_auc_score`** | Uses predicted probabilities, not just yes/no | Measures ranking quality across all thresholds, giving a fuller picture than one fixed cutoff |
| **`cross_val_score`** | Repeats train/test splits 5 times | Checks if performance is consistent, and is a first practical check against overfitting |

---

## Part 4: Code, Explanation, and Insight

```python
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_score

# Confusion matrix: shows actual vs predicted counts in a 2x2 grid
# rows = actual class, columns = predicted class
cm = confusion_matrix(y_test, y_pred_tree)
print(cm)

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix — Decision Tree")
plt.show()
```

**Our result:**

| | Predicted: No Churn | Predicted: Churn |
|---|---|---|
| **Actual: No Churn** | 941 (TN) | 94 (FP) |
| **Actual: Churn** | 198 (FN) | 176 (TP) |

**Insight we take:** Out of 374 customers who actually churned, the model correctly caught only 176 and **missed 198** — more than half. This is the single most important number in this step: the model is much better at confirming who *stays* than at catching who *leaves*.

```python
# classification_report gives precision, recall, and F1-score per class in one go
print(classification_report(y_test, y_pred_tree, target_names=["No Churn", "Churn"]))
```

**Our result:**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| No Churn | 0.83 | 0.91 | 0.87 |
| **Churn** | **0.65** | **0.47** | **0.55** |

**Insight we take:** Precision (0.65) is decent — when the model flags churn, it's right about two-thirds of the time. But **recall (0.47)** is weak — it only catches 47% of actual churners. For a business trying to prevent churn, this recall gap means over half of at-risk customers go completely unflagged by this model as-is.

```python
# predict_proba gives the probability of churn (not just yes/no),
# which roc_auc_score needs to measure how well the model ranks
# churners above non-churners across all thresholds.
y_proba = tree_model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_proba)
print(f"ROC-AUC: {auc:.3f}")
```

**Our result:** **ROC-AUC = 0.826**

**Insight we take:** This is a strong score (0.5 = random guessing, 1.0 = perfect). It tells us the model is actually quite good at *ranking* customers by churn risk overall — the weak recall above is really about where we set the decision threshold (currently 0.5 by default), not that the model lacks useful signal. In a real project, we could lower this threshold to flag more customers as "at risk," trading some precision for much better recall.

```python
# cross_val_score trains and tests the model 5 separate times on different
# slices of the data, so we get a more reliable performance estimate
# than a single train/test split — and it shows if performance is unstable.
cv_scores = cross_val_score(tree_model, X, y, cv=5, scoring="accuracy")
print(f"Cross-val scores: {cv_scores}")
print(f"Mean: {cv_scores.mean():.3f}, Std: {cv_scores.std():.3f}")
```

**Our result:** Scores: [0.789, 0.778, 0.774, 0.794, 0.800] → **Mean = 0.787, Std = 0.010**

**Insight we take:** The very low standard deviation (0.010) means the model performs almost identically across five different data splits — this is a good sign of **stability**, not overfitting to one lucky train/test split. The mean (0.787) is also close to our original single-split accuracy (0.793), confirming that earlier result wasn't a fluke.

---

## Part 5: Interview Questions and Answers

**Q1: Why is accuracy misleading on imbalanced data?**
> Because a model can score high accuracy just by always predicting the majority class. In our data, always predicting "No Churn" would already give ~73% accuracy without learning anything — so accuracy alone doesn't tell us if the model is actually useful.

**Q2: What's the difference between precision and recall, and when would you prioritize one over the other?**
> Precision measures how many predicted positives were actually correct; recall measures how many actual positives were caught. In churn prediction, missing a real churner (low recall) usually costs more than a false alarm (low precision), so I'd prioritize recall — even if it means accepting more false positives.

**Q3: What is ROC-AUC measuring that accuracy doesn't?**
> ROC-AUC measures how well the model ranks positive cases above negative cases across all possible decision thresholds, not just the default 0.5 cutoff. A high ROC-AUC with weak recall at 0.5 often means the model has good signal, but the threshold needs adjusting.

**Q4: What is overfitting, and how did you check for it here?**
> Overfitting is when a model learns the training data too specifically, including its noise, so it performs worse on new data. I checked for it using 5-fold cross-validation — the very low standard deviation across folds (0.010) shows the model generalizes consistently rather than performing well on just one lucky split.

**Q5: If recall is low, what would you do next?**
> I'd try lowering the classification threshold below 0.5 to flag more customers as at-risk, try a different model like Random Forest, or use class-weighting techniques that penalize the model more for missing churners during training.

---

## Part 6: What We Can Learn More

| Topic | Why It's Worth Learning |
|---|---|
| **Threshold tuning** | Adjusting the 0.5 cutoff based on precision/recall trade-offs to match business priorities |
| **Class weighting / SMOTE** | Techniques to handle imbalanced data during training, not just during evaluation |
| **Precision-Recall curve** | More informative than ROC-AUC specifically when classes are imbalanced |
| **Random Forest / Gradient Boosting** | Ensemble methods that typically improve on a single Decision Tree's recall and stability |

---

## What We Observed

| Metric | Value |
|---|---|
| Confusion Matrix | TN=941, FP=94, FN=198, TP=176 |
| Precision (Churn) | 0.65 |
| Recall (Churn) | 0.47 |
| F1 (Churn) | 0.55 |
| ROC-AUC | 0.826 |
| Cross-val Mean / Std | 0.787 / 0.010 |

**Insight sentence:** *"Our model has strong overall ranking ability (ROC-AUC 0.826) and stable performance across folds (std 0.010), but weak recall (0.47) means it misses over half of actual churners — for a real retention campaign, we'd lower the decision threshold to catch more at-risk customers, accepting a precision trade-off."*

---

## Key Terms

| Term | Meaning |
|---|---|
| **Confusion Matrix** | Grid comparing actual vs predicted classes |
| **False Negative** | Missed a real positive case (here: missed churner) |
| **Precision** | Accuracy of positive predictions |
| **Recall** | Coverage of actual positive cases |
| **ROC-AUC** | Ranking quality across all thresholds |
| **Overfitting** | Model memorizes training data, fails to generalize |
| **Cross-Validation** | Testing across multiple data splits for a reliable estimate |

---

## Interview One-Liner

> "I never judge a model on accuracy alone, especially with imbalanced data — I look at precision, recall, and ROC-AUC together, and use cross-validation to confirm the performance is stable, not a one-off lucky split."