# Telco Customer Churn Analytics — Project Explainer & Interview Q&A

Use this as your single revision sheet before any interview or project defense. Read the "How to Explain This Project" section out loud a few times until it feels natural, then go through the Q&A.

---

## How to Explain This Project (30-Second Pitch)

> "I built an end-to-end churn analysis project on telecom customer data — about 7,000 customers. I cleaned the data, explored it to find patterns, ran statistical tests, and built a Decision Tree model to predict which customers are likely to leave. The biggest finding was that contract type is the strongest driver of churn — month-to-month customers churn at 43%, versus under 3% for two-year contracts. My model reaches 79% accuracy and a 0.826 ROC-AUC score, though I found it misses over half of actual churners at the default threshold, which I'd fix by adjusting the decision threshold in a real deployment."

### 2-Minute Version (if asked to go deeper)

> "The business problem was: a telecom company wants to know why customers churn and who's likely to churn next.
>
> I started with the data analytics life cycle — defined the problem, then cleaned the data. One issue was `TotalCharges` loading as text instead of numbers because of blank values; I found those belonged to brand-new customers with zero tenure, so I filled them with 0 instead of blindly using the average.
>
> Next, I did EDA — visualizing churn against contract type, internet service, and tenure. Month-to-month contracts and low tenure stood out immediately as the biggest churn signals.
>
> I backed that up with descriptive statistics — churned customers had less than half the average tenure of retained customers.
>
> Then I moved to modelling. I started with Naive Bayes on categorical features, got 77% accuracy, then moved to a Decision Tree with more features, which improved to 79% and let me see feature importances directly — Contract alone explained almost 59% of the model's decisions.
>
> Finally, I evaluated it properly — not just accuracy, but confusion matrix, precision, recall, and ROC-AUC, because the data is imbalanced. I found recall on the churn class was only 47%, meaning we miss over half of actual churners, even though ROC-AUC (0.826) shows the model ranks risk well. I also used 5-fold cross-validation to confirm the model is stable, not overfitted to one split.
>
> The last step was building a Power BI dashboard to present these findings visually to a non-technical audience."

---

## Q&A by Topic

### 1. Project Overview

| Question | Short Answer |
|---|---|
| What is this project about? | Predicting which telecom customers will churn (leave) using their account and usage data |
| Why does this matter to a business? | Keeping an existing customer is cheaper than acquiring a new one, so predicting churn lets a company act early with retention offers |
| What dataset did you use? | The IBM Telco Customer Churn dataset, ~7,043 rows, 21 columns |
| What was your end-to-end process? | Data Analytics Life Cycle: define problem → clean data → EDA → statistics → modelling → evaluation → dashboard |

### 2. Data Cleaning

| Question | Short Answer |
|---|---|
| What problems did you find in the raw data? | `TotalCharges` loaded as text due to blank values, and a few columns needed type/label fixes |
| Why was `TotalCharges` blank for some rows? | Those customers had `tenure = 0` — brand new, not billed yet |
| Why fill with 0 instead of the average? | Because the blanks weren't random — they represented real customers with genuinely zero charges so far; using the average would have been factually wrong |
| How do you handle duplicates in pandas? | `df.duplicated().sum()` to count them, `df.drop_duplicates()` to remove them |
| What does `errors="coerce"` do in `pd.to_numeric()`? | Converts invalid values to `NaN` instead of crashing the program |

### 3. EDA & Visualization

| Question | Short Answer |
|---|---|
| What is EDA? | Exploring data with charts and summaries to find patterns before modelling |
| What's the difference between univariate and bivariate analysis? | Univariate looks at one column alone; bivariate looks at the relationship between two columns |
| What's the difference between a histogram and a boxplot? | A histogram shows the full distribution shape; a boxplot summarizes it into 5 numbers and flags outliers |
| Why is class imbalance important to notice early? | A model can get high accuracy just by predicting the majority class, so accuracy alone becomes misleading |
| What was your biggest EDA insight? | Month-to-month contracts churn at 42.7%, vs 11.3% (one year) and 2.8% (two year) |

### 4. Descriptive Statistics

| Question | Short Answer |
|---|---|
| What's the difference between mean, median, and mode? | Mean = average, median = middle value when sorted, mode = most frequent value |
| When do you use median instead of mean? | When data is skewed or has outliers, since mean gets pulled toward extreme values |
| What is IQR and why use it? | Interquartile Range (Q3-Q1) — the spread of the middle 50% of data; it's the standard basis for detecting outliers |
| What does Coefficient of Variation (CV) tell you? | Spread relative to the mean, as a percentage — lets you compare variability across columns with different units |
| What did your grouped statistics show? | Churned customers had less than half the average tenure (18 vs 37.6 months) and paid ~$13/month more than retained customers |

### 5. Probability & Naive Bayes

| Question | Short Answer |
|---|---|
| What is conditional probability? | The chance of an event given that another event already happened, e.g. P(Churn \| Month-to-month) |
| What did you calculate manually, and what was the result? | P(Churn \| Month-to-month) = 0.427 — matched the EDA chart exactly |
| Why is Naive Bayes called "naive"? | It assumes every feature is independent of the others, which isn't perfectly true in real data |
| What accuracy did Naive Bayes get? | 77.2% |
| Why is 77.2% not that impressive on its own? | Because always predicting "No Churn" would already give ~73% accuracy, since the data is imbalanced |

### 6. Decision Tree / Predictive Modelling

| Question | Short Answer |
|---|---|
| Why move from Naive Bayes to a Decision Tree? | Trees can capture interactions between features (e.g. Contract + tenure together), unlike Naive Bayes' independence assumption |
| What accuracy did the Decision Tree get? | 79.3%, about 2 points better than Naive Bayes |
| What is `max_depth` and why limit it? | Limits how many splits deep the tree can go; prevents it from memorizing the training data (overfitting) |
| What is Gini impurity? | A measure of how mixed the classes are in a node — 0 means pure, higher means more mixed |
| What was the most important feature? | Contract type, at 58.8% importance — by far the biggest driver |
| Which feature had zero importance? | PaymentMethod — a candidate to drop in a simpler model |

### 7. Model Evaluation

| Question | Short Answer |
|---|---|
| Why not judge a model on accuracy alone? | On imbalanced data, accuracy can look good even when the model isn't actually useful |
| What is a confusion matrix? | A grid comparing actual vs predicted classes: True Positive, True Negative, False Positive, False Negative |
| What's the difference between precision and recall? | Precision = how many predicted positives were correct; Recall = how many actual positives were caught |
| Which matters more for churn prediction, precision or recall? | Recall — missing a real churner (False Negative) usually costs more than a false alarm |
| What was your model's recall on the Churn class, and what does it mean? | 0.47 — the model missed more than half of actual churners |
| What is ROC-AUC measuring? | How well the model ranks churners above non-churners across all thresholds, not just one fixed cutoff |
| What was your ROC-AUC, and what does it tell you? | 0.826 — the model has strong underlying signal, even though default-threshold recall was weak |
| What is overfitting? | When a model performs well on training data but poorly on new, unseen data |
| How did you check for overfitting? | 5-fold cross-validation — very low standard deviation (0.010) showed the model is stable across different data splits |
| If recall is too low, what would you do next? | Lower the classification threshold below 0.5, try class-weighting, or use a stronger model like Random Forest |

### 8. General / Wrap-Up Questions

| Question | Short Answer |
|---|---|
| What was the single biggest insight from this project? | Contract type is the dominant churn driver — confirmed independently by EDA, probability, and the Decision Tree's feature importance |
| What would you do differently with more time? | Try Random Forest or class-weighting to improve recall, and add regression/forecasting for revenue impact |
| What tools did you use? | Python (Pandas, NumPy, Matplotlib, Seaborn, scikit-learn) and Power BI for the dashboard |
| How did you make sure your work is trustworthy? | Cross-validation for stability, comparing model accuracy against a majority-class baseline, and choosing metrics (recall, ROC-AUC) suited to imbalanced data |
| What business action would you recommend from this analysis? | Target month-to-month, low-tenure, higher-paying customers with retention offers or incentives to move to longer contracts |

---

## Tips for Using This Sheet

- Practice the 30-second pitch until you don't need to read it — that's usually the first thing an interviewer asks.
- If asked a "why" question, always connect back to the **business reason**, not just the technical step.
- If you don't remember an exact number, it's fine to say "around X%" — interviewers usually care more about your reasoning than the decimal point.