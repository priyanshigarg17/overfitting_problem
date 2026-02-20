

# 📌 Feature Selection using SelectFromModel (L1 vs L2 Regularization)

## 🚀 Overview

This project explains how **SelectFromModel** from Scikit-learn can be used for feature selection using linear models with L1 and L2 regularization.

The primary objective is to understand:

* How model-based feature selection works
* Why L1 regularization naturally performs feature selection
* How L2 regularization can still be used with threshold-based selection
* When to prefer L1 over L2 in practical machine learning scenarios

---

## 🧠 What is SelectFromModel?

SelectFromModel is a meta-transformer used for feature selection.

It selects features based on importance weights learned by an estimator.
The estimator must expose either:

* Coefficients (for linear models)
* Feature importances (for tree-based models)

The method keeps only those features whose importance values exceed a defined threshold.

---

## 🔬 Understanding Regularization

Regularization is used to:

* Prevent overfitting
* Improve model generalization
* Control model complexity

Two commonly used types are:

### 🔹 L1 Regularization (Lasso)

* Adds absolute value penalty to the loss function
* Shrinks some coefficients to exactly zero
* Produces sparse models
* Performs natural feature selection

Because some coefficients become zero, irrelevant features are automatically removed.

---

### 🔹 L2 Regularization (Ridge)

* Adds squared penalty to the loss function
* Shrinks coefficients toward zero
* Rarely makes coefficients exactly zero
* Reduces model variance

L2 does not inherently remove features, but reduces their influence.

---

## ⚖️ L1 vs L2 Comparison

| Aspect                      | L1 Regularization | L2 Regularization |
| --------------------------- | ----------------- | ----------------- |
| Shrinks Coefficients        | Yes               | Yes               |
| Produces Exact Zeros        | Yes               | Rarely            |
| Automatic Feature Selection | Yes               | No                |
| Handles Correlated Features | Moderate          | Better            |
| Model Stability             | Moderate          | High              |

---

## 🎯 Does SelectFromModel Work with L2?

Yes, SelectFromModel works with L2 regularization.

However:

* Since L2 rarely produces zero coefficients,
* Feature selection depends on a chosen threshold,
* Features with small coefficient magnitudes can be removed manually based on that threshold.

Therefore, L2-based feature selection is threshold-driven, not sparsity-driven.

---

## 📊 Key Observations

* L1 is ideal for high-dimensional datasets where feature reduction is important.
* L2 is better when features are correlated and model stability is required.
* SelectFromModel is most effective with L1 for true feature elimination.
* With L2, careful threshold tuning is required.

---

## 📌 When to Use Each Approach

### Use L1 When:

* You need feature selection.
* You are working with sparse or high-dimensional data.
* Interpretability is important.

### Use L2 When:

* You want stable coefficient estimates.
* Features are highly correlated.
* The focus is on generalization rather than elimination.

---

## 💼 Practical Machine Learning Insight

In real-world ML systems:

* L1 is often used for dimensionality reduction.
* L2 is commonly used to improve generalization performance.
* A combination (Elastic Net) can also be applied when both sparsity and stability are required.

Understanding the trade-offs between L1 and L2 is crucial for building robust and interpretable machine learning models.

