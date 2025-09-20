# Machine_Learning_1_Group
Sentiment Analysis Project – Session 5
📌 Overview

This project focuses on implementing and comparing different approaches to Sentiment Analysis on Twitter data.
The work is divided among five members (A–E), each responsible for specific tasks, and concludes with a unified UI demo and report.

👥 Task Assignment
Bạn A: Tasks 1–3

Task 1: Logistic Regression with scikit-learn vs. from-scratch implementation

Load Twitter dataset (train/test CSV).

Train Logistic Regression (sklearn.linear_model.LogisticRegression) with features = positive/negative frequency.

Set solver = 'liblinear', max_iter = 10000.

Evaluate (accuracy, precision, recall, F1).

Compare with scratch version: loss curves, runtime, accuracy.

Comment pros/cons: sklearn is faster/easier, scratch offers full control.

Task 2: Debugging "divide by zero" with gradient descent

Issue: very small learning rate (1e-9) + many iterations (100k) → unstable sigmoid/log cost.

Fix: increase learning rate (e.g. 1e-6) or add epsilon (1e-10) in log.

Re-run, plot loss, and show stability.

Task 3: Normalizing positive/negative frequencies

Normalize with denominator N = train_set_length * sentence_length.

Train new model, evaluate, plot feature scatter before/after normalization.

Comment: may reduce bias from sentence length, but could shrink values too much (risk underfitting).

Bạn B: Tasks 4–5

Task 4: Apply feature scaling

Techniques: Min-Max Scaler and Standard Scaler (from sklearn.preprocessing).

Compare performance across: no scaling vs. Min-Max vs. Standard.

Comment: scaling helps convergence and sometimes boosts accuracy.

Task 5: Rule-based decision function

Simple rule: predict positive if pos_freq > neg_freq.

Evaluate precision on test set.

Compare with logistic regression: logistic is superior because it learns weights and bias.

Bạn C: Task 6

Task 6: Feature engineering with 6 features

Original features: pos/neg frequency.

Add 4 new features:

Presence of specific keywords.

Pronoun count.

Binary indicator if none of the above.

log(sentence length).

Train logistic model, analyze feature importance, evaluate improvements.

Comment: more features can increase accuracy but risk overfitting.

Bạn D: Task 7

Task 7: Try multiple ML models

Models: SVM, Random Forest, XGBoost, Naive Bayes.

Train with base and extended features.

Use GridSearchCV for tuning.

Compare metrics and choose the best model (expected: XGBoost).

Comment: ensemble/complex models may outperform logistic regression on harder data.

Bạn E: Task 8

Task 8: Benchmark with LLM (e.g., ChatGPT)

Use API to classify tweets.

Measure precision on test set.

Compare with custom models.

Comment: LLMs achieve higher accuracy thanks to context understanding, but are costly/slower.

Joint Task: Task 9 – UI & Report

UI (A, B, C, D together):

Build demo app using Streamlit.

Input: text box for a tweet.

Backend: load best-trained model (from Task 7).

Output: sentiment (Positive/Negative), confidence score, feature visualization.

Add model comparison tab.

Deploy with Streamlit Sharing.

Report (E):

Consolidate results, plots, and comments into a structured report (PDF/Markdown).

Sections: Introduction, Methods, Results, Conclusion (best model, LLM vs. custom).

Final deadline: 25/09/2025.
