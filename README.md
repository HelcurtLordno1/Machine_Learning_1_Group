# 📖 Machine_Learning_1_Group  
## Sentiment Analysis Project – Session 5  

### 📌 Overview  
This project focuses on implementing and comparing different approaches to **Sentiment Analysis** on Twitter data.  
The work is divided among five members (A–E), each responsible for specific tasks, and concludes with a unified **UI demo** and **final report**.  

---

## 👥 Task Assignment  

### 🅰️ Bạn A – Tasks 1–3  

**Task 1: Logistic Regression (scikit-learn vs. from-scratch)**  
- Load Twitter dataset (train/test CSV).  
- Train Logistic Regression (`sklearn.linear_model.LogisticRegression`) with features = positive/negative frequency.  
- Parameters: `solver='liblinear'`, `max_iter=10000`.  
- Evaluate metrics: accuracy, precision, recall, F1.  
- Compare with scratch version (loss curves, runtime, accuracy).  
- **Comment:** scikit-learn is faster/easier; scratch offers more control.  

**Task 2: Debugging "divide by zero" in Gradient Descent**  
- Issue: very small learning rate (1e-9) + many iterations (100k) → unstable sigmoid/log cost.  
- Fix: increase learning rate (e.g., `1e-6`) or add epsilon (`1e-10`) in log.  
- Re-run, plot loss, and confirm stability.  

**Task 3: Normalizing Positive/Negative Frequencies**  
- Normalize with denominator `N = train_set_length * sentence_length`.  
- Train new model, evaluate, and plot scatter features before/after normalization.  
- **Comment:** reduces bias from sentence length but may shrink values too much → risk of underfitting.  

---

### 🅱️ Bạn B – Tasks 4–5  

**Task 4: Feature Scaling**  
- Techniques: `MinMaxScaler` and `StandardScaler` (from `sklearn.preprocessing`).  
- Compare: no scaling vs. Min-Max vs. Standard.  
- **Comment:** scaling speeds up convergence and may boost accuracy.  

**Task 5: Rule-based Decision Function**  
- Simple rule: predict **positive** if `pos_freq > neg_freq`.  
- Evaluate precision on test set.  
- Compare with Logistic Regression.  
- **Comment:** logistic is superior because it learns weights + bias, rule-based is too simplistic.  

---

### 🅲 Bạn C – Task 6  

**Task 6: Extended Feature Engineering (6 features)**  
- Base features: positive/negative frequency.  
- Add 4 new features:  
  1. Presence of specific keywords.  
  2. Pronoun count.  
  3. Binary indicator (if none of the above).  
  4. `log(sentence_length)`.  
- Train logistic model, analyze feature importance, evaluate results.  
- **Comment:** more features may improve accuracy but risk overfitting.  

---

### 🅳 Bạn D – Task 7  

**Task 7: Trying Multiple ML Models**  
- Models: SVM, Random Forest, XGBoost, Naive Bayes.  
- Train with both base and extended features.  
- Tune hyperparameters with `GridSearchCV`.  
- Compare metrics and select the best (expected: XGBoost).  
- **Comment:** ensemble/complex models may outperform Logistic Regression on complex datasets.  

---

### 🅴 Bạn E – Task 8  

**Task 8: Benchmark with LLM (e.g., ChatGPT)**  
- Use API to classify tweets.  
- Measure precision on test set.  
- Compare with custom ML models.  
- **Comment:** LLMs provide higher accuracy thanks to contextual understanding, but are slower and costly.  

---

### 🤝 Joint Task – Task 9 (UI & Report)  

**UI (by A, B, C, D):**  
- Build demo app using **Streamlit**.  
- Input: text box for a tweet.  
- Backend: load best-trained model (from Task 7).  
- Output: sentiment prediction (Positive/Negative) + confidence score.  
- Add comparison tab for different models.  
- Deploy with Streamlit Sharing.  

**Report (by E):**  
- Collect all results, plots, and comments.  
- Structure: **Introduction – Methods – Results – Conclusion**.  
- Compare best model vs. LLM benchmark.  
- Deliver in **PDF/Markdown** format.  
- **Deadline:** 25/09/2025.  

---

## 📂 Deliverables  
- ✅ Source code (per task)  
- ✅ Streamlit UI app (`app.py`)  
- ✅ Final report (`report.pdf` / `report.md`)  
