# Imbalanced Text Classification for Disaster Tweet Detection

**Team Members:** Kola Pranathi, Kommula Devamani, Bokka Roshini Sreeja

---

## 1. Introduction

The goal of this project is to build and compare binary text classification models on the **Disaster Tweets** dataset, which exhibits significant class imbalance (majority non‑disaster, minority disaster). Class imbalance often causes models to be biased toward the majority class, leading to poor recall for the minority class – a critical issue when detecting real disasters.

We implement three classifiers: **Logistic Regression**, **Random Forest**, and a **Neural Network with a fixed architecture** (64→32→16). For each, we evaluate performance **without** imbalance handling, then apply **two different techniques**: class weighting (or random oversampling) and SMOTE. Performance is measured using accuracy, precision, recall, F1‑score, ROC‑AUC, PR‑AUC, and confusion matrices.

---

## 2. Dataset Description

The dataset is sourced from Kaggle: [Disaster Tweets](https://www.kaggle.com/datasets/vstepanenko/disaster-tweets). It contains **11,370 tweets**, each labeled as:

- **0** – Non‑disaster tweet
- **1** – Real disaster tweet

**Class distribution:**

- Non‑disaster: 9,256 (81.4%)
- Disaster: 2,114 (18.6%)

**Imbalance ratio:** 4.38 : 1 (majority:minority)

This imbalance is substantial and will affect model learning unless handled properly.

---

## 3. Preprocessing

The following text preprocessing steps were applied to each tweet:

1. **Lowercasing** – convert all characters to lowercase.
2. **URL removal** – remove `http://` or `https://` links.
3. **Punctuation & number removal** – keep only alphabetic characters.
4. **Stopword removal** – remove common English stopwords (using NLTK).
5. **TF‑IDF vectorization** – convert cleaned text into numerical features:
   - `max_features = 5000`
   - `ngram_range = (1,2)` (unigrams and bigrams)

**Example:**

| Original tweet | Cleaned text |
|----------------|---------------|
| "Communal violence in Bhainsa, Telangana. Stones pelted..." | "communal violence bhainsa telangana stones pel" |

---

## 4. Methodology

### 4.1 Models Implemented

- **Logistic Regression** – linear classifier with L2 regularization (tunable: regularization type, strength, solver, decision threshold). We used default `penalty='l2'`, `C=1.0`, `solver='lbfgs'`, threshold=0.5.
- **Random Forest** – ensemble of 100 decision trees (tunable: number of trees, max depth, min samples split, min samples leaf). We used `n_estimators=100`, `max_depth=None`, `min_samples_split=2`, `min_samples_leaf=1`.
- **Neural Network (Fixed Architecture)** – built with Keras, architecture fixed as:
  - Input layer (vectorized text)
  - Hidden Layer 1: 64 neurons
  - Hidden Layer 2: 32 neurons
  - Hidden Layer 3: 16 neurons
  - Output layer: 1 neuron with sigmoid activation
  - Tunable hyperparameters: activation (ReLU chosen), optimizer (Adam), learning rate (0.001), batch size (32), epochs (20), weight initialization (glorot_uniform), dropout (0.0), L2 regularization (0.0), decision threshold (0.5).

### 4.2 Imbalance Handling Techniques

#### Part A – No Imbalance Handling
All models trained on the original imbalanced training set.

#### Part B – With Imbalance Handling
We applied **two distinct techniques**:

1. **Class Weighting** (for Logistic Regression and Random Forest) – using `class_weight='balanced'` to adjust weights inversely proportional to class frequencies.  
   For the Neural Network, we used **SMOTE** as the first technique (since class weighting is not directly applicable to Keras without custom weighting).

2. **SMOTE (Synthetic Minority Oversampling Technique)** – applied to the TF‑IDF vectors of the training set for **all models** (Logistic Regression, Random Forest, and Neural Network).  
   Additionally, we demonstrated **Random Oversampling** for the Neural Network as a variant.

All models are evaluated on the **same test set** (stratified split, 20% of data) to ensure fair comparison.

---

## 5. Experimental Setup

- **Train / Test split:** 80% / 20% (stratified by target class)  
  - Training size: 9,096 tweets  
  - Test size: 2,274 tweets  
- **Random seed:** 42 (for reproducibility)  
- **TF‑IDF parameters:** max_features = 5000, ngram_range = (1,2)  
- **Evaluation metrics:**  
  - Accuracy, Precision (per class), Recall (per class), F1‑score (macro & weighted)  
  - Confusion Matrix  
  - ROC‑AUC, PR‑AUC, Precision‑Recall curve  
- **Hardware:** Google Colab (CPU runtime)

---

## 6. Results

### 6.1 Performance Without Imbalance Handling

| Model | Accuracy | F1 Macro | Recall (Disaster) | PR‑AUC |
|-------|----------|----------|-------------------|--------|
| Logistic Regression | 0.8690 | 0.7131 | 0.3546 | 0.7256 |
| Random Forest | 0.8791 | 0.7654 | 0.4917 | 0.7155 |
| Neural Network (Fixed Arch) | 0.8624 | 0.7651 | 0.5887 | 0.7022 |

**Observation:** Random Forest achieves the highest accuracy (87.9%). The Neural Network has the best recall for disaster tweets (58.9%) without any balancing, significantly better than Logistic Regression (35.5%). However, its PR‑AUC is slightly lower, indicating a less favorable precision‑recall trade‑off.

### 6.2 Performance With Imbalance Handling

#### Technique 1: Class Weighting (LR, RF) + SMOTE (NN)

| Model | Technique | Accuracy | F1 Macro | Recall (Disaster) | PR‑AUC |
|-------|-----------|----------|----------|-------------------|--------|
| Logistic Regression | Class Weighting | 0.8465 | **0.7739** | **0.7518** | 0.7192 |
| Random Forest | Class Weighting | **0.8804** | 0.7637 | 0.4775 | 0.7298 |
| Neural Network | SMOTE | 0.8571 | 0.7616 | 0.6028 | 0.6979 |

**Observation:** 
- **Logistic Regression with class weighting** achieves the highest recall (75.2%) and F1 macro (0.7739), making it the best model for detecting disasters.
- **Random Forest with class weighting** maintains the highest overall accuracy (88.0%) but its recall for the disaster class remains low (47.8%).
- Neural Network with SMOTE improves recall slightly (from 58.9% to 60.3%) but still underperforms compared to Logistic Regression.

#### Technique 2: SMOTE (all models) + Random Oversampling (NN variant)

| Model | Technique | Accuracy | F1 Macro | Recall (Disaster) | PR‑AUC |
|-------|-----------|----------|----------|-------------------|--------|
| Logistic Regression | SMOTE | 0.8509 | 0.7732 | 0.7139 | 0.7223 |
| Random Forest | SMOTE | 0.8760 | 0.7694 | 0.5272 | 0.7304 |
| Neural Network | Random Oversampling | 0.8619 | 0.7655 | 0.5934 | 0.6997 |

**Observation:** 
- SMOTE improves Logistic Regression recall to 71.4% (slightly less than class weighting's 75.2%). 
- Random Forest with SMOTE improves recall to 52.7%, still far behind Logistic Regression.
- Neural Network with Random Oversampling performs similarly to SMOTE (recall ~59%).

### 6.3 Best Performance Summary

| Metric | Best Model | Technique | Value |
|--------|-----------|-----------|-------|
| Accuracy | Random Forest | Class Weighting | 0.8804 |
| F1 Macro | Logistic Regression | Class Weighting | 0.7739 |
| Recall (Disaster) | Logistic Regression | Class Weighting | 0.7518 |

### 6.4 Confusion Matrices, ROC Curves, PR Curves

*(Plots are generated by the code – include them in your final PDF.)*

Key observations from plots:
- Without balancing, Logistic Regression shows many false negatives (disasters missed).
- Class weighting for Logistic Regression drastically reduces false negatives, though false positives increase.
- Random Forest maintains high accuracy but struggles to identify disaster tweets even after balancing.
- The Neural Network achieves moderate recall but lower PR‑AUC indicates poor precision at higher recall levels.

---

## 7. Analysis and Discussion

### 7.1 Effect of Class Imbalance
Imbalance causes models to favour the majority (non‑disaster) class. Logistic Regression without balancing has only 35.5% recall for disasters – in a real‑world scenario, nearly two out of three actual disasters would be missed. Random Forest does better (49.2% recall) but still insufficient. The Neural Network achieves 58.9% recall even without balancing, suggesting its non‑linearity helps capture minority patterns.

### 7.2 Which Imbalance Technique Works Best?
- **For recall (most important for disaster detection):** Logistic Regression with class weighting achieves the highest recall (75.2%). This is the best choice for catching real disasters.
- **For balanced metrics (F1 macro):** Again, Logistic Regression with class weighting gives the highest F1 macro (0.7739), indicating the best overall trade‑off between precision and recall.
- **For accuracy:** Random Forest with class weighting gives the highest accuracy (88.0%), but this is misleading because its recall for disasters is low (47.8%).
- **Neural Network** underperforms compared to Logistic Regression, possibly due to the need for more hyperparameter tuning (e.g., learning rate, dropout, L2) or more training epochs.

### 7.3 SMOTE vs. Class Weighting
- **Class weighting** works exceptionally well for Logistic Regression, boosting recall from 35.5% to 75.2% while maintaining decent precision (56.6%).
- **SMOTE** also improves Logistic Regression (71.4% recall) but slightly less than class weighting.
- For Random Forest, neither technique improves recall substantially (still below 53%), suggesting that tree‑based models may need different handling (e.g., threshold moving).
- For the Neural Network, SMOTE and Random Oversampling give similar modest improvements (recall ~60%), far below Logistic Regression's performance.

### 7.4 Precision‑Recall Trade‑off
- Improving recall inevitably reduces precision. Logistic Regression with class weighting has precision of 56.6% for disasters – meaning about 43% of its disaster alerts are false positives. This is acceptable in disaster detection: **it is better to issue a false alarm than to miss a real disaster**.
- The PR‑AUC metric confirms that Logistic Regression with class weighting (0.719) and SMOTE (0.722) offer a good trade‑off, though Random Forest with SMOTE has a slightly higher PR‑AUC (0.730) due to better precision at lower recall levels.

---

## 8. Conclusion

This project successfully demonstrated the impact of class imbalance on text classification and the effectiveness of various mitigation techniques.

**Key findings:**
- Without imbalance handling, all models have poor recall for the disaster class, especially Logistic Regression (35.5%).
- **Logistic Regression with class weighting** is the overall best model, achieving the highest recall (75.2%), highest F1 macro (0.7739), and competitive PR‑AUC. It is the most suitable for real‑world disaster detection where catching disasters is critical.
- Random Forest achieves the highest accuracy (88.0%) but fails to improve disaster recall beyond 53%, making it less useful for this application.
- The fixed‑architecture Neural Network (64‑32‑16) shows moderate recall (max 60.3% with SMOTE) but requires more tuning to outperform Logistic Regression.
- Class weighting is simpler and more effective than SMOTE for Logistic Regression; for Random Forest, neither technique works well.

---

## References

1. Kaggle Dataset: [Disaster Tweets](https://www.kaggle.com/datasets/vstepanenko/disaster-tweets)
2. Scikit‑learn documentation: [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html), [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
3. TensorFlow/Keras documentation: [Sequential model](https://www.tensorflow.org/guide/keras/sequential_model)
4. Imbalanced‑learn documentation: [SMOTE, RandomOverSampler](https://imbalanced-learn.org/stable/)
