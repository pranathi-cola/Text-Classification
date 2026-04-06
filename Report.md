# Imbalanced Text Classification for Disaster Tweet Detection

**Team Members:** Kola Pranathi, Kommula Devamani, Bokka Roshini Sreeja

---

## 1. Introduction

The goal of this project is to build and compare binary text classification models on the **Disaster Tweets dataset**, which exhibits significant class imbalance (majority non‑disaster, minority disaster). Class imbalance often causes models to be biased toward the majority class, leading to poor recall for the minority class – a critical issue when detecting real disasters.

We implement three classifiers: Logistic Regression, Naive Bayes, and Linear SVM. For each, we evaluate performance **without** imbalance handling, then apply **two different techniques**: class weighting (or random oversampling) and SMOTE. Performance is measured using accuracy, precision, recall, F1‑score, ROC‑AUC, PR‑AUC, and confusion matrices.

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

Example:

| Original tweet | Cleaned text |
|----------------|---------------|
| "Communal violence in Bhainsa, Telangana. Stones pelted..." | "communal violence bhainsa telangana stones pel" |

---

## 4. Methodology

### 4.1 Models Implemented
- **Logistic Regression** – linear classifier with L2 regularization.
- **Multinomial Naive Bayes** – suitable for multinomial distributions (TF‑IDF features).
- **Linear Support Vector Machine (LinearSVC)** – maximizes margin between classes.

### 4.2 Imbalance Handling Techniques

#### Part A – No Imbalance Handling
All models trained on the original imbalanced training set.

#### Part B – With Imbalance Handling
We applied **two distinct techniques**:

1. **Class Weighting / Random Oversampling**  
   - For Logistic Regression and Linear SVM: `class_weight='balanced'` (adjusts weights inversely proportional to class frequencies).  
   - For Naive Bayes (which does not support class_weight): **Random Oversampling** of the minority class in the training set.

2. **SMOTE (Synthetic Minority Oversampling Technique)**  
   - Applied to the TF‑IDF vectors of the training set.  
   - Generates synthetic samples for the minority class by interpolating between existing minority instances.  
   - Results in a perfectly balanced training set (7,405 samples per class).

All models are evaluated on the **same test set** (stratified split, 20% of data) to ensure fair comparison.

---

## 5. Experimental Setup

- **Train / Test split:** 80% / 20% (stratified by target class)  
  - Training size: 9,096 tweets  
  - Test size: 2,274 tweets  
- **Random seed:** 42 
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
| Naive Bayes | 0.8703 | 0.7227 | 0.3783 | 0.7317 |
| **Linear SVM** | **0.8826** | **0.7866** | **0.5697** | **0.7141** |

- **Observation:** Linear SVM achieves the highest accuracy and F1 macro, but recall for disaster tweets is only 57%. Logistic Regression and Naive Bayes have very low recall (~35–38%), missing most real disasters.

### 6.2 Performance With Imbalance Handling

#### Technique 1: Class Weighting / Random Oversampling

| Model | Technique | Accuracy | F1 Macro | Recall (Disaster) | PR‑AUC |
|-------|-----------|----------|----------|-------------------|--------|
| Logistic Regression | Class Weighting | 0.8465 | 0.7739 | 0.7518 | 0.7192 |
| Linear SVM | Class Weighting | 0.8439 | 0.7644 | 0.7069 | 0.6865 |
| **Naive Bayes** | **Random Oversampling** | **0.8347** | **0.7627** | **0.7636** | **0.7383** |

- **Observation:** All techniques significantly boost recall for the disaster class (from ~35% to >70%). Naive Bayes + Random Oversampling gives the highest recall (76.4%) and PR‑AUC (0.7383). Accuracy drops slightly but is still acceptable.

#### Technique 2: SMOTE

| Model | Technique | Accuracy | F1 Macro | Recall (Disaster) | PR‑AUC |
|-------|-----------|----------|----------|-------------------|--------|
| Logistic Regression | SMOTE | 0.8509 | 0.7732 | 0.7139 | 0.7223 |
| Naive Bayes | SMOTE | 0.8412 | 0.7671 | 0.7447 | 0.7388 |
| Linear SVM | SMOTE | 0.8514 | 0.7682 | 0.6785 | 0.6869 |

- **Observation:** SMOTE also improves recall substantially, though slightly lower than random oversampling for Naive Bayes. For SVM, SMOTE gives a balanced trade‑off between precision and recall.

### 6.3 Confusion Matrices, ROC Curves, PR Curves

*(Plots are generated by the code – include them in your final PDF.)*

Key observations from plots:
- Without balancing, the confusion matrix shows many false negatives (disasters missed).
- With balancing, false negatives decrease, but false positives increase slightly.
- ROC‑AUC remains high (>0.88) for all models, but PR‑AUC is more sensitive to imbalance and improves with balancing.

---

## 7. Analysis and Discussion

### 7.1 Effect of Class Imbalance
Imbalance causes models to favour the majority (non‑disaster) class. Even though accuracy is high (>87%), the **recall for disaster tweets is disastrously low** (35–57%). This means in a real‑world scenario, most actual disasters would go undetected – unacceptable for an emergency alert system.

### 7.2 Which Imbalance Technique Works Best?
- **For recall:** Naive Bayes + Random Oversampling achieves the highest recall (76.4%), making it the best choice if the goal is to catch as many disasters as possible (even at the cost of some false alarms).
- **For balanced metrics (F1 macro):** Linear SVM without balancing gives the highest F1 macro (0.7866), but this is because it still favours the majority class. After balancing, F1 macro improves for minority class but overall macro drops slightly due to lower precision on majority class.
- **Class weighting** and **SMOTE** perform similarly for Logistic Regression and SVM. For Naive Bayes, random oversampling edges out SMOTE.

### 7.3 SMOTE vs. Class Weighting / Oversampling
- **SMOTE** generates synthetic samples, which can help generalisation but may introduce noise if not tuned. In our experiments, SMOTE yields slightly lower recall for SVM but more stable precision.
- **Class weighting** is simpler and computationally cheaper. It works well for LR and SVM.
- **Random oversampling** (for NB) is straightforward and effective, though it can lead to overfitting if the same minority samples are repeated; however, the test performance remains good.

### 7.4 Precision‑Recall Trade‑off
- Improving recall for the disaster class inevitably reduces precision (more false positives). This is acceptable in disaster detection: **it is better to issue a false alarm than to miss a real disaster**.
- The PR‑AUC metric confirms that balancing techniques improve the overall precision‑recall trade‑off. The highest PR‑AUC (0.7388) is achieved by Naive Bayes + Random Oversampling / SMOTE.

---

## 8. Conclusion

This project successfully demonstrated the impact of class imbalance on text classification and the effectiveness of various mitigation techniques.

- Without imbalance handling, models have high accuracy but very poor recall for the minority (disaster) class.
- **Naive Bayes combined with Random Oversampling** provides the best recall (76.4%) and PR‑AUC, making it the most suitable model for real‑world disaster tweet detection where catching disasters is critical.
- Class weighting and SMOTE also improve recall significantly and are viable alternatives depending on the desired precision‑recall balance.

---

## References
1. Kaggle Dataset: [Disaster Tweets](https://www.kaggle.com/datasets/vstepanenko/disaster-tweets)
2. Scikit‑learn documentation: [Class Weight](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
3. Imbalanced‑learn documentation: [RandomOverSampler, SMOTE](https://imbalanced-learn.org/stable/)
