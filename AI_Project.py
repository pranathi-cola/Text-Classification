!pip install -q kagglehub imbalanced-learn tensorflow

import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import glorot_uniform, he_normal, random_uniform
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, classification_report,
    confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, ConfusionMatrixDisplay
)
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings('ignore')

nltk.download('stopwords')
from nltk.corpus import stopwords

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

import kagglehub
path = kagglehub.dataset_download("vstepanenko/disaster-tweets")
print("Path to dataset files:", path)
import os
csv_file = None
for f in os.listdir(path):
    if f.endswith('.csv'):
        csv_file = os.path.join(path, f)
        break
if csv_file is None:
    raise FileNotFoundError("No CSV file found in the downloaded dataset")

df = pd.read_csv(csv_file)
print(f"Dataset shape: {df.shape}")
df.head()

target_counts = df['target'].value_counts()
print("Class distribution:\n", target_counts)
print(f"Imbalance ratio (majority/minority): {target_counts[0]/target_counts[1]:.2f}:1")

sns.barplot(x=target_counts.index, y=target_counts.values)
plt.title('Class Distribution (0 = Non-disaster, 1 = Disaster)')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

df['clean_text'] = df['text'].apply(clean_text)
print("Sample after cleaning:\n")
print(df[['text', 'clean_text']].iloc[0])

X = df['clean_text']
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
print(f"Train class distribution:\n{y_train.value_counts()}")

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)
print(f"TF-IDF shape: {X_train_tfidf.shape}")

def build_nn(input_dim, activation='relu', optimizer='adam', learning_rate=0.001,
             dropout_rate=0.0, l2_reg=0.0, init='glorot_uniform'):
    model = Sequential()
    model.add(Input(shape=(input_dim,)))

    model.add(Dense(64, activation=activation, kernel_initializer=init,
                    kernel_regularizer=l2(l2_reg)))
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))

    model.add(Dense(32, activation=activation, kernel_initializer=init,
                    kernel_regularizer=l2(l2_reg)))
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))

    model.add(Dense(16, activation=activation, kernel_initializer=init,
                    kernel_regularizer=l2(l2_reg)))
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))

    model.add(Dense(1, activation='sigmoid'))

    if optimizer.lower() == 'adam':
        opt = Adam(learning_rate=learning_rate)
    elif optimizer.lower() == 'sgd':
        opt = SGD(learning_rate=learning_rate)
    elif optimizer.lower() == 'rmsprop':
        opt = RMSprop(learning_rate=learning_rate)
    else:
        opt = Adam(learning_rate=learning_rate)

    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model

def evaluate_sklearn_model(model, X_train_vec, y_train, X_test_vec, y_test, model_name="Model", technique=""):
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test_vec)[:, 1]
    else:
        y_proba = model.decision_function(X_test_vec)

    acc = accuracy_score(y_test, y_pred)
    precision, recall, fbeta, _ = precision_recall_fscore_support(y_test, y_pred, average=None, labels=[0,1])
    f1_macro = precision_recall_fscore_support(y_test, y_pred, average='macro')[2]
    f1_weighted = precision_recall_fscore_support(y_test, y_pred, average='weighted')[2]
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)

    print(f"{model_name} {technique}")
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")
    print(f"F1 Macro: {f1_macro:.4f} | F1 Weighted: {f1_weighted:.4f}")
    print("\nPer-class Precision/Recall/F1:")
    print(f"Class 0 (Non-disaster): P={precision[0]:.4f}, R={recall[0]:.4f}, F1={fbeta[0]:.4f}")
    print(f"Class 1 (Disaster):     P={precision[1]:.4f}, R={recall[1]:.4f}, F1={fbeta[1]:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Non-disaster', 'Disaster']))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-disaster', 'Disaster'])
    disp.plot(cmap='Blues')
    plt.title(f'Confusion Matrix: {model_name} {technique}')
    plt.show()

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.3f})')
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve: {model_name} {technique}')
    plt.legend()
    plt.show()

    prec, rec, _ = precision_recall_curve(y_test, y_proba)
    plt.plot(rec, prec, label=f'PR (AUC = {pr_auc:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve: {model_name} {technique}')
    plt.legend()
    plt.show()

    return {
        'model': model_name,
        'technique': technique,
        'accuracy': acc,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision_disaster': precision[1],
        'recall_disaster': recall[1]
    }

def evaluate_nn(X_train_vec, y_train, X_test_vec, y_test,
                activation='relu', optimizer='adam', learning_rate=0.001,
                batch_size=32, epochs=20, dropout_rate=0.0, l2_reg=0.0,
                init='glorot_uniform', threshold=0.5, technique=""):
    X_train_dense = X_train_vec.toarray()
    X_test_dense = X_test_vec.toarray()

    model = build_nn(X_train_dense.shape[1], activation, optimizer, learning_rate,
                     dropout_rate, l2_reg, init)

    history = model.fit(X_train_dense, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_split=0.1,
                        verbose=0)

    y_proba = model.predict(X_test_dense).flatten()
    y_pred = (y_proba >= threshold).astype(int)

    acc = accuracy_score(y_test, y_pred)
    precision, recall, fbeta, _ = precision_recall_fscore_support(y_test, y_pred, average=None, labels=[0,1])
    f1_macro = precision_recall_fscore_support(y_test, y_pred, average='macro')[2]
    f1_weighted = precision_recall_fscore_support(y_test, y_pred, average='weighted')[2]
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)

    print(f"Neural Network (Fixed Arch) {technique}")
    print(f"Params: {activation}, {optimizer}, lr={learning_rate}, bs={batch_size}, epochs={epochs}, dropout={dropout_rate}, l2={l2_reg}, threshold={threshold}")
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")
    print(f"F1 Macro: {f1_macro:.4f} | F1 Weighted: {f1_weighted:.4f}")
    print("\nPer-class Precision/Recall/F1:")
    print(f"Class 0 (Non-disaster): P={precision[0]:.4f}, R={recall[0]:.4f}, F1={fbeta[0]:.4f}")
    print(f"Class 1 (Disaster):     P={precision[1]:.4f}, R={recall[1]:.4f}, F1={fbeta[1]:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Non-disaster', 'Disaster']))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-disaster', 'Disaster'])
    disp.plot(cmap='Blues')
    plt.title(f'Confusion Matrix: Neural Network (Fixed Arch) {technique}')
    plt.show()

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.3f})')
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve: Neural Network (Fixed Arch) {technique}')
    plt.legend()
    plt.show()

    prec, rec, _ = precision_recall_curve(y_test, y_proba)
    plt.plot(rec, prec, label=f'PR (AUC = {pr_auc:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve: Neural Network (Fixed Arch) {technique}')
    plt.legend()
    plt.show()

    return {
        'model': 'Neural Network (Fixed Arch)',
        'technique': technique + f" ({activation}/{optimizer}/bs{batch_size}/ep{epochs}/drop{dropout_rate}/l2{l2_reg}/th{threshold})",
        'accuracy': acc,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision_disaster': precision[1],
        'recall_disaster': recall[1]
    }

lr_default = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=1000, random_state=42)
results_a_lr = evaluate_sklearn_model(lr_default, X_train_tfidf, y_train, X_test_tfidf, y_test,
                                      model_name="Logistic Regression", technique="(No balancing)")

rf_default = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2,
                                    min_samples_leaf=1, random_state=42)
results_a_rf = evaluate_sklearn_model(rf_default, X_train_tfidf, y_train, X_test_tfidf, y_test,
                                      model_name="Random Forest", technique="(No balancing)")

results_a_nn = evaluate_nn(X_train_tfidf, y_train, X_test_tfidf, y_test,
                           activation='relu', optimizer='adam', learning_rate=0.001,
                           batch_size=32, epochs=20, dropout_rate=0.0, l2_reg=0.0,
                           init='glorot_uniform', threshold=0.5, technique="(No balancing)")

results_a = [results_a_lr, results_a_rf, results_a_nn]

results_b1 = []

lr_cw = LogisticRegression(class_weight='balanced', penalty='l2', C=1.0, solver='lbfgs', max_iter=1000, random_state=42)
metrics_lr_cw = evaluate_sklearn_model(lr_cw, X_train_tfidf, y_train, X_test_tfidf, y_test,
                                       model_name="Logistic Regression", technique="(Class Weighting)")
results_b1.append(metrics_lr_cw)

rf_cw = RandomForestClassifier(class_weight='balanced', n_estimators=100, max_depth=None,
                               min_samples_split=2, min_samples_leaf=1, random_state=42)
metrics_rf_cw = evaluate_sklearn_model(rf_cw, X_train_tfidf, y_train, X_test_tfidf, y_test,
                                       model_name="Random Forest", technique="(Class Weighting)")
results_b1.append(metrics_rf_cw)

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_tfidf, y_train)
print(f"After SMOTE: X shape {X_train_smote.shape}, y distribution {pd.Series(y_train_smote).value_counts().to_dict()}")
metrics_nn_smote = evaluate_nn(X_train_smote, y_train_smote, X_test_tfidf, y_test,
                               activation='relu', optimizer='adam', learning_rate=0.001,
                               batch_size=32, epochs=20, dropout_rate=0.0, l2_reg=0.0,
                               init='glorot_uniform', threshold=0.5, technique="(SMOTE)")
results_b1.append(metrics_nn_smote)

results_b2 = []

lr_smote = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=1000, random_state=42)
metrics_lr_smote = evaluate_sklearn_model(lr_smote, X_train_smote, y_train_smote, X_test_tfidf, y_test,
                                          model_name="Logistic Regression", technique="(SMOTE)")
results_b2.append(metrics_lr_smote)

rf_smote = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2,
                                   min_samples_leaf=1, random_state=42)
metrics_rf_smote = evaluate_sklearn_model(rf_smote, X_train_smote, y_train_smote, X_test_tfidf, y_test,
                                          model_name="Random Forest", technique="(SMOTE)")
results_b2.append(metrics_rf_smote)

ros = RandomOverSampler(random_state=42)
X_train_ros, y_train_ros = ros.fit_resample(X_train_tfidf, y_train)
metrics_nn_ros = evaluate_nn(X_train_ros, y_train_ros, X_test_tfidf, y_test,
                             activation='relu', optimizer='adam', learning_rate=0.001,
                             batch_size=32, epochs=20, dropout_rate=0.0, l2_reg=0.0,
                             init='glorot_uniform', threshold=0.5, technique="(Random Oversampling)")
results_b2.append(metrics_nn_ros)

all_results = results_a + results_b1 + results_b2
df_results = pd.DataFrame(all_results)
df_results = df_results[['model', 'technique', 'accuracy', 'roc_auc', 'pr_auc',
                         'f1_macro', 'f1_weighted', 'precision_disaster', 'recall_disaster']]
df_results

print("\nBest Accuracy")
print(df_results.loc[df_results['accuracy'].idxmax()])

print("\nBest F1 Macro")
print(df_results.loc[df_results['f1_macro'].idxmax()])

print("\nBest Recall for Disaster Class")
print(df_results.loc[df_results['recall_disaster'].idxmax()])
