from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer, recall_score, classification_report


def calc_auc_pr(y_test, y_prob):
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
    auc_pr = average_precision_score(y_test, y_prob)
    return auc_pr

def calc_auc_roc(y_test, y_prob):
    auc_roc = roc_auc_score(y_test, y_prob)
    return auc_roc

def calc_f1(y_test, y_pred):
    f1 = f1_score(y_test, y_pred)
    return f1

def calc_sensitivity(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    return tp / (tp + fn) if (tp + fn) > 0 else 0

def calc_specificity(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    return tn / (tn + fp) if (tn + fp) > 0 else 0

def generate_confusion_matrix(y_test, y_pred, img_path=None):
    cm = confusion_matrix(y_test, y_pred)

    if img_path:
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Non-Fraud', 'Fraud'], yticklabels=['Non-Fraud', 'Fraud'])
        
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        plt.savefig(img_path)
        plt.show()
    return cm

def identify_false_predictions(orig_data_path, test_data_path, y_test, y_pred):
    # load data from the paths specified for inference
    df_orig = pd.read_csv(orig_data_path)
    df_dev = pd.read_csv(test_data_path)

    # merge with prediction
    df_dev = df_dev.loc[y_test.index]
    df_dev['fraud_predicted'] = y_pred

    # we will only keep the orig data cols together with the prediction
    col_to_keep = list(df_orig.columns) + ['fraud_predicted']
    df_inferred = pd.merge(df_orig, df_dev, how='inner', on='id', suffixes=('', '_dev'))

    # beautify the report
    target = 'fraud'
    gt_data = df_inferred.pop(target)
    df_inferred.insert(len(df_inferred.columns) - 1, target, gt_data)

    return df_inferred[df_inferred.fraud != df_inferred.fraud_predicted]

def plot_auc_pr_curve(y_test, y_prob):
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
    auc_pr = average_precision_score(y_test, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f"AUC-PR = {auc_pr:.2f}", color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='best')
    plt.grid()
    plt.show()

def cross_validation(X_train, y_train, data_scaler, smote_scaler, clf):
    # Prepare for cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cross_val_metrics = []
    
    for train_index, val_index in skf.split(X_train, y_train):
        # Split data into training and validation sets
        X_train_k, X_val_k = X_train.to_numpy()[train_index], X_train.to_numpy()[val_index]
        y_train_k, y_val_k = y_train.to_numpy()[train_index], y_train.to_numpy()[val_index]
    
        # Standardize and balance the data
        X_train_k_standardized, X_val_k_standardize = data_scaler.transform(X_train_k), data_scaler.transform(X_val_k)
        X_train_k_standardized_balanced, y_train_k_standardized_balanced = smote_scaler.fit_resample(X_train_k_standardized, y_train_k)
        
        # Train and validate the model
        clf.fit(X_train_k_standardized_balanced, y_train_k_standardized_balanced)
        y_pred_val_k = clf.predict(X_val_k_standardize)
        
        # Collect metrics
        report = classification_report(y_val_k, y_pred_val_k, output_dict=True)
        cross_val_metrics.append(report)

    return cross_val_metrics