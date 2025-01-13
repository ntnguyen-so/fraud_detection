from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


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
