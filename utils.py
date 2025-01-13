import pandas as pd
import numpy as np
import joblib
from feature_engineering import FeatureEngineering
import pickle

def load_data(nonfraud_path, fraud_path, fillna=True):
    def load_raw_data(data_path, is_fraud=False):
        datasets = []
        for file in data_path:
            datasets.append(pd.read_csv(file))
    
        df = pd.concat(datasets)
        if is_fraud:
            df['fraud'] = 1
        else:
            df['fraud'] = 0
            
        df = df.fillna(0)
        return df
        
    nonfraud_df = load_raw_data(nonfraud_path, is_fraud=False)
    fraud_df = load_raw_data(fraud_path, is_fraud=True)
    df = pd.concat([nonfraud_df, fraud_df])
    
    if fillna:
        df = df.fillna(0)
    return df

def transform_data(df, new_path, is_fit=False):
    fe = FeatureEngineering(df)
    if is_fit:
        transformed_df = fe.fit_transform()
    else:
        transformed_df = fe.transform()
    transformed_df.to_csv(new_path, index=False)
    return transformed_df

def generate_prediction_report(orig_data_path, test_data_path, y_test, y_pred, y_pred_proba, result_path):
    # load data from the paths specified for inference
    df_orig = pd.read_csv(orig_data_path)
    df_dev = pd.read_csv(test_data_path)
    
    # merge with prediction
    df_dev = df_dev.loc[y_test.index]
    df_dev['fraud_predicted'] = y_pred
    df_dev['fraud_predicted_proba'] = y_pred_proba[:, 1]
    
    # we will only keep the orig data cols together with the prediction
    col_to_keep = list(df_orig.columns) + ['fraud_predicted', 'fraud_predicted_proba', 'fraud_rank']
    df_inferred = pd.merge(df_orig, df_dev, how='inner', on='id', suffixes=('', '_dev'))
    df_inferred['fraud_rank'] = df_inferred['fraud_predicted_proba'].rank(ascending=False, method='first').astype(int)
    
    # beautify the report
    target = 'fraud'
    gt_data = df_inferred.pop(target)
    df_inferred.insert(len(df_inferred.columns) - 1, target, gt_data)
    df_inferred = df_inferred.sort_values('fraud_rank')

    df_inferred = df_inferred[col_to_keep]
    df_inferred.to_csv(result_path, index=False)
    return df_inferred

def save_model(model, path):
    joblib.dump(model, path)

def load_model(path):
    model = joblib.load(path)
    return model

def save_scaler(scaler, path):
    joblib.dump(scaler, path)

def load_scaler(path):
    scaler = joblib.load(path)
    return scaler
    