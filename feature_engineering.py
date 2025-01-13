import pandas as pd
import os
import glob
import pandas as pds
import numpy as np
import utils
import pickle
import joblib

class FeatureEngineering:
    def __init__(self, data):
        self.data = data
        self.is_fit_running = False
        self.feature_statistic_resource = 'assets/feature_statistic_resource.pkl'
        self.flowname_encoder = 'assets/flowname_encoder.pkl'

        self.feature_statistics = dict()

    def convert_type(self):
        self.data.timestamp = pd.to_datetime(self.data.timestamp)
        self.data = self.data.sort_values('timestamp')

    def indicate_outlier(self, features): 
        for feature in features:
            if self.is_fit_running:
                Q1 = self.data[feature].quantile(0.25)
                Q3 = self.data[feature].quantile(0.75)
                IQR = Q3 - Q1
                self.feature_statistics[feature + '_outlier'] = {'Q1': Q1, 'Q3': Q3, 'IQR': IQR}
            else:
                Q1 = self.feature_statistics[feature + '_outlier']['Q1']
                Q3 = self.feature_statistics[feature + '_outlier']['Q3']
                IQR = self.feature_statistics[feature + '_outlier']['IQR']
                
            outliers = self.data[(self.data[feature] < (Q1 - 1.5 * IQR)) | (self.data[feature] > (Q3 + 1.5 * IQR))]
            self.data[feature + '_outlier'] = ((self.data[feature] < (Q1 - 1.5 * IQR)) | (self.data[feature] > (Q3 + 1.5 * IQR))).astype(int)

        return self.data

    def calc_z_score_indiv_feature(self, features): # [amount, sender_age, receiver_age]
        for feature in features:
            if self.is_fit_running:
                mean = self.data[feature].mean()
                std = self.data[feature].std()
                self.feature_statistics[feature + '_zscore'] = {'mean': mean, 'std': std}
            else:
                mean = self.feature_statistics[feature + '_zscore']['mean']
                std = self.feature_statistics[feature + '_zscore']['std']

            self.data[feature + '_zscore'] = (self.data[feature] - mean) / std

        return self.data

    def calc_z_score_pair(self, features_pairs): 
        for features_pair in features_pairs:
            first, second = features_pair
            if self.is_fit_running:
                mean = self.data.groupby(second)[first].transform("mean")
                std = self.data.groupby(second)[first].transform("std")
                self.feature_statistics[f"{first}_zscore_{second}"] = {'mean': mean, 'std': std}
                self.data[first + '_zscore_' + second] = (self.data[first] - mean) / std
                self.data[first + '_zscore_' + second] = self.data[first + '_zscore_' + second].fillna(0)
            else:
                mean = self.feature_statistics[f"{first}_zscore_{second}"]['mean']
                std = self.feature_statistics[f"{first}_zscore_{second}"]['std'] 

            self.data[first + '_zscore_' + second] = (self.data[first] - mean) / std
            self.data[first + '_zscore_' + second] = self.data[first + '_zscore_' + second].fillna(0)
            
        return self.data

    # This function is to ensure that every class to be encoded should exist; otherwise, -1 is set
    def _safe_encode(self, encoder, data, column_name):
        classes = list(encoder.classes_)
        transformed = data[column_name].apply(lambda x: encoder.transform([x])[0] if x in classes else -1)
        return transformed

    def _encode_flowname(self):
        if not os.path.isfile(self.flowname_encoder):
            raise FileNotFoundError('No encoder file for flowname or undefined!')
        with open(self.flowname_encoder, 'rb') as f:
            flowname_encoder = pickle.load(f)
            self.data['flowname'] = self._safe_encode(flowname_encoder, self.data, 'flowname')

        return self.data

    def generate_custom_features(self):
        self.data['fee_per_transaction'] = self.data['bankfees'] / self.data['amount']
        self.data['age_diff_sender_receiver'] = self.data['sender_age'] - self.data['receiver_age']
        self.data["transaction_frequency"] = self.data.groupby(["sender_id", "receiver_id"]).cumcount()
        self.data['is_pension'] = (self.data['sender_age'] >= 67).astype(int)
        self.data['sender_total_failed_transactions'] = (
            self.data['status'].eq('PaymentFailedV0')  
            .groupby(self.data['sender_id'])           
            .cumsum()                           
        )

        self.data = self._encode_flowname()
            
        self.data['flow_fraud_rate'] = self.data.groupby('flowname')['fraud'].transform('sum') / self.data.groupby('flowname')['flowname'].transform('count')
        
        return self.data

    # A function to calculate pay date of pension money
    # The money is paid on the 20th day of the month if day of the transaction is after 20
    # Or 20th day of the previous month
    def _calculate_pension_pay_day(self, date):
        # Move the date to the previous month
        previous_month = date - pd.offsets.MonthBegin(2)  
        pay_day = previous_month.replace(day=20)  
        
        # Check if the 20th is a Friday
        if pay_day.weekday() == 4:  # 0=Monday, 4=Friday
            # If Friday, adjust to the day before (Thursday)
            pay_day = pay_day - pd.Timedelta(days=1)
        
        return pay_day

    def calc_diff_transaction_pension_payment(self):
        # difference from the transaction time to pension payout
        self.data['last_pension_pay_date'] = self.data['timestamp'].apply(self._calculate_pension_pay_day)
        self.data['pensionpaid_transaction_days_diff'] = (self.data['timestamp'] - self.data['last_pension_pay_date']).dt.days - 1
        self.data['pensionpaid_transaction_days_diff'] = self.data['pensionpaid_transaction_days_diff'] * self.data['is_pension']
        self.data.drop('last_pension_pay_date', axis=1, inplace=True)

        return self.data

    def calc_diff_transaction(self):
        self.data = self.data.reset_index(drop=True)
        # difference between two transactions made by the same pair of people
        self.data['transaction_time_diff'] = (
            self.data.sort_values('timestamp')
            .groupby(['sender_id', 'receiver_id'])['timestamp']
            .diff()
        )
        self.data['transaction_time_diff'] = self.data['transaction_time_diff'].dt.total_seconds()
        self.data['transaction_time_diff'] = self.data['transaction_time_diff'].fillna(0)
        return self.data

    def calc_reattempt_transaction_diff(self):
        failed_payments = self.data[self.data['status'] == 'PaymentFailedV0']
        merged = pd.merge(self.data, failed_payments, on='sender_id', suffixes=('', '_failed'), how='left')
        merged = merged[merged['timestamp_failed'] < merged['timestamp']]
        last_failed_times = merged.groupby(['sender_id', 'timestamp'])['timestamp_failed'].max().reset_index()
        self.data = pd.merge(self.data, last_failed_times[['sender_id', 'timestamp', 'timestamp_failed']], 
                       on=['sender_id', 'timestamp'], 
                       how='left')
        self.data['last_failed_time'] = self.data['timestamp_failed']
        self.data['last_failed_time'].fillna(self.data['timestamp'], inplace=True)
        self.data['seconds_diff_last_failed'] = (self.data['timestamp'] - self.data['last_failed_time']).dt.seconds
        self.data.drop(columns=['timestamp_failed', 'last_failed_time'], inplace=True)

        return self.data

    def extract_time_components(self):
        self.data['weekday'] = self.data['timestamp'].dt.weekday
        self.data['day'] = self.data['timestamp'].dt.day
        self.data['month'] = self.data['timestamp'].dt.month
        self.data['hour'] = self.data['timestamp'].dt.hour

        self.data = self.calc_diff_transaction()
        self.data = self.calc_reattempt_transaction_diff()
        self.data = self.calc_diff_transaction_pension_payment()       

        return self.data

    def cleanup(self):
        self.data.drop(columns=['sender_id', 'receiver_id', 'status', 'timestamp'], axis=1, inplace=True)
        return self.data
        
    def fit_transform(self):
        # Indicate the fitting process is ongoing
        self.is_fit_running = True
        
        # Transforming data
        self.convert_type()
        self.indicate_outlier(['amount'])
        self.calc_z_score_indiv_feature(['amount', 'sender_age', 'receiver_age'])
        self.calc_z_score_pair([
            ['amount', 'sender_age'], ['amount', 'receiver_age'],
            ['bankfees', 'sender_age'], ['bankfees', 'receiver_age'],
            ['sender_age', 'receiver_age']
        ])
        self.generate_custom_features()
        self.extract_time_components()
        self.cleanup()

        # Save characteristics
        joblib.dump(self.feature_statistics, self.feature_statistic_resource)
        return self.data

    def transform(self):
        self.feature_statistics = joblib.load(self.feature_statistic_resource)
        
        # Transforming data
        self.convert_type()
        self.indicate_outlier(['amount'])
        self.calc_z_score_indiv_feature(['amount', 'sender_age', 'receiver_age'])
        self.calc_z_score_pair([
            ['amount', 'sender_age'], ['amount', 'receiver_age'],
            ['bankfees', 'sender_age'], ['bankfees', 'receiver_age'],
            ['sender_age', 'receiver_age']
        ])
        self.generate_custom_features()
        self.extract_time_components()
        self.cleanup()


        return self.data

if __name__ == "__main__":
    df = utils.load_data(nonfraud_path=glob.glob("datasets/*_nonfraud_*.csv"), fraud_path=glob.glob("datasets/*_fraud*.csv"))
    
    fe = FeatureEngineering(df)
    transformed_df = fe.transform() #fe.fit_transform()

    transformed_df.to_csv('data4dev_test__.csv', index=False)
