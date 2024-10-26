from typing import List, Tuple
import sqlite3
import boto3
import csv
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from scipy.stats import skew
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import (  # ,LabelEncoder, MinMaxScaler
    OneHotEncoder, StandardScaler)


class DataReader:
    def read_csv(self, file_path):
        return pd.read_csv(file_path)
    
    def read_database(self, db_path, query):
        conn = sqlite3.connect(db_path)
        data = pd.read_sql_query(query, conn)
        conn.close()
        return data

    def read_s3_bucket(self, bucket_name, file_key):
        s3 = boto3.client('s3')
        obj = s3.get_object(Bucket=bucket_name, Key=file_key)
        return pd.read_csv(obj['Body'])

class DataProcessor:
    def __init__(self, data):
        if isinstance(data, pd.DataFrame):
            self.data = data
        else:
            raise ValueError("Input data must be a pandas DataFrame")

    def remove_missing_rows(self): #tested
        self.data = self.data.dropna()
        return self.data
    
    def fill_missing_columns(self, method: str = 'mean') -> pd.DataFrame: #tested
        """
        Treats missing data in a DataFrame based on the specified method.
        If the method is not recognized, missing rows are dropped by default.
        
        Args:
        df (pd.DataFrame): The DataFrame with missing data.
        method (str): The method to use for filling missing data.
        Options are 'mean', 'median', 'mode', or 'drop'.
        
        Returns:
            pd.DataFrame: The DataFrame with treated missing data.
        """
        # Create a copy of the DataFrame
        filled_df = self.data.copy()
        
        # Default to dropping missing values if method is not recognized
        if method not in ['mean', 'median', 'mode']:
            print("Method not recognized. Dropping all rows with missing values.")
            return remove_missing_rows()
        
        # Iterate over each column in the DataFrame
        for col in filled_df.columns:
            if filled_df[col].dtype in [np.float64, np.int64]:
                # Numeric data
                if method == 'mean':
                    filled_df[col].fillna(filled_df[col].mean(), inplace=True)
                elif method == 'median':
                    filled_df[col].fillna(filled_df[col].median(), inplace=True)
                elif method == 'mode':
                    filled_df[col].fillna(filled_df[col].mode()[0], inplace=True)
                
            if filled_df[col].dtype not in [np.float64, np.int64]:
                # Categorical data
                filled_df[col].fillna(filled_df[col].mode()[0], inplace=True)
        
        return filled_df

    def encode_categorical(self, method='pd'):
        if method == 'pd':
            self.data = pd.get_dummies(self.data)
        elif method == 'onehot':
            encoder = OneHotEncoder()
            encoded_data = encoder.fit_transform(self.data.select_dtypes(include=['object']))
            encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(self.data.select_dtypes(include=['object']).columns))
            self.data = pd.concat([self.data.select_dtypes(exclude=['object']).reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)
        return self.data
    
    def rename_columns(self, columns_dict):
        self.data = self.data.rename(columns=columns_dict)
        return self.data
    
    def remove_columns(self, columns_list):
        self.data = self.data.drop(columns=columns_list)
        return self.data

    def remove_unwanted_columns(self, columns_to_remove: List[str]) -> pd.DataFrame:
        """
        Removes specified columns from a DataFrame.
    
        Args:
            df (pd.DataFrame): The input DataFrame.
            columns_to_remove (list): A list of column names to be removed.
    
        Returns:
            pd.DataFrame: A DataFrame with the specified columns removed.
        """
        df = self.data.copy()
        # Check if all specified columns exist in the DataFrame
        existing_columns = [col for col in columns_to_remove if col in df.columns]
    
        # Drop the unwanted columns
        df = df.drop(columns=existing_columns, axis=1)
    
        return df

    def winsorize_outliers(self, limit=0.05):
        from scipy.stats.mstats import winsorize
        self.data = self.data.apply(lambda x: winsorize(x, limits=limit) if pd.api.types.is_numeric_dtype(x) else x)
        return self.data

    def change_column_type(self, column_name, new_type):
        self.data[column_name] = self.data[column_name].astype(new_type)
        return self.data

    @staticmethod
    def handle_class_imbalance(X, y, technique: str = 'SMOTE', random_state: int = None):
        """
        Handles class imbalance using the specified technique.
    
        Args:
            X (pd.DataFrame): Features dataframe.
            y (pd.Series): Target series.
            technique (str): The technique for handling class imbalance ('undersampling', 'oversampling', 'SMOTE').
            random_state (int): The seed used by the random number generator.
    
        Returns:
            X_resampled (np.ndarray): Resampled features.
            y_resampled (np.ndarray): Resampled target.
        """
        if technique == 'SMOTE':
            sampler = SMOTE(random_state=random_state)
        elif technique == 'oversampling':
            sampler = RandomOverSampler(random_state=random_state)
        elif technique == 'undersampling':
            sampler = RandomUnderSampler(random_state=random_state)
        else:
            raise ValueError("Invalid technique specified. Choose from 'undersampling', 'oversampling', or 'SMOTE'.")
    
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        return X_resampled, y_resampled
    
    def other_transformations(self):
        # Add your other useful transformations here
        return self.data