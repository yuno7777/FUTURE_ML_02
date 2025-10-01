"""
Data Preprocessing Module for Churn Prediction System

This module handles preprocessing for both Telco and Bank churn datasets:
- Automatic dataset detection
- Data cleaning and missing value handling
- Categorical encoding (OneHot/Label)
- Numerical feature scaling
- Feature type identification

Author: AI Data Scientist
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

class ChurnDataPreprocessor:
    """
    A comprehensive preprocessor for churn prediction datasets.
    Supports both Telco and Bank customer churn datasets.
    """
    
    def __init__(self):
        self.dataset_type = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.column_transformer = None
        self.feature_names = []
        self.target_column = None
        
    def detect_dataset_type(self, df):
        """
        Automatically detect whether the dataset is Telco or Bank churn data
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            str: 'telco' or 'bank'
        """
        # Check for distinctive columns in each dataset
        telco_indicators = ['customerID', 'MonthlyCharges', 'TotalCharges', 
                           'InternetService', 'Contract', 'PaymentMethod']
        bank_indicators = ['CustomerId', 'Surname', 'CreditScore', 'Geography', 
                          'Balance', 'EstimatedSalary', 'Exited']
        
        telco_score = sum(1 for col in telco_indicators if col in df.columns)
        bank_score = sum(1 for col in bank_indicators if col in df.columns)
        
        if bank_score > telco_score:
            self.dataset_type = 'bank'
            self.target_column = 'Exited'
        else:
            self.dataset_type = 'telco'
            self.target_column = 'Churn'
            
        print(f"🔍 Dataset detected: {self.dataset_type.upper()} churn dataset")
        return self.dataset_type
    
    def clean_data(self, df):
        """
        Clean the dataset by handling missing values and data type issues
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Cleaned dataframe
        """
        df_clean = df.copy()
        
        if self.dataset_type == 'telco':
            # Handle TotalCharges column (often stored as string with spaces)
            if 'TotalCharges' in df_clean.columns:
                df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
                df_clean['TotalCharges'].fillna(df_clean['TotalCharges'].median(), inplace=True)
            
            # Convert target to binary
            if 'Churn' in df_clean.columns:
                df_clean['Churn'] = df_clean['Churn'].map({'Yes': 1, 'No': 0})
            
            # Convert SeniorCitizen to proper categorical
            if 'SeniorCitizen' in df_clean.columns:
                df_clean['SeniorCitizen'] = df_clean['SeniorCitizen'].map({1: 'Yes', 0: 'No'})
                
        elif self.dataset_type == 'bank':
            # Handle missing values for bank dataset
            numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if col != self.target_column:
                    df_clean[col].fillna(df_clean[col].median(), inplace=True)
            
            # Remove unnecessary columns
            cols_to_drop = ['RowNumber', 'CustomerId', 'Surname']
            df_clean = df_clean.drop([col for col in cols_to_drop if col in df_clean.columns], axis=1)
        
        # General cleaning
        # Remove duplicates
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        final_rows = len(df_clean)
        
        if initial_rows != final_rows:
            print(f"🧹 Removed {initial_rows - final_rows} duplicate rows")
        
        print(f"✅ Data cleaning completed. Shape: {df_clean.shape}")
        return df_clean
    
    def identify_column_types(self, df):
        """
        Identify numerical and categorical columns
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            tuple: (numerical_columns, categorical_columns)
        """
        # Exclude target column
        feature_columns = [col for col in df.columns if col != self.target_column]
        
        numerical_columns = []
        categorical_columns = []
        
        for col in feature_columns:
            if df[col].dtype in ['int64', 'float64']:
                # Check if it's actually categorical (like binary 0/1)
                unique_values = df[col].nunique()
                if unique_values <= 10 and df[col].dtype == 'int64':
                    categorical_columns.append(col)
                else:
                    numerical_columns.append(col)
            else:
                categorical_columns.append(col)
        
        # Special handling for customerID type columns
        if self.dataset_type == 'telco':
            if 'customerID' in categorical_columns:
                categorical_columns.remove('customerID')
        
        print(f"📊 Numerical columns ({len(numerical_columns)}): {numerical_columns}")
        print(f"📋 Categorical columns ({len(categorical_columns)}): {categorical_columns}")
        
        return numerical_columns, categorical_columns
    
    def encode_features(self, df):
        """
        Encode categorical features and scale numerical features
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            tuple: (processed_df, feature_names)
        """
        # Separate features and target
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]
        
        # Remove customerID if present
        if 'customerID' in X.columns:
            X = X.drop(columns=['customerID'])
        
        # Identify column types
        numerical_columns, categorical_columns = self.identify_column_types(df)
        
        # Create preprocessing pipelines
        numerical_transformer = StandardScaler()
        
        # Use OneHotEncoder for categorical variables
        categorical_transformer = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
        
        # Create column transformer
        self.column_transformer = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_columns),
                ('cat', categorical_transformer, categorical_columns)
            ]
        )
        
        # Fit and transform the data
        X_processed = self.column_transformer.fit_transform(X)
        
        # Get feature names
        feature_names = []
        
        # Add numerical feature names
        feature_names.extend(numerical_columns)
        
        # Add categorical feature names
        if categorical_columns:
            cat_feature_names = self.column_transformer.named_transformers_['cat'].get_feature_names_out(categorical_columns)
            feature_names.extend(cat_feature_names)
        
        self.feature_names = feature_names
        
        # Create processed dataframe
        X_processed_df = pd.DataFrame(X_processed, columns=feature_names, index=X.index)
        processed_df = pd.concat([X_processed_df, y.reset_index(drop=True)], axis=1)
        
        print(f"🔧 Feature encoding completed. Final shape: {processed_df.shape}")
        print(f"📈 Total features after encoding: {len(feature_names)}")
        
        return processed_df, feature_names
    
    def preprocess_data(self, file_path):
        """
        Complete preprocessing pipeline
        
        Args:
            file_path (str): Path to the dataset file
            
        Returns:
            tuple: (processed_df, dataset_type, feature_names)
        """
        print(f"🚀 Starting preprocessing for: {file_path}")
        
        # Load data
        df = pd.read_csv(file_path)
        print(f"📁 Loaded data with shape: {df.shape}")
        
        # Detect dataset type
        dataset_type = self.detect_dataset_type(df)
        
        # Clean data
        df_clean = self.clean_data(df)
        
        # Encode features
        processed_df, feature_names = self.encode_features(df_clean)
        
        return processed_df, dataset_type, feature_names
    
    def save_processed_data(self, df, dataset_type, output_dir='.'):
        """
        Save processed data to CSV
        
        Args:
            df (pd.DataFrame): Processed dataframe
            dataset_type (str): Type of dataset ('telco' or 'bank')
            output_dir (str): Output directory
        """
        output_path = f"{output_dir}/processed_{dataset_type}.csv"
        df.to_csv(output_path, index=False)
        print(f"💾 Processed data saved to: {output_path}")
        return output_path


def main():
    """
    Demo function to test the preprocessor
    """
    preprocessor = ChurnDataPreprocessor()
    
    # Test with telco dataset
    print("="*50)
    print("TESTING WITH TELCO DATASET")
    print("="*50)
    
    try:
        processed_df, dataset_type, features = preprocessor.preprocess_data(
            'telco-customer/WA_Fn-UseC_-Telco-Customer-Churn.csv'
        )
        print(f"\n✅ Telco preprocessing successful!")
        print(f"Dataset type: {dataset_type}")
        print(f"Final shape: {processed_df.shape}")
        print(f"Features: {len(features)}")
        
        # Save processed data
        preprocessor.save_processed_data(processed_df, dataset_type)
        
    except Exception as e:
        print(f"❌ Error processing telco data: {e}")
    
    # Test with bank dataset
    print("\n" + "="*50)
    print("TESTING WITH BANK DATASET")
    print("="*50)
    
    try:
        preprocessor_bank = ChurnDataPreprocessor()
        processed_df, dataset_type, features = preprocessor_bank.preprocess_data(
            'bank-turnover/Churn_Modelling.csv'
        )
        print(f"\n✅ Bank preprocessing successful!")
        print(f"Dataset type: {dataset_type}")
        print(f"Final shape: {processed_df.shape}")
        print(f"Features: {len(features)}")
        
        # Save processed data
        preprocessor_bank.save_processed_data(processed_df, dataset_type)
        
    except Exception as e:
        print(f"❌ Error processing bank data: {e}")


if __name__ == "__main__":
    main()