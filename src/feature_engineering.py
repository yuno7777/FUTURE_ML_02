"""
Feature Engineering Module for Churn Prediction System

This module creates additional engineered features to improve model performance:
- Tenure buckets (Short/Medium/Long)
- Total services count
- Payment method types (Online/Offline)
- Contract length indicators
- Average monthly spend metrics
- Customer value segments

Author: AI Data Scientist
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class ChurnFeatureEngineer:
    """
    Feature engineering class for creating additional predictive features
    """
    
    def __init__(self, dataset_type):
        self.dataset_type = dataset_type
        self.feature_mappings = {}
        
    def create_tenure_buckets(self, df):
        """
        Create tenure buckets: Short (0-12), Medium (13-36), Long (36+)
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with tenure buckets
        """
        df_enhanced = df.copy()
        
        if 'tenure' in df.columns:
            def categorize_tenure(tenure):
                if tenure <= 12:
                    return 'Short'
                elif tenure <= 36:
                    return 'Medium'
                else:
                    return 'Long'
            
            df_enhanced['tenure_bucket'] = df['tenure'].apply(categorize_tenure)
            print("✅ Created tenure buckets: Short (0-12), Medium (13-36), Long (36+)")
            
        elif 'Tenure' in df.columns:  # Bank dataset
            def categorize_tenure_bank(tenure):
                if tenure <= 3:
                    return 'Short'
                elif tenure <= 7:
                    return 'Medium'
                else:
                    return 'Long'
            
            df_enhanced['tenure_bucket'] = df['Tenure'].apply(categorize_tenure_bank)
            print("✅ Created tenure buckets for bank data: Short (0-3), Medium (4-7), Long (8+)")
        
        return df_enhanced
    
    def create_services_count(self, df):
        """
        Create total services count for telco customers
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with services count
        """
        df_enhanced = df.copy()
        
        if self.dataset_type == 'telco':
            # List of service columns
            service_columns = [
                'PhoneService', 'MultipleLines', 'InternetService',
                'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport', 'StreamingTV', 'StreamingMovies'
            ]
            
            # Count services (excluding 'No' and 'No internet service')
            df_enhanced['total_services'] = 0
            
            for col in service_columns:
                if col in df.columns:
                    # Count as service if not 'No' or 'No internet service' or 'No phone service'
                    service_mask = ~df[col].isin(['No', 'No internet service', 'No phone service'])
                    df_enhanced['total_services'] += service_mask.astype(int)
            
            print(f"✅ Created total services count (0-{df_enhanced['total_services'].max()})")
            
        elif self.dataset_type == 'bank':
            # For bank data, create product-related features
            if 'NumOfProducts' in df.columns:
                df_enhanced['products_bucket'] = df['NumOfProducts'].apply(
                    lambda x: 'Single' if x == 1 else 'Multiple' if x <= 2 else 'High'
                )
                print("✅ Created product buckets for bank data")
                
        return df_enhanced
    
    def create_payment_method_type(self, df):
        """
        Categorize payment methods as Online vs Offline
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with payment method types
        """
        df_enhanced = df.copy()
        
        if 'PaymentMethod' in df.columns:
            def categorize_payment(payment_method):
                online_methods = ['Electronic check', 'Credit card (automatic)', 'Bank transfer (automatic)']
                return 'Online' if payment_method in online_methods else 'Offline'
            
            df_enhanced['payment_method_type'] = df['PaymentMethod'].apply(categorize_payment)
            print("✅ Created payment method types: Online vs Offline")
            
        return df_enhanced
    
    def create_contract_indicators(self, df):
        """
        Create contract length indicators
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with contract indicators
        """
        df_enhanced = df.copy()
        
        if 'Contract' in df.columns:
            # Create binary indicators for contract types
            df_enhanced['is_month_to_month'] = (df['Contract'] == 'Month-to-month').astype(int)
            df_enhanced['is_long_term'] = (df['Contract'].isin(['One year', 'Two year'])).astype(int)
            
            # Create contract length score
            contract_mapping = {'Month-to-month': 1, 'One year': 12, 'Two year': 24}
            df_enhanced['contract_length_months'] = df['Contract'].map(contract_mapping)
            
            print("✅ Created contract length indicators")
            
        return df_enhanced
    
    def create_spending_metrics(self, df):
        """
        Create average monthly spend and spending-related features
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with spending metrics
        """
        df_enhanced = df.copy()
        
        if self.dataset_type == 'telco':
            if 'MonthlyCharges' in df.columns and 'TotalCharges' in df.columns and 'tenure' in df.columns:
                # Average monthly spend (total charges / tenure)
                df_enhanced['avg_monthly_spend'] = np.where(
                    df['tenure'] > 0, 
                    df['TotalCharges'] / df['tenure'], 
                    df['MonthlyCharges']
                )
                
                # Spending efficiency (monthly charges vs average)
                avg_monthly_charge = df['MonthlyCharges'].mean()
                df_enhanced['spending_ratio'] = df['MonthlyCharges'] / avg_monthly_charge
                
                # High value customer flag
                df_enhanced['is_high_value'] = (df['MonthlyCharges'] > df['MonthlyCharges'].quantile(0.75)).astype(int)
                
                print("✅ Created spending metrics for telco data")
                
        elif self.dataset_type == 'bank':
            if 'Balance' in df.columns and 'EstimatedSalary' in df.columns:
                # Balance to salary ratio
                df_enhanced['balance_salary_ratio'] = np.where(
                    df['EstimatedSalary'] > 0,
                    df['Balance'] / df['EstimatedSalary'],
                    0
                )
                
                # High balance customer
                df_enhanced['is_high_balance'] = (df['Balance'] > df['Balance'].quantile(0.75)).astype(int)
                
                # Salary bucket
                salary_median = df['EstimatedSalary'].median()
                df_enhanced['salary_bucket'] = np.where(
                    df['EstimatedSalary'] > salary_median, 'High', 'Low'
                )
                
                print("✅ Created financial metrics for bank data")
                
        return df_enhanced
    
    def create_customer_segments(self, df):
        """
        Create customer value segments based on multiple factors
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with customer segments
        """
        df_enhanced = df.copy()
        
        if self.dataset_type == 'telco':
            # Create customer segments based on tenure, charges, and services
            conditions = []
            
            if all(col in df.columns for col in ['tenure', 'MonthlyCharges', 'total_services']):
                # High value: long tenure, high charges, many services
                high_value = (
                    (df['tenure'] > 24) & 
                    (df['MonthlyCharges'] > df['MonthlyCharges'].quantile(0.7)) &
                    (df.get('total_services', 0) > 3)
                )
                
                # At risk: short tenure, high charges
                at_risk = (
                    (df['tenure'] <= 12) & 
                    (df['MonthlyCharges'] > df['MonthlyCharges'].median())
                )
                
                # New customer: very short tenure
                new_customer = df['tenure'] <= 6
                
                # Create segments
                df_enhanced['customer_segment'] = 'Standard'
                df_enhanced.loc[high_value, 'customer_segment'] = 'High_Value'
                df_enhanced.loc[at_risk, 'customer_segment'] = 'At_Risk'
                df_enhanced.loc[new_customer, 'customer_segment'] = 'New'
                
                print("✅ Created customer segments: High_Value, At_Risk, New, Standard")
                
        elif self.dataset_type == 'bank':
            # Create segments based on balance, salary, and products
            if all(col in df.columns for col in ['Balance', 'EstimatedSalary', 'NumOfProducts']):
                # Premium: high balance and salary
                premium = (
                    (df['Balance'] > df['Balance'].quantile(0.8)) &
                    (df['EstimatedSalary'] > df['EstimatedSalary'].quantile(0.7))
                )
                
                # At risk: low balance, single product
                at_risk = (
                    (df['Balance'] < df['Balance'].quantile(0.3)) &
                    (df['NumOfProducts'] == 1)
                )
                
                # Create segments
                df_enhanced['customer_segment'] = 'Standard'
                df_enhanced.loc[premium, 'customer_segment'] = 'Premium'
                df_enhanced.loc[at_risk, 'customer_segment'] = 'At_Risk'
                
                print("✅ Created customer segments: Premium, At_Risk, Standard")
                
        return df_enhanced
    
    def create_interaction_features(self, df):
        """
        Create interaction features between important variables
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with interaction features
        """
        df_enhanced = df.copy()
        
        if self.dataset_type == 'telco':
            # Tenure × Monthly Charges interaction
            if 'tenure' in df.columns and 'MonthlyCharges' in df.columns:
                df_enhanced['tenure_charges_interaction'] = df['tenure'] * df['MonthlyCharges']
            
            # Services × Charges interaction
            if 'total_services' in df.columns and 'MonthlyCharges' in df.columns:
                df_enhanced['services_charges_interaction'] = df['total_services'] * df['MonthlyCharges']
                
        elif self.dataset_type == 'bank':
            # Age × Balance interaction
            if 'Age' in df.columns and 'Balance' in df.columns:
                df_enhanced['age_balance_interaction'] = df['Age'] * df['Balance']
            
            # CreditScore × Balance interaction
            if 'CreditScore' in df.columns and 'Balance' in df.columns:
                df_enhanced['credit_balance_interaction'] = df['CreditScore'] * df['Balance']
        
        print("✅ Created interaction features")
        return df_enhanced
    
    def engineer_all_features(self, df):
        """
        Apply all feature engineering techniques
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with all engineered features
        """
        print(f"🔧 Starting feature engineering for {self.dataset_type} dataset...")
        print(f"Initial shape: {df.shape}")
        
        # Apply all feature engineering steps
        df_enhanced = df.copy()
        
        # 1. Create tenure buckets
        df_enhanced = self.create_tenure_buckets(df_enhanced)
        
        # 2. Create services count
        df_enhanced = self.create_services_count(df_enhanced)
        
        # 3. Create payment method types
        df_enhanced = self.create_payment_method_type(df_enhanced)
        
        # 4. Create contract indicators
        df_enhanced = self.create_contract_indicators(df_enhanced)
        
        # 5. Create spending metrics
        df_enhanced = self.create_spending_metrics(df_enhanced)
        
        # 6. Create customer segments
        df_enhanced = self.create_customer_segments(df_enhanced)
        
        # 7. Create interaction features
        df_enhanced = self.create_interaction_features(df_enhanced)
        
        print(f"🎯 Feature engineering completed!")
        print(f"Final shape: {df_enhanced.shape}")
        print(f"Added {df_enhanced.shape[1] - df.shape[1]} new features")
        
        # Display new feature summary
        new_features = [col for col in df_enhanced.columns if col not in df.columns]
        if new_features:
            print(f"🆕 New features created: {new_features}")
        
        return df_enhanced


def main():
    """
    Demo function to test feature engineering
    """
    # Test with both datasets
    datasets = [
        ('telco-customer/WA_Fn-UseC_-Telco-Customer-Churn.csv', 'telco'),
        ('bank-turnover/Churn_Modelling.csv', 'bank')
    ]
    
    for file_path, dataset_type in datasets:
        print("="*60)
        print(f"TESTING FEATURE ENGINEERING: {dataset_type.upper()} DATASET")
        print("="*60)
        
        try:
            # Load data
            df = pd.read_csv(file_path)
            print(f"Loaded {dataset_type} data: {df.shape}")
            
            # Initialize feature engineer
            engineer = ChurnFeatureEngineer(dataset_type)
            
            # Apply feature engineering
            df_enhanced = engineer.engineer_all_features(df)
            
            # Save enhanced data
            output_path = f"enhanced_{dataset_type}.csv"
            df_enhanced.to_csv(output_path, index=False)
            print(f"💾 Enhanced data saved to: {output_path}")
            
            # Show sample of new features
            print("\n📊 Sample of enhanced data:")
            print(df_enhanced.head(3))
            
        except Exception as e:
            print(f"❌ Error in feature engineering for {dataset_type}: {e}")
        
        print("\n")


if __name__ == "__main__":
    main()