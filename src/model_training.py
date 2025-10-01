"""
Model Training Module for Churn Prediction System

This module handles:
- Training multiple classification models (Logistic Regression, Random Forest, XGBoost)
- Stratified train-test split
- Model evaluation with multiple metrics
- Model comparison and selection
- Best model saving

Author: AI Data Scientist
"""

import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime

# Scikit-learn imports
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, classification_report, confusion_matrix
)

# XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost not available. Will skip XGBoost training.")

import warnings
warnings.filterwarnings('ignore')

class ChurnModelTrainer:
    """
    Comprehensive model training class for churn prediction
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        
    def prepare_data(self, df, target_column, test_size=0.2):
        """
        Prepare data for training with stratified split
        
        Args:
            df (pd.DataFrame): Input dataframe
            target_column (str): Name of target column
            test_size (float): Test set size ratio
        """
        print(f"📊 Preparing data for training...")
        
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        self.feature_names = X.columns.tolist()
        
        # Stratified train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        print(f"✅ Data split completed:")
        print(f"   Training set: {self.X_train.shape}")
        print(f"   Test set: {self.X_test.shape}")
        print(f"   Class distribution in training set:")
        print(f"   {self.y_train.value_counts().to_dict()}")
        
    def initialize_models(self):
        """
        Initialize all models with optimal hyperparameters
        """
        print("🤖 Initializing models...")
        
        # Logistic Regression
        self.models['Logistic Regression'] = LogisticRegression(
            random_state=self.random_state,
            max_iter=1000,
            class_weight='balanced'
        )
        
        # Random Forest
        self.models['Random Forest'] = RandomForestClassifier(
            n_estimators=100,
            random_state=self.random_state,
            class_weight='balanced',
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2
        )
        
        # XGBoost (if available)
        if XGBOOST_AVAILABLE:
            self.models['XGBoost'] = xgb.XGBClassifier(
                random_state=self.random_state,
                eval_metric='logloss',
                scale_pos_weight=1,
                max_depth=6,
                learning_rate=0.1,
                n_estimators=100
            )
        
        print(f"✅ Initialized {len(self.models)} models")
        
    def train_models(self):
        """
        Train all models and evaluate performance
        """
        print("🏋️‍♂️ Training models...")
        
        for name, model in self.models.items():
            print(f"\n🔄 Training {name}...")
            
            try:
                # Train the model
                model.fit(self.X_train, self.y_train)
                
                # Make predictions
                y_pred = model.predict(self.X_test)
                y_pred_proba = model.predict_proba(self.X_test)[:, 1]
                
                # Calculate metrics
                metrics = {
                    'accuracy': accuracy_score(self.y_test, y_pred),
                    'precision': precision_score(self.y_test, y_pred),
                    'recall': recall_score(self.y_test, y_pred),
                    'f1_score': f1_score(self.y_test, y_pred),
                    'roc_auc': roc_auc_score(self.y_test, y_pred_proba)
                }
                
                # Cross-validation score
                cv_scores = cross_val_score(
                    model, self.X_train, self.y_train, 
                    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
                    scoring='roc_auc'
                )
                metrics['cv_auc_mean'] = cv_scores.mean()
                metrics['cv_auc_std'] = cv_scores.std()
                
                # Store results
                self.results[name] = {
                    'model': model,
                    'metrics': metrics,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba,
                    'confusion_matrix': confusion_matrix(self.y_test, y_pred)
                }
                
                print(f"✅ {name} trained successfully")
                print(f"   Accuracy: {metrics['accuracy']:.4f}")
                print(f"   ROC-AUC: {metrics['roc_auc']:.4f}")
                print(f"   F1-Score: {metrics['f1_score']:.4f}")
                
            except Exception as e:
                print(f"❌ Error training {name}: {e}")
                
    def select_best_model(self, metric='roc_auc'):
        """
        Select the best model based on specified metric
        
        Args:
            metric (str): Metric to use for selection
        """
        print(f"\n🏆 Selecting best model based on {metric}...")
        
        best_score = 0
        best_name = None
        
        for name, result in self.results.items():
            score = result['metrics'][metric]
            if score > best_score:
                best_score = score
                best_name = name
        
        if best_name:
            self.best_model = self.results[best_name]['model']
            self.best_model_name = best_name
            
            print(f"🥇 Best model: {best_name}")
            print(f"   {metric.upper()}: {best_score:.4f}")
            
            return self.best_model, best_name
        
        return None, None
    
    def get_model_comparison(self):
        """
        Create a comparison dataframe of all models
        
        Returns:
            pd.DataFrame: Model comparison results
        """
        comparison_data = []
        
        for name, result in self.results.items():
            row = {'Model': name}
            row.update(result['metrics'])
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Sort by ROC-AUC (descending)
        comparison_df = comparison_df.sort_values('roc_auc', ascending=False)
        
        print("\n📊 Model Comparison Results:")
        print("="*80)
        print(comparison_df.round(4))
        
        return comparison_df
    
    def get_feature_importance(self, top_n=20):
        """
        Get feature importance from the best model
        
        Args:
            top_n (int): Number of top features to return
            
        Returns:
            pd.DataFrame: Feature importance dataframe
        """
        if not self.best_model:
            print("❌ No best model selected yet!")
            return None
        
        importance_data = []
        
        if hasattr(self.best_model, 'feature_importances_'):
            # Tree-based models (Random Forest, XGBoost)
            importances = self.best_model.feature_importances_
            importance_type = 'Feature Importance'
            
        elif hasattr(self.best_model, 'coef_'):
            # Linear models (Logistic Regression)
            importances = np.abs(self.best_model.coef_[0])
            importance_type = 'Coefficient Magnitude'
            
        else:
            print("❌ Selected model doesn't support feature importance!")
            return None
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            importance_type: importances
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values(importance_type, ascending=False)
        
        print(f"\n📈 Top {min(top_n, len(importance_df))} Feature Importances ({self.best_model_name}):")
        print("="*60)
        print(importance_df.head(top_n))
        
        return importance_df.head(top_n)
    
    def generate_classification_report(self):
        """
        Generate detailed classification report for the best model
        """
        if not self.best_model:
            print("❌ No best model selected yet!")
            return None
        
        y_pred = self.results[self.best_model_name]['predictions']
        
        print(f"\n📋 Detailed Classification Report ({self.best_model_name}):")
        print("="*60)
        print(classification_report(self.y_test, y_pred))
        
        return classification_report(self.y_test, y_pred, output_dict=True)
    
    def save_best_model(self, filepath='models/churn_model.pkl'):
        """
        Save the best model to disk
        
        Args:
            filepath (str): Path to save the model
        """
        if not self.best_model:
            print("❌ No best model to save!")
            return False
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Prepare model package
            model_package = {
                'model': self.best_model,
                'model_name': self.best_model_name,
                'feature_names': self.feature_names,
                'metrics': self.results[self.best_model_name]['metrics'],
                'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'train_shape': self.X_train.shape,
                'test_shape': self.X_test.shape
            }
            
            # Save model
            with open(filepath, 'wb') as f:
                pickle.dump(model_package, f)
            
            print(f"💾 Best model saved to: {filepath}")
            print(f"   Model: {self.best_model_name}")
            print(f"   ROC-AUC: {self.results[self.best_model_name]['metrics']['roc_auc']:.4f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            return False
    
    def train_complete_pipeline(self, df, target_column, model_save_path='models/churn_model.pkl'):
        """
        Complete training pipeline
        
        Args:
            df (pd.DataFrame): Input dataframe
            target_column (str): Target column name
            model_save_path (str): Path to save the best model
            
        Returns:
            dict: Training results summary
        """
        print("🚀 Starting complete training pipeline...")
        print("="*60)
        
        # 1. Prepare data
        self.prepare_data(df, target_column)
        
        # 2. Initialize models
        self.initialize_models()
        
        # 3. Train models
        self.train_models()
        
        # 4. Select best model
        best_model, best_name = self.select_best_model()
        
        # 5. Show comparison
        comparison_df = self.get_model_comparison()
        
        # 6. Show feature importance
        feature_importance = self.get_feature_importance()
        
        # 7. Generate classification report
        class_report = self.generate_classification_report()
        
        # 8. Save best model
        save_success = self.save_best_model(model_save_path)
        
        # Summary
        summary = {
            'best_model': best_name,
            'best_metrics': self.results[best_name]['metrics'] if best_name else None,
            'model_comparison': comparison_df,
            'feature_importance': feature_importance,
            'classification_report': class_report,
            'model_saved': save_success,
            'model_path': model_save_path if save_success else None
        }
        
        print("\n" + "="*60)
        print("🎉 Training pipeline completed successfully!")
        
        return summary


def load_saved_model(filepath='models/churn_model.pkl'):
    """
    Load a saved model from disk
    
    Args:
        filepath (str): Path to the saved model
        
    Returns:
        dict: Model package
    """
    try:
        with open(filepath, 'rb') as f:
            model_package = pickle.load(f)
        
        print(f"✅ Model loaded successfully from: {filepath}")
        print(f"   Model: {model_package['model_name']}")
        print(f"   Training date: {model_package['training_date']}")
        print(f"   ROC-AUC: {model_package['metrics']['roc_auc']:.4f}")
        
        return model_package
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None


def main():
    """
    Demo function to test model training
    """
    from data_preprocessing import ChurnDataPreprocessor
    from feature_engineering import ChurnFeatureEngineer
    
    # Test with telco dataset
    print("="*70)
    print("TESTING MODEL TRAINING: TELCO DATASET")
    print("="*70)
    
    try:
        # 1. Load and preprocess data
        preprocessor = ChurnDataPreprocessor()
        processed_df, dataset_type, features = preprocessor.preprocess_data(
            'telco-customer/WA_Fn-UseC_-Telco-Customer-Churn.csv'
        )
        
        # 2. Feature engineering
        engineer = ChurnFeatureEngineer(dataset_type)
        enhanced_df = engineer.engineer_all_features(processed_df)
        
        # 3. Train models
        trainer = ChurnModelTrainer()
        target_col = 'Churn' if dataset_type == 'telco' else 'Exited'
        
        summary = trainer.train_complete_pipeline(
            enhanced_df, target_col, 'models/telco_churn_model.pkl'
        )
        
        print(f"\n✅ Telco model training completed!")
        
    except Exception as e:
        print(f"❌ Error in telco model training: {e}")


if __name__ == "__main__":
    main()