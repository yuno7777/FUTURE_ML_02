"""
Evaluation Module for Churn Prediction System

This module handles:
- Model performance evaluation and visualization
- Confusion matrix plots
- ROC curve plots
- Feature importance visualization
- Churn distribution analysis
- Model comparison charts

Author: AI Data Scientist
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, classification_report,
    precision_recall_curve, average_precision_score
)
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for better looking plots
plt.style.use('default')
sns.set_palette("husl")

class ChurnModelEvaluator:
    """
    Comprehensive model evaluation and visualization class
    """
    
    def __init__(self, output_dir='reports/figures'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def plot_confusion_matrix(self, y_true, y_pred, model_name, save_plot=True):
        """
        Plot confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name (str): Name of the model
            save_plot (bool): Whether to save the plot
        """
        plt.figure(figsize=(8, 6))
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Churn', 'Churn'],
                   yticklabels=['No Churn', 'Churn'])
        
        plt.title(f'Confusion Matrix - {model_name}', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        
        # Add accuracy text
        accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
        plt.text(0.5, -0.1, f'Accuracy: {accuracy:.3f}', 
                ha='center', transform=plt.gca().transAxes, fontsize=12)
        
        plt.tight_layout()
        
        if save_plot:
            filename = f'{self.output_dir}/confusion_matrix_{model_name.lower().replace(" ", "_")}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"💾 Confusion matrix saved: {filename}")
        
        plt.show()
        return plt.gcf()
    
    def plot_roc_curve(self, models_results, save_plot=True):
        """
        Plot ROC curves for multiple models
        
        Args:
            models_results (dict): Dictionary containing model results
            save_plot (bool): Whether to save the plot
        """
        plt.figure(figsize=(10, 8))
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for i, (model_name, results) in enumerate(models_results.items()):
            if 'y_test' in results and 'probabilities' in results:
                y_test = results['y_test']
                y_prob = results['probabilities']
                
                # Calculate ROC curve
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                roc_auc = auc(fpr, tpr)
                
                # Plot
                plt.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
                        label=f'{model_name} (AUC = {roc_auc:.3f})')
        
        # Plot diagonal line
        plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--', alpha=0.5)
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves Comparison', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            filename = f'{self.output_dir}/roc_curves_comparison.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"💾 ROC curves saved: {filename}")
        
        plt.show()
        return plt.gcf()
    
    def plot_feature_importance(self, importance_df, model_name, top_n=15, save_plot=True):
        """
        Plot feature importance
        
        Args:
            importance_df (pd.DataFrame): Feature importance dataframe
            model_name (str): Name of the model
            top_n (int): Number of top features to plot
            save_plot (bool): Whether to save the plot
        """
        plt.figure(figsize=(12, max(8, top_n * 0.4)))
        
        # Get top features
        top_features = importance_df.head(top_n)
        
        # Get importance column name
        importance_col = [col for col in top_features.columns if col != 'Feature'][0]
        
        # Create horizontal bar plot
        bars = plt.barh(range(len(top_features)), top_features[importance_col], 
                       color='skyblue', edgecolor='navy', alpha=0.7)
        
        # Customize plot
        plt.yticks(range(len(top_features)), top_features['Feature'])
        plt.xlabel(importance_col, fontsize=12)
        plt.title(f'Top {top_n} Feature Importance - {model_name}', 
                 fontsize=16, fontweight='bold')
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + width*0.01, bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', ha='left', va='center', fontsize=9)
        
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        if save_plot:
            filename = f'{self.output_dir}/feature_importance_{model_name.lower().replace(" ", "_")}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"💾 Feature importance saved: {filename}")
        
        plt.show()
        return plt.gcf()
    
    def plot_churn_distribution(self, df, dataset_type, categorical_cols=None, save_plot=True):
        """
        Plot churn distribution by key categorical variables
        
        Args:
            df (pd.DataFrame): Input dataframe
            dataset_type (str): Type of dataset ('telco' or 'bank')
            categorical_cols (list): List of categorical columns to analyze
            save_plot (bool): Whether to save the plot
        """
        target_col = 'Churn' if dataset_type == 'telco' else 'Exited'
        
        if categorical_cols is None:
            if dataset_type == 'telco':
                categorical_cols = ['Contract', 'PaymentMethod', 'InternetService', 'gender']
            else:
                categorical_cols = ['Geography', 'Gender', 'HasCrCard', 'IsActiveMember']
        
        # Filter existing columns
        categorical_cols = [col for col in categorical_cols if col in df.columns]
        
        if not categorical_cols:
            print("❌ No categorical columns found for distribution analysis")
            return None
        
        # Create subplots
        n_cols = min(2, len(categorical_cols))
        n_rows = (len(categorical_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(categorical_cols):
            ax = axes[i] if len(categorical_cols) > 1 else axes[0]
            
            # Calculate churn rate by category
            churn_data = df.groupby(col)[target_col].agg(['count', 'sum']).reset_index()
            churn_data['churn_rate'] = churn_data['sum'] / churn_data['count']
            
            # Create bar plot
            bars = ax.bar(churn_data[col], churn_data['churn_rate'], 
                         color='coral', alpha=0.7, edgecolor='darkred')
            
            ax.set_title(f'Churn Rate by {col}', fontsize=12, fontweight='bold')
            ax.set_ylabel('Churn Rate', fontsize=10)
            ax.set_xlabel(col, fontsize=10)
            
            # Add value labels on bars
            for bar, rate in zip(bars, churn_data['churn_rate']):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{rate:.2%}', ha='center', va='bottom', fontsize=9)
            
            # Rotate x-axis labels if needed
            if len(churn_data[col].astype(str).str.len().max()) > 8:
                ax.tick_params(axis='x', rotation=45)
            
            ax.grid(axis='y', alpha=0.3)
        
        # Hide unused subplots
        for j in range(len(categorical_cols), len(axes)):
            axes[j].set_visible(False)
        
        plt.suptitle(f'Churn Distribution Analysis - {dataset_type.title()} Dataset', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_plot:
            filename = f'{self.output_dir}/churn_distribution_{dataset_type}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"💾 Churn distribution saved: {filename}")
        
        plt.show()
        return fig
    
    def plot_model_comparison(self, comparison_df, save_plot=True):
        """
        Plot model comparison across multiple metrics
        
        Args:
            comparison_df (pd.DataFrame): Model comparison dataframe
            save_plot (bool): Whether to save the plot
        """
        # Select key metrics for comparison
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        # Filter available metrics
        available_metrics = [m for m in metrics_to_plot if m in comparison_df.columns]
        
        if not available_metrics:
            print("❌ No metrics available for comparison plot")
            return None
        
        # Prepare data for plotting
        models = comparison_df['Model'].tolist()
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Set width of bars
        bar_width = 0.15
        positions = np.arange(len(models))
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        # Create bars for each metric
        for i, metric in enumerate(available_metrics):
            values = comparison_df[metric].tolist()
            bars = ax.bar(positions + i * bar_width, values, bar_width, 
                         label=metric.replace('_', ' ').title(), 
                         color=colors[i % len(colors)], alpha=0.8)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Customize plot
        ax.set_xlabel('Models', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Model Performance Comparison', fontsize=16, fontweight='bold')
        ax.set_xticks(positions + bar_width * (len(available_metrics) - 1) / 2)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 1.1)
        
        plt.tight_layout()
        
        if save_plot:
            filename = f'{self.output_dir}/model_comparison.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"💾 Model comparison saved: {filename}")
        
        plt.show()
        return fig
    
    def create_evaluation_summary(self, trainer_results, dataset_type):
        """
        Create a comprehensive evaluation summary with all visualizations
        
        Args:
            trainer_results (dict): Results from model trainer
            dataset_type (str): Type of dataset
        """
        print("🎨 Creating comprehensive evaluation summary...")
        print("="*60)
        
        # 1. Model comparison plot
        if 'model_comparison' in trainer_results:
            self.plot_model_comparison(trainer_results['model_comparison'])
        
        # 2. Feature importance plot
        if 'feature_importance' in trainer_results and trainer_results['feature_importance'] is not None:
            model_name = trainer_results.get('best_model', 'Best Model')
            self.plot_feature_importance(trainer_results['feature_importance'], model_name)
        
        print("\n📊 Evaluation summary completed!")
        print(f"📁 All plots saved to: {self.output_dir}")
        
        return True
    
    def generate_evaluation_report(self, trainer_results, dataset_type, original_df=None):
        """
        Generate a complete evaluation report with metrics and visualizations
        
        Args:
            trainer_results (dict): Results from model trainer
            dataset_type (str): Type of dataset
            original_df (pd.DataFrame): Original dataframe for distribution analysis
        """
        print("📋 Generating comprehensive evaluation report...")
        print("="*70)
        
        # Create timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # 1. Print summary header
        print(f"\n🎯 CHURN PREDICTION MODEL EVALUATION REPORT")
        print(f"📅 Generated: {timestamp}")
        print(f"📊 Dataset: {dataset_type.upper()}")
        print("="*70)
        
        # 2. Best model summary
        if 'best_model' in trainer_results and 'best_metrics' in trainer_results:
            best_model = trainer_results['best_model']
            metrics = trainer_results['best_metrics']
            
            print(f"\n🏆 BEST MODEL: {best_model}")
            print("-" * 40)
            print(f"Accuracy:  {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall:    {metrics['recall']:.4f}")
            print(f"F1-Score:  {metrics['f1_score']:.4f}")
            print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
            
            if 'cv_auc_mean' in metrics:
                print(f"CV AUC:    {metrics['cv_auc_mean']:.4f} ± {metrics['cv_auc_std']:.4f}")
        
        # 3. Model comparison
        if 'model_comparison' in trainer_results:
            print(f"\n📊 MODEL COMPARISON:")
            print("-" * 40)
            comparison_df = trainer_results['model_comparison']
            print(comparison_df.round(4).to_string(index=False))
        
        # 4. Top features
        if 'feature_importance' in trainer_results and trainer_results['feature_importance'] is not None:
            print(f"\n📈 TOP 10 IMPORTANT FEATURES:")
            print("-" * 40)
            importance_df = trainer_results['feature_importance']
            print(importance_df.head(10).to_string(index=False))
        
        # 5. Generate visualizations
        self.create_evaluation_summary(trainer_results, dataset_type)
        
        # 6. Generate churn distribution if original data provided
        if original_df is not None:
            self.plot_churn_distribution(original_df, dataset_type)
        
        print("\n" + "="*70)
        print("✅ EVALUATION REPORT COMPLETED!")
        print(f"📁 All visualizations saved to: {self.output_dir}")
        
        return True


def main():
    """
    Demo function to test evaluation module
    """
    # This would typically be called after model training
    print("🧪 Testing evaluation module...")
    
    # Create sample data for demonstration
    evaluator = ChurnModelEvaluator()
    
    # Test creating output directory
    print(f"✅ Evaluator initialized. Output directory: {evaluator.output_dir}")
    
    print("\n📝 Note: This module is designed to work with model training results.")
    print("   Run the complete pipeline to see full evaluation in action!")


if __name__ == "__main__":
    main()