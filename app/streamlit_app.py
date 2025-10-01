"""
Complete Churn Prediction System - Integrated Streamlit Dashboard

End-to-end ML workflow: Data Upload → Preprocessing → Training → Evaluation → Prediction
All within the Streamlit UI - no terminal commands needed!
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import io
from datetime import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, roc_curve, classification_report
)
import warnings
warnings.filterwarnings('ignore')

# Try to import XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# Try to import ReportLab for PDF generation
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="🎯 Churn Prediction System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .tab-header {
        font-size: 1.8rem;
        color: #2e7d32;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .prediction-result {
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .high-risk {
        background-color: #ffebeb;
        color: #d32f2f;
        border: 2px solid #d32f2f;
    }
    .low-risk {
        background-color: #e8f5e8;
        color: #2e7d32;
        border: 2px solid #2e7d32;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'model_results' not in st.session_state:
    st.session_state.model_results = None
if 'dataset_type' not in st.session_state:
    st.session_state.dataset_type = None
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'encoders' not in st.session_state:
    st.session_state.encoders = {}

@st.cache_data
def load_dataset(dataset_choice, uploaded_file=None):
    """Load dataset based on user choice"""
    try:
        if dataset_choice == "Telco Dataset":
            if os.path.exists('../telco-customer/WA_Fn-UseC_-Telco-Customer-Churn.csv'):
                df = pd.read_csv('../telco-customer/WA_Fn-UseC_-Telco-Customer-Churn.csv')
                return df, 'telco'
            elif os.path.exists('telco-customer/WA_Fn-UseC_-Telco-Customer-Churn.csv'):
                df = pd.read_csv('telco-customer/WA_Fn-UseC_-Telco-Customer-Churn.csv')
                return df, 'telco'
            else:
                st.error("❌ Telco dataset not found in expected directories")
                return None, None
        elif dataset_choice == "Bank Dataset":
            if os.path.exists('../bank-turnover/Churn_Modelling.csv'):
                df = pd.read_csv('../bank-turnover/Churn_Modelling.csv')
                return df, 'bank'
            elif os.path.exists('bank-turnover/Churn_Modelling.csv'):
                df = pd.read_csv('bank-turnover/Churn_Modelling.csv')
                return df, 'bank'
            else:
                st.error("❌ Bank dataset not found in expected directories")
                return None, None
        elif dataset_choice == "Upload Custom CSV" and uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            # Auto-detect dataset type based on columns
            if 'Churn' in df.columns:
                return df, 'telco'
            elif 'Exited' in df.columns:
                return df, 'bank'
            else:
                return df, 'custom'
    except Exception as e:
        st.error(f"❌ Error loading dataset: {e}")
        return None, None

def detect_dataset_type(df):
    """Automatically detect dataset type"""
    telco_indicators = ['customerID', 'MonthlyCharges', 'TotalCharges', 'InternetService']
    bank_indicators = ['CustomerId', 'Surname', 'CreditScore', 'Geography', 'Balance']
    
    telco_score = sum(1 for col in telco_indicators if col in df.columns)
    bank_score = sum(1 for col in bank_indicators if col in df.columns)
    
    if bank_score > telco_score:
        return 'bank'
    else:
        return 'telco'

@st.cache_data
def preprocess_data(df, dataset_type):
    """Preprocess the data"""
    df_clean = df.copy()
    
    # Determine target column
    if dataset_type == 'telco':
        target_col = 'Churn'
        # Handle TotalCharges
        if 'TotalCharges' in df_clean.columns:
            df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
            df_clean['TotalCharges'].fillna(df_clean['TotalCharges'].median(), inplace=True)
        # Convert target
        if target_col in df_clean.columns:
            df_clean[target_col] = df_clean[target_col].map({'Yes': 1, 'No': 0})
        # Convert SeniorCitizen
        if 'SeniorCitizen' in df_clean.columns:
            df_clean['SeniorCitizen'] = df_clean['SeniorCitizen'].map({1: 'Yes', 0: 'No'})
    else:  # bank
        target_col = 'Exited'
        # Remove unnecessary columns
        cols_to_drop = ['RowNumber', 'CustomerId', 'Surname']
        df_clean = df_clean.drop([col for col in cols_to_drop if col in df_clean.columns], axis=1)
    
    # Remove customerID for telco
    if 'customerID' in df_clean.columns:
        df_clean = df_clean.drop(columns=['customerID'])
    
    # Handle missing values
    numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if col != target_col:
            df_clean[col].fillna(df_clean[col].median(), inplace=True)
    
    categorical_columns = df_clean.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        if col != target_col:
            df_clean[col].fillna(df_clean[col].mode()[0] if len(df_clean[col].mode()) > 0 else 'Unknown', inplace=True)
    
    return df_clean, target_col
    
def data_upload_tab():
    """Tab 1: Data Upload & Preview"""
    st.markdown('<div class="tab-header">📁 Data Upload & Preview</div>', unsafe_allow_html=True)
    
    # Dataset selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        dataset_choice = st.selectbox(
            "Choose Dataset:",
            ["Telco Dataset", "Bank Dataset", "Upload Custom CSV"]
        )
    
    with col2:
        st.write("")
        st.write("")
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
    
    uploaded_file = None
    if dataset_choice == "Upload Custom CSV":
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    # Load data
    if dataset_choice != "Upload Custom CSV" or uploaded_file is not None:
        with st.spinner("Loading dataset..."):
            df, dataset_type = load_dataset(dataset_choice, uploaded_file)
        
        if df is not None:
            st.session_state.data = df
            st.session_state.dataset_type = dataset_type if dataset_type else detect_dataset_type(df)
            
            st.success(f"✅ Dataset loaded successfully! Type: {st.session_state.dataset_type.upper()}")
            
            # Display dataset info
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📊 Rows", f"{len(df):,}")
            with col2:
                st.metric("📋 Columns", f"{len(df.columns):,}")
            with col3:
                missing_count = df.isnull().sum().sum()
                st.metric("❓ Missing Values", f"{missing_count:,}")
            with col4:
                if st.session_state.dataset_type == 'telco' and 'Churn' in df.columns:
                    churn_rate = (df['Churn'] == 'Yes').mean()
                    st.metric("🎯 Churn Rate", f"{churn_rate:.1%}")
                elif st.session_state.dataset_type == 'bank' and 'Exited' in df.columns:
                    churn_rate = df['Exited'].mean()
                    st.metric("🎯 Churn Rate", f"{churn_rate:.1%}")
                else:
                    st.metric("🎯 Churn Rate", "N/A")
            
            # Display first 10 rows
            st.subheader("📋 Dataset Preview (First 10 rows)")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Display column info
            with st.expander("📈 Column Information"):
                col_info = pd.DataFrame({
                    'Column': df.columns,
                    'Data Type': df.dtypes,
                    'Non-Null Count': df.count(),
                    'Null Count': df.isnull().sum(),
                    'Unique Values': df.nunique()
                })
                st.dataframe(col_info, use_container_width=True)
            
            # Preprocess button
            st.markdown("---")
            if st.button("🔧 Preprocess Data", type="primary"):
                with st.spinner("Preprocessing data..."):
                    processed_df, target_col = preprocess_data(df, st.session_state.dataset_type)
                    st.session_state.processed_data = processed_df
                    st.session_state.target_column = target_col
                
                st.success("✅ Data successfully preprocessed!")
                st.balloons()
                
                # Show preprocessing results
                st.subheader("🎯 Preprocessing Results")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Original Shape", f"{df.shape}")
                    st.metric("Processed Shape", f"{processed_df.shape}")
                
                with col2:
                    missing_after = processed_df.isnull().sum().sum()
                    st.metric("Missing Values After", f"{missing_after:,}")
                    st.metric("Target Column", target_col)
        else:
            st.warning("⚠️ Please select a valid dataset to continue.")
    else:
        st.info("👆 Please choose a dataset or upload a CSV file to get started.")
    
def data_preprocessing_tab():
    """Tab 2: Data Preprocessing"""
    st.markdown('<div class="tab-header">🔧 Data Preprocessing</div>', unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.warning("⚠️ Please upload and load a dataset first in the 'Data Upload & Preview' tab.")
        return
    
    df = st.session_state.data
    
    st.subheader("📊 Current Dataset Status")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Dataset Type", st.session_state.dataset_type.upper())
    with col2:
        st.metric("Rows × Columns", f"{df.shape[0]} × {df.shape[1]}")
    with col3:
        missing_vals = df.isnull().sum().sum()
        st.metric("Missing Values", f"{missing_vals:,}")
    
    # Show missing value details
    if missing_vals > 0:
        st.subheader("❓ Missing Values by Column")
        missing_info = df.isnull().sum()
        missing_info = missing_info[missing_info > 0].sort_values(ascending=False)
        
        if len(missing_info) > 0:
            fig = px.bar(
                x=missing_info.index,
                y=missing_info.values,
                title="Missing Values by Column",
                labels={'x': 'Columns', 'y': 'Missing Count'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Preprocessing options
    st.subheader("⚙️ Preprocessing Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Missing Value Strategy:**")
        numeric_strategy = st.selectbox(
            "Numeric columns:",
            ["Median", "Mean", "Drop rows"]
        )
        categorical_strategy = st.selectbox(
            "Categorical columns:",
            ["Most frequent", "Drop rows", "Fill with 'Unknown'"]
        )
    
    with col2:
        st.write("**Encoding Strategy:**")
        encoding_strategy = st.selectbox(
            "Categorical encoding:",
            ["One-Hot Encoding", "Label Encoding"]
        )
        scaling_strategy = st.selectbox(
            "Numeric scaling:",
            ["Standard Scaler", "Min-Max Scaler", "No Scaling"]
        )
    
    # Process data button
    if st.button("🚀 Start Preprocessing", type="primary"):
        with st.spinner("Processing data..."):
            try:
                processed_df, target_col = preprocess_data(df, st.session_state.dataset_type)
                
                # Apply advanced preprocessing based on user choices
                # Separate features and target
                if target_col in processed_df.columns:
                    X = processed_df.drop(columns=[target_col])
                    y = processed_df[target_col]
                else:
                    st.error(f"❌ Target column '{target_col}' not found in dataset")
                    return
                
                # Handle categorical and numeric columns
                categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
                numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
                
                # Encode categorical variables
                encoders = {}
                if len(categorical_cols) > 0:
                    if encoding_strategy == "One-Hot Encoding":
                        X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
                    else:  # Label Encoding
                        X_encoded = X.copy()
                        for col in categorical_cols:
                            le = LabelEncoder()
                            X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
                            encoders[col] = le
                else:
                    X_encoded = X.copy()
                
                # Scale numeric features
                scaler = None
                if scaling_strategy != "No Scaling" and len(numeric_cols) > 0:
                    if scaling_strategy == "Standard Scaler":
                        scaler = StandardScaler()
                    # For simplicity, we'll just use StandardScaler
                    X_encoded[numeric_cols] = scaler.fit_transform(X_encoded[numeric_cols])
                
                # Store processed data
                final_df = pd.concat([X_encoded, y], axis=1)
                
                st.session_state.processed_data = final_df
                st.session_state.target_column = target_col
                st.session_state.feature_names = X_encoded.columns.tolist()
                st.session_state.scaler = scaler
                st.session_state.encoders = encoders
                
                st.success("✅ Data successfully preprocessed!")
                st.balloons()
                
                # Display results
                st.subheader("🎯 Preprocessing Results")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Features After Encoding", len(X_encoded.columns))
                with col2:
                    st.metric("Target Column", target_col)
                with col3:
                    missing_after = final_df.isnull().sum().sum()
                    st.metric("Missing Values", missing_after)
                
                # Show feature types
                st.subheader("📋 Feature Summary")
                feature_info = pd.DataFrame({
                    'Feature': X_encoded.columns,
                    'Data Type': X_encoded.dtypes,
                    'Non-Null Count': X_encoded.count(),
                    'Unique Values': X_encoded.nunique()
                })
                st.dataframe(feature_info, use_container_width=True)
                
                # Train model button
                st.markdown("---")
                if st.button("🎯 Train Model", type="primary"):
                    st.info("👉 Please go to the 'Model Training & Evaluation' tab to train your model.")
                
            except Exception as e:
                st.error(f"❌ Error during preprocessing: {str(e)}")
                st.write("Error details:", e)
    
    # Show current preprocessing status
    if st.session_state.processed_data is not None:
        st.success("✅ Data is already preprocessed and ready for training!")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Processed Features", len(st.session_state.feature_names))
        with col2:
            st.metric("Target Column", st.session_state.target_column)

def model_training_tab():
    """Tab 3: Model Training & Evaluation"""
    st.markdown('<div class="tab-header">🤖 Model Training & Evaluation</div>', unsafe_allow_html=True)
    
    if st.session_state.processed_data is None:
        st.warning("⚠️ Please preprocess your data first in the 'Data Preprocessing' tab.")
        return
    
    df = st.session_state.processed_data
    target_col = st.session_state.target_column
    
    st.subheader("📊 Training Data Status")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Samples", f"{len(df):,}")
    with col2:
        st.metric("Features", len(st.session_state.feature_names))
    with col3:
        churn_rate = df[target_col].mean()
        st.metric("Churn Rate", f"{churn_rate:.1%}")
    
    # Training configuration
    st.subheader("⚙️ Training Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)
        random_state = st.number_input("Random State", value=42, min_value=0)
    
    with col2:
        models_to_train = st.multiselect(
            "Select Models to Train:",
            ["Logistic Regression", "Random Forest", "XGBoost"],
            default=["Logistic Regression", "Random Forest", "XGBoost" if XGBOOST_AVAILABLE else None]
        )
        
        if "XGBoost" in models_to_train and not XGBOOST_AVAILABLE:
            st.warning("⚠️ XGBoost not available. Install with: pip install xgboost")
            models_to_train.remove("XGBoost")
    
    # Training button
    if st.button("🚀 Train Models", type="primary"):
        if not models_to_train:
            st.error("❌ Please select at least one model to train.")
            return
        
        with st.spinner("Training models... This may take a few minutes."):
            try:
                # Prepare data
                X = df.drop(columns=[target_col])
                y = df[target_col]
                
                # Train-test split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state, stratify=y
                )
                
                # Initialize models
                models = {}
                if "Logistic Regression" in models_to_train:
                    models["Logistic Regression"] = LogisticRegression(
                        random_state=random_state, max_iter=1000, class_weight='balanced'
                    )
                
                if "Random Forest" in models_to_train:
                    models["Random Forest"] = RandomForestClassifier(
                        n_estimators=100, random_state=random_state, class_weight='balanced'
                    )
                
                if "XGBoost" in models_to_train and XGBOOST_AVAILABLE:
                    models["XGBoost"] = xgb.XGBClassifier(
                        random_state=random_state, eval_metric='logloss'
                    )
                
                # Train and evaluate models
                results = {}
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, (name, model) in enumerate(models.items()):
                    status_text.text(f"Training {name}...")
                    
                    # Train model
                    model.fit(X_train, y_train)
                    
                    # Make predictions
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                    
                    # Calculate metrics
                    metrics = {
                        'accuracy': accuracy_score(y_test, y_pred),
                        'precision': precision_score(y_test, y_pred),
                        'recall': recall_score(y_test, y_pred),
                        'f1_score': f1_score(y_test, y_pred),
                        'roc_auc': roc_auc_score(y_test, y_pred_proba)
                    }
                    
                    # Store results
                    results[name] = {
                        'model': model,
                        'metrics': metrics,
                        'predictions': y_pred,
                        'probabilities': y_pred_proba,
                        'confusion_matrix': confusion_matrix(y_test, y_pred)
                    }
                    
                    progress_bar.progress((i + 1) / len(models))
                
                status_text.text("Training completed!")
                
                # Select best model
                best_model_name = max(results.keys(), key=lambda x: results[x]['metrics']['roc_auc'])
                best_model = results[best_model_name]['model']
                
                # Save best model
                os.makedirs('../models', exist_ok=True)
                model_data = {
                    'model': best_model,
                    'model_name': best_model_name,
                    'feature_names': st.session_state.feature_names,
                    'scaler': st.session_state.scaler,
                    'encoders': st.session_state.encoders,
                    'metrics': results[best_model_name]['metrics'],
                    'dataset_type': st.session_state.dataset_type,
                    'target_column': target_col,
                    'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                try:
                    with open('../models/churn_model.pkl', 'wb') as f:
                        pickle.dump(model_data, f)
                    model_saved = True
                except:
                    try:
                        with open('models/churn_model.pkl', 'wb') as f:
                            pickle.dump(model_data, f)
                        model_saved = True
                    except:
                        model_saved = False
                
                st.session_state.model = best_model
                st.session_state.model_results = results
                
                st.success(f"✅ Training completed! Best model: {best_model_name}")
                if model_saved:
                    st.success("✅ Model saved successfully!")
                else:
                    st.warning("⚠️ Could not save model to disk.")
                
                # Display results
                st.subheader("📈 Model Comparison")
                
                # Create comparison dataframe
                comparison_data = []
                for name, result in results.items():
                    row = {'Model': name}
                    row.update(result['metrics'])
                    comparison_data.append(row)
                
                comparison_df = pd.DataFrame(comparison_data)
                comparison_df = comparison_df.sort_values('roc_auc', ascending=False)
                
                # Display table
                st.dataframe(comparison_df.round(4), use_container_width=True)
                
                # Create comparison chart
                metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
                fig = go.Figure()
                
                for metric in metrics_to_plot:
                    fig.add_trace(go.Bar(
                        name=metric.replace('_', ' ').title(),
                        x=comparison_df['Model'],
                        y=comparison_df[metric],
                        text=comparison_df[metric].round(3),
                        textposition='auto'
                    ))
                
                fig.update_layout(
                    title="Model Performance Comparison",
                    xaxis_title="Models",
                    yaxis_title="Score",
                    barmode='group',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show detailed results for best model
                st.subheader(f"🏆 Best Model Details: {best_model_name}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Confusion Matrix
                    st.write("**Confusion Matrix**")
                    cm = results[best_model_name]['confusion_matrix']
                    
                    fig, ax = plt.subplots(figsize=(6, 4))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                               xticklabels=['No Churn', 'Churn'],
                               yticklabels=['No Churn', 'Churn'])
                    ax.set_title('Confusion Matrix')
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('Actual')
                    
                    st.pyplot(fig)
                    plt.close()
                
                with col2:
                    # ROC Curve
                    st.write("**ROC Curve**")
                    
                    fig, ax = plt.subplots(figsize=(6, 4))
                    
                    for name, result in results.items():
                        fpr, tpr, _ = roc_curve(y_test, result['probabilities'])
                        auc_score = result['metrics']['roc_auc']
                        ax.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.3f})')
                    
                    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
                    ax.set_xlabel('False Positive Rate')
                    ax.set_ylabel('True Positive Rate')
                    ax.set_title('ROC Curves')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    st.pyplot(fig)
                    plt.close()
                
                # Feature importance (if available)
                if hasattr(best_model, 'feature_importances_'):
                    st.subheader("🎯 Feature Importance")
                    
                    importance_df = pd.DataFrame({
                        'Feature': st.session_state.feature_names,
                        'Importance': best_model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    fig = px.bar(
                        importance_df.head(15),
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title="Top 15 Feature Importance"
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Generate report button
                st.markdown("---")
                if st.button("📄 Generate Report"):
                    generate_simple_report(results, best_model_name, st.session_state.dataset_type)
                
            except Exception as e:
                st.error(f"❌ Error during training: {str(e)}")
                st.write("Error details:", e)
    
    # Show current model status
    if st.session_state.model is not None:
        st.info("✅ Model is trained and ready for predictions!")
        if st.button("🔄 Re-train Model"):
            st.session_state.model = None
            st.session_state.model_results = None
            st.rerun()

def predict_churn_tab():
    """Tab 4: Predict Churn"""
    st.markdown('<div class="tab-header">🔮 Predict Customer Churn</div>', unsafe_allow_html=True)
    
    # Try to load model
    model_data = None
    
    # Check session state first
    if st.session_state.model is not None:
        model_data = {
            'model': st.session_state.model,
            'feature_names': st.session_state.feature_names,
            'scaler': st.session_state.scaler,
            'encoders': st.session_state.encoders,
            'dataset_type': st.session_state.dataset_type,
            'target_column': st.session_state.target_column
        }
    else:
        # Try to load from file
        model_paths = ['../models/churn_model.pkl', 'models/churn_model.pkl']
        for path in model_paths:
            if os.path.exists(path):
                try:
                    with open(path, 'rb') as f:
                        model_data = pickle.load(f)
                    break
                except:
                    continue
    
    if model_data is None:
        st.warning("⚠️ No trained model found. Please train a model first in the 'Model Training & Evaluation' tab.")
        return
    
    model = model_data['model']
    feature_names = model_data.get('feature_names', [])
    scaler = model_data.get('scaler')
    encoders = model_data.get('encoders', {})
    dataset_type = model_data.get('dataset_type', 'telco')
    
    st.success(f"✅ Model loaded successfully! Dataset type: {dataset_type.upper()}")
    
    # Display model info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model Type", model_data.get('model_name', 'Unknown'))
    with col2:
        st.metric("Features", len(feature_names))
    with col3:
        if 'metrics' in model_data:
            accuracy = model_data['metrics'].get('roc_auc', 0)
            st.metric("ROC-AUC", f"{accuracy:.3f}")
    
    st.markdown("---")
    st.subheader("📋 Customer Information")
    st.write("Fill in the customer details below to predict churn probability:")
    
    # Create input form based on dataset type
    if dataset_type == 'telco':
        create_telco_input_form(model, feature_names, scaler, encoders)
    elif dataset_type == 'bank':
        create_bank_input_form(model, feature_names, scaler, encoders)
    else:
        st.error("❌ Unknown dataset type. Cannot create input form.")

def create_telco_input_form(model, feature_names, scaler, encoders):
    """Create input form for Telco dataset"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Demographics**")
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Has Partner", ["No", "Yes"])
        dependents = st.selectbox("Has Dependents", ["No", "Yes"])
        
        st.write("**Account Information**")
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=float(monthly_charges * tenure))
    
    with col2:
        st.write("**Services**")
        phone_service = st.selectbox("Phone Service", ["No", "Yes"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        
        st.write("**Contract & Billing**")
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
        payment_method = st.selectbox("Payment Method", 
                                    ["Electronic check", "Mailed check", 
                                     "Bank transfer (automatic)", "Credit card (automatic)"])
    
    if st.button("🔮 Predict Churn", type="primary"):
        # Create input dataframe
        input_data = {
            'gender': gender,
            'SeniorCitizen': senior_citizen,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'PhoneService': phone_service,
            'MultipleLines': multiple_lines,
            'InternetService': internet_service,
            'OnlineSecurity': online_security,
            'OnlineBackup': online_backup,
            'DeviceProtection': device_protection,
            'TechSupport': tech_support,
            'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies,
            'Contract': contract,
            'PaperlessBilling': paperless_billing,
            'PaymentMethod': payment_method,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges
        }
        
        make_prediction(input_data, model, feature_names, scaler, encoders, 'telco')

def create_bank_input_form(model, feature_names, scaler, encoders):
    """Create input form for Bank dataset"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Demographics**")
        geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.slider("Age", 18, 95, 35)
        
        st.write("**Financial Information**")
        credit_score = st.slider("Credit Score", 350, 850, 650)
        balance = st.number_input("Account Balance ($)", min_value=0.0, value=50000.0)
        estimated_salary = st.number_input("Estimated Salary ($)", min_value=0.0, value=75000.0)
    
    with col2:
        st.write("**Banking Details**")
        tenure = st.slider("Tenure (years)", 0, 10, 5)
        num_of_products = st.selectbox("Number of Products", [1, 2, 3, 4])
        has_cr_card = st.selectbox("Has Credit Card", [0, 1])
        is_active_member = st.selectbox("Is Active Member", [0, 1])
    
    if st.button("🔮 Predict Churn", type="primary"):
        # Create input dataframe
        input_data = {
            'Geography': geography,
            'Gender': gender,
            'Age': age,
            'Tenure': tenure,
            'Balance': balance,
            'NumOfProducts': num_of_products,
            'HasCrCard': has_cr_card,
            'IsActiveMember': is_active_member,
            'EstimatedSalary': estimated_salary,
            'CreditScore': credit_score
        }
        
        make_prediction(input_data, model, feature_names, scaler, encoders, 'bank')

def make_prediction(input_data, model, feature_names, scaler, encoders, dataset_type):
    """Make churn prediction"""
    try:
        with st.spinner("Making prediction..."):
            # Convert to DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Apply same preprocessing as training data
            processed_input = preprocess_input(input_df, feature_names, scaler, encoders, dataset_type)
            
            # Make prediction
            prediction = model.predict(processed_input)[0]
            probability = model.predict_proba(processed_input)[0][1]
            
            # Display results
            st.markdown("---")
            st.subheader("🎯 Prediction Results")
            
            if prediction == 1:
                st.markdown(f'''
                <div class="prediction-result high-risk">
                    ⚠️ HIGH CHURN RISK<br>
                    Probability: {probability:.1%}
                </div>
                ''', unsafe_allow_html=True)
                
                st.error("😨 **High Risk Customer!** This customer has a high probability of churning.")
                
                # Recommendations for high-risk customers
                st.subheader("💡 Recommended Actions")
                if dataset_type == 'telco':
                    recommendations = [
                        "🎁 Offer contract upgrade incentives",
                        "💳 Promote automatic payment methods",
                        "📢 Proactive customer service outreach",
                        "🎆 Provide loyalty rewards or discounts",
                        "📞 Schedule retention call within 48 hours"
                    ]
                else:
                    recommendations = [
                        "🎯 Cross-sell additional banking products",
                        "📱 Increase digital engagement",
                        "🏆 Offer VIP banking status",
                        "💰 Provide financial advisory services",
                        "🔄 Schedule relationship manager meeting"
                    ]
                
                for rec in recommendations:
                    st.write(f"• {rec}")
                    
            else:
                st.markdown(f'''
                <div class="prediction-result low-risk">
                    ✅ LOW CHURN RISK<br>
                    Probability: {probability:.1%}
                </div>
                ''', unsafe_allow_html=True)
                
                st.success("😊 **Low Risk Customer!** This customer has a low probability of churning.")
                
                # Recommendations for low-risk customers
                st.subheader("💡 Recommended Actions")
                maintenance_actions = [
                    "🌟 Maintain current service quality",
                    "📧 Send satisfaction surveys",
                    "🎆 Consider upselling opportunities",
                    "👥 Include in referral programs",
                    "📊 Monitor for any changes in behavior"
                ]
                
                for action in maintenance_actions:
                    st.write(f"• {action}")
            
            # Show confidence and key factors
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Churn Probability", f"{probability:.1%}")
                st.metric("Confidence Level", "High" if abs(probability - 0.5) > 0.2 else "Medium")
            
            with col2:
                st.write("**Key Risk Factors:**")
                if dataset_type == 'telco':
                    risk_factors = []
                    if input_data.get('Contract') == 'Month-to-month':
                        risk_factors.append("📅 Month-to-month contract")
                    if input_data.get('PaymentMethod') == 'Electronic check':
                        risk_factors.append("💳 Electronic check payment")
                    if input_data.get('tenure', 0) < 12:
                        risk_factors.append("⏰ Short tenure")
                    if input_data.get('MonthlyCharges', 0) > 70:
                        risk_factors.append("💰 High monthly charges")
                else:
                    risk_factors = []
                    if input_data.get('NumOfProducts') == 1:
                        risk_factors.append("🎯 Single product")
                    if input_data.get('IsActiveMember') == 0:
                        risk_factors.append("😴 Inactive member")
                    if input_data.get('Age', 0) > 50:
                        risk_factors.append("🎂 Senior customer")
                
                if risk_factors:
                    for factor in risk_factors[:3]:  # Show top 3
                        st.write(f"• {factor}")
                else:
                    st.write("• ✅ No major risk factors")
    
    except Exception as e:
        st.error(f"❌ Error making prediction: {str(e)}")
        st.write("Please check your input data and try again.")

def preprocess_input(input_df, feature_names, scaler, encoders, dataset_type):
    """Preprocess input data to match training format"""
    processed_df = input_df.copy()
    
    # Handle categorical encoding using stored encoders first
    for col, encoder in encoders.items():
        if col in processed_df.columns:
            try:
                processed_df[col] = encoder.transform(processed_df[col].astype(str))
            except ValueError:
                # Handle unseen categories by using the most frequent class
                processed_df[col] = encoder.transform([encoder.classes_[0]])[0]
    
    # Apply one-hot encoding for remaining categorical columns
    categorical_cols = processed_df.select_dtypes(include=['object']).columns.tolist()
    if len(categorical_cols) > 0:
        processed_df = pd.get_dummies(processed_df, columns=categorical_cols, drop_first=True)
    
    # Ensure ALL training features are present with correct values
    for feature in feature_names:
        if feature not in processed_df.columns:
            processed_df[feature] = 0  # Default value for missing features
    
    # Remove any extra columns not in training features
    extra_cols = [col for col in processed_df.columns if col not in feature_names]
    if extra_cols:
        processed_df = processed_df.drop(columns=extra_cols)
    
    # Reorder columns to match training order exactly
    processed_df = processed_df[feature_names]
    
    # Apply scaling if scaler exists
    if scaler is not None:
        # Only scale numeric columns that exist in the original training data
        try:
            processed_df = pd.DataFrame(
                scaler.transform(processed_df),
                columns=processed_df.columns,
                index=processed_df.index
            )
        except Exception as e:
            # If scaling fails, continue without scaling
            print(f"Warning: Scaling failed: {e}")
    
    return processed_df

def generate_simple_report(results, best_model_name, dataset_type):
    """Generate a comprehensive report with charts and download options"""
    try:
        st.subheader("📊 Comprehensive Model Report")
        
        # Create comprehensive text report
        report_text = f"""CHURN PREDICTION MODEL REPORT
{'=' * 50}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Dataset Type: {dataset_type.upper()}
Best Performing Model: {best_model_name}

EXECUTIVE SUMMARY
{'-' * 20}
This report presents the results of training multiple machine learning models
for customer churn prediction. The best performing model was {best_model_name}
with an ROC-AUC score of {results[best_model_name]['metrics']['roc_auc']:.4f}.

BEST MODEL PERFORMANCE
{'-' * 25}
"""
        
        best_metrics = results[best_model_name]['metrics']
        for metric, value in best_metrics.items():
            metric_name = metric.replace('_', ' ').title()
            report_text += f"{metric_name:<15}: {value:.4f} ({value:.1%} if applicable)\n"
        
        report_text += f"\nMODEL COMPARISON\n{'-' * 20}\n"
        
        # Create comparison table
        comparison_data = []
        for name, result in results.items():
            comparison_data.append({
                'Model': name,
                'Accuracy': f"{result['metrics']['accuracy']:.4f}",
                'Precision': f"{result['metrics']['precision']:.4f}", 
                'Recall': f"{result['metrics']['recall']:.4f}",
                'F1-Score': f"{result['metrics']['f1_score']:.4f}",
                'ROC-AUC': f"{result['metrics']['roc_auc']:.4f}"
            })
        
        # Add formatted table to report
        report_text += f"{'Model':<20} {'Accuracy':<10} {'Precision':<11} {'Recall':<10} {'F1-Score':<10} {'ROC-AUC':<10}\n"
        report_text += f"{'-' * 80}\n"
        for data in comparison_data:
            report_text += f"{data['Model']:<20} {data['Accuracy']:<10} {data['Precision']:<11} {data['Recall']:<10} {data['F1-Score']:<10} {data['ROC-AUC']:<10}\n"
        
        report_text += f"\nDETAILED ANALYSIS\n{'-' * 20}\n"
        for name, result in results.items():
            report_text += f"\n{name.upper()}:\n"
            for metric, value in result['metrics'].items():
                interpretation = ""
                if metric == 'accuracy':
                    interpretation = "(Overall correctness)"
                elif metric == 'precision':
                    interpretation = "(Accuracy of churn predictions)"
                elif metric == 'recall':
                    interpretation = "(Coverage of actual churners)"
                elif metric == 'f1_score':
                    interpretation = "(Balance of precision and recall)"
                elif metric == 'roc_auc':
                    interpretation = "(Overall discriminative ability)"
                
                report_text += f"  {metric.replace('_', ' ').title():<15}: {value:.4f} {interpretation}\n"
        
        report_text += f"\nRECOMMendations\n{'-' * 15}\n"
        
        best_auc = results[best_model_name]['metrics']['roc_auc']
        if best_auc >= 0.85:
            report_text += "• EXCELLENT: Model shows excellent discriminative ability (AUC ≥ 0.85)\n"
            report_text += "• RECOMMENDATION: Deploy model for production use\n"
        elif best_auc >= 0.75:
            report_text += "• GOOD: Model shows good discriminative ability (AUC ≥ 0.75)\n"
            report_text += "• RECOMMENDATION: Consider deployment with monitoring\n"
        elif best_auc >= 0.65:
            report_text += "• FAIR: Model shows fair discriminative ability (AUC ≥ 0.65)\n"
            report_text += "• RECOMMENDATION: Improve feature engineering or collect more data\n"
        else:
            report_text += "• POOR: Model shows poor discriminative ability (AUC < 0.65)\n"
            report_text += "• RECOMMENDATION: Revisit data quality and feature selection\n"
        
        best_precision = results[best_model_name]['metrics']['precision']
        best_recall = results[best_model_name]['metrics']['recall']
        
        if best_precision > best_recall:
            report_text += "• Model is conservative - high precision, lower recall\n"
            report_text += "• Good for minimizing false alarms but may miss some churners\n"
        elif best_recall > best_precision:
            report_text += "• Model is aggressive - high recall, lower precision\n"
            report_text += "• Good for catching churners but may have false alarms\n"
        else:
            report_text += "• Model is balanced between precision and recall\n"
        
        report_text += f"\nMODEL DEPLOYMENT NOTES\n{'-' * 25}\n"
        report_text += f"• Model Type: {best_model_name}\n"
        report_text += f"• Training Date: {datetime.now().strftime('%Y-%m-%d')}\n"
        report_text += f"• Dataset: {dataset_type.upper()}\n"
        report_text += f"• Recommended Review Frequency: Monthly\n"
        report_text += f"• Model Monitoring: Track prediction drift and performance\n\n"
        
        report_text += f"END OF REPORT\n{'=' * 50}"
        
        # Display the report in Streamlit
        with st.expander("📄 View Full Report", expanded=True):
            st.text_area("Model Report", report_text, height=500)
        
        # Create downloadable files
        col1, col2 = st.columns(2)
        
        with col1:
            # Text report download
            st.download_button(
                label="📄 Download Text Report",
                data=report_text,
                file_name=f"churn_report_{dataset_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        with col2:
            # CSV report download
            comparison_df = pd.DataFrame(comparison_data)
            csv_data = comparison_df.to_csv(index=False)
            st.download_button(
                label="📊 Download CSV Data",
                data=csv_data,
                file_name=f"churn_metrics_{dataset_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        # Interactive comparison chart
        st.subheader("📈 Interactive Model Comparison")
        
        comparison_df_melted = pd.melt(
            comparison_df, 
            id_vars=['Model'], 
            value_vars=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
            var_name='Metric', 
            value_name='Score'
        )
        comparison_df_melted['Score'] = comparison_df_melted['Score'].astype(float)
        
        fig = px.bar(
            comparison_df_melted,
            x='Model',
            y='Score', 
            color='Metric',
            title="Model Performance Comparison",
            barmode='group',
            height=500
        )
        fig.update_layout(
            xaxis_title="Models",
            yaxis_title="Score",
            legend_title="Metrics"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Success message
        st.success("✅ Report generated successfully! You can download the report files above.")
        
        # Additional insights
        with st.expander("🔍 Additional Insights"):
            st.write("**Key Takeaways:**")
            st.write(f"• Best performing model: **{best_model_name}**")
            st.write(f"• Highest ROC-AUC score: **{best_auc:.4f}**")
            st.write(f"• Model precision: **{best_precision:.4f}** (accuracy of churn predictions)")
            st.write(f"• Model recall: **{best_recall:.4f}** (coverage of actual churners)")
            
            if len(results) > 1:
                sorted_models = sorted(results.items(), key=lambda x: x[1]['metrics']['roc_auc'], reverse=True)
                st.write(f"• Performance ranking: {' > '.join([name for name, _ in sorted_models])}")
        
    except Exception as e:
        st.error(f"❌ Error generating report: {str(e)}")
        st.write("Please try again or contact support if the issue persists.")

def main():
    """Main Streamlit app"""
    # Header
    st.markdown('<h1 class="main-header">Churn Prediction System</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem; color: #666;">
            End-to-end ML workflow: Upload 툙2 Preprocess 툙2 Train 툙2 Evaluate 툙2 Predict
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📁 Data Upload & Preview",
        "🔧 Data Preprocessing", 
        "🤖 Model Training & Evaluation",
        "🔮 Predict Churn"
    ])
    
    with tab1:
        data_upload_tab()
    
    with tab2:
        data_preprocessing_tab()
    
    with tab3:
        model_training_tab()
    
    with tab4:
        predict_churn_tab()
    
    # Sidebar with info
    with st.sidebar:
        st.markdown("### 📊 System Status")
        
        # Data status
        if st.session_state.data is not None:
            st.success("✅ Data loaded")
            st.write(f"Dataset: {st.session_state.dataset_type.upper()}")
            st.write(f"Rows: {len(st.session_state.data):,}")
        else:
            st.warning("⚠️ No data loaded")
        
        # Preprocessing status
        if st.session_state.processed_data is not None:
            st.success("✅ Data preprocessed")
            st.write(f"Features: {len(st.session_state.feature_names)}")
        else:
            st.info("🔄 Data not preprocessed")
        
        # Model status
        if st.session_state.model is not None:
            st.success("✅ Model trained")
        else:
            st.info("🎯 No model trained")
        
        st.markdown("---")
        st.markdown("### 🛠️ Quick Actions")
        
        if st.button("🔄 Reset All"):
            for key in st.session_state.keys():
                del st.session_state[key]
            st.rerun()
        
        if st.button("📊 Clear Cache"):
            st.cache_data.clear()
            st.success("Cache cleared!")
        
        st.markdown("---")
        st.markdown("""
        ### 📚 Instructions
        
        1. **Upload Data**: Choose dataset or upload CSV
        2. **Preprocess**: Clean and prepare data
        3. **Train Models**: Compare ML algorithms
        4. **Predict**: Test with new customer data
        
        ### 📈 Supported Datasets
        - Telco Customer Churn
        - Bank Customer Churn
        - Custom CSV files
        """)

if __name__ == "__main__":
    main()