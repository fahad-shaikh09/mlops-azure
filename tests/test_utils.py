"""Test utilities for model validation"""
import pandas as pd
import numpy as np
import os
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from data_preprocessing import DataPreprocessor
except ImportError:
    DataPreprocessor = None

def create_sample_data():
    """Create sample data that matches the preprocessing pipeline"""
    return pd.DataFrame({
        'feature1': [1.0, 2.0, 3.0, 4.0, 5.0, 1.5, 2.5, 3.5, 4.5, 5.5, 1.2, 2.2, 3.2, 4.2, 5.2],
        'feature2': [7.0, 3.0, 8.0, 11.0, 4.0, 6.5, 3.5, 8.5, 11.5, 4.5, 6.2, 3.2, 8.2, 11.2, 4.2],
        'categorical': ['A', 'B', 'A', 'C', 'B', 'A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A'],
        'target': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    })

def get_processed_test_data(n_samples=2):
    """Get processed test data that matches the model's expected input format"""
    # First try to use actual processed data files
    test_data_path = 'data/processed/X_test.csv'
    train_data_path = 'data/processed/X_train.csv'
    
    try:
        if os.path.exists(test_data_path):
            df = pd.read_csv(test_data_path)
            if len(df) >= n_samples:
                return df.iloc[:n_samples].values
            elif os.path.exists(train_data_path):
                # If test data is too small, use train data
                df = pd.read_csv(train_data_path)
                return df.iloc[:n_samples].values
        elif os.path.exists(train_data_path):
            # Use train data if test data doesn't exist
            df = pd.read_csv(train_data_path)
            return df.iloc[:n_samples].values
    except Exception as e:
        print(f"Warning: Could not load processed data files: {e}")
    
    # Fallback: generate data using preprocessing pipeline
    if DataPreprocessor is None:
        # Final fallback: return random data with expected shape (25 features based on actual data)
        return np.random.rand(n_samples, 25)
    
    try:
        # Create and process sample data
        preprocessor = DataPreprocessor()
        df = create_sample_data()
        
        # Apply same preprocessing as training
        df_clean = preprocessor.clean_data(df)
        df_features = preprocessor.feature_engineering(df_clean)
        df_encoded = preprocessor.encode_categorical(df_features, target_column='target')
        
        # Split and get features only
        X_train, X_test, y_train, y_test = preprocessor.split_data(df_encoded, 'target')
        
        # Return first n_samples from test set, or fallback to train set if test is too small
        if len(X_test) >= n_samples:
            data = X_test.iloc[:n_samples].values if hasattr(X_test, 'iloc') else X_test[:n_samples]
        else:
            data = X_train.iloc[:n_samples].values if hasattr(X_train, 'iloc') else X_train[:n_samples]
        
        return data
        
    except Exception as e:
        # Final fallback: return random data with expected shape
        print(f"Warning: Using fallback test data due to preprocessing error: {e}")
        return np.random.rand(n_samples, 25)  # Updated to 25 features based on actual data

def get_test_data_for_api(n_samples=2):
    """Get test data formatted for API calls"""
    processed_data = get_processed_test_data(n_samples)
    return {
        "data": processed_data.tolist()
    }