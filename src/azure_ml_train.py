"""
Azure ML training script that runs on Azure ML compute
This script is executed remotely on Azure ML compute clusters
"""
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
# import mlflow
# import mlflow.sklearn  # Commented out to avoid sync conflicts with Azure ML
from azureml.core import Run, Dataset, Model
import argparse
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data_from_azure_ml(run, dataset_name=None):
    """Load data from Azure ML Dataset or default dataset"""
    try:
        if dataset_name:
            # Load from registered dataset
            dataset = Dataset.get_by_name(run.experiment.workspace, dataset_name)
            df = dataset.to_pandas_dataframe()
            logger.info(f"Loaded dataset: {dataset_name}")
        else:
            # Create sample dataset for demo purposes
            logger.info("Creating sample dataset...")
            df = pd.DataFrame({
                'feature1': [1.2, 2.3, 3.1, 4.5, 5.0, 1.8, 2.9, 3.7, 4.2, 5.5, 0.9, 6.1, 2.4, 3.8, 4.7, 1.5, 5.2, 2.6, 3.3, 4.8],
                'feature2': [7.8, 3.4, 8.9, 11.2, 4.1, 6.7, 9.3, 5.5, 12.1, 2.8, 8.2, 7.4, 10.6, 4.9, 6.3, 9.8, 3.7, 11.5, 5.1, 8.7],
                'categorical': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'B', 'A', 'C', 'B', 'A', 'C', 'B', 'A', 'C', 'B', 'A', 'C', 'B'],
                'target': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
            })
        
        return df
    except Exception as e:
        logger.error(f"Failed to load data: {str(e)}")
        raise

def preprocess_data(df, target_column='target'):
    """Preprocess the data"""
    logger.info("Starting data preprocessing...")
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle missing values for numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if col != target_column:
            df[col] = df[col].fillna(df[col].median())
    
    # Handle missing values for categorical columns
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        if col != target_column:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
    
    # Feature engineering - create squared features
    for col in numeric_columns:
        if col != target_column:
            df[f'{col}_squared'] = df[col] ** 2
    
    # One-hot encode categorical variables (excluding target)
    categorical_to_encode = [col for col in categorical_columns if col != target_column]
    if categorical_to_encode:
        df = pd.get_dummies(df, columns=categorical_to_encode, drop_first=True)
    
    logger.info(f"Preprocessing completed. Final shape: {df.shape}")
    return df

def split_data(df, target_column, test_size=0.2, random_state=42):
    """Split data into train and test sets"""
    from sklearn.model_selection import train_test_split
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Check if we can use stratification
    min_class_count = y.value_counts().min()
    use_stratify = min_class_count >= 2
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=y if use_stratify else None
    )
    
    logger.info(f"Data split - Train: {len(X_train)}, Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, model_type='random_forest', **kwargs):
    """Train the model"""
    logger.info(f"Training {model_type} model...")
    
    if model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=kwargs.get('n_estimators', 100),
            max_depth=kwargs.get('max_depth', 10),
            random_state=42
        )
    elif model_type == 'logistic_regression':
        model = LogisticRegression(
            C=kwargs.get('C', 1.0),
            max_iter=kwargs.get('max_iter', 1000),
            random_state=42
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    model.fit(X_train, y_train)
    logger.info("Model training completed")
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model"""
    logger.info("Evaluating model...")
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1_score': f1_score(y_test, y_pred, average='weighted')
    }
    
    if y_pred_proba is not None and len(np.unique(y_test)) == 2:
        metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
    
    logger.info(f"Model metrics: {metrics}")
    return metrics

def register_model_in_azure_ml(run, model, model_name, metrics, model_path=None):
    """Register model in Azure ML Model Registry"""
    logger.info(f"Registering model: {model_name}")
    
    # Save model locally first
    if model_path is None:
        model_path = f'outputs/{model_name}.pkl'
    
    os.makedirs('outputs', exist_ok=True)
    joblib.dump(model, model_path)
    logger.info(f"Model saved to: {model_path}")
    
    # Log model artifact to Azure ML run (skip MLflow to avoid conflicts)
    logger.info("Logging model artifact to Azure ML run")
    
    # Register model in Azure ML (remove run_id parameter - deprecated)
    registered_model = Model.register(
        workspace=run.experiment.workspace,
        model_path=model_path,
        model_name=model_name,
        description=f"Model trained on Azure ML compute. Accuracy: {metrics.get('accuracy', 'N/A'):.3f}",
        tags={
            'framework': 'scikit-learn',
            'type': 'classification',
            'training_compute': 'azure_ml',
            'accuracy': f"{metrics.get('accuracy', 0):.3f}",
            'f1_score': f"{metrics.get('f1_score', 0):.3f}",
            'run_id': run.id
        }
    )
    
    logger.info(f"Model registered: {registered_model.name} v{registered_model.version}")
    logger.info(f"Model URI: {registered_model.url}")
    
    return registered_model

def main():
    parser = argparse.ArgumentParser(description="Train model on Azure ML")
    parser.add_argument("--dataset-name", type=str, help="Name of registered dataset")
    parser.add_argument("--model-type", type=str, default="random_forest", choices=['random_forest', 'logistic_regression'])
    parser.add_argument("--model-name", type=str, default="trained-model", help="Name for model registration")
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--register-model", type=str, default="true", help="Whether to register model")
    
    args = parser.parse_args()
    
    # Get Azure ML run context
    run = Run.get_context()
    
    # Note: MLflow tracking is automatically enabled by Azure ML
    # We'll use only Azure ML run context to avoid sync conflicts
    
    try:
        # Load data
        df = load_data_from_azure_ml(run, args.dataset_name)
        
        # Preprocess data
        df_processed = preprocess_data(df)
        
        # Split data
        X_train, X_test, y_train, y_test = split_data(df_processed, 'target')
        
        # Log data info
        run.log("train_samples", len(X_train))
        run.log("test_samples", len(X_test))
        run.log("n_features", X_train.shape[1])
        
        # Train model
        model_params = {
            'n_estimators': args.n_estimators,
            'max_depth': args.max_depth,
            'C': args.C,
            'max_iter': args.max_iter
        }
        
        model = train_model(X_train, y_train, args.model_type, **model_params)
        
        # Log parameters (use only Azure ML run context to avoid sync issues)
        for param_name, param_value in model_params.items():
            run.log(param_name, param_value)
        
        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test)
        
        # Log metrics (use only Azure ML run context to avoid sync issues)
        for metric_name, metric_value in metrics.items():
            run.log(metric_name, metric_value)
        
        # Save metrics to outputs folder
        os.makedirs('outputs', exist_ok=True)
        with open('outputs/metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Register model if requested
        if args.register_model.lower() == 'true':
            registered_model = register_model_in_azure_ml(run, model, args.model_name, metrics)
            run.log("registered_model_name", registered_model.name)
            run.log("registered_model_version", registered_model.version)
            run.log("registered_model_id", registered_model.id)
        
        # Log training completion
        run.log("training_status", "completed")
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        run.log("training_status", "failed")
        run.log("error_message", str(e))
        raise
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        run.log("training_status", "failed")
        run.log("error_message", str(e))
        raise

if __name__ == "__main__":
    main()