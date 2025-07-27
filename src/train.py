import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
try:
    import mlflow
    import mlflow.sklearn
except ImportError:  # pragma: no cover - optional dependency
    mlflow = None

try:
    from azureml.core import Run, Workspace, Dataset, Experiment
    from azureml.core.model import Model
except ImportError:  # pragma: no cover - optional dependency
    Run = None
    Workspace = None
    Dataset = None
    Experiment = None
    Model = None
import argparse
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, workspace=None):
        self.workspace = workspace
        self.run = Run.get_context() if Run else None
        
    def load_data(self, data_path):
        """Load processed training data"""
        X_train = pd.read_csv(os.path.join(data_path, 'X_train.csv'))
        X_test = pd.read_csv(os.path.join(data_path, 'X_test.csv'))
        y_train = pd.read_csv(os.path.join(data_path, 'y_train.csv')).values.ravel()
        y_test = pd.read_csv(os.path.join(data_path, 'y_test.csv')).values.ravel()
        
        logger.info(f"Loaded training data - Train: {X_train.shape}, Test: {X_test.shape}")
        return X_train, X_test, y_train, y_test
    
    def train_model(self, X_train, y_train, model_type='random_forest', **kwargs):
        """Train the model"""
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
        
        logger.info(f"Training {model_type} model...")
        model.fit(X_train, y_train)
        logger.info("Model training completed")
        
        return model
    
    def evaluate_model(self, model, X_test, y_test):
        """Evaluate the trained model"""
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
        
        logger.info(f"Model evaluation metrics: {metrics}")
        return metrics
    
    def log_metrics(self, metrics):
        """Log metrics to MLflow and Azure ML"""
        for metric_name, metric_value in metrics.items():
            if mlflow:
                mlflow.log_metric(metric_name, metric_value)
            if self.run:
                self.run.log(metric_name, metric_value)
    
    def save_model(self, model, model_name, output_dir='outputs'):
        """Save the trained model"""
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, f'{model_name}.pkl')
        
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Log model to MLflow
        if mlflow:
            mlflow.sklearn.log_model(model, model_name)
        
        return model_path
    
    def register_model(self, model_path, model_name, model_version=None):
        """Register model in Azure ML"""
        if self.workspace and Model:
            try:
                model = Model.register(
                    workspace=self.workspace,
                    model_path=model_path,
                    model_name=model_name,
                    model_version=model_version,
                    description=f"Model trained using MLOps pipeline",
                    tags={'framework': 'scikit-learn', 'type': 'classification'}
                )
                logger.info(f"Model registered: {model.name} v{model.version}")
                return model
            except Exception as e:
                logger.error(f"Model registration failed: {str(e)}")
                return None
        else:
            logger.warning("Workspace or Azure ML not available, skipping model registration")
            return None

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train ML model')
    parser.add_argument('--data-path', type=str, default='data/processed', help='Path to processed data')
    parser.add_argument('--model-type', type=str, default='random_forest', choices=['random_forest', 'logistic_regression'])
    parser.add_argument('--model-name', type=str, default='trained_model', help='Name for the model')
    parser.add_argument('--n-estimators', type=int, default=100, help='Number of estimators for RandomForest')
    parser.add_argument('--max-depth', type=int, default=10, help='Max depth for RandomForest')
    parser.add_argument('--C', type=float, default=1.0, help='Regularization parameter for LogisticRegression')
    parser.add_argument('--max-iter', type=int, default=1000, help='Max iterations for LogisticRegression')
    
    return parser.parse_args()

def main():
    """Main training pipeline"""
    args = parse_args()
    
    # Initialize MLflow
    if mlflow:
        mlflow.start_run()
    
    try:
        # Get Azure ML workspace (if available)
        ws = None
        if Workspace:
            try:
                ws = Workspace.from_config()
                logger.info("Connected to Azure ML workspace")
            except Exception as e:
                logger.warning(f"Could not connect to Azure ML workspace: {str(e)}")
        else:
            logger.warning("Azure ML not available")
        
        # Initialize trainer
        trainer = ModelTrainer(workspace=ws)
        
        # Load data
        X_train, X_test, y_train, y_test = trainer.load_data(args.data_path)
        
        # Train model
        model_params = {
            'n_estimators': args.n_estimators,
            'max_depth': args.max_depth,
            'C': args.C,
            'max_iter': args.max_iter
        }
        
        model = trainer.train_model(X_train, y_train, args.model_type, **model_params)
        
        # Log parameters
        for param_name, param_value in model_params.items():
            if mlflow:
                mlflow.log_param(param_name, param_value)
            if trainer.run:
                trainer.run.log(param_name, param_value)
        
        # Evaluate model
        metrics = trainer.evaluate_model(model, X_test, y_test)
        
        # Log metrics
        trainer.log_metrics(metrics)
        
        # Save model
        model_path = trainer.save_model(model, args.model_name)
        
        # Register model in Azure ML
        registered_model = trainer.register_model(model_path, args.model_name)
        
        # Save metrics to file
        with open('outputs/metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info("Training pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}")
        raise
    finally:
        if mlflow:
            mlflow.end_run()

if __name__ == "__main__":
    main()