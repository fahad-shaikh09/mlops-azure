import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import joblib
import os
import argparse

try:
    from azure.storage.blob import BlobServiceClient  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    BlobServiceClient = None

try:
    from azureml.core import Dataset, Workspace  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    Dataset = None
    Workspace = None
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self, workspace=None):
        self.workspace = workspace
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.imputer = SimpleImputer(strategy='median')
        
    def load_data(self, dataset_name=None, file_path=None):
        """Load data from Azure ML Dataset or local file"""
        if dataset_name and self.workspace:
            if Dataset is None:
                raise ImportError(
                    "azureml-core is required to load data from an Azure ML Dataset"
                )
            dataset = Dataset.get_by_name(self.workspace, dataset_name)
            df = dataset.to_pandas_dataframe()
            logger.info(f"Loaded dataset {dataset_name} from Azure ML")
        elif file_path:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Data file not found: {file_path}")
            df = pd.read_csv(file_path)
            logger.info(f"Loaded data from {file_path}")
        else:
            raise ValueError("Either dataset_name or file_path must be provided")
        
        return df
    
    def clean_data(self, df):
        """Clean and preprocess the data"""
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Handle missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = self.imputer.fit_transform(df[numeric_columns])
        
        # Fill categorical missing values
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
        
        logger.info("Data cleaning completed")
        return df
    
    def feature_engineering(self, df):
        """Create new features and transform existing ones"""
        # Example feature engineering - customize based on your data
        for col in df.select_dtypes(include=[np.number]).columns:
            # Create squared features for numeric columns
            df[f'{col}_squared'] = df[col] ** 2
            
        logger.info("Feature engineering completed")
        return df
    
    def encode_categorical(self, df, target_column=None):
        """Encode categorical variables"""
        categorical_columns = df.select_dtypes(include=['object']).columns

        if target_column and target_column in categorical_columns:
            categorical_columns = categorical_columns.drop(target_column)
        
        # One-hot encode categorical variables
        df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
        
        # Encode target variable if specified
        if target_column and target_column in df.columns:
            df_encoded[target_column] = self.label_encoder.fit_transform(df[target_column])
        
        logger.info("Categorical encoding completed")
        return df_encoded
    
    def scale_features(self, X_train, X_test=None):
        """Scale numerical features"""
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled
    
    def split_data(self, df, target_column, test_size=0.2, random_state=42):
        """Split data into train and test sets"""
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
        if not use_stratify:
            logger.warning("Stratification disabled due to insufficient samples per class")
        
        return X_train, X_test, y_train, y_test
    
    def save_preprocessors(self, output_dir='models'):
        """Save preprocessing objects"""
        os.makedirs(output_dir, exist_ok=True)
        
        joblib.dump(self.scaler, os.path.join(output_dir, 'scaler.pkl'))
        joblib.dump(self.label_encoder, os.path.join(output_dir, 'label_encoder.pkl'))
        joblib.dump(self.imputer, os.path.join(output_dir, 'imputer.pkl'))
        
        logger.info(f"Preprocessors saved to {output_dir}")
    
    def load_preprocessors(self, model_dir='models'):
        """Load preprocessing objects"""
        self.scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
        self.label_encoder = joblib.load(os.path.join(model_dir, 'label_encoder.pkl'))
        self.imputer = joblib.load(os.path.join(model_dir, 'imputer.pkl'))
        
        logger.info(f"Preprocessors loaded from {model_dir}")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run data preprocessing")
    parser.add_argument(
        "--data-file",
        type=str,
        default="data/raw_data.csv",
        help="Path to raw CSV file",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Name of Azure ML dataset",
    )
    return parser.parse_args()


def main():
    """Main preprocessing pipeline"""
    args = parse_args()

    # Initialize preprocessor
    preprocessor = DataPreprocessor()

    # Load data from file or Azure ML
    df = preprocessor.load_data(dataset_name=args.dataset_name, file_path=args.data_file)
    
    # Preprocess data
    df_clean = preprocessor.clean_data(df)
    df_features = preprocessor.feature_engineering(df_clean)
    df_encoded = preprocessor.encode_categorical(df_features, target_column='target')
    
    # Split data
    X_train, X_test, y_train, y_test = preprocessor.split_data(df_encoded, 'target')
    
    # Scale features
    X_train_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_test)
    
    # Save processed data
    os.makedirs('data/processed', exist_ok=True)
    pd.DataFrame(X_train_scaled).to_csv('data/processed/X_train.csv', index=False)
    pd.DataFrame(X_test_scaled).to_csv('data/processed/X_test.csv', index=False)
    pd.DataFrame(y_train).to_csv('data/processed/y_train.csv', index=False)
    pd.DataFrame(y_test).to_csv('data/processed/y_test.csv', index=False)
    
    # Save preprocessors
    preprocessor.save_preprocessors()
    
    logger.info("Data preprocessing pipeline completed successfully")

if __name__ == "__main__":
    main()