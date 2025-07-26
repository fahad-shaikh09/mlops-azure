import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from data_preprocessing import DataPreprocessor

class TestDataValidation:
    
    def setup_method(self):
        """Setup test data"""
        self.preprocessor = DataPreprocessor()
        
        # Create sample test data - Change
        self.test_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [7, 3, 8, 11, 4],
            'categorical': ['A', 'B', 'A', 'C', 'B'],
            'target': [0, 1, 0, 1, 0]
        })
    
    def test_data_schema_validation(self):
        """Test if data has required columns"""
        required_columns = ['feature1', 'feature2', 'categorical', 'target']
        
        for col in required_columns:
            assert col in self.test_data.columns, f"Missing required column: {col}"
    
    def test_data_types(self):
        """Test data type validation"""
        assert pd.api.types.is_numeric_dtype(self.test_data['feature1']), "feature1 should be numeric"
        assert pd.api.types.is_numeric_dtype(self.test_data['feature2']), "feature2 should be numeric"
        assert pd.api.types.is_object_dtype(self.test_data['categorical']), "categorical should be object type"
    
    def test_data_quality_checks(self):
        """Test data quality metrics"""
        # Check for minimum number of samples
        assert len(self.test_data) >= 5, "Insufficient number of samples"
        
        # Check for data completeness
        completeness = (1 - self.test_data.isnull().sum() / len(self.test_data)) * 100
        assert all(completeness >= 80), "Data completeness below threshold"
    
    def test_target_distribution(self):
        """Test target variable distribution"""
        target_counts = self.test_data['target'].value_counts()
        
        # Check if target has both classes
        assert len(target_counts) >= 2, "Target should have at least 2 classes"
        
        # Check for class imbalance (no class should be less than 10% of total)
        min_class_ratio = target_counts.min() / len(self.test_data)
        assert min_class_ratio >= 0.1, "Severe class imbalance detected"
    
    def test_data_preprocessing_pipeline(self):
        """Test the data preprocessing pipeline"""
        df_clean = self.preprocessor.clean_data(self.test_data.copy())
        df_features = self.preprocessor.feature_engineering(df_clean)
        df_encoded = self.preprocessor.encode_categorical(df_features, target_column='target')
        
        # Check if preprocessing completed without errors
        assert df_encoded is not None, "Preprocessing pipeline failed"
        assert len(df_encoded) > 0, "No data after preprocessing"
    
    def test_outlier_detection(self):
        """Test outlier detection"""
        # Simple outlier detection using IQR
        numeric_columns = self.test_data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if col != 'target':  # Skip target column
                Q1 = self.test_data[col].quantile(0.25)
                Q3 = self.test_data[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = self.test_data[(self.test_data[col] < lower_bound) | 
                                        (self.test_data[col] > upper_bound)]
                
                outlier_ratio = len(outliers) / len(self.test_data)
                assert outlier_ratio < 0.1, f"Too many outliers in {col}: {outlier_ratio:.2%}"
    
    def test_feature_correlation(self):
        """Test feature correlation"""
        numeric_data = self.test_data.select_dtypes(include=[np.number])
        correlation_matrix = numeric_data.corr()
        
        # Check for highly correlated features (excluding diagonal)
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                if abs(correlation_matrix.iloc[i, j]) > 0.95:
                    high_corr_pairs.append((correlation_matrix.columns[i], correlation_matrix.columns[j]))
        
        assert len(high_corr_pairs) == 0, f"High correlation detected: {high_corr_pairs}"

if __name__ == "__main__":
    pytest.main([__file__])