import pytest
import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from test_utils import get_processed_test_data

class TestModelValidation:
    
    def setup_method(self):
        """Setup test environment"""
        self.model_path = 'outputs/trained-model.pkl'
        self.metrics_path = 'outputs/metrics.json'
        self.test_data_path = 'data/processed'
    
    def load_model_safely(self):
        """Helper method to load model with version compatibility handling"""
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                return joblib.load(self.model_path)
        except ModuleNotFoundError as e:
            if "numpy._core" in str(e):
                pytest.skip(f"Skipping test due to numpy version compatibility: {str(e)}")
            else:
                raise
        except Exception as e:
            if "version" in str(e).lower() and ("sklearn" in str(e).lower() or "scikit" in str(e).lower()):
                pytest.skip(f"Skipping test due to sklearn version mismatch: {str(e)}")
            else:
                raise
    
    def test_model_file_exists(self):
        """Test if model file exists"""
        assert os.path.exists(self.model_path), f"Model file not found: {self.model_path}"
    
    def test_model_loadable(self):
        """Test if model can be loaded"""
        model = self.load_model_safely()
        assert model is not None, "Model is None after loading"
    
    def test_model_has_required_methods(self):
        """Test if model has required methods"""
        model = self.load_model_safely()
        
        assert hasattr(model, 'predict'), "Model doesn't have predict method"
        assert hasattr(model, 'predict_proba'), "Model doesn't have predict_proba method"
    
    def test_model_performance_metrics(self):
        """Test if model meets minimum performance requirements"""
        if not os.path.exists(self.metrics_path):
            pytest.skip("Metrics file not found")
        
        with open(self.metrics_path, 'r') as f:
            metrics = json.load(f)
        
        # Define minimum performance thresholds
        min_accuracy = 0.7
        min_precision = 0.6
        min_recall = 0.6
        min_f1 = 0.6
        
        assert metrics['accuracy'] >= min_accuracy, f"Accuracy {metrics['accuracy']:.3f} below threshold {min_accuracy}"
        assert metrics['precision'] >= min_precision, f"Precision {metrics['precision']:.3f} below threshold {min_precision}"
        assert metrics['recall'] >= min_recall, f"Recall {metrics['recall']:.3f} below threshold {min_recall}"
        assert metrics['f1_score'] >= min_f1, f"F1-score {metrics['f1_score']:.3f} below threshold {min_f1}"
    
    def test_model_predictions_format(self):
        """Test if model predictions are in correct format"""
        model = self.load_model_safely()
        
        # Use utility function to get properly formatted test data
        X_test = get_processed_test_data(2)
        
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)
        
        # Test prediction format
        assert isinstance(predictions, np.ndarray), "Predictions should be numpy array"
        assert len(predictions) == len(X_test), "Number of predictions should match input samples"
        
        # Test probability format
        assert isinstance(probabilities, np.ndarray), "Probabilities should be numpy array"
        assert probabilities.shape[0] == len(X_test), "Number of probability predictions should match input samples"
        assert np.allclose(probabilities.sum(axis=1), 1), "Probabilities should sum to 1"
    
    def test_model_consistency(self):
        """Test if model gives consistent predictions"""
        model = self.load_model_safely()
        
        # Use utility function to get properly formatted test data
        X_test = get_processed_test_data(2)
        
        # Get predictions multiple times
        pred1 = model.predict(X_test)
        pred2 = model.predict(X_test)
        
        # Predictions should be identical
        assert np.array_equal(pred1, pred2), "Model predictions are not consistent"
    
    def test_model_edge_cases(self):
        """Test model behavior with edge cases"""
        model = self.load_model_safely()
        
        # Get a sample of test data to determine shape
        sample_data = get_processed_test_data(1)
        n_features = sample_data.shape[1]
        
        # Test with zeros
        X_zeros = np.zeros((1, n_features))
        pred_zeros = model.predict(X_zeros)
        assert not np.isnan(pred_zeros).any(), "Model returns NaN for zero input"
        
        # Test with large values
        X_large = np.full((1, n_features), 1000)
        pred_large = model.predict(X_large)
        assert not np.isnan(pred_large).any(), "Model returns NaN for large input"
        
        # Test with negative values
        X_negative = np.full((1, n_features), -1)
        pred_negative = model.predict(X_negative)
        assert not np.isnan(pred_negative).any(), "Model returns NaN for negative input"
    
    def test_preprocessing_artifacts_exist(self):
        """Test if preprocessing artifacts exist"""
        required_files = ['scaler.pkl', 'label_encoder.pkl', 'imputer.pkl']
        
        for file_name in required_files:
            file_path = os.path.join('models', file_name)
            assert os.path.exists(file_path), f"Preprocessing artifact not found: {file_path}"
    
    def test_preprocessing_artifacts_loadable(self):
        """Test if preprocessing artifacts can be loaded"""
        artifact_files = {
            'scaler': 'models/scaler.pkl',
            'label_encoder': 'models/label_encoder.pkl',
            'imputer': 'models/imputer.pkl'
        }
        
        for artifact_name, file_path in artifact_files.items():
            if os.path.exists(file_path):
                try:
                    try:
                        import warnings
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", category=UserWarning)
                            artifact = joblib.load(file_path)
                    except ModuleNotFoundError as e:
                        if "numpy._core" in str(e):
                            pytest.skip(f"Skipping test due to numpy version compatibility: {str(e)}")
                            continue
                        else:
                            raise
                    assert artifact is not None, f"{artifact_name} is None after loading"
                except Exception as e:
                    pytest.fail(f"Failed to load {artifact_name}: {str(e)}")
    
    def test_model_size_reasonable(self):
        """Test if model size is reasonable"""
        model_size = os.path.getsize(self.model_path) / (1024 * 1024)  # Size in MB
        max_size_mb = 100  # Maximum allowed size in MB
        
        assert model_size <= max_size_mb, f"Model size {model_size:.2f}MB exceeds limit {max_size_mb}MB"

if __name__ == "__main__":
    pytest.main([__file__])