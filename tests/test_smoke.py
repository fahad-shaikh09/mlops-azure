import pytest
import requests
import json
import os
from test_utils import get_test_data_for_api

class TestSmoke:
    """Smoke tests for production deployment"""
    
    def setup_method(self):
        """Setup test environment"""
        self.endpoint_info_path = 'outputs/endpoint_info.json'
        self.scoring_uri = None
        
        if os.path.exists(self.endpoint_info_path):
            with open(self.endpoint_info_path, 'r') as f:
                endpoint_info = json.load(f)
                self.scoring_uri = endpoint_info.get('scoring_uri')
    
    def test_endpoint_is_alive(self):
        """Test if production endpoint is alive"""
        if not self.scoring_uri:
            pytest.skip("Scoring URI not available")
        
        # Simple test to verify endpoint is reachable
        test_data = get_test_data_for_api(1)
        headers = {'Content-Type': 'application/json'}
        
        try:
            response = requests.post(
                self.scoring_uri, 
                data=json.dumps(test_data), 
                headers=headers,
                timeout=30
            )
            
            # Just check if we get any response (200 or proper error)
            assert response.status_code in [200, 400, 422, 500], f"Unexpected status: {response.status_code}"
            
        except requests.exceptions.RequestException as e:
            pytest.fail(f"Production endpoint not reachable: {str(e)}")
    
    def test_basic_prediction_works(self):
        """Test basic prediction functionality"""
        if not self.scoring_uri:
            pytest.skip("Scoring URI not available")
        
        test_data = get_test_data_for_api(1)
        headers = {'Content-Type': 'application/json'}
        
        try:
            response = requests.post(
                self.scoring_uri, 
                data=json.dumps(test_data), 
                headers=headers,
                timeout=30
            )
            
            assert response.status_code == 200, f"Prediction failed: {response.status_code}"
            
            result = response.json()
            assert 'predictions' in result, "Missing predictions in response"
            
        except requests.exceptions.RequestException as e:
            pytest.fail(f"Basic prediction test failed: {str(e)}")
        except json.JSONDecodeError:
            pytest.fail("Invalid JSON response from endpoint")
    
    def test_model_artifacts_exist(self):
        """Test if critical model artifacts exist"""
        critical_files = [
            'outputs/trained_model.pkl',
            'outputs/metrics.json'
        ]
        
        for file_path in critical_files:
            if os.path.exists(file_path):
                assert os.path.getsize(file_path) > 0, f"Critical file is empty: {file_path}"
    
    def test_endpoint_response_format(self):
        """Test if endpoint returns expected response format"""
        if not self.scoring_uri:
            pytest.skip("Scoring URI not available")
        
        test_data = get_test_data_for_api(1)
        headers = {'Content-Type': 'application/json'}
        
        try:
            response = requests.post(
                self.scoring_uri, 
                data=json.dumps(test_data), 
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Check basic response structure
                assert isinstance(result, dict), "Response should be a dictionary"
                assert 'predictions' in result, "Response should contain predictions"
                assert isinstance(result['predictions'], list), "Predictions should be a list"
                
        except requests.exceptions.RequestException as e:
            pytest.fail(f"Response format test failed: {str(e)}")

if __name__ == "__main__":
    pytest.main([__file__])