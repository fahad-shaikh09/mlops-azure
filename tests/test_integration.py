import pytest
import requests
import json
import time
import os
from test_utils import get_test_data_for_api

class TestIntegration:
    
    def setup_method(self):
        """Setup test environment"""
        # Read endpoint info if available
        self.endpoint_info_path = 'outputs/endpoint_info.json'
        self.scoring_uri = None
        
        if os.path.exists(self.endpoint_info_path):
            with open(self.endpoint_info_path, 'r') as f:
                endpoint_info = json.load(f)
                self.scoring_uri = endpoint_info.get('scoring_uri')
    
    def test_endpoint_health(self):
        """Test if endpoint is healthy and accessible"""
        if not self.scoring_uri:
            pytest.skip("Scoring URI not available")
        
        # Basic health check - try to reach the endpoint
        try:
            # Some endpoints have a health check endpoint
            health_uri = self.scoring_uri.replace('/score', '/health')
            response = requests.get(health_uri, timeout=30)
            
            # If health endpoint doesn't exist, that's okay
            if response.status_code == 404:
                pytest.skip("Health endpoint not available")
            
            assert response.status_code == 200, f"Health check failed with status: {response.status_code}"
        except requests.exceptions.RequestException as e:
            pytest.fail(f"Failed to reach endpoint: {str(e)}")
    
    def test_endpoint_prediction_valid_input(self):
        """Test endpoint with valid input data"""
        if not self.scoring_uri:
            pytest.skip("Scoring URI not available")
        
        # Test data - use properly formatted data
        test_data = get_test_data_for_api(2)
        
        headers = {'Content-Type': 'application/json'}
        
        try:
            response = requests.post(
                self.scoring_uri, 
                data=json.dumps(test_data), 
                headers=headers,
                timeout=60
            )
            
            assert response.status_code == 200, f"Prediction failed with status: {response.status_code}"
            
            result = response.json()
            assert 'predictions' in result, "Response missing predictions"
            assert len(result['predictions']) == 2, "Wrong number of predictions returned"
            
        except requests.exceptions.RequestException as e:
            pytest.fail(f"Request failed: {str(e)}")
        except json.JSONDecodeError as e:
            pytest.fail(f"Invalid JSON response: {str(e)}")
    
    def test_endpoint_prediction_single_sample(self):
        """Test endpoint with single sample"""
        if not self.scoring_uri:
            pytest.skip("Scoring URI not available")
        
        # Single sample test data
        test_data = get_test_data_for_api(1)
        
        headers = {'Content-Type': 'application/json'}
        
        try:
            response = requests.post(
                self.scoring_uri, 
                data=json.dumps(test_data), 
                headers=headers,
                timeout=60
            )
            
            assert response.status_code == 200, f"Single sample prediction failed: {response.status_code}"
            
            result = response.json()
            assert 'predictions' in result, "Response missing predictions"
            assert len(result['predictions']) == 1, "Wrong number of predictions for single sample"
            
        except requests.exceptions.RequestException as e:
            pytest.fail(f"Single sample request failed: {str(e)}")
    
    def test_endpoint_error_handling_invalid_input(self):
        """Test endpoint error handling with invalid input"""
        if not self.scoring_uri:
            pytest.skip("Scoring URI not available")
        
        # Invalid test data (missing required fields)
        invalid_data = {
            "invalid_field": [1, 2, 3]
        }
        
        headers = {'Content-Type': 'application/json'}
        
        try:
            response = requests.post(
                self.scoring_uri, 
                data=json.dumps(invalid_data), 
                headers=headers,
                timeout=60
            )
            
            # Should return an error status or error message
            if response.status_code == 200:
                result = response.json()
                assert 'error' in result, "Expected error message for invalid input"
            else:
                assert response.status_code in [400, 422, 500], f"Unexpected status code: {response.status_code}"
                
        except requests.exceptions.RequestException as e:
            pytest.fail(f"Error handling test failed: {str(e)}")
    
    def test_endpoint_response_time(self):
        """Test endpoint response time"""
        if not self.scoring_uri:
            pytest.skip("Scoring URI not available")
        
        test_data = get_test_data_for_api(1)
        
        headers = {'Content-Type': 'application/json'}
        max_response_time = 30  # seconds
        
        try:
            start_time = time.time()
            response = requests.post(
                self.scoring_uri, 
                data=json.dumps(test_data), 
                headers=headers,
                timeout=max_response_time
            )
            end_time = time.time()
            
            response_time = end_time - start_time
            
            assert response.status_code == 200, f"Response time test failed: {response.status_code}"
            assert response_time <= max_response_time, f"Response time {response_time:.2f}s exceeds limit {max_response_time}s"
            
        except requests.exceptions.Timeout:
            pytest.fail(f"Request timed out after {max_response_time} seconds")
        except requests.exceptions.RequestException as e:
            pytest.fail(f"Response time test failed: {str(e)}")
    
    def test_endpoint_concurrent_requests(self):
        """Test endpoint with concurrent requests"""
        if not self.scoring_uri:
            pytest.skip("Scoring URI not available")
        
        import concurrent.futures
        import threading
        
        test_data = get_test_data_for_api(1)
        
        headers = {'Content-Type': 'application/json'}
        num_concurrent_requests = 5
        
        def make_request():
            try:
                response = requests.post(
                    self.scoring_uri, 
                    data=json.dumps(test_data), 
                    headers=headers,
                    timeout=60
                )
                return response.status_code == 200
            except:
                return False
        
        # Make concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent_requests) as executor:
            futures = [executor.submit(make_request) for _ in range(num_concurrent_requests)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        successful_requests = sum(results)
        success_rate = successful_requests / num_concurrent_requests
        
        assert success_rate >= 0.8, f"Concurrent request success rate {success_rate:.2%} below threshold 80%"
    
    def test_endpoint_data_validation(self):
        """Test endpoint input data validation"""
        if not self.scoring_uri:
            pytest.skip("Scoring URI not available")
        
        # Test with wrong number of features
        wrong_features_data = {
            "data": [[1.0, 2.0]]  # Too few features
        }
        
        headers = {'Content-Type': 'application/json'}
        
        try:
            response = requests.post(
                self.scoring_uri, 
                data=json.dumps(wrong_features_data), 
                headers=headers,
                timeout=60
            )
            
            # Should handle gracefully - either return error status or error message
            if response.status_code == 200:
                result = response.json()
                assert 'error' in result, "Expected error for wrong number of features"
            else:
                assert response.status_code in [400, 422, 500], f"Unexpected status for wrong features: {response.status_code}"
                
        except requests.exceptions.RequestException as e:
            pytest.fail(f"Data validation test failed: {str(e)}")

if __name__ == "__main__":
    pytest.main([__file__])