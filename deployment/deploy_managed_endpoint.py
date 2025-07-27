"""
Deploy models using Azure ML Managed Endpoints (SDK v2)
This is the modern approach that doesn't require ACI permissions
"""
import os
import yaml
import argparse
import logging
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Model,
    Environment,
    CodeConfiguration,
    OnlineRequestSettings,
    ProbeSettings
)
from azure.identity import DefaultAzureCredential
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_deployment_config(config_path='deployment_config_production.yml'):
    """Load deployment configuration"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def create_ml_client():
    """Create ML Client using Azure ML SDK v2"""
    try:
        # Try to get workspace info from config
        config_path = '../.azureml/config.json'
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            ml_client = MLClient(
                credential=DefaultAzureCredential(),
                subscription_id=config['subscription_id'],
                resource_group_name=config['resource_group'],
                workspace_name=config['workspace_name']
            )
        else:
            # Fallback to environment variables
            ml_client = MLClient(
                credential=DefaultAzureCredential(),
                subscription_id=os.environ.get('AZURE_SUBSCRIPTION_ID'),
                resource_group_name=os.environ.get('AZURE_RESOURCE_GROUP'),
                workspace_name=os.environ.get('AZURE_ML_WORKSPACE')
            )
        
        logger.info(f"✅ Connected to workspace: {ml_client.workspace_name}")
        return ml_client
        
    except Exception as e:
        logger.error(f"Failed to create ML client: {e}")
        raise


def create_custom_environment_for_deployment(ml_client):
    """Create a custom environment for deployment that matches training environment"""
    try:
        logger.info("Creating custom sklearn environment for deployment...")
        
        # Create environment with conda specification
        conda_spec = """
name: sklearn-deployment
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.8
  - pip
  - pip:
    - azureml-defaults>=1.38.0
    - scikit-learn==1.0.2
    - pandas
    - numpy
    - joblib
    - inference-schema[image-support]
"""
        
        # Write conda spec to a temporary file
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            f.write(conda_spec)
            conda_file_path = f.name
        
        try:
            # Create custom environment with proper parameters
            env = Environment(
                name="sklearn-deployment-env",
                description="Custom sklearn environment for deployment (Python 3.8, scikit-learn 1.0.2)",
                conda_file=conda_file_path,
                image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest"
            )
            
            # Create or update the environment
            created_env = ml_client.environments.create_or_update(env)
            logger.info(f"✅ Created custom sklearn deployment environment: {created_env.name}")
            
            # Return the environment reference in the correct format for SDK v2
            return f"{created_env.name}@latest"
            
        finally:
            # Clean up temp file
            if os.path.exists(conda_file_path):
                os.unlink(conda_file_path)
            
    except Exception as e:
        logger.error(f"Failed to create custom environment: {e}")
        logger.info("Falling back to trying curated environments...")
        
        # Try to find a working sklearn environment from the registry
        try:
            # List environments and find sklearn ones
            environments = ml_client.environments.list()
            sklearn_envs = []
            
            for env in environments:
                env_name = env.name.lower()
                if ('sklearn' in env_name or 'scikit' in env_name) and '1.0' in env_name:
                    sklearn_envs.append(env)
                    
            if sklearn_envs:
                best_env = sklearn_envs[0]
                logger.info(f"Using curated environment: {best_env.name}:{best_env.version}")
                return f"{best_env.name}:{best_env.version}"
            else:
                logger.warning("No suitable sklearn environments found, using minimal environment")
                return "AzureML-Minimal@latest"
                
        except Exception as env_e:
            logger.error(f"Failed to find fallback environment: {env_e}")
            # Last resort - use minimal environment
            return "AzureML-Minimal@latest"

def cleanup_existing_endpoints(ml_client, endpoint_name):
    """Clean up existing endpoints and compute clusters to free quota before deployment"""
    try:
        logger.info(f"🧹 Cleaning up existing resources to free quota...")
        
        # 1. Clean up all existing endpoints first
        logger.info("Deleting all existing endpoints...")
        endpoints = ml_client.online_endpoints.list()
        
        for endpoint in endpoints:
            logger.info(f"Deleting endpoint: {endpoint.name}")
            try:
                # Delete deployments first
                deployments = ml_client.online_deployments.list(endpoint_name=endpoint.name)
                for deployment in deployments:
                    logger.info(f"  Deleting deployment: {deployment.name}")
                    ml_client.online_deployments.begin_delete(
                        name=deployment.name,
                        endpoint_name=endpoint.name
                    ).result()
                
                # Delete the endpoint
                ml_client.online_endpoints.begin_delete(name=endpoint.name).result()
                logger.info(f"✅ Deleted endpoint: {endpoint.name}")
            except Exception as e:
                logger.warning(f"Could not delete endpoint {endpoint.name}: {e}")
        
        # 2. Clean up compute clusters to free quota
        logger.info("Checking compute clusters...")
        try:
            from azure.ai.ml.entities import AmlCompute
            
            # List compute instances that might be consuming quota
            computes = ml_client.compute.list()
            for compute in computes:
                if hasattr(compute, 'size') and compute.size in ['Standard_DS2_v2', 'Standard_DS3_v2', 'Standard_F2s_v2']:
                    logger.info(f"Found compute using quota: {compute.name} ({compute.size})")
                    
                    # For training compute clusters, we just scale them down instead of deleting
                    if compute.name == 'cpu-cluster':
                        logger.info(f"Scaling down compute cluster: {compute.name}")
                        try:
                            # Update the compute to scale down to 0 nodes
                            compute_update = AmlCompute(
                                name=compute.name,
                                min_instances=0,
                                max_instances=1  # Keep small max for future use
                            )
                            ml_client.compute.begin_create_or_update(compute_update).result()
                            logger.info(f"✅ Scaled down compute cluster: {compute.name}")
                        except Exception as e:
                            logger.warning(f"Could not scale down compute {compute.name}: {e}")
                    
        except Exception as e:
            logger.warning(f"Could not cleanup compute resources: {e}")
        
        logger.info("🧹 Cleanup completed - quota should be freed")
        
    except Exception as e:
        logger.warning(f"Cleanup encountered issues: {e}")
        # Continue anyway, cleanup is best-effort

def get_optimal_instance_type():
    """Get optimal instance type from the quota check or use fallback"""
    try:
        # Try to load the optimal instance config if it exists
        config_file = 'optimal_instance_config.json'
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
            optimal_instance = config.get('instance_type', 'Standard_F2s_v2')
            logger.info(f"💰 Using cost-optimized instance: {optimal_instance}")
            return optimal_instance
        else:
            logger.info("📊 No quota optimization config found, using default Azure ML supported instance")
            return 'Standard_F2s_v2'  # Most cost-effective Azure ML supported default
    except Exception as e:
        logger.warning(f"Could not load optimal instance config: {e}")
        return 'Standard_F2s_v2'  # Safe Azure ML supported fallback

def deploy_managed_endpoint(ml_client, model_name, endpoint_name, config):
    """Deploy model to Azure ML Managed Endpoint"""
    try:
        # Clean up existing endpoints first to free quota
        cleanup_existing_endpoints(ml_client, endpoint_name)
        
        # Get the latest version of the model
        model = ml_client.models.get(name=model_name, label="latest")
        logger.info(f"Using model: {model.name} v{model.version}")
        
        # Get optimal instance type for cost efficiency
        optimal_instance_type = get_optimal_instance_type()
        
        # Create a custom environment since curated ones have naming issues with SDK v2
        environment_ref = create_custom_environment_for_deployment(ml_client)
        
        # Create or get endpoint
        try:
            endpoint = ml_client.online_endpoints.get(name=endpoint_name)
            logger.info(f"Using existing endpoint: {endpoint_name}")
        except Exception:
            logger.info(f"Creating new endpoint: {endpoint_name}")
            endpoint = ManagedOnlineEndpoint(
                name=endpoint_name,
                description=config.get('description', 'Production ML Model Endpoint'),
                tags={
                    'model_name': model.name,
                    'model_version': str(model.version),
                    'deployment_type': 'managed_endpoint',
                    'environment': 'production'
                }
            )
            ml_client.online_endpoints.begin_create_or_update(endpoint).result()
            logger.info("✅ Endpoint created successfully")
        
        # Create deployment
        deployment_name = "production"  # Single deployment name
        
        logger.info(f"Creating deployment: {deployment_name}")
        deployment = ManagedOnlineDeployment(
            name=deployment_name,
            endpoint_name=endpoint_name,
            model=model,
            code_configuration=CodeConfiguration(
                code="../src",
                scoring_script="score.py"
            ),
            environment=environment_ref,  # Use dynamically selected environment
            instance_type=optimal_instance_type,  # Cost-optimized instance from quota analysis
            instance_count=1,  # Always use 1 instance to minimize quota usage
            request_settings=OnlineRequestSettings(
                request_timeout_ms=5000,
                max_concurrent_requests_per_instance=1,
                max_queue_wait_ms=1000
            ),
            liveness_probe=ProbeSettings(
                initial_delay=10,
                period=10,
                timeout=5,
                failure_threshold=3
            ),
            readiness_probe=ProbeSettings(
                initial_delay=10,
                period=10,
                timeout=5,
                failure_threshold=3
            )
        )
        
        # Deploy
        logger.info("Starting deployment...")
        ml_client.online_deployments.begin_create_or_update(deployment).result()
        
        # Set traffic to 100% for this deployment
        endpoint.traffic = {deployment_name: 100}
        ml_client.online_endpoints.begin_create_or_update(endpoint).result()
        
        # Get endpoint details
        endpoint = ml_client.online_endpoints.get(name=endpoint_name)
        
        logger.info("✅ Deployment completed successfully!")
        logger.info(f"Endpoint name: {endpoint.name}")
        logger.info(f"Scoring URI: {endpoint.scoring_uri}")
        logger.info(f"Swagger URI: {endpoint.openapi_uri}")
        
        # Save deployment info
        deployment_info = {
            'endpoint_name': endpoint.name,
            'scoring_uri': endpoint.scoring_uri,
            'swagger_uri': endpoint.openapi_uri,
            'model_name': model.name,
            'model_version': model.version,
            'deployment_type': 'managed_endpoint',
            'deployment_name': deployment_name
        }
        
        os.makedirs('outputs', exist_ok=True)
        with open('outputs/deployment_info.json', 'w') as f:
            json.dump(deployment_info, f, indent=2)
        
        return endpoint
        
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        raise

def test_endpoint(ml_client, endpoint_name):
    """Test the deployed endpoint"""
    try:
        # Sample test data (adjust based on your model's input format)
        import json
        test_data = {
            "data": [
                [1.0, 2.0, 3.0, 4.0, 5.0, 1.0]  # 6 features to match trained model
            ]
        }
        
        # Create a temporary test file
        test_file = "/tmp/test_data.json"
        with open(test_file, 'w') as f:
            json.dump(test_data, f)
        
        result = ml_client.online_endpoints.invoke(
            endpoint_name=endpoint_name,
            request_file=test_file,
            deployment_name="production"
        )
        
        # Clean up test file
        os.remove(test_file)
        
        logger.info("✅ Endpoint test successful!")
        logger.info(f"Test result: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Endpoint test failed: {e}")
        return None

def list_endpoints(ml_client):
    """List all endpoints"""
    try:
        endpoints = ml_client.online_endpoints.list()
        
        logger.info(f"\n{'='*80}")
        logger.info(f"MANAGED ENDPOINTS IN WORKSPACE")
        logger.info(f"{'='*80}")
        
        for endpoint in endpoints:
            logger.info(f"\n📊 Endpoint: {endpoint.name}")
            logger.info(f"   Scoring URI: {endpoint.scoring_uri}")
            logger.info(f"   Traffic: {endpoint.traffic}")
            logger.info(f"   Tags: {endpoint.tags}")
            
    except Exception as e:
        logger.error(f"Failed to list endpoints: {e}")

def delete_endpoint(ml_client, endpoint_name):
    """Delete an endpoint"""
    try:
        logger.info(f"Deleting endpoint: {endpoint_name}")
        ml_client.online_endpoints.begin_delete(name=endpoint_name).result()
        logger.info(f"✅ Endpoint {endpoint_name} deleted successfully")
    except Exception as e:
        logger.error(f"Failed to delete endpoint {endpoint_name}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Deploy model using Azure ML Managed Endpoints")
    parser.add_argument("--action", choices=['deploy', 'list', 'test', 'delete', 'cleanup'], default='deploy',
                       help="Action to perform")
    parser.add_argument("--model-name", type=str, help="Name of the registered model")
    parser.add_argument("--endpoint-name", type=str, help="Name for the managed endpoint")
    parser.add_argument("--config-file", type=str, default="deployment_config_production.yml",
                       help="Path to deployment configuration file")
    
    args = parser.parse_args()
    
    try:
        # Create ML client
        ml_client = create_ml_client()
        
        if args.action == 'list':
            list_endpoints(ml_client)
        
        elif args.action == 'deploy':
            if not args.model_name or not args.endpoint_name:
                logger.error("Error: --model-name and --endpoint-name are required for deployment")
                return
            
            # Load deployment configuration
            config = load_deployment_config(args.config_file)
            
            # Deploy model
            endpoint = deploy_managed_endpoint(ml_client, args.model_name, args.endpoint_name, config)
            
            # Test the endpoint
            logger.info("Testing deployed endpoint...")
            test_endpoint(ml_client, args.endpoint_name)
        
        elif args.action == 'test':
            if not args.endpoint_name:
                logger.error("Error: --endpoint-name is required for testing")
                return
            
            test_endpoint(ml_client, args.endpoint_name)
        
        elif args.action == 'delete':
            if not args.endpoint_name:
                logger.error("Error: --endpoint-name is required for deletion")
                return
            
            delete_endpoint(ml_client, args.endpoint_name)
        
        elif args.action == 'cleanup':
            # Clean up all endpoints to free quota
            endpoint_name = args.endpoint_name or "dummy-endpoint-name"
            cleanup_existing_endpoints(ml_client, endpoint_name)
        
    except Exception as e:
        logger.error(f"Operation failed: {e}")
        raise

if __name__ == "__main__":
    main()