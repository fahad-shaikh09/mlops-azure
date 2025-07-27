#!/usr/bin/env python3
"""
Script to check available Azure ML instance types and quota
Automatically selects the most cost-effective instance type that fits within quota
"""
import json
import logging
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Azure ML Managed Endpoints Supported Instance Types (cheapest first)
# Based on Azure ML documentation and pricing
COST_EFFECTIVE_INSTANCES = [
    # F-series (compute optimized, good price/performance) - Supported by Azure ML
    "Standard_F2s_v2",      # 2 vCPUs, 4 GB RAM - Cost effective, supported
    "Standard_F4s_v2",      # 4 vCPUs, 8 GB RAM - Good value
    "Standard_F8s_v2",      # 8 vCPUs, 16 GB RAM - Higher performance
    
    # D-series v2 (general purpose, balanced) - Supported by Azure ML
    "Standard_DS2_v2",      # 2 vCPUs, 7 GB RAM - Good value, well-tested
    "Standard_DS3_v2",      # 4 vCPUs, 14 GB RAM - Standard choice
    "Standard_DS4_v2",      # 8 vCPUs, 28 GB RAM - Higher memory
    
    # D-series v3 (newer generation) - Supported by Azure ML
    "Standard_D2s_v3",      # 2 vCPUs, 8 GB RAM - Good value
    "Standard_D4s_v3",      # 4 vCPUs, 16 GB RAM - Balanced
    "Standard_D8s_v3",      # 8 vCPUs, 32 GB RAM - High performance
    
    # D-series v4 (latest generation) - Supported by Azure ML  
    "Standard_D2s_v4",      # 2 vCPUs, 8 GB RAM - Latest gen
    "Standard_D4s_v4",      # 4 vCPUs, 16 GB RAM - Latest gen
    
    # E-series (memory optimized) - For larger models
    "Standard_E2s_v3",      # 2 vCPUs, 16 GB RAM - Memory optimized
    "Standard_E4s_v3",      # 4 vCPUs, 32 GB RAM - High memory
]

# Instance specifications for quota calculation (Azure ML supported only)
INSTANCE_SPECS = {
    "Standard_F2s_v2":  {"vcpus": 2, "ram_gb": 4},
    "Standard_F4s_v2":  {"vcpus": 4, "ram_gb": 8},
    "Standard_F8s_v2":  {"vcpus": 8, "ram_gb": 16},
    "Standard_DS2_v2":  {"vcpus": 2, "ram_gb": 7},
    "Standard_DS3_v2":  {"vcpus": 4, "ram_gb": 14},
    "Standard_DS4_v2":  {"vcpus": 8, "ram_gb": 28},
    "Standard_D2s_v3":  {"vcpus": 2, "ram_gb": 8},
    "Standard_D4s_v3":  {"vcpus": 4, "ram_gb": 16},
    "Standard_D8s_v3":  {"vcpus": 8, "ram_gb": 32},
    "Standard_D2s_v4":  {"vcpus": 2, "ram_gb": 8},
    "Standard_D4s_v4":  {"vcpus": 4, "ram_gb": 16},
    "Standard_E2s_v3":  {"vcpus": 2, "ram_gb": 16},
    "Standard_E4s_v3":  {"vcpus": 4, "ram_gb": 32},
}

def create_ml_client():
    """Create ML Client"""
    try:
        config_path = '../.azureml/config.json'
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        ml_client = MLClient(
            credential=DefaultAzureCredential(),
            subscription_id=config['subscription_id'],
            resource_group_name=config['resource_group'],
            workspace_name=config['workspace_name']
        )
        
        logger.info(f"✅ Connected to workspace: {ml_client.workspace_name}")
        return ml_client
    except Exception as e:
        logger.error(f"Failed to create ML client: {e}")
        raise

def check_current_quota_usage(ml_client):
    """Check current quota usage for compute resources"""
    try:
        logger.info("🔍 Checking current compute resource usage...")
        
        total_vcpus_used = 0
        resources_found = []
        
        # Check online endpoints (managed endpoints)
        try:
            endpoints = ml_client.online_endpoints.list()
            for endpoint in endpoints:
                try:
                    deployments = ml_client.online_deployments.list(endpoint_name=endpoint.name)
                    for deployment in deployments:
                        if hasattr(deployment, 'instance_type') and hasattr(deployment, 'instance_count'):
                            instance_type = deployment.instance_type
                            instance_count = deployment.instance_count or 1
                            
                            if instance_type in INSTANCE_SPECS:
                                vcpus_per_instance = INSTANCE_SPECS[instance_type]["vcpus"]
                                total_vcpus = vcpus_per_instance * instance_count
                                total_vcpus_used += total_vcpus
                                
                                resources_found.append({
                                    'type': 'Managed Endpoint',
                                    'name': f"{endpoint.name}/{deployment.name}",
                                    'instance_type': instance_type,
                                    'instance_count': instance_count,
                                    'vcpus_used': total_vcpus
                                })
                except Exception as e:
                    logger.warning(f"Could not check deployments for endpoint {endpoint.name}: {e}")
        except Exception as e:
            logger.warning(f"Could not list endpoints: {e}")
        
        # Check compute clusters
        try:
            computes = ml_client.compute.list()
            for compute in computes:
                if hasattr(compute, 'size'):
                    # For compute clusters, check current node count
                    if hasattr(compute, 'current_node_count') and compute.current_node_count > 0:
                        instance_type = compute.size
                        node_count = compute.current_node_count
                        
                        if instance_type in INSTANCE_SPECS:
                            vcpus_per_node = INSTANCE_SPECS[instance_type]["vcpus"]
                            total_vcpus = vcpus_per_node * node_count
                            total_vcpus_used += total_vcpus
                            
                            resources_found.append({
                                'type': 'Compute Cluster',
                                'name': compute.name,
                                'instance_type': instance_type,
                                'instance_count': node_count,
                                'vcpus_used': total_vcpus
                            })
        except Exception as e:
            logger.warning(f"Could not list compute clusters: {e}")
        
        return total_vcpus_used, resources_found
        
    except Exception as e:
        logger.error(f"Failed to check quota usage: {e}")
        return 0, []

def get_available_quota(subscription_id, location="uksouth"):
    """
    Estimate available quota based on common Azure ML limits
    In practice, you'd need to call Azure Resource Manager API for exact quotas
    """
    # Common quota limits for different subscription types
    # These are conservative estimates - actual limits may vary
    QUOTA_ESTIMATES = {
        'free_trial': 6,      # Free tier
        'pay_as_you_go': 20,  # Standard subscription
        'enterprise': 100,    # Enterprise subscription
    }
    
    # For this script, we'll use a conservative estimate
    # In production, you'd call the Azure Resource Manager API
    estimated_quota = QUOTA_ESTIMATES['pay_as_you_go']  # Conservative estimate
    
    logger.info(f"📊 Estimated total vCPU quota: {estimated_quota} cores")
    logger.info("💡 Note: This is an estimate. Actual quota may differ.")
    
    return estimated_quota

def find_optimal_instance_type(available_vcpus, min_vcpus=1):
    """Find the most cost-effective instance type that fits within available quota"""
    
    logger.info(f"🔍 Finding optimal instance type with {available_vcpus} available vCPUs...")
    
    suitable_instances = []
    
    for instance_type in COST_EFFECTIVE_INSTANCES:
        if instance_type in INSTANCE_SPECS:
            specs = INSTANCE_SPECS[instance_type]
            required_vcpus = specs["vcpus"]
            
            if required_vcpus <= available_vcpus and required_vcpus >= min_vcpus:
                suitable_instances.append({
                    'instance_type': instance_type,
                    'vcpus': required_vcpus,
                    'ram_gb': specs["ram_gb"],
                    'cost_rank': COST_EFFECTIVE_INSTANCES.index(instance_type)  # Lower = cheaper
                })
    
    if not suitable_instances:
        logger.warning(f"⚠️ No suitable instances found for {available_vcpus} vCPUs")
        return None
    
    # Sort by cost rank (cheapest first)
    suitable_instances.sort(key=lambda x: x['cost_rank'])
    
    logger.info(f"💰 Found {len(suitable_instances)} suitable cost-effective instances:")
    for i, instance in enumerate(suitable_instances[:3]):  # Show top 3
        rank_emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
        logger.info(f"  {rank_emoji} {instance['instance_type']}: {instance['vcpus']} vCPUs, {instance['ram_gb']} GB RAM")
    
    optimal_instance = suitable_instances[0]
    logger.info(f"✅ Recommended: {optimal_instance['instance_type']} (cheapest option)")
    
    return optimal_instance

def generate_deployment_config(instance_type, instance_count=1):
    """Generate deployment configuration with optimal instance type"""
    # Determine cost level based on Azure ML supported instances
    if instance_type in ['Standard_F2s_v2', 'Standard_DS2_v2']:
        cost_level = 'Low'
    elif instance_type in ['Standard_D2s_v3', 'Standard_D2s_v4']:
        cost_level = 'Low-Medium'
    elif instance_type in ['Standard_F4s_v2', 'Standard_DS3_v2', 'Standard_D4s_v3']:
        cost_level = 'Medium'
    else:
        cost_level = 'Higher'
    
    config = {
        'instance_type': instance_type,
        'instance_count': instance_count,
        'estimated_cost': cost_level,
        'use_case': 'Azure ML managed endpoint inference',
        'recommendation': f"Using {instance_type} - most cost-effective Azure ML supported instance"
    }
    return config

def main():
    """Main function to check quota and recommend optimal instance type"""
    try:
        logger.info("🚀 Azure ML Quota and Instance Optimization Tool")
        logger.info("=" * 60)
        
        # Create ML client
        ml_client = create_ml_client()
        
        # Check current usage
        vcpus_used, resources = check_current_quota_usage(ml_client)
        
        logger.info(f"📊 Current vCPU usage: {vcpus_used} cores")
        
        if resources:
            logger.info("📋 Current resources:")
            for resource in resources:
                logger.info(f"  • {resource['type']}: {resource['name']}")
                logger.info(f"    Instance: {resource['instance_type']} x{resource['instance_count']} = {resource['vcpus_used']} vCPUs")
        else:
            logger.info("✅ No active resources found - quota is free!")
        
        # Get estimated quota
        total_quota = get_available_quota(ml_client.subscription_id)
        available_quota = max(0, total_quota - vcpus_used)
        
        logger.info(f"💡 Available quota: {available_quota} vCPUs (of {total_quota} total)")
        
        # Find optimal instance types for different scenarios
        logger.info("\n🎯 RECOMMENDATIONS:")
        logger.info("-" * 40)
        
        # For training workloads (need at least 2 vCPUs for reasonable performance)
        training_instance = find_optimal_instance_type(available_quota, min_vcpus=2)
        if training_instance:
            logger.info(f"🏃 Training: {training_instance['instance_type']} ({training_instance['vcpus']} vCPUs)")
        
        # For inference workloads (can work with 1 vCPU)
        inference_instance = find_optimal_instance_type(available_quota, min_vcpus=1)
        if inference_instance:
            logger.info(f"🎯 Inference: {inference_instance['instance_type']} ({inference_instance['vcpus']} vCPUs)")
        
        # Generate configuration
        if inference_instance:
            config = generate_deployment_config(inference_instance['instance_type'])
            
            logger.info("\n⚙️ DEPLOYMENT CONFIG:")
            logger.info("-" * 25)
            for key, value in config.items():
                logger.info(f"  {key}: {value}")
            
            # Save configuration to file
            output_file = 'optimal_instance_config.json'
            with open(output_file, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info(f"\n💾 Configuration saved to: {output_file}")
            
            return inference_instance['instance_type']
        else:
            logger.error("❌ No suitable instances found within quota limits")
            logger.info("💡 Suggestions:")
            logger.info("  1. Clean up existing resources to free quota")
            logger.info("  2. Request quota increase from Azure support")
            logger.info("  3. Use a different region with available quota")
            return None
            
    except Exception as e:
        logger.error(f"❌ Script failed: {e}")
        return None

if __name__ == "__main__":
    optimal_instance = main()
    if optimal_instance:
        print(f"\n🎉 SUCCESS: Use {optimal_instance} for cost-effective deployment!")
    else:
        print("\n😞 No optimal instance found. Check quota and cleanup resources.")