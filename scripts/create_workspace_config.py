#!/usr/bin/env python3
"""
Script to create Azure ML workspace configuration file safely
"""
import json
import os
import sys

def create_config():
    """Create Azure ML workspace configuration"""
    try:
        # Get environment variables
        subscription_id = os.environ.get('AZURE_SUBSCRIPTION_ID')
        resource_group = os.environ.get('AZURE_RESOURCE_GROUP')
        workspace_name = os.environ.get('AZURE_ML_WORKSPACE')
        
        # Validate required variables
        if not all([subscription_id, resource_group, workspace_name]):
            print("❌ Missing required environment variables:")
            if not subscription_id:
                print("  - AZURE_SUBSCRIPTION_ID")
            if not resource_group:
                print("  - AZURE_RESOURCE_GROUP")
            if not workspace_name:
                print("  - AZURE_ML_WORKSPACE")
            return False
        
        # Clean the values (remove any potential control characters)
        subscription_id = subscription_id.strip()
        resource_group = resource_group.strip()
        workspace_name = workspace_name.strip()
        
        # Create config dictionary
        config = {
            'subscription_id': subscription_id,
            'resource_group': resource_group,
            'workspace_name': workspace_name
        }
        
        # Ensure directory exists
        os.makedirs('.azureml', exist_ok=True)
        
        # Write config file
        with open('.azureml/config.json', 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=True)
        
        print("✅ Azure ML workspace config created successfully")
        print(f"   Subscription ID: {subscription_id}")
        print(f"   Resource Group: {resource_group}")
        print(f"   Workspace Name: {workspace_name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating workspace config: {e}")
        return False

def main():
    """Main function"""
    print("🔧 Creating Azure ML workspace configuration...")
    
    if create_config():
        print("🎉 Configuration created successfully!")
        sys.exit(0)
    else:
        print("💥 Configuration creation failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()