#!/usr/bin/env python3
"""
Script to validate Azure ML workspace configuration
"""
import json
import os
import sys

def validate_config(config_path='.azureml/config.json'):
    """Validate Azure ML workspace configuration"""
    try:
        if not os.path.exists(config_path):
            print(f"❌ Configuration file not found: {config_path}")
            return False
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        required_fields = ['subscription_id', 'resource_group', 'workspace_name']
        missing_fields = []
        
        for field in required_fields:
            if field not in config or not config[field]:
                missing_fields.append(field)
        
        if missing_fields:
            print(f"❌ Missing required fields: {', '.join(missing_fields)}")
            return False
        
        print("✅ Azure ML workspace configuration is valid")
        print(f"   Subscription ID: {config['subscription_id']}")
        print(f"   Resource Group: {config['resource_group']}")
        print(f"   Workspace Name: {config['workspace_name']}")
        
        return True
        
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in configuration file: {e}")
        return False
    except Exception as e:
        print(f"❌ Error validating configuration: {e}")
        return False

def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else '.azureml/config.json'
    
    print("🔍 Validating Azure ML workspace configuration...")
    
    if validate_config(config_path):
        print("🎉 Configuration validation successful!")
        sys.exit(0)
    else:
        print("💥 Configuration validation failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()