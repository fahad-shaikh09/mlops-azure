"""
Azure ML Model Management utilities
This script helps you find, list, and manage models in Azure ML Model Registry
"""
import argparse
import logging
from azureml.core import Workspace, Model
from azureml.core.model import ModelProfile
import pandas as pd
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AzureMLModelManager:
    def __init__(self, workspace=None):
        self.workspace = workspace or Workspace.from_config()
        logger.info(f"Connected to workspace: {self.workspace.name}")
    
    def list_all_models(self, model_name=None, tags=None):
        """List all models in the registry"""
        logger.info("Retrieving models from Azure ML Model Registry...")
        
        if model_name:
            models = Model.list(self.workspace, name=model_name, tags=tags)
        else:
            models = Model.list(self.workspace, tags=tags)
        
        if not models:
            logger.info("No models found in the registry")
            return []
        
        model_info = []
        for model in models:
            info = {
                'name': model.name,
                'version': model.version,
                'id': model.id,
                'created_time': model.created_time,
                'description': model.description,
                'tags': model.tags,
                'url': model.url,
                'run_id': getattr(model, 'run_id', 'N/A')
            }
            model_info.append(info)
        
        # Sort by creation time (newest first)
        model_info.sort(key=lambda x: x['created_time'], reverse=True)
        
        return model_info
    
    def get_model_details(self, model_name, version=None):
        """Get detailed information about a specific model"""
        try:
            if version:
                model = Model(self.workspace, name=model_name, version=version)
            else:
                # Get latest version
                model = Model(self.workspace, name=model_name)
            
            details = {
                'name': model.name,
                'version': model.version,
                'id': model.id,
                'created_time': model.created_time,
                'modified_time': getattr(model, 'modified_time', None),
                'description': model.description,
                'tags': model.tags,
                'properties': getattr(model, 'properties', {}),
                'url': model.url,
                'run_id': getattr(model, 'run_id', 'N/A'),
                'framework': model.tags.get('framework', 'Unknown'),
                'accuracy': model.tags.get('accuracy', 'N/A'),
                'f1_score': model.tags.get('f1_score', 'N/A')
            }
            
            return details
            
        except Exception as e:
            logger.error(f"Failed to get model details: {str(e)}")
            return None
    
    def download_model(self, model_name, version=None, target_dir='./downloaded_models'):
        """Download a model from the registry"""
        import os
        
        try:
            if version:
                model = Model(self.workspace, name=model_name, version=version)
            else:
                model = Model(self.workspace, name=model_name)
            
            # Create target directory
            os.makedirs(target_dir, exist_ok=True)
            
            # Download model
            model_path = model.download(target_dir=target_dir, exist_ok=True)
            logger.info(f"Model downloaded to: {model_path}")
            
            return model_path
            
        except Exception as e:
            logger.error(f"Failed to download model: {str(e)}")
            return None
    
    def compare_model_versions(self, model_name, versions=None):
        """Compare different versions of a model"""
        if not versions:
            # Get all versions of the model
            models = Model.list(self.workspace, name=model_name)
            versions = [model.version for model in models]
        
        comparison = []
        for version in versions:
            details = self.get_model_details(model_name, version)
            if details:
                comparison.append({
                    'version': details['version'],
                    'created_time': details['created_time'],
                    'accuracy': details['accuracy'],
                    'f1_score': details['f1_score'],
                    'run_id': details['run_id'],
                    'description': details['description']
                })
        
        return comparison
    
    def print_model_summary(self, models):
        """Print a summary of models"""
        if not models:
            print("No models found.")
            return
        
        print(f"\n{'='*80}")
        print(f"AZURE ML MODEL REGISTRY SUMMARY")
        print(f"{'='*80}")
        print(f"Workspace: {self.workspace.name}")
        print(f"Resource Group: {self.workspace.resource_group}")
        print(f"Subscription: {self.workspace.subscription_id}")
        print(f"Total Models: {len(models)}")
        print(f"{'='*80}")
        
        for i, model in enumerate(models, 1):
            print(f"\n{i}. Model: {model['name']} (v{model['version']})")
            print(f"   ID: {model['id']}")
            print(f"   Created: {model['created_time']}")
            print(f"   Framework: {model['tags'].get('framework', 'Unknown')}")
            print(f"   Accuracy: {model['tags'].get('accuracy', 'N/A')}")
            print(f"   F1 Score: {model['tags'].get('f1_score', 'N/A')}")
            print(f"   Run ID: {model['run_id']}")
            print(f"   Description: {model['description'] or 'No description'}")
            print(f"   URL: {model['url']}")
            
            if model['tags']:
                print(f"   Tags: {model['tags']}")
            print(f"   {'-'*60}")

def main():
    parser = argparse.ArgumentParser(description="Azure ML Model Management")
    parser.add_argument("--action", choices=['list', 'details', 'download', 'compare'], 
                       default='list', help="Action to perform")
    parser.add_argument("--model-name", type=str, help="Specific model name")
    parser.add_argument("--version", type=int, help="Model version")
    parser.add_argument("--target-dir", type=str, default='./downloaded_models', 
                       help="Directory to download model to")
    parser.add_argument("--output-format", choices=['table', 'json'], default='table',
                       help="Output format")
    parser.add_argument("--tags", type=str, help="Filter by tags (key=value format)")
    
    args = parser.parse_args()
    
    try:
        manager = AzureMLModelManager()
        
        if args.action == 'list':
            # Parse tags if provided
            tags = None
            if args.tags:
                tag_pairs = args.tags.split(',')
                tags = {}
                for pair in tag_pairs:
                    key, value = pair.split('=')
                    tags[key.strip()] = value.strip()
            
            models = manager.list_all_models(args.model_name, tags)
            
            if args.output_format == 'json':
                print(json.dumps(models, indent=2, default=str))
            else:
                manager.print_model_summary(models)
        
        elif args.action == 'details':
            if not args.model_name:
                print("Error: --model-name is required for details action")
                return
            
            details = manager.get_model_details(args.model_name, args.version)
            if details:
                if args.output_format == 'json':
                    print(json.dumps(details, indent=2, default=str))
                else:
                    print("\nMODEL DETAILS:")
                    print("=" * 50)
                    for key, value in details.items():
                        print(f"{key.capitalize().replace('_', ' ')}: {value}")
            else:
                print(f"Model '{args.model_name}' not found")
        
        elif args.action == 'download':
            if not args.model_name:
                print("Error: --model-name is required for download action")
                return
            
            model_path = manager.download_model(args.model_name, args.version, args.target_dir)
            if model_path:
                print(f"Model successfully downloaded to: {model_path}")
        
        elif args.action == 'compare':
            if not args.model_name:
                print("Error: --model-name is required for compare action")
                return
            
            comparison = manager.compare_model_versions(args.model_name)
            if comparison:
                if args.output_format == 'json':
                    print(json.dumps(comparison, indent=2, default=str))
                else:
                    print(f"\nMODEL VERSION COMPARISON: {args.model_name}")
                    print("=" * 60)
                    df = pd.DataFrame(comparison)
                    print(df.to_string(index=False))
            else:
                print(f"No versions found for model '{args.model_name}'")
    
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        print(f"Failed to execute action '{args.action}': {str(e)}")

if __name__ == "__main__":
    main()