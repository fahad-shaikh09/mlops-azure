# Example Setup Guide

This file shows you exactly what values to replace in the configuration files.

## 1. Update config/workspace_config.json

Replace the template:
```json
{
    "subscription_id": "",
    "resource_group": "",
    "workspace_name": "",
    "location": "East US"
}
```

With your actual values:
```json
{
    "subscription_id": "12345678-1234-1234-1234-123456789012",
    "resource_group": "my-ml-resource-group",
    "workspace_name": "my-ml-workspace",
    "location": "East US"
}
```

## 2. GitHub Secrets Setup

In your GitHub repository settings, add these secrets:

| Secret Name | Example Value | Description |
|-------------|---------------|-------------|
| `AZURE_SUBSCRIPTION_ID` | `12345678-1234-1234-1234-123456789012` | Your Azure subscription ID |
| `AZURE_RESOURCE_GROUP` | `my-ml-resource-group` | Your resource group name |
| `AZURE_ML_WORKSPACE` | `my-ml-workspace` | Your Azure ML workspace name |
| `AZURE_CREDENTIALS` | `{"clientId": "...", "clientSecret": "...", ...}` | Complete service principal JSON |

## 3. Customize for Your Use Case

### Training Data
- Update `src/data_preprocessing.py` to load your data
- Modify `data/raw_data.csv` path or data source

### Model Configuration  
- Edit `src/train.py` model parameters
- Adjust hyperparameters in `azure_ml/train_job.py`

### Deployment Settings
- Modify `deployment/deployment_config_production.yml`
- Update health check endpoints in `src/score.py`

## 4. Cost Optimization

The pipeline automatically selects the most cost-effective Azure ML instance types:
- `Standard_F2s_v2` (2 vCPUs, 4GB) - Most economical for inference
- `Standard_DS2_v2` (2 vCPUs, 7GB) - Good for training workloads

You can override this by editing `deployment/check_quota_and_instances.py`.

## 5. Testing Your Setup

```bash
# Validate configuration
python scripts/validate_workspace_config.py

# Test quota optimization
python deployment/check_quota_and_instances.py

# Run local tests
python -m pytest tests/ -v
```

## 6. Production Deployment

Once everything is configured:
1. Push to `main` branch to trigger production deployment
2. Monitor the GitHub Actions workflow
3. Check Azure ML Studio for deployment status
4. Test the deployed endpoint

## Security Notes

- Never commit actual credentials to git
- Use GitHub Secrets for all sensitive information
- Regularly rotate service principal credentials
- Monitor access logs in Azure