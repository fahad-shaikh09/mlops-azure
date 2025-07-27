# MLOps Pipeline with Azure ML and GitHub Actions

A production-ready MLOps pipeline using Azure Machine Learning and GitHub Actions for automated training, deployment, and monitoring with cost optimization and quota management.

## 🌟 Key Features

- **Azure ML SDK v2** support with managed endpoints
- **Automated quota management** and cost optimization
- **Azure ML supported instance types** selection
- **Comprehensive CI/CD** with GitHub Actions
- **Production-ready deployment** with health checks
- **Security-first** approach with no hardcoded secrets

## 🏗️ Architecture Overview

```
├── src/                              # Source code
│   ├── data_preprocessing.py         # Data preprocessing pipeline
│   ├── train.py                     # Model training script
│   ├── azure_ml_train.py           # Azure ML training script
│   └── score.py                     # Model scoring script (SDK v2)
├── deployment/                       # Deployment configurations  
│   ├── deploy_managed_endpoint.py   # Azure ML managed endpoints (SDK v2)
│   ├── check_quota_and_instances.py # Quota optimization script
│   ├── deployment_config_production.yml # Production deployment config
│   └── environment.yml             # Conda environment
├── azure_ml/                        # Azure ML specific scripts
│   ├── train_job.py                # Azure ML training job submission
│   └── model_management.py         # Model management utilities
├── monitoring/                       # Monitoring and drift detection
│   ├── data_drift_detector.py      # Data drift monitoring
│   └── model_monitor.py            # Model performance monitoring
├── tests/                           # Comprehensive test suites
│   ├── test_data_validation.py     # Data validation tests
│   ├── test_model_validation.py    # Model validation tests
│   ├── test_integration.py         # Integration tests
│   └── test_smoke.py              # Smoke tests
├── .github/workflows/              # GitHub Actions workflows
│   └── ml-pipeline.yml            # Complete ML pipeline with quota management
├── config/                         # Configuration templates
│   └── workspace_config.json      # Azure ML workspace config template
└── scripts/                        # Utility scripts
    ├── create_workspace_config.py  # Workspace configuration script
    └── validate_workspace_config.py # Configuration validation
```

## 🚀 Features

- **Azure ML SDK v2**: Modern managed endpoints deployment
- **Quota Optimization**: Automatic quota analysis and cost-effective instance selection
- **Instance Type Management**: Uses only Azure ML supported instance types
- **Automated Training**: End-to-end model training with Azure ML compute clusters
- **Managed Endpoints**: Production-ready deployments with health checks
- **Cost Optimization**: Intelligent selection of cheapest suitable VM sizes
- **Data Validation**: Comprehensive data quality and schema validation
- **Model Monitoring**: Performance tracking and drift detection
- **CI/CD Pipeline**: Complete GitHub Actions workflow with cleanup
- **Security**: No hardcoded secrets, environment-based configuration

## 📋 Prerequisites

1. **Azure Account** with active subscription
2. **Azure ML Workspace** created
3. **Service Principal** with appropriate permissions
4. **GitHub Repository** with Actions enabled
5. **GitHub CLI** installed (for setup script)

## 🛠️ Quick Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/mlops-azure.git
cd mlops-azure
```

### 2. Set Up Azure Resources

Create an Azure ML workspace and service principal:

```bash
# Create resource group
az group create --name your-resource-group --location eastus

# Create Azure ML workspace
az ml workspace create --name your-workspace --resource-group your-resource-group

# Create service principal for GitHub Actions
az ad sp create-for-rbac --name "github-mlops-sp" \
  --role contributor \
  --scopes /subscriptions/<your-subscription-id>/resourceGroups/<your-resource-group> \
  --sdk-auth
```

### 3. Configure GitHub Secrets

In your GitHub repository, go to Settings > Secrets and Variables > Actions and add:

**Required Secrets:**
- `AZURE_CREDENTIALS`: Complete JSON output from service principal creation
- `AZURE_SUBSCRIPTION_ID`: Your Azure subscription ID  
- `AZURE_RESOURCE_GROUP`: Your resource group name
- `AZURE_ML_WORKSPACE`: Your Azure ML workspace name

**Example AZURE_CREDENTIALS format:**
```json
{
  "clientId": "your-client-id",
  "clientSecret": "your-client-secret", 
  "subscriptionId": "your-subscription-id",
  "tenantId": "your-tenant-id"
}
```

### 4. Update Configuration Files

Update `config/workspace_config.json` with your Azure details:

```json
{
    "subscription_id": "your-subscription-id",
    "resource_group": "your-resource-group", 
    "workspace_name": "your-workspace-name",
    "location": "eastus"
}
```

**⚠️ Security Note**: Never commit actual credentials to git. The template files contain placeholders only.

## 💰 Cost Optimization & Quota Management

This pipeline includes intelligent cost optimization features:

### Automated Instance Selection
- **Quota Analysis**: Checks current Azure ML compute usage
- **Cost Ranking**: Ranks VM sizes by cost-effectiveness
- **Smart Selection**: Automatically chooses cheapest suitable instance type
- **Azure ML Compatibility**: Uses only supported instance types

### Supported Instance Types (Cost-Optimized)
```
Standard_F2s_v2   → 2 vCPUs,  4 GB RAM  (Most cost-effective)
Standard_DS2_v2   → 2 vCPUs,  7 GB RAM  (Good value)
Standard_D2s_v3   → 2 vCPUs,  8 GB RAM  (Latest generation)
Standard_F4s_v2   → 4 vCPUs,  8 GB RAM  (Higher performance)
...
```

### Quota Management Scripts
```bash
# Check current quota usage and get recommendations
python deployment/check_quota_and_instances.py

# Clean up resources to free quota
python deployment/deploy_managed_endpoint.py --action cleanup
```

## 🔄 Pipeline Workflows

### Main ML Pipeline (`ml-pipeline.yml`)

Triggered on:
- Push to `main` or `develop` branches  
- Pull requests to `main`
- Manual dispatch

**Enhanced Stages:**
1. **Data Validation**: Validates input data quality and schema
2. **Resource Cleanup**: Frees quota by cleaning up old endpoints
3. **Azure ML Training**: Submits training job to remote compute cluster
4. **Model Registration**: Automatically registers model in Azure ML Registry
5. **Quota Analysis**: Analyzes quota and selects optimal instance types
6. **Staging Deployment**: Deploys from registry using managed endpoints
7. **Integration Tests**: Runs end-to-end integration tests
8. **Production Deployment**: Deploys to production (main branch only)
9. **Smoke Tests**: Basic production functionality tests

### ⚡ **Key Features:**
- **Remote Training**: Models trained on Azure ML compute clusters (not locally)
- **Registry-Based Deployment**: Models deployed directly from Azure ML Registry
- **Quota Management**: Automatic cleanup and resource optimization
- **Cost Optimization**: Intelligent selection of cheapest suitable VMs
- **Azure ML SDK v2**: Modern managed endpoints with health checks
- **Zero Downtime**: Rolling deployments with traffic management
- **No Local Artifacts**: All models stored and managed in Azure ML Registry

### Monitoring Pipeline (`model-monitoring.yml`)

Runs daily at 2 AM UTC:
- **Data Drift Detection**: Monitors for changes in input data distribution
- **Model Performance Monitoring**: Tracks model accuracy and performance metrics
- **Alert Generation**: Sends notifications for detected issues

## 📊 Monitoring and Alerting

### Data Drift Detection

The pipeline monitors for:
- Statistical drift using Kolmogorov-Smirnov tests
- Population Stability Index (PSI) scores
- Categorical variable distribution changes

### Model Performance Monitoring

Tracks:
- Accuracy, precision, recall, F1-score
- Response times and error rates
- Resource utilization (CPU, memory)
- Prediction confidence scores

### Thresholds

Default alert thresholds:
- Performance degradation: >10% drop from baseline
- PSI score: >0.2 indicates significant drift
- Error rate: >3% triggers alert
- CPU usage: >80% triggers resource alert

## 🎯 **Where to Find Your Models**

Your trained models are stored in **Azure ML Model Registry** and can be accessed through:

### **Azure ML Studio** (Web UI)
- Navigate to `https://ml.azure.com`
- Go to "Models" → Find your model (`trained_model`)
- View metrics, versions, and deployment status

### **Python SDK**
```python
from azureml.core import Workspace, Model
ws = Workspace.from_config()
model = Model(ws, name='trained_model')
print(f"Model: {model.name} v{model.version}")
```

### **Management Scripts**
```bash
# List all models
python azure_ml/model_management.py --action list

# Download a model
./scripts/quick_model_access.sh --action download --model trained_model

# Deploy a model
./scripts/quick_model_access.sh --action deploy --model trained_model --service my-service
```

### **Model Locations**
- **Training**: Azure ML compute clusters (`cpu-cluster`)
- **Storage**: Azure ML Model Registry (versioned)
- **Staging**: `trained_model-staging` service
- **Production**: `trained_model-production` service

📚 **Detailed guide**: [docs/MODEL_LOCATIONS.md](docs/MODEL_LOCATIONS.md)

## 🧪 Testing Strategy

### Test Types

1. **Data Validation Tests** (`test_data_validation.py`)
   - Schema validation
   - Data quality checks
   - Outlier detection
   - Feature correlation analysis

2. **Model Validation Tests** (`test_model_validation.py`)
   - Model file integrity
   - Performance thresholds
   - Prediction format validation
   - Edge case handling

3. **Integration Tests** (`test_integration.py`)
   - End-to-end API testing
   - Endpoint health checks
   - Concurrent request handling
   - Response time validation

4. **Smoke Tests** (`test_smoke.py`)
   - Basic functionality verification
   - Production deployment validation

### Running Tests Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run all tests
python -m pytest tests/ -v

# Run specific test suite
python -m pytest tests/test_data_validation.py -v
```

## 🔧 Local Development

### Environment Setup

```bash
# Create conda environment
conda env create -f deployment/environment.yml
conda activate model-env

# Or use pip
pip install -r requirements.txt
```

### Data Preprocessing

```bash
python src/data_preprocessing.py
```

### Model Training

```bash
python src/train.py --model-type random_forest --n-estimators 100
```

### Local Deployment Testing

```bash
cd deployment
python deploy.py
```

## 📈 Performance Optimization

### Model Training

- **Hyperparameter Tuning**: Extend `train.py` with automated hyperparameter optimization
- **Feature Selection**: Implement automated feature selection in preprocessing
- **Model Ensembling**: Combine multiple models for better performance

### Deployment Optimization

- **Auto-scaling**: Configure auto-scaling based on request volume
- **Model Caching**: Implement response caching for frequently requested predictions
- **Load Balancing**: Use Azure Load Balancer for high availability

### Monitoring Optimization

- **Real-time Alerts**: Integrate with Azure Monitor for real-time alerting
- **Custom Metrics**: Add business-specific metrics tracking
- **Dashboard Integration**: Connect to Power BI or Grafana for visualization

## 🔒 Security & Privacy

### Security Features
- **Zero Hardcoded Secrets**: All credentials managed via GitHub Secrets
- **Environment-Based Config**: Configuration loaded from environment variables
- **Service Principal Authentication**: Secure Azure authentication
- **Least Privilege Access**: RBAC with minimal required permissions
- **Audit Trail**: All operations logged for compliance

### What's Not Included (On Purpose)
❌ No actual credentials or API keys  
❌ No hardcoded subscription IDs  
❌ No production data or models  
❌ No personal/company-specific configurations  

### Making It Production-Ready
1. **Replace placeholders** in `config/workspace_config.json`
2. **Set up GitHub Secrets** as described in setup section
3. **Configure your data sources** in the training scripts
4. **Customize model parameters** for your use case
5. **Set up monitoring** for your specific metrics

## 🚨 Troubleshooting

### Common Issues

1. **Authentication Errors**
   - Verify service principal credentials
   - Check RBAC permissions on Azure ML workspace

2. **Deployment Failures**
   - Check Azure ML compute availability
   - Verify Docker image dependencies

3. **Test Failures**
   - Ensure test data is available
   - Check endpoint accessibility

4. **Monitoring Issues**
   - Verify data sources are accessible
   - Check monitoring thresholds configuration

### Debug Mode

Enable debug logging by setting environment variable:
```bash
export LOGGING_LEVEL=DEBUG
```

## 📝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with appropriate tests
4. Submit a pull request

### Code Standards

- Follow PEP 8 for Python code
- Add docstrings to all functions
- Include unit tests for new features
- Update README for significant changes

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Support

For issues and questions:
1. Check the troubleshooting section
2. Review existing GitHub issues
3. Create a new issue with detailed description

## 🔗 Useful Links

- [Azure Machine Learning Documentation](https://docs.microsoft.com/en-us/azure/machine-learning/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [MLOps Best Practices](https://docs.microsoft.com/en-us/azure/machine-learning/concept-ml-pipelines)
- [Azure ML Python SDK](https://docs.microsoft.com/en-us/python/api/overview/azure/ml/)

---

**Note**: This is a template repository. Customize the code, configurations, and documentation according to your specific use case and requirements.