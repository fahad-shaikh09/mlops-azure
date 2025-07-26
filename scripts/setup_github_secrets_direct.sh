#!/bin/bash

# Script to set up GitHub repository secrets for Azure ML operations
# Uses the service principal credentials we just created

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🔐 Setting up GitHub repository secrets for Azure ML...${NC}"

# Check if gh CLI is installed and authenticated
if ! command -v gh &> /dev/null; then
    echo -e "${RED}❌ GitHub CLI (gh) is not installed. Please install it first.${NC}"
    echo "Install with: brew install gh"
    exit 1
fi

if ! gh auth status &> /dev/null; then
    echo -e "${RED}❌ GitHub CLI is not authenticated. Please run 'gh auth login' first.${NC}"
    exit 1
fi

# Pre-configured Azure values from your service principal
AZURE_SUBSCRIPTION_ID=""
AZURE_RESOURCE_GROUP=""
AZURE_ML_WORKSPACE=""
AZURE_TENANT_ID=""
AZURE_CLIENT_ID=""
AZURE_CLIENT_SECRET=""

echo -e "${YELLOW}📋 Configuration to be set:${NC}"
echo "  Subscription ID: $AZURE_SUBSCRIPTION_ID"
echo "  Resource Group: $AZURE_RESOURCE_GROUP"
echo "  ML Workspace: $AZURE_ML_WORKSPACE"
echo "  Tenant ID: $AZURE_TENANT_ID"
echo "  Client ID: $AZURE_CLIENT_ID"
echo

# Set GitHub repository secrets
echo -e "${YELLOW}🔧 Setting GitHub repository secrets...${NC}"

secrets=(
    "AZURE_SUBSCRIPTION_ID:$AZURE_SUBSCRIPTION_ID"
    "AZURE_RESOURCE_GROUP:$AZURE_RESOURCE_GROUP"
    "AZURE_ML_WORKSPACE:$AZURE_ML_WORKSPACE"
    "AZURE_TENANT_ID:$AZURE_TENANT_ID"
    "AZURE_CLIENT_ID:$AZURE_CLIENT_ID"
    "AZURE_CLIENT_SECRET:$AZURE_CLIENT_SECRET"
)

for secret in "${secrets[@]}"; do
    IFS=':' read -r secret_name secret_value <<< "$secret"
    if gh secret set "$secret_name" -b "$secret_value"; then
        echo -e "${GREEN}✅ Set secret: $secret_name${NC}"
    else
        echo -e "${RED}❌ Failed to set secret: $secret_name${NC}"
        exit 1
    fi
done

echo
echo -e "${GREEN}🎉 All GitHub repository secrets have been set successfully!${NC}"
echo
echo -e "${YELLOW}📝 Secrets created:${NC}"
echo "  • AZURE_SUBSCRIPTION_ID"
echo "  • AZURE_RESOURCE_GROUP" 
echo "  • AZURE_ML_WORKSPACE"
echo "  • AZURE_TENANT_ID"
echo "  • AZURE_CLIENT_ID"
echo "  • AZURE_CLIENT_SECRET"
echo
echo -e "${YELLOW}💡 You can now use these secrets in your GitHub Actions workflows!${NC}"