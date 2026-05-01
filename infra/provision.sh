#!/usr/bin/env bash
# infra/provision.sh
#
# Provisions every Azure resource needed by the Multi-Agent Stock AI system.
#
# Prerequisites:
#   - Azure CLI installed  →  https://learn.microsoft.com/en-us/cli/azure/install-azure-cli
#   - Logged in           →  az login
#   - openssl available   →  brew install openssl  (macOS) or apt install openssl
#
# Usage:
#   chmod +x infra/provision.sh
#   ./infra/provision.sh [project-prefix]   # default prefix: stockai
#
# The script writes a .env file at the repo root on completion.

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────────
PREFIX="${1:-stockai}"
LOCATION="eastus"                         # Azure OpenAI is available in eastus
RG="${PREFIX}-rg"

POSTGRES_SERVER="${PREFIX}-pg"
POSTGRES_DB="stockai"

OPENAI_ACCOUNT="${PREFIX}-openai"
OPENAI_DEPLOYMENT="gpt-4o"
OPENAI_MODEL_VERSION="2024-11-20"

SERVICEBUS_NS="${PREFIX}-sb"
SERVICEBUS_QUEUE="pipeline-events"

KEYVAULT="${PREFIX}-kv"

# ACR names: globally unique, lowercase alphanumeric only (no hyphens)
ACR="${PREFIX}acr$(openssl rand -hex 3)"

LOG_ANALYTICS="${PREFIX}-logs"
CONTAINERAPPS_ENV="${PREFIX}-cae"

# ── Terminal colours ───────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; BOLD='\033[1m'; NC='\033[0m'

info()    { echo -e "${BLUE}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }
header()  { echo -e "\n${BOLD}── $* ──${NC}"; }

# ── Preflight checks ───────────────────────────────────────────────────────────
header "Preflight"

command -v az      &>/dev/null || error "Azure CLI not found. See: https://learn.microsoft.com/en-us/cli/azure/install-azure-cli"
command -v openssl &>/dev/null || error "openssl not found. Install via brew or apt."

az account show &>/dev/null || error "Not logged in to Azure. Run: az login"

SUBSCRIPTION=$(az account show --query id   -o tsv)
TENANT=$(az account show       --query tenantId -o tsv)

info "Subscription : $SUBSCRIPTION"
info "Tenant       : $TENANT"
info "Prefix       : $PREFIX"
info "Location     : $LOCATION"
info "Resource Grp : $RG"

echo ""
read -rp "Proceed with provisioning? [y/N] " REPLY
echo ""
[[ $REPLY =~ ^[Yy]$ ]] || { info "Aborted."; exit 0; }

# ── Resource Group ─────────────────────────────────────────────────────────────
header "Resource Group"
az group create \
  --name "$RG" \
  --location "$LOCATION" \
  --output none
success "Resource group: $RG"

# ── PostgreSQL Flexible Server ─────────────────────────────────────────────────
header "PostgreSQL Flexible Server"

POSTGRES_ADMIN_USER="${PREFIX}admin"
# Generate a 24-char password with letters, digits, and safe specials
POSTGRES_ADMIN_PASSWORD=$(openssl rand -base64 32 | tr -dc 'A-Za-z0-9!@#%' | head -c 24)

info "Creating server $POSTGRES_SERVER (this takes ~3-5 minutes)..."
az postgres flexible-server create \
  --resource-group  "$RG" \
  --name            "$POSTGRES_SERVER" \
  --location        "$LOCATION" \
  --admin-user      "$POSTGRES_ADMIN_USER" \
  --admin-password  "$POSTGRES_ADMIN_PASSWORD" \
  --sku-name        "Standard_B1ms" \
  --tier            "Burstable" \
  --storage-size    32 \
  --version         "16" \
  --public-access   "0.0.0.0" \
  --output none

az postgres flexible-server db create \
  --resource-group "$RG" \
  --server-name    "$POSTGRES_SERVER" \
  --database-name  "$POSTGRES_DB" \
  --output none

POSTGRES_HOST="${POSTGRES_SERVER}.postgres.database.azure.com"
success "PostgreSQL: $POSTGRES_HOST  (db: $POSTGRES_DB, user: $POSTGRES_ADMIN_USER)"

# ── Azure OpenAI ───────────────────────────────────────────────────────────────
header "Azure OpenAI"

info "Creating Cognitive Services account: $OPENAI_ACCOUNT"
az cognitiveservices account create \
  --name           "$OPENAI_ACCOUNT" \
  --resource-group "$RG" \
  --location       "$LOCATION" \
  --kind           "OpenAI" \
  --sku            "S0" \
  --output none

info "Deploying model: $OPENAI_DEPLOYMENT ($OPENAI_MODEL_VERSION)"
az cognitiveservices account deployment create \
  --name            "$OPENAI_ACCOUNT" \
  --resource-group  "$RG" \
  --deployment-name "$OPENAI_DEPLOYMENT" \
  --model-name      "gpt-4o" \
  --model-version   "$OPENAI_MODEL_VERSION" \
  --model-format    "OpenAI" \
  --sku-capacity    10 \
  --sku-name        "Standard" \
  --output none

OPENAI_ENDPOINT=$(az cognitiveservices account show \
  --name "$OPENAI_ACCOUNT" --resource-group "$RG" \
  --query "properties.endpoint" -o tsv)

OPENAI_KEY=$(az cognitiveservices account keys list \
  --name "$OPENAI_ACCOUNT" --resource-group "$RG" \
  --query "key1" -o tsv)

success "Azure OpenAI: $OPENAI_ENDPOINT (deployment: $OPENAI_DEPLOYMENT)"

# ── Azure Service Bus ──────────────────────────────────────────────────────────
header "Azure Service Bus"

az servicebus namespace create \
  --name           "$SERVICEBUS_NS" \
  --resource-group "$RG" \
  --location       "$LOCATION" \
  --sku            "Basic" \
  --output none

az servicebus queue create \
  --name            "$SERVICEBUS_QUEUE" \
  --namespace-name  "$SERVICEBUS_NS" \
  --resource-group  "$RG" \
  --output none

SB_CONN=$(az servicebus namespace authorization-rule keys list \
  --name            "RootManageSharedAccessKey" \
  --namespace-name  "$SERVICEBUS_NS" \
  --resource-group  "$RG" \
  --query           "primaryConnectionString" -o tsv)

success "Service Bus: $SERVICEBUS_NS  queue: $SERVICEBUS_QUEUE"

# ── Azure Key Vault ────────────────────────────────────────────────────────────
header "Azure Key Vault"

CURRENT_USER_OID=$(az ad signed-in-user show --query id -o tsv)

az keyvault create \
  --name                    "$KEYVAULT" \
  --resource-group          "$RG" \
  --location                "$LOCATION" \
  --enable-rbac-authorization false \
  --output none

az keyvault set-policy \
  --name         "$KEYVAULT" \
  --object-id    "$CURRENT_USER_OID" \
  --secret-permissions get set list delete purge \
  --output none

# Store all secrets centrally
az keyvault secret set --vault-name "$KEYVAULT" --name "DB-HOST"           --value "$POSTGRES_HOST"           --output none
az keyvault secret set --vault-name "$KEYVAULT" --name "DB-USER"           --value "$POSTGRES_ADMIN_USER"     --output none
az keyvault secret set --vault-name "$KEYVAULT" --name "DB-PASSWORD"       --value "$POSTGRES_ADMIN_PASSWORD" --output none
az keyvault secret set --vault-name "$KEYVAULT" --name "AZURE-OPENAI-KEY"  --value "$OPENAI_KEY"              --output none
az keyvault secret set --vault-name "$KEYVAULT" --name "SB-CONN-STRING"    --value "$SB_CONN"                 --output none

KV_URL="https://${KEYVAULT}.vault.azure.net/"
success "Key Vault: $KV_URL  (5 secrets stored)"

# ── Azure Container Registry ───────────────────────────────────────────────────
header "Azure Container Registry  (used in Phase 7)"

az acr create \
  --name           "$ACR" \
  --resource-group "$RG" \
  --sku            "Basic" \
  --admin-enabled  true \
  --output none

ACR_LOGIN_SERVER=$(az acr show --name "$ACR" --resource-group "$RG" --query loginServer -o tsv)
ACR_PASSWORD=$(az acr credential show --name "$ACR" --query "passwords[0].value" -o tsv)

success "ACR: $ACR_LOGIN_SERVER"

# ── Log Analytics Workspace ────────────────────────────────────────────────────
header "Log Analytics  (for Container Apps + App Insights)"

az monitor log-analytics workspace create \
  --workspace-name "$LOG_ANALYTICS" \
  --resource-group "$RG" \
  --location       "$LOCATION" \
  --output none

LOG_ANALYTICS_ID=$(az monitor log-analytics workspace show \
  --workspace-name "$LOG_ANALYTICS" --resource-group "$RG" --query customerId -o tsv)
LOG_ANALYTICS_KEY=$(az monitor log-analytics workspace get-shared-keys \
  --workspace-name "$LOG_ANALYTICS" --resource-group "$RG" --query primarySharedKey -o tsv)

success "Log Analytics workspace: $LOG_ANALYTICS"

# ── Container Apps Environment ─────────────────────────────────────────────────
header "Container Apps Environment  (used in Phase 7)"

az containerapp env create \
  --name                  "$CONTAINERAPPS_ENV" \
  --resource-group        "$RG" \
  --location              "$LOCATION" \
  --logs-workspace-id     "$LOG_ANALYTICS_ID" \
  --logs-workspace-key    "$LOG_ANALYTICS_KEY" \
  --output none

success "Container Apps Env: $CONTAINERAPPS_ENV"

# ── Write .env ─────────────────────────────────────────────────────────────────
header "Writing .env"

cat > .env <<EOF
# ── App ────────────────────────────────────────────────────────────────────────
APP_ENV=development
LOG_LEVEL=INFO

# ── Azure OpenAI ───────────────────────────────────────────────────────────────
AZURE_OPENAI_ENDPOINT=${OPENAI_ENDPOINT}
AZURE_OPENAI_API_KEY=${OPENAI_KEY}
AZURE_OPENAI_DEPLOYMENT=${OPENAI_DEPLOYMENT}
AZURE_OPENAI_API_VERSION=2024-02-15-preview

# ── Azure PostgreSQL ───────────────────────────────────────────────────────────
DB_HOST=${POSTGRES_HOST}
DB_PORT=5432
DB_NAME=${POSTGRES_DB}
DB_USER=${POSTGRES_ADMIN_USER}
DB_PASSWORD=${POSTGRES_ADMIN_PASSWORD}
DB_SSL_MODE=require

# ── Azure Service Bus ──────────────────────────────────────────────────────────
AZURE_SERVICEBUS_CONNECTION_STRING=${SB_CONN}
AZURE_SERVICEBUS_QUEUE_NAME=${SERVICEBUS_QUEUE}

# ── Azure Key Vault ────────────────────────────────────────────────────────────
AZURE_KEYVAULT_URL=${KV_URL}

# ── Azure Monitor / App Insights ──────────────────────────────────────────────
# Create an App Insights resource in the portal, then paste the connection string here.
APPLICATIONINSIGHTS_CONNECTION_STRING=

# ── External APIs ──────────────────────────────────────────────────────────────
COINGECKO_API_KEY=
ALPHA_VANTAGE_API_KEY=

# ── Ingestion config ───────────────────────────────────────────────────────────
STOCK_SYMBOLS=AAPL,MSFT,GOOGL,TSLA,NVDA
CRYPTO_SYMBOLS=bitcoin,ethereum,solana,cardano
INGESTION_INTERVAL_MINUTES=15

# ── Azure Container Registry (Phase 7) ────────────────────────────────────────
# ACR_LOGIN_SERVER=${ACR_LOGIN_SERVER}
# ACR_PASSWORD=${ACR_PASSWORD}
EOF

success ".env written to repo root"

# ── Summary ────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}${BOLD}════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}${BOLD}  Azure Provisioning Complete                       ${NC}"
echo -e "${GREEN}${BOLD}════════════════════════════════════════════════════${NC}"
echo ""
printf "  %-30s %s\n" "Resource Group:"       "$RG"
printf "  %-30s %s\n" "PostgreSQL Host:"      "$POSTGRES_HOST"
printf "  %-30s %s\n" "Azure OpenAI:"         "$OPENAI_ENDPOINT"
printf "  %-30s %s\n" "Service Bus:"          "$SERVICEBUS_NS"
printf "  %-30s %s\n" "Key Vault:"            "$KV_URL"
printf "  %-30s %s\n" "Container Registry:"   "$ACR_LOGIN_SERVER"
printf "  %-30s %s\n" "Container Apps Env:"   "$CONTAINERAPPS_ENV"
echo ""
echo -e "${YELLOW}  ⚠  .env contains secrets — it is already in .gitignore. Never commit it.${NC}"
echo ""
echo "  Next steps:"
echo "    1. pip install -r requirements.txt"
echo "    2. python main.py status"
echo "    3. Proceed to Phase 1 (Ingestion Agent)"
echo ""
