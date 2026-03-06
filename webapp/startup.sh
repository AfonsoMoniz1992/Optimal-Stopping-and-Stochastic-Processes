#!/bin/bash
# Azure App Service startup script for Streamlit.
# Set as the "Startup Command" in Azure Portal → App Service → Configuration → General settings.
# Or run directly: bash startup.sh

export PORT=${PORT:-8000}

python -m streamlit run app.py \
  --server.port "$PORT" \
  --server.address 0.0.0.0 \
  --server.headless true \
  --browser.gatherUsageStats false
