#!/bin/bash
# launch_webhook_server.sh

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Set environment variables
export GOOGLE_DOC_ID="1fk0TkyC7xsKgw2yGV6lzTn99zU6u2SKuNU2RdYxMj9w"
export GOOGLE_CREDENTIALS_PATH="credentials.json"

# Launch webhook server
python -m trisolaris.integration.webhook_server 