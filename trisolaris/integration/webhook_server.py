"""
Webhook server for Google Docs notifications.
"""
from flask import Flask, request, jsonify
import os
import json
import logging
from typing import Dict, Any

from trisolaris.integration.google_docs_sync import GoogleDocsSyncService

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('webhook_server')

# Initialize Flask app
app = Flask(__name__)

# Initialize sync service
sync_service = GoogleDocsSyncService(
    doc_id=os.environ.get('GOOGLE_DOC_ID', '1fk0TkyC7xsKgw2yGV6lzTn99zU6u2SKuNU2RdYxMj9w'),
    credentials_path=os.environ.get('GOOGLE_CREDENTIALS_PATH', 'credentials.json'),
    update_interval=300  # 5 minutes
)

@app.route('/webhook', methods=['POST'])
def webhook():
    """Handle webhook notifications from Google Docs."""
    try:
        data = request.json
        
        # Check for Drive/Docs notifications
        if 'drive' in data or 'docs' in data:
            # Force refresh of document
            sync_service.fetch_document(force=True)
            
        return jsonify({"status": "success"})
        
    except Exception as e:
        logger.error(f"Error processing webhook: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/content', methods=['GET'])
def get_content():
    """Get the current document content."""
    content = sync_service.fetch_document()
    return jsonify({"content": content})

@app.route('/update', methods=['POST'])
def update_content():
    """Update the document content."""
    try:
        data = request.json
        content = data.get('content', '')
        position = data.get('position', 1)
        
        success = sync_service.update_document(content, position)
        
        if success:
            return jsonify({"status": "success"})
        else:
            return jsonify({"status": "error", "message": "Failed to update document"}), 500
            
    except Exception as e:
        logger.error(f"Error updating document: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

def start_server(host: str = '0.0.0.0', port: int = 5000):
    """Start the webhook server."""
    app.run(host=host, port=port)

if __name__ == '__main__':
    start_server() 