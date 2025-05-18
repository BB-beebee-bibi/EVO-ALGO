"""
Setup script for the TRISOLARIS project.
"""
import os
import sys
import argparse
import logging
import subprocess
from typing import Dict, Any, List
import json

def setup_project(args):
    """Set up the TRISOLARIS project."""
    # Create directories
    directories = [
        "trisolaris",
        "trisolaris/core",
        "trisolaris/core/meta_control",
        "trisolaris/core/population",
        "trisolaris/validation",
        "trisolaris/tasks",
        "trisolaris/integration",
        "tests",
        "data",
        "data/text_files"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        
    # Create __init__.py files
    for directory in directories:
        if directory.startswith("trisolaris"):
            init_file = os.path.join(directory, "__init__.py")
            if not os.path.exists(init_file):
                with open(init_file, "w") as f:
                    f.write("# TRISOLARIS Evolutionary Code Engine\n")
                    
    # Install dependencies
    if args.install_deps:
        dependencies = [
            "numpy>=1.21.0",
            "scipy>=1.7.0",
            "pytest>=7.0.0",
            "astor>=0.8.1",
            "flask>=2.0.0",
            "requests>=2.26.0",
            "google-api-python-client>=2.0.0",
            "google-auth>=2.0.0"
        ]
        
        print("Installing dependencies...")
        for dep in dependencies:
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
            
    # Set up Google Docs integration
    if args.setup_google_docs:
        setup_google_docs(args.credentials_path, args.doc_id)
        
    print("TRISOLARIS project set up successfully!")
    
def setup_google_docs(credentials_path, doc_id):
    """Set up Google Docs integration."""
    if credentials_path and os.path.exists(credentials_path):
        # Create configuration
        config = {
            "google_docs": {
                "doc_id": doc_id,
                "credentials_path": credentials_path,
                "update_interval": 300
            },
            "webhook_server": {
                "host": "0.0.0.0",
                "port": 5000
            }
        }
        
        # Save configuration
        with open("trisolaris_config.json", "w") as f:
            json.dump(config, f, indent=2)
            
        print(f"Google Docs integration configured with doc ID: {doc_id}")
    else:
        print("Warning: Google credentials file not found. Google Docs integration not configured.")
        
def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Set up the TRISOLARIS project")
    parser.add_argument("--install-deps", action="store_true", help="Install dependencies")
    parser.add_argument("--setup-google-docs", action="store_true", help="Set up Google Docs integration")
    parser.add_argument("--credentials-path", default="credentials.json", help="Path to Google API credentials")
    parser.add_argument("--doc-id", default="1fk0TkyC7xsKgw2yGV6lzTn99zU6u2SKuNU2RdYxMj9w", help="Google Doc ID")
    
    args = parser.parse_args()
    setup_project(args)
    
if __name__ == "__main__":
    main() 