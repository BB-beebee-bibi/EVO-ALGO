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
import setuptools

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
        
setuptools.setup(
    name="trisolaris",
    version="0.2.0",
    author="TriSolaris Team",
    author_email="team@example.com",
    description="Evolutionary Code Generation System",
    long_description="A system for evolving Python programs from natural language requests.",
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/trisolaris",  # Update as needed
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires=[
        "astor>=0.8.1",
        "numpy>=1.20.0",
        "pandas>=1.2.0",
        "scipy>=1.6.0",
        "hypothesis>=6.0.0",
    ],
    extras_require={
        'dev': [
            'pytest>=6.2',
            'flake8>=3.9',
            'mypy>=0.900',
            'black>=21.0b0',
            'tox>=3.20',
        ]
    },
)

# --- Custom logic for direct execution only ---
if __name__ == "__main__":
    import argparse
    import sys
    import json
    def custom_setup_logic(args):
        print("Running custom setup logic...")
        # Place any project-specific setup logic here
        # (e.g., Google Docs integration, directory creation, etc.)
        # This will NOT interfere with pip install
        # Example:
        if args.setup_google_docs:
            print("Would run Google Docs setup here.")
    parser = argparse.ArgumentParser(description="TriSolaris Project Setup & Management")
    parser.add_argument('--setup-google-docs', action='store_true', help='Set up Google Docs integration')
    args = parser.parse_args()
    if args.setup_google_docs:
        custom_setup_logic(args)
    else:
        print("setup.py executed directly. No custom logic triggered.") 