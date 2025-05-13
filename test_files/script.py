#!/usr/bin/env python3
"""
Simple file organization script for testing purposes.
"""
import os
import shutil
from pathlib import Path
import mimetypes

def organize_by_type(directory):
    """Organize files by their MIME type."""
    # Get all files in the directory
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    
    # Create category directories
    categories = {
        'text': os.path.join(directory, 'text'),
        'image': os.path.join(directory, 'image'),
        'application': os.path.join(directory, 'application'),
        'other': os.path.join(directory, 'other')
    }
    
    for category in categories.values():
        os.makedirs(category, exist_ok=True)
    
    # Move files to appropriate directories
    for file in files:
        if file == os.path.basename(__file__):
            continue  # Skip this script
            
        file_path = os.path.join(directory, file)
        mime_type, _ = mimetypes.guess_type(file_path)
        
        if mime_type:
            category = mime_type.split('/')[0]
            if category in categories:
                dest_dir = categories[category]
            else:
                dest_dir = categories['other']
        else:
            dest_dir = categories['other']
            
        dest_path = os.path.join(dest_dir, file)
        shutil.copy2(file_path, dest_path)
        print(f"Copied {file} to {dest_dir}")

if __name__ == "__main__":
    organize_by_type(os.path.dirname(os.path.abspath(__file__)))
    print("File organization complete!")