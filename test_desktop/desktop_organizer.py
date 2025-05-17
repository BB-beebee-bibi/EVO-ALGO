#!/usr/bin/env python3
"""
Desktop File Organizer
Organizes files on the desktop by their content type.
"""
import os
import shutil
from pathlib import Path
import mimetypes
import magic  # python-magic library for better file type detection
from typing import List, Dict, Set, Any, Optional

# Initialize mimetypes
mimetypes.init()

# Define common file categories
CATEGORIES = {
    'images': {'image/jpeg', 'image/png', 'image/gif', 'image/bmp', 'image/webp'},
    'documents': {'application/pdf', 'application/msword', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'text/plain'},
    'audio': {'audio/mpeg', 'audio/wav', 'audio/ogg', 'audio/midi'},
    'video': {'video/mp4', 'video/quicktime', 'video/x-msvideo', 'video/x-matroska'},
    'code': {'text/x-python', 'text/x-java', 'text/x-c++', 'text/x-c', 'text/javascript'},
    'archives': {'application/zip', 'application/x-rar-compressed', 'application/x-7z-compressed'},
    'spreadsheets': {'application/vnd.ms-excel', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'},
    'presentations': {'application/vnd.ms-powerpoint', 'application/vnd.openxmlformats-officedocument.presentationml.presentation'}
}

def get_file_type(file_path: str) -> str:
    """
    Determine the type of file based on its content.
    Returns a category name like 'images', 'documents', etc.
    """
    try:
        # Try using python-magic first
        mime = magic.from_file(file_path, mime=True)
    except Exception:
        # Fall back to mimetypes
        mime, _ = mimetypes.guess_type(file_path)
        if not mime:
            return 'unknown'
    
    # Find the category for this mime type
    for category, mime_types in CATEGORIES.items():
        if mime in mime_types:
            return category
    
    return 'unknown'

def create_category_dirs(base_path: str, categories: List[str]) -> None:
    """
    Create directories for each category if they don't exist.
    """
    for category in categories:
        category_path = os.path.join(base_path, category)
        if not os.path.exists(category_path):
            os.makedirs(category_path)

def organize_files(desktop_path: str) -> Dict[str, List[str]]:
    """
    Organize files on the desktop into categories.
    Returns a dictionary mapping categories to lists of files.
    """
    # Initialize result dictionary
    organized_files = {category: [] for category in CATEGORIES.keys()}
    organized_files['unknown'] = []
    
    # Create category directories
    create_category_dirs(desktop_path, list(CATEGORIES.keys()) + ['unknown'])
    
    # Process each file
    for filename in os.listdir(desktop_path):
        file_path = os.path.join(desktop_path, filename)
        
        # Skip directories
        if os.path.isdir(file_path):
            continue
        
        # Get file category
        category = get_file_type(file_path)
        
        # Move file to appropriate directory
        target_dir = os.path.join(desktop_path, category)
        target_path = os.path.join(target_dir, filename)
        
        try:
            shutil.move(file_path, target_path)
            organized_files[category].append(target_path)
        except Exception as e:
            print(f"Error moving {filename}: {e}")
    
    return organized_files

def main():
    """
    Main function to organize desktop files.
    """
    # Get desktop path
    desktop_path = os.path.expanduser("~/Desktop")
    
    # Organize files
    organized_files = organize_files(desktop_path)
    
    # Print results
    print("\nFile organization complete!")
    for category, files in organized_files.items():
        if files:
            print(f"\n{category.upper()}:")
            for file in files:
                print(f"  - {os.path.basename(file)}")

if __name__ == "__main__":
    main()
