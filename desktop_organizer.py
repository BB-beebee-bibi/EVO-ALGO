#!/usr/bin/env python3
"""
Desktop File Organizer
Organizes files on the desktop by their content type.
"""
import os
import shutil
from pathlib import Path
from typing import List, Dict, Set, Any, Optional

# Define file extensions for each category
CATEGORIES = {
    'images': {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'},
    'documents': {'.pdf', '.doc', '.docx', '.txt'},
    'audio': {'.mp3', '.wav', '.ogg', '.midi'},
    'video': {'.mp4', '.mov', '.avi', '.mkv'},
    'code': {'.py', '.java', '.cpp', '.c', '.js'},
    'archives': {'.zip', '.rar', '.7z'},
    'spreadsheets': {'.xlsx', '.xls', '.csv'},
    'presentations': {'.pptx', '.ppt'}
}

def get_file_type(file_path: str) -> str:
    """
    Determine the type of file based on its extension.
    Returns a category name like 'images', 'documents', etc.
    """
    ext = os.path.splitext(file_path)[1].lower()
    
    # Find the category for this extension
    for category, extensions in CATEGORIES.items():
        if ext in extensions:
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
        
        # Skip directories and our own script
        if os.path.isdir(file_path) or filename in ['desktop_organizer.py', 'create_test_files.py']:
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
    # Use current directory instead of desktop
    current_dir = os.getcwd()
    
    # Organize files
    organized_files = organize_files(current_dir)
    
    # Print results
    print("\nFile organization complete!")
    for category, files in organized_files.items():
        if files:
            print(f"\n{category.upper()}:")
            for file in files:
                print(f"  - {os.path.basename(file)}")

if __name__ == "__main__":
    main() 