#!/usr/bin/env python3
"""
Desktop File Organizer
Organizes files on the desktop by their content type.
"""
import os
import shutil
from pathlib import Path
from typing import List, Dict

# Define file categories by extension
CATEGORIES = {
    'images': {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'},
    'documents': {'.pdf', '.doc', '.docx', '.txt', '.rtf'},
    'audio': {'.mp3', '.wav', '.ogg', '.midi', '.m4a'},
    'video': {'.mp4', '.mov', '.avi', '.mkv', '.wmv'},
    'code': {'.py', '.java', '.cpp', '.c', '.js', '.html', '.css'},
    'archives': {'.zip', '.rar', '.7z', '.tar', '.gz'},
    'spreadsheets': {'.xls', '.xlsx', '.csv'},
    'presentations': {'.ppt', '.pptx', '.key'}
}

def get_file_type(file_path: str) -> str:
    """Determine the type of file based on its extension."""
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    
    for category, extensions in CATEGORIES.items():
        if ext in extensions:
            return category
    return 'unknown'

def create_category_dirs(base_path: str) -> None:
    """Create directories for each category."""
    for category in CATEGORIES.keys():
        category_path = os.path.join(base_path, category)
        os.makedirs(category_path, exist_ok=True)
    os.makedirs(os.path.join(base_path, 'unknown'), exist_ok=True)

def organize_files(desktop_path: str) -> Dict[str, List[str]]:
    """Organize files into categories."""
    organized_files = {category: [] for category in CATEGORIES.keys()}
    organized_files['unknown'] = []
    
    create_category_dirs(desktop_path)
    
    for filename in os.listdir(desktop_path):
        file_path = os.path.join(desktop_path, filename)
        
        if os.path.isdir(file_path):
            continue
            
        category = get_file_type(file_path)
        target_dir = os.path.join(desktop_path, category)
        target_path = os.path.join(target_dir, filename)
        
        shutil.move(file_path, target_path)
        organized_files[category].append(target_path)
    
    return organized_files

def main():
    """Main function to organize desktop files."""
    desktop_path = os.path.expanduser("~/Desktop")
    organized_files = organize_files(desktop_path)
    
    print("
File organization complete!")
    for category, files in organized_files.items():
        if files:
            print(f"
{category.upper()}:")
            for file in files:
                print(f"  - {os.path.basename(file)}")

if __name__ == "__main__":
    main()
