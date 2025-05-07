"""
USB Scanner Template

This template provides a basic structure for the USB scanner code to be evolved.
"""

import os
import platform
import datetime
from typing import List, Dict, Any

def find_usb_drives() -> List[str]:
    """
    Find USB drives connected to the system.
    
    Returns:
        List of drive letters for USB drives
    """
    # Example implementation for finding USB drives on Windows
    drives = []
    
    # Try to list all drives and find removable ones
    if platform.system() == "Windows":
        # Add at least drive E which we know is a USB drive
        drives.append("E")
        
        # You could also try to detect all removable drives
        # using commands like wmic or checking drive types
        
    return drives

def get_drive_info(drive_letter: str) -> Dict[str, Any]:
    """
    Get information about a drive including available space, type, etc.
    
    Args:
        drive_letter: Letter of the drive (e.g., "E")
        
    Returns:
        Dictionary with drive information
    """
    # Example implementation for getting drive info
    drive_info = {
        "path": drive_letter,
        "type": "removable"
    }
    
    # Try to get more information about the drive
    drive_path = f"{drive_letter}:"
    try:
        # Try to get drive statistics like free space and total size
        if os.path.exists(drive_path):
            drive_info["exists"] = True
            
            # Add more drive information if possible
            # For example, free space and total size
    except Exception as e:
        drive_info["error"] = str(e)
    
    return drive_info

def scan_directory(dir_path: str) -> Dict[str, Any]:
    """
    Scan a directory and return its structure.
    
    Args:
        dir_path: Path to the directory
        
    Returns:
        Dictionary with directory structure
    """
    # Example implementation for scanning a directory
    result = {
        "type": "directory",
        "path": dir_path,
        "contents": []
    }
    
    # Try to scan the directory contents
    try:
        if os.path.exists(dir_path) and os.path.isdir(dir_path):
            # List files and folders in the directory
            items = os.listdir(dir_path)
            
            # Add items to the result
            for item in items:
                item_path = os.path.join(dir_path, item)
                if os.path.isdir(item_path):
                    # This is a subdirectory
                    result["contents"].append({
                        "name": item,
                        "type": "directory"
                    })
                else:
                    # This is a file
                    result["contents"].append({
                        "name": item,
                        "type": "file"
                    })
    except Exception as e:
        result["error"] = str(e)
    
    return result

def main():
    """Main function to run the USB scanner."""
    print("USB Drive Scanner")
    print("----------------")
    
    # Find USB drives
    drives = find_usb_drives()
    if not drives:
        print("No USB drives found.")
        return
    
    print(f"Found {len(drives)} USB drive(s): {', '.join(drives)}")
    
    # Get info and scan each drive
    for drive in drives:
        print(f"\nDrive {drive}:")
        
        # Get drive info
        info = get_drive_info(drive)
        print(f"Drive info: {info}")
        
        # Scan the drive
        drive_path = f"{drive}:"
        contents = scan_directory(drive_path)
        
        # Print summary of contents
        if "contents" in contents and contents["contents"]:
            files = [item for item in contents["contents"] if item["type"] == "file"]
            dirs = [item for item in contents["contents"] if item["type"] == "directory"]
            print(f"Found {len(files)} files and {len(dirs)} directories")
        else:
            print("No contents found or error occurred")

if __name__ == "__main__":
    main() 