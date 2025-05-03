"""
Minja USB Drive Scanner

A simple module to scan a USB drive and return its contents
"""

import os
import platform
import json
import datetime
from typing import Dict, List, Any

def get_drive_info(drive_path: str) -> Dict[str, Any]:
    """
    Get information about a drive including available space, type, etc.
    
    Args:
        drive_path: Path to the drive
        
    Returns:
        Dictionary with drive information
    """
    try:
        total, used, free = os.popen(f'wmic logicaldisk where "DeviceID=\'{drive_path}:\'" get size,freespace /format:csv').read().strip().split('\n')
        # Parse the CSV output
        _, free_space, total_size = free.split(',')
        free_space = int(free_space) / (1024 * 1024 * 1024)  # Convert to GB
        total_size = int(total_size) / (1024 * 1024 * 1024)  # Convert to GB
        
        info = {
            "path": drive_path,
            "total_size_gb": round(total_size, 2),
            "free_space_gb": round(free_space, 2),
            "used_space_gb": round(total_size - free_space, 2),
            "format": os.popen(f'wmic logicaldisk where "DeviceID=\'{drive_path}:\'" get FileSystem').read().strip().split('\n')[1]
        }
        return info
    except Exception as e:
        return {
            "path": drive_path,
            "error": str(e)
        }

def scan_directory(dir_path: str, max_depth: int = 2, current_depth: int = 0) -> Dict[str, Any]:
    """
    Scan a directory and return its structure
    
    Args:
        dir_path: Path to the directory
        max_depth: Maximum depth to scan
        current_depth: Current scanning depth
        
    Returns:
        Dictionary with directory structure
    """
    result = {
        "name": os.path.basename(dir_path) or dir_path,
        "path": dir_path,
        "type": "directory",
        "size": 0,
        "contents": []
    }
    
    try:
        # Get directory stats
        stats = os.stat(dir_path)
        result["created"] = datetime.datetime.fromtimestamp(stats.st_ctime).isoformat()
        result["modified"] = datetime.datetime.fromtimestamp(stats.st_mtime).isoformat()
        
        # List contents
        if current_depth < max_depth:
            for item in os.listdir(dir_path):
                item_path = os.path.join(dir_path, item)
                
                try:
                    if os.path.isdir(item_path):
                        # For directories, recursively scan
                        subdir = scan_directory(item_path, max_depth, current_depth + 1)
                        result["contents"].append(subdir)
                        result["size"] += subdir["size"]
                    else:
                        # For files, get basic info
                        file_stats = os.stat(item_path)
                        file_size = file_stats.st_size
                        result["size"] += file_size
                        
                        result["contents"].append({
                            "name": item,
                            "path": item_path,
                            "type": "file",
                            "size": file_size,
                            "created": datetime.datetime.fromtimestamp(file_stats.st_ctime).isoformat(),
                            "modified": datetime.datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
                            "extension": os.path.splitext(item)[1].lower()
                        })
                except Exception as e:
                    # Add error entry for items we can't process
                    result["contents"].append({
                        "name": item,
                        "path": item_path,
                        "type": "error",
                        "error": str(e)
                    })
        else:
            # At max depth, just count files and directories
            files = []
            dirs = []
            for item in os.listdir(dir_path):
                item_path = os.path.join(dir_path, item)
                if os.path.isdir(item_path):
                    dirs.append(item)
                else:
                    files.append(item)
            
            result["_summary"] = {
                "num_files": len(files),
                "num_dirs": len(dirs),
                "max_depth_reached": True
            }
    
    except Exception as e:
        result["error"] = str(e)
    
    return result

def find_usb_drives() -> List[str]:
    """
    Find USB drives connected to the system
    
    Returns:
        List of drive letters for USB drives
    """
    if platform.system() == "Windows":
        # Use PowerShell to find removable drives
        output = os.popen('wmic logicaldisk where "drivetype=2" get deviceid, volumename, description /format:csv').read()
        drives = []
        for line in output.strip().split('\n')[1:]:  # Skip header
            if line:
                parts = line.split(',')
                if len(parts) >= 2:
                    drive_letter = parts[1].strip()[0]  # Extract just the letter
                    drives.append(drive_letter)
        return drives
    else:
        # For non-Windows systems, this would be different
        return []

def scan_usb_drive(drive_letter: str = None, output_file: str = None) -> Dict[str, Any]:
    """
    Scan a USB drive and return its contents
    
    Args:
        drive_letter: Drive letter to scan, or None to scan the first USB drive
        output_file: Optional file to save the scan results to
        
    Returns:
        Dictionary with scan results
    """
    # Find USB drives if not specified
    if not drive_letter:
        drives = find_usb_drives()
        if not drives:
            return {"error": "No USB drives found"}
        drive_letter = drives[0]
    
    # Format the drive path
    drive_path = f"{drive_letter}:"
    
    # Get drive information
    result = {
        "timestamp": datetime.datetime.now().isoformat(),
        "system": platform.system(),
        "drive_info": get_drive_info(drive_letter),
        "contents": scan_directory(drive_path, max_depth=2)
    }
    
    # Save to file if requested
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
    
    return result

if __name__ == "__main__":
    # Basic CLI functionality
    import sys
    
    drive = None
    output = "usb_scan_result.json"
    
    if len(sys.argv) > 1:
        drive = sys.argv[1]
    
    if len(sys.argv) > 2:
        output = sys.argv[2]
    
    print(f"Scanning USB drive {'(auto-detect)' if not drive else drive}")
    result = scan_usb_drive(drive, output)
    
    print(f"Scan complete. Results saved to {output}")
    
    # Print summary
    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        drive_info = result["drive_info"]
        print(f"Drive: {drive_info['path']}")
        print(f"Total size: {drive_info['total_size_gb']} GB")
        print(f"Free space: {drive_info['free_space_gb']} GB")
        print(f"Format: {drive_info['format']}")
        
        # Print top-level directories
        contents = result["contents"]
        print("\nTop-level directories:")
        for item in contents["contents"]:
            if item["type"] == "directory":
                print(f"  {item['name']} ({len(item['contents'])} items)") 