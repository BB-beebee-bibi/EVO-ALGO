#!/usr/bin/env python3
"""
Drive Scanner

A program that scans all connected storage devices, lets the user choose one,
and creates a detailed snapshot of the selected drive's contents.
"""

import os
import sys
import json
import datetime
import subprocess
import stat
import shutil
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple

def get_drives() -> List[Dict[str, str]]:
    """
    Get a list of all connected drives and their mount points.
    
    Returns:
        List of dictionaries with drive information
    """
    drives = []
    
    try:
        # Run lsblk command to get drive information
        output = subprocess.check_output(
            ["lsblk", "-J", "-o", "NAME,SIZE,TYPE,MOUNTPOINT,LABEL"],
            universal_newlines=True
        )
        
        # Parse JSON output
        lsblk_data = json.loads(output)
        
        # Process drive information
        for device in lsblk_data.get("blockdevices", []):
            if device["type"] == "disk":
                drive_info = {
                    "name": device["name"],
                    "path": f"/dev/{device['name']}",
                    "size": device["size"],
                    "type": device["type"],
                    "partitions": []
                }
                
                # Add partitions if available
                if "children" in device:
                    for partition in device["children"]:
                        if partition.get("mountpoint"):
                            part_info = {
                                "name": partition["name"],
                                "path": f"/dev/{partition['name']}",
                                "size": partition["size"],
                                "type": partition["type"],
                                "mountpoint": partition["mountpoint"],
                                "label": partition.get("label", "")
                            }
                            drive_info["partitions"].append(part_info)
                
                # Only add drives with mounted partitions
                if drive_info["partitions"]:
                    drives.append(drive_info)
        
    except subprocess.CalledProcessError as e:
        print(f"Error: Failed to get drive information: {e}")
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse drive information: {e}")
    
    return drives

def get_drive_usage(mountpoint: str) -> Dict[str, Any]:
    """
    Get usage information for a drive.
    
    Args:
        mountpoint: Path where the drive is mounted
        
    Returns:
        Dictionary with usage information
    """
    try:
        total, used, free = shutil.disk_usage(mountpoint)
        return {
            "total_bytes": total,
            "total_gb": round(total / (1024**3), 2),
            "used_bytes": used,
            "used_gb": round(used / (1024**3), 2),
            "free_bytes": free,
            "free_gb": round(free / (1024**3), 2),
            "percent_used": round((used / total) * 100, 1)
        }
    except Exception as e:
        print(f"Error: Failed to get disk usage: {e}")
        return {}

def scan_directory(path: str, max_depth: int = 3, current_depth: int = 0) -> Dict[str, Any]:
    """
    Scan a directory recursively and collect information about its contents.
    
    Args:
        path: Directory path to scan
        max_depth: Maximum depth for recursion
        current_depth: Current recursion depth
        
    Returns:
        Dictionary with directory structure and file information
    """
    result = {
        "name": os.path.basename(path) or path,
        "path": path,
        "type": "directory",
        "size": 0,
        "contents": []
    }
    
    try:
        # Get directory stats
        stats = os.stat(path)
        result["created"] = datetime.datetime.fromtimestamp(stats.st_ctime).isoformat()
        result["modified"] = datetime.datetime.fromtimestamp(stats.st_mtime).isoformat()
        result["accessed"] = datetime.datetime.fromtimestamp(stats.st_atime).isoformat()
        result["permissions"] = stat.filemode(stats.st_mode)
        
        # Stop recursion if max depth is reached
        if current_depth >= max_depth:
            file_count = 0
            dir_count = 0
            
            for item in os.listdir(path):
                item_path = os.path.join(path, item)
                if os.path.isdir(item_path):
                    dir_count += 1
                else:
                    file_count += 1
                    try:
                        file_stats = os.stat(item_path)
                        result["size"] += file_stats.st_size
                    except:
                        pass
            
            result["_summary"] = {
                "files": file_count,
                "directories": dir_count,
                "max_depth_reached": True
            }
            
            return result
        
        # List contents
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            
            try:
                if os.path.isdir(item_path):
                    # Recursively scan subdirectories
                    subdir = scan_directory(item_path, max_depth, current_depth + 1)
                    result["contents"].append(subdir)
                    result["size"] += subdir["size"]
                else:
                    # Get file information
                    file_stats = os.stat(item_path)
                    file_size = file_stats.st_size
                    result["size"] += file_size
                    
                    file_info = {
                        "name": item,
                        "path": item_path,
                        "type": "file",
                        "size": file_size,
                        "size_human": format_size(file_size),
                        "created": datetime.datetime.fromtimestamp(file_stats.st_ctime).isoformat(),
                        "modified": datetime.datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
                        "accessed": datetime.datetime.fromtimestamp(file_stats.st_atime).isoformat(),
                        "permissions": stat.filemode(file_stats.st_mode),
                        "extension": os.path.splitext(item)[1].lower()
                    }
                    
                    result["contents"].append(file_info)
            except Exception as e:
                # Add error entry for items we can't process
                result["contents"].append({
                    "name": item,
                    "path": item_path,
                    "type": "error",
                    "error": str(e)
                })
    
    except Exception as e:
        result["error"] = str(e)
    
    # Add human-readable size
    result["size_human"] = format_size(result["size"])
    
    return result

def analyze_drive_contents(scan_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze the drive contents to provide summary statistics.
    
    Args:
        scan_result: Result from scan_directory
        
    Returns:
        Dictionary with analysis results
    """
    analysis = {
        "total_size": scan_result["size"],
        "total_size_human": format_size(scan_result["size"]),
        "file_count": 0,
        "directory_count": 0,
        "file_types": defaultdict(int),
        "file_types_size": defaultdict(int),
        "largest_files": [],
        "newest_files": [],
        "oldest_files": []
    }
    
    # Helper function to process each item
    def process_item(item):
        if item["type"] == "file":
            analysis["file_count"] += 1
            ext = item.get("extension", "").lower() or "(no extension)"
            analysis["file_types"][ext] += 1
            analysis["file_types_size"][ext] += item["size"]
            
            # Track largest files
            analysis["largest_files"].append((item["size"], item["path"]))
            analysis["largest_files"].sort(reverse=True)
            analysis["largest_files"] = analysis["largest_files"][:10]
            
            # Track newest and oldest files
            if "modified" in item:
                mod_time = datetime.datetime.fromisoformat(item["modified"])
                analysis["newest_files"].append((mod_time, item["path"]))
                analysis["oldest_files"].append((mod_time, item["path"]))
                
                analysis["newest_files"].sort(reverse=True)
                analysis["oldest_files"].sort()
                
                analysis["newest_files"] = analysis["newest_files"][:10]
                analysis["oldest_files"] = analysis["oldest_files"][:10]
                
        elif item["type"] == "directory":
            analysis["directory_count"] += 1
            if "contents" in item:
                for subitem in item["contents"]:
                    process_item(subitem)
    
    # Process the scan result
    process_item(scan_result)
    
    # Format the results for output
    analysis["largest_files"] = [
        {"path": path, "size": size, "size_human": format_size(size)}
        for size, path in analysis["largest_files"]
    ]
    
    analysis["newest_files"] = [
        {"path": path, "modified": dt.isoformat()}
        for dt, path in analysis["newest_files"]
    ]
    
    analysis["oldest_files"] = [
        {"path": path, "modified": dt.isoformat()}
        for dt, path in analysis["oldest_files"]
    ]
    
    # Convert defaultdicts to regular dicts for JSON serialization
    analysis["file_types"] = dict(analysis["file_types"])
    analysis["file_types_size"] = {
        ext: {"size": size, "size_human": format_size(size)}
        for ext, size in analysis["file_types_size"].items()
    }
    
    return analysis

def format_size(size_bytes: int) -> str:
    """
    Format a size in bytes to a human-readable string.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Human-readable size string
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    size = float(size_bytes)
    unit_index = 0
    
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1
    
    return f"{size:.2f} {units[unit_index]}"

def save_scan_result(result: Dict[str, Any], filename: str) -> None:
    """
    Save scan result to a JSON file.
    
    Args:
        result: Scan result dictionary
        filename: Target filename
    """
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
        print(f"Scan result saved to {filename}")
    except Exception as e:
        print(f"Error saving scan result: {e}")

def main():
    """Main function to run the drive scanner."""
    print("Drive Scanner")
    print("============")
    
    # Get all connected drives
    drives = get_drives()
    
    if not drives:
        print("No mounted drives found.")
        return
    
    # Display available drives
    print("\nAvailable drives:")
    for i, drive in enumerate(drives):
        print(f"[{i+1}] {drive['name']} ({drive['size']})")
        for j, partition in enumerate(drive['partitions']):
            label = f" - {partition['label']}" if partition['label'] else ""
            print(f"    [{i+1}.{j+1}] {partition['name']} ({partition['size']}) mounted at {partition['mountpoint']}{label}")
    
    # Prompt user to select a drive partition
    try:
        choice = input("\nSelect a drive partition (e.g., 1.1): ")
        if '.' in choice:
            drive_idx, part_idx = map(int, choice.split('.'))
            drive_idx -= 1
            part_idx -= 1
            
            if 0 <= drive_idx < len(drives) and 0 <= part_idx < len(drives[drive_idx]['partitions']):
                selected_partition = drives[drive_idx]['partitions'][part_idx]
                mountpoint = selected_partition['mountpoint']
                
                print(f"\nScanning {selected_partition['name']} mounted at {mountpoint}...")
                
                # Get usage information
                usage = get_drive_usage(mountpoint)
                if usage:
                    print(f"Total size: {usage['total_gb']} GB")
                    print(f"Used: {usage['used_gb']} GB ({usage['percent_used']}%)")
                    print(f"Free: {usage['free_gb']} GB")
                
                # Set scan depth
                max_depth = 3
                depth_input = input("\nEnter maximum scan depth (default: 3): ").strip()
                if depth_input and depth_input.isdigit():
                    max_depth = int(depth_input)
                
                print(f"\nScanning with max depth: {max_depth}")
                print("This may take a while for large drives...")
                
                # Scan the selected drive
                scan_start = datetime.datetime.now()
                scan_result = scan_directory(mountpoint, max_depth=max_depth)
                scan_duration = (datetime.datetime.now() - scan_start).total_seconds()
                
                # Analyze the scan results
                print("\nAnalyzing results...")
                analysis = analyze_drive_contents(scan_result)
                
                # Add metadata
                result = {
                    "scan_time": datetime.datetime.now().isoformat(),
                    "scan_duration_seconds": scan_duration,
                    "drive_info": selected_partition,
                    "usage": usage,
                    "max_depth": max_depth,
                    "scan_result": scan_result,
                    "analysis": analysis
                }
                
                # Display summary
                print(f"\nScan completed in {scan_duration:.2f} seconds")
                print(f"Found {analysis['file_count']} files in {analysis['directory_count']} directories")
                print(f"Total size: {analysis['total_size_human']}")
                
                print("\nFile types found:")
                for ext, count in sorted(analysis['file_types'].items(), key=lambda x: x[1], reverse=True)[:10]:
                    ext_size = analysis['file_types_size'][ext]['size_human']
                    print(f"  {ext}: {count} files ({ext_size})")
                
                # Save result to file
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                drive_name = selected_partition['name'].replace('/', '_')
                filename = f"drive_scan_{drive_name}_{timestamp}.json"
                
                save_scan_result(result, filename)
                
                print(f"\nScan complete! Full results saved to {filename}")
            else:
                print("Invalid selection.")
        else:
            print("Invalid selection format. Use drive.partition notation (e.g., 1.1).")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 