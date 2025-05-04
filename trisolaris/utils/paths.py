"""
Path utilities for the TRISOLARIS framework.

This module provides utilities for creating and managing file paths, 
especially for timestamped output directories.
"""

import os
import datetime
import logging
import re
from typing import Optional

# Configure logging
logger = logging.getLogger(__name__)

def create_timestamped_output_dir(base_dir: str = "outputs") -> str:
    """
    Create a timestamped output directory for evolution runs.
    
    Args:
        base_dir: Base directory for all outputs
        
    Returns:
        Path to the created timestamped directory
    """
    # Make sure the base directory exists
    os.makedirs(base_dir, exist_ok=True)
    
    # Create a timestamp in the format YYYYMMDD_HHMMSS
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"run_{timestamp}")
    
    # Create the directory
    os.makedirs(run_dir, exist_ok=True)
    logger.info(f"Created timestamped output directory: {run_dir}")
    
    return run_dir

def create_generation_dir(run_dir: str, generation: int) -> str:
    """
    Create a directory for a specific generation within a run.
    
    Args:
        run_dir: Path to the run directory
        generation: Generation number
        
    Returns:
        Path to the created generation directory
    """
    gen_dir = os.path.join(run_dir, f"generation_{generation}")
    os.makedirs(gen_dir, exist_ok=True)
    return gen_dir

def get_latest_run_dir(base_dir: str = "outputs") -> Optional[str]:
    """
    Find the most recent run directory.
    
    Args:
        base_dir: Base directory for all outputs
        
    Returns:
        Path to the most recent run directory, or None if none exists
    """
    if not os.path.exists(base_dir):
        return None
        
    run_dirs = []
    pattern = re.compile(r"run_(\d{8}_\d{6})")
    
    for item in os.listdir(base_dir):
        full_path = os.path.join(base_dir, item)
        if os.path.isdir(full_path):
            match = pattern.match(item)
            if match:
                timestamp = match.group(1)
                run_dirs.append((full_path, timestamp))
    
    if not run_dirs:
        return None
        
    # Sort by timestamp (newest first)
    run_dirs.sort(key=lambda x: x[1], reverse=True)
    
    return run_dirs[0][0]

def get_latest_generation_dir(run_dir: str) -> Optional[str]:
    """
    Find the directory for the latest generation in a run.
    
    Args:
        run_dir: Path to the run directory
        
    Returns:
        Path to the latest generation directory, or None if none exists
    """
    if not os.path.exists(run_dir):
        return None
        
    gen_dirs = []
    pattern = re.compile(r"generation_(\d+)")
    
    for item in os.listdir(run_dir):
        full_path = os.path.join(run_dir, item)
        if os.path.isdir(full_path):
            match = pattern.match(item)
            if match:
                gen_num = int(match.group(1))
                gen_dirs.append((full_path, gen_num))
    
    if not gen_dirs:
        return None
        
    # Sort by generation number (highest first)
    gen_dirs.sort(key=lambda x: x[1], reverse=True)
    
    return gen_dirs[0][0]

def get_best_solution_path(run_dir: str = None, base_dir: str = "outputs") -> Optional[str]:
    """
    Find the path to the best solution in the most recent run.
    
    Args:
        run_dir: Path to a specific run directory (optional)
        base_dir: Base directory for all outputs (used if run_dir is None)
        
    Returns:
        Path to the best solution file, or None if not found
    """
    # If no specific run directory is provided, find the latest
    if run_dir is None:
        run_dir = get_latest_run_dir(base_dir)
        
    if not run_dir or not os.path.exists(run_dir):
        return None
    
    # Check for best.py file directly in the run directory (final best solution)
    best_path = os.path.join(run_dir, "best.py")
    if os.path.exists(best_path):
        return best_path
    
    # If not found, check the latest generation directory
    latest_gen_dir = get_latest_generation_dir(run_dir)
    if latest_gen_dir:
        best_path = os.path.join(latest_gen_dir, "best.py")
        if os.path.exists(best_path):
            return best_path
    
    return None

def resolve_relative_path(path: str, base_dir: str = None) -> str:
    """
    Resolve a potentially relative path against a base directory.
    
    Args:
        path: Path to resolve (may be absolute or relative)
        base_dir: Base directory for relative paths
        
    Returns:
        Resolved absolute path
    """
    if os.path.isabs(path):
        return path
    
    if base_dir:
        return os.path.join(base_dir, path)
    
    # If no base_dir is provided, use current working directory
    return os.path.join(os.getcwd(), path)

def ensure_dir_exists(path: str) -> str:
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        path: Path to the directory
        
    Returns:
        Path to the directory (same as input)
    """
    os.makedirs(path, exist_ok=True)
    return path
