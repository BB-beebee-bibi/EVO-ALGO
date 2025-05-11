#!/usr/bin/env python
"""
Utilities for Enhanced Progremon System

This module provides utility functions shared across the Progremon system,
including color output for terminals, logging configuration, ID generation,
and directory management.
"""

import os
import sys
import uuid
import time
import random
import logging
import datetime
from pathlib import Path
from typing import Optional, Union, Dict, Any, List


class Colors:
    """ANSI color codes for terminal output."""
    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    YELLOW = "\033[0;33m"
    BLUE = "\033[0;34m"
    MAGENTA = "\033[0;35m"
    CYAN = "\033[0;36m"
    WHITE = "\033[0;37m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    RESET = "\033[0m"


def print_color(text: str, color: str, bold: bool = False, underline: bool = False, end: str = "\n"):
    """
    Print text with the specified color and styling.
    
    Args:
        text: Text to print
        color: ANSI color code from Colors class
        bold: Whether to make the text bold
        underline: Whether to underline the text
        end: String appended after the final value
    """
    style = ""
    if bold:
        style += Colors.BOLD
    if underline:
        style += Colors.UNDERLINE
    
    formatted_text = f"{style}{color}{text}{Colors.RESET}"
    print(formatted_text, end=end)


def print_table(headers: List[str], rows: List[List[Any]], colors: Optional[List[str]] = None):
    """
    Print a formatted table with headers and rows.
    
    Args:
        headers: List of column headers
        rows: List of lists containing row values
        colors: Optional list of colors for each row
    """
    if not rows:
        return
    
    # Calculate column widths
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))
    
    # Create line format
    line_format = " | ".join(f"{{:{w}}}" for w in col_widths)
    
    # Print header
    header_line = line_format.format(*headers)
    print_color(header_line, Colors.CYAN, bold=True)
    print_color("-" * len(header_line), Colors.CYAN)
    
    # Print rows
    for i, row in enumerate(rows):
        color = colors[i] if colors and i < len(colors) else Colors.WHITE
        print_color(line_format.format(*[str(cell) for cell in row]), color)


def print_banner():
    """Print the Progremon ASCII banner."""
    banner = r"""
    ____                                                  
   / __ \_________  ____ ________  ____ ___  ____  ____  
  / /_/ / ___/ __ \/ __ `/ ___/ / / / __ `__ \/ __ \/ __ \ 
 / ____/ /  / /_/ / /_/ / /  / /_/ / / / / / / /_/ / / / / 
/_/   /_/   \____/\__, /_/   \__,_/_/ /_/ /_/\____/_/ /_/  
                 /____/                                    
                                     
      * ~ Gotta evolve 'em all! ~ *
    """
    print_color(banner, Colors.CYAN, bold=True)


def setup_logging(log_file: Optional[Union[str, Path]] = None, 
                level: int = logging.INFO) -> logging.Logger:
    """
    Configure the logging system for Progremon.
    
    Args:
        log_file: Optional path to a log file
        level: Logging level
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger('progremon')
    logger.setLevel(level)
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    
    # Add console handler to logger
    logger.addHandler(console_handler)
    
    # Add file handler if specified
    if log_file:
        if isinstance(log_file, str):
            log_file = Path(log_file)
        
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def generate_unique_id(prefix: str = "") -> str:
    """
    Generate a unique ID with an optional prefix.
    
    Args:
        prefix: Optional prefix for the ID
        
    Returns:
        Unique ID string
    """
    # Alternatives for generating unique IDs
    methods = [
        # Method 1: Random number + timestamp
        lambda: f"{random.randint(1000, 9999)}_{int(time.time())}",
        
        # Method 2: UUID-based
        lambda: str(uuid.uuid4())[:8],
        
        # Method 3: Timestamp + random
        lambda: f"{int(time.time())}_{random.randint(10, 99)}"
    ]
    
    # Choose a method randomly to increase uniqueness
    unique_id = random.choice(methods)()
    
    if prefix:
        return f"{prefix}_{unique_id}"
    return unique_id


def create_timestamped_directory(base_dir: Union[str, Path], prefix: str = "") -> Path:
    """
    Create a directory with a timestamp in the name.
    
    Args:
        base_dir: Base directory
        prefix: Optional prefix for directory name
        
    Returns:
        Path to created directory
    """
    if isinstance(base_dir, str):
        base_dir = Path(base_dir)
    
    base_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if prefix:
        dir_name = f"{prefix}_{timestamp}"
    else:
        dir_name = timestamp
    
    path = base_dir / dir_name
    path.mkdir(exist_ok=True)
    
    return path


def save_json(data: Any, file_path: Union[str, Path]) -> None:
    """
    Save data to a JSON file.
    
    Args:
        data: Data to save
        file_path: Path to save to
    """
    import json
    
    if isinstance(file_path, str):
        file_path = Path(file_path)
    
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(file_path: Union[str, Path]) -> Any:
    """
    Load data from a JSON file.
    
    Args:
        file_path: Path to load from
        
    Returns:
        Loaded data
    """
    import json
    
    with open(file_path, 'r') as f:
        return json.load(f)


def confirm_action(prompt: str, default: bool = True) -> bool:
    """
    Ask for user confirmation.
    
    Args:
        prompt: Prompt to display
        default: Default action if user presses Enter
        
    Returns:
        True if confirmed, False otherwise
    """
    yes_options = ['y', 'yes']
    no_options = ['n', 'no']
    
    default_prompt = "[Y/n]" if default else "[y/N]"
    full_prompt = f"{prompt} {default_prompt} "
    
    while True:
        print_color(full_prompt, Colors.GREEN, end="")
        response = input().strip().lower()
        
        if not response:
            return default
        
        if response in yes_options:
            return True
        if response in no_options:
            return False
        
        print_color("Please answer 'y' or 'n'", Colors.YELLOW)


def format_time_delta(seconds: float) -> str:
    """
    Format a time delta in a human-readable way.
    
    Args:
        seconds: Number of seconds
        
    Returns:
        Formatted string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    
    minutes = seconds // 60
    seconds = seconds % 60
    
    if minutes < 60:
        return f"{int(minutes)}m {int(seconds)}s"
    
    hours = minutes // 60
    minutes = minutes % 60
    
    return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"


def get_platform_info() -> Dict[str, str]:
    """
    Get information about the current platform.
    
    Returns:
        Dictionary with platform information
    """
    import platform
    
    return {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version()
    }


# Simple test code to verify functionality
if __name__ == "__main__":
    # Test color printing
    print_color("This is RED text", Colors.RED)
    print_color("This is GREEN text", Colors.GREEN)
    print_color("This is BLUE text", Colors.BLUE)
    print_color("This is BOLD text", Colors.WHITE, bold=True)
    
    # Test banner
    print_banner()
    
    # Test table printing
    headers = ["Name", "Value", "Status"]
    rows = [
        ["Item 1", 123, "OK"],
        ["Item 2", 456, "Error"],
        ["Item 3", 789, "Warning"]
    ]
    colors = [Colors.GREEN, Colors.RED, Colors.YELLOW]
    print_table(headers, rows, colors)
    
    # Test logging
    logger = setup_logging()
    logger.info("Test log message")
    logger.warning("Test warning")
    logger.error("Test error")
    
    # Test unique ID generation
    print_color("\nGenerated IDs:", Colors.BLUE)
    for i in range(5):
        unique_id = generate_unique_id("test")
        print_color(f"  {unique_id}", Colors.CYAN)
    
    # Test directory creation
    print_color("\nCreating test directory...", Colors.BLUE)
    test_dir = create_timestamped_directory("test_output", "utils_test")
    print_color(f"Created directory: {test_dir}", Colors.CYAN)
    
    # Test confirmation
    if confirm_action("Run additional tests?", False):
        print_color("\nRunning additional tests...", Colors.BLUE)
        # Additional tests would go here
    else:
        print_color("\nSkipping additional tests.", Colors.YELLOW)
    
    # Test time formatting
    print_color("\nFormatted times:", Colors.BLUE)
    times = [5.2, 65, 3600, 7262]
    for t in times:
        print_color(f"  {t:.1f}s -> {format_time_delta(t)}", Colors.CYAN)
    
    # Test platform info
    print_color("\nPlatform information:", Colors.BLUE)
    platform_info = get_platform_info()
    for key, value in platform_info.items():
        print_color(f"  {key}: {value}", Colors.CYAN)