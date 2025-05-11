#!/usr/bin/env python
"""
Task Template Loader for Enhanced Progremon System

This module provides the TaskTemplateLoader class, which is responsible for
loading and managing task-specific code templates. It supports template caching,
metadata retrieval, and task type detection from natural language descriptions.
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

# Configure logger
logger = logging.getLogger('progremon.template_loader')


class TaskTemplateLoader:
    """
    Loads and configures task-specific code templates.
    
    This class handles loading templates for different types of tasks,
    caching templates for improved performance, and performing task type
    detection from natural language descriptions.
    """
    
    def __init__(self, templates_dir: str = "guidance"):
        """
        Initialize the TaskTemplateLoader.
        
        Args:
            templates_dir: Directory containing template files
        """
        self.templates_dir = templates_dir
        
        # Map of task types to template files
        self.templates = {
            "bluetooth_scan": "bluetooth_scanner_template.py",
            "usb_scan": "usb_scanner_template.py",
            "web": "web_template.py",
            "game": "game_template.py",
            "general": None  # General tasks don't use a specific template
        }
        
        # Template metadata
        self.template_metadata = {
            "bluetooth_scan": {
                "requires_libraries": ["bluetooth"],
                "platform_specific": True,
                "description": "Template for Bluetooth device scanning"
            },
            "usb_scan": {
                "requires_libraries": ["pyusb"],
                "platform_specific": True,
                "description": "Template for USB device scanning"
            },
            "web": {
                "requires_libraries": ["flask"],
                "platform_specific": False,
                "description": "Template for web applications"
            },
            "game": {
                "requires_libraries": ["pygame"],
                "platform_specific": False,
                "description": "Template for simple games"
            }
        }
        
        # Internal template cache
        self._template_cache = {}
        
        # Task detection patterns
        self._task_detection_patterns = {
            "bluetooth_scan": [
                r'bluetooth',
                r'ble',
                r'discover devices', 
                r'nearby devices',
                r'scan for devices'
            ],
            "usb_scan": [
                r'usb',
                r'detect usb',
                r'find devices',
                r'connected devices'
            ],
            "web": [
                r'web',
                r'website',
                r'html',
                r'css',
                r'javascript',
                r'flask',
                r'django'
            ],
            "game": [
                r'game',
                r'pygame',
                r'arcade',
                r'play'
            ]
        }
    
    def load_template(self, task_type: str) -> Optional[str]:
        """
        Load template code with caching for better performance.
        
        Args:
            task_type: Type of task to load template for
            
        Returns:
            Template code as string, or None if not found
        """
        # Check cache first
        if task_type in self._template_cache:
            logger.debug(f"Returning cached template for {task_type}")
            return self._template_cache[task_type]
            
        # Check if task type exists and has a template
        if task_type not in self.templates or not self.templates[task_type]:
            logger.warning(f"No template defined for task type: {task_type}")
            return None
            
        # Construct path to template file
        template_file = Path(self.templates_dir) / self.templates[task_type]
        
        # Check if template file exists
        if not template_file.exists():
            logger.warning(f"Template file not found: {template_file}")
            return None
            
        # Load template
        try:
            with open(template_file, 'r') as f:
                template = f.read()
                
            # Cache template for future use
            self._template_cache[task_type] = template
            logger.debug(f"Loaded and cached template for {task_type}")
            
            return template
        except Exception as e:
            logger.error(f"Error loading template: {e}")
            return None
    
    def reload_template(self, task_type: str) -> Optional[str]:
        """
        Force reload a template, bypassing the cache.
        
        Args:
            task_type: Type of task to reload template for
            
        Returns:
            Template code as string, or None if not found
        """
        # Clear cache entry
        if task_type in self._template_cache:
            del self._template_cache[task_type]
            
        # Load template
        return self.load_template(task_type)
    
    def get_template_metadata(self, task_type: str) -> Dict[str, Any]:
        """
        Get metadata for a template.
        
        Args:
            task_type: Type of task to get metadata for
            
        Returns:
            Dictionary of metadata, empty if not found
        """
        if task_type in self.template_metadata:
            return self.template_metadata[task_type]
        return {}
    
    def get_task_type_from_description(self, description: str) -> str:
        """
        Detect the most likely task type from a natural language description.
        
        Args:
            description: Natural language description
            
        Returns:
            Detected task type, or "general" if no specific type detected
        """
        # Convert description to lowercase for case-insensitive matching
        description = description.lower()
        
        # Count matches for each task type
        match_counts = {}
        
        for task_type, patterns in self._task_detection_patterns.items():
            count = 0
            for pattern in patterns:
                count += len(re.findall(pattern, description))
            match_counts[task_type] = count
        
        # Find task type with highest match count, if any
        best_match = "general"
        best_count = 0
        
        for task_type, count in match_counts.items():
            if count > best_count:
                best_count = count
                best_match = task_type
        
        logger.debug(f"Task detection: {best_match} (score: {best_count})")
        return best_match
    
    def list_available_templates(self) -> List[str]:
        """
        List all available templates.
        
        Returns:
            List of available task types
        """
        return [task for task, file in self.templates.items() if file]
    
    def get_template_info(self, task_type: str) -> Dict[str, Any]:
        """
        Get comprehensive information about a template.
        
        Args:
            task_type: Type of task to get information for
            
        Returns:
            Dictionary with template information
        """
        info = {
            "task_type": task_type,
            "has_template": task_type in self.templates and self.templates[task_type] is not None
        }
        
        # Add template file info if available
        if info["has_template"]:
            template_file = Path(self.templates_dir) / self.templates[task_type]
            info["template_file"] = str(template_file)
            info["exists"] = template_file.exists()
            
            if info["exists"]:
                info["size"] = template_file.stat().st_size
                info["modified"] = template_file.stat().st_mtime
        
        # Add metadata if available
        if task_type in self.template_metadata:
            info["metadata"] = self.template_metadata[task_type]
        
        return info
    

# Simple test code to verify functionality
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create template loader
    loader = TaskTemplateLoader()
    
    # List available templates
    print("Available templates:")
    for template in loader.list_available_templates():
        print(f"  - {template}")
    
    # Test task type detection
    test_descriptions = [
        "Create a bluetooth scanner to find nearby devices",
        "Build a web application with HTML and CSS",
        "Make a USB device detector",
        "Generate a sorting algorithm"
    ]
    
    print("\nTask type detection:")
    for desc in test_descriptions:
        task_type = loader.get_task_type_from_description(desc)
        print(f"  \"{desc}\" -> {task_type}")
    
    # Test loading a template
    print("\nAttempting to load bluetooth_scan template...")
    template = loader.load_template("bluetooth_scan")
    if template:
        print(f"  Success! Template size: {len(template)} characters")
        print(f"  First few lines:\n  {template.split('\\n')[0]}\n  {template.split('\\n')[1]}")
    else:
        print("  Template not found")