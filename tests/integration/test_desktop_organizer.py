#!/usr/bin/env python3
"""
Integration tests for the desktop organizer task.
"""
import os
import shutil
from pathlib import Path
import pytest
from trisolaris.tasks.desktop_organizer_task import DesktopOrganizerTask

@pytest.fixture
def test_desktop():
    """Create a temporary desktop directory with test files."""
    desktop = Path("test_desktop")
    
    # Clean up any existing test directory
    if desktop.exists():
        shutil.rmtree(desktop)
    
    desktop.mkdir(exist_ok=True)
    
    # Create test files
    test_files = {
        'document.txt': 'text/plain',
        'image.jpg': 'image/jpeg',
        'document.pdf': 'application/pdf',
        'code.py': 'text/x-python',
        'audio.mp3': 'audio/mpeg',
        'video.mp4': 'video/mp4',
        'archive.zip': 'application/zip',
        'spreadsheet.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    }
    
    for filename in test_files:
        (desktop / filename).touch()
    
    yield desktop
    
    # Cleanup
    try:
        if desktop.exists():
            shutil.rmtree(desktop)
    except Exception as e:
        print(f"Warning: Failed to clean up test directory: {e}")

def test_desktop_organizer_task():
    """Test the desktop organizer task."""
    task = DesktopOrganizerTask()
    
    # Test template generation
    template = task.get_template()
    assert template is not None
    assert "def organize_files" in template
    
    # Test evolution parameters
    params = task.get_evolution_params()
    assert 'population_size' in params
    assert 'num_generations' in params
    assert 'mutation_rate' in params
    assert 'crossover_rate' in params
    
    # Test fitness weights
    weights = task.get_fitness_weights()
    assert 'functionality' in weights
    assert 'efficiency' in weights
    assert 'robustness' in weights
    assert 'maintainability' in weights
    
    # Test required boundaries
    boundaries = task.get_required_boundaries()
    assert 'max_file_size' in boundaries
    assert 'allowed_imports' in boundaries
    assert 'forbidden_imports' in boundaries

def test_evolved_solution(test_desktop):
    """Test an evolved solution on the test desktop."""
    task = DesktopOrganizerTask()
    
    # Get the template code
    code = task.get_template()
    
    # Evaluate fitness
    fitness, scores = task.evaluate_fitness(code)
    
    # Basic assertions
    assert 0 <= fitness <= 1
    assert all(0 <= score <= 1 for score in scores.values())
    
    # Test that the code can be executed
    try:
        exec(code, {'__builtins__': __builtins__})
    except Exception as e:
        pytest.fail(f"Code execution failed: {e}") 