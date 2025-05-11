#!/usr/bin/env python
"""
Evolution Session Management for Enhanced Progremon System

This module provides the EvolutionSession class, which manages the lifecycle,
data storage, and organization of evolution runs. It handles session creation,
metadata tracking, solution saving, and checkpoint creation.
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

# Import utilities
try:
    from utils import generate_unique_id, save_json, load_json, format_time_delta
except ImportError:
    # Fallback implementations if utils module not available
    from uuid import uuid4
    def generate_unique_id(prefix=""): 
        return f"{prefix}_{str(uuid4())[:8]}"
    def save_json(data, file_path):
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    def load_json(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    def format_time_delta(seconds):
        return f"{seconds:.1f}s"

# Configure logger
logger = logging.getLogger('progremon.evolution_session')


class EvolutionSession:
    """
    Manages the lifecycle and data for an evolution session.
    
    This class handles session creation, metadata tracking, solution saving,
    directory organization, and checkpointing for evolution runs. It provides
    a consistent interface for managing session data across the system.
    """
    
    def __init__(self, base_dir: str = "evolved_output", task_type: str = "general"):
        """
        Initialize a new evolution session.
        
        Args:
            base_dir: Base directory for output
            task_type: Type of task being evolved
        """
        self.base_dir = Path(base_dir)
        self.task_type = task_type
        self.start_time = datetime.now()
        self.end_time = None
        
        # Generate a unique session ID
        self.session_id = f"trial_{generate_unique_id()}"
        
        # Create session directories
        self.session_dir = self.base_dir / self.session_id
        self.generations_dir = self.session_dir / "generations"
        self.best_solution_dir = self.session_dir / "best_solution"
        
        # Create directories
        self._create_directories()
        
        # Initialize metadata
        self.metadata = {
            "session_id": self.session_id,
            "task_type": self.task_type,
            "start_time": self.start_time.isoformat(),
            "end_time": None,
            "duration_seconds": None,
            "generations": 0,
            "best_fitness": 0.0,
            "fitness_history": [],
            "settings": {},
            "best_solution_path": None
        }
        
        # Save initial metadata
        self._save_metadata()
        
        logger.info(f"Created new evolution session: {self.session_id}")
    
    def _create_directories(self):
        """Create necessary directories for the session."""
        # Create main session directory
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.generations_dir.mkdir(exist_ok=True)
        self.best_solution_dir.mkdir(exist_ok=True)
    
    def _save_metadata(self):
        """Save session metadata to disk."""
        metadata_path = self.session_dir / "session_info.json"
        save_json(self.metadata, metadata_path)
    
    def update_settings(self, settings: Dict[str, Any]):
        """
        Update session settings.
        
        Args:
            settings: Dictionary of evolution settings
        """
        # Filter out large text fields like description to keep metadata concise
        filtered_settings = {}
        for key, value in settings.items():
            if key == "description" and isinstance(value, str) and len(value) > 100:
                filtered_settings[key] = value[:100] + "..."
            else:
                filtered_settings[key] = value
        
        self.metadata["settings"] = filtered_settings
        self._save_metadata()
        logger.debug(f"Updated session settings: {self.session_id}")
    
    def record_generation(self, generation_number: int, best_fitness: float, 
                        avg_fitness: float, population_size: int):
        """
        Record information about a completed generation.
        
        Args:
            generation_number: Generation number (0-based)
            best_fitness: Best fitness achieved in this generation
            avg_fitness: Average fitness across population
            population_size: Size of the population
        """
        # Create generation directory
        gen_dir = self.generations_dir / f"generation_{generation_number}"
        gen_dir.mkdir(exist_ok=True)
        
        # Update metadata
        self.metadata["generations"] = max(self.metadata["generations"], generation_number + 1)
        
        if best_fitness > self.metadata["best_fitness"]:
            self.metadata["best_fitness"] = best_fitness
        
        # Record fitness history
        self.metadata["fitness_history"].append({
            "generation": generation_number,
            "best_fitness": best_fitness,
            "avg_fitness": avg_fitness,
            "population_size": population_size,
            "timestamp": datetime.now().isoformat()
        })
        
        # Save metadata
        self._save_metadata()
        logger.debug(f"Recorded generation {generation_number} (best fitness: {best_fitness:.4f})")
    
    def save_solution(self, code: str, generation: int, fitness: float, is_best: bool = False) -> Path:
        """
        Save a solution to disk.
        
        Args:
            code: Solution code to save
            generation: Generation number
            fitness: Solution fitness
            is_best: Whether this is the best solution so far
            
        Returns:
            Path to the saved solution
        """
        # Format timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Determine file name and path
        file_name = f"{self.task_type}_{timestamp}.py"
        meta_name = f"{self.task_type}_{timestamp}_meta.json"
        
        if is_best:
            # Save to best solution directory
            solution_path = self.best_solution_dir / file_name
            meta_path = self.best_solution_dir / meta_name
            
            # Update metadata
            self.metadata["best_solution_path"] = str(solution_path)
        else:
            # Save to generation directory
            gen_dir = self.generations_dir / f"generation_{generation}"
            gen_dir.mkdir(exist_ok=True)
            
            solution_path = gen_dir / file_name
            meta_path = gen_dir / meta_name
        
        # Save solution code
        with open(solution_path, 'w') as f:
            f.write(code)
        
        # Save solution metadata
        solution_meta = {
            "fitness": fitness,
            "generation": generation,
            "task_type": self.task_type,
            "timestamp": timestamp,
            "is_best": is_best
        }
        save_json(solution_meta, meta_path)
        
        if is_best:
            logger.info(f"Saved best solution: {solution_path} (fitness: {fitness:.4f})")
        else:
            logger.debug(f"Saved solution: {solution_path} (fitness: {fitness:.4f})")
        
        return solution_path
    
    def create_checkpoint(self) -> Path:
        """
        Create a checkpoint of the current session state.
        
        Returns:
            Path to the checkpoint file
        """
        # Format timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create checkpoint file
        checkpoint_path = self.session_dir / f"checkpoint_{timestamp}.json"
        
        # Create checkpoint data (deep copy of metadata with additional info)
        checkpoint_data = dict(self.metadata)
        checkpoint_data["checkpoint_time"] = timestamp
        
        # Save checkpoint
        save_json(checkpoint_data, checkpoint_path)
        logger.info(f"Created session checkpoint: {checkpoint_path}")
        
        return checkpoint_path
    
    def finalize(self):
        """Finalize the session and update completion metadata."""
        # Record end time
        self.end_time = datetime.now()
        
        # Update metadata
        self.metadata["end_time"] = self.end_time.isoformat()
        
        # Calculate duration
        duration = (self.end_time - self.start_time).total_seconds()
        self.metadata["duration_seconds"] = duration
        self.metadata["duration_formatted"] = format_time_delta(duration)
        
        # Final metadata save
        self._save_metadata()
        logger.info(f"Finalized session {self.session_id}, duration: {format_time_delta(duration)}")
    
    @classmethod
    def load_from_id(cls, session_id: str, base_dir: str = "evolved_output") -> 'EvolutionSession':
        """
        Load an existing session from its ID.
        
        Args:
            session_id: ID of the session to load
            base_dir: Base directory for output
            
        Returns:
            Loaded EvolutionSession object
            
        Raises:
            FileNotFoundError: If the session directory or metadata not found
        """
        # Construct session path
        session_path = Path(base_dir) / session_id
        metadata_path = session_path / "session_info.json"
        
        # Check if session exists
        if not session_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(f"Session not found: {session_id}")
        
        # Load metadata
        metadata = load_json(metadata_path)
        
        # Create session object
        session = cls.__new__(cls)  # Create instance without calling __init__
        
        # Set attributes
        session.base_dir = Path(base_dir)
        session.session_id = session_id
        session.task_type = metadata.get("task_type", "general")
        session.session_dir = session_path
        session.generations_dir = session_path / "generations"
        session.best_solution_dir = session_path / "best_solution"
        session.metadata = metadata
        
        # Parse timestamps
        try:
            session.start_time = datetime.fromisoformat(metadata["start_time"])
            if metadata["end_time"]:
                session.end_time = datetime.fromisoformat(metadata["end_time"])
            else:
                session.end_time = None
        except (KeyError, ValueError):
            # Fallback if timestamps are missing or invalid
            session.start_time = datetime.now()
            session.end_time = None
        
        logger.info(f"Loaded existing session: {session_id}")
        return session
    
    @classmethod
    def list_sessions(cls, base_dir: str = "evolved_output") -> List[Dict[str, Any]]:
        """
        List all existing sessions.
        
        Args:
            base_dir: Base directory for output
            
        Returns:
            List of dictionaries with session information
        """
        # Convert to Path
        base_path = Path(base_dir)
        
        # Check if base directory exists
        if not base_path.exists():
            logger.warning(f"Base directory not found: {base_dir}")
            return []
        
        # Find all subdirectories that contain session_info.json
        sessions = []
        for session_dir in base_path.iterdir():
            if session_dir.is_dir():
                metadata_path = session_dir / "session_info.json"
                if metadata_path.exists():
                    try:
                        metadata = load_json(metadata_path)
                        sessions.append({
                            "session_id": session_dir.name,
                            "task_type": metadata.get("task_type", "unknown"),
                            "start_time": metadata.get("start_time"),
                            "end_time": metadata.get("end_time"),
                            "generations": metadata.get("generations", 0),
                            "best_fitness": metadata.get("best_fitness", 0.0),
                            "directory": str(session_dir)
                        })
                    except Exception as e:
                        logger.warning(f"Error loading session metadata: {e}")
                        continue
        
        # Sort by start time (most recent first)
        sessions.sort(key=lambda s: s.get("start_time", ""), reverse=True)
        
        return sessions


# Simple test code to verify functionality
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    print("Testing EvolutionSession...")
    
    # Create a test session
    print("Creating test session...")
    session = EvolutionSession(base_dir="test_output", task_type="test")
    print(f"Session ID: {session.session_id}")
    
    # Update settings
    print("Updating settings...")
    session.update_settings({
        "pop_size": 50,
        "gens": 25,
        "description": "This is a test description that's quite long and should be truncated in the metadata"
    })
    
    # Record some generations
    print("Recording generations...")
    for i in range(5):
        session.record_generation(
            generation_number=i,
            best_fitness=0.5 + i * 0.1,
            avg_fitness=0.3 + i * 0.1,
            population_size=50
        )
    
    # Save a solution
    print("Saving solution...")
    code = "def test():\n    return 'Hello World'\n"
    path = session.save_solution(
        code=code,
        generation=4,
        fitness=0.9,
        is_best=True
    )
    print(f"Solution saved to: {path}")
    
    # Create a checkpoint
    print("Creating checkpoint...")
    checkpoint_path = session.create_checkpoint()
    print(f"Checkpoint saved to: {checkpoint_path}")
    
    # Finalize session
    print("Finalizing session...")
    session.finalize()
    print(f"Session duration: {session.metadata['duration_formatted']}")
    
    # List all sessions
    print("\nListing all sessions...")
    sessions = EvolutionSession.list_sessions("test_output")
    for s in sessions:
        print(f"  {s['session_id']} ({s['task_type']}) - {s['best_fitness']} fitness")
    
    # Load the session we just created
    print("\nLoading session...")
    loaded_session = EvolutionSession.load_from_id(session.session_id, "test_output")
    print(f"Loaded session: {loaded_session.session_id}")
    print(f"Best fitness: {loaded_session.metadata['best_fitness']}")
    
    print("\nTest completed successfully!")