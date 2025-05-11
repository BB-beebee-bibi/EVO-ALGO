#!/usr/bin/env python
"""
Test Suite for Enhanced Progremon System

This module contains tests for verifying functionality of the enhanced
Progremon system, including component integration, template loading,
session management, and evolution processes.
"""

import os
import sys
import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add parent directory to path if needed
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import system components
try:
    from task_template_loader import TaskTemplateLoader
    from evolution_session import EvolutionSession
    from utils import generate_unique_id, Colors, print_color
    from progremon_enhanced import ProgemonTrainer
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure you're running this from the project root directory")
    sys.exit(1)


class TestTaskTemplateLoader(unittest.TestCase):
    """Tests for the TaskTemplateLoader component."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for test templates
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test template files
        test_template = "# Test Template\ndef test_function():\n    return 'Hello World'\n"
        
        # Write test templates
        template_path = Path(self.temp_dir) / "test_template.py"
        with open(template_path, 'w') as f:
            f.write(test_template)
        
        # Create loader with test directory
        self.loader = TaskTemplateLoader(templates_dir=self.temp_dir)
        
        # Override templates mapping
        self.loader.templates = {
            "test": "test_template.py",
            "general": None
        }
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_load_template(self):
        """Test loading a template."""
        template = self.loader.load_template("test")
        self.assertIsNotNone(template)
        self.assertIn("test_function", template)
    
    def test_load_nonexistent_template(self):
        """Test loading a nonexistent template."""
        template = self.loader.load_template("nonexistent")
        self.assertIsNone(template)
    
    def test_template_caching(self):
        """Test that templates are cached."""
        # Load the template once
        template1 = self.loader.load_template("test")
        
        # Modify the template file
        template_path = Path(self.temp_dir) / "test_template.py"
        with open(template_path, 'w') as f:
            f.write("# Modified Template\n")
        
        # Load the template again - should return cached version
        template2 = self.loader.load_template("test")
        
        # Templates should be identical
        self.assertEqual(template1, template2)
        
        # Force reload should get the new version
        template3 = self.loader.reload_template("test")
        self.assertNotEqual(template1, template3)
        self.assertIn("Modified Template", template3)
    
    def test_template_metadata(self):
        """Test retrieving template metadata."""
        # Set up test metadata
        self.loader.template_metadata = {
            "test": {
                "requires_libraries": ["pytest"],
                "platform_specific": False
            }
        }
        
        # Get metadata
        metadata = self.loader.get_template_metadata("test")
        self.assertEqual(metadata["requires_libraries"], ["pytest"])
        self.assertFalse(metadata["platform_specific"])
    
    def test_task_type_detection(self):
        """Test detecting task type from description."""
        bluetooth_task = "Create a bluetooth scanner to find nearby devices"
        web_task = "Build a simple website with HTML and CSS"
        usb_task = "Make a USB device detector"
        general_task = "Generate a sorting algorithm"
        
        self.assertEqual(self.loader.get_task_type_from_description(bluetooth_task), "bluetooth_scan")
        self.assertEqual(self.loader.get_task_type_from_description(web_task), "web")
        self.assertEqual(self.loader.get_task_type_from_description(usb_task), "usb_scan")
        self.assertEqual(self.loader.get_task_type_from_description(general_task), "general")


class TestEvolutionSession(unittest.TestCase):
    """Tests for the EvolutionSession component."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_session_initialization(self):
        """Test that a session is properly initialized."""
        session = EvolutionSession(base_dir=self.temp_dir, task_type="test")
        
        # Check that the session ID was generated
        self.assertIsNotNone(session.session_id)
        
        # Check that directories were created
        session_dir = Path(self.temp_dir) / session.session_id
        self.assertTrue(session_dir.exists())
        self.assertTrue((session_dir / "generations").exists())
        self.assertTrue((session_dir / "best_solution").exists())
        
        # Check that session info was saved
        session_info_file = session_dir / "session_info.json"
        self.assertTrue(session_info_file.exists())
    
    def test_recording_generation(self):
        """Test recording generation information."""
        session = EvolutionSession(base_dir=self.temp_dir, task_type="test")
        
        # Record a generation
        session.record_generation(
            generation_number=1,
            best_fitness=0.75,
            avg_fitness=0.5,
            population_size=50
        )
        
        # Check that the generation was recorded
        self.assertEqual(session.metadata["generations"], 1)
        self.assertEqual(session.metadata["best_fitness"], 0.75)
        self.assertEqual(len(session.metadata["fitness_history"]), 1)
        
        # Check that the generation directory was created
        gen_dir = Path(self.temp_dir) / session.session_id / "generations" / "generation_1"
        self.assertTrue(gen_dir.exists())
    
    def test_saving_solution(self):
        """Test saving a solution."""
        session = EvolutionSession(base_dir=self.temp_dir, task_type="test")
        
        # Save a solution
        code = "def hello():\n    print('Hello, World!')\n"
        solution_path = session.save_solution(
            code=code,
            generation=1,
            fitness=0.8,
            is_best=True
        )
        
        # Check that the solution was saved
        self.assertTrue(solution_path.exists())
        
        # Check that the best solution was also saved
        best_solution_dir = Path(self.temp_dir) / session.session_id / "best_solution"
        self.assertTrue(list(best_solution_dir.glob("*.py")))
        
        # Check that the metadata was updated
        self.assertIn("best_solution_path", session.metadata)
    
    def test_finalizing_session(self):
        """Test finalizing a session."""
        session = EvolutionSession(base_dir=self.temp_dir, task_type="test")
        
        # Finalize the session
        session.finalize()
        
        # Check that the end time was set
        self.assertIsNotNone(session.end_time)
        self.assertIsNotNone(session.metadata["end_time"])
        self.assertIn("duration_seconds", session.metadata)
    
    def test_loading_session(self):
        """Test loading an existing session."""
        # Create a session
        original_session = EvolutionSession(base_dir=self.temp_dir, task_type="test")
        session_id = original_session.session_id
        
        # Record a generation
        original_session.record_generation(
            generation_number=1,
            best_fitness=0.75,
            avg_fitness=0.5,
            population_size=50
        )
        
        # Load the session
        loaded_session = EvolutionSession.load_from_id(session_id, base_dir=self.temp_dir)
        
        # Check that the session was loaded correctly
        self.assertEqual(loaded_session.session_id, session_id)
        self.assertEqual(loaded_session.task_type, "test")
        self.assertEqual(loaded_session.metadata["generations"], 1)
        self.assertEqual(loaded_session.metadata["best_fitness"], 0.75)


class TestProgemonTrainer(unittest.TestCase):
    """Tests for the ProgemonTrainer component."""
    
    def setUp(self):
        """Set up test environment."""
        # Mock external dependencies
        self.mock_engine = MagicMock()
        self.mock_evaluator = MagicMock()
        self.mock_enforcer = MagicMock()
        
        # Create a temporary directory for output
        self.temp_dir = tempfile.mkdtemp()
        
        # Create the trainer
        self.trainer = ProgemonTrainer()
        
        # Override output directory
        self.trainer.settings["output_dir"] = self.temp_dir
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_parse_request(self):
        """Test parsing a user request."""
        description = "Create a bluetooth scanner to find nearby devices"
        settings = self.trainer.parse_request(description)
        
        self.assertEqual(settings["task"], "bluetooth_scan")
        self.assertEqual(settings["description"], description)
        
        # Check that task-specific settings were applied
        self.assertEqual(settings["ethics_level"], "strict")
    
    @patch('progremon_enhanced.EvolutionEngine')
    @patch('progremon_enhanced.FitnessEvaluator')
    @patch('progremon_enhanced.EthicalBoundaryEnforcer')
    def test_configure_ethical_boundaries(self, mock_enforcer_class, mock_evaluator_class, mock_engine_class):
        """Test configuring ethical boundaries."""
        # Create mocks
        mock_enforcer = mock_enforcer_class.return_value
        
        # Configure ethical boundaries
        settings = {
            "ethics_level": "strict",
            "task": "bluetooth_scan"
        }
        self.trainer._configure_ethical_boundaries(mock_enforcer, settings)
        
        # Check that the enforcer was configured
        mock_enforcer.set_level.assert_called_with("strict")
        
        # Check that task-specific restrictions were added
        self.assertTrue(mock_enforcer.add_restriction.called)
        self.assertTrue(mock_enforcer.set_resource_limit.called)
        self.assertTrue(mock_enforcer.enable_progressive_ethics.called)
    
    @patch('progremon_enhanced.EvolutionEngine')
    @patch('progremon_enhanced.FitnessEvaluator')
    @patch('progremon_enhanced.EthicalBoundaryEnforcer')
    def test_run_evolution_basic(self, mock_enforcer_class, mock_evaluator_class, mock_engine_class):
        """Test running a simple evolution process."""
        # Configure mocks
        mock_engine = mock_engine_class.return_value
        mock_engine.get_best_fitness.return_value = 0.8
        mock_engine.get_average_fitness.return_value = 0.6
        mock_engine.calculate_diversity.return_value = 0.5
        mock_best_solution = MagicMock()
        mock_best_solution.code = "def test(): pass"
        mock_best_solution.fitness = 0.8
        mock_engine.get_best_solution.return_value = mock_best_solution
        
        # Run evolution
        settings = {
            "output_dir": self.temp_dir,
            "task": "general",
            "description": "Generate a test function",
            "pop_size": 10,
            "gens": 2,
            "mutation_rate": 0.1,
            "crossover_rate": 0.7,
            "ethics_level": "basic"
        }
        success = self.trainer.run_evolution(settings)
        
        # Check that evolution was successful
        self.assertTrue(success)
        
        # Check that a session was created
        self.assertIsNotNone(self.trainer.session)
        
        # Check that the engine was properly configured
        mock_engine_class.assert_called_once()
        self.assertTrue(mock_engine.initialize_population.called)
        self.assertEqual(mock_engine.create_next_generation.call_count, settings["gens"])
        
        # Check that a solution was saved
        self.assertTrue(any(Path(self.temp_dir).glob("**/best_solution/*.py")))


def run_tests():
    """Run all tests."""
    print_color("\nRunning tests for Enhanced Progremon System", Colors.CYAN, bold=True)
    
    # Create test suite
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestTaskTemplateLoader))
    suite.addTest(unittest.makeSuite(TestEvolutionSession))
    suite.addTest(unittest.makeSuite(TestProgemonTrainer))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)


if __name__ == "__main__":
    run_tests()