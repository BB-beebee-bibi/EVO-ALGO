#!/usr/bin/env python
"""
Progremon System Diagnostic Script

This script tests the basic functionality of the enhanced Progremon system
by attempting imports and basic operations. It helps identify integration
issues and missing dependencies.
"""

import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('progremon_diagnosis')


def check_directories():
    """Check if all required directories exist."""
    logger.info("Checking required directories...")
    
    required_dirs = [
        "guidance",
        "logs"
    ]
    
    for directory in required_dirs:
        path = Path(directory)
        if not path.exists():
            logger.warning(f"Directory '{directory}' does not exist. Creating it...")
            path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")
        else:
            logger.info(f"Found directory: {directory}")


def test_imports():
    """Test importing all key components."""
    logger.info("Testing module imports...")
    
    import_tests = [
        ("utils module", "from utils import Colors, print_color, print_banner, setup_logging, generate_unique_id"),
        ("task_template_loader module", "from task_template_loader import TaskTemplateLoader"),
        ("evolution_session module", "from evolution_session import EvolutionSession"),
        ("progremon_enhanced module", "from progremon_enhanced import ProgemonTrainer"),
        ("AdaptiveTweaker module", "from adaptive_tweaker_fix import AdaptiveTweaker, EvolutionMetrics"),
        ("Trisolaris core modules", "from trisolaris.core import EvolutionEngine, CodeGenome"),
        ("Trisolaris evaluation modules", "from trisolaris.evaluation import FitnessEvaluator"),
        ("Trisolaris boundary enforcer", "from trisolaris.evaluation.boundary_enforcer import EthicalBoundaryEnforcer")
    ]
    
    for test_name, import_statement in import_tests:
        try:
            logger.info(f"Testing {test_name}...")
            exec(import_statement)
            logger.info(f"✅ Successfully imported {test_name}")
        except Exception as e:
            logger.error(f"❌ Failed to import {test_name}: {e}")


def test_template_loading():
    """Test loading templates."""
    logger.info("Testing template loading...")
    
    try:
        from task_template_loader import TaskTemplateLoader
        
        loader = TaskTemplateLoader()
        
        # Test task detection
        test_descriptions = [
            "Create a bluetooth scanner to find nearby devices",
            "Generate a sorting algorithm"
        ]
        
        for desc in test_descriptions:
            task_type = loader.get_task_type_from_description(desc)
            logger.info(f"Task type for '{desc}': {task_type}")
        
        # Test template loading
        template = loader.load_template("bluetooth_scan")
        if template:
            logger.info(f"✅ Successfully loaded bluetooth_scan template ({len(template)} bytes)")
            # Check if it contains the expected class
            if "BluetoothScanner" in template:
                logger.info("✅ Template contains BluetoothScanner class")
            else:
                logger.warning("❌ Template does not contain BluetoothScanner class")
        else:
            logger.error("❌ Failed to load bluetooth_scan template")
            
    except Exception as e:
        logger.error(f"❌ Error testing template loading: {e}")


def test_session_management():
    """Test session management."""
    logger.info("Testing session management...")
    
    try:
        from evolution_session import EvolutionSession
        import shutil
        
        # Create a test directory
        test_dir = Path("test_diagnose_output")
        test_dir.mkdir(exist_ok=True)
        
        # Create a session
        session = EvolutionSession(base_dir=test_dir, task_type="test")
        logger.info(f"✅ Successfully created session with ID: {session.session_id}")
        
        # Test recording a generation
        session.record_generation(
            generation_number=0,
            best_fitness=0.5,
            avg_fitness=0.3,
            population_size=10
        )
        logger.info("✅ Successfully recorded generation")
        
        # Test saving a solution
        code = "def test(): pass"
        solution_path = session.save_solution(
            code=code,
            generation=0,
            fitness=0.5,
            is_best=True
        )
        logger.info(f"✅ Successfully saved solution to {solution_path}")
        
        # Test finalizing
        session.finalize()
        logger.info("✅ Successfully finalized session")
        
        # Clean up test directory
        shutil.rmtree(test_dir)
        logger.info("Cleaned up test directory")
        
    except Exception as e:
        logger.error(f"❌ Error testing session management: {e}")


def test_adaptive_tweaker():
    """Test adaptive tweaker integration."""
    logger.info("Testing AdaptiveTweaker integration...")
    
    try:
        try:
            from adaptive_tweaker_fix import AdaptiveTweaker, EvolutionMetrics
            logger.info("✅ Successfully imported AdaptiveTweaker from fixed module")
            using_fix = True
        except ImportError:
            logger.warning("Could not import from adaptive_tweaker_fix, trying original module")
            from adaptive_tweaker import AdaptiveTweaker, EvolutionMetrics
            logger.info("✅ Successfully imported AdaptiveTweaker from original module")
            using_fix = False
        
        # Create an instance
        settings = {
            "pop_size": 50,
            "gens": 25,
            "mutation_rate": 0.15,
            "crossover_rate": 0.7
        }
        
        tweaker = AdaptiveTweaker(settings)
        logger.info("✅ Successfully created AdaptiveTweaker instance")
        
        # Reset and set initial params
        tweaker.reset()
        tweaker.set_initial_params(settings)
        logger.info("✅ Successfully reset and set initial parameters")
        
        # Test parameter adjustment
        params = tweaker.adjust_parameters(
            generation=0,
            best_fitness=0.5,
            avg_fitness=0.3,
            diversity=0.7
        )
        logger.info(f"✅ Successfully adjusted parameters: {params}")
        logger.info(f"Using {'fixed' if using_fix else 'original'} AdaptiveTweaker implementation")
        
    except Exception as e:
        logger.error(f"❌ Error testing AdaptiveTweaker integration: {e}")


def test_progremon_trainer():
    """Test ProgemonTrainer basic functionality."""
    logger.info("Testing ProgemonTrainer basic functionality...")
    
    try:
        from progremon_enhanced import ProgemonTrainer
        
        # Create an instance
        trainer = ProgemonTrainer()
        logger.info("✅ Successfully created ProgemonTrainer instance")
        
        # Test parsing request
        description = "Create a simple bluetooth scanner"
        settings = trainer.parse_request(description)
        
        logger.info(f"✅ Successfully parsed request: {settings['task']}")
        
        # Don't run full evolution as it might take too long
        logger.info("Skipping full evolution test")
        
    except Exception as e:
        logger.error(f"❌ Error testing ProgemonTrainer: {e}")


def main():
    """Run all diagnostic tests."""
    logger.info("Starting Progremon system diagnostics...")
    
    # Run tests
    check_directories()
    test_imports()
    test_template_loading()
    test_session_management()
    test_adaptive_tweaker()
    test_progremon_trainer()
    
    logger.info("Diagnostic tests completed!")


if __name__ == "__main__":
    main()