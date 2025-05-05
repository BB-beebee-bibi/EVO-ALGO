#!/usr/bin/env python3
"""
Integration tests for the Bluetooth Scanner Task
"""
import os
import sys
import unittest
import tempfile
import json
from unittest.mock import patch, MagicMock

# Add the parent directory to the path so we can import trisolaris modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from trisolaris.tasks import BluetoothScannerTask
from trisolaris.core import CodeGenome, EvolutionEngine
from trisolaris.evaluation import FitnessEvaluator


# Sample Bluetooth device data for mocking
SAMPLE_DEVICES = [
    {
        "address": "00:11:22:33:44:55",
        "name": "Test Device 1",
        "rssi": -65,
        "device_type": "AUDIO",
        "services": ["0000110b-0000-1000-8000-00805f9b34fb"],
        "vulnerabilities": ["CVE-2017-0785"],
    },
    {
        "address": "AA:BB:CC:DD:EE:FF",
        "name": "Test Device 2",
        "rssi": -72,
        "device_type": "PHONE",
        "services": ["0000110c-0000-1000-8000-00805f9b34fb"],
        "vulnerabilities": [],
    }
]


class TestBluetoothScannerTask(unittest.TestCase):
    """Integration tests for the Bluetooth Scanner Task."""
    
    def setUp(self):
        """Set up test environment."""
        self.task = BluetoothScannerTask()
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test environment."""
        # Remove temporary directory
        import shutil
        shutil.rmtree(self.temp_dir)
    
    @patch('subprocess.run')
    @patch('bluepy.btle.Scanner')
    def test_task_template(self, mock_scanner, mock_subprocess):
        """Test that the task provides a valid template."""
        # Get the template
        template_code = self.task.get_template()
        
        # Verify it's a string with content
        self.assertIsInstance(template_code, str)
        self.assertGreater(len(template_code), 100)
        
        # Create a genome from the template 
        genome = CodeGenome.from_source(template_code)
        
        # Should be valid Python code
        self.assertIsNotNone(genome)
    
    def test_task_evaluation_parameters(self):
        """Test task evaluation parameters."""
        # Check evolution parameters
        params = self.task.get_evolution_params()
        
        self.assertIsInstance(params, dict)
        self.assertIn('population_size', params)
        self.assertIn('num_generations', params)
        self.assertIn('mutation_rate', params)
        self.assertIn('crossover_rate', params)
        
        # Check fitness weights
        weights = self.task.get_fitness_weights()
        
        self.assertIsInstance(weights, dict)
        self.assertIn('functionality', weights)
        self.assertIn('efficiency', weights)
        self.assertIn('alignment', weights)
        
        # Weights should sum to approximately 1.0
        total_weight = sum(weights.values())
        self.assertAlmostEqual(total_weight, 1.0, delta=0.01)
    
    def test_task_boundaries(self):
        """Test that the task defines ethical boundaries."""
        boundaries = self.task.get_required_boundaries()
        
        self.assertIsInstance(boundaries, dict)
        self.assertGreater(len(boundaries), 0)
        
        # Should define allowed imports
        imports = self.task.get_allowed_imports()
        self.assertIsInstance(imports, list)
        self.assertGreater(len(imports), 0)
    
    @patch('subprocess.run')
    def test_fitness_evaluation(self, mock_subprocess):
        """Test the fitness evaluation with a basic scanner."""
        # Mock discovery of devices
        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stdout = json.dumps(SAMPLE_DEVICES)
        
        # Create a basic bluetooth scanner code
        basic_code = """
import json
import subprocess

def scan_bluetooth_devices():
    devices = []
    try:
        # Mock scanning by returning sample data
        devices = [{
            "address": "00:11:22:33:44:55",
            "name": "Test Device 1",
            "rssi": -65,
            "device_type": "AUDIO"
        }]
    except Exception as e:
        print(f"Error: {e}")
    return devices

def main():
    devices = scan_bluetooth_devices()
    print(json.dumps(devices))

if __name__ == "__main__":
    main()
"""
        
        # Evaluate the code
        fitness, _ = self.task.evaluate_fitness(basic_code)
        
        # Should return a valid fitness score
        self.assertIsInstance(fitness, float)
        self.assertGreaterEqual(fitness, 0.0)
        self.assertLessEqual(fitness, 1.0)
    
    def test_post_processing(self):
        """Test post-processing of evolved code."""
        # Sample code to post-process
        code = """
def scan_bluetooth_devices():
    return []

def main():
    scan_bluetooth_devices()
"""
        
        # Post-process the code
        processed_code = self.task.post_process(code)
        
        # Should still be a string and contain the original code
        self.assertIsInstance(processed_code, str)
        self.assertIn("scan_bluetooth_devices", processed_code)
        
        # Should add imports and shebang if they're missing
        self.assertIn("#!/usr/bin", processed_code)
    
    @patch('bluepy.btle.Scanner')
    def test_evolution_integration(self, mock_scanner):
        """Test integrating the task with the evolution engine."""
        # Setup mock to return sample devices
        mock_device1 = MagicMock()
        mock_device1.addr = SAMPLE_DEVICES[0]["address"]
        mock_device1.addrType = "public"
        mock_device1.rssi = SAMPLE_DEVICES[0]["rssi"]
        mock_device1.getValueText.return_value = SAMPLE_DEVICES[0]["name"]
        
        mock_device2 = MagicMock()
        mock_device2.addr = SAMPLE_DEVICES[1]["address"]
        mock_device2.addrType = "public"
        mock_device2.rssi = SAMPLE_DEVICES[1]["rssi"]
        mock_device2.getValueText.return_value = SAMPLE_DEVICES[1]["name"]
        
        mock_scanner.return_value.scan.return_value = [mock_device1, mock_device2]
        
        # Create a small integration test with the evolution engine
        evaluator = FitnessEvaluator()
        
        # Add a custom alignment measure that uses the task's fitness evaluation
        def evaluate_task_fitness(genome):
            if hasattr(genome, 'to_source'):
                code = genome.to_source()
            else:
                code = str(genome)
            
            fitness, _ = self.task.evaluate_fitness(code)
            return fitness
        
        evaluator.add_alignment_measure(
            evaluate_task_fitness,
            weight=1.0,
            name="bluetooth_scanner_fitness"
        )
        
        # Create a small engine with the task's template
        engine = EvolutionEngine(
            population_size=5,
            evaluator=evaluator,
            mutation_rate=0.2,
            crossover_rate=0.7,
            genome_class=CodeGenome
        )
        
        # Create the initial population from the template
        template = self.task.get_template()
        base_genome = CodeGenome.from_source(template)
        
        # Create population with template and variants
        engine.population = [base_genome.clone() for _ in range(5)]
        for i in range(1, 5):
            engine.population[i].mutate(mutation_rate=0.1 * i)
        
        # Run a single generation
        engine.evaluate_population()
        
        # Should have fitness values
        for genome in engine.population:
            self.assertTrue(hasattr(genome, 'fitness'))
            self.assertIsNotNone(genome.fitness)
        
        # Select parents and create offspring
        parents = engine.select_parents()
        offspring = engine.create_offspring(parents)
        
        # Should have created offspring
        self.assertGreater(len(offspring), 0)


if __name__ == "__main__":
    unittest.main() 