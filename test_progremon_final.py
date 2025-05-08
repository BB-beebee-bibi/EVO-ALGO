#!/usr/bin/env python
"""
Test script for the rewritten Progremon implementation.
"""

import os
import sys

def main():
    # Import the new implementation
    try:
        from progremon_final import ProgemonTrainer
        print("Successfully imported ProgemonTrainer from progremon_final")
    except ImportError as e:
        print(f"Error importing ProgemonTrainer: {e}")
        return False
    
    # Create a trainer instance
    trainer = ProgemonTrainer()
    print("Successfully created ProgemonTrainer instance")
    
    # Simulate a request for bluetooth scanning
    request = "Create a bluetooth scanner that finds all nearby devices with signal strength information, updates every 0.5 seconds, and displays the results in a table format"
    print(f"\nProcessing request: {request}")
    
    # Process the request
    config = trainer.process_request(request)
    print("\nDetected configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Configure with minimal settings for testing
    test_settings = {
        "pop_size": 5,      # Small population for quick testing
        "gens": 3,          # Few generations for quick testing
        "save_all_generations": True,
        "output_dir": "test_evolution_output",
        "task": config["task"],
        "description": config["description"],
        "mutation_rate": 0.1,
        "crossover_rate": 0.7,
        "ethics_level": "standard"
    }
    
    # Apply detected configurations
    for key in config:
        if key not in ["task", "description"]:
            test_settings[key] = config[key]
    
    print("\nTest settings:")
    for key, value in test_settings.items():
        print(f"  {key}: {value}")
    
    # Create output directory if it doesn't exist
    os.makedirs(test_settings["output_dir"], exist_ok=True)
    
    # Run a quick evolution
    print("\nRunning test evolution...")
    result = trainer.run_evolution(test_settings)
    
    if result:
        print("\nEvolution completed successfully!")
        
        # Check for the best solution file
        best_path = os.path.join(test_settings["output_dir"], trainer.session.session_id, "best_solution.py")
        if os.path.exists(best_path):
            print(f"Best solution saved to: {best_path}")
            
            # Display first few lines of the solution
            with open(best_path, "r") as f:
                code = f.read()
                lines = code.split("\n")[:10]
                print("\nSolution preview:")
                print("-" * 40)
                print("\n".join(lines))
                print("...")
                print("-" * 40)
        else:
            print(f"Warning: Best solution file not found at {best_path}")
    else:
        print("\nEvolution failed!")
    
    return result

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)