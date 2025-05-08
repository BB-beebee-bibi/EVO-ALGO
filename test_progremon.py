def main():
    # Simulate user input for the Progremon interface
    request = "tell me what all bluetooth devices within range are, including signal strength and device type. Update every 0.20 seconds, results should be formatted as an easy-to-read table"
    
    # Run Progremon with simulated input
    import progremon
    trainer = progremon.ProgemonTrainer()
    
    # Simulate the main loop
    print("\nWhat would you like to create?")
    print(request)
    
    # Process the request
    config = trainer.process_request(request)
    settings = trainer.configure_evolution(config)
    
    # Simulate user confirmation
    print("\nWould you like to customize any of these settings? (y/n)")
    print("n")
    
    # Simulate the main loop with the 'n' response
    trainer._customize_settings(settings)
    
    # Run evolution
    if trainer.run_evolution(settings):
        print("\nEvolution complete! Check the output directory for results.")

if __name__ == "__main__":
    main()
