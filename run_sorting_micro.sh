#!/bin/bash

# Run the text sorting evolution with specified parameters
python -m trisolaris.examples.text_sorting_evolution \
  --pop 30 \
  --gens 15 \
  --mutation_rate 0.3 \
  --crossover_rate 0.7 \
  --log_every 1 \
  --output_csv evolution_stats.csv

# Print summary
echo "Evolution run complete. Results saved to evolution_stats.csv"
echo "Fitness progression:"
tail -n +2 evolution_stats.csv | awk -F',' '{print "Generation", $1, "Min:", $2, "Mean:", $3, "Max:", $4}' 