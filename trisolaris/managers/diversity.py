"""
Diversity Guardian for the TRISOLARIS framework.

This module implements mechanisms to track population diversity and maintain 
adequate genetic variation during the evolutionary process.
"""

import logging
import random
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Callable, Set

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DiversityGuardian:
    """
    Monitors and maintains genetic diversity in the evolving population.
    
    This class tracks population diversity metrics and implements strategies
    to prevent premature convergence by maintaining adequate genetic variation.
    """
    
    def __init__(
        self,
        min_diversity: float = 0.2,  # Minimum diversity threshold
        metrics: List[str] = None,    # Diversity metrics to track
        injection_strategies: List[str] = None,  # Strategies for diversity injection
        novelty_search_weight: float = 0.3,  # Weight for novelty in selection
    ):
        """
        Initialize the Diversity Guardian.
        
        Args:
            min_diversity: Minimum acceptable diversity threshold
            metrics: List of diversity metrics to track (default: 'genotypic', 'phenotypic')
            injection_strategies: Diversity injection strategies (default: 'mutation', 'immigration', 'restart')
            novelty_search_weight: Weight for novelty in fitness calculations
        """
        self.min_diversity = min_diversity
        self.metrics = metrics or ['genotypic', 'phenotypic']
        self.injection_strategies = injection_strategies or ['mutation', 'immigration', 'restart']
        self.novelty_search_weight = novelty_search_weight
        
        # Diversity history
        self.history = {
            'genotypic_diversity': [],
            'phenotypic_diversity': [],
            'unique_ratio': [],
            'injections': []
        }
        
        # Set of unique solution hashes for tracking
        self.unique_solutions = set()
        
        # Track how many times diversity has been injected
        self.injection_count = 0
        
        # Current strategy index
        self.current_strategy_idx = 0
    
    def measure_diversity(self, population) -> float:
        """
        Measure the diversity of a population using configured metrics.
        
        Args:
            population: List of genomes
            
        Returns:
            Composite diversity score (0.0 to 1.0)
        """
        if not population or len(population) < 2:
            return 0.0
        
        scores = []
        
        # Track unique solution hashes for unique ratio calculation
        solution_hashes = set()
        for genome in population:
            try:
                solution_hash = hash(genome.to_source())
                solution_hashes.add(solution_hash)
                self.unique_solutions.add(solution_hash)
            except:
                pass
        
        # Genotypic diversity (based on source code)
        if 'genotypic' in self.metrics:
            # Compare source code pairwise - more different = more diverse
            try:
                genotypic_diversity = self._calculate_genotypic_diversity(population)
                scores.append(genotypic_diversity)
                self.history['genotypic_diversity'].append(genotypic_diversity)
            except Exception as e:
                logger.warning(f"Error calculating genotypic diversity: {str(e)}")
        
        # Phenotypic diversity (based on behavior)
        if 'phenotypic' in self.metrics:
            try:
                phenotypic_diversity = self._calculate_phenotypic_diversity(population)
                scores.append(phenotypic_diversity)
                self.history['phenotypic_diversity'].append(phenotypic_diversity)
            except Exception as e:
                logger.warning(f"Error calculating phenotypic diversity: {str(e)}")
        
        # Calculate unique ratio
        unique_ratio = len(solution_hashes) / len(population)
        self.history['unique_ratio'].append(unique_ratio)
        scores.append(unique_ratio)
        
        # Average all available diversity metrics
        diversity_score = sum(scores) / len(scores) if scores else 0.0
        
        return diversity_score
    
    def _calculate_genotypic_diversity(self, population) -> float:
        """
        Calculate genotypic diversity based on code structure.
        
        Args:
            population: List of genomes
            
        Returns:
            Genotypic diversity score (0.0 to 1.0)
        """
        # Simple version: Compare source code differences
        if len(population) < 2:
            return 0.0
            
        total_diff = 0
        comparisons = 0
        
        # Compare each pair of genomes
        for i in range(len(population)):
            for j in range(i+1, len(population)):
                try:
                    source_i = population[i].to_source()
                    source_j = population[j].to_source()
                    
                    # Calculate normalized edit distance
                    diff = self._normalized_levenshtein_distance(source_i, source_j)
                    total_diff += diff
                    comparisons += 1
                except Exception as e:
                    logger.debug(f"Error comparing genomes {i} and {j}: {str(e)}")
                    continue
        
        if comparisons == 0:
            return 0.0
            
        return total_diff / comparisons
    
    def _calculate_phenotypic_diversity(self, population) -> float:
        """
        Calculate phenotypic diversity based on behavior signatures.
        
        Args:
            population: List of genomes
            
        Returns:
            Phenotypic diversity score (0.0 to 1.0)
        """
        # If genomes have behavior signatures, compare those
        # Otherwise, fall back to simple structural diversity
        
        has_signatures = False
        for genome in population:
            if hasattr(genome, 'behavior_signature'):
                has_signatures = True
                break
        
        if not has_signatures:
            # Fall back to structural diversity
            return self._calculate_structural_diversity(population)
        
        # Compare behavior signatures
        total_diff = 0
        comparisons = 0
        
        for i in range(len(population)):
            for j in range(i+1, len(population)):
                try:
                    sig_i = population[i].behavior_signature
                    sig_j = population[j].behavior_signature
                    
                    # Calculate signature difference
                    if isinstance(sig_i, (list, tuple)) and isinstance(sig_j, (list, tuple)):
                        # Handle vector signatures
                        sig_i_arr = np.array(sig_i)
                        sig_j_arr = np.array(sig_j)
                        diff = np.linalg.norm(sig_i_arr - sig_j_arr) / np.sqrt(len(sig_i))
                        diff = min(1.0, diff)  # Clamp to 1.0 max
                    else:
                        # Handle other signature types with simple equality check
                        diff = 0.0 if sig_i == sig_j else 1.0
                    
                    total_diff += diff
                    comparisons += 1
                except Exception as e:
                    logger.debug(f"Error comparing behavior signatures: {str(e)}")
                    continue
        
        if comparisons == 0:
            return 0.0
            
        return total_diff / comparisons
    
    def _calculate_structural_diversity(self, population) -> float:
        """
        Calculate diversity based on code structure characteristics.
        
        Args:
            population: List of genomes
            
        Returns:
            Structural diversity score (0.0 to 1.0)
        """
        # Extract structural features like AST node counts, function counts, etc.
        features = []
        for genome in population:
            try:
                # Collect various structural metrics
                source = genome.to_source()
                feature = {
                    'length': len(source),
                    'lines': source.count('\n') + 1,
                    'functions': source.count('def '),
                    'classes': source.count('class '),
                    'loops': source.count('for ') + source.count('while '),
                    'imports': source.count('import ') + source.count('from '),
                }
                features.append(feature)
            except Exception as e:
                logger.debug(f"Error extracting structural features: {str(e)}")
                # Add empty feature set to maintain indices
                features.append({
                    'length': 0, 'lines': 0, 'functions': 0,
                    'classes': 0, 'loops': 0, 'imports': 0
                })
        
        # Calculate diversity as coefficient of variation across features
        diversity_scores = []
        
        for feature_name in ['length', 'lines', 'functions', 'classes', 'loops', 'imports']:
            values = [f[feature_name] for f in features]
            if not values or sum(values) == 0:
                continue
                
            # Calculate coefficient of variation
            mean = sum(values) / len(values)
            if mean == 0:
                continue
                
            variance = sum((x - mean) ** 2 for x in values) / len(values)
            std_dev = variance ** 0.5
            cv = std_dev / mean if mean > 0 else 0
            
            # Convert to 0-1 score (higher CV = higher diversity)
            # CV can be > 1, so we use a mapping function
            diversity_score = min(1.0, cv / 2.0)
            diversity_scores.append(diversity_score)
        
        # Average across all feature dimensions
        return sum(diversity_scores) / len(diversity_scores) if diversity_scores else 0.0
    
    def _normalized_levenshtein_distance(self, s1: str, s2: str) -> float:
        """
        Calculate normalized Levenshtein distance between two strings.
        
        Args:
            s1: First string
            s2: Second string
            
        Returns:
            Normalized distance (0.0 to 1.0)
        """
        # For very long strings, use a sampling approach
        if len(s1) > 1000 or len(s2) > 1000:
            return self._sample_string_difference(s1, s2)
        
        # Dynamic programming implementation of Levenshtein distance
        m, n = len(s1), len(s2)
        
        # Handle edge cases
        if m == 0 or n == 0:
            return 1.0 if max(m, n) > 0 else 0.0
        
        # Create matrix
        d = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
        
        # Initialize first row and column
        for i in range(m + 1):
            d[i][0] = i
        for j in range(n + 1):
            d[0][j] = j
        
        # Fill matrix
        for j in range(1, n + 1):
            for i in range(1, m + 1):
                if s1[i - 1] == s2[j - 1]:
                    d[i][j] = d[i - 1][j - 1]
                else:
                    d[i][j] = min(
                        d[i - 1][j] + 1,      # Deletion
                        d[i][j - 1] + 1,      # Insertion
                        d[i - 1][j - 1] + 1   # Substitution
                    )
        
        # Normalize by the length of the longer string
        return d[m][n] / max(m, n)
    
    def _sample_string_difference(self, s1: str, s2: str, samples: int = 10) -> float:
        """
        Estimate string difference by sampling segments.
        
        Args:
            s1: First string
            s2: Second string
            samples: Number of segments to sample
            
        Returns:
            Estimated normalized difference (0.0 to 1.0)
        """
        total_diff = 0.0
        
        # Handle empty strings
        if not s1 or not s2:
            return 1.0 if s1 or s2 else 0.0
        
        # Handle different string lengths
        len_diff = abs(len(s1) - len(s2)) / max(len(s1), len(s2))
        total_diff += len_diff * 0.5  # Length difference contributes 50% to the score
        
        # Sample segments from both strings
        s1_segments = self._sample_segments(s1, samples)
        s2_segments = self._sample_segments(s2, samples)
        
        # Compare sampled segments
        segment_diffs = []
        for seg1, seg2 in zip(s1_segments, s2_segments):
            if seg1 and seg2:
                # Calculate Levenshtein distance for each segment pair
                segment_diff = self._normalized_levenshtein_distance(seg1, seg2)
                segment_diffs.append(segment_diff)
        
        # Average segment differences contribute 50% to the score
        if segment_diffs:
            total_diff += sum(segment_diffs) / len(segment_diffs) * 0.5
        else:
            total_diff += 0.5  # Default if no valid segments to compare
        
        return total_diff
    
    def _sample_segments(self, text: str, num_segments: int) -> List[str]:
        """
        Sample segments from a text string.
        
        Args:
            text: Source text
            num_segments: Number of segments to sample
            
        Returns:
            List of sampled text segments
        """
        segments = []
        segment_length = min(100, max(10, len(text) // 20))  # Adaptive segment length
        
        # Try to get key segments from the text structure
        lines = text.split('\n')
        if len(lines) >= num_segments:
            # Sample evenly across the file
            step = len(lines) // num_segments
            for i in range(0, len(lines), step):
                if len(segments) < num_segments:
                    segment = '\n'.join(lines[i:i+3])[:segment_length]  # Get a few lines
                    segments.append(segment)
        else:
            # Fall back to character-based sampling
            step = max(1, len(text) // num_segments)
            for i in range(0, len(text), step):
                if len(segments) < num_segments:
                    end = min(i + segment_length, len(text))
                    segments.append(text[i:end])
        
        return segments
    
    def calculate_novelty_scores(self, population) -> List[float]:
        """
        Calculate novelty scores for each individual in the population.
        
        Args:
            population: List of genomes
            
        Returns:
            List of novelty scores (0.0 to 1.0)
        """
        if not population or len(population) < 2:
            return [0.0] * len(population)
        
        novelty_scores = []
        
        for i, genome in enumerate(population):
            # Calculate average distance to k-nearest neighbors
            distances = []
            
            for j, other_genome in enumerate(population):
                if i == j:
                    continue
                
                try:
                    source_i = genome.to_source()
                    source_j = other_genome.to_source()
                    
                    # Calculate normalized edit distance
                    dist = self._normalized_levenshtein_distance(source_i, source_j)
                    distances.append(dist)
                except Exception as e:
                    logger.debug(f"Error comparing genomes {i} and {j}: {str(e)}")
                    continue
            
            # If we couldn't calculate any distances, assign zero novelty
            if not distances:
                novelty_scores.append(0.0)
                continue
                
            # Calculate average distance to k-nearest neighbors
            k = min(5, len(distances))
            if k > 0:
                distances.sort()  # Sort in ascending order
                avg_dist = sum(distances[:k]) / k
                novelty_scores.append(avg_dist)
            else:
                novelty_scores.append(0.0)
        
        return novelty_scores
    
    def adjust_fitness_with_novelty(self, population, fitness_scores: List[float]) -> List[float]:
        """
        Adjust fitness scores to incorporate novelty.
        
        Args:
            population: List of genomes
            fitness_scores: Original fitness scores
            
        Returns:
            Adjusted fitness scores incorporating novelty
        """
        if not population or len(population) < 2:
            return fitness_scores
            
        novelty_scores = self.calculate_novelty_scores(population)
        
        # Normalize novelty scores to 0-1 range
        max_novelty = max(novelty_scores) if novelty_scores else 1.0
        if max_novelty > 0:
            normalized_novelty = [n / max_novelty for n in novelty_scores]
        else:
            normalized_novelty = [0.0] * len(novelty_scores)
        
        # Combine with original fitness using the novelty weight
        w = self.novelty_search_weight
        adjusted_scores = [
            (1 - w) * fs + w * ns
            for fs, ns in zip(fitness_scores, normalized_novelty)
        ]
        
        return adjusted_scores
    
    def inject_diversity(self, population) -> List:
        """
        Inject diversity into the population using selected strategies.
        
        Args:
            population: List of genomes to diversify
            
        Returns:
            Updated population with injected diversity
        """
        if not population:
            return population
            
        self.injection_count += 1
        logger.info(f"Injecting diversity (#{self.injection_count})")
        
        # Choose next strategy in rotation
        strategy = self.injection_strategies[self.current_strategy_idx % len(self.injection_strategies)]
        self.current_strategy_idx += 1
        
        # Record the injection event
        self.history['injections'].append({
            'count': self.injection_count,
            'strategy': strategy,
            'population_size': len(population)
        })
        
        # Apply the selected strategy
        if strategy == 'mutation':
            return self._inject_with_mutation(population)
        elif strategy == 'immigration':
            return self._inject_with_immigration(population)
        elif strategy == 'restart':
            return self._inject_with_restart(population)
        else:
            logger.warning(f"Unknown diversity injection strategy: {strategy}")
            return population
    
    def _inject_with_mutation(self, population) -> List:
        """
        Inject diversity using higher mutation rates on a subset of the population.
        
        Args:
            population: List of genomes
            
        Returns:
            Updated population
        """
        # Select a subset of the population for high mutation
        mutation_candidates = random.sample(
            population,
            k=max(1, int(len(population) * 0.3))  # Mutate 30% of population
        )
        
        # Create new variants through increased mutation
        for genome in mutation_candidates:
            try:
                # Apply higher mutation rate
                genome.mutate(mutation_rate=0.5, mutate_structure=True)
            except Exception as e:
                logger.warning(f"Error during diversity mutation: {str(e)}")
        
        return population
    
    def _inject_with_immigration(self, population) -> List:
        """
        Inject diversity by introducing new random genomes.
        
        Args:
            population: List of genomes
            
        Returns:
            Updated population
        """
        # Determine how many immigrants to create
        num_immigrants = max(1, int(len(population) * 0.2))  # 20% immigration rate
        
        # Create new random genomes
        try:
            genome_class = population[0].__class__
            immigrants = [genome_class() for _ in range(num_immigrants)]
            
            # Replace the lowest fitness individuals
            population = sorted(population, key=lambda x: getattr(x, 'fitness', 0.0), reverse=True)
            population = population[:-num_immigrants] + immigrants
        except Exception as e:
            logger.warning(f"Error during diversity immigration: {str(e)}")
        
        return population
    
    def _inject_with_restart(self, population) -> List:
        """
        Inject diversity by partial restart with new random genomes but keeping the best.
        
        Args:
            population: List of genomes
            
        Returns:
            Updated population
        """
        try:
            # Keep the top performers
            num_keep = max(1, int(len(population) * 0.2))  # Keep top 20%
            
            # Sort by fitness
            population = sorted(population, key=lambda x: getattr(x, 'fitness', 0.0), reverse=True)
            top_performers = population[:num_keep]
            
            # Generate new random genomes
            genome_class = population[0].__class__
            new_genomes = [genome_class() for _ in range(len(population) - num_keep)]
            
            # Combine
            population = top_performers + new_genomes
            
        except Exception as e:
            logger.warning(f"Error during diversity restart: {str(e)}")
        
        return population
    
    def needs_diversity_injection(self, population) -> bool:
        """
        Check if the population needs diversity injection.
        
        Args:
            population: List of genomes
            
        Returns:
            True if diversity is too low and injection is needed
        """
        diversity = self.measure_diversity(population)
        return diversity < self.min_diversity
    
    def get_diversity_history(self) -> Dict[str, List]:
        """
        Get the history of diversity measurements and interventions.
        
        Returns:
            Dictionary with historical diversity data
        """
        return self.history
    
    def report(self) -> str:
        """
        Generate a summary report of diversity metrics.
        
        Returns:
            Diversity report as a string
        """
        lines = ["=== DIVERSITY REPORT ==="]
        
        # Add current thresholds
        lines.append(f"Minimum diversity threshold: {self.min_diversity}")
        lines.append(f"Novelty search weight: {self.novelty_search_weight}")
        lines.append(f"Active metrics: {', '.join(self.metrics)}")
        lines.append(f"Injection strategies: {', '.join(self.injection_strategies)}")
        
        # Add history stats
        if self.history['genotypic_diversity']:
            avg_geno = sum(self.history['genotypic_diversity']) / len(self.history['genotypic_diversity'])
            lines.append(f"Average genotypic diversity: {avg_geno:.4f}")
        
        if self.history['phenotypic_diversity']:
            avg_pheno = sum(self.history['phenotypic_diversity']) / len(self.history['phenotypic_diversity'])
            lines.append(f"Average phenotypic diversity: {avg_pheno:.4f}")
        
        if self.history['unique_ratio']:
            avg_unique = sum(self.history['unique_ratio']) / len(self.history['unique_ratio'])
            lines.append(f"Average unique solution ratio: {avg_unique:.4f}")
        
        # Add injection summary
        lines.append(f"Total diversity injections: {self.injection_count}")
        
        # Add trend indicators
        if len(self.history['genotypic_diversity']) >= 2:
            first_half = self.history['genotypic_diversity'][:len(self.history['genotypic_diversity'])//2]
            second_half = self.history['genotypic_diversity'][len(self.history['genotypic_diversity'])//2:]
            
            first_avg = sum(first_half) / len(first_half) if first_half else 0
            second_avg = sum(second_half) / len(second_half) if second_half else 0
            
            trend = "increasing" if second_avg > first_avg else "decreasing" if second_avg < first_avg else "stable"
            lines.append(f"Diversity trend: {trend}")
        
        lines.append("=====================")
        
        return "\n".join(lines)
    
    def __str__(self) -> str:
        """String representation of the diversity guardian."""
        activation_status = "active" if any(self.history.values()) else "initialized"
        return f"DiversityGuardian({activation_status}, min_diversity={self.min_diversity}, injections={self.injection_count})"
