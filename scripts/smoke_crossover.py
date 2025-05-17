"""
Diagnostic script for analyzing AST-based crossover operator behavior.
This script helps identify why crossover might be failing to produce novel offspring.
"""
import os
import sys
from pathlib import Path
import ast
import random
import hashlib
import logging
from typing import Dict, List, Tuple, Set

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from trisolaris.core.program_representation import ProgramAST, validate_ast
from trisolaris.core.ast_helpers import get_all_nodes, get_leaf_nodes

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("crossover_diagnosis")

def bytehash(node: ast.AST) -> str:
    """Generate a hash of the AST node for comparison."""
    return hashlib.sha1(ast.unparse(node).encode()).hexdigest()

def node_count_by_type(ast_node: ast.AST) -> Dict[str, int]:
    """Count nodes by type to characterize AST complexity."""
    type_counts = {}
    for node in ast.walk(ast_node):
        node_type = type(node).__name__
        type_counts[node_type] = type_counts.get(node_type, 0) + 1
    return type_counts

def analyze_subtree_compatibility(parent1: ProgramAST, parent2: ProgramAST) -> Dict[str, int]:
    """Analyze the compatibility of subtrees between parents."""
    compatibility_stats = {
        'total_nodes_p1': 0,
        'total_nodes_p2': 0,
        'compatible_pairs': 0,
        'compatible_types': set()
    }
    
    # Get all nodes from both parents
    nodes1 = get_all_nodes(parent1.ast_tree)
    nodes2 = get_all_nodes(parent2.ast_tree)
    
    compatibility_stats['total_nodes_p1'] = len(nodes1)
    compatibility_stats['total_nodes_p2'] = len(nodes2)
    
    # Find compatible pairs
    for node1 in nodes1:
        # Skip module and function definition nodes
        if isinstance(node1, (ast.Module, ast.FunctionDef)):
            continue
            
        # Count nodes of same type in parent2
        compatible_nodes = [n for n in nodes2 
                          if isinstance(n, type(node1)) 
                          and not isinstance(n, (ast.Module, ast.FunctionDef))]
        
        if compatible_nodes:
            compatibility_stats['compatible_pairs'] += len(compatible_nodes)
            compatibility_stats['compatible_types'].add(type(node1).__name__)
    
    return compatibility_stats

def analyze_crossover_attempts(parent1: ProgramAST, parent2: ProgramAST, num_attempts: int = 50) -> Dict[str, int]:
    """Analyze multiple crossover attempts and collect statistics."""
    stats = {
        'total_attempts': 0,
        'successful_crossovers': 0,
        'identical_to_p1': 0,
        'identical_to_p2': 0,
        'novel_offspring': 0,
        'validation_failures': 0,
        'compatible_pairs': 0  # Add this to track compatible pairs
    }
    
    p1_hash = bytehash(parent1.ast_tree)
    p2_hash = bytehash(parent2.ast_tree)
    
    # Get initial compatible pairs count
    compatibility_stats = analyze_subtree_compatibility(parent1, parent2)
    stats['compatible_pairs'] = compatibility_stats['compatible_pairs']
    
    for attempt in range(num_attempts):
        stats['total_attempts'] += 1
        try:
            logger.debug(f"\nCrossover attempt {attempt + 1}:")
            c1, c2 = ProgramAST.crossover(parent1, parent2)
            
            # Analyze each child
            for child_idx, child in enumerate((c1, c2)):
                child_hash = bytehash(child.ast_tree)
                is_valid, error = validate_ast(child.ast_tree)
                
                if not is_valid:
                    stats['validation_failures'] += 1
                    logger.debug(f"Child {child_idx + 1} validation failed: {error}")
                    continue
                
                if child_hash == p1_hash:
                    stats['identical_to_p1'] += 1
                    logger.debug(f"Child {child_idx + 1} identical to Parent 1")
                elif child_hash == p2_hash:
                    stats['identical_to_p2'] += 1
                    logger.debug(f"Child {child_idx + 1} identical to Parent 2")
                else:
                    stats['novel_offspring'] += 1
                    stats['successful_crossovers'] += 1
                    logger.info(f"Child {child_idx + 1} is novel offspring with hash: {child_hash}")
                    
        except Exception as e:
            logger.error(f"Crossover attempt {attempt + 1} failed with error: {e}")
            continue
    
    return stats

def main():
    """Run the diagnostic analysis."""
    logger.info("Starting crossover diagnostic analysis")
    
    # Create parent programs
    p1 = ProgramAST()
    p2 = ProgramAST()
    # Apply multiple mutations to p2 to ensure diversity
    for _ in range(3):
        p2 = p2.mutate(mutation_rate=1.0)

    # Log source code for visual confirmation
    logger.info(f"P1 source:\n{p1.to_source()}")
    logger.info(f"P2 source:\n{p2.to_source()}")
    
    # Verify parent diversity
    p1_hash = bytehash(p1.ast_tree)
    p2_hash = bytehash(p2.ast_tree)
    logger.info(f"Parent hashes: {p1_hash=}, {p2_hash=}")
    logger.info(f"Parents identical: {p1_hash == p2_hash}")
    
    # Print structural characteristics
    logger.info(f"P1 node distribution: {node_count_by_type(p1.ast_tree)}")
    logger.info(f"P2 node distribution: {node_count_by_type(p2.ast_tree)}")
    
    # Analyze subtree compatibility
    compatibility_stats = analyze_subtree_compatibility(p1, p2)
    logger.info("\nSubtree Compatibility Analysis:")
    logger.info(f"Total nodes in P1: {compatibility_stats['total_nodes_p1']}")
    logger.info(f"Total nodes in P2: {compatibility_stats['total_nodes_p2']}")
    logger.info(f"Compatible subtree pairs: {compatibility_stats['compatible_pairs']}")
    logger.info(f"Compatible node types: {compatibility_stats['compatible_types']}")
    
    # Analyze crossover attempts
    crossover_stats = analyze_crossover_attempts(p1, p2)
    logger.info("\nCrossover Analysis:")
    logger.info(f"Total attempts: {crossover_stats['total_attempts'] * 2}")  # *2 because each attempt produces 2 children
    logger.info(f"Successful crossovers: {crossover_stats['successful_crossovers']}")
    logger.info(f"Identical to Parent 1: {crossover_stats['identical_to_p1']} ({crossover_stats['identical_to_p1']/(crossover_stats['total_attempts']*2)*100:.1f}%)")
    logger.info(f"Identical to Parent 2: {crossover_stats['identical_to_p2']} ({crossover_stats['identical_to_p2']/(crossover_stats['total_attempts']*2)*100:.1f}%)")
    logger.info(f"Novel offspring: {crossover_stats['novel_offspring']} ({crossover_stats['novel_offspring']/(crossover_stats['total_attempts']*2)*100:.1f}%)")
    logger.info(f"Validation failures: {crossover_stats['validation_failures']}")
    
    # Print summary
    if crossover_stats['novel_offspring'] > 0:
        logger.info("\n✅ Diagnostic complete: Novel offspring are being generated")
    else:
        logger.warning("\n⚠️ Diagnostic complete: No novel offspring generated")
        if crossover_stats['compatible_pairs'] == 0:
            logger.warning("Root cause: No compatible subtrees found between parents")
        elif crossover_stats['validation_failures'] > 0:
            logger.warning("Root cause: Crossover produces invalid ASTs")
        else:
            logger.warning("Root cause: Crossover always falls back to parent copies")

if __name__ == "__main__":
    main() 