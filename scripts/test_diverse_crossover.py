"""
Strategic diagnostic test for genetic crossover using paradigmatically diverse parents.
This script tests whether the crossover operator can produce novel offspring when given
radically different parent programs implementing opposing algorithmic paradigms.
"""
import os
import sys
from pathlib import Path
import ast
import hashlib
import logging
from typing import Dict, List, Tuple, Set

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from trisolaris.core.program_representation import ProgramAST, validate_ast

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("diverse_crossover_test")

# Create two algorithmically diverse sorting implementations
BUILTIN_SORT = """
def sort_files(file_list):
    return sorted(file_list)
"""

BUBBLE_SORT = """
def sort_files(file_list):
    n = len(file_list)
    for i in range(n):
        for j in range(0, n-i-1):
            if file_list[j] > file_list[j+1]:
                file_list[j], file_list[j+1] = file_list[j+1], file_list[j]
    return file_list
"""

def bytehash(node: ast.AST) -> str:
    """Generate a hash of the AST node for comparison."""
    return hashlib.sha1(ast.unparse(node).encode()).hexdigest()

def analyze_ast_structure(tree: ast.AST) -> Dict[str, int]:
    """Analyze the structural composition of an AST."""
    node_counts = {}
    for node in ast.walk(tree):
        node_type = type(node).__name__
        node_counts[node_type] = node_counts.get(node_type, 0) + 1
    return node_counts

def main():
    """Run the diverse crossover test."""
    logger.info("Starting diverse crossover test with paradigmatically different parents")
    
    # Parse source code into ASTs
    p1_ast = ast.parse(BUILTIN_SORT)
    p2_ast = ast.parse(BUBBLE_SORT)
    
    # Create parent programs
    p1 = ProgramAST(ast_tree=p1_ast)
    p2 = ProgramAST(ast_tree=p2_ast)
    
    # Verify parent diversity
    logger.info("\nParent algorithms:")
    logger.info(f"Parent 1 (built-in sort):\n{ast.unparse(p1.ast_tree)}\n")
    logger.info(f"Parent 2 (bubble sort):\n{ast.unparse(p2.ast_tree)}\n")
    
    # Compare parent hashes
    p1_hash = bytehash(p1.ast_tree)
    p2_hash = bytehash(p2.ast_tree)
    logger.info("Parent hashes:")
    logger.info(f"• p1 (builtin): {p1_hash}")
    logger.info(f"• p2 (bubble): {p2_hash}")
    logger.info(f"Parents identical: {p1_hash == p2_hash}")
    
    # Analyze AST structure differences
    p1_nodes = analyze_ast_structure(p1.ast_tree)
    p2_nodes = analyze_ast_structure(p2.ast_tree)
    
    logger.info("\nStructural composition:")
    for node_type in sorted(set(list(p1_nodes.keys()) + list(p2_nodes.keys()))):
        p1_count = p1_nodes.get(node_type, 0)
        p2_count = p2_nodes.get(node_type, 0)
        logger.info(f"  {node_type}: {p1_count} vs {p2_count}")
    
    # Run multiple crossovers and analyze offspring
    novel = 0
    identical_to_p1 = 0
    identical_to_p2 = 0
    total = 0
    novel_examples = []
    validation_failures = 0
    
    logger.info("\nRunning crossover experiments...")
    for i in range(50):
        try:
            c1, c2 = ProgramAST.crossover(p1, p2)
            
            for kid in (c1, c2):
                total += 1
                h = bytehash(kid.ast_tree)
                
                # Verify offspring validity
                is_valid, error = validate_ast(kid.ast_tree)
                if not is_valid:
                    validation_failures += 1
                    logger.error(f"Invalid offspring: {error}")
                    continue
                
                if h == p1_hash:
                    identical_to_p1 += 1
                    logger.debug(f"Offspring {total} identical to Parent 1")
                elif h == p2_hash:
                    identical_to_p2 += 1
                    logger.debug(f"Offspring {total} identical to Parent 2")
                else:
                    novel += 1
                    logger.info(f"Novel offspring {total} produced!")
                    # Store first 3 novel examples
                    if len(novel_examples) < 3:
                        novel_examples.append(ast.unparse(kid.ast_tree))
                        
        except Exception as e:
            logger.error(f"Crossover attempt {i+1} failed with error: {e}")
            continue
    
    # Report results
    logger.info(f"\nCROSSOVER RESULTS:")
    logger.info(f"Total offspring: {total}")
    logger.info(f"Identical to Parent 1: {identical_to_p1} ({identical_to_p1/total*100:.1f}%)")
    logger.info(f"Identical to Parent 2: {identical_to_p2} ({identical_to_p2/total*100:.1f}%)")
    logger.info(f"Novel offspring: {novel} ({novel/total*100:.1f}%)")
    logger.info(f"Validation failures: {validation_failures}")
    
    if novel == 0:
        logger.warning("\n❌ FAILURE: All offspring identical to parents. Crossover appears non-functional.")
        logger.warning("Potential root causes:")
        logger.warning("1. get_compatible_subtrees() may be too restrictive")
        logger.warning("2. validate_ast() might reject valid crossovers")
        logger.warning("3. Subtree replacement logic may have errors")
        logger.warning("4. Node type compatibility checks may be overly conservative")
    else:
        logger.info("\n✅ SUCCESS: Crossover produced novel offspring!")
        logger.info("\nEXAMPLE NOVEL OFFSPRING:")
        for i, example in enumerate(novel_examples):
            logger.info(f"\nNovel #{i+1}:\n{example}")

if __name__ == "__main__":
    main() 