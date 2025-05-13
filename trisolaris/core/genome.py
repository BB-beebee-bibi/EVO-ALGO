"""
CodeGenome representation for the TRISOLARIS framework.

This module provides the CodeGenome class that represents individual solutions 
in the form of code snippets or programs.
"""

import random
import copy
import ast
import logging
from typing import Tuple, List, Dict, Any, Optional, Union
import os

# Import modern AST utilities for Python 3.8+ compatibility
from trisolaris.core.ast_utils import ModernMutationTransformer, ModernAstCrossover, ModernSubtreeCollector
from trisolaris.core.syntax_validator import SyntaxValidator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CodeGenome:
    """
    Represents a single solution in the form of code.
    
    CodeGenome provides methods for initialization, mutation, crossover, and conversion
    between source code and abstract syntax tree (AST) representations.
    """
    
    def __init__(self, ast_tree=None, source_code=None):
        """
        Initialize a new CodeGenome.
        
        Args:
            ast_tree: Optional AST representation of the code
            source_code: Optional source code string
            
        Note: At least one of ast_tree or source_code should be provided,
              unless random initialization is desired.
        """
        if ast_tree:
            self.ast_tree = ast_tree
            self._source_code = None
        elif source_code:
            self._source_code = source_code
            try:
                self.ast_tree = ast.parse(source_code)
            except SyntaxError:
                # Fallback to storing just source code if parsing fails
                self.ast_tree = None
        else:
            # Create a minimal random function if nothing is provided
            self._create_random_genome()
    
    def _create_random_genome(self):
        """Create a simple random function as a starting point."""
        # Generate a simple function that returns a random value
        function_name = ''.join(random.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(5))
        num_params = random.randint(0, 3)
        param_names = [''.join(random.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(3)) 
                      for _ in range(num_params)]
        params = ", ".join(param_names)
        
        return_value = random.choice([
            "0", 
            "1", 
            "True", 
            "False", 
            '"Hello"', 
            f"{random.randint(0, 100)}"
        ])
        
        self._source_code = f"def {function_name}({params}):\n    return {return_value}"
        try:
            self.ast_tree = ast.parse(self._source_code)
        except SyntaxError:
            # Fallback if somehow the generated code is invalid
            self._source_code = "def fallback():\n    return 0"
            self.ast_tree = ast.parse(self._source_code)
    
    @classmethod
    def from_source(cls, source_code: str) -> 'CodeGenome':
        """
        Create a CodeGenome from source code.
        
        Args:
            source_code: The source code string
            
        Returns:
            A new CodeGenome instance
        """
        return cls(source_code=source_code)
    
    @classmethod
    def from_directory(cls, directory_path: str) -> 'CodeGenome':
        """
        Create a CodeGenome from all Python files in a directory.
        
        Args:
            directory_path: Path to the directory containing Python files
            
        Returns:
            A new CodeGenome instance with combined code from all files
        """
        # Check if directory exists
        if not os.path.isdir(directory_path):
            raise ValueError(f"Directory not found: {directory_path}")
        
        # Initialize empty combined source
        combined_source = f"# Combined code from {directory_path}\n\n"
        
        # Track files processed
        files_processed = []
        
        # Walk through directory and collect Python files
        for root, _, files in os.walk(directory_path):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            file_content = f.read()
                            rel_path = os.path.relpath(file_path, directory_path)
                            combined_source += f"# File: {rel_path}\n{file_content}\n\n"
                            files_processed.append(rel_path)
                    except Exception as e:
                        combined_source += f"# Error reading {file_path}: {str(e)}\n\n"
        
        if not files_processed:
            # If no Python files were found, look for text files that might contain code
            for root, _, files in os.walk(directory_path):
                for file in files:
                    if file.endswith(('.txt', '.md')):
                        file_path = os.path.join(root, file)
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                file_content = f.read()
                                if "def " in file_content or "class " in file_content:
                                    rel_path = os.path.relpath(file_path, directory_path)
                                    combined_source += f"# File: {rel_path}\n{file_content}\n\n"
                                    files_processed.append(rel_path)
                        except Exception:
                            pass
        
        # If still no files found, create a minimal sample
        if not files_processed:
            combined_source += (
                "# No Python files found, creating minimal example\n"
                "def process_directory(path):\n"
                f"    print('Processing directory: {directory_path}')\n"
                "    return {'status': 'success', 'files_found': 0}\n"
            )
        
        return cls(source_code=combined_source)
    
    def to_source(self) -> str:
        """
        Convert the genome to source code.
        
        Returns:
            Source code string
        """
        if self._source_code is None and self.ast_tree:
            # Generate source code from AST
            self._source_code = ast.unparse(self.ast_tree)
        
        return self._source_code
    
    def clone(self) -> 'CodeGenome':
        """
        Create a deep copy of this genome.
        
        Returns:
            A new CodeGenome instance with the same code
        """
        if self.ast_tree:
            return CodeGenome(ast_tree=copy.deepcopy(self.ast_tree))
        else:
            return CodeGenome(source_code=self._source_code)
    
    def mutate(self, rate: float = 0.1) -> None:
        """
        Apply random mutations to the genome with the given probability.
        
        Args:
            rate: Probability of mutation (0-1)
            
        Note: This modifies the genome in-place.
        """
        if not self.ast_tree:
            # Try to parse the source code if we don't have an AST
            try:
                self.ast_tree = ast.parse(self._source_code)
            except SyntaxError:
                # Can't mutate without a valid AST
                return
        
        # Apply AST-based mutations
        mutator = AstMutator(rate)
        self.ast_tree = mutator.mutate(self.ast_tree)
        
        # The source code is now outdated
        self._source_code = None
        
        # Validate and repair the code after mutation
        self._validate_and_repair_code()
    
    def crossover(self, other: 'CodeGenome') -> Tuple['CodeGenome', 'CodeGenome']:
        """
        Perform crossover with another genome.
        
        Args:
            other: Another CodeGenome to crossover with
            
        Returns:
            A tuple of two new CodeGenome instances (children)
        """
        if not self.ast_tree or not other.ast_tree:
            # Ensure both parents have valid ASTs
            if not self.ast_tree and self._source_code:
                try:
                    self.ast_tree = ast.parse(self._source_code)
                except SyntaxError:
                    # Return clones if we can't parse
                    return self.clone(), other.clone()
            
            if not other.ast_tree and other._source_code:
                try:
                    other.ast_tree = ast.parse(other._source_code)
                except SyntaxError:
                    # Return clones if we can't parse
                    return self.clone(), other.clone()
        
        # Create copies to avoid modifying the originals
        child1_ast = copy.deepcopy(self.ast_tree) if self.ast_tree else None
        child2_ast = copy.deepcopy(other.ast_tree) if other.ast_tree else None
        
        if child1_ast and child2_ast:
            # Perform AST-based crossover
            crossover_operator = AstCrossover()
            child1_ast, child2_ast = crossover_operator.crossover(child1_ast, child2_ast)
            
            # Create new genomes
            child1 = CodeGenome(ast_tree=child1_ast)
            child2 = CodeGenome(ast_tree=child2_ast)
            
            # Validate and repair the code after crossover
            child1._validate_and_repair_code()
            child2._validate_and_repair_code()
            
            return child1, child2
        else:
            # Fallback to source-level crossover if AST is not available
            src1 = self.to_source().split('\n')
            src2 = other.to_source().split('\n')
            
            # Simple line-based crossover
            crossover_point = min(len(src1) // 2, len(src2) // 2)
            
            child1_src = '\n'.join(src1[:crossover_point] + src2[crossover_point:])
            child2_src = '\n'.join(src2[:crossover_point] + src1[crossover_point:])
            
            # Create new genomes
            child1 = CodeGenome(source_code=child1_src)
            child2 = CodeGenome(source_code=child2_src)
            
            # Validate and repair the code after crossover
            child1._validate_and_repair_code()
            child2._validate_and_repair_code()
            
            return child1, child2


class AstMutator:
    """Helper class to apply mutations to AST trees."""
    
    def __init__(self, mutation_rate: float = 0.1):
        """
        Initialize the AST mutator.
        
        Args:
            mutation_rate: Probability of mutation for each node
        """
        self.mutation_rate = mutation_rate
    
    def mutate(self, tree: ast.AST) -> ast.AST:
        """
        Apply mutations to an AST tree.
        
        Args:
            tree: The AST to mutate
            
        Returns:
            Mutated AST
        """
        # Create a deep copy to avoid modifying the original
        tree_copy = copy.deepcopy(tree)
        
        # Apply mutations using the modern node transformer for future compatibility
        transformer = ModernMutationTransformer(self.mutation_rate)
        return ast.fix_missing_locations(transformer.visit(tree_copy))


class AstCrossover:
    """Implements crossover operations for AST trees."""
    
    def __init__(self):
        """Initialize with a modern crossover implementation."""
        self.modern_crossover = ModernAstCrossover()
    
    def crossover(self, tree1: ast.AST, tree2: ast.AST) -> Tuple[ast.AST, ast.AST]:
        """
        Perform crossover between two AST trees using the modern implementation.
        
        Args:
            tree1: First parent AST
            tree2: Second parent AST
            
        Returns:
            Tuple of two child ASTs
        """
        return self.modern_crossover.crossover(tree1, tree2)
    
    def _collect_subtrees(self, tree: ast.AST) -> List[Tuple[ast.AST, Union[str, list], Optional[int]]]:
        """
        Collect all valid subtrees for crossover using the modern implementation.
        
        Args:
            tree: The AST to analyze
            
        Returns:
            List of tuples (parent_node, field_name_or_list, index_or_None)
        """
        return self.modern_crossover._collect_subtrees(tree)


class SubtreeCollector(ast.NodeVisitor):
    """
    Collector for AST subtrees suitable for crossover.
    Thin wrapper around ModernSubtreeCollector for backward compatibility.
    """
    
    def __init__(self):
        """Initialize the collector with a modern implementation."""
        self.modern_collector = ModernSubtreeCollector()
        # Expose the subtrees property directly
        self.subtrees = self.modern_collector.subtrees

    def visit(self, node):
        """Delegate to the modern implementation."""
        self.modern_collector.visit(node)
        # Update our reference to the modern collector's subtrees
        self.subtrees = self.modern_collector.subtrees


class SyntaxAwareCodeGenome(CodeGenome):
    """
    An extension of CodeGenome that is syntax-aware and ensures
    that all genetic operations produce syntactically valid code.
    """
    
    def __init__(self, ast_tree=None, source_code=None):
        """Initialize a new SyntaxAwareCodeGenome."""
        super().__init__(ast_tree, source_code)
        self._validate_and_repair_code()
    
    def _validate_and_repair_code(self):
        """Validate and repair the code if necessary."""
        if not self._source_code and self.ast_tree:
            try:
                self._source_code = ast.unparse(self.ast_tree)
            except Exception as e:
                logger.warning(f"Failed to unparse AST: {e}")
                return
        
        if self._source_code:
            valid_code, was_valid, repairs = SyntaxValidator.validate_and_repair(self._source_code)
            
            if not was_valid and repairs:
                logger.info(f"Repaired code with {len(repairs)} fixes: {', '.join(repairs)}")
                self._source_code = valid_code
                
                # Update AST with repaired code
                try:
                    self.ast_tree = ast.parse(self._source_code)
                except SyntaxError:
                    logger.warning("Failed to parse repaired code")


# Add _validate_and_repair_code method to the original CodeGenome class
def _validate_and_repair_code(self):
    """Validate and repair the code if necessary."""
    if not self._source_code and self.ast_tree:
        try:
            self._source_code = ast.unparse(self.ast_tree)
        except Exception as e:
            logger.warning(f"Failed to unparse AST: {e}")
            return
    
    if self._source_code:
        valid_code, was_valid, repairs = SyntaxValidator.validate_and_repair(self._source_code)
        
        if not was_valid and repairs:
            logger.info(f"Repaired code with {len(repairs)} fixes: {', '.join(repairs)}")
            self._source_code = valid_code
            
            # Update AST with repaired code
            try:
                self.ast_tree = ast.parse(self._source_code)
            except SyntaxError:
                logger.warning("Failed to parse repaired code")

# Add the method to the CodeGenome class
CodeGenome._validate_and_repair_code = _validate_and_repair_code
