"""
CodeGenome representation for the TRISOLARIS framework.

This module provides the CodeGenome class that represents individual solutions 
in the form of code snippets or programs.
"""

import random
import copy
import ast
from typing import Tuple, List, Dict, Any, Optional, Union
import os

class CodeGenome:
    """
    Represents a single solution in the form of code.
    
    CodeGenome provides methods for initialization, mutation, crossover, and conversion
    between source code and abstract syntax tree (AST) representations.
    """
    
    def __init__(self, ast_tree=None, source_code=None, task_description=None, task_type=None, template_code=None, **kwargs):
        """
        Initialize a new CodeGenome.
        
        Args:
            ast_tree: Optional AST representation of the code
            source_code: Optional source code string
            task_description: Optional natural language description of the task
            task_type: Optional identifier for the type of task (e.g., 'bluetooth_scan')
            template_code: Optional template code to use as a starting point
            **kwargs: Additional parameters for specialized genome creation
            
        Note: At least one of ast_tree or source_code should be provided,
              unless random initialization or template-based initialization is desired.
        """
        # Store task information
        self.task_description = task_description
        self.task_type = task_type
        self.code = None  # Public access to code
        
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
        elif template_code:
            # Use the provided template code as a starting point
            self._source_code = template_code
            try:
                self.ast_tree = ast.parse(template_code)
            except SyntaxError:
                # Fallback to storing just source code if parsing fails
                self.ast_tree = None
                
            # Add task-specific information as a comment if available
            if task_description:
                self._source_code = f"# Task: {task_description}\n{self._source_code}"
        else:
            # Create a task-specific or random genome
            self._create_task_specific_genome()
        
        # Store the source code as the accessible code property
        if self._source_code:
            self.code = self._source_code
            
        # Initialize fitness as negative infinity (worst possible)
        self.fitness = float('-inf')
    
    def _create_task_specific_genome(self):
        """Create a genome that is either task-specific or random."""
        if self.task_type == "bluetooth_scan":
            # Create a specialized Bluetooth scanning program
            self._create_bluetooth_scan_genome()
        elif self.task_type == "usb_scan":
            # Create a specialized USB scanning program
            self._create_usb_scan_genome()
        elif self.task_description and len(self.task_description) > 10:
            # Create a more tailored starter code based on description
            self._create_description_based_genome()
        else:
            # Fall back to classic random genome
            self._create_random_genome()
            
        try:
            self.ast_tree = ast.parse(self._source_code)
        except SyntaxError:
            # Fallback if somehow the generated code is invalid
            self._source_code = "def fallback():\n    # Task could not be parsed correctly\n    return 0"
            self.ast_tree = ast.parse(self._source_code)
            
        # Store the source code as the accessible code property
        self.code = self._source_code
    
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
        
        self._source_code = f"# Random starter code\n"
        if self.task_description:
            self._source_code += f"# Task: {self.task_description}\n"
        self._source_code += f"def {function_name}({params}):\n    return {return_value}"
    
    def _create_bluetooth_scan_genome(self):
        """Create specialized starter code for Bluetooth scanning."""
        self._source_code = """# Bluetooth scanner program
    import bluetooth
    
    def scan_for_devices(duration=10):
        # Scan for nearby Bluetooth devices
        print("Scanning for Bluetooth devices...")
        devices = bluetooth.discover_devices(
            duration=duration,
            lookup_names=True
        )
        
        results = []
        for addr, name in devices:
            results.append({
                "address": addr,
                "name": name
            })
        
        return results
    
    def main():
        devices = scan_for_devices()
        print(f"Found {len(devices)} devices:")
        for device in devices:
            print(f"  {device['name']} - {device['address']}")
        
    if __name__ == "__main__":
        main()
    """
        # Add task description as a comment if available
        if self.task_description:
            self._source_code = f"# Task: {self.task_description}\n{self._source_code}"
    
    def _create_usb_scan_genome(self):
        """Create specialized starter code for USB scanning."""
        self._source_code = """# USB device scanner program
    import usb.core
    import usb.util
    
    def list_usb_devices():
        # List all connected USB devices
        print("Scanning for USB devices...")
        devices = usb.core.find(find_all=True)
        
        results = []
        for device in devices:
            try:
                manufacturer = usb.util.get_string(device, device.iManufacturer)
            except:
                manufacturer = "Unknown"
                
            try:
                product = usb.util.get_string(device, device.iProduct)
            except:
                product = "Unknown"
                
            results.append({
                "vendor_id": device.idVendor,
                "product_id": device.idProduct,
                "manufacturer": manufacturer,
                "product": product
            })
        
        return results
    
    def main():
        devices = list_usb_devices()
        print(f"Found {len(devices)} USB devices:")
        for device in devices:
            print(f"  {device['product']} ({device['manufacturer']}) - "
                  f"ID: {device['vendor_id']:04x}:{device['product_id']:04x}")
        
    if __name__ == "__main__":
        main()
    """
        # Add task description as a comment if available
        if self.task_description:
            self._source_code = f"# Task: {self.task_description}\n{self._source_code}"
    
    def _create_description_based_genome(self):
        """Create starter code based on the task description."""
        words = self.task_description.lower().split()
        
        # Extract key terms from description
        key_terms = [w for w in words if len(w) > 3]
        if not key_terms:
            key_terms = ["task"]
        
        function_name = f"process_{key_terms[0].replace('.', '').replace(',', '')}"
        
        # Generate function parameters based on key terms (up to 3)
        param_names = []
        for term in key_terms[1:4]:
            clean_term = term.replace('.', '').replace(',', '')
            if clean_term not in function_name and clean_term.isalnum():
                param_names.append(clean_term)
        
        params = ", ".join(param_names) if param_names else "input_data"
        
        # Generate a descriptive comment
        comment = f"# Task: {self.task_description}\n"
        
        # Create function body
        body = "    # TODO: Implement based on task description\n"
        body += "    result = {}\n"
        for term in key_terms[:3]:
            clean_term = term.replace('.', '').replace(',', '')
            if clean_term not in function_name and clean_term not in param_names and clean_term.isalnum():
                body += f"    result['{clean_term}'] = True\n"
        body += "    return result"
        
        self._source_code = f"{comment}\ndef {function_name}({params}):\n{body}"
    
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
        Convert the AST back to source code.
        
        Returns:
            The source code string representation
        """
        if self._source_code:
            return self._source_code
        try:
            return ast.unparse(self.ast_tree)
        except Exception:
            return "def fallback():\n    return 0"
    
    def set_fitness(self, fitness: float) -> None:
        """
        Set the fitness score for this genome.
        
        Args:
            fitness: The fitness score to set
        """
        self.fitness = fitness
    
    def get_fitness(self) -> float:
        """
        Get the current fitness score.
        
        Returns:
            The current fitness score
        """
        return self.fitness 
    
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
            
            # Create and return new genomes
            return CodeGenome(ast_tree=child1_ast), CodeGenome(ast_tree=child2_ast)
        else:
            # Fallback to source-level crossover if AST is not available
            src1 = self.to_source().split('\n')
            src2 = other.to_source().split('\n')
            
            # Simple line-based crossover
            crossover_point = min(len(src1) // 2, len(src2) // 2)
            
            child1_src = '\n'.join(src1[:crossover_point] + src2[crossover_point:])
            child2_src = '\n'.join(src2[:crossover_point] + src1[crossover_point:])
            
            return CodeGenome(source_code=child1_src), CodeGenome(source_code=child2_src)


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
        
        # Apply mutations using a node transformer
        transformer = MutationTransformer(self.mutation_rate)
        return ast.fix_missing_locations(transformer.visit(tree_copy))


class MutationTransformer(ast.NodeTransformer):
    """AST transformer that applies random mutations."""
    
    def __init__(self, mutation_rate: float = 0.1):
        """
        Initialize the mutation transformer.
        
        Args:
            mutation_rate: Probability of mutation for each node
        """
        self.mutation_rate = mutation_rate
        
        # Define possible literal mutations
        self.num_mutators = [
            lambda n: n + 1,
            lambda n: n - 1,
            lambda n: n * 2,
            lambda n: n // 2 if n != 0 else 1
        ]
        
        self.str_mutators = [
            lambda s: s + "X",
            lambda s: s[:-1] if len(s) > 1 else s,
            lambda s: "X" + s,
            lambda s: s.replace("a", "A").replace("e", "E")
        ]
        
        self.bool_mutators = [
            lambda b: not b
        ]
    
    def generic_visit(self, node):
        """Visit all nodes and possibly apply mutations."""
        # Call the parent method to continue traversal
        node = super().generic_visit(node)
        
        # Randomly decide whether to mutate this node
        if random.random() < self.mutation_rate:
            node = self.mutate_node(node)
        
        return node
    
    def mutate_node(self, node):
        """Apply appropriate mutation based on node type."""
        # Handle different node types
        if isinstance(node, ast.Num):
            return self.mutate_num(node)
        elif isinstance(node, ast.Str):
            return self.mutate_str(node)
        elif isinstance(node, ast.NameConstant) and isinstance(node.value, bool):
            return self.mutate_bool(node)
        elif isinstance(node, ast.BinOp):
            return self.mutate_binop(node)
        elif isinstance(node, ast.Compare):
            return self.mutate_compare(node)
        elif isinstance(node, ast.Name):
            return self.mutate_name(node)
        
        # Return unchanged for other node types
        return node
    
    def mutate_num(self, node):
        """Mutate a numeric literal."""
        mutator = random.choice(self.num_mutators)
        try:
            new_value = mutator(node.n)
            return ast.Num(n=new_value)
        except:
            return node
    
    def mutate_str(self, node):
        """Mutate a string literal."""
        mutator = random.choice(self.str_mutators)
        try:
            new_value = mutator(node.s)
            return ast.Str(s=new_value)
        except:
            return node
    
    def mutate_bool(self, node):
        """Mutate a boolean literal."""
        return ast.NameConstant(value=not node.value)
    
    def mutate_binop(self, node):
        """Mutate a binary operation by changing the operator."""
        op_map = {
            ast.Add: ast.Sub,
            ast.Sub: ast.Add,
            ast.Mult: ast.Div,
            ast.Div: ast.Mult,
            ast.FloorDiv: ast.Div,
            ast.Mod: ast.FloorDiv
        }
        
        if type(node.op) in op_map:
            node.op = op_map[type(node.op)]()
        
        return node
    
    def mutate_compare(self, node):
        """Mutate a comparison by changing the operator."""
        if not node.ops:
            return node
            
        op_map = {
            ast.Eq: ast.NotEq,
            ast.NotEq: ast.Eq,
            ast.Lt: ast.Gt,
            ast.Gt: ast.Lt,
            ast.LtE: ast.GtE,
            ast.GtE: ast.LtE,
            ast.Is: ast.IsNot,
            ast.IsNot: ast.Is
        }
        
        if type(node.ops[0]) in op_map:
            node.ops[0] = op_map[type(node.ops[0])]()
        
        return node
    
    def mutate_name(self, node):
        """Potentially add a prefix or suffix to a variable name."""
        # Be careful about changing names - only do it for certain contexts
        if random.random() < 0.1:  # Very low probability
            # Only modify names that appear to be variables, not built-ins
            if not node.id.startswith('__') and node.id not in dir(__builtins__):
                node.id = f"var_{node.id}"
        
        return node


class AstCrossover:
    """Implements crossover operations for AST trees."""
    
    def crossover(self, tree1: ast.AST, tree2: ast.AST) -> Tuple[ast.AST, ast.AST]:
        """
        Perform crossover between two AST trees.
        
        Args:
            tree1: First parent AST
            tree2: Second parent AST
            
        Returns:
            Tuple of two child ASTs
        """
        # Create deep copies to avoid modifying the originals
        child1 = copy.deepcopy(tree1)
        child2 = copy.deepcopy(tree2)
        
        # Get all suitable subtrees for crossover
        subtrees1 = self._collect_subtrees(child1)
        subtrees2 = self._collect_subtrees(child2)
        
        if not subtrees1 or not subtrees2:
            # Not enough subtrees for crossover
            return child1, child2
        
        # Select random subtrees
        parent1_node, parent1_field_or_list, parent1_index = random.choice(subtrees1)
        parent2_node, parent2_field_or_list, parent2_index = random.choice(subtrees2)
        
        # Extract the subtrees to swap
        # If parent1_index is not None, it's a list element swap
        if parent1_index is not None:
            subtree1 = parent1_field_or_list[parent1_index]
        else: # It's an attribute swap
            subtree1 = getattr(parent1_node, parent1_field_or_list)
            
        if parent2_index is not None:
            subtree2 = parent2_field_or_list[parent2_index]
        else: # It's an attribute swap
            subtree2 = getattr(parent2_node, parent2_field_or_list)
        
        # Ensure the types are compatible for swap if possible (simple check)
        if type(subtree1) != type(subtree2):
             # If types don't match, might be risky to swap; return clones
             # A more sophisticated check could allow swapping compatible types (e.g., Expr vs Expr)
             # For now, we abort the swap for safety.
             return child1, child2

        # Perform the swap
        if parent1_index is not None:
            parent1_field_or_list[parent1_index] = subtree2
        else:
            setattr(parent1_node, parent1_field_or_list, subtree2)
            
        if parent2_index is not None:
            parent2_field_or_list[parent2_index] = subtree1
        else:
            setattr(parent2_node, parent2_field_or_list, subtree1)
        
        # Fix the AST structure
        ast.fix_missing_locations(child1)
        ast.fix_missing_locations(child2)
        
        return child1, child2
    
    def _collect_subtrees(self, tree: ast.AST) -> List[Tuple[ast.AST, Union[str, list], Optional[int]]]:
        """
        Collect all valid subtrees for crossover.
        Subtrees can be direct attributes or elements within list attributes.
        
        Args:
            tree: The AST to analyze
            
        Returns:
            List of tuples (parent_node, field_name_or_list, index_or_None)
            If index_or_None is None, field_name_or_list is the attribute name (str).
            If index_or_None is an int, field_name_or_list is the list itself,
            and index_or_None is the index within that list.
        """
        collector = SubtreeCollector()
        collector.visit(tree)
        return collector.subtrees


class SubtreeCollector(ast.NodeVisitor):
    """Collects subtrees from an AST that are suitable for crossover."""
    
    def __init__(self):
        # Stores tuples: (parent_node, field_name_or_list, index_or_None)
        self.subtrees: List[Tuple[ast.AST, Union[str, list], Optional[int]]] = []

    def visit(self, node):
        """Override visit to process attributes and list elements."""
        # First, visit children to collect deeper subtrees
        super().visit(node)

        # Then, process the current node's direct attributes and list elements
        for field, value in ast.iter_fields(node):
            if isinstance(value, ast.AST):
                # Add direct AST attributes as potential swap points
                self.subtrees.append((node, field, None)) 
            elif isinstance(value, list) and value and all(isinstance(x, ast.AST) for x in value):
                # Add individual elements of AST lists as potential swap points
                for i, item in enumerate(value):
                     # Ensure the item is actually an AST node before adding
                    if isinstance(item, ast.AST):
                         self.subtrees.append((node, value, i)) # Store the list itself and the index

    # Remove specific visit methods like visit_Module, visit_FunctionDef etc.
    # as the generic visit handles collecting subtrees more comprehensively.
    # The generic_visit method of NodeVisitor will traverse the tree correctly.

    # We keep generic_visit from the parent class (NodeVisitor)
    # The custom visit method above adds the logic for collecting our specific tuples. 