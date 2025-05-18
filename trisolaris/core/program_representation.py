import ast
import random
import copy
from typing import List, Any, Optional, Tuple, Set, Dict, Union
import astor  # For converting AST back to source code
from .ast_helpers import (
    get_all_nodes, get_leaf_nodes, get_subtrees,
    validate_ast, replace_subtree, clone_ast
)
import logging
import hashlib
import os

logger = logging.getLogger(__name__)

class ProgramAST:
    """
    Represents Python programs as ASTs for evolutionary operations.
    """
    def __init__(self, source: str = None, tree: ast.Module = None):
        """Initialize from source code or an existing AST."""
        self.stats = {
            'mutation_attempts': 0,
            'mutation_effective': 0,
            'mutation_noop': 0,
            'crossover_attempts': 0,
            'crossover_novel': 0,
            'crossover_parent1': 0,
            'crossover_parent2': 0,
            'crossover_validation_failures': 0
        }
        
        if source is not None:
            self.tree = ast.parse(source)
            ast.fix_missing_locations(self.tree)
        elif tree is not None:
            self.tree = copy.deepcopy(tree)
            ast.fix_missing_locations(self.tree)
        else:
            # Create minimal valid program with sort_files function
            self.tree = ast.parse("""
def sort_files(file_list):
    return sorted(file_list)
""")
            
        # Initialize function mapping
        self.ast_nodes = self._map_functions()
        
        # Validate the AST
        is_valid, error = validate_ast(self.tree)
        if not is_valid:
            raise ValueError(f"Invalid AST: {error}")
    
    def _map_functions(self) -> Dict[str, ast.FunctionDef]:
        """Create a mapping of function names to their AST nodes."""
        functions = {}
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef):
                functions[node.name] = node
        return functions
    
    def to_source(self) -> str:
        """Convert the AST back to source code."""
        return astor.to_source(self.tree)
    
    def _generate_random_constant(self) -> ast.AST:
        """Generate a random constant value."""
        constant_types = [
            (ast.Constant, lambda: random.randint(0, 100)),  # int
            (ast.Constant, lambda: random.random()),  # float
            (ast.Constant, lambda: ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=5))),  # str
            (ast.Constant, lambda: random.choice([True, False])),  # bool
        ]
        const_type, generator = random.choice(constant_types)
        return const_type(value=generator())

    def _generate_random_name(self) -> ast.Name:
        """Generate a random variable name."""
        name = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=5))
        return ast.Name(id=name, ctx=ast.Load())

    def _generate_random_operator(self) -> ast.AST:
        """Generate a random operator."""
        operators = [
            ast.Add(), ast.Sub(), ast.Mult(), ast.Div(),
            ast.Mod(), ast.Pow(), ast.FloorDiv(),
            ast.Eq(), ast.NotEq(), ast.Lt(), ast.LtE(),
            ast.Gt(), ast.GtE(), ast.Is(), ast.IsNot(),
            ast.In(), ast.NotIn()
        ]
        return random.choice(operators)

    def _generate_random_expression(self, max_depth: int = 2) -> ast.AST:
        """Generate a random expression."""
        if max_depth <= 0:
            return random.choice([
                self._generate_random_constant(),
                self._generate_random_name()
            ])
        
        expression_types = [
            # Binary operation
            lambda: ast.BinOp(
                left=self._generate_random_expression(max_depth - 1),
                op=self._generate_random_operator(),
                right=self._generate_random_expression(max_depth - 1)
            ),
            # Compare operation
            lambda: ast.Compare(
                left=self._generate_random_expression(max_depth - 1),
                ops=[self._generate_random_operator()],
                comparators=[self._generate_random_expression(max_depth - 1)]
            ),
            # Call operation
            lambda: ast.Call(
                func=self._generate_random_name(),
                args=[self._generate_random_expression(max_depth - 1)],
                keywords=[]
            ),
            # Simple expression
            lambda: random.choice([
                self._generate_random_constant(),
                self._generate_random_name()
            ])
        ]
        return random.choice(expression_types)()

    def _point_mutate(self, node: ast.AST) -> ast.AST:
        """Perform point mutation on a leaf node."""
        if isinstance(node, ast.Constant):
            # Generate a different constant value
            while True:
                new_node = self._generate_random_constant()
                if new_node.value != node.value:
                    return new_node
        elif isinstance(node, ast.Name):
            # Generate a different variable name
            while True:
                new_node = self._generate_random_name()
                if new_node.id != node.id:
                    return new_node
        return node

    def _subtree_mutate(self, node: ast.AST) -> ast.AST:
        """Replace a subtree with a newly generated one."""
        # Generate a new subtree with different structure
        new_subtree = self._generate_random_expression(max_depth=3)
        # Ensure the new subtree is different
        if ast.dump(new_subtree) == ast.dump(node):
            # If identical, modify one of its children
            if isinstance(new_subtree, ast.BinOp):
                new_subtree.op = self._generate_random_operator()
            elif isinstance(new_subtree, ast.Compare):
                new_subtree.ops = [self._generate_random_operator()]
            elif isinstance(new_subtree, ast.Call):
                new_subtree.func = self._generate_random_name()
        return new_subtree

    def _functional_mutate(self, node: ast.AST) -> ast.AST:
        """Change operators or function calls while preserving arity."""
        if isinstance(node, ast.BinOp):
            # Try different operators until we find one that's different
            while True:
                new_op = self._generate_random_operator()
                if not isinstance(new_op, type(node.op)):
                    return ast.BinOp(
                        left=node.left,
                        op=new_op,
                        right=node.right
                    )
        elif isinstance(node, ast.Compare):
            # Try different comparison operators
            while True:
                new_op = self._generate_random_operator()
                if not isinstance(new_op, type(node.ops[0])):
                    return ast.Compare(
                        left=node.left,
                        ops=[new_op],
                        comparators=node.comparators
                    )
        elif isinstance(node, ast.Call):
            # Try different function names or attributes
            func = node.func
            if isinstance(func, ast.Name):
                while True:
                    new_func = self._generate_random_name()
                    if new_func.id != func.id:
                        return ast.Call(
                            func=new_func,
                            args=node.args,
                            keywords=node.keywords
                        )
            elif isinstance(func, ast.Attribute):
                # Mutate the attribute name or the value
                if random.random() < 0.5:
                    # Change the attribute name
                    new_attr = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=5))
                    return ast.Call(
                        func=ast.Attribute(value=func.value, attr=new_attr, ctx=func.ctx),
                        args=node.args,
                        keywords=node.keywords
                    )
                else:
                    # Change the value (e.g., os -> random name)
                    new_value = self._generate_random_name()
                    return ast.Call(
                        func=ast.Attribute(value=new_value, attr=func.attr, ctx=func.ctx),
                        args=node.args,
                        keywords=node.keywords
                    )
        return node

    def _attempt_macro_mutation(self) -> Optional[ast.AST]:
        """Attempt a more substantial mutation when normal mutations fail."""
        try:
            all_nodes = get_all_nodes(self.tree)
            if not all_nodes:
                return None
            # Try to mutate import statements
            for node in all_nodes:
                if isinstance(node, ast.Import):
                    # Change the imported module name
                    for alias in node.names:
                        alias.name = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=5))
                    return self.tree
            # Try to mutate function signature
            for node in all_nodes:
                if isinstance(node, ast.FunctionDef):
                    # Change the function name
                    node.name = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=5))
                    # Change argument names
                    for arg in node.args.args:
                        arg.arg = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=5))
                    return self.tree
            # As a last resort, generate a new random program
            return self._generate_random_program()
        except Exception as e:
            logger.error(f"Macro mutation failed: {e}")
            return None

    def mutate(self, mutation_rate: float = 0.1, max_retries: int = 10) -> 'ProgramAST':
        """Apply random mutations to the program AST. Prevent infinite recursion by limiting retries."""
        self.stats['mutation_attempts'] += 1
        
        if random.random() > mutation_rate:
            self.stats['mutation_noop'] += 1
            return ProgramAST(tree=ast.copy_location(
                ast.fix_missing_locations(clone_ast(self.tree)),
                self.tree
            ))

        all_nodes = get_all_nodes(self.tree)
        if not all_nodes:
            self.stats['mutation_noop'] += 1
            return self

        original_hash = hash(ast.dump(self.tree))
        mutated = None
        for attempt in range(max_retries):
            target_node = random.choice(all_nodes)
            mutation_type = random.choice(['point', 'subtree', 'functional'])
            if mutation_type == 'point' and len(list(ast.iter_child_nodes(target_node))) == 0:
                mutated_node = self._point_mutate(target_node)
            elif mutation_type == 'subtree':
                mutated_node = self._subtree_mutate(target_node)
            else:
                mutated_node = self._functional_mutate(target_node)
            new_ast = replace_subtree(self.tree, target_node, mutated_node)
            is_valid, error = validate_ast(new_ast)
            if is_valid:
                mutated = ProgramAST(tree=new_ast)
                if hash(ast.dump(mutated.tree)) != original_hash:
                    self.stats['mutation_effective'] += 1
                    return mutated
            # On final attempt, try a more substantial mutation
            if attempt == max_retries - 2:
                new_ast = self._attempt_macro_mutation()
                if new_ast is not None:
                    is_valid, error = validate_ast(new_ast)
                    if is_valid:
                        mutated = ProgramAST(tree=new_ast)
                        if hash(ast.dump(mutated.tree)) != original_hash:
                            self.stats['mutation_effective'] += 1
                            return mutated
        # If all retries fail, return a clone of the original
        self.stats['mutation_noop'] += 1
        return ProgramAST(tree=ast.copy_location(
            ast.fix_missing_locations(clone_ast(self.tree)),
            self.tree
        ))

    @classmethod
    def crossover(cls, parent1: 'ProgramAST', parent2: 'ProgramAST') -> Tuple['ProgramAST', 'ProgramAST']:
        """
        Perform subtree crossover between this program and another parent.
        
        Note: Not every crossover operation will produce offspring different 
        from both parents. This is normal genetic programming behavior, especially 
        with simple programs or limited compatible subtrees. The crossover success 
        rate typically ranges from 30-70% depending on program complexity and diversity.
        
        Returns:
            tuple: (child1, child2) - Two new ProgramAST instances
        """
        parent1.stats['crossover_attempts'] += 1
        parent2.stats['crossover_attempts'] += 1
        
        # Enable debug mode if environment variable is set
        debug_mode = os.environ.get('DEBUG_CROSSOVER', '').lower() == 'true'
        
        if debug_mode:
            logger.info("Crossover debug mode enabled")
            logger.info(f"Parent 1 AST:\n{ast.dump(parent1.tree, include_attributes=True)}")
            logger.info(f"Parent 2 AST:\n{ast.dump(parent2.tree, include_attributes=True)}")
        
        # Get compatible subtrees from both parents with location information
        p1_subtrees = cls.get_compatible_subtrees(parent1.tree)
        p2_subtrees = cls.get_compatible_subtrees(parent2.tree)
        
        # Log the number of compatible subtrees found
        logger.debug(f"Parent 1 compatible subtrees: {len(p1_subtrees)}")
        logger.debug(f"Parent 2 compatible subtrees: {len(p2_subtrees)}")
        
        # If no compatible subtrees, return copies of parents
        if not p1_subtrees or not p2_subtrees:
            logger.warning("No compatible subtrees found for crossover. Returning parent copies.")
            parent1.stats['crossover_parent1'] += 1
            parent2.stats['crossover_parent2'] += 1
            return cls(tree=copy.deepcopy(parent1.tree)), cls(tree=copy.deepcopy(parent2.tree))
        
        # Try up to 5 times to get valid offspring
        max_attempts = 5
        for attempt in range(max_attempts):
            # Select random subtrees from each parent
            p1_subtree_info = random.choice(p1_subtrees)
            p2_subtree_info = random.choice(p2_subtrees)
            
            p1_subtree, p1_parent, p1_field, p1_index = p1_subtree_info
            p2_subtree, p2_parent, p2_field, p2_index = p2_subtree_info
            
            # Log the selected subtrees
            logger.debug(f"Selected subtree from Parent 1: {ast.unparse(p1_subtree)}")
            logger.debug(f"Selected subtree from Parent 2: {ast.unparse(p2_subtree)}")
            
            # Create copies of parent ASTs
            child1_ast = copy.deepcopy(parent1.tree)
            child2_ast = copy.deepcopy(parent2.tree)
            
            # Replace subtrees using structural equality and location tracking
            child1_ast = cls.replace_subtree_with_location(child1_ast, p1_subtree, p2_subtree, p1_parent, p1_field, p1_index)
            child2_ast = cls.replace_subtree_with_location(child2_ast, p2_subtree, p1_subtree, p2_parent, p2_field, p2_index)
            
            # Log the resulting ASTs
            logger.debug(f"Child 1 AST after crossover: {ast.unparse(child1_ast)}")
            logger.debug(f"Child 2 AST after crossover: {ast.unparse(child2_ast)}")
            
            # Validate resulting ASTs
            is_valid1, error1 = validate_ast(child1_ast)
            is_valid2, error2 = validate_ast(child2_ast)
            
            if not is_valid1 or not is_valid2:
                logger.warning(f"Invalid offspring produced. Child 1 error: {error1}, Child 2 error: {error2}")
                parent1.stats['crossover_validation_failures'] += 1
                parent2.stats['crossover_validation_failures'] += 1
                continue
            
            child1 = cls(tree=child1_ast)
            child2 = cls(tree=child2_ast)
            
            # Check if offspring are novel
            c1_novel = ast.dump(child1.tree) != ast.dump(parent1.tree) and ast.dump(child1.tree) != ast.dump(parent2.tree)
            c2_novel = ast.dump(child2.tree) != ast.dump(parent1.tree) and ast.dump(child2.tree) != ast.dump(parent2.tree)
            
            if c1_novel:
                parent1.stats['crossover_novel'] += 1
            if c2_novel:
                parent2.stats['crossover_novel'] += 1
            
            # If we have at least one novel offspring, return them
            if c1_novel or c2_novel:
                if debug_mode:
                    logger.info("Successfully produced novel offspring")
                    logger.info(f"Child 1 AST:\n{ast.dump(child1.tree, include_attributes=True)}")
                    logger.info(f"Child 2 AST:\n{ast.dump(child2.tree, include_attributes=True)}")
                return child1, child2
            
            # If we're on the last attempt and still no novel offspring, return the best we have
            if attempt == max_attempts - 1:
                logger.warning("Failed to produce novel offspring after maximum attempts")
                return child1, child2
        
        # If all attempts failed, return copies of parents
        return cls(tree=copy.deepcopy(parent1.tree)), cls(tree=copy.deepcopy(parent2.tree))

    @staticmethod
    def replace_subtree_with_location(
        ast_tree: ast.AST,
        old_subtree: ast.AST,
        new_subtree: ast.AST,
        parent: ast.AST,
        field: str,
        index: Optional[int]
    ) -> ast.AST:
        """
        Replace a subtree in an AST with another subtree using structural equality and location tracking.
        """
        new_ast = ast.copy_location(
            ast.fix_missing_locations(clone_ast(ast_tree)),
            ast_tree
        )
        
        # Find the exact node to replace using location information
        for node in ast.walk(new_ast):
            if isinstance(node, type(parent)):
                for f, value in ast.iter_fields(node):
                    if f == field:
                        if isinstance(value, list):
                            for i, item in enumerate(value):
                                if isinstance(item, ast.AST) and ast.dump(item) == ast.dump(old_subtree):
                                    if index is None or i == index:
                                        value[i] = ast.copy_location(
                                            ast.fix_missing_locations(clone_ast(new_subtree)),
                                            item
                                        )
                        elif isinstance(value, ast.AST) and ast.dump(value) == ast.dump(old_subtree):
                            setattr(node, field, ast.copy_location(
                                ast.fix_missing_locations(clone_ast(new_subtree)),
                                value
                            ))
        
        return new_ast

    @staticmethod
    def get_compatible_subtrees(ast_tree: ast.AST) -> List[Tuple[ast.AST, ast.AST, str, Optional[int]]]:
        """
        Get all valid subtrees that can be swapped during crossover.
        Returns a list of tuples containing (node, parent, field_name, index).
        
        Node filtering rules:
        1. Skip module and function definition nodes to maintain program structure
        2. Skip context nodes (Load, Store, Del) as they're rarely meaningful targets
        3. Skip operator nodes when they're standalone, but allow them within expressions
        4. Allow expression nodes containing operators (BinOp, Compare, Call)
        5. Allow statement nodes that don't affect program structure (Expr, Assign)
        6. Skip nodes that would break program semantics (e.g., function parameters)
        
        Returns:
            List[Tuple[ast.AST, ast.AST, str, Optional[int]]]: List of compatible subtrees
        """
        # Nodes that should never be swapped
        skip_types = (
            ast.Module,  # Top-level structure
            ast.FunctionDef,  # Function definitions
            ast.arg,  # Function parameters
            ast.Load, ast.Store, ast.Del,  # Context nodes
            ast.operator,  # Operator tokens
            ast.cmpop,  # Comparison operators
            ast.boolop,  # Boolean operators
            ast.unaryop,  # Unary operators
        )
        
        # Nodes that can be swapped if they're part of an expression
        expression_nodes = (
            ast.BinOp,  # Binary operations
            ast.Compare,  # Comparisons
            ast.Call,  # Function calls
            ast.Constant,  # Constants
            ast.Name,  # Variable names
            ast.Attribute,  # Attribute access
            ast.Subscript,  # Subscripting
            ast.List, ast.Tuple, ast.Dict,  # Collection literals
            ast.IfExp,  # Conditional expressions
            ast.Lambda,  # Lambda expressions
        )
        
        # Nodes that can be swapped as standalone statements
        statement_nodes = (
            ast.Expr,  # Expression statements
            ast.Assign,  # Assignment statements
            ast.AugAssign,  # Augmented assignment
            ast.Return,  # Return statements
            ast.If,  # If statements
            ast.For,  # For loops
            ast.While,  # While loops
            ast.Try,  # Try/except blocks
        )
        
        compatible_subtrees = []
        
        for parent in ast.walk(ast_tree):
            for field, value in ast.iter_fields(parent):
                if isinstance(value, list):
                    for i, item in enumerate(value):
                        if isinstance(item, ast.AST):
                            # Skip nodes that should never be swapped
                            if isinstance(item, skip_types):
                                continue
                                
                            # Allow expression nodes and their children
                            if isinstance(item, expression_nodes):
                                compatible_subtrees.append((item, parent, field, i))
                            # Allow statement nodes
                            elif isinstance(item, statement_nodes):
                                compatible_subtrees.append((item, parent, field, i))
                            # Allow operators within expressions
                            elif isinstance(parent, expression_nodes):
                                compatible_subtrees.append((item, parent, field, i))
                                
                elif isinstance(value, ast.AST):
                    # Skip nodes that should never be swapped
                    if isinstance(value, skip_types):
                        continue
                        
                    # Allow expression nodes and their children
                    if isinstance(value, expression_nodes):
                        compatible_subtrees.append((value, parent, field, None))
                    # Allow statement nodes
                    elif isinstance(value, statement_nodes):
                        compatible_subtrees.append((value, parent, field, None))
                    # Allow operators within expressions
                    elif isinstance(parent, expression_nodes):
                        compatible_subtrees.append((value, parent, field, None))
                        
        return compatible_subtrees

    def print_genetic_stats(self):
        """Report on genetic operator effectiveness."""
        mut_success_rate = self.stats['mutation_effective'] / max(1, self.stats['mutation_attempts'])
        cross_novel_rate = self.stats['crossover_novel'] / max(1, self.stats['crossover_attempts'])
        
        logger.info(f"Mutation success rate: {mut_success_rate:.1%}")
        logger.info(f"Novel crossover rate: {cross_novel_rate:.1%}")
        logger.info(f"Mutation no-op rate: {self.stats['mutation_noop'] / max(1, self.stats['mutation_attempts']):.1%}")
        logger.info(f"Crossover validation failures: {self.stats['crossover_validation_failures']}")

    @property
    def ast_tree(self) -> ast.AST:
        """Alias for self.tree to match test expectations."""
        return self.tree

class ProgramGenerator:
    """
    Generates and manipulates program ASTs for the evolutionary process.
    """
    @staticmethod
    def generate_population(size: int) -> List[ProgramAST]:
        """Generate a population of random program ASTs."""
        return [ProgramAST() for _ in range(size)]
    
    @staticmethod
    def mutate_population(population: List[ProgramAST], mutation_rate: float) -> List[ProgramAST]:
        """Apply mutations to a population of programs."""
        return [program.mutate(mutation_rate) for program in population]
    
    @staticmethod
    def crossover_population(parents: List[ProgramAST]) -> List[ProgramAST]:
        """Perform crossover operations on a population of programs."""
        children = []
        for i in range(0, len(parents) - 1, 2):
            child1, child2 = ProgramAST.crossover(parents[i], parents[i + 1])
            children.extend([child1, child2])
        return children 