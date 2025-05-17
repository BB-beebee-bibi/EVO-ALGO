import ast
import random
import copy
from typing import List, Any, Optional, Tuple, Set, Dict
import astor  # For converting AST back to source code
from .ast_helpers import (
    get_all_nodes, get_leaf_nodes, get_subtrees,
    validate_ast, replace_subtree, clone_ast
)

class ProgramAST:
    """
    Represents a program as an AST for evolutionary operations.
    """
    def __init__(self, ast_tree: Optional[ast.AST] = None):
        self.ast_tree = ast_tree or self._generate_random_program()
        # Validate the AST
        is_valid, error = validate_ast(self.ast_tree)
        if not is_valid:
            raise ValueError(f"Invalid AST: {error}")
    
    def _generate_random_program(self) -> ast.AST:
        """Generate a random program AST for text file sorting."""
        # Basic program structure
        program = ast.Module(
            body=[
                # Import statements
                ast.Import(names=[ast.alias(name='os', asname=None)]),
                ast.Import(names=[ast.alias(name='re', asname=None)]),
                
                # Main function definition
                ast.FunctionDef(
                    name='sort_files',
                    args=ast.arguments(
                        posonlyargs=[],
                        args=[ast.arg(arg='file_list', annotation=None)],
                        kwonlyargs=[],
                        kw_defaults=[],
                        defaults=[]
                    ),
                    body=[
                        # Initialize result dictionary
                        ast.Assign(
                            targets=[ast.Name(id='result', ctx=ast.Store())],
                            value=ast.Dict(keys=[], values=[])
                        ),
                        
                        # Process each file
                        ast.For(
                            target=ast.Name(id='filename', ctx=ast.Store()),
                            iter=ast.Name(id='file_list', ctx=ast.Load()),
                            body=[
                                # Read file content
                                ast.Assign(
                                    targets=[ast.Name(id='content', ctx=ast.Store())],
                                    value=ast.Call(
                                        func=ast.Name(id='open', ctx=ast.Load()),
                                        args=[
                                            ast.Call(
                                                func=ast.Attribute(
                                                    value=ast.Name(id='os', ctx=ast.Load()),
                                                    attr='path',
                                                    ctx=ast.Load()
                                                ),
                                                args=[
                                                    ast.Name(id='filename', ctx=ast.Load())
                                                ],
                                                keywords=[]
                                            )
                                        ],
                                        keywords=[]
                                    )
                                ),
                                
                                # Add classification logic (placeholder)
                                ast.Assign(
                                    targets=[ast.Name(id='category', ctx=ast.Store())],
                                    value=ast.Constant(value='unknown')
                                ),
                                
                                # Store result
                                ast.Expr(
                                    value=ast.Call(
                                        func=ast.Attribute(
                                            value=ast.Name(id='result', ctx=ast.Load()),
                                            attr='update',
                                            ctx=ast.Load()
                                        ),
                                        args=[
                                            ast.Dict(
                                                keys=[ast.Constant(value='filename')],
                                                values=[ast.Name(id='category', ctx=ast.Load())]
                                            )
                                        ],
                                        keywords=[]
                                    )
                                )
                            ],
                            orelse=[]
                        ),
                        
                        # Return result
                        ast.Return(value=ast.Name(id='result', ctx=ast.Load()))
                    ],
                    decorator_list=[],
                    returns=None
                )
            ],
            type_ignores=[]
        )
        
        return program
    
    def to_source(self) -> str:
        """Convert AST back to source code."""
        return astor.to_source(self.ast_tree)
    
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
            return self._generate_random_constant()
        elif isinstance(node, ast.Name):
            return self._generate_random_name()
        return node

    def _subtree_mutate(self, node: ast.AST) -> ast.AST:
        """Replace a subtree with a newly generated one."""
        return self._generate_random_expression(max_depth=2)

    def _functional_mutate(self, node: ast.AST) -> ast.AST:
        """Change operators or function calls while preserving arity."""
        if isinstance(node, ast.BinOp):
            return ast.BinOp(
                left=node.left,
                op=self._generate_random_operator(),
                right=node.right
            )
        elif isinstance(node, ast.Compare):
            return ast.Compare(
                left=node.left,
                ops=[self._generate_random_operator()],
                comparators=node.comparators
            )
        elif isinstance(node, ast.Call):
            return ast.Call(
                func=self._generate_random_name(),
                args=node.args,
                keywords=node.keywords
            )
        return node

    def mutate(self, mutation_rate: float = 0.1) -> 'ProgramAST':
        """Apply random mutations to the program AST."""
        if random.random() > mutation_rate:
            return ProgramAST(ast_tree=ast.copy_location(
                ast.fix_missing_locations(clone_ast(self.ast_tree)),
                self.ast_tree
            ))

        # Get all nodes in the AST
        all_nodes = get_all_nodes(self.ast_tree)
        if not all_nodes:
            return self

        # Select a random node to mutate
        target_node = random.choice(all_nodes)

        # Choose mutation type
        mutation_type = random.choice(['point', 'subtree', 'functional'])
        
        # Apply mutation
        if mutation_type == 'point' and len(list(ast.iter_child_nodes(target_node))) == 0:
            mutated_node = self._point_mutate(target_node)
        elif mutation_type == 'subtree':
            mutated_node = self._subtree_mutate(target_node)
        else:  # functional
            mutated_node = self._functional_mutate(target_node)

        # Create new AST with mutation
        new_ast = replace_subtree(self.ast_tree, target_node, mutated_node)
        
        # Validate the mutated AST
        is_valid, error = validate_ast(new_ast)
        if not is_valid:
            # If mutation is invalid, try again with a different node
            return self.mutate(mutation_rate)
            
        return ProgramAST(ast_tree=new_ast)
    
    @staticmethod
    def crossover(parent1: 'ProgramAST', parent2: 'ProgramAST') -> Tuple['ProgramAST', 'ProgramAST']:
        """
        Perform subtree crossover between two program ASTs.
        
        This implementation:
        1. Selects compatible subtrees from both parents
        2. Ensures type safety by checking node types
        3. Maintains program validity through validation
        4. Handles different AST node types appropriately
        
        Args:
            parent1: First parent program AST
            parent2: Second parent program AST
            
        Returns:
            Tuple of two new ProgramAST instances resulting from crossover
        """
        def get_compatible_subtrees(ast_tree: ast.AST) -> List[Tuple[ast.AST, ast.AST]]:
            """Get all valid subtree pairs that can be swapped."""
            compatible_pairs = []
            nodes1 = get_all_nodes(ast_tree)
            
            for node1 in nodes1:
                # Skip module and function definition nodes to maintain program structure
                if isinstance(node1, (ast.Module, ast.FunctionDef)):
                    continue
                    
                # Get all nodes from parent2 that are of the same type
                nodes2 = [n for n in get_all_nodes(parent2.ast_tree) 
                         if isinstance(n, type(node1)) and not isinstance(n, (ast.Module, ast.FunctionDef))]
                
                for node2 in nodes2:
                    compatible_pairs.append((node1, node2))
            
            return compatible_pairs
        
        def perform_crossover(ast1: ast.AST, ast2: ast.AST) -> Tuple[ast.AST, ast.AST]:
            """Perform the actual subtree swap between two ASTs."""
            # Get compatible subtree pairs
            compatible_pairs = get_compatible_subtrees(ast1)
            if not compatible_pairs:
                return ast1, ast2
                
            # Select a random pair of compatible subtrees
            node1, node2 = random.choice(compatible_pairs)
            
            # Create copies of the ASTs
            new_ast1 = clone_ast(ast1)
            new_ast2 = clone_ast(ast2)
            
            # Find corresponding nodes in the copied ASTs
            for node in ast.walk(new_ast1):
                if isinstance(node, type(node1)) and ast.dump(node) == ast.dump(node1):
                    node1 = node
                    break
                    
            for node in ast.walk(new_ast2):
                if isinstance(node, type(node2)) and ast.dump(node) == ast.dump(node2):
                    node2 = node
                    break
            
            # Perform the swap
            new_ast1 = replace_subtree(new_ast1, node1, node2)
            new_ast2 = replace_subtree(new_ast2, node2, node1)
            
            return new_ast1, new_ast2
        
        # Perform crossover
        child1_ast, child2_ast = perform_crossover(parent1.ast_tree, parent2.ast_tree)
        
        # Validate the resulting ASTs
        is_valid1, error1 = validate_ast(child1_ast)
        is_valid2, error2 = validate_ast(child2_ast)
        
        # If either child is invalid, return copies of the parents
        if not (is_valid1 and is_valid2):
            return (
                ProgramAST(ast_tree=clone_ast(parent1.ast_tree)),
                ProgramAST(ast_tree=clone_ast(parent2.ast_tree))
            )
        
        return ProgramAST(ast_tree=child1_ast), ProgramAST(ast_tree=child2_ast)

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