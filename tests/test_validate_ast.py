import pytest
from trisolaris.core.program_representation import validate_ast
import ast

def test_validate_ast_valid():
    """Test validate_ast with valid ASTs."""
    # Create a simple valid AST
    valid_ast = ast.Module(
        body=[
            ast.FunctionDef(
                name='sort_files',
                args=ast.arguments(
                    posonlyargs=[],
                    args=[ast.arg(arg='file_list', annotation=None)],
                    kwonlyargs=[],
                    kw_defaults=[],
                    defaults=[]
                ),
                body=[ast.Return(value=ast.Constant(value=42))],
                decorator_list=[],
                returns=None
            )
        ],
        type_ignores=[]
    )
    valid_ast = ast.fix_missing_locations(valid_ast)
    is_valid, error = validate_ast(valid_ast)
    assert is_valid, f"Valid AST rejected: {error}"
    assert error is None

def test_validate_ast_invalid():
    """Test validate_ast with invalid ASTs."""
    # Create an invalid AST (missing required attributes)
    invalid_ast = ast.Module(
        body=[
            ast.FunctionDef(
                name='sort_files',
                args=None,  # Invalid: args should be an ast.arguments object
                body=[ast.Return(value=ast.Constant(value=42))],
                decorator_list=[],
                returns=None
            )
        ],
        type_ignores=[]
    )
    invalid_ast = ast.fix_missing_locations(invalid_ast)
    is_valid, error = validate_ast(invalid_ast)
    assert not is_valid, "Invalid AST accepted"
    assert error is not None

def test_validate_ast_roundtrip():
    """Test that ASTs can be compiled and parsed back."""
    # Create a valid AST
    original_ast = ast.Module(
        body=[
            ast.FunctionDef(
                name='sort_files',
                args=ast.arguments(
                    posonlyargs=[],
                    args=[ast.arg(arg='file_list', annotation=None)],
                    kwonlyargs=[],
                    kw_defaults=[],
                    defaults=[]
                ),
                body=[ast.Return(value=ast.Constant(value=42))],
                decorator_list=[],
                returns=None
            )
        ],
        type_ignores=[]
    )
    original_ast = ast.fix_missing_locations(original_ast)
    # Convert to source and back
    source = ast.unparse(original_ast)
    parsed_ast = ast.parse(source)
    
    # Validate the round-tripped AST
    is_valid, error = validate_ast(parsed_ast)
    assert is_valid, f"Round-tripped AST invalid: {error}"
    assert error is None

def test_validate_ast_edge_cases():
    """Test validate_ast with edge cases."""
    # Test with empty module (should be invalid if validator requires sort_files)
    empty_ast = ast.Module(body=[], type_ignores=[])
    empty_ast = ast.fix_missing_locations(empty_ast)
    is_valid, error = validate_ast(empty_ast)
    assert not is_valid, "Empty module accepted when it should be rejected"
    # Test with module containing only imports (should be invalid if validator requires sort_files)
    import_ast = ast.Module(
        body=[
            ast.Import(names=[ast.alias(name='os', asname=None)]),
            ast.ImportFrom(
                module='sys',
                names=[ast.alias(name='path', asname=None)],
                level=0
            )
        ],
        type_ignores=[]
    )
    import_ast = ast.fix_missing_locations(import_ast)
    is_valid, error = validate_ast(import_ast)
    assert not is_valid, "Module with only imports accepted when it should be rejected"
    # Test with a minimal valid module
    minimal_valid_ast = ast.Module(
        body=[
            ast.FunctionDef(
                name='sort_files',
                args=ast.arguments(
                    posonlyargs=[],
                    args=[ast.arg(arg='file_list', annotation=None)],
                    kwonlyargs=[],
                    kw_defaults=[],
                    defaults=[]
                ),
                body=[ast.Return(value=ast.Constant(value=0))],
                decorator_list=[],
                returns=None
            )
        ],
        type_ignores=[]
    )
    minimal_valid_ast = ast.fix_missing_locations(minimal_valid_ast)
    is_valid, error = validate_ast(minimal_valid_ast)
    assert is_valid, f"Minimal valid module rejected: {error}" 