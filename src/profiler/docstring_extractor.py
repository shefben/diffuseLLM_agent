import ast
from typing import Dict, List, Optional, Union

def extract_docstrings_from_source(source_code: str) -> Dict[str, Optional[str]]:
    """
    Parses Python source code and extracts docstrings from modules,
    classes, and functions.

    Args:
        source_code: A string containing the Python source code.

    Returns:
        A dictionary where keys are the names of the entities (e.g.,
        'module', 'ClassName', 'function_name', 'ClassName.method_name')
        and values are their docstrings. If an entity has no docstring,
        the value is None.
    """
    docstrings = {}

    try:
        tree = ast.parse(source_code)
    except SyntaxError as e:
        # Handle cases where the source code cannot be parsed
        return {"error": f"SyntaxError: {e}"}

    # Module docstring
    module_docstring = ast.get_docstring(tree)
    docstrings["module"] = module_docstring

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # For top-level functions or methods within classes
            path_parts = []
            current = node
            while current:
                if isinstance(current, ast.ClassDef):
                    path_parts.insert(0, current.name)
                elif isinstance(current, ast.FunctionDef):
                    path_parts.insert(0, current.name)

                # Try to find parent, this is a bit simplified
                # For robust parent tracking, one might need to build a parent map first
                parent = None
                for parent_node in ast.walk(tree):
                    for child in ast.iter_child_nodes(parent_node):
                        if child == current:
                            parent = parent_node
                            break
                    if parent:
                        break

                if isinstance(parent, (ast.ClassDef, ast.FunctionDef)):
                    current = parent
                else: # Reached module level or unsupported nesting
                    current = None

            name = ".".join(path_parts)
            if not name: # Should not happen if node is FunctionDef
                 name = node.name # Fallback to just node name

            docstrings[name] = ast.get_docstring(node)
        elif isinstance(node, ast.ClassDef):
            docstrings[node.name] = ast.get_docstring(node)
            # Docstrings for methods within this class are handled by the ast.FunctionDef check
            # by constructing the 'ClassName.method_name' path.

    return docstrings

def extract_docstrings_from_file(file_path: str) -> Dict[str, Optional[str]]:
    """
    Reads a Python file and extracts docstrings from its content.

    Args:
        file_path: The path to the Python file.

    Returns:
        A dictionary of docstrings, similar to extract_docstrings_from_source.
        Returns a dictionary with an 'error' key if the file cannot be read
        or parsed.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            source_code = f.read()
        return extract_docstrings_from_source(source_code)
    except FileNotFoundError:
        return {"error": f"File not found: {file_path}"}
    except Exception as e:
        return {"error": f"Failed to read file {file_path}: {e}"}

if __name__ == '__main__':
    example_code = """
"""Module docstring."""

class MyClass:
    """Class docstring."""

    def __init__(self, value):
        """Constructor docstring."""
        self.value = value

    def my_method(self, x):
        """Method docstring."""
        return x * self.value

def my_function(y):
    """Function docstring."""
    return y + 1

class NestedClass:
    """Nested Class docstring."""
    class InnerClass:
        """Inner Class docstring."""
        def inner_method(self):
            """Inner method docstring."""
            pass
"""

    # Test with source code
    print("--- Docstrings from source ---")
    docstrings_src = extract_docstrings_from_source(example_code)
    for name, doc in docstrings_src.items():
        print(f"{name}: {repr(doc)}")

    # Test with a dummy file
    dummy_file = "dummy_docstring_test.py"
    with open(dummy_file, "w", encoding="utf-8") as f:
        f.write(example_code)

    print("\n--- Docstrings from file ---")
    docstrings_file = extract_docstrings_from_file(dummy_file)
    for name, doc in docstrings_file.items():
        print(f"{name}: {repr(doc)}")

    # Test with a non-existent file
    print("\n--- Docstrings from non-existent file ---")
    docstrings_non_existent = extract_docstrings_from_file("non_existent.py")
    print(docstrings_non_existent)

    # Test with syntactically incorrect file
    error_code = "def func( :"
    print("\n--- Docstrings from code with syntax error ---")
    docstrings_error = extract_docstrings_from_source(error_code)
    print(docstrings_error)

    import os
    os.remove(dummy_file)
