import ast
from typing import Dict, Set

def extract_identifiers_from_source(source_code: str) -> Dict[str, Set[str]]:
    """
    Parses Python source code and extracts identifiers for classes, functions,
    parameters, and variables.

    Args:
        source_code: A string containing the Python source code.

    Returns:
        A dictionary where keys are identifier types (e.g., 'classes',
        'functions', 'parameters', 'variables') and values are sets of
        identified names.
        Returns a dictionary with an 'error' key if parsing fails.
    """
    identifiers: Dict[str, Set[str]] = {
        "classes": set(),
        "functions": set(),
        "parameters": set(),
        "variables": set(), # Includes local variables and attributes
    }

    try:
        tree = ast.parse(source_code)
    except SyntaxError as e:
        return {"error": f"SyntaxError: {e}"}

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            identifiers["classes"].add(node.name)
            # Attributes defined in class body (e.g., class variables)
            for sub_node in node.body:
                if isinstance(sub_node, ast.Assign):
                    for target in sub_node.targets:
                        if isinstance(target, ast.Name):
                            identifiers["variables"].add(target.id)
                        elif isinstance(target, ast.Attribute):
                            # Could record target.attr for class attributes if needed
                            pass
                elif isinstance(sub_node, ast.AnnAssign):
                     if isinstance(sub_node.target, ast.Name):
                        identifiers["variables"].add(sub_node.target.id)


        elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
            identifiers["functions"].add(node.name)
            # Parameters
            for arg in node.args.args:
                identifiers["parameters"].add(arg.arg)
            if node.args.vararg:
                identifiers["parameters"].add(node.args.vararg.arg)
            if node.args.kwarg:
                identifiers["parameters"].add(node.args.kwarg.arg)
            for arg in node.args.kwonlyargs:
                identifiers["parameters"].add(arg.arg)

            # Variables defined within the function
            for sub_node in ast.walk(node): # Walk only within the function scope
                if isinstance(sub_node, ast.Assign):
                    for target in sub_node.targets:
                        if isinstance(target, ast.Name):
                            identifiers["variables"].add(target.id)
                elif isinstance(sub_node, ast.AnnAssign):
                     if isinstance(sub_node.target, ast.Name):
                        identifiers["variables"].add(sub_node.target.id)
                elif isinstance(sub_node, ast.Attribute) and isinstance(sub_node.value, ast.Name) and sub_node.value.id == 'self':
                    # Could potentially capture self.attribute_name as a specific type of variable
                    identifiers["variables"].add(sub_node.attr)


        elif isinstance(node, ast.Assign):
            # Global/module-level variables (or reassigned variables if not in func/class scope)
            # This might catch some things already caught in class/function bodies if not careful about scope.
            # The current ast.walk processes nodes, so this will find all assignments.
            # We could add scope checks here if needed (e.g. by checking parent nodes).
            for target in node.targets:
                if isinstance(target, ast.Name):
                    # Check if it's not already a function or class name at module level
                    # This simple check might not be robust for complex scoping.
                    is_function_or_class = False
                    for body_item in tree.body:
                        if isinstance(body_item, (ast.FunctionDef, ast.ClassDef)) and body_item.name == target.id:
                            is_function_or_class = True
                            break
                    if not is_function_or_class:
                         identifiers["variables"].add(target.id)

        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                is_function_or_class = False
                for body_item in tree.body:
                    if isinstance(body_item, (ast.FunctionDef, ast.ClassDef)) and body_item.name == node.target.id:
                        is_function_or_class = True
                        break
                if not is_function_or_class:
                    identifiers["variables"].add(node.target.id)


    return identifiers

def extract_identifiers_from_file(file_path: str) -> Dict[str, Set[str]]:
    """
    Reads a Python file and extracts identifiers from its content.

    Args:
        file_path: The path to the Python file.

    Returns:
        A dictionary of identifiers, similar to extract_identifiers_from_source.
        Returns a dictionary with an 'error' key if the file cannot be read
        or parsed.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            source_code = f.read()
        return extract_identifiers_from_source(source_code)
    except FileNotFoundError:
        return {"error": f"File not found: {file_path}"}
    except Exception as e:
        return {"error": f"Failed to read file {file_path}: {e}"}

if __name__ == '__main__':
    example_code = """
MY_CONSTANT = 100
_global_var = "test"

class MyClass:
    cls_attr: int = 10

    def __init__(self, param1, param2="default"):
        self.instance_var = param1
        local_to_init = param2

    def method_one(self, arg1, *args, **kwargs):
        local_var = arg1 + MY_CONSTANT
        self.another_attr = local_var
        return self.another_attr

def top_level_function(func_param, _ignored_param):
    function_local = func_param * 2
    return function_local

async def async_function(async_arg):
    await_var = async_arg
    return await_var

another_var_after_defs = "done"
"""

    print("--- Identifiers from source ---")
    ids_src = extract_identifiers_from_source(example_code)
    if "error" in ids_src:
        print(ids_src["error"])
    else:
        for id_type, id_set in ids_src.items():
            print(f"{id_type.capitalize()}: {sorted(list(id_set))}")

    dummy_file = "dummy_identifier_test.py"
    with open(dummy_file, "w", encoding="utf-8") as f:
        f.write(example_code)

    print("\n--- Identifiers from file ---")
    ids_file = extract_identifiers_from_file(dummy_file)
    if "error" in ids_file:
        print(ids_file["error"])
    else:
        for id_type, id_set in ids_file.items():
            print(f"{id_type.capitalize()}: {sorted(list(id_set))}")

    # Test with syntactically incorrect file
    error_code = "class BadSyntax( :"
    print("\n--- Identifiers from code with syntax error ---")
    ids_error = extract_identifiers_from_source(error_code)
    print(ids_error)

    import os
    os.remove(dummy_file)
