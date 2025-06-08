# src/planner/refactor_grammar.py
from typing import List, Dict, Any, Union, Optional, Type # Added Type for REFACTOR_OPERATION_MAP type hint
from pydantic import BaseModel, Field, validator

class BaseRefactorOperation(BaseModel):
    name: str
    description: str
    required_parameters: List[str]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Check if attributes are defined as class variables, not as Pydantic fields
        if not isinstance(getattr(cls, 'name', None), str) or \
           not isinstance(getattr(cls, 'description', None), str) or \
           not isinstance(getattr(cls, 'required_parameters', None), list):
            raise TypeError(f"Subclasses of BaseRefactorOperation must define 'name' (str), 'description' (str), and 'required_parameters' (List[str]) as class attributes.")


    def validate_parameters(self, params: Dict[str, Any]) -> bool:
        """
        Validates if the provided parameters are sufficient and correct for this operation.
        Base implementation checks for presence of all required_parameters.
        Specific operations should override this for more detailed validation.
        """
        for req_param in self.required_parameters:
            if req_param not in params:
                print(f"Error: Missing required parameter '{req_param}' for operation '{self.name}'.")
                return False
        return True

    class Config:
        extra = "forbid"
        arbitrary_types_allowed = True


# --- Specific Refactor Operation Examples ---

class AddDecorator(BaseRefactorOperation):
    name: str = "add_decorator"
    description: str = "Adds a decorator to a specified function or method."
    required_parameters: List[str] = ["target_function_name", "decorator_name"]
    # Optional class attribute, not a Pydantic field for the instance data model
    decorator_import_statement: Optional[str] = None

    def validate_parameters(self, params: Dict[str, Any]) -> bool:
        if not super().validate_parameters(params):
            return False
        if not isinstance(params.get("target_function_name"), str):
            print(f"Error: 'target_function_name' must be a string for '{self.name}'.")
            return False
        if not isinstance(params.get("decorator_name"), str):
            print(f"Error: 'decorator_name' must be a string for '{self.name}'.")
            return False
        # Parameters passed in the 'params' dict can be validated here.
        # 'decorator_import_statement' is a class attribute here, not typically in 'params'
        # unless it's passed as an override. If it's meant to be part of the operation's data,
        # it should be a Pydantic field in the model.
        if "decorator_import_statement_param" in params and \
           params["decorator_import_statement_param"] is not None and \
           not isinstance(params["decorator_import_statement_param"], str):
            print(f"Error: 'decorator_import_statement_param' must be a string if provided for '{self.name}'.")
            return False
        return True

class ExtractMethod(BaseRefactorOperation):
    name: str = "extract_method"
    description: str = "Extracts a block of code into a new method within the same class/module."
    required_parameters: List[str] = ["source_function_name", "start_line", "end_line", "new_method_name"]

    def validate_parameters(self, params: Dict[str, Any]) -> bool:
        if not super().validate_parameters(params):
            return False
        if not isinstance(params.get("start_line"), int) or not isinstance(params.get("end_line"), int):
            print(f"Error: 'start_line' and 'end_line' must be integers for '{self.name}'.")
            return False
        if params["start_line"] >= params["end_line"]:
            print(f"Error: 'start_line' must be less than 'end_line' for '{self.name}'.")
            return False
        if not isinstance(params.get("new_method_name"), str) or not params.get("new_method_name").isidentifier():
            print(f"Error: 'new_method_name' must be a valid Python identifier string for '{self.name}'.")
            return False
        if not isinstance(params.get("source_function_name"), str):
            print(f"Error: 'source_function_name' must be a string for '{self.name}'.")
            return False
        return True

class UpdateDocstring(BaseRefactorOperation):
    name: str = "update_docstring"
    description: str = "Updates the docstring of a specified function, method, or class."
    required_parameters: List[str] = ["target_name", "new_docstring"]

    def validate_parameters(self, params: Dict[str, Any]) -> bool:
        if not super().validate_parameters(params):
            return False
        if not isinstance(params.get("new_docstring"), str):
            print(f"Error: 'new_docstring' must be a string for '{self.name}'.")
            return False
        if not isinstance(params.get("target_name"), str):
            print(f"Error: 'target_name' must be a string for '{self.name}'.")
            return False
        return True

class AddImport(BaseRefactorOperation):
    name: str = "add_import"
    description: str = "Adds an import statement to a file."
    required_parameters: List[str] = ["import_statement"]

    def validate_parameters(self, params: Dict[str, Any]) -> bool:
        if not super().validate_parameters(params):
            return False
        import_stmt = params.get("import_statement")
        if not isinstance(import_stmt, str) or not (import_stmt.startswith("import ") or import_stmt.startswith("from ")):
            print(f"Error: 'import_statement' ('{import_stmt}') must be a valid Python import string for '{self.name}'.")
            return False
        return True

# --- Refactor Operation Map ---

# We instantiate the classes here to store instances in the map,
# allowing access to their methods and class attributes directly.
REFACTOR_OPERATION_INSTANCES: Dict[str, BaseRefactorOperation] = {
    op.name: op() for op in [AddDecorator, ExtractMethod, UpdateDocstring, AddImport]
}

# To store types instead (useful for creating new instances with specific data):
REFACTOR_OPERATION_CLASSES: Dict[str, Type[BaseRefactorOperation]] = {
    AddDecorator.name: AddDecorator,
    ExtractMethod.name: ExtractMethod,
    UpdateDocstring.name: UpdateDocstring,
    AddImport.name: AddImport,
}


if __name__ == "__main__":
    import json # For model_json_schema

    print("Refactor Operation Grammar loaded.")
    print(f"Available operation instances: {list(REFACTOR_OPERATION_INSTANCES.keys())}")
    print(f"Available operation classes: {list(REFACTOR_OPERATION_CLASSES.keys())}")


    add_decorator_op = REFACTOR_OPERATION_INSTANCES[AddDecorator.name]
    print(f"\nOperation: {add_decorator_op.name} - {add_decorator_op.description}")
    print(f"  Class attribute decorator_import_statement: {AddDecorator.decorator_import_statement}") # Access class attribute

    valid_params_decorator = {
        "target_function_name": "my_func",
        "decorator_name": "@logged",
        "decorator_import_statement_param": "from logging import logged" # Example if passed as param
    }
    print(f"Params: {valid_params_decorator}, Valid? {add_decorator_op.validate_parameters(valid_params_decorator)}")

    invalid_params_decorator = {"target_function_name": "my_func"}
    print(f"Params: {invalid_params_decorator}, Valid? {add_decorator_op.validate_parameters(invalid_params_decorator)}")

    invalid_type_params_decorator = {"target_function_name": 123, "decorator_name": "@logged"}
    print(f"Params: {invalid_type_params_decorator}, Valid? {add_decorator_op.validate_parameters(invalid_type_params_decorator)}")


    extract_method_op = REFACTOR_OPERATION_INSTANCES[ExtractMethod.name]
    print(f"\nOperation: {extract_method_op.name} - {extract_method_op.description}")
    valid_params_extract = {
        "source_function_name": "big_func",
        "start_line": 10,
        "end_line": 20,
        "new_method_name": "extracted_part"
    }
    print(f"Params: {valid_params_extract}, Valid? {extract_method_op.validate_parameters(valid_params_extract)}")

    invalid_params_extract = {
        "source_function_name": "big_func",
        "start_line": 20,
        "end_line": 10,
        "new_method_name": "extracted_part"
    }
    print(f"Params: {invalid_params_extract}, Valid? {extract_method_op.validate_parameters(invalid_params_extract)}")

    add_import_op = REFACTOR_OPERATION_INSTANCES[AddImport.name]
    print(f"\nOperation: {add_import_op.name} - {add_import_op.description}")
    valid_import_params = {"import_statement": "from my_module import my_class"}
    print(f"Params: {valid_import_params}, Valid? {add_import_op.validate_parameters(valid_import_params)}")
    invalid_import_params = {"import_statement": "my_module.my_class"}
    print(f"Params: {invalid_import_params}, Valid? {add_import_op.validate_parameters(invalid_import_params)}")

    update_docstring_op = REFACTOR_OPERATION_INSTANCES[UpdateDocstring.name]
    print(f"\nOperation: {update_docstring_op.name} - {update_docstring_op.description}")
    valid_docstring_params = {"target_name": "my_function", "new_docstring": "This is a new docstring."}
    print(f"Params: {valid_docstring_params}, Valid? {update_docstring_op.validate_parameters(valid_docstring_params)}")
    invalid_docstring_params = {"target_name": "my_function", "new_docstring": 123}
    print(f"Params: {invalid_docstring_params}, Valid? {update_docstring_op.validate_parameters(invalid_docstring_params)}")

    # Example of accessing schema from class type
    print("\n--- Schema for AddDecorator (from class) ---")
    # Pydantic V2 uses model_json_schema(), V1 used schema_json()
    if hasattr(AddDecorator, 'model_json_schema'):
        print(json.dumps(AddDecorator.model_json_schema(), indent=2))
    elif hasattr(AddDecorator, 'schema_json'): # Fallback for Pydantic V1
        print(AddDecorator.schema_json(indent=2))
