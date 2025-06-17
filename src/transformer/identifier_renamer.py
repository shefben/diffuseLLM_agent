import libcst as cst
from libcst.metadata import (
    PositionProvider,
    ScopeProvider,
    ParentNodeProvider,
)  # Will be needed for more advanced scope
import re
import sqlite3
from pathlib import Path
from typing import Dict, Optional, Tuple

# Assuming NAMING_CONVENTIONS_REGEX and is_identifier_matching_convention are importable
# Adjust path based on final project structure.
from src.profiler.naming_conventions import is_identifier_matching_convention


# --- Name Conversion Helpers ---
def to_snake_case(name: str) -> str:
    """Converts a PascalCase or camelCase string to snake_case."""
    if not name:
        return ""
    # Add underscore before uppercase letters, except if it's the first char or preceded by an underscore or another uppercase letter
    s1 = re.sub(r"([^_A-Z])([A-Z][a-z]+)", r"\1_\2", name)
    # Add underscore between multiple uppercase letters followed by lowercase (e.g. HTTPResponse -> HTTP_Response)
    s2 = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", s1)
    # Handle cases like "HTTPRequest" -> "http_request" (all caps sequence)
    s3 = re.sub(
        r"([a-z0-9])([A-Z])", r"\1_\2", s2
    )  # Handles MyVar -> my_var, and then lower.
    return s3.lower()


def to_pascal_case(name: str) -> str:
    """Converts a snake_case or camelCase string to PascalCase."""
    if not name:
        return ""
    if "_" in name:  # snake_case
        return "".join(word.capitalize() for word in name.split("_"))
    else:  # camelCase or already PascalCase
        return name[0].upper() + name[1:]


def to_camel_case(name: str) -> str:
    """Converts a snake_case or PascalCase string to camelCase."""
    if not name:
        return ""
    if "_" in name:  # snake_case
        parts = name.split("_")
        return parts[0].lower() + "".join(word.capitalize() for word in parts[1:])
    else:  # PascalCase or already camelCase
        return name[0].lower() + name[1:]


# --- LibCST Transformer ---
class IdentifierRenamingTransformer(cst.CSTTransformer):
    METADATA_DEPENDENCIES = (
        PositionProvider,
        ScopeProvider,
    )  # Enable scope metadata if needed

    def __init__(
        self, active_rules: Dict[str, Tuple[str, str]]
    ):  # convention_name, regex_pattern
        super().__init__()
        self.active_rules = active_rules
        # For collision avoidance (simplified): track renames in current processing unit (module)
        # This doesn't handle cross-file or complex shadowing.
        self.renamed_in_module: Dict[str, str] = {}
        self.scope_renames: Dict[cst.metadata.Scope, Dict[str, str]] = {}

    def _get_target_convention(self, identifier_type: str) -> Optional[Tuple[str, str]]:
        return self.active_rules.get(identifier_type)

    def _convert_name(self, current_name: str, target_convention_name: str) -> str:
        if target_convention_name == "SNAKE_CASE":
            return to_snake_case(current_name)
        elif target_convention_name == "PASCAL_CASE":
            return to_pascal_case(current_name)
        elif target_convention_name == "CAMEL_CASE":
            return to_camel_case(current_name)
        # UPPER_SNAKE_CASE for constants often involves more than just case change (e.g. from var name)
        # For now, focus on function/class/variable casing.
        # Test conventions would also need specific converters if we rename *to* them.
        return current_name  # No change if target convention unknown or not convertible easily

    def leave_FunctionName(
        self, original_node: cst.FunctionName, updated_node: cst.FunctionName
    ) -> cst.FunctionName:
        target_conv_info = self._get_target_convention("function")
        if not target_conv_info:
            return updated_node

        target_convention_name, target_regex = target_conv_info
        current_name = original_node.value

        # Use the placeholder or imported version of is_identifier_matching_convention
        if not is_identifier_matching_convention(
            current_name, target_convention_name
        ):  # MODIFIED
            # Check if it matches a dunder pattern, if so, usually leave it.
            if re.fullmatch(r"__([a-zA-Z0-9_]+)__", current_name):
                return updated_node

            new_name = self._convert_name(current_name, target_convention_name)
            if new_name != current_name:
                print(
                    f"Renaming function: '{current_name}' -> '{new_name}' (to {target_convention_name})"
                )
                # Basic collision check (very simplified)
                # if new_name in self.renamed_in_module.values() and self.renamed_in_module.get(current_name) != new_name:
                #     print(f"Warning: Potential collision for new name '{new_name}'. Skipping rename of '{current_name}'.")
                #     return updated_node
                self.renamed_in_module[current_name] = new_name
                return updated_node.with_changes(value=new_name)
        return updated_node

    def leave_ClassName(
        self, original_node: cst.ClassName, updated_node: cst.ClassName
    ) -> cst.ClassName:
        target_conv_info = self._get_target_convention("class")
        if not target_conv_info:
            return updated_node

        target_convention_name, target_regex = target_conv_info
        current_name = original_node.value

        if not is_identifier_matching_convention(
            current_name, target_convention_name
        ):  # MODIFIED
            new_name = self._convert_name(current_name, target_convention_name)
            if new_name != current_name:
                print(
                    f"Renaming class: '{current_name}' -> '{new_name}' (to {target_convention_name})"
                )
                self.renamed_in_module[current_name] = new_name
                return updated_node.with_changes(value=new_name)
        return updated_node

    def leave_Name(
        self, original_node: cst.Name, updated_node: cst.Name
    ) -> cst.BaseExpression:
        parent = self.get_metadata(ParentNodeProvider, original_node, None)
        scope = self.get_metadata(ScopeProvider, original_node, None)

        identifier_type = None
        if isinstance(parent, cst.Param) or isinstance(parent, cst.AssignTarget):
            identifier_type = "variable"

        # Determine if this Name needs renaming based on stored mapping
        if (
            scope
            and scope in self.scope_renames
            and original_node.value in self.scope_renames[scope]
        ):
            new_name = self.scope_renames[scope][original_node.value]
            return updated_node.with_changes(value=new_name)

        if identifier_type:
            target_conv = self._get_target_convention(identifier_type)
            if target_conv:
                target_conv_name, _ = target_conv
                current_name = original_node.value
                if not is_identifier_matching_convention(
                    current_name, target_conv_name
                ):
                    new_name = self._convert_name(current_name, target_conv_name)
                    if new_name != current_name:
                        if scope:
                            self.scope_renames.setdefault(scope, {})[current_name] = (
                                new_name
                            )
                        self.renamed_in_module[current_name] = new_name
                        return updated_node.with_changes(value=new_name)
        return updated_node


# --- Main Orchestration Function ---
def rename_identifiers_in_code(source_code: str, db_path: Path) -> str:
    """
    Loads active naming rules from the DB and uses LibCST to rename identifiers
    in the source_code that do not conform to these active rules.
    """
    active_rules: Dict[str, Tuple[str, str]] = {}  # convention_name, regex_pattern
    conn = None
    try:
        if not db_path.exists():
            print(
                f"Warning: Naming conventions DB not found at {db_path}. No renaming will occur."
            )
            return source_code

        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute(
            "SELECT identifier_type, convention_name, regex_pattern FROM naming_rules WHERE is_active = TRUE"
        )
        for row in cursor.fetchall():
            id_type, conv_name, regex_p = row
            active_rules[id_type] = (conv_name, regex_p)
    except sqlite3.Error as e:
        print(f"SQLite error when loading naming rules: {e}. No renaming will occur.")
        if conn:
            conn.close()  # Ensure connection is closed on error too
        return source_code
    finally:
        if conn:
            conn.close()

    if not active_rules:
        print("No active naming rules found in the database. No renaming will occur.")
        return source_code

    try:
        module_cst = cst.parse_module(source_code)
        # For more advanced scope-aware renaming, a MetadataWrapper would be needed here:
        # wrapper = cst.metadata.Wrapper(module_cst)
        # renamer_transformer = IdentifierRenamingTransformer(active_rules)
        # modified_module = wrapper.visit(renamer_transformer)

        # Simpler visit for now, without full scope metadata for leave_Name
        renamer_transformer = IdentifierRenamingTransformer(active_rules)
        modified_module = module_cst.visit(renamer_transformer)

        return modified_module.code
    except cst.ParserSyntaxError as e:
        print(f"LibCST parsing error: {e}. Returning original code.")
        return source_code
    except Exception as e_global:
        print(f"Unexpected error during renaming: {e_global}. Returning original code.")
        return source_code


if __name__ == "__main__":
    # Example usage (requires a dummy DB to be set up)
    # from ..database_setup import create_naming_conventions_db # Relative import
    # from ..profiler.naming_conventions import populate_naming_rules_from_profile, NAMING_CONVENTIONS_REGEX

    print("--- Name Conversion Helpers ---")
    print(
        f"to_snake_case('MyPascalCase'): {to_snake_case('MyPascalCase')}"
    )  # my_pascal_case
    print(
        f"to_snake_case('myCamelCase'): {to_snake_case('myCamelCase')}"
    )  # my_camel_case
    print(
        f"to_pascal_case('my_snake_case'): {to_pascal_case('my_snake_case')}"
    )  # MySnakeCase
    print(
        f"to_pascal_case('myCamelCase'): {to_pascal_case('myCamelCase')}"
    )  # MyCamelCase
    print(
        f"to_camel_case('MyPascalCase'): {to_camel_case('MyPascalCase')}"
    )  # myPascalCase
    print(
        f"to_camel_case('my_snake_case'): {to_camel_case('my_snake_case')}"
    )  # mySnakeCase

    # --- LibCST Renaming Example (Conceptual - needs DB setup) ---
    # test_db = Path("temp_renamer_test.db")
    # if test_db.exists(): test_db.unlink()
    # create_naming_conventions_db(db_path=test_db)
    # mock_profile = {"identifier_snake_case_pct": 0.8, "identifier_camelCase_pct": 0.1} # To make snake_case dominant for func/var
    # populate_naming_rules_from_profile(test_db, mock_profile) # Populates with SNAKE_CASE for func/var, PASCAL_CASE for class

    # example_code = """
    # class my_class_to_rename: # Should become MyClassToRename
    #     def MyFunctionToRename(self): # Should become my_function_to_rename
    #         VarToRename = 1 # Placeholder for future variable renaming
    #         return VarToRename
    #
    # def AnotherFunction(): # Should become another_function
    #     pass
    # """
    # print(f"\nOriginal code:\n{example_code}")
    # # modified_code = rename_identifiers_in_code(example_code, test_db)
    # # print(f"\nModified code:\n{modified_code}")
    # print("\n(LibCST renaming test commented out as it requires live DB setup in __main__)")

    # if test_db.exists(): test_db.unlink()
    pass  # End of main
