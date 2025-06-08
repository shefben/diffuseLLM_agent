# src/profiler/naming_conventions.py
import re
from typing import Dict

# Define standard naming conventions and their regex patterns
# These regexes aim to be reasonably practical but might not cover every nuanced edge case.
# They generally assume ASCII identifiers for simplicity, but can be expanded for Unicode.

NAMING_CONVENTIONS_REGEX: Dict[str, str] = {
    "SNAKE_CASE": r"^[a-z_][a-z0-9_]*$",
    # Explanation: Starts with a lowercase letter or underscore,
    # followed by zero or more lowercase letters, digits, or underscores.
    # Examples: my_variable, _internal_var, calculate_value, value_01

    "PASCAL_CASE": r"^[A-Z][a-zA-Z0-9]*$",
    # Explanation: Starts with an uppercase letter,
    # followed by zero or more letters (uppercase or lowercase) or digits.
    # Examples: MyClass, AnotherExample, HttpRequestHandler

    "CAMEL_CASE": r"^[a-z]+[A-Za-z0-9_]*$", # Allow underscore in middle/end for flexibility
    # Explanation: Starts with one or more lowercase letters,
    # followed by zero or more letters (uppercase or lowercase), digits, or underscores.
    # An uppercase letter usually appears after the initial lowercase part if it's truly camelCase.
    # Examples: myVariable, anotherExample, calculateValue, value01, an_Object_Like_This (less common but matches)

    "UPPER_SNAKE_CASE": r"^[A-Z_][A-Z0-9_]*$",
    # Explanation: Starts with an uppercase letter or underscore,
    # followed by zero or more uppercase letters, digits, or underscores.
    # Examples: MY_CONSTANT, ANOTHER_VALUE, _INTERNAL_CONSTANT, API_KEY_01

    # Specific conventions for tests
    "TEST_SNAKE_CASE": r"^test_[a-z_][a-z0-9_]*$",
    # Explanation: Must start with "test_", followed by snake_case.
    # Examples: test_my_function, test_another_feature

    "TEST_PASCAL_CASE": r"^Test[A-Z][a-zA-Z0-9]*$",
    # Explanation: Must start with "Test", followed by PascalCase.
    # Examples: TestMyClass, TestAnotherComponent

    # Optional: Dunder methods (usually exempt from casing rules or have their own)
    # "DUNDER_METHOD": r"^__[a-z0-9_]+__$",
    # For now, these are not actively enforced as "styles" but rather special names.
}

# Helper function to validate an identifier against a named convention
def is_identifier_matching_convention(identifier: str, convention_name: str) -> bool:
    """
    Checks if a given identifier matches a specified naming convention.

    Args:
        identifier: The identifier string to check.
        convention_name: The name of the convention (e.g., "SNAKE_CASE").

    Returns:
        True if the identifier matches the convention, False otherwise.
        Returns False if the convention_name is unknown.
    """
    if convention_name not in NAMING_CONVENTIONS_REGEX:
        return False
    regex_pattern = NAMING_CONVENTIONS_REGEX[convention_name]
    return bool(re.fullmatch(regex_pattern, identifier))

if __name__ == '__main__':
    # Example usage and testing of the regexes
    test_cases = {
        "SNAKE_CASE": ["my_var", "_private", "var1", "a_b_c", "a", "_"],
        "PASCAL_CASE": ["MyClass", "MyVar", "Class01", "A"],
        "CAMEL_CASE": ["myVar", "anotherExample", "var01", "aValue", "a_b"], # a_b matches current CAMEL_CASE
        "UPPER_SNAKE_CASE": ["MY_CONST", "_PRIVATE_CONST", "CONST01", "A", "_"],
        "TEST_SNAKE_CASE": ["test_my_func", "test_another"],
        "TEST_PASCAL_CASE": ["TestMyClass", "TestAPI"],
    }

    invalid_cases = {
        "SNAKE_CASE": ["MyVar", "myVAr", "1var", "my var", ""],
        "PASCAL_CASE": ["myClass", "_MyClass", "1Class", "My Class", ""],
        "CAMEL_CASE": ["MyVar", "_myVar", "1var", "my Var", ""], # MyVar fails CAMEL_CASE (starts uppercase)
        "UPPER_SNAKE_CASE": ["myConst", "MY_CONST_lower", "1CONST", "MY CONST", ""],
        "TEST_SNAKE_CASE": ["my_test_func", "testMyFunc", "Test_Func", "test_"],
        "TEST_PASCAL_CASE": ["myTestClass", "TestmyClass", "test_Class", "Test"],
    }

    for convention, identifiers in test_cases.items():
        print(f"\n--- Testing {convention} (Valid Cases) ---")
        for ident in identifiers:
            matches = is_identifier_matching_convention(ident, convention)
            print(f"'{ident}': {'Matches' if matches else 'Does NOT match'}")
            if not matches:
                 print(f"ERROR: Expected '{ident}' to match {convention} but it did not.")


    for convention, identifiers in invalid_cases.items():
        print(f"\n--- Testing {convention} (Invalid Cases) ---")
        for ident in identifiers:
            matches = is_identifier_matching_convention(ident, convention)
            print(f"'{ident}': {'Matches' if matches else 'Does NOT match'}")
            if matches:
                 print(f"ERROR: Expected '{ident}' NOT to match {convention} but it did.")

# Add new imports
import sqlite3
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List # Added List

# Define identifier types used in the database
IDENTIFIER_TYPES = [
    "constant",
    "class",
    "function",
    "variable",
    "test_function",
    "test_class"
]

def populate_naming_rules_from_profile(
    db_path: Path,
    unified_profile: Dict[str, Any],
    default_active_threshold: float = 0.40 # Min percentage to be considered 'dominant' if no single one is >50%
) -> bool:
    """
    Populates the naming_rules table in the SQLite database based on the
    dominant conventions found in the unified_profile.

    Args:
        db_path: Path to the SQLite database (e.g., 'config/naming_conventions.db').
                 The database schema must already exist.
        unified_profile: A dictionary representing the unified style profile,
                         containing identifier percentage statistics.
        default_active_threshold: Minimum percentage for a style to be considered dominant
                                  if no style has a clear majority.

    Returns:
        True if population was successful, False otherwise.
    """
    if not db_path.exists():
        print(f"Error: Database file not found at {db_path}. Please create it first.")
        return False

    # Extract relevant percentages from the profile, defaulting to 0.0 if missing
    snake_pct = unified_profile.get("identifier_snake_case_pct", 0.0) or 0.0
    camel_pct = unified_profile.get("identifier_camelCase_pct", 0.0) or 0.0
    upper_snake_pct = unified_profile.get("identifier_UPPER_SNAKE_CASE_pct", 0.0) or 0.0
    # PascalCase for classes is assumed, not directly from these percentages for now.

    rules_to_insert: List[Tuple[str, str, str, str, bool]] = [] # identifier_type, convention_name, regex, description, is_active

    # --- Determine rules for each identifier type ---

    # 1. Constants
    # Primarily driven by UPPER_SNAKE_CASE_pct.
    # If very low, perhaps no strong constant convention, or it's mixed.
    # For now, if upper_snake_pct is present and > a small threshold, assume it.
    if upper_snake_pct > 0.01: # Small threshold to indicate presence
        rules_to_insert.append(("constant", "UPPER_SNAKE_CASE", NAMING_CONVENTIONS_REGEX["UPPER_SNAKE_CASE"], "Global constant naming convention", True))
    else: # Default or if no strong signal
        rules_to_insert.append(("constant", "UPPER_SNAKE_CASE", NAMING_CONVENTIONS_REGEX["UPPER_SNAKE_CASE"], "Default global constant naming convention", True))


    # 2. Classes
    # Assume PascalCase as a strong default for classes.
    rules_to_insert.append(("class", "PASCAL_CASE", NAMING_CONVENTIONS_REGEX["PASCAL_CASE"], "Global class naming convention", True))

    # 3. Functions and Variables (often share a style or have related styles)
    # Determine dominant between snake_case and camelCase
    func_var_convention = None
    if snake_pct >= camel_pct and snake_pct >= default_active_threshold:
        func_var_convention = "SNAKE_CASE"
    elif camel_pct > snake_pct and camel_pct >= default_active_threshold:
        func_var_convention = "CAMEL_CASE"
    elif snake_pct > 0 or camel_pct > 0 : # If neither reaches threshold but there's some usage, pick the max
        func_var_convention = "SNAKE_CASE" if snake_pct >= camel_pct else "CAMEL_CASE"

    if func_var_convention:
        rules_to_insert.append(("function", func_var_convention, NAMING_CONVENTIONS_REGEX[func_var_convention], f"Dominant global function naming: {func_var_convention}", True))
        rules_to_insert.append(("variable", func_var_convention, NAMING_CONVENTIONS_REGEX[func_var_convention], f"Dominant global variable naming: {func_var_convention}", True))

        # Add the other convention as inactive if it also had some presence
        if func_var_convention == "SNAKE_CASE" and camel_pct > 0.05: # Threshold for "some presence"
             rules_to_insert.append(("function", "CAMEL_CASE", NAMING_CONVENTIONS_REGEX["CAMEL_CASE"], "Secondary function naming: CAMEL_CASE", False))
             rules_to_insert.append(("variable", "CAMEL_CASE", NAMING_CONVENTIONS_REGEX["CAMEL_CASE"], "Secondary variable naming: CAMEL_CASE", False))
        elif func_var_convention == "CAMEL_CASE" and snake_pct > 0.05:
             rules_to_insert.append(("function", "SNAKE_CASE", NAMING_CONVENTIONS_REGEX["SNAKE_CASE"], "Secondary function naming: SNAKE_CASE", False))
             rules_to_insert.append(("variable", "SNAKE_CASE", NAMING_CONVENTIONS_REGEX["SNAKE_CASE"], "Secondary variable naming: SNAKE_CASE", False))

    else: # No clear dominant style for functions/variables, or percentages too low
        print("Warning: No clear dominant naming convention for functions/variables based on profile stats. Adding common defaults as inactive.")
        rules_to_insert.append(("function", "SNAKE_CASE", NAMING_CONVENTIONS_REGEX["SNAKE_CASE"], "Default function naming (snake_case)", False))
        rules_to_insert.append(("variable", "SNAKE_CASE", NAMING_CONVENTIONS_REGEX["SNAKE_CASE"], "Default variable naming (snake_case)", False))


    # 4. Test Functions and Test Classes (add with default active status)
    rules_to_insert.append(("test_function", "TEST_SNAKE_CASE", NAMING_CONVENTIONS_REGEX["TEST_SNAKE_CASE"], "Convention for test function names", True))
    rules_to_insert.append(("test_class", "TEST_PASCAL_CASE", NAMING_CONVENTIONS_REGEX["TEST_PASCAL_CASE"], "Convention for test class names", True))

    conn = None # Ensure conn is defined for finally block
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        for id_type in IDENTIFIER_TYPES:
            cursor.execute("UPDATE naming_rules SET is_active = FALSE WHERE identifier_type = ?", (id_type,))

        for rule_data in rules_to_insert:
            identifier_type, convention_name, regex_pattern, description, is_active = rule_data
            cursor.execute("""
                INSERT OR REPLACE INTO naming_rules
                (identifier_type, convention_name, regex_pattern, description, is_active)
                VALUES (?, ?, ?, ?, ?)
            """, (identifier_type, convention_name, regex_pattern, description, is_active))

        conn.commit()
        print(f"Naming rules populated in {db_path.resolve()} based on profile.")
        return True
    except sqlite3.Error as e:
        print(f"SQLite error while populating naming rules: {e}")
        return False
    except KeyError as e:
        print(f"Error: Naming convention key missing from NAMING_CONVENTIONS_REGEX: {e}")
        return False
    finally:
        if conn: # Check if conn was successfully assigned
            conn.close()

# (The existing if __name__ == '__main__' block for testing regexes can be kept.
#  A new section could be added to it to test populate_naming_rules_from_profile,
#  requiring database_setup.create_naming_conventions_db to be callable.)
# Example for __main__ extension:
#
# if __name__ == '__main__':
#     # ... (existing regex test code) ...
#
#     print("\n\n--- Testing DB Population ---")
#     # Create a dummy in-memory DB for this test
#     # Need to import create_naming_conventions_db or ensure it's run
#     # from ..database_setup import create_naming_conventions_db # Relative import if structured as package
#
#     # For standalone running of this file, we might need to mock create_naming_conventions_db
#     # or have a way to set up an in-memory db directly.
#
#     # Simplest for direct run: use a file, ensure schema exists via database_setup.py first.
#     test_db = Path("temp_naming_rules_test.db")
#     if test_db.exists(): test_db.unlink()
#
#     # conn_setup = sqlite3.connect(str(test_db)) # Create file
#     # from src.profiler.database_setup import create_naming_conventions_db # Assume it can be imported
#     # create_naming_conventions_db(db_path=test_db) # Call it
#     # conn_setup.close()
#
#     # Mock profile 1: snake_case dominant for func/var
#     profile1 = {
#         "identifier_snake_case_pct": 0.8,
#         "identifier_camelCase_pct": 0.1,
#         "identifier_UPPER_SNAKE_CASE_pct": 0.9
#     }
#     # populate_naming_rules_from_profile(test_db, profile1)
#     # Add asserts here by querying the DB
#
#     # if test_db.exists(): test_db.unlink() # Cleanup
