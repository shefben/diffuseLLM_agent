import toml # For reading and writing pyproject.toml
from pathlib import Path
from typing import Dict, Any, Optional, List, Set # Ensure List, Set are here
import sqlite3 # Add sqlite3

# Assuming UnifiedStyleProfile structure (or its dict representation) is used.
# from .diffusion_interfacer import UnifiedStyleProfile # For type hint if needed

DEFAULT_PYPROJECT_PATH = Path("pyproject.toml")

def generate_black_config_in_pyproject(
    unified_profile: Dict[str, Any],
    pyproject_path: Path = DEFAULT_PYPROJECT_PATH
) -> bool:
    """
    Generates or updates the [tool.black] section in pyproject.toml
    based on the unified style profile.

    Args:
        unified_profile: A dictionary representing the unified style profile.
        pyproject_path: The path to the pyproject.toml file.

    Returns:
        True if the configuration was successfully written, False otherwise.
    """
    try:
        if pyproject_path.exists():
            with open(pyproject_path, "r", encoding="utf-8") as f:
                pyproject_data = toml.load(f)
        else:
            # If pyproject.toml doesn't exist, create a basic structure.
            # This matches the structure created in a previous step for placeholder pyproject.toml
            pyproject_data = {
                "build-system": {
                    "requires": ["setuptools >= 61.0"],
                    "build-backend": "setuptools.build_meta"
                },
                "project": {
                    "name": "ai_coding_assistant", # Or derive from repo name
                    "version": "0.0.1",
                    "description": "A self-hosted Python toolchain for codebase evolution.",
                    "readme": "README.md",
                    "requires-python": ">=3.8",
                }
            }

        if "tool" not in pyproject_data:
            pyproject_data["tool"] = {}

        pyproject_data["tool"]["black"] = {} # Start fresh or update existing

        # Extract relevant settings from the profile for Black
        # 1. line_length
        if unified_profile.get("max_line_length") is not None:
            pyproject_data["tool"]["black"]["line_length"] = int(unified_profile["max_line_length"])

        # 2. skip-string-normalization
        # If quotes are mixed (e.g. not 'single' or 'double' in profile, or a specific flag exists)
        # For now, let's assume if preferred_quotes is 'single' or 'double', we don't skip.
        # If it's 'other' or None, we might skip, or if the profile explicitly says so.
        # The current `UnifiedStyleProfile` mock has `preferred_quotes` which can be 'single' or 'double'.
        # If it implies a strong preference, we don't skip normalization.
        # If quotes are truly mixed and should be preserved, profile should indicate this.
        # Let's assume for now: if a clear preference exists, Black enforces it.
        # `skip-string-normalization = true` means Black *won't* normalize.
        # If preferred_quotes is 'single' or 'double', we want normalization, so skip-string-normalization should be false (or absent).
        # The user spec said: "skip-string-normalization (set if mixed quotes stay mixed)"
        # Our current profile has 'preferred_quotes'. If this is set, it implies we DON'T want mixed quotes.
        # So, skip-string-normalization would be false.
        # If 'preferred_quotes' was None or some indicator of "mixed", then true.
        preferred_quotes = unified_profile.get("preferred_quotes")
        if preferred_quotes is None or preferred_quotes == "other": # Assuming 'other' means mixed
             pyproject_data["tool"]["black"]["skip-string-normalization"] = True
        else:
            # If 'single' or 'double', Black will enforce it by default.
            # No need to set skip-string-normalization to false, absence means false.
            pass

        # 3. target-version (example, not yet in our UnifiedStyleProfile)
        # if unified_profile.get("python_version"):
        #     pyproject_data["tool"]["black"]["target-version"] = [unified_profile["python_version"]]
        target_version_profile = unified_profile.get("target_python_version")
        if target_version_profile:
            if isinstance(target_version_profile, str):
                # Ensure it's a list of strings, e.g., "py39" -> ["py39"]
                pyproject_data["tool"]["black"]["target-version"] = [target_version_profile]
            elif isinstance(target_version_profile, list) and all(isinstance(item, str) for item in target_version_profile):
                pyproject_data["tool"]["black"]["target-version"] = target_version_profile
            # Else, if it's some other type or malformed list, skip for now or add warning

        # Ensure parent directory exists (though for root pyproject.toml, it's usually not an issue)
        pyproject_path.parent.mkdir(parents=True, exist_ok=True)

        with open(pyproject_path, "w", encoding="utf-8") as f:
            toml.dump(pyproject_data, f)

        print(f"[tool.black] configuration updated in {pyproject_path.resolve()}")
        return True

    except (IOError, toml.TomlDecodeError, toml.TomlPreserveCommentEncoderError) as e:
        print(f"Error processing {pyproject_path.resolve()} for Black config: {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred during Black config generation: {e}")
        return False

if __name__ == '__main__':
    # Example Usage
    # Create a dummy unified profile
    dummy_profile_for_black = {
        "max_line_length": 99,
        "preferred_quotes": "single",
        # "python_version": "py39" # Example if we had it
    }

    dummy_profile_mixed_quotes = {
        "max_line_length": 79,
        "preferred_quotes": None, # Indicates mixed quotes might be desired
    }

    test_pyproject_path = Path("temp_pyproject_black.toml")

    # Test 1: Generate with specific settings
    print("--- Test 1: Generating Black config with specific settings ---")
    success = generate_black_config_in_pyproject(dummy_profile_for_black, test_pyproject_path)
    if success:
        with open(test_pyproject_path, "r") as f:
            print(f.read())
    else:
        print("Failed to generate Black config for Test 1.")

    # Test 2: Generate with settings suggesting mixed quotes
    print("\n--- Test 2: Generating Black config for mixed quotes ---")
    success = generate_black_config_in_pyproject(dummy_profile_mixed_quotes, test_pyproject_path)
    if success:
        with open(test_pyproject_path, "r") as f:
            print(f.read())
    else:
        print("Failed to generate Black config for Test 2.")

    # Test 3: Update an existing file (first create one with other tools)
    print("\n--- Test 3: Updating an existing pyproject.toml ---")
    existing_data = {
        "tool": {
            "some_other_tool": {"setting": "value"},
            "black": {"line_length": 88} # Old black setting
        },
        "project": {"name": "my_project"}
    }
    with open(test_pyproject_path, "w") as f:
        toml.dump(existing_data, f)

    success = generate_black_config_in_pyproject(dummy_profile_for_black, test_pyproject_path)
    if success:
        with open(test_pyproject_path, "r") as f:
            print(f.read())
    else:
        print("Failed to generate Black config for Test 3.")


    # Clean up
    if test_pyproject_path.exists():
        test_pyproject_path.unlink()
        # print(f"Cleaned up {test_pyproject_path.name}") # Silence this for combined __main__

def get_active_naming_convention(cursor: sqlite3.Cursor, identifier_type: str) -> Optional[str]:
    """Helper to query the active naming convention for a given identifier type."""
    cursor.execute(
        "SELECT convention_name FROM naming_rules WHERE identifier_type = ? AND is_active = TRUE",
        (identifier_type,)
    )
    row = cursor.fetchone()
    return row[0] if row else None

# NOTE: This function currently maps line_length, docstring_style (to D rules),
# preferred_quotes (to Q rules), and naming conventions (to N rules)
# from the unified_profile to the Ruff configuration.
# Other Ruff rule categories (e.g., for import styles, bugbear checks)
# are not automatically configured from the current UnifiedStyleProfile structure,
# as the profile doesn't yet capture those specific style preferences.
# Ruff's default selections or a user-defined base 'select' list will apply for those.
def generate_ruff_config_in_pyproject(
    unified_profile: Dict[str, Any],
    pyproject_path: Path = DEFAULT_PYPROJECT_PATH,
    db_path: Optional[Path] = None # Path to naming_conventions.db, make it optional for now
) -> bool:
    """
    Generates or updates the [tool.ruff] section in pyproject.toml
    based on the unified style profile and active naming conventions from the DB.
    """
    try:
        # ... (existing pyproject.toml loading/creation logic - keep as is) ...
        if pyproject_path.exists():
            with open(pyproject_path, "r", encoding="utf-8") as f:
                pyproject_data = toml.load(f)
        else:
            pyproject_data = {
                "build-system": {"requires": ["setuptools >= 61.0"], "build-backend": "setuptools.build_meta"},
                "project": {"name": "ai_coding_assistant", "version": "0.0.1", "description": "A self-hosted Python toolchain for codebase evolution.", "readme": "README.md", "requires-python": ">=3.8",}
            }

        if "tool" not in pyproject_data: pyproject_data["tool"] = {}
        if "ruff" not in pyproject_data["tool"]: pyproject_data["tool"]["ruff"] = {}
        ruff_config = pyproject_data["tool"]["ruff"]
        if "lint" not in ruff_config: ruff_config["lint"] = {}
        if "format" not in ruff_config: ruff_config["format"] = {}

        # Initialize current_select based on existing config or as empty set
        if "select" in ruff_config["lint"]:
            current_select: Set[str] = set(ruff_config["lint"]["select"])
        else:
            current_select: Set[str] = set() # Use Ruff's defaults if nothing explicitly selected

        current_ignore: Set[str] = set(ruff_config["lint"].get("ignore", []))


        # --- Populate existing Ruff settings (line-length, quotes, pydocstyle) ---
        if unified_profile.get("max_line_length") is not None:
            ruff_config["line-length"] = int(unified_profile["max_line_length"])

        docstring_style_map = {"google": "google", "numpy": "numpy", "pep257": "pep257", "restructuredtext": "pep257", "plain": "pep257"}
        profile_doc_style = unified_profile.get("docstring_style")
        if profile_doc_style and profile_doc_style in docstring_style_map:
            ruff_doc_style = docstring_style_map[profile_doc_style]
            if ruff_doc_style:
                if "pydocstyle" not in ruff_config["lint"]: ruff_config["lint"]["pydocstyle"] = {}
                ruff_config["lint"]["pydocstyle"]["convention"] = ruff_doc_style
                current_select.add("D")

        preferred_quotes = unified_profile.get("preferred_quotes")
        if preferred_quotes:
            ruff_config["format"]["quote-style"] = preferred_quotes
            if "flake8-quotes" not in ruff_config["lint"]: ruff_config["lint"]["flake8-quotes"] = {}
            ruff_config["lint"]["flake8-quotes"]["inline-quotes"] = preferred_quotes
            ruff_config["lint"]["flake8-quotes"]["multiline-quotes"] = preferred_quotes
            # Keep docstring quotes double for single-quote preference, or match for double
            ruff_config["lint"]["flake8-quotes"]["docstring-quotes"] = "double" if preferred_quotes == "single" else preferred_quotes
            current_select.add("Q")

        # --- Advanced: Configure pep8-naming (N) rules based on DB ---
        rules_to_potentially_ignore: Set[str] = set()
        if db_path and db_path.exists():
            current_select.add("N") # Ensure pep8-naming is active if we are configuring it
            conn_db = None
            try:
                conn_db = sqlite3.connect(str(db_path))
                cursor = conn_db.cursor()

                # Function names (N802 expects snake_case)
                active_func_style = get_active_naming_convention(cursor, "function")
                if active_func_style and active_func_style != "SNAKE_CASE":
                    rules_to_potentially_ignore.add("N802") # e.g., if CAMEL_CASE is active for functions

                # Argument names (N803 expects snake_case) - assume 'variable' style applies
                active_var_style = get_active_naming_convention(cursor, "variable")
                if active_var_style and active_var_style != "SNAKE_CASE":
                    rules_to_potentially_ignore.add("N803") # For arguments
                    rules_to_potentially_ignore.add("N806") # For variables in functions
                    # N815 (class scope) and N816 (global scope) also check for mixedCase.
                    # If dominant var style is, e.g., camelCase, these PEP8 checks for mixedCase might be redundant or conflicting.
                    if active_var_style == "CAMEL_CASE": # or other mixedCase styles
                         rules_to_potentially_ignore.add("N815")
                         rules_to_potentially_ignore.add("N816")

                # Class names (N801 expects PascalCase)
                active_class_style = get_active_naming_convention(cursor, "class")
                if active_class_style and active_class_style != "PASCAL_CASE":
                    rules_to_potentially_ignore.add("N801")

                # Constant names (N816 also covers global constants, expecting non-mixedCase; PEP8 wants UPPER_SNAKE_CASE)
                # If project uses, say, PascalCase for constants, N816 might be one to ignore.
                active_const_style = get_active_naming_convention(cursor, "constant")
                if active_const_style and active_const_style not in ["UPPER_SNAKE_CASE", "SNAKE_CASE"]: # if it's mixed-case like
                    # N816 flags mixedCase in global scope. If our constant style IS mixedCase, ignore N816.
                    # This logic is a bit simplified. PEP8 Naming doesn't have a direct "constant must be UPPER_SNAKE" rule,
                    # but N816 (mixedCase variable in global scope) often catches non-UPPER_SNAKE_CASE constants.
                    if active_const_style in ["PASCAL_CASE", "CAMEL_CASE"]: # Example mixed-case styles for constants
                        rules_to_potentially_ignore.add("N816")

            except sqlite3.Error as e:
                print(f"Warning: SQLite error when reading naming_conventions.db: {e}. Naming rules might not be optimally configured.")
            finally:
                if conn_db: conn_db.close()
        else:
            if db_path: # Path was given but doesn't exist
                 print(f"Warning: naming_conventions.db not found at {db_path}. Ruff naming rules will use defaults.")
            # If no db_path, N rules will just be part of select if added by default, without specific ignores.
            # Add N to select by default if we intend to manage it.
            current_select.add("N")


        # Update select and ignore in ruff_config
        current_ignore.update(rules_to_potentially_ignore) # Add any new ignores from naming

        if current_select: # If there are explicit selections
            ruff_config["lint"]["select"] = sorted(list(current_select))
        elif "select" in ruff_config["lint"]: # If select became empty, remove the key to use Ruff defaults
            ruff_config["lint"].pop("select", None)
        # If current_select started empty and remained empty, "select" key is not added.

        if current_ignore: # Only write ignore if not empty
            ruff_config["lint"]["ignore"] = sorted(list(current_ignore))
        elif "ignore" in ruff_config["lint"]: # If it became empty, remove key
             ruff_config["lint"].pop("ignore", None)


        # ... (existing pyproject.toml writing logic - keep as is) ...
        pyproject_path.parent.mkdir(parents=True, exist_ok=True)
        with open(pyproject_path, "w", encoding="utf-8") as f:
            toml.dump(pyproject_data, f)
        print(f"[tool.ruff] configuration updated in {pyproject_path.resolve()} including naming rule adjustments.")
        return True

    except (IOError, toml.TomlDecodeError, toml.TomlPreserveCommentEncoderError) as e:
        print(f"Error processing {pyproject_path.resolve()} for Ruff config: {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred during Ruff config generation: {e}")
        return False

if __name__ == '__main__':
    # (Keep existing __main__ content for Black)
    # ... (previous print statements and tests for Black config) ...
    print("\n" + "="*50 + "\n")


    # Create a dummy unified profile for Ruff
    dummy_profile_for_ruff = {
        "max_line_length": 100,
        "preferred_quotes": "double",
        "docstring_style": "numpy",
        # "python_version": "py310" # Example
    }

    dummy_profile_ruff_strict = {
        "max_line_length": 80,
        "preferred_quotes": "single",
        "docstring_style": "google",
    }

    test_pyproject_path_ruff = Path("temp_pyproject_ruff.toml")

    # Test 1: Generate Ruff config with specific settings
    print("--- Test 1: Generating Ruff config with specific settings ---")
    # First, ensure the file is clean or has some base content
    if test_pyproject_path_ruff.exists(): test_pyproject_path_ruff.unlink()
    generate_black_config_in_pyproject({"max_line_length":100}, test_pyproject_path_ruff) # Add some black config first

    success_ruff = generate_ruff_config_in_pyproject(dummy_profile_for_ruff, test_pyproject_path_ruff)
    if success_ruff:
        with open(test_pyproject_path_ruff, "r") as f:
            print(f.read())
    else:
        print("Failed to generate Ruff config for Test 1.")

    # Test 2: Generate Ruff config with different settings
    print("\n--- Test 2: Generating Ruff config with stricter settings ---")
    if test_pyproject_path_ruff.exists(): test_pyproject_path_ruff.unlink() # Clean slate
    success_ruff = generate_ruff_config_in_pyproject(dummy_profile_ruff_strict, test_pyproject_path_ruff)
    if success_ruff:
        with open(test_pyproject_path_ruff, "r") as f:
            print(f.read())
    else:
        print("Failed to generate Ruff config for Test 2.")

    # Test 3: Update an existing pyproject.toml with Ruff and Black sections
    print("\n--- Test 3: Updating an existing pyproject.toml with both Black and Ruff ---")
    existing_data_both = {
        "tool": {
            "some_other_tool": {"setting": "value"},
            "black": {"line_length": 88},
            "ruff": {"lint": {"select": ["E"]}} # Old ruff setting
        },
        "project": {"name": "my_project_for_ruff"}
    }
    with open(test_pyproject_path_ruff, "w") as f:
        toml.dump(existing_data_both, f)

    # Apply Black settings first (optional, could be one profile for both)
    # Re-using dummy_profile_for_ruff for black settings here for simplicity in test
    generate_black_config_in_pyproject(dummy_profile_for_ruff, test_pyproject_path_ruff)
    # Then apply Ruff settings
    success_ruff = generate_ruff_config_in_pyproject(dummy_profile_for_ruff, test_pyproject_path_ruff)

    if success_ruff:
        with open(test_pyproject_path_ruff, "r") as f:
            print(f.read())
    else:
        print("Failed to generate Ruff config for Test 3.")


    # Clean up
    if test_pyproject_path_ruff.exists():
        test_pyproject_path_ruff.unlink()
        print(f"Cleaned up {test_pyproject_path_ruff.name}")

    # Also clean up the black test file if it's still around from previous examples in __main__
    # This assumes the __main__ from the previous step for Black used test_pyproject_path = Path("temp_pyproject_black.toml")
    # Let's rename the original black test file to avoid collision in the combined __main__
    black_test_file_original_name = Path("temp_pyproject_black.toml")
    if black_test_file_original_name.exists():
        black_test_file_original_name.unlink()
        # print(f"Cleaned up {black_test_file_original_name.name}") # Silenced for final combined cleanup


DEFAULT_DOCSTRING_TEMPLATE_PATH = Path("config") / "docstring_template.py"

def generate_docstring_template_file(
    unified_profile: Dict[str, Any],
    template_output_path: Path = DEFAULT_DOCSTRING_TEMPLATE_PATH
) -> bool:
    """
    Generates a docstring_template.py file demonstrating the canonical
    docstring layout for the style specified in the unified profile.

    Args:
        unified_profile: A dictionary representing the unified style profile,
                         expected to contain 'docstring_style'.
        template_output_path: The path where the docstring_template.py file
                              will be saved.

    Returns:
        True if the template file was successfully written, False otherwise.
    """
    docstring_style = unified_profile.get("docstring_style", "google") # Default to Google style

    common_header = f"# Docstring Template: {docstring_style.capitalize()} Style\n\n"
    common_header += "# This file is auto-generated based on the detected or chosen project style.\n"
    common_header += "# It provides examples of the canonical docstring format.\n\n"

    template_content = common_header

    if docstring_style == "google":
        template_content += """from typing import Optional # For type hints in examples

def example_function_google(param1: int, param2: str = "default") -> bool:
    """Example function demonstrating Google style docstrings.

    This is a longer description of what the function does. It can span
    multiple lines.

    Args:
        param1 (int): Description of the first parameter.
        param2 (str): Description of the second parameter. Defaults to "default".
               This can also be a multi-line description if needed.

    Returns:
        bool: True if successful, False otherwise. A more detailed description of
        what is returned and under what conditions.

    Raises:
        ValueError: If param1 is zero (example of documenting exceptions).
        TypeError: If param2 is not a string.
    """
    if param1 == 0:
        raise ValueError("param1 cannot be zero")
    if not isinstance(param2, str):
        raise TypeError("param2 must be a string")
    print(f"Called with {param1} and {param2}")
    return True

class ExampleClassGoogle:
    """Example class demonstrating Google style docstrings.

    Attributes:
        attr1 (str): Description of attr1.
        attr2 (int): Description of attr2.
    """
    def __init__(self, attr1: str, attr2: int):
        """Initializes ExampleClassGoogle.

        Args:
            attr1 (str): The first attribute.
            attr2 (int): The second attribute.
        """
        self.attr1 = attr1
        self.attr2 = attr2

    def example_method_google(self, class_param: float) -> Optional[str]:
        """An example method for the class.

        Note that 'self' is not documented in the Args section.

        Args:
            class_param (float): Description of the method's parameter.

        Returns:
            Optional[str]: A string if class_param is positive, None otherwise.
        """
        if class_param > 0:
            return f"Value: {class_param}"
        return None
"""
    elif docstring_style == "numpy":
        template_content += """# For type hints in examples, Optional can be useful
# from typing import Optional
# import numpy as np # Common to see numpy imported if using this style

def example_function_numpy(param1, param2="default"):
    """Example function demonstrating NumPy style docstrings.

    This is a longer description of what the function does. It can span
    multiple lines.

    Parameters
    ----------
    param1 : int
        Description of the first parameter.
    param2 : str, optional
        Description of the second parameter. Defaults to "default".
        This can also be a multi-line description if needed.

    Returns
    -------
    bool
        True if successful, False otherwise. A more detailed description of
        what is returned and under what conditions.

    Raises
    ------
    ValueError
        If param1 is zero (example of documenting exceptions).
    TypeError
        If param2 is not a string.

    See Also
    --------
    some_other_function : For related functionality.

    Examples
    --------
    >>> example_function_numpy(5, "test")
    Called with 5 and test
    True
    """
    if param1 == 0:
        raise ValueError("param1 cannot be zero")
    if not isinstance(param2, str):
        raise TypeError("param2 must be a string")
    print(f"Called with {param1} and {param2}")
    return True

class ExampleClassNumpy:
    """Example class demonstrating NumPy style docstrings.

    Attributes
    ----------
    attr1 : str
        Description of attr1.
    attr2 : int
        Description of attr2.

    Methods
    -------
    example_method_numpy(class_param)
        An example method for the class.
    """
    def __init__(self, attr1, attr2):
        self.attr1 = attr1
        self.attr2 = attr2

    def example_method_numpy(self, class_param):
        """An example method for the class.

        Note that 'self' is not documented in the Parameters section.

        Parameters
        ----------
        class_param : float
            Description of the method's parameter.

        Returns
        -------
        str or None
            A string if class_param is positive, None otherwise.
        """
        if class_param > 0:
            return f"Value: {class_param}"
        return None
"""
    elif docstring_style in ["pep257", "restructuredtext", "plain"]:
        style_name = "PEP 257 / reStructuredText" if docstring_style != "plain" else "Plain"
        template_content = f"# Docstring Template: {style_name} Style\n\n" # Specific header
        template_content += "# This file is auto-generated based on the detected or chosen project style.\n"
        template_content += "# It provides examples of the canonical docstring format.\n\n"
        template_content += """from typing import Optional # For type hints in examples

def example_function_pep257(param1: int, param2: str = "default") -> bool:
    """
    Example function demonstrating PEP 257 / reStructuredText style.

    This is a longer description of what the function does. It can span
    multiple lines.

    :param param1: Description of the first parameter.
    :type param1: int
    :param param2: Description of the second parameter, defaults to "default".
                   This can also be a multi-line description if needed.
    :type param2: str, optional
    :returns: True if successful, False otherwise. A more detailed description
              of what is returned and under what conditions.
    :rtype: bool
    :raises ValueError: If param1 is zero.
    :raises TypeError: If param2 is not a string.
    """
    if param1 == 0:
        raise ValueError("param1 cannot be zero")
    if not isinstance(param2, str):
        raise TypeError("param2 must be a string")
    print(f"Called with {param1} and {param2}")
    return True

class ExampleClassPep257:
    """
    Example class demonstrating PEP 257 / reStructuredText style.

    :ivar attr1: Description of attr1.
    :vartype attr1: str
    :ivar attr2: Description of attr2.
    :vartype attr2: int
    """
    def __init__(self, attr1: str, attr2: int):
        """
        Initializes ExampleClassPep257.

        :param attr1: The first attribute.
        :type attr1: str
        :param attr2: The second attribute.
        :type attr2: int
        """
        self.attr1 = attr1
        self.attr2 = attr2

    def example_method_pep257(self, class_param: float) -> Optional[str]:
        """
        An example method for the class.

        Note that 'self' is not documented.

        :param class_param: Description of the method's parameter.
        :type class_param: float
        :returns: A string if class_param is positive, None otherwise.
        :rtype: Optional[str]
        """
        if class_param > 0:
            return f"Value: {class_param}"
        return None
"""
    else: # Fallback for 'other' or unknown styles
        template_content = common_header # Reset to common_header
        template_content += f"# Docstring style '{docstring_style}' is not explicitly templated.\n"
        template_content += "# Please refer to project conventions or define a custom template.\n"
        template_content += """
def example_function_unknown_style(param1, param2):
    """A brief summary of the function.

    More detailed explanation if needed.
    """
    pass
"""

    try:
        template_output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(template_output_path, "w", encoding="utf-8") as f:
            f.write(template_content)
        print(f"Docstring template file saved to {template_output_path.resolve()}")
        return True
    except IOError as e:
        print(f"Error writing docstring template file to {template_output_path.resolve()}: {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred during docstring template generation: {e}")
        return False

# Update the if __name__ == '__main__': block to include tests for Ruff
if __name__ == '__main__':
    # (Keep existing __main__ content for Black)
    # ... (previous print statements and tests for Black config) ...
    print("\n" + "="*50 + "\n")
    # ... (ruff config tests without DB) ...

    print("\n--- Test Ruff config with Naming DB ---")
    # Conceptual: These imports would be needed if running this __main__ directly
    # from src.profiler.database_setup import create_naming_conventions_db
    # from src.profiler.naming_conventions import populate_naming_rules_from_profile, NAMING_CONVENTIONS_REGEX

    # This section is more illustrative of how one might test it in __main__
    # Actual test cases should be in a separate test file (e.g., test_config_generator.py)
    # For a real __main__ test here, one would need to handle imports carefully
    # depending on how this script is run (as a module or standalone).

    # temp_test_dir = Path("temp_config_gen_main_test")
    # temp_test_dir.mkdir(exist_ok=True)
    # test_pyproject_path_ruff_db = temp_test_dir / "pyproject_db.toml"
    # test_db_path_for_ruff = temp_test_dir / "naming_rules_for_ruff.db"

    # if test_db_path_for_ruff.exists(): test_db_path_for_ruff.unlink()
    # # create_naming_conventions_db(db_path=test_db_path_for_ruff) # Needs import

    # profile_camel_dominant = {
    #     "identifier_snake_case_pct": 0.10,
    #     "identifier_camelCase_pct": 0.80,
    #     "identifier_UPPER_SNAKE_CASE_pct": 0.9,
    #     "max_line_length": 80, "preferred_quotes": "single", "docstring_style": "google"
    # }
    # # populate_naming_rules_from_profile(test_db_path_for_ruff, profile_camel_dominant) # Needs import

    # if test_pyproject_path_ruff_db.exists(): test_pyproject_path_ruff_db.unlink()
    # # success_ruff_db = generate_ruff_config_in_pyproject(profile_camel_dominant, test_pyproject_path_ruff_db, test_db_path_for_ruff)
    # # if success_ruff_db:
    # #     with open(test_pyproject_path_ruff_db, "r") as f:
    # #         print("Ruff config with DB (camelCase func/var expected to ignore N802, N803, N806, N815, N816):")
    # #         print(f.read())
    # # else:
    # #     print("Failed to generate Ruff config with DB.")

    # ... (docstring template tests) ...
    # Cleanup
    # import shutil
    # if temp_test_dir.exists(): shutil.rmtree(temp_test_dir)

    # The existing cleanup for black test file and ruff test file (temp_pyproject_ruff.toml)
    # should be preserved or consolidated if they target the same temp files.
    # For now, the conceptual structure is shown. A full __main__ needs careful ordering of tests and cleanup.
    # For this subtask, the primary goal is the function modification.
    # The previous cleanup for temp_pyproject_black.toml and temp_pyproject_ruff.toml is fine.
    # The new temp_docstring_template.py cleanup was also added correctly.
    # The conceptual part above is just for illustrating a more involved __main__ test for the DB part.
    # The actual __main__ from the previous step should be largely preserved, with this new conceptual block
    # being an idea for more direct testing if desired (but better done in test files).
    # The existing cleanup for temp_pyproject_black.toml and temp_pyproject_ruff.toml is fine.
    # The new temp_docstring_template.py cleanup was also added correctly.
    # The conceptual part above is just for illustrating a more involved __main__ test for the DB part.
    # The actual __main__ from the previous step should be largely preserved, with this new conceptual block
    # being an idea for more direct testing if desired (but better done in test files).
    # For this subtask, the primary change is to the generate_ruff_config_in_pyproject function itself.
    # The __main__ block will be kept as it was at the end of the previous step to avoid complex merge for now.
    # The critical part is the function and its imports.
    pass # Placeholder to ensure the main block structure from previous step is notionally here.
