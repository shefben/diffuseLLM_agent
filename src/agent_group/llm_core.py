from pathlib import Path
from typing import Any, Dict, Tuple, Optional, List # Added List
import json # Add json import for dumping dicts in prompt

# Attempt to import the new LLM interfacer function
try:
    from src.profiler.llm_interfacer import get_llm_code_fix_suggestion
except ImportError:
    get_llm_code_fix_suggestion = None
    print("LLMCore Warning: Failed to import get_llm_code_fix_suggestion. LLM-based repair will be disabled.")


class LLMCore:
    def __init__(self,
                 style_profile: Dict[str, Any],
                 naming_conventions_db_path: Path,
                 config: Optional[Dict[str, Any]] = None,
                 llm_model_path: Optional[str] = None): # New parameter
        """
        Initializes the LLMCore.
        Args:
            style_profile: Dictionary containing style profile information.
            naming_conventions_db_path: Path to the naming conventions database.
            config: Optional dictionary for LLM core specific configurations.
            llm_model_path: Optional path to GGUF model for repairs.
        """
        self.style_profile = style_profile
        self.naming_conventions_db_path = naming_conventions_db_path
        self.config = config if config else {}
        self.llm_model_path = llm_model_path # Store new parameter
        print(f"LLMCore initialized with style_profile, naming_conventions_db_path: {naming_conventions_db_path}, config: {self.config}, llm_model_path: {self.llm_model_path}")

    def generate_scaffold_patch(self, context_data: Dict[str, Any]) -> Tuple[Optional[str], Optional[List[str]]]:
        """
        Generates a scaffold patch (LibCST edit script string) and an edit summary.
        Args:
            context_data: Comprehensive context data dictionary.
        Returns:
            A tuple containing the mock LibCST script string (or None) and an edit summary list (or None).
        """
        print(f"LLMCore.generate_scaffold_patch called. Context keys: {list(context_data.keys())}")

        # Extract relevant information from context_data
        phase_description = context_data.get("phase_description", "No phase description provided.")
        target_file = context_data.get("phase_target_file", "No target file specified.")
        parameters = context_data.get("phase_parameters", {})
        code_snippets = context_data.get("retrieved_code_snippets", {"info": "No snippets retrieved."})
        style_profile = context_data.get("style_profile", {})

        # Construct a detailed prompt string
        prompt = f"""[LLM TASK: Generate LibCST Python Edit Script and Edit Summary]

Objective:
Generate a Python script using LibCST that performs a specific code modification.
The script should define a VisitorBasedCodemodCommand class to implement the change.
For complex logic or sections that require further refinement (e.g., by a diffusion model),
use placeholders like '__HOLE_0__', '__HOLE_1__', etc., in the generated code within the LibCST script.

Output Format:
1.  A Python string variable containing the complete LibCST script.
2.  A Python list of strings variable named "edit_summary", where each string describes a high-level change.
    Example: edit_summary = ["ADD_DECORATOR @transactional to func:process_payment line 33 in payment_service.py", "REPLACE_BODY func:get_data in data_utils.py with placeholder __HOLE_0__"]

Task Details:
----------------
Phase Description: {phase_description}
Target File: {target_file}
Phase Parameters: {json.dumps(parameters, indent=2)}

Relevant Code Snippets:
-------------------------
{json.dumps(code_snippets, indent=2)}

Style Profile (guidelines for generated code within the CST script, if applicable, and for the modification itself):
---------------------------------------------------------------------------------------------------------------
Indent Width: {style_profile.get('indent_size', style_profile.get('tab_width', 4))}
Quote Style: {style_profile.get('quote_style', 'double')}
(Other relevant style elements could be included here)

Instructions for LibCST script:
-------------------------------
- Ensure all necessary LibCST imports are included (e.g., libcst as cst, libcst.matchers as m, VisitorBasedCodemodCommand).
- The command should be parameterizable via its __init__ method if details from 'Phase Parameters' are needed.
- Use placeholders like '__HOLE_0__' for parts of the new code that are complex or require later expansion.
- The script should be executable and define at least one class inheriting from VisitorBasedCodemodCommand.

Begin Script and Summary:
"""
        print("\n--- LLMCore: Constructed Prompt for Scaffold Generation ---")
        print(prompt)
        print("--- End of Prompt ---\n")

        # Mock LLM Interaction
        print("LLMCore: Using MOCK LLM response for scaffold generation.")

        # Example: If phase is to add a function (could be inferred from phase_description or parameters)
        func_name_param = parameters.get("function_name", "new_mock_function")

        mock_cst_script_str = f'''
import libcst as cst
import libcst.matchers as m
from libcst.codemod import CodemodContext, VisitorBasedCodemodCommand

class AddFunctionWithPlaceholderCommand(VisitorBasedCodemodCommand):
    DESCRIPTION: str = "Adds a new function with a placeholder body, using a parameter for the function name."

    def __init__(self, context: CodemodContext, function_name: str, placeholder_content: str):
        super().__init__(context)
        self.function_name = function_name
        self.placeholder_content = placeholder_content # e.g., "__HOLE_0__"

    def leave_Module(self, original_node: cst.Module, updated_node: cst.Module) -> cst.Module:
        new_function_code = f"""
def {self.function_name}():
    # {self.placeholder_content}
    pass
"""
        new_function_def = cst.parse_statement(new_function_code)

        # Example: Add to the end of the module
        # More sophisticated logic might insert it relative to other elements.
        return updated_node.with_changes(body=list(updated_node.body) + [new_function_def])

# How to instantiate (example):
# cmd_context = CodemodContext()
# command_instance = AddFunctionWithPlaceholderCommand(cmd_context, function_name="{func_name_param}", placeholder_content="__HOLE_0__ # Business logic for {func_name_param}")
'''

        mock_edit_summary_list = [
            f"ADD_FUNCTION name:{func_name_param} in file:{target_file}",
            f"INSERT_PLACEHOLDER __HOLE_0__ in {func_name_param}"
        ]

        # Handle a hypothetical 'modify_class' operation
        if "modify_class" in phase_description.lower() or parameters.get("operation_type") == "modify_class":
            class_name_param = parameters.get("class_name", "ExistingMockClass")
            method_name_param = parameters.get("method_name", "new_method_in_class")
            mock_cst_script_str = f'''
import libcst as cst
import libcst.matchers as m
from libcst.codemod import CodemodContext, VisitorBasedCodemodCommand

class ModifyClassAddMethodCommand(VisitorBasedCodemodCommand):
    DESCRIPTION: str = "Adds a new method with a placeholder to an existing class."

    def __init__(self, context: CodemodContext, class_name: str, method_name: str, placeholder_content: str):
        super().__init__(context)
        self.class_name = class_name
        self.method_name = method_name
        self.placeholder_content = placeholder_content

    @m.call_if_inside(m.ClassDef(name=m.Name(value=lambda val: val == self.class_name)))
    def leave_ClassDef(self, original_node: cst.ClassDef, updated_node: cst.ClassDef) -> cst.ClassDef:
        new_method_code = f"""
    def {self.method_name}(self):
        # {self.placeholder_content}
        return None
"""
        # WARNING: cst.parse_statement expects a statement. A method def is part of a ClassDef body.
        # This direct parsing and appending might need refinement for correct indentation and structure.
        # A safer way is to construct CST nodes directly or use a more complex parsing logic.
        # For this mock, we'll assume a simplified approach that might need fixing in a real scenario.
        try:
            # This is not ideal, parsing a full class just to get a method, then extracting.
            temp_class_def = cst.parse_module(f"class Temp:\\n{new_method_code.strip()}").body[0]
            if isinstance(temp_class_def, cst.ClassDef) and temp_class_def.body.body:
                new_method_node = temp_class_def.body.body[0]
                # Ensure it's a FunctionDef before adding
                if isinstance(new_method_node, cst.FunctionDef):
                    # Add to existing methods
                    updated_body_elements = list(updated_node.body.body) + [new_method_node]
                    return updated_node.with_changes(body=updated_node.body.with_changes(body=updated_body_elements))
        except Exception as e:
            print(f"Mock LLM: Error constructing class method node: {e}")

        return updated_node # Return original if modification failed

# How to instantiate (example):
# cmd_context = CodemodContext()
# command_instance = ModifyClassAddMethodCommand(cmd_context, class_name="{class_name_param}", method_name="{method_name_param}", placeholder_content="__HOLE_CLASS_METHOD_0__")
'''
            mock_edit_summary_list = [
                f"MODIFY_CLASS name:{class_name_param} in file:{target_file}",
                f"ADD_METHOD name:{method_name_param} to class:{class_name_param}",
                f"INSERT_PLACEHOLDER __HOLE_CLASS_METHOD_0__ in {class_name_param}.{method_name_param}"
            ]

        # Simulate returning the script and summary
        return mock_cst_script_str, mock_edit_summary_list

    def polish_patch(self, completed_patch_script: Optional[str], context_data: Dict[str, Any]) -> Optional[str]:
        """
        Polishes a LibCST Python edit script.
        Args:
            completed_patch_script: The LibCST script string, potentially after diffusion expansion.
            context_data: Comprehensive context data dictionary.
        Returns:
            The polished LibCST script string.
        """
        print(f"LLMCore.polish_patch called. Context keys: {list(context_data.keys())}")

        if not completed_patch_script:
            print("LLMCore.polish_patch Warning: Received None or empty completed_patch_script. Returning as is.")
            return completed_patch_script

        # Extract relevant information from context_data
        style_profile = context_data.get("style_profile", {})
        naming_conventions_db_path = context_data.get("naming_conventions_db_path") # Path object
        # repository_digester = context_data.get("repository_digester") # Digester instance
        # target_file = context_data.get("phase_target_file")

        prompt = f"""[LLM TASK: Polish LibCST Python Edit Script]

Objective:
Review and polish the provided LibCST Python edit script. Focus on:
1.  Harmonizing identifier styles (e.g., variable names, function names within the script) according to conventions (see Naming Conventions DB path).
2.  Inserting any obviously missing standard library imports required by the CST script itself (e.g., `typing.List` if used).
3.  Repairing obvious type mismatches or syntax issues *within the LibCST script code itself*, not the code it generates.
    (Assume type information for the target codebase is accessible via the provided digester, if needed for context, but prioritize script correctness).

Input LibCST Script to Polish:
------------------------------
{completed_patch_script}

Contextual Information:
-----------------------
Style Profile (for generated code by the script, but also hints for script style):
{json.dumps(style_profile, indent=2)}

Naming Conventions Database Path (for harmonizing identifiers in the script):
{str(naming_conventions_db_path) if naming_conventions_db_path else "Not provided"}

Target File for the Original Codemod: {context_data.get("phase_target_file", "N/A")}
Phase Parameters for original Codemod: {json.dumps(context_data.get("phase_parameters", {}), indent=2)}


Instructions for Polishing:
---------------------------
- Ensure the script remains a valid Python script that uses LibCST.
- Do NOT alter placeholders like '__HOLE_0__'.
- Focus on improving the LibCST script's internal quality, adherence to conventions, and robustness.

Begin Polished Script:
"""
        print("\n--- LLMCore: Constructed Prompt for Polishing Pass ---")
        print(prompt[:1000] + "..." if len(prompt) > 1000 else prompt) # Print snippet if too long
        print("--- End of Polishing Prompt ---\n")

        # Mock LLM Interaction
        print("LLMCore: Using MOCK LLM response for polishing.")

        polished_script = "# Script polished by mock LLM (LLMCore.polish_patch)\n" + completed_patch_script

        # Simulate identifier harmonization (very superficial mock)
        # Example: if the scaffold/expansion used a generic name like "placeholder_var" or "temp_method_name"
        # This would typically require parsing the script, identifying symbols, and applying rules.
        if "placeholder_content" in polished_script and "placeholder_content =" not in polished_script : # Avoid replacing variable assignment
             polished_script = polished_script.replace("placeholder_content", "polished_placeholder_content_var")
             print("  Mock Polish: Replaced 'placeholder_content' with 'polished_placeholder_content_var'")

        # Simulate adding a missing import (very superficial mock)
        if "VisitorBasedCodemodCommand" in polished_script and "from libcst.codemod import CodemodContext, VisitorBasedCodemodCommand" not in polished_script:
            if "import libcst.matchers as m" in polished_script: # find a line to insert after
                 polished_script = polished_script.replace(
                    "import libcst.matchers as m",
                    "import libcst.matchers as m\nfrom libcst.codemod import CodemodContext, VisitorBasedCodemodCommand"
                )
            else: # prepend
                 polished_script = "from libcst.codemod import CodemodContext, VisitorBasedCodemodCommand\n" + polished_script
            print("  Mock Polish: Added 'from libcst.codemod import CodemodContext, VisitorBasedCodemodCommand'")


        print(f"LLMCore: Polishing complete. Returning script (len: {len(polished_script)}).")
        return polished_script

    def propose_repair_diff(self, traceback_str: str, context_data: Dict[str, Any]) -> Optional[str]: # Return type changed to Optional[str] for script
        """
        Placeholder for proposing a repair diff (as a new script string) based on a traceback.
        Args:
            traceback_str: The traceback string from a failed validation or specific error signal.
            context_data: Comprehensive context data dictionary.
        Returns:
            A mock repair script string (or None).
        """
        print(f"LLMCore.propose_repair_diff called with traceback: '{traceback_str}'. Context keys: {list(context_data.keys())}")

        if "DUPLICATE_DETECTED: REUSE_EXISTING_HELPER" in traceback_str:
            print("LLMCore: Received DUPLICATE_DETECTED. Attempting to generate a patch that reuses existing helper.")
            target_file = context_data.get("phase_target_file", "unknown_target.py")
            original_planned_func_name = context_data.get("phase_parameters", {}).get("function_name", "original_planned_function")

            mock_reuse_script = f'''# Mock repair for DUPLICATE_DETECTED from LLMCore.
# Original intent might have been to create: {original_planned_func_name} in {target_file}
# This script should ideally modify the call site to use 'existing_helper_function_abc'
# For now, it's just a placeholder script.
import libcst as cst
class ReuseHelperInsteadCommand(cst.VisitorBasedCodemodCommand):
    DESCRIPTION = "Replaces a new function call with a call to an existing helper."
    def __init__(self, context): super().__init__(context) # Simplified init
    def leave_Module(self, original_node, updated_node):
        comment = cst.Comment("# LLMCore: Code modified to use existing_helper_function_abc() instead of new {original_planned_func_name}().")
        new_body = list(updated_node.body) + [cst.EmptyLine(comment=comment)]
        return updated_node.with_changes(body=new_body)
# Instantiation: ReuseHelperInsteadCommand(CodemodContext())
'''
            return mock_reuse_script

        # Generic error handling using LLM if model path is configured
        if not self.llm_model_path or get_llm_code_fix_suggestion is None:
            print("LLMCore Warning: LLM model path not configured or get_llm_code_fix_suggestion not available. Falling back to generic mock fix for other errors.")
            if "SyntaxError" in traceback_str: # Keep basic syntax error mock as fallback
                print("LLMCore: Proposing a generic syntax fix (fallback).")
                return f"# Mock syntax repair for: {traceback_str[:100]}\nimport libcst as cst\nclass MinimalSyntaxFix(cst.VisitorBasedCodemodCommand):\n    pass"
            return f"# LLM model path not configured. Generic mock repair for: {traceback_str[:100]}"

        print(f"LLMCore: Attempting LLM-based repair for traceback: {traceback_str[:200]}...")
        current_failed_script = context_data.get('current_patch_candidate', '# No script provided in context_data["current_patch_candidate"]')
        if not isinstance(current_failed_script, str): # Ensure it's a string
            current_failed_script = str(current_failed_script)

        phase_description = context_data.get("phase_description", "N/A")
        target_file = context_data.get("phase_target_file")

        # Prepare additional context for the fix suggestion more carefully
        additional_llm_context = {
            "style_profile": context_data.get("style_profile"),
            "code_snippets": context_data.get("retrieved_code_snippets")
            # Avoid passing the whole digester or complex objects like handles directly to LLM prompt
        }

        suggested_fix_script = get_llm_code_fix_suggestion(
            model_path=self.llm_model_path,
            original_code_script=current_failed_script,
            error_traceback=traceback_str,
            phase_description=phase_description,
            target_file=target_file,
            additional_context=additional_llm_context,
            # n_gpu_layers, max_tokens, temperature can be sourced from self.config if needed
            n_gpu_layers=self.config.get("llm_repair_n_gpu_layers", -1),
            max_tokens=self.config.get("llm_repair_max_tokens", 1024),
            temperature=self.config.get("llm_repair_temperature", 0.4),
            verbose=self.config.get("llm_repair_verbose", False)
        )

        if suggested_fix_script:
            print(f"LLMCore: LLM suggested a fix script (len: {len(suggested_fix_script)}).")
            # print(f"  Suggested fix script preview:\n{suggested_fix_script[:300]}...") # For debugging
        else:
            print("LLMCore: LLM did not return a suggestion. Returning None.")

        return suggested_fix_script
