from pathlib import Path
from typing import Any, Dict, Tuple, Optional, List # Added List
import json # Add json import for dumping dicts in prompt

# Attempt to import the new LLM interfacer function
try:
    from src.profiler.llm_interfacer import get_llm_code_fix_suggestion, get_llm_cst_scaffold
except ImportError:
    get_llm_code_fix_suggestion = None
    get_llm_cst_scaffold = None # Also handle if this specific one is missing
    print("LLMCore Warning: Failed to import one or more LLM interfacer functions. Related LLM capabilities will be disabled.")


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
        Generates a LibCST scaffold script and an edit summary using an LLM.
        Args:
            context_data: Comprehensive context data dictionary.
        Returns:
            A tuple (cst_script_str, edit_summary_list), or (None, None) on failure.
        """
        print(f"LLMCore.generate_scaffold_patch called. Context keys: {list(context_data.keys())}")

        if get_llm_cst_scaffold is None or not self.llm_model_path:
            print("LLMCore Error: get_llm_cst_scaffold is not available or llm_model_path not configured. Cannot generate scaffold.")
            return None, None

        phase_description = context_data.get("phase_description", "N/A")
        target_file = str(context_data.get("phase_target_file", "N/A")) # Ensure string for prompt
        parameters = context_data.get("phase_parameters", {})
        code_snippets = context_data.get("retrieved_code_snippets", {})
        style_profile = context_data.get("style_profile", {})

        # Refined Prompt for LibCST script and delimited edit summary
        prompt = f"""[TASK]
You are an expert Python programmer specializing in code generation using LibCST.
Your goal is to generate a Python script that defines a LibCST VisitorBasedCodemodCommand or MatcherDecoratableCommand class.
This script will be used to perform a specific code modification as described in the phase details.
For complex logic or sections that will be filled in later (e.g., by another AI model), use placeholders like '__HOLE_0__', '__HOLE_1__', etc., in the generated *code within the LibCST command's methods*.

[OUTPUT FORMAT]
Provide your response as two distinct parts, separated by a specific delimiter:
1.  The LibCST Python script.
2.  An edit summary list enclosed in XML-like tags.

Example:
```python
# LibCST Script part
import libcst as cst
from libcst.codemod import VisitorBasedCodemodCommand, CodemodContext

class AddCommentCommand(VisitorBasedCodemodCommand):
    # ... (rest of the class definition) ...
    def leave_FunctionDef(self, original_node, updated_node):
        # ... logic using placeholders like self.new_code_with_hole = "__HOLE_0__" ...
        return updated_node # ... with changes ...
```
# SCRIPT_END_EDIT_SUMMARY_START
# - ADD_FUNCTION: new_dummy_function in module/file.py
# - INSERT_PLACEHOLDER: __HOLE_0__ for core logic in new_dummy_function
# EDIT_SUMMARY_END

[PHASE DETAILS]
Description: {phase_description}
Target File: {target_file}
Parameters: {json.dumps(parameters, indent=2)}

[CONTEXTUAL INFORMATION]
Relevant Code Snippets:
{json.dumps(code_snippets, indent=2)}

Style Profile (for generated code within the CST script, if applicable, and for the modification itself):
Indent Width: {style_profile.get('indent_size', style_profile.get('tab_width', 4))}
Quote Style: {style_profile.get('quote_style', 'double')}

[INSTRUCTIONS]
- Ensure the script is syntactically correct Python and uses LibCST correctly.
- The class should be parameterizable via its __init__ if details from 'Phase Parameters' are needed.
- Use placeholders like '__HOLE_0__' for complex logic within the generated code segments in the CST script.
- Ensure the edit summary is concise and uses the specified format.
"""
        if self.config.get("verbose_prompts", False): # Control verbosity of prompt logging
            print("\n--- LLMCore: Constructed Prompt for Scaffold Generation ---")
            print(prompt)
            print("--- End of Prompt ---\n")
        else:
            print("\n--- LLMCore: Constructed Prompt for Scaffold Generation (summary shown) ---")
            print(prompt[:300] + "...")
            print("--- End of Prompt Summary ---\n")


        llm_params = self.config.get("llm_scaffold_params", {})
        raw_llm_output = get_llm_cst_scaffold(
            model_path=self.llm_model_path,
            prompt=prompt,
            verbose=self.config.get("llm_verbose", False),
            n_gpu_layers=self.config.get("llm_n_gpu_layers", -1),
            n_ctx=llm_params.get("n_ctx", 4096),
            max_tokens_for_scaffold=llm_params.get("max_tokens", 2048),
            temperature=llm_params.get("temperature", 0.3)
        )

        if raw_llm_output is None:
            print("LLMCore Error: get_llm_cst_scaffold returned None. Failed to generate scaffold from LLM.")
            return None, None

        # Parse the raw output
        script_marker = "# SCRIPT_END_EDIT_SUMMARY_START"
        summary_start_marker = "# EDIT_SUMMARY_START" # Redundant if script_marker is primary
        summary_end_marker = "# EDIT_SUMMARY_END"

        extracted_cst_script_str: Optional[str] = None
        extracted_edit_summary_list: Optional[List[str]] = None

        try:
            if script_marker in raw_llm_output:
                parts = raw_llm_output.split(script_marker, 1)
                extracted_cst_script_str = parts[0].strip()

                summary_part = parts[1]
                # Summary part might still contain the start marker if it was also a stop token
                if summary_start_marker in summary_part:
                     summary_part = summary_part.split(summary_start_marker,1)[1]

                if summary_end_marker in summary_part:
                    summary_part = summary_part.split(summary_end_marker, 1)[0]

                extracted_edit_summary_list = [
                    line.strip() for line in summary_part.strip().splitlines()
                    if line.strip() and line.strip().startswith("- ") # Ensure lines are actual summary items
                ]
                # Clean up "- " prefix
                extracted_edit_summary_list = [line[2:] for line in extracted_edit_summary_list]

            else: # Fallback if markers are not perfectly produced
                print("LLMCore Warning: Script/summary delimiter not found in LLM output. Attempting heuristic parsing.")
                # Attempt to find Python code block for script
                code_block_match = re.search(r"```python\n(.*?)\n```", raw_llm_output, re.DOTALL)
                if code_block_match:
                    extracted_cst_script_str = code_block_match.group(1).strip()
                else: # Assume the whole thing might be the script if no markers
                    extracted_cst_script_str = raw_llm_output.strip()

                # For summary, if no markers, it's hard to get. Maybe it's after the script.
                # This part is highly unreliable without markers.
                # For now, if no markers, summary might be None or extracted based on other heuristics.
                # Let's assume for now if markers are missing, summary extraction is skipped or results in empty.
                if extracted_cst_script_str and len(raw_llm_output) > len(extracted_cst_script_str):
                    potential_summary_part = raw_llm_output[len(extracted_cst_script_str):].strip()
                    if potential_summary_part.startswith("#"): # Common for comments
                         extracted_edit_summary_list = [
                            line.strip()[2:] for line in potential_summary_part.splitlines()
                            if line.strip().startswith("# - ")
                         ]


            if not extracted_cst_script_str:
                print("LLMCore Error: Could not extract LibCST script from LLM output.")
                if self.verbose: print(f"LLM Raw Output for script extraction failure:\n{raw_llm_output}")
                return None, None

            if not extracted_edit_summary_list: # Default to a generic summary if none parsed
                print("LLMCore Warning: Could not extract edit summary from LLM output, or it was empty.")
                extracted_edit_summary_list = [f"GENERIC_EDIT_APPLIED to {target_file}"]


            # Basic validation of the script
            try:
                ast.parse(extracted_cst_script_str)
                if self.verbose: print(f"LLMCore Info: Extracted CST script is valid Python syntax. Length: {len(extracted_cst_script_str)}")
            except SyntaxError as e_syn:
                print(f"LLMCore Error: Extracted CST script has syntax errors: {e_syn}")
                if self.verbose: print(f"Problematic CST script:\n{extracted_cst_script_str}")
                return None, None

            print(f"LLMCore Info: Successfully generated scaffold. Script length: {len(extracted_cst_script_str)}, Summary items: {len(extracted_edit_summary_list)}")
            return extracted_cst_script_str, extracted_edit_summary_list

        except Exception as e_parse:
            print(f"LLMCore Error: Failed to parse LLM output for scaffold: {e_parse}")
            if self.verbose: print(f"LLM Raw Output for parsing failure:\n{raw_llm_output}")
            return None, None

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
