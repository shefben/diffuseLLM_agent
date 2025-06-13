from pathlib import Path
from typing import Any, Dict, Tuple, Optional, List # Added List
import json # Add json import for dumping dicts in prompt
import sqlite3 # Added for polish_patch naming conventions
import ast # For validating polished script syntax
import re # For parsing LLM scaffold output (already present)

# Attempt to import the new LLM interfacer function
try:
    from src.profiler.llm_interfacer import get_llm_code_fix_suggestion, get_llm_cst_scaffold, get_llm_polished_cst_script
    from src.utils.config_loader import DEFAULT_APP_CONFIG # For default model paths
except ImportError:
    get_llm_code_fix_suggestion = None
    get_llm_cst_scaffold = None
    get_llm_polished_cst_script = None
    DEFAULT_APP_CONFIG = {"models": {"agent_llm_gguf": "./models/placeholder_llm_agent.gguf"}} # Basic fallback
    print("LLMCore Warning: Failed to import one or more LLM interfacer functions or DEFAULT_APP_CONFIG. Related LLM capabilities will be disabled or use basic fallbacks.")


class LLMCore:
    def __init__(self,
                 app_config: Dict[str, Any],
                 style_profile: Dict[str, Any],
                 naming_conventions_db_path: Path
                 ):
        """
        Initializes the LLMCore.
        Args:
            app_config: The main application configuration dictionary.
            style_profile: Dictionary containing style profile information.
            naming_conventions_db_path: Path to the naming conventions database.
        """
        self.app_config = app_config
        self.style_profile = style_profile
        self.naming_conventions_db_path = naming_conventions_db_path

        self.verbose = self.app_config.get("general", {}).get("verbose", False)

        default_agent_llm_path = DEFAULT_APP_CONFIG.get("models", {}).get("agent_llm_gguf", "./models/placeholder_llm_agent.gguf")
        self.llm_model_path = self.app_config.get("models", {}).get("agent_llm_gguf", default_agent_llm_path)

        if self.verbose:
            print(f"LLMCore initialized. Verbose: {self.verbose}, Model Path: {self.llm_model_path}")
            print(f"  Style profile keys: {list(self.style_profile.keys())}")
            print(f"  Naming conventions DB: {self.naming_conventions_db_path}")

    def generate_scaffold_patch(self, context_data: Dict[str, Any]) -> Tuple[Optional[str], Optional[List[str]]]:
        """
        Generates a LibCST scaffold script and an edit summary using an LLM.
        Args:
            context_data: Comprehensive context data dictionary.
        Returns:
            A tuple (cst_script_str, edit_summary_list), or (None, None) on failure.
        """
        if self.verbose: print(f"LLMCore.generate_scaffold_patch called. Context keys: {list(context_data.keys())}")

        if get_llm_cst_scaffold is None or not self.llm_model_path:
            print("LLMCore Error: get_llm_cst_scaffold is not available or llm_model_path not configured. Cannot generate scaffold.")
            return None, None

        llm_params = self.app_config.get("llm_params", {})
        # General verbosity for LLMCore's own prints is self.verbose
        # llm_interfacer_verbose is for the get_llm_... function's internal logging
        llm_interfacer_verbose = self.app_config.get("general", {}).get("verbose_llm_calls", self.verbose)


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
Indent Width: {style_profile.get('indent_size', style_profile.get('tab_width', DEFAULT_APP_CONFIG.get('style_profile_defaults', {}).get('indent_size', 4)))}
Quote Style: {style_profile.get('quote_style', DEFAULT_APP_CONFIG.get('style_profile_defaults', {}).get('quote_style', 'double'))}

[INSTRUCTIONS]
- Ensure the script is syntactically correct Python and uses LibCST correctly.
- The class should be parameterizable via its __init__ if details from 'Phase Parameters' are needed.
- Use placeholders like '__HOLE_0__' for complex logic within the generated code segments in the CST script.
- Ensure the edit summary is concise and uses the specified format.
"""
        if llm_params.get("verbose_prompts", self.verbose): # Control verbosity of prompt logging
            print("\n--- LLMCore: Constructed Prompt for Scaffold Generation ---")
            print(prompt)
            print("--- End of Prompt ---\n")
        else:
            if self.verbose: print("\n--- LLMCore: Constructed Prompt for Scaffold Generation (summary shown) ---")
            if self.verbose: print(prompt[:300] + "...")
            if self.verbose: print("--- End of Prompt Summary ---\n")

        scaffold_model_path = self.llm_model_path
        scaffold_n_gpu_layers = llm_params.get("n_gpu_layers_default", DEFAULT_APP_CONFIG["llm_params"]["n_gpu_layers_default"])
        scaffold_n_ctx = llm_params.get("n_ctx_default", DEFAULT_APP_CONFIG["llm_params"]["n_ctx_default"])
        scaffold_max_tokens = llm_params.get("agent_scaffold_max_tokens", DEFAULT_APP_CONFIG["llm_params"]["agent_scaffold_max_tokens"])
        scaffold_temp = llm_params.get("agent_scaffold_temp", llm_params.get("temperature_default", DEFAULT_APP_CONFIG["llm_params"]["temperature_default"]))

        raw_llm_output = get_llm_cst_scaffold(
            model_path=scaffold_model_path,
            prompt=prompt,
            verbose=llm_interfacer_verbose, # For the interfacer function's logging
            n_gpu_layers=scaffold_n_gpu_layers,
            n_ctx=scaffold_n_ctx,
            max_tokens_for_scaffold=scaffold_max_tokens,
            temperature=scaffold_temp
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
        if self.verbose: print(f"LLMCore.polish_patch called. Context keys: {list(context_data.keys())}")

        if not completed_patch_script:
            if self.verbose: print("LLMCore.polish_patch Warning: Received None or empty completed_patch_script. Returning as is.")
            return completed_patch_script

        llm_params = self.app_config.get("llm_params", {})
        llm_interfacer_verbose = self.app_config.get("general", {}).get("verbose_llm_calls", self.verbose)

        # Extract relevant information from context_data
        style_profile = context_data.get("style_profile", self.style_profile) # Use instance style_profile as fallback
        naming_conventions_db_path = context_data.get("naming_conventions_db_path", self.naming_conventions_db_path) # Use instance path as fallback

        active_naming_conventions_str = "Naming conventions database not available or no active rules found."
        if naming_conventions_db_path and naming_conventions_db_path.exists():
            try:
                conn = sqlite3.connect(str(naming_conventions_db_path))
                cursor = conn.cursor()
                cursor.execute("SELECT identifier_type, convention_name FROM naming_rules WHERE is_active = TRUE")
                rules = cursor.fetchall()
                conn.close()
                if rules:
                    formatted_rules = [f"  - {identifier_type.capitalize()}: {convention_name}" for identifier_type, convention_name in rules]
                    active_naming_conventions_str = "Active Naming Conventions (guideline for script's own identifiers):\n" + "\n".join(formatted_rules)
                    active_naming_conventions_str += "\n(If specific conventions are not listed, use standard Python best practices.)"
                else:
                    active_naming_conventions_str = "No active naming conventions found in the database. Use standard Python best practices."
            except sqlite3.Error as e_sqlite:
                active_naming_conventions_str = f"Error accessing naming conventions DB: {e_sqlite}. Use standard Python best practices."
        elif naming_conventions_db_path: # Path provided but does not exist
             active_naming_conventions_str = f"Naming conventions DB not found at {naming_conventions_db_path}. Use standard Python best practices."


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
        # This prompt is complex, so verbose printing is more likely useful
        if llm_params.get("verbose_prompts", self.verbose):
            print("\n--- LLMCore: Constructed Prompt for Polishing Pass ---")
            print(prompt)
            print("--- End of Polishing Prompt ---\n")
        else:
            if self.verbose: print("\n--- LLMCore: Constructed Prompt for Polishing Pass (summary shown) ---")
            if self.verbose: print(prompt[:500] + "...")
            if self.verbose: print("--- End of Polishing Prompt Summary ---\n")

        if get_llm_polished_cst_script is None:
            if self.verbose: print("LLMCore Warning: get_llm_polished_cst_script not available. Skipping real LLM polishing.")
            return completed_patch_script

        polishing_model_path = self.llm_model_path # Uses the main agent LLM
        if not polishing_model_path: # Should be caught by __init__ or earlier checks
            print("LLMCore Warning: No model path configured for polishing. Skipping real LLM polishing.")
            return completed_patch_script

        # Parameters for the call to get_llm_polished_cst_script
        polish_n_gpu_layers = llm_params.get("n_gpu_layers_default", DEFAULT_APP_CONFIG["llm_params"]["n_gpu_layers_default"])
        polish_n_ctx = llm_params.get("n_ctx_default", DEFAULT_APP_CONFIG["llm_params"]["n_ctx_default"])
        polish_max_tokens = llm_params.get("agent_polish_max_tokens", DEFAULT_APP_CONFIG["llm_params"]["agent_polish_max_tokens"])
        polish_temp = llm_params.get("agent_polish_temp", llm_params.get("temperature_default", DEFAULT_APP_CONFIG["llm_params"]["temperature_default"]))

        # Re-construct the prompt for polishing with potentially updated context_data values
        # (style_profile and naming_conventions_db_path are now correctly sourced before this point)
        prompt = f"""[TASK DESCRIPTION]
You are an expert Python programmer specializing in reviewing and polishing LibCST (Concrete Syntax Tree) refactoring scripts.
Your goal is to refine the provided LibCST script for clarity, correctness, and adherence to coding conventions.

[INPUT LibCST SCRIPT TO POLISH]
```python
{completed_patch_script}
```

[CONTEXTUAL INFORMATION FOR POLISHING]
1.  **Style Profile (for the script's own code, if applicable, and for generated code):**
    ```json
    {json.dumps(style_profile, indent=2)}
    ```
2.  **Naming Conventions (guidance for identifier naming within the LibCST script itself):**
    {active_naming_conventions_str}

3.  **Target File for the Original Codemod Operation:** {context_data.get("phase_target_file", "N/A")}
4.  **Phase Parameters for the Original Codemod Operation:**
    ```json
    {json.dumps(context_data.get("phase_parameters", {}), indent=2)}
    ```

[POLISHING INSTRUCTIONS]
1.  **Harmonize Identifier Style:** Review variable names, function names, and class names *within the LibCST script itself*. Ensure they are consistent and follow standard Python conventions (e.g., snake_case for functions and variables, CamelCase for classes).
2.  **Ensure Necessary Imports:** Check if the script uses Python modules that require imports (e.g., `typing.List`, `collections.defaultdict`). If such imports are obviously missing from the top of the script, add them. Alternatively, if you are unsure about global import scope, list them in a comment block: `# REQUIRED_IMPORTS_START\n# import os\n# import typing\n# REQUIRED_IMPORTS_END`.
3.  **Review and Repair Obvious Issues:** Correct any clear type mismatches, syntax errors, or minor logical flaws *within the LibCST script's own logic*. This does NOT mean altering the intended refactoring behavior of the script on the target codebase.
4.  **Do NOT alter placeholders like `__HOLE_0__`, `__HOLE_1__`, etc.** These are intentional and will be filled by another process.
5.  **Output ONLY the complete, polished Python LibCST script string.** Do not add any explanations, apologies, or markdown formatting before or after the script block.

[POLISHED LibCST SCRIPT]
```python
"""

        llm_output_string = get_llm_polished_cst_script(
            model_path=polishing_model_path,
            prompt=prompt,
            verbose=llm_interfacer_verbose,
            n_gpu_layers=polish_n_gpu_layers,
            n_ctx=polish_n_ctx,
            max_tokens_for_polished_script=polish_max_tokens,
            temperature=polish_temp
        )

        if not llm_output_string: # Handles None or empty string
            if self.verbose: print("LLMCore Warning: LLM polishing returned no content. Proceeding with the unpolished script.")
            return completed_patch_script

        # LLM might sometimes include the closing ``` in its output.
        if llm_output_string.endswith("```"):
            llm_output_string = llm_output_string[:-3].rstrip()

        polished_script = llm_output_string

        # Basic validation of the polished script
        try:
            ast.parse(polished_script)
            if self.verbose: print(f"LLMCore: Polished script is valid Python syntax. Length: {len(polished_script)}.")
        except SyntaxError as e_syn:
            print(f"LLMCore Error: Polished script has syntax errors: {e_syn}. Returning original script.")
            if llm_interfacer_verbose: print(f"Problematic polished script:\n{polished_script}")
            return completed_patch_script

        if self.verbose: print(f"LLMCore: Polishing complete. Returning script (len: {len(polished_script)}).")
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
        if self.verbose: print(f"LLMCore.propose_repair_diff called with traceback: '{traceback_str}'. Context keys: {list(context_data.keys())}")

        llm_params = self.app_config.get("llm_params", {})
        llm_interfacer_verbose = self.app_config.get("general", {}).get("verbose_llm_calls", self.verbose)

        if "DUPLICATE_DETECTED: REUSE_EXISTING_HELPER" in traceback_str:
            if self.verbose: print("LLMCore: Received DUPLICATE_DETECTED. Generating LibCST script to comment out previous attempt.")

            entity_name_str = context_data.get("phase_parameters", {}).get("function_name") or \
                              context_data.get("phase_parameters", {}).get("class_name") or \
                              "unknown_entity"
            current_failed_script_content = str(context_data.get('current_patch_candidate', '# Original script could not be retrieved from context.'))

            header_comment_text = f"Original script commented out due to DUPLICATE_DETECTED for entity: {entity_name_str}.\nPlease refactor the original logic to use the existing helper or rename the entity."

            script_lines_to_embed = current_failed_script_content.splitlines()

            empty_line_calls_str_list = []
            for line_content in header_comment_text.split('\n'):
                # Escape for embedding inside an f-string within another f-string, then inside cst.Comment
                escaped_line = line_content.replace('\\', '\\\\').replace('"', '\\"').replace("'", "\\'")
                empty_line_calls_str_list.append(f'cst.EmptyLine(comment=cst.Comment(f"# {escaped_line}"))')

            empty_line_calls_str_list.append('cst.EmptyLine()') # Blank line for separation

            for line_content in script_lines_to_embed:
                escaped_line = line_content.replace('\\', '\\\\').replace('"', '\\"').replace("'", "\\'")
                # The line itself is already a comment line if we are commenting out a script
                # but if current_failed_script_content is not a script but some other text, prefixing is safer.
                # The previous logic prefixed with "# ". Here, we are generating a CST node.
                # The cst.Comment will handle the "#".
                empty_line_calls_str_list.append(f'cst.EmptyLine(comment=cst.Comment(f"# {escaped_line}"))')

            new_body_elements_initializer_str = "[\n            " + ",\n            ".join(empty_line_calls_str_list) + "\n        ]"

            generated_repair_script = f"""
import libcst as cst
from libcst.codemod import CodemodContext, VisitorBasedCodemodCommand

class ReplaceContentWithCommentedOutScriptCommand(VisitorBasedCodemodCommand):
    DESCRIPTION = "Replaces the module's content with a pre-defined commented-out script and header due to a duplicate entity error."

    # No __init__ needed if content is embedded directly or not parameterized further

    def leave_Module(self, original_node: cst.Module, updated_node: cst.Module) -> cst.Module:
        # This list of cst.EmptyLine objects will form the new body of the module.
        # Each string passed to cst.Comment() will be prefixed with a '#' by LibCST.
        new_body_elements = {new_body_elements_initializer_str}

        return updated_node.with_changes(body=new_body_elements, header=[], footer=[])

# To ensure the Patcher can find this class by a known name if needed.
COMMAND_CLASS_NAME = "ReplaceContentWithCommentedOutScriptCommand"
"""
            if self.verbose: print(f"LLMCore: Generated repair LibCST script for DUPLICATE_DETECTED (length: {len(generated_repair_script)}).")
            return generated_repair_script.strip()

        # Generic error handling using LLM if model path is configured
        # Use the specific repair_llm_gguf if defined, else fallback to agent_llm_gguf
        repair_model_path = self.app_config.get("models", {}).get("repair_llm_gguf", self.llm_model_path)

        if not repair_model_path or get_llm_code_fix_suggestion is None:
            if self.verbose: print("LLMCore Warning: LLM repair model path not configured or get_llm_code_fix_suggestion not available. Falling back to generic mock fix for other errors.")
            if "SyntaxError" in traceback_str:
                if self.verbose: print("LLMCore: Proposing a generic syntax fix (fallback).")
                return f"# Mock syntax repair for: {traceback_str[:100]}\nimport libcst as cst\nclass MinimalSyntaxFix(cst.VisitorBasedCodemodCommand):\n    pass"
            return f"# LLM model path not configured. Generic mock repair for: {traceback_str[:100]}"

        if self.verbose: print(f"LLMCore: Attempting LLM-based repair for traceback: {traceback_str[:200]}... using model {repair_model_path}")

        current_failed_script_from_context = context_data.get('current_patch_candidate')
        if isinstance(current_failed_script_from_context, str) and current_failed_script_from_context.strip():
            current_failed_script = current_failed_script_from_context
        elif current_failed_script_from_context is not None:
            current_failed_script = str(current_failed_script_from_context)
            if not current_failed_script.strip():
                 current_failed_script = "# Original script was empty or whitespace."
        else:
            current_failed_script = "# Original script was not provided in context_data['current_patch_candidate']."
            if self.verbose: print("LLMCore Warning: 'current_patch_candidate' not found or is None in context_data for repair.")

        phase_description = context_data.get("phase_description", "N/A")
        target_file = context_data.get("phase_target_file")

        additional_llm_context = {
            "style_profile": context_data.get("style_profile", self.style_profile),
            "code_snippets": context_data.get("retrieved_code_snippets")
        }

        repair_n_gpu_layers = llm_params.get("agent_repair_n_gpu_layers", llm_params.get("n_gpu_layers_default", DEFAULT_APP_CONFIG["llm_params"]["n_gpu_layers_default"]))
        repair_n_ctx = llm_params.get("agent_repair_n_ctx", llm_params.get("n_ctx_default", DEFAULT_APP_CONFIG["llm_params"]["n_ctx_default"]))
        repair_max_tokens = llm_params.get("agent_repair_max_tokens", DEFAULT_APP_CONFIG["llm_params"]["agent_repair_max_tokens"])
        repair_temp = llm_params.get("agent_repair_temp", llm_params.get("temperature_default", DEFAULT_APP_CONFIG["llm_params"]["temperature_default"]))

        suggested_fix_script = get_llm_code_fix_suggestion(
            model_path=repair_model_path,
            original_code_script=current_failed_script,
            error_traceback=traceback_str,
            phase_description=phase_description,
            target_file=target_file,
            additional_context=additional_llm_context,
            n_gpu_layers=repair_n_gpu_layers,
            n_ctx=repair_n_ctx,
            max_tokens=repair_max_tokens,
            temperature=repair_temp,
            verbose=llm_interfacer_verbose
        )

        if suggested_fix_script:
            if self.verbose: print(f"LLMCore: LLM suggested a fix script (len: {len(suggested_fix_script)}).")
        else:
            if self.verbose: print("LLMCore: LLM did not return a suggestion. Returning None.")

        return suggested_fix_script
