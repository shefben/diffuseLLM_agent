from typing import Any, Dict, List, Optional
import re

# Attempt to import get_llm_code_infill and get_divot5_code_infill, handle if not available
try:
    from src.profiler.llm_interfacer import get_llm_code_infill, get_divot5_code_infill
    from src.utils.config_loader import DEFAULT_APP_CONFIG
except ImportError:
    get_llm_code_infill = None
    get_divot5_code_infill = None
    DEFAULT_APP_CONFIG = { # Basic fallback for model paths if main import fails
        "models": {
            "agent_llm_gguf": "./models/placeholder_llm_agent.gguf",
            "divot5_infill_model_dir": "./models/placeholder_divot5_infill/"
        },
        "llm_params": {}, "divot5_fim_params": {} # Empty dicts for params
    }
    print("DiffusionCore Warning: Failed to import LLM interfacer functions or DEFAULT_APP_CONFIG. LLM-based infilling will be disabled or use basic fallbacks.")


class DiffusionCore:
    def __init__(self, app_config: Dict[str, Any], style_profile: Dict[str, Any]):
        """
        Initializes the DiffusionCore.
        Args:
            app_config: The main application configuration dictionary.
            style_profile: Dictionary containing style profile information.
        """
        self.app_config = app_config
        self.style_profile = style_profile # Store style_profile if needed by other methods
        self.verbose = self.app_config.get("general", {}).get("verbose", False)
        if self.verbose:
            print(f"DiffusionCore initialized. Verbose: {self.verbose}")
            print(f"  Style profile keys: {list(self.style_profile.keys())}")

    def expand_scaffold(self, scaffold_patch_script: Optional[str], edit_summary_str: Optional[str], context_data: Dict[str, Any]) -> Optional[str]:
        """
        Expands placeholders in a scaffold LibCST patch script using LLM-based code infilling.
        Args:
            scaffold_patch_script: The LibCST script string with placeholders (e.g., __HOLE_0__).
            edit_summary_str: String representation of the edit summary from the LLM scaffolding pass.
            context_data: Comprehensive context data dictionary.
        Returns:
            The completed LibCST script string with placeholders filled, or the original/partially filled on error.
        """
        print(f"DiffusionCore.expand_scaffold called. Context keys: {list(context_data.keys())}. Edit summary: '{edit_summary_str}'")

        if not scaffold_patch_script:
            if self.verbose: print("DiffusionCore Warning: Received None or empty scaffold_patch_script. Returning as is.")
            return scaffold_patch_script

        if self.verbose: print(f"  Initial scaffold script (first 200 chars): '{scaffold_patch_script[:200]}...'")

        agent_infill_config = self.app_config.get("agent_infill", {})
        infill_type = agent_infill_config.get("type", "gguf").lower()
        llm_params = self.app_config.get("llm_params", {})
        divot5_params = self.app_config.get("divot5_fim_params", {})
        llm_interfacer_verbose = self.app_config.get("general", {}).get("verbose_llm_calls", self.verbose)

        infill_model_path: Optional[str] = None
        can_infill = False

        if infill_type == "gguf":
            infill_model_path = self.app_config.get("models", {}).get("infill_llm_gguf",
                                                                    self.app_config.get("models", {}).get("agent_llm_gguf",
                                                                                                          DEFAULT_APP_CONFIG["models"]["agent_llm_gguf"]))
            can_infill = get_llm_code_infill is not None and bool(infill_model_path)
            if not can_infill and self.verbose:
                if get_llm_code_infill is None: print("DiffusionCore Info: get_llm_code_infill (for GGUF) not available.")
                if not infill_model_path: print("DiffusionCore Info: GGUF infill model path not configured.")
        elif infill_type == "divot5":
            infill_model_path = self.app_config.get("models", {}).get("divot5_infill_model_dir", DEFAULT_APP_CONFIG["models"]["divot5_infill_model_dir"])
            can_infill = get_divot5_code_infill is not None and bool(infill_model_path)
            if not can_infill and self.verbose:
                if get_divot5_code_infill is None: print("DiffusionCore Info: get_divot5_code_infill not available.")
                if not infill_model_path: print("DiffusionCore Info: DivoT5 infill model path not configured.")
        else:
            print(f"DiffusionCore Warning: Unknown infill type '{infill_type}' configured. Skipping real infill.")

        if not can_infill and self.verbose:
            print("DiffusionCore Info: Real LLM infill will be skipped due to missing function or model path.")

        current_script_variant = scaffold_patch_script

        # Iterate reversed to handle potential length changes, though replacing unique markers is generally safe.
        # Using re.finditer to get match objects for start/end positions.
        hole_pattern = r"(__HOLE_(\d+)__)"
        matches = list(re.finditer(hole_pattern, current_script_variant))

        if not matches:
            print("DiffusionCore: No placeholders like __HOLE_n__ found in the script.")
            return current_script_variant

        for match in reversed(matches):
            hole_marker = match.group(1) # e.g., "__HOLE_0__"
            hole_number_str = match.group(2) # e.g., "0"

            code_before_hole = current_script_variant[:match.start()]
            code_after_hole = current_script_variant[match.end():]

            # Attempt to find relevant part of edit_summary for this specific hole
            # This is a heuristic; more advanced context mapping might be needed.
            summary_context_for_hole = f"Context for {hole_marker}: The overall goal is described by the summary: '{edit_summary_str}'. This specific hole, {hole_marker}, needs to be filled with Python code that logically completes the surrounding script."

            # Construct the prompt based on infill_type. For now, assume a generic FIM-like prompt structure.
            # DivoT5 might use specific tokens like <PREFIX>, <SUFFIX>, <MIDDLE>.
            # GGUF FIM models might expect a specific format too (e.g., CodeLlama FIM).
            # This part needs careful alignment with the chosen models.
            # For now, a simple text prompt:
            prompt = f"""You are an expert Python code completion assistant.
Your task is to fill in the placeholder `{hole_marker}` in the provided Python (LibCST) script.
The script is intended to perform a refactoring operation.
The overall edit summary for the script is: {edit_summary_str if edit_summary_str else "No summary provided."}
Context specific to this hole: {summary_context_for_hole}

Consider the code immediately before and after the placeholder to ensure logical consistency and correct syntax.
Provide *only* the Python code snippet that should replace `{hole_marker}`. Do not include the marker itself in your response.
Ensure the generated code is correctly indented relative to its position in the script if possible, though this might be adjusted later.

Code before placeholder:
```python
{code_before_hole[-500:]}
```

Placeholder: {hole_marker}

Code after placeholder:
```python
{code_after_hole[:500]}
```

Provide only the code snippet to replace `{hole_marker}`:"""

            infilled_code_snippet = None
            llm_call_succeeded = False

            if can_infill and infill_model_path:
                if infill_type == "gguf":
                    gguf_n_gpu_layers = llm_params.get("n_gpu_layers_default", DEFAULT_APP_CONFIG["llm_params"]["n_gpu_layers_default"])
                    gguf_n_ctx = llm_params.get("n_ctx_default", DEFAULT_APP_CONFIG["llm_params"]["n_ctx_default"])
                    gguf_max_tokens = llm_params.get("agent_infill_gguf_max_tokens", DEFAULT_APP_CONFIG["llm_params"]["agent_infill_gguf_max_tokens"])
                    gguf_temp = llm_params.get("agent_infill_gguf_temp", DEFAULT_APP_CONFIG["llm_params"]["temperature_default"])
                    # stop_sequences might need to be configured in llm_params too
                    stop_sequences = llm_params.get("agent_infill_gguf_stop_sequences", ["\n```", "\n# End of infill"])

                    if get_llm_code_infill:
                        infilled_code_snippet = get_llm_code_infill(
                            model_path=infill_model_path,
                            prompt=prompt, # This prompt might need adjustment for GGUF FIM models
                            verbose=llm_interfacer_verbose,
                            n_gpu_layers=gguf_n_gpu_layers,
                            n_ctx=gguf_n_ctx,
                            max_tokens_for_infill=gguf_max_tokens,
                            temperature=gguf_temp,
                            stop=stop_sequences
                        )
                        llm_call_succeeded = infilled_code_snippet is not None

                elif infill_type == "divot5":
                    # DivoT5 prompt might need to be structured with <PREFIX>, <SUFFIX>, <MIDDLE>
                    # For now, using the same generic prompt; this will likely need refinement.
                    # Example DivoT5 prompt structure: f"<PREFIX>{code_before_hole[-500:]}<SUFFIX>{code_after_hole[:500]}<MIDDLE>"
                    divot5_prompt = f"<PREFIX>{code_before_hole[-500:]}<SUFFIX>{code_after_hole[:500]}<MIDDLE>" # Example specific prompt

                    divot5_max_length = divot5_params.get("infill_max_length", DEFAULT_APP_CONFIG["divot5_fim_params"]["infill_max_length"])
                    divot5_num_beams = divot5_params.get("infill_num_beams", DEFAULT_APP_CONFIG["divot5_fim_params"]["infill_num_beams"])
                    divot5_temp = divot5_params.get("infill_temperature", DEFAULT_APP_CONFIG["divot5_fim_params"]["infill_temperature"])

                    if get_divot5_code_infill:
                        infilled_code_snippet = get_divot5_code_infill(
                            model_dir_path=infill_model_path, # DivoT5 usually loaded from a directory
                            prompt=divot5_prompt, # Use the DivoT5 specific prompt
                            max_length=divot5_max_length,
                            num_beams=divot5_num_beams,
                            temperature=divot5_temp,
                            verbose=llm_interfacer_verbose
                        )
                        llm_call_succeeded = infilled_code_snippet is not None

            if llm_call_succeeded and isinstance(infilled_code_snippet, str) and infilled_code_snippet.strip():
                # TODO: Add intelligent indentation adjustment for infilled_code_snippet.
                #       This might involve:
                #       1. Determining the indentation of the line containing the hole_marker.
                #       2. Ensuring the first line of infilled_code_snippet matches this.
                #       3. Adjusting subsequent lines of infilled_code_snippet relative to its first line.
                #       For now, we rely on the LLM to provide reasonably indented code or manual post-processing.
                # A simple heuristic:
                lines_before = code_before_hole.splitlines()
                original_indent_str = ""
                if lines_before:
                    last_line_before = lines_before[-1]
                    match_indent = re.match(r"^(\s*)", last_line_before)
                    if match_indent:
                        # If the hole is on a new line, this indent is likely correct.
                        # If the hole is mid-line, this needs more sophistication.
                        # Assuming hole is usually on its own indented line for now.
                        original_indent_str = match_indent.group(1)

                # Prepend indent to each line of the snippet if not already indented similarly
                adjusted_snippet_lines = []
                for i, line in enumerate(infilled_code_snippet.splitlines()):
                    if i == 0 and not line.startswith(original_indent_str) and line.strip(): # First line, ensure it has the base indent
                         adjusted_snippet_lines.append(original_indent_str + line)
                    elif i > 0 and line.strip(): # Subsequent lines, ensure they also have at least base indent if they are not empty
                         adjusted_snippet_lines.append(original_indent_str + line) # This simple version might over-indent if snippet is already multi-line indented
                    else:
                         adjusted_snippet_lines.append(line) # Keep empty lines as they are

                # This indentation logic is very basic. A more robust solution would parse the snippet
                # and adjust its internal relative indentation to match the insertion point's base indent.
                # For now, we'll use this simplified approach. A better approach might be to pass indentation
                # instructions to the LLM or use a code formatter after infilling.
                # For now, let's just join what we have, it's better than nothing.
                # The hole normally implies the LLM should provide content starting at that indent level.

                current_script_variant = code_before_hole + infilled_code_snippet + code_after_hole
                print(f"DiffusionCore: Successfully filled {hole_marker} with LLM output (length {len(infilled_code_snippet)}).")
            else:
                failure_reason = "LLM returned no content"
                if not llm_infill_available: failure_reason = "LLM infill function not available"
                elif not infill_model_path: failure_reason = "Infill model path not configured"
                elif not llm_call_succeeded : failure_reason = "LLM call failed or returned None"

                replacement_comment = f"# FAILED_TO_FILL_HOLE_{hole_number_str} - Reason: {failure_reason}"
                current_script_variant = code_before_hole + replacement_comment + code_after_hole
                print(f"DiffusionCore: Failed to fill {hole_marker}. Replaced with comment: '{replacement_comment}'. Reason: {failure_reason}")

        completed_patch_script = current_script_variant
        print(f"DiffusionCore: Finished expanding scaffold. Final script (first 200 chars): '{completed_patch_script[:200]}...'")
        return completed_patch_script

    def re_denoise_spans(self,
                         failed_patch_script: Optional[str],
                         proposed_fix_script: Optional[str],
                         context_data: Dict[str, Any]
                         ) -> Optional[str]:
        """
        Currently, this method passes through the proposed_fix_script from LLMCore.
        Future enhancements could involve more sophisticated merging or diffusion-based refinement.

        Args:
            failed_patch_script: The LibCST script that was polished and then failed validation.
            proposed_fix_script: The new script suggested by LLMCore.propose_repair_diff.
            context_data: Comprehensive context data dictionary.
        Returns:
            The script to be used for the next repair attempt (currently, proposed_fix_script), or None.
        """
        if self.verbose:
            failed_script_preview = f"(length: {len(failed_patch_script)}) {failed_patch_script[:100]}..." if failed_patch_script else "None"
            proposed_script_preview = f"(length: {len(proposed_fix_script)}) {proposed_fix_script[:100]}..." if proposed_fix_script else "None"
            print(f"DiffusionCore.re_denoise_spans called. Context keys: {list(context_data.keys())}.")
            print(f"  Verbose: Failed patch script (preview): {failed_script_preview}")
            print(f"  Verbose: Proposed fix script (preview): {proposed_script_preview}")
        else:
            print(f"DiffusionCore.re_denoise_spans called.")

        # This method acts as a pass-through for the script proposed by LLMCore.
        # The primary role of LLMCore.propose_repair_diff is to provide a complete, revised script.
        # DiffusionCore here confirms that proposal and passes it on.
        # Future enhancements for more complex merging/denoising are out of scope for this refinement.

        if proposed_fix_script is None:
            print("DiffusionCore: No proposed fix script received from LLMCore. Passing through None.")
            return None
        else:
            # Ensure proposed_fix_script is a string, though type hints suggest it should be.
            if not isinstance(proposed_fix_script, str):
                print(f"DiffusionCore Warning: proposed_fix_script is not a string (type: {type(proposed_fix_script)}). Attempting to cast. This is unexpected.")
                try:
                    proposed_fix_script = str(proposed_fix_script)
                except Exception as e_cast:
                    print(f"DiffusionCore Error: Failed to cast proposed_fix_script to string: {e_cast}. Returning None.")
                    return None

            print(f"DiffusionCore: Passing through proposed fix script (length: {len(proposed_fix_script)}) from LLMCore as the 're-denoised' script.")
            return proposed_fix_script
