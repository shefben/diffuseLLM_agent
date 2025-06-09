from typing import Any, Dict, List, Optional
import re

# Attempt to import get_llm_code_infill, handle if not available
try:
    from src.profiler.llm_interfacer import get_llm_code_infill
except ImportError:
    get_llm_code_infill = None
    print("DiffusionCore Warning: Failed to import get_llm_code_infill from src.profiler.llm_interfacer. LLM-based infilling will be disabled.")


class DiffusionCore:
    def __init__(self, style_profile: Dict[str, Any], config: Optional[Dict[str, Any]] = None):
        """
        Initializes the DiffusionCore.
        Args:
            style_profile: Dictionary containing style profile information.
            config: Optional dictionary for Diffusion core specific configurations.
                    Expected keys for expand_scaffold:
                    - "infill_model_path" (Optional[str]): Path to GGUF model for infilling.
                    - "llm_model_path" (Optional[str]): Fallback path if infill_model_path is not set.
                    - "verbose" (bool): LLM verbosity.
                    - "n_gpu_layers" (int): GPU layers for LLM.
                    - "n_ctx" (int): Context size for LLM.
                    - "max_tokens_for_infill" (int): Max tokens for generated infill.
                    - "temperature" (float): Sampling temperature for LLM.
                    - "stop_sequences_for_infill" (Optional[List[str]]): Stop sequences for infill.
        """
        self.style_profile = style_profile
        self.config = config if config else {}
        # Ensure essential config keys have defaults if not provided, or handle their absence in methods.
        # For this refactor, we assume expand_scaffold will handle missing keys gracefully.
        print(f"DiffusionCore initialized with style_profile, config: {self.config}")

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
            print("DiffusionCore Warning: Received None or empty scaffold_patch_script. Returning as is.")
            return scaffold_patch_script

        print(f"  Initial scaffold script (first 200 chars): '{scaffold_patch_script[:200]}...'")

        llm_infill_available = get_llm_code_infill is not None
        infill_model_path = self.config.get("infill_model_path", self.config.get("llm_model_path"))

        if not llm_infill_available:
            print("DiffusionCore Warning: get_llm_code_infill function is not available (likely missing LlamaCPP in llm_interfacer). Real LLM infill will be skipped.")
        if not infill_model_path:
            print("DiffusionCore Warning: No model path configured for infilling ('infill_model_path' or 'llm_model_path' in config). Real LLM infill will be skipped.")

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
            # Example: if edit_summary_str is a list of strings, one per hole:
            # try:
            #    summary_context_for_hole = edit_summary_list[int(hole_number_str)]
            # except (IndexError, ValueError, TypeError): # Handle cases where summary is not a list or index is bad
            #    pass # Stick with default summary_context_for_hole

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

            if llm_infill_available and infill_model_path:
                # Get LLM parameters from config with defaults
                llm_verbose = self.config.get("verbose", False)
                n_gpu_layers = self.config.get("n_gpu_layers", -1) # Default to -1 (all layers if possible)
                n_ctx = self.config.get("n_ctx", 4096) # Default context size
                max_tokens = self.config.get("max_tokens_for_infill", 512)
                temperature = self.config.get("temperature", 0.4)
                stop_sequences = self.config.get("stop_sequences_for_infill", ["\n```", "\n# End of infill"]) # Example stop sequences

                if get_llm_code_infill: # Final check on the function pointer
                    infilled_code_snippet = get_llm_code_infill(
                        model_path=infill_model_path,
                        prompt=prompt,
                        verbose=llm_verbose,
                        n_gpu_layers=n_gpu_layers,
                        n_ctx=n_ctx,
                        max_tokens_for_infill=max_tokens,
                        temperature=temperature,
                        stop=stop_sequences
                    )
                    llm_call_succeeded = infilled_code_snippet is not None # True if LLM returned something (even empty string)

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
            The script to be used for the next repair attempt (currently, proposed_fix_script).
        """
        print(f"DiffusionCore.re_denoise_spans called. Context keys: {list(context_data.keys())}.")
        # print(f"  Failed patch script (start): '{failed_patch_script[:100] if failed_patch_script else 'None'}...'")
        # print(f"  Proposed fix script (start): '{proposed_fix_script[:100] if proposed_fix_script else 'None'}...'")

        # Current pass-through implementation:
        # The LLMCore.propose_repair_diff (via get_llm_code_fix_suggestion) is already tasked
        # with returning a complete, revised script. So, re_denoise_spans will just pass this through.
        # A more complex future version might:
        # 1. Identify specific "spans" or "holes" in the original failed_patch_script that correspond
        #    to the error described in the traceback (from context_data if needed).
        # 2. Use the proposed_fix_script as a strong hint or guide.
        # 3. Apply a diffusion model or another LLM to "in-fill" or "re-denoise" only those spans,
        #    potentially merging the LLM's broad fix with the more structured parts of failed_patch_script.
        # For now, the "repair" is wholesale replacement by the LLM's suggested script.

        if proposed_fix_script is not None:
            print(f"DiffusionCore: Passing through proposed fix script from LLMCore. Length: {len(proposed_fix_script)}")
            return proposed_fix_script
        else:
            print("DiffusionCore: No proposed fix script from LLMCore. Returning the original failed script for retry if applicable.")
            # This implies LLMCore couldn't suggest a fix. Returning the failed script might lead to a loop
            # if not handled carefully in CollaborativeAgentGroup (e.g., by max_repair_attempts).
            # However, CollaborativeAgentGroup's logic already handles None from propose_repair_diff by breaking the loop.
            # So, this path (returning failed_patch_script if proposed_fix_script is None) might be less common.
            # It's safer to return None if proposed_fix_script is None, aligning with LLMCore's inability to fix.
            return None
        return patch_with_failed_validation
