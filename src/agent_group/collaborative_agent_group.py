from typing import TYPE_CHECKING, Any, Dict, Tuple, Callable, Optional
from pathlib import Path

# Local imports
from .llm_core import LLMCore
from .diffusion_core import DiffusionCore

if TYPE_CHECKING:
    from src.planner.phase_model import Phase
    from src.digester.repository_digester import RepositoryDigester
    # Define Path from pathlib for type hinting if not already globally available
    # from pathlib import Path

class CollaborativeAgentGroup:
    def __init__(self,
                 style_profile: Dict[str, Any],
                 naming_conventions_db_path: Path,
                 llm_core_config: Optional[Dict[str, Any]] = None,
                 diffusion_core_config: Optional[Dict[str, Any]] = None
                 ):
        """
        Initializes the CollaborativeAgentGroup.
        Args:
            style_profile: Dictionary containing style profile information.
            naming_conventions_db_path: Path to the naming conventions database.
            llm_core_config: Optional configuration for LLMCore.
            diffusion_core_config: Optional configuration for DiffusionCore.
        """
        self.style_profile = style_profile
        self.naming_conventions_db_path = naming_conventions_db_path

        # Initialize Core Agents
        self.llm_agent = LLMCore(
            style_profile=self.style_profile,
            naming_conventions_db_path=self.naming_conventions_db_path,
            config=llm_core_config
        )
        self.diffusion_agent = DiffusionCore(
            style_profile=self.style_profile,
            config=diffusion_core_config
        )

        self.current_patch_candidate: Optional[Any] = None
        self.patch_history: list = [] # To store (patch, validation_result, score) tuples

        print("CollaborativeAgentGroup initialized with LLMCore and DiffusionCore.")

    def _perform_duplicate_guard(self, patch_script_str: Optional[str], context_data: Dict[str, Any]) -> bool:
        """
        Checks if the proposed patch script might be creating a duplicate function/method.
        This is a mock implementation.
        """
        if not patch_script_str:
            return False

        print(f"CollaborativeAgentGroup._perform_duplicate_guard called for script (first 100 chars): '{patch_script_str[:100]}...'")

        digester = context_data.get("repository_digester")
        if not digester or not hasattr(digester, 'signature_trie'):
            print("  DuplicateGuard: Digester or SignatureTrie not available in context. Skipping check.")
            return False

        # Mock duplicate detection logic:
        # Try to find if the script defines a function named "new_mock_function"
        # This is highly simplified. A real implementation would parse the script,
        # extract defined function/method signatures, and check them.
        if "def new_mock_function(" in patch_script_str or "function_name=\"new_mock_function\"" in patch_script_str :
            # This specific signature is hardcoded in SignatureTrie.search to return True for testing
            mock_function_signature = "new_mock_function(arg1: int) -> str"
            print(f"  DuplicateGuard: Mocking check for signature: '{mock_function_signature}'")

            try:
                # The SignatureTrie.search method was updated to return bool
                is_duplicate = digester.signature_trie.search(mock_function_signature)
                print(f"  DuplicateGuard: SignatureTrie search result for '{mock_function_signature}': {is_duplicate}")
                if is_duplicate:
                    print("  DuplicateGuard: Mock duplicate signature detected!")
                    return True
            except Exception as e:
                print(f"  DuplicateGuard: Error during SignatureTrie search: {e}")
                return False # Fail safe

        print("  DuplicateGuard: No duplicate detected by mock logic.")
        return False

    def run(
        self,
        phase_ctx: 'Phase',
        digester: 'RepositoryDigester',
        validator_handle: Callable[[Any, 'RepositoryDigester', 'Phase'], Tuple[bool, Any, Optional[str]]],
        score_style_handle: Callable[[Any, Dict[str, Any]], float]
    ) -> Optional[Any]:
        # Initialize variables to store outputs from agent calls
        scaffold_patch_script: Optional[str] = None
        edit_summary_list: Optional[List[str]] = None
        completed_patch_script: Optional[str] = None # For output of diffusion pass
        # self.current_patch_candidate is initialized in __init__ and will hold the evolving script/patch

        """
        Main execution loop for the agent group to generate and refine a patch.
        Args:
            phase_ctx: The current phase context.
            digester: The repository digester instance.
            validator_handle: A callable for validating the patch.
                              Expected signature: (patch, digester, phase_ctx) -> (is_valid, validation_payload, error_traceback_or_None)
            score_style_handle: A callable for scoring the patch against style.
                                Expected signature: (patch, style_profile) -> score (float)
                                (Note: style_profile is accessed from self.style_profile within this method)
        Returns:
            The final validated and scored patch, or None if no suitable patch could be generated.
        """
        print(f"CollaborativeAgentGroup.run called for phase: {phase_ctx.operation_name} on {phase_ctx.target_file}")
        print(f"CollaborativeAgentGroup using internal style_profile (keys: {list(self.style_profile.keys())}) and naming_conventions_db: {self.naming_conventions_db_path}")

        # --- Phase 5: Step 1: Context Broadcast ---
        # Initialize and populate comprehensive context_data
        context_data: Dict[str, Any] = {
            "phase_description": getattr(phase_ctx, 'description', 'N/A'),
            "phase_target_file": getattr(phase_ctx, 'target_file', None),
            "phase_parameters": getattr(phase_ctx, 'parameters', {}),
            "style_profile": self.style_profile,
            "naming_conventions_db_path": self.naming_conventions_db_path,
            "score_style_handle": score_style_handle, # Passed from PhasePlanner
            "validator_handle": validator_handle,     # Passed from PhasePlanner
            "repository_digester": digester           # Pass the digester instance
        }

        # Add PDG slice and code snippets from digester
        # These calls assume digester methods handle potential errors or missing data gracefully.
        context_data["retrieved_code_snippets"] = digester.get_code_snippets_for_phase(phase_ctx)
        context_data["pdg_slice"] = digester.get_pdg_slice_for_phase(phase_ctx)

        print(f"Step 1: Context Broadcast prepared. Context keys: {list(context_data.keys())}")
        if context_data["retrieved_code_snippets"]:
             print(f"  Retrieved snippets for: {list(context_data['retrieved_code_snippets'].keys())}")
        if context_data["pdg_slice"]:
             print(f"  Retrieved PDG slice info: {context_data['pdg_slice'].get('info', 'N/A')}")


        # --- LLM Core generates scaffold patch (was previously Step 1, now after full context prep) ---
        # Note: The first argument to generate_scaffold_patch was phase_ctx.
        # We can pass phase_ctx itself, or pass relevant parts from context_data if preferred.
        # For now, passing phase_ctx directly as per its original design, plus the full context_data.
        # LLMCore.generate_scaffold_patch now only takes context_data.
        scaffold_patch_script, edit_summary_list = self.llm_agent.generate_scaffold_patch(context_data)

        if scaffold_patch_script is None or edit_summary_list is None:
            print("LLMCore failed to generate an initial scaffold script or edit summary. Aborting.")
            return None

        print(f"Step 2: LLM scaffolding pass complete. Received script (len: {len(scaffold_patch_script)}) and summary (items: {len(edit_summary_list)}).")
        print(f"   Edit Summary: {edit_summary_list}")
        # print(f"   Scaffold Script:\n{scaffold_patch_script}") # Potentially very long

        # --- Phase 5: Step 2 (continued, now Step 3 in Phase 5 doc): Diffusion Core expands scaffold ---
        # Ensure edit_summary_list is passed correctly (e.g. as a string or processed if needed by expand_scaffold)
        # DiffusionCore.expand_scaffold expects edit_summary as str. Let's join the list.
        diffusion_edit_summary_str = "; ".join(edit_summary_list) if edit_summary_list else ""

        # Call expand_scaffold
        completed_patch_script = self.diffusion_agent.expand_scaffold(scaffold_patch_script, diffusion_edit_summary_str, context_data)

        if completed_patch_script is None:
            print("DiffusionCore failed to expand/complete the scaffold script. Aborting.")
            return None
        print(f"Step 3: Diffusion expansion pass complete. Completed script (len: {len(completed_patch_script)}).")
        # print(f"   Completed Script:\n{completed_patch_script}") # Potentially very long

        # At this point, completed_patch_script is a string (the LibCST script).
        # For the iterative refinement loop, self.current_patch_candidate needs to be
        # the actual patch object/data structure that the validator_handle expects.
        # For now, the placeholder validator_handle in PhasePlanner might accept this string,
        # or we assume a (not-yet-implemented) step here to "execute" or "interpret" this script
        # to produce a patch object.
        # Let's assume for this subtask, current_patch_candidate will store the script string.
        # This will need adjustment in Phase 5 Step 3 (Validate Patch).
        self.current_patch_candidate = completed_patch_script

        # --- Iterative Refinement Loop (Simplified Placeholder) ---
        max_iterations = 3
        for i in range(max_iterations):
            print(f"\n--- Iteration {i + 1} ---")

            # --- Step 5 (Part A): Duplicate Guard ---
            # self.current_patch_candidate holds the script string from diffusion or previous repair
            is_duplicate_detected = self._perform_duplicate_guard(self.current_patch_candidate, context_data)

            # Initialize validation results for the iteration
            is_valid = True
            error_traceback: Optional[str] = None
            validation_payload: Any = None

            if is_duplicate_detected:
                print("CollaborativeAgentGroup: Duplicate detected by guard! Attempting repair to reuse existing functionality.")
                is_valid = False # Mark as not valid to trigger repair logic
                error_traceback = "DUPLICATE_DETECTED: REUSE_EXISTING_HELPER"
                # Skip direct validation if duplicate is found, proceed to repair logic.
                # The style score might not be relevant here, or could be set to a low value.
                style_score = 0.0 # Or skip scoring for duplicates
                self.patch_history.append((self.current_patch_candidate, is_valid, style_score, error_traceback))
            else:
                # --- Step 5 (Part B): Validate Patch (if not a duplicate) ---
                # The validator_handle currently in PhasePlanner is a mock that takes (patch, digester, phase_ctx)
                # The 'patch' it receives will be the completed_patch_script string.
                # This mock validator needs to be aware it might receive a script string.
                is_valid, validation_payload, error_traceback = validator_handle(
                    self.current_patch_candidate, # This is the script string
                    digester,
                    phase_ctx
                )
                print(f"Loop Step A (Validation): Validation result for script: is_valid={is_valid}, payload={validation_payload}, error='{bool(error_traceback)}'")

            # --- Phase 5: Step 4 (Score Patch Style - was Step 4, now part of loop) ---
            # This step is now after duplicate check and potential direct validation skip.
            # If duplicate, style_score was set to 0.0. Otherwise, calculate it.
            if not is_duplicate_detected:
                style_score = score_style_handle(
                    self.current_patch_candidate, # This is the script string
                    self.style_profile
                )
                print(f"Loop Step B (Style Score): Style score for script: {style_score:.2f}")
                self.patch_history.append((self.current_patch_candidate, is_valid, style_score, error_traceback))

            # Decision logic based on validation (and duplicate check)
            if is_valid: # This means not a duplicate AND validator_handle returned True
                print(f"Script validated successfully in iteration {i+1}. Proceeding to polish.")
                # --- Phase 5: Step 4 (LLM Polishing Pass - was Step 5 in old numbering) ---
            # The score_style_handle also takes the patch (script string here)
            style_score = score_style_handle(
                self.current_patch_candidate, # This is the script string
                self.style_profile
            )
            print(f"Loop Step B (Style Score): Style score for script: {style_score:.2f}")

            self.patch_history.append((self.current_patch_candidate, is_valid, style_score, error_traceback))

            if is_valid:
                print(f"Script validated successfully in iteration {i+1}. Proceeding to polish.")
                # --- Phase 5: Step 4 (LLM Polishing Pass - was Step 5 in old numbering) ---
                polished_script = self.llm_agent.polish_patch(self.current_patch_candidate, context_data)
                print(f"Loop Step C (Polish): LLMCore polished the script. New length: {len(polished_script) if polished_script else 'N/A'}.")
                self.current_patch_candidate = polished_script # Update current candidate to the polished script

                # Re-validate and re-score after polish
                # Ensure current_patch_candidate is not None before validating
                if self.current_patch_candidate is None:
                    print("Polishing returned None. Cannot proceed with validation. Aborting iteration.")
                    error_traceback = "Polishing step resulted in a None script."
                    is_valid = False # Mark as not valid to trigger repair or end loop
                    # Skip directly to repair logic or end of loop processing
                else:
                    final_is_valid, validation_payload, final_error_traceback = validator_handle(self.current_patch_candidate, digester, phase_ctx)
                    final_style_score = score_style_handle(self.current_patch_candidate, self.style_profile)
                    print(f"Loop Step C.1 (Post-Polish Validation): Valid: {final_is_valid}, Style: {final_style_score:.2f}")

                    # Update history with the polished attempt
                    self.patch_history.append((self.current_patch_candidate, final_is_valid, final_style_score, final_error_traceback))
                    is_valid = final_is_valid # Update is_valid based on post-polish validation
                    error_traceback = final_error_traceback if final_error_traceback else error_traceback # Persist new error if any

                    if final_is_valid:
                        print(f"Polished script is valid. Final style score: {final_style_score:.2f}. This script is the result of this phase.")
                        return self.current_patch_candidate
                    else:
                        print("Polished script failed validation.")
                        # error_traceback should be set from final_error_traceback for the repair step
                        # Fall through to repair logic for the polished_but_failed_script

            # If not valid after initial check, or if polish made it invalid / returned None:
            if not is_valid: # Combined check
                print(f"Script requires repair (or loop exit) after iteration {i+1} (is_valid={is_valid}). Error: {error_traceback}")
                if error_traceback:
                    # --- Phase 5: Step 5 (LLM Proposes Repair - was Step 6a) ---
                print("Loop Step D (Attempt Repair): Attempting repair with LLMCore.")
                # propose_repair_diff takes traceback and context_data.
                # The current_patch_candidate at this point is the one that failed (either initial or polished).
                repair_diff_script_or_new_script = self.llm_agent.propose_repair_diff(error_traceback, context_data)
                if repair_diff_script_or_new_script:
                    print(f"LLMCore proposed a repair/new script.")
                    self.current_patch_candidate = repair_diff_script_or_new_script # This becomes the candidate for the next iteration
                else:
                    print("LLMCore could not propose a repair. Trying DiffusionCore re-denoise (if applicable).")
                    # --- Phase 5: Step 6b (Diffusion Re-Denoises - was Step 6b) ---
                    # re_denoise_spans expects the patch that failed and affected_spans.
                    # This part is tricky as 'affected_spans' needs to come from validation_payload.
                    # And the patch is a script string.
                    affected_spans = validation_payload.get("affected_spans", []) if isinstance(validation_payload, dict) else []
                    if affected_spans: # Only call if there are specific spans
                         print("Loop Step E (Re-Denoise): Attempting re-denoise with DiffusionCore.")
                         self.current_patch_candidate = self.diffusion_agent.re_denoise_spans(self.current_patch_candidate, affected_spans, context_data)
                    else:
                        print("No specific affected spans for DiffusionCore re-denoise, or LLM repair failed. Aborting iteration if no progress.")
                        # If no repair from LLM and no specific spans for Diffusion, we might be stuck.
                        # End the loop or try a more general re-denoise if that's an option.
                        break
            else: # No error_traceback from validation (e.g. style score too low but otherwise valid)
                print("Loop Step D/E (No Repair Path): Validation failed without a traceback (e.g. style issue), or no repair proposed. No further repair attempts in this iteration.")
                # If style is the only issue, and we don't have a specific re-denoise for style, we might break.
                # Or, if a previous version was good enough, that might be selected later.
                break # Break if no clear path to repair/improve

            if i == max_iterations - 1:
                print("Max iterations reached. Could not produce a valid and satisfactory script.")
                # Fallback: return the best script from history based on validity and score.
                best_historical_script = None
                best_historical_score = -1.0
                was_valid_in_history = False
                for p_hist, v_hist, s_hist, _tb_hist in self.patch_history:
                    # Prioritize valid scripts, then highest score.
                    if v_hist and s_hist > best_historical_score:
                        best_historical_script = p_hist
                        best_historical_score = s_hist
                        was_valid_in_history = True
                    elif not was_valid_in_history and s_hist > best_historical_score: # If no valid ones yet, take best score overall
                        best_historical_script = p_hist
                        best_historical_score = s_hist

                if best_historical_script:
                    print(f"Returning best script from history (Valid: {was_valid_in_history}, Score: {best_historical_score:.2f}).")
                    return best_historical_script
                return None # No suitable script found after all iterations

        print("Exited refinement loop. No suitable script generated.")
        return None # Should ideally return the best effort from history if loop finishes. Covered by above.


    def generate_patch_preview(self) -> str:
        """Placeholder: Generates a preview of the current patch candidate."""
        if self.current_patch_candidate:
            # In a real scenario, this would format the patch (e.g., as a diff)
            return f"Preview of current patch candidate: {str(self.current_patch_candidate)[:200]}..."
        return "Patch preview not yet implemented. No current patch candidate."

    def abort_and_rollback(self) -> None:
        """Placeholder: Aborts the current operation and rolls back any changes."""
        print("CollaborativeAgentGroup.abort_and_rollback called. Rolling back (placeholder).")
        self.current_patch_candidate = None
        self.patch_history = []
        # Actual rollback logic would depend on how changes are staged/applied.
        pass

# Example Usage (Conceptual - requires mock objects for Phase, Digester etc.)
if __name__ == '__main__':
    print("\n--- CollaborativeAgentGroup Example Usage (Conceptual) ---")

    # Mock Phase (replace with actual Phase import and instantiation if available)
    class MockPhase:
        def __init__(self, op_name, target, params=None):
            self.operation_name = op_name
            self.target_file = target
            self.parameters = params if params else {}

    # Mock Digester (replace with actual Digester import and instantiation)
    class MockDigester:
        def get_project_overview(self): return {"files": 2, "language": "python"}
        def get_file_content(self, path: Path): return f"# Content of {path}\npass" if path else None

    # Mock validator and scorer
    def mock_validator(patch, digester, phase_ctx):
        print(f"MockValidator: Validating patch: {patch}")
        if patch.get("value") and "ERROR" in patch["value"]:
            return False, {"reason": "Contains ERROR string"}, "Traceback: Something went wrong due to ERROR"
        if patch.get("path") == "/src/invalid.py":
            return False, {"reason": "Invalid path"}, "Traceback: InvalidPathError"
        return True, {"lines_changed": 1}, None

    def mock_score_style(patch, style_profile):
        print(f"MockScoreStyle: Scoring patch: {patch} with profile: {style_profile}")
        if patch.get("value") and "bad_style" in patch["value"]:
            return 0.2
        return 0.85

    # Setup
    mock_llm_config = {"other_common_context": {"api_key": "llm_mock_key"}}
    mock_diffusion_config = {"other_common_context": {"model_strength": 0.7}}
    mock_style_profile = {"line_length": 88, "indent_style": "space"}
    mock_naming_db_path = Path("mock_naming_db.json")

    # Create dummy naming db file
    with open(mock_naming_db_path, "w") as f:
        f.write('{"function_prefix": "get_"}')

    agent_group = CollaborativeAgentGroup(
        llm_config=mock_llm_config,
        diffusion_config=mock_diffusion_config,
        style_profile=mock_style_profile,
        naming_conventions_db_path=mock_naming_db_path
    )

    mock_phase_ctx = MockPhase(op_name="add_function", target="src/example.py", params={"function_name": "my_func"})
    mock_digester_instance = MockDigester()

    print("\nStarting agent_group.run...")
    final_patch_result = agent_group.run(
        phase_ctx=mock_phase_ctx,
        digester=mock_digester_instance,
        validator_handle=mock_validator,
        score_style_handle=mock_score_style
    )

    print(f"\nAgent group run completed. Final patch: {final_patch_result}")
    print(f"Patch preview: {agent_group.generate_patch_preview()}")

    # Example of a patch that might fail validation then get repaired (conceptually)
    print("\n--- Example with failing patch ---")
    # LLMCore's generate_scaffold_patch would need to be influenced to produce this,
    # or we'd need to mock the internal agents more deeply.
    # For this example, assume the flow leads to a patch that fails.
    # We can simulate this by how mock_validator works with "ERROR" in value.

    # To test repair, we'd need to modify LLMCore/DiffusionCore mocks or have them produce failing patches.
    # The current placeholder LLMCore.generate_scaffold_patch produces a valid-looking patch.
    # Let's imagine a scenario where a patch initially contains "ERROR"
    # This would require deeper mocking of the internal agent calls or specific test setup.
    # For now, the existing run shows the loop. A more specific test for repair:

    # agent_group.llm_agent.generate_scaffold_patch = lambda pc, cd: ({"op": "add", "path": "/src/example.py", "value": "def new_function():\n    ERROR\n    pass"}, "Scaffold with error")
    # print("\nRestarting agent_group.run with a patch designed to fail then repair...")
    # final_patch_fail_repair = agent_group.run( mock_phase_ctx, mock_digester_instance, mock_validator, mock_score_style)
    # print(f"\nAgent group run (fail-repair scenario) completed. Final patch: {final_patch_fail_repair}")


    agent_group.abort_and_rollback()

    # Cleanup dummy file
    if mock_naming_db_path.exists():
        mock_naming_db_path.unlink()

    print("\n--- CollaborativeAgentGroup Example Done ---")
