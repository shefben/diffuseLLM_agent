from typing import TYPE_CHECKING, Any, Dict, Tuple, Callable, Optional, List # Added List
from pathlib import Path

# Local imports
from .exceptions import PhaseFailure # New import
from .llm_core import LLMCore
from .diffusion_core import DiffusionCore

if TYPE_CHECKING:
    from src.planner.phase_model import Phase
    from src.digester.repository_digester import RepositoryDigester
    from src.validator.validator import Validator # For type hinting
    # Define Path from pathlib for type hinting if not already globally available
    # from pathlib import Path

class CollaborativeAgentGroup:
    def __init__(self,
                 app_config: Dict[str, Any],
                 digester: 'RepositoryDigester', # Added digester
                 style_profile: Dict[str, Any],
                 naming_conventions_db_path: Path,
                 validator_instance: 'Validator'
                 ):
        """
        Initializes the CollaborativeAgentGroup.
        Args:
            app_config: The main application configuration dictionary.
            digester: An instance of RepositoryDigester.
            style_profile: Dictionary containing style profile information.
            naming_conventions_db_path: Path to the naming conventions database.
            validator_instance: An instance of the Validator class.
        """
        self.app_config = app_config
        self.digester = digester # Store digester instance, will also be passed in run()
        self.style_profile = style_profile
        self.naming_conventions_db_path = naming_conventions_db_path
        self.validator = validator_instance
        self.max_repair_attempts = self.app_config.get("agent_group", {}).get("max_repair_attempts", 3)
        self.verbose = self.app_config.get("general", {}).get("verbose", False)

        # Initialize Core Agents using app_config
        self.llm_agent = LLMCore(
            app_config=self.app_config,
            style_profile=self.style_profile,
            naming_conventions_db_path=self.naming_conventions_db_path
        )
        # Assuming DiffusionCore will also be refactored to take app_config
        # For now, its existing signature might be config and style_profile.
        # We'll pass app_config as its 'config' for now.
        self.diffusion_agent = DiffusionCore(
            # app_config=self.app_config, # If DiffusionCore is updated
            config=self.app_config, # Passing app_config as the 'config' dict
            style_profile=self.style_profile
        )

        self.current_patch_candidate: Optional[Any] = None
        self.patch_history: list = [] # To store (script, validation_result, score, error) tuples
        self._current_run_context_data: Optional[Dict[str, Any]] = None # For preview context

        if self.verbose:
            print(f"CollaborativeAgentGroup initialized. Max repair attempts: {self.max_repair_attempts}")

            app_config=self.app_config, # If DiffusionCore is updated
            style_profile=self.style_profile
        )
        self.style_validator_agent = StyleValidatorCore(app_config=self.app_config, style_profile=self.style_profile)

        self.current_patch_candidate: Optional[Any] = None
        self.patch_history: list = [] # To store (script, validation_result, score, error) tuples
        self._current_run_context_data: Optional[Dict[str, Any]] = None # For preview context

        print("CollaborativeAgentGroup initialized with LLMCore, DiffusionCore, and max_repair_attempts.")

    def _perform_duplicate_guard(self, modified_code_str: Optional[str], target_file_path: Path, context_data: Dict[str, Any]) -> bool:
        """
        Checks if the modified code string contains any function/method signatures
        that already exist in the project's signature trie.
        """
        if not modified_code_str:
            print("DuplicateGuard: No modified code string provided. Skipping check.")
            return False

        digester = context_data.get("repository_digester")
        if not digester or not hasattr(digester, 'signature_trie') or not hasattr(digester, 'repo_path'):
            print("DuplicateGuard: Digester, signature_trie, or repo_path not available in context. Skipping check.")
            return False

        print(f"DuplicateGuard: Checking for duplicate signatures in modified code for target: {target_file_path.name}")

        try:
            module_ast = ast.parse(modified_code_str, filename=str(target_file_path))
        except SyntaxError as e:
            print(f"DuplicateGuard: SyntaxError parsing modified code for {target_file_path.name}: {e}. Cannot check for duplicates.")
            return False # Let validator handle syntax error

        project_root = digester.repo_path
        # Assuming _get_module_qname_from_path is a static or instance method on digester
        module_qname = digester._get_module_qname_from_path(target_file_path, project_root)

        def simple_type_resolver(node: ast.AST, hint_category: str) -> Optional[str]:
            # For nodes within modified_code_str, we don't have Pyanalyze types yet.
            # Fallback to "Any" or try to get text from annotation if present.
            # Node is the annotation node itself if category is 'return_annotation'
            # Node is ast.arg if category is parameter-related.
            annotation_node = None
            if category_hint.endswith("return_annotation"):
                annotation_node = node # node is the annotation itself
            elif hasattr(node, 'annotation') and node.annotation:
                annotation_node = node.annotation

            if annotation_node:
                if isinstance(annotation_node, ast.Name): return annotation_node.id
                if isinstance(annotation_node, ast.Constant): # Python 3.8+ for str/None consts in annotations
                    if isinstance(annotation_node.value, str): return annotation_node.value
                    return str(annotation_node.value)
                if hasattr(ast, 'unparse'): # Attempt to unparse more complex annotations
                    try: return ast.unparse(annotation_node)
                    except: pass # Fall through if unparse fails
                # For older Pythons or complex types not handled by unparse if simple,
                # a more basic representation might be needed, or just default to Any.
                # For now, ast.unparse is a good general attempt for Py 3.9+.
            return "typing.Any" # Default to Any

        for item_node in module_ast.body:
            if isinstance(item_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # For top-level functions
                # The generate_function_signature_string expects func_node, type_resolver, and optional class_name
                signature_str = generate_function_signature_string(item_node, simple_type_resolver)
                # SignatureTrie.search returns List[str] of FQNs if found, or empty list.
                # We need to check if the list is non-empty.
                # The mock in SignatureTrie was returning bool, this needs to be aligned.
                # Assuming search now returns List[str] as per its typical design.
                if digester.signature_trie.search(signature_str): # If list is not empty, a duplicate exists
                    print(f"DuplicateGuard: Duplicate top-level function signature found for '{module_qname}.{item_node.name}': {signature_str}")
                    return True
            elif isinstance(item_node, ast.ClassDef):
                class_name_simple = item_node.name
                # class_fqn_prefix = f"{module_qname}.{class_name_simple}" # Full FQN not needed for prefix arg
                for method_node in item_node.body:
                    if isinstance(method_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        # Pass simple class name as prefix for method name part of signature string
                        signature_str = generate_function_signature_string(method_node, simple_type_resolver, class_fqn_prefix_for_method_name=class_name_simple)
                        if digester.signature_trie.search(signature_str):
                            print(f"DuplicateGuard: Duplicate method signature found for '{module_qname}.{class_name_simple}.{method_node.name}': {signature_str}")
                            return True

        print(f"DuplicateGuard: No duplicate signatures found in {target_file_path.name}.")
        return False

    def run(
        self,
        phase_ctx: 'Phase',
        digester: 'RepositoryDigester',
        validator_handle: Callable[[Optional[str], Path, 'RepositoryDigester', Path], Tuple[bool, Optional[str]]], # Updated signature from R5.7
        score_style_handle: Callable[[Any, Dict[str, Any]], float]
    ) -> Tuple[Optional[str], Optional[str]]: # Returns (script_string, source_info_string)
        final_patch_script: Optional[str] = None
        patch_source_info: Optional[str] = None

        # self.current_patch_candidate will hold the LibCST SCRIPT string.
        # It's initialized to None in __init__.
        # Patch history stores (script_str, is_valid, score, error_traceback_str)
        # For this subtask, patch_source_info is not added to patch_history records yet.

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
        context_data["retrieved_code_snippets"] = digester.get_code_snippets_for_phase(phase_ctx)
        context_data["pdg_slice"] = digester.get_pdg_slice_for_phase(phase_ctx)

        # Store context for potential use in utility methods like generate_patch_preview
        self._current_run_context_data = context_data

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
            return None, "LLMCore_scaffold_failed" # Return None script, source info
        patch_source_info = "LLMCore_scaffold_v1"

        print(f"Step 2: LLM scaffolding pass complete. Script len: {len(scaffold_patch_script)}, Summary: {edit_summary_list}")

        # --- Diffusion Core expands scaffold ---
        diffusion_edit_summary_str = "; ".join(edit_summary_list) if edit_summary_list else ""
        completed_patch_script = self.diffusion_agent.expand_scaffold(scaffold_patch_script, diffusion_edit_summary_str, context_data)

        if completed_patch_script is None:
            print("DiffusionCore failed to expand/complete the scaffold script. Aborting.")
            return None, patch_source_info # Keep last known source

        if completed_patch_script != scaffold_patch_script : # Update source if script changed
            patch_source_info = "DiffusionCore_expansion_v1"
        self.current_patch_candidate = completed_patch_script
        print(f"Step 3: Diffusion expansion pass complete. Script len: {len(self.current_patch_candidate)}. Source: {patch_source_info}")

        # --- Iterative Refinement Loop ---
        # self.max_repair_attempts defined in __init__
        # self.current_patch_candidate holds the LibCST SCRIPT string.

        self.digester = digester # Store digester instance for access in other methods if needed, or pass via context

        for i in range(self.max_repair_attempts):
            print(f"\n--- Repair Iteration {i + 1}/{self.max_repair_attempts} ---")

            modified_code_content_for_iteration: Optional[str] = None
            error_traceback: Optional[str] = None
            validation_payload: Any = None
            is_duplicate_detected = False # Initialize

            target_file_str = context_data.get("phase_target_file")
            original_content: Optional[str] = None

            if not target_file_str:
                is_valid = False
                error_traceback = "PhaseConfigurationError: Missing target_file in phase context."
                print(f"CollaborativeAgentGroup Error: {error_traceback}")
            elif not self.current_patch_candidate: # Should be a LibCST script string
                is_valid = False
                error_traceback = "AgentError: No current patch candidate (script) to apply."
                print(f"CollaborativeAgentGroup Error: {error_traceback}")
            else:
                # Attempt to apply the patch script
                original_content = self.digester.get_file_content(Path(target_file_str))
                if original_content is None:
                    # Check if it's a new file scenario (target might not exist yet)
                    # For new files, original_content should be effectively empty.
                    # This needs to be determined based on phase intention, e.g. "create_file" op.
                    # For now, if get_file_content returns None, and it's not explicitly a new file op, it's an issue.
                    # Assuming for now that if target_file is specified, it should exist unless it's a new file op.
                    # Let's assume for new file, original_content would be "" (handled by get_file_content or here)
                    # If it's not a new file op and content is None, that's an issue.
                    # For simplicity, if get_file_content returns None, we'll treat as empty for application.
                    print(f"CollaborativeAgentGroup Info: Original content for '{target_file_str}' not found. Assuming empty for patch application (e.g. new file).")
                    original_content = ""

                try:
                    modified_code_content_for_iteration = apply_libcst_codemod_script(
                        original_content,
                        self.current_patch_candidate # This is the LibCST script string
                    )
                    print(f"CollaborativeAgentGroup: Successfully applied script for iteration {i+1}.")
                except PatchApplicationError as pae:
                    is_valid = False
                    error_traceback = f"PatchApplicationError: Failed to apply LibCST script: {pae}"
                    modified_code_content_for_iteration = None # Ensure it's None on failure
                    print(f"CollaborativeAgentGroup Error: {error_traceback}")
                except Exception as e_apply: # Catch other unexpected errors during application
                    is_valid = False
                    error_traceback = f"UnexpectedErrorApplyingPatch: {type(e_apply).__name__} - {e_apply}"
                    modified_code_content_for_iteration = None
                    print(f"CollaborativeAgentGroup Error: {error_traceback}")

            # --- Step 5 (Part A): Duplicate Guard (if script application was successful) ---
            if is_valid and modified_code_content_for_iteration is not None and target_file_str:
                is_duplicate_detected = self._perform_duplicate_guard(modified_code_content_for_iteration, Path(target_file_str), context_data)
                if is_duplicate_detected:
                    is_valid = False
                    error_traceback = "DUPLICATE_DETECTED: REUSE_EXISTING_HELPER"
                    print(f"CollaborativeAgentGroup: {error_traceback}")

            # --- Step 5 (Part B): Validate Patch (if script applied and no duplicate) ---
            # This now expects validator_handle to take modified_code_content_str.
            # The actual change to Validator.validate_patch's signature is the next step.
            if is_valid and modified_code_content_for_iteration is not None and target_file_str:
                # Assuming validator_handle's signature will be:
                # (modified_code_content: str, target_file: Path, digester: 'RepositoryDigester', project_root: Path)
                # -> (bool, Optional[str], Optional[str]) where payload is now the error string.
                # For now, the payload from validator_handle is still Any.
                is_valid, validation_payload, error_traceback_validator = validator_handle(
                    modified_code_content_for_iteration, # Pass applied code content
                    Path(target_file_str), # Pass Path object
                    digester,
                    digester.repo_path # Pass project_root
                )
                if not is_valid and error_traceback_validator: # If validator provides an error, use it.
                    error_traceback = error_traceback_validator
                print(f"Loop Step A (Validation on content): is_valid={is_valid}, payload={validation_payload}, error='{bool(error_traceback)}'")
            elif is_valid and modified_code_content_for_iteration is None: # Should have been caught by patch application checks
                is_valid = False
                error_traceback = "InternalError: modified_code_content is None despite is_valid being True before validation call."
                print(f"CollaborativeAgentGroup Error: {error_traceback}")


            # --- Score Patch Style ---
            # Style scoring should ideally run on the modified_code_content_for_iteration.
            # However, score_style_handle expects the "patch" which has been the script.
            # For now, we continue passing the script to style scorer. This could be refined.
            style_score_content_to_score = self.current_patch_candidate # Default to script
            if is_duplicate_detected : # If duplicate, style score is low.
                 style_score = 0.0
            else:
                 style_score = score_style_handle(style_score_content_to_score, self.style_profile)

            print(f"Loop Step B (Style Score on script): Style score: {style_score:.2f}")
            # Record history using the SCRIPT that was attempted.
            self.patch_history.append((self.current_patch_candidate, is_valid, style_score, error_traceback))

            # --- Decision logic based on validation ---
            if is_valid:
                print(f"Script validated successfully in iteration {i+1}. Proceeding to polish the SCRIPT.")
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
                    # Validator_handle now expects: (modified_code_content: str, target_file: Path, digester: 'RepositoryDigester', project_root: Path)
                    # We need to apply the polished script first.
                    # This re-application logic is similar to the start of the loop.

                    temp_error_applying_polished: Optional[str] = None
                    modified_content_after_polish: Optional[str] = None
                    if target_file_str: # target_file_str defined at the start of the loop
                        try:
                            # original_content was fetched at the start of the loop iteration.
                            # If current_patch_candidate (polished script) is None, this block is skipped.
                            modified_content_after_polish = apply_libcst_codemod_script(original_content if original_content is not None else "", self.current_patch_candidate)
                        except PatchApplicationError as pae_polish:
                            temp_error_applying_polished = f"PatchApplicationError after polish: {pae_polish}"
                        except Exception as e_polish_apply:
                             temp_error_applying_polished = f"Unexpected error applying polished script: {e_polish_apply}"
                    else: # Should not happen if initial checks for target_file_str are done
                        temp_error_applying_polished = "Target file string was missing for post-polish validation."

                    if temp_error_applying_polished:
                        final_is_valid = False
                        final_error_traceback = temp_error_applying_polished
                        final_style_score = 0.0 # Or previous score
                        print(f"CollaborativeAgentGroup Error: {final_error_traceback}")
                    elif modified_content_after_polish is not None and target_file_str:
                        final_is_valid, final_error_traceback = validator_handle( # type: ignore
                            modified_content_after_polish, Path(target_file_str), digester, digester.repo_path
                        )
                        final_style_score = score_style_handle(self.current_patch_candidate, self.style_profile) # Score the script
                    else: # Should not be reached if logic is correct
                        final_is_valid = False
                        final_error_traceback = "Internal error: Could not re-apply polished script for final validation."
                        final_style_score = 0.0
                        print(f"CollaborativeAgentGroup Error: {final_error_traceback}")


                    print(f"Loop Step C.1 (Post-Polish Validation): Valid: {final_is_valid}, Style: {final_style_score:.2f}")
                    self.patch_history.append((self.current_patch_candidate, final_is_valid, final_style_score, final_error_traceback))
                    is_valid = final_is_valid
                    error_traceback = final_error_traceback if final_error_traceback else error_traceback

                    if final_is_valid:
                        print(f"Polished script is valid. Final style score: {final_style_score:.2f}. This script is the result of this phase.")
                        patch_source_info = f"LLMCore_polish_iteration_{i + 1}" # Or more specific based on pre-repair/post-repair
                        return self.current_patch_candidate, patch_source_info
                    else:
                        print("Polished script failed validation.")
                        # Fall through to repair logic for the polished_but_failed_script

            # If not valid after initial check, or if polish made it invalid / returned None:
            if not is_valid: # Combined check
                print(f"Script requires repair (or loop exit) after iteration {i+1} (is_valid={is_valid}). Error: {error_traceback}")
                if error_traceback:
                    # --- Phase 5: Step 5 (LLM Proposes Repair - was Step 6a) ---
                    print("Loop Step D (Attempt Repair): Attempting repair with LLMCore.")

                    # Ensure context_data for repair includes the script that just failed
                    context_data_for_repair = context_data.copy()
                    context_data_for_repair['current_patch_candidate_DEBUG'] = self.current_patch_candidate # Add the failing script for LLM to see

                    target_file_str = context_data.get("phase_target_file")

                    # --- Attempt Heuristic Fixes (New Step before LLM repair) ---
                    heuristically_fixed_script: Optional[str] = None
                    if target_file_str and error_traceback: # Ensure necessary info is available
                        print("Loop Step D.1 (Heuristic Fix Attempt): Attempting heuristic fixes via Validator.")
                        heuristically_fixed_script = self.validator.attempt_heuristic_fixes(
                            error_traceback,
                            self.current_patch_candidate, # The script that just failed (after polish)
                            target_file_str
                        )

                    if heuristically_fixed_script:
                        print("CollaborativeAgentGroup: Validator proposed a heuristic fix. Updating current_patch_candidate and restarting iteration for re-validation.")
                        # Add a history entry for the heuristic attempt itself.
                        # The outcome of this attempt will be recorded in the next iteration's main validation.
                        # Use previous style score as a placeholder, or a specific marker.
                        previous_style_score = self.patch_history[-1][2] if self.patch_history else 0.0
                        self.patch_history.append((
                            heuristically_fixed_script, # The script *after* heuristic fix
                            "pending_revalidation",     # Validation status
                            previous_style_score,       # Placeholder score
                            "Heuristic fix applied, pending re-validation" # Traceback
                        ))
                        self.current_patch_candidate = heuristically_fixed_script
                        continue # Restart the loop to re-evaluate the heuristically fixed script from the top

                    # If no heuristic fix was applied or successful, proceed to LLM-based repair
                    # This 'if' condition is implicitly met if 'continue' was not hit.
                    # The 'is_valid' and 'error_traceback' are from the original failure (or failed polish)
                    # that triggered this repair block.
                    print("Loop Step D.2 (LLM Repair Attempt): No heuristic fix applied or taken. Attempting LLM repair.")
                    llm_proposed_fix_script = self.llm_agent.propose_repair_diff(error_traceback, context_data_for_repair)

                        if llm_proposed_fix_script:
                            print(f"LLMCore proposed a fix script (len: {len(llm_proposed_fix_script)}).")
                            print("Loop Step E (Diffusion Re-Denoise/Merge): Processing LLM's proposed fix script.")
                            merged_fixed_script = self.diffusion_agent.re_denoise_spans(
                                failed_patch_script=self.current_patch_candidate, # Script before LLM proposal
                                proposed_fix_script=llm_proposed_fix_script,
                                context_data=context_data_for_repair
                            )

                            if merged_fixed_script:
                                print(f"DiffusionCore processed the fix. Resulting script len: {len(merged_fixed_script)}.")
                                self.current_patch_candidate = merged_fixed_script
                            else:
                                print("DiffusionCore did not return a script after processing LLM's fix. Breaking repair loop.")
                                break
                        else:
                            print("LLMCore could not propose a repair. Breaking repair loop.")
                            break
                    elif not error_traceback and not is_valid: # e.g. heuristic fix removed error but still not valid for other reasons
                        print("Loop Step D.3 (Skipping LLM Repair): No error traceback, but script still not valid. Cannot proceed with LLM repair. Breaking loop.")
                        break

            else: # No error_traceback from validation (e.g. style score too low but otherwise valid, or duplicate without specific repair path)
                print("Loop Step D/E (No Repair Path): Validation failed without an actionable traceback, or duplicate guard triggered generic error. No further repair attempts in this iteration.")
                # If style is the only issue, and we don't have a specific re-denoise for style, we might break.
                # Or, if a previous version was good enough, that might be selected later.
                break # Break if no clear path to repair/improve

            if i == self.max_repair_attempts - 1 and not is_valid: # Check is_valid for the last attempt
                print(f"Max repair attempts ({self.max_repair_attempts}) reached. Could not produce a valid and satisfactory script.")
                # Log the last error that prevented success
                last_error_for_failure = error_traceback if error_traceback else "Unknown error after max repair attempts."

                # Fallback: Check if any script in history was valid and had a decent score
                # This logic is already present for returning best from history if loop finishes
                # But here we explicitly raise PhaseFailure

                # Find best from history before failing
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

                if best_historical_script: # This check is actually done after the loop by existing logic
                    print(f"Best historical script was (Valid: {was_valid_in_history}, Score: {best_historical_score:.2f}), but phase still failed on last attempt.")

                raise PhaseFailure(f"Failed to validate/repair patch after {self.max_repair_attempts} attempts. Last error: {last_error_for_failure}")

        # After loop, if we exited due to success (is_valid became True and returned), this part is skipped.
        # If loop finishes due to max_iterations and no valid patch was found and returned within the loop:
        print("Exited refinement loop. Selecting best script from history or returning None.")
        best_historical_script = None
        best_historical_score = -1.0
        was_valid_in_history = False
        for p_hist, v_hist, s_hist, _tb_hist in self.patch_history:
            if v_hist and s_hist > best_historical_score:
                best_historical_script = p_hist
                best_historical_score = s_hist
                was_valid_in_history = True
            elif not was_valid_in_history and s_hist > best_historical_score:
                best_historical_script = p_hist
                best_historical_score = s_hist

        if best_historical_script:
            print(f"Returning best script from history (Valid: {was_valid_in_history}, Score: {best_historical_score:.2f}).")
            return best_historical_script

        # If no script in history was ever good (e.g. all failed initial validation, no repairs worked)
        # and loop finished without returning a valid one.
        # This case should ideally be covered by the PhaseFailure exception if max_repair_attempts is the sole reason for exiting without success.
        # However, if the loop breaks for other reasons (e.g., no repair path found before max_attempts), this is the final fallback.
        if not self.patch_history or not any(h[1] for h in self.patch_history) : # if no history or no valid patch in history
             # Ensure error_traceback is defined; it might not be if the loop was never fully entered or broke very early.
             final_error_msg = error_traceback if 'error_traceback' in locals() and error_traceback else "No successful patch and no specific error traceback recorded."
             raise PhaseFailure(f"No valid patch could be generated after loop. Last error: {final_error_msg}")

        return None # Should be unreachable if PhaseFailure is raised or a script is returned.


    def generate_patch_preview(self) -> str:
        """Generates a mock preview of the current patch candidate (LibCST script)."""
        target_file_name = "N/A"
        if self._current_run_context_data:
            target_file_name = self._current_run_context_data.get("phase_target_file", "N/A")

        script_content_preview = "No patch candidate available."
        if self.current_patch_candidate and isinstance(self.current_patch_candidate, str):
            script_content_preview = self.current_patch_candidate[:1000] + ("..." if len(self.current_patch_candidate) > 1000 else "")
        elif self.current_patch_candidate:
            script_content_preview = f"Patch candidate is not a string (type: {type(self.current_patch_candidate)}). Preview unavailable."


        preview = f"""
Patch Preview (Mock):
---------------------
Target File: {target_file_name}
Patch Type: LibCST Script (Python code to perform an edit)

Script Content (first 1000 chars):
{script_content_preview}

(Note: This is a mock preview of the LibCST script.
 A full preview would show a diff of the target file after applying this script.)
"""
        return preview.strip()

    def abort_and_rollback(self) -> None:
        """Placeholder: Aborts the current operation and conceptually rolls back changes."""
        patch_status_info = "No active patch candidate."
        if self.current_patch_candidate and isinstance(self.current_patch_candidate, str):
            patch_status_info = f"Current patch candidate (script) length: {len(self.current_patch_candidate)}."
        elif self.current_patch_candidate:
            patch_status_info = f"Current patch candidate type: {type(self.current_patch_candidate)}."

        print(f"CollaborativeAgentGroup: Abort and Rollback called. {patch_status_info}")
        print("  (Mock: No file operations to roll back at this stage as scripts are not yet applied.)")

        # Reset internal state related to the current run
        self.current_patch_candidate = None
        self.patch_history = []
        self._current_run_context_data = None # Clear the context of the aborted run

        print("  Internal state (patch candidate, history, context) has been reset.")
        # In a real scenario, this might involve:
        # - Deleting temporary files
        # - Reverting any applied changes in a sandbox environment
        # - Signaling to other components that the phase was aborted
        pass

# Example Usage (Conceptual - requires mock objects for Phase, Digester etc.)
if __name__ == '__main__':
    from src.utils.config_loader import load_app_config # For __main__
    from src.validator.validator import Validator # For __main__

    print("\n--- CollaborativeAgentGroup Example Usage (Conceptual) ---")

    app_cfg_main = load_app_config()
    app_cfg_main["general"]["verbose"] = True
    if not app_cfg_main["general"].get("project_root"):
        app_cfg_main["general"]["project_root"] = str(Path.cwd())


    # Mock Phase (replace with actual Phase import and instantiation if available)
    class MockPhase:
        def __init__(self, op_name, target, params=None, description="Mock phase description"):
            self.operation_name = op_name
            self.target_file = target
            self.parameters = params if params else {}
            self.description = description


    # Mock Digester (replace with actual Digester import and instantiation)
    class MockDigester:
        def __init__(self, app_config_param): # Mock accepts app_config
            self.app_config = app_config_param
            self.repo_path = Path(app_config_param.get("general",{}).get("project_root", "."))
            self.verbose = app_config_param.get("general",{}).get("verbose", False)
            if self.verbose: print(f"MockDigester initialized for CollaborativeAgentGroup main, repo_path: {self.repo_path}")

        def get_project_overview(self): return {"files": 2, "language": "python", "mock_overview": True}
        def get_file_content(self, path: Path): return f"# Content of {path}\npass" if path else None
        def get_code_snippets_for_phase(self, phase_ctx_param: Any) -> Dict[str, str]:
            return {"mock_snippet.py": "def mock_func(): pass"}
        def get_pdg_slice_for_phase(self, phase_ctx_param: Any) -> Dict[str, Any]:
            return {"nodes": [], "edges": [], "info": "Mock PDG"}
        def _get_module_qname_from_path(self, file_path: Path, project_root: Path) -> str:
            return file_path.stem # Simplified for mock

    # Mock validator and scorer (Validator itself will be refactored later)
    mock_validator_instance = Validator(app_config=app_cfg_main) # Pass app_config

    def mock_validator_handle(modified_code: str, target_file: Path, digester_param: Any, project_root_param: Path) -> Tuple[bool, Optional[str], Optional[str]]:
        if "ERROR" in modified_code:
            return False, "Contains ERROR string", "Traceback: Something went wrong due to ERROR"
        if target_file.name == "invalid.py":
            return False, "Invalid path", "Traceback: InvalidPathError"
        return True, "Validated by mock", None

    def mock_score_style(patch_script: Any, style_profile_param: Dict[str, Any]) -> float:
        if isinstance(patch_script, str) and "bad_style" in patch_script:
            return 0.2
        return 0.85

    # Setup
    mock_style_profile = {"line_length": 88, "indent_style": "space"}
    mock_naming_db_path = Path(app_cfg_main["general"]["project_root"]) / "mock_naming_db.json" # Relative to project root

    # Create dummy naming db file
    mock_naming_db_path.parent.mkdir(parents=True, exist_ok=True)
    with open(mock_naming_db_path, "w") as f:
        json.dump({"function_prefix": "get_"}, f)

    mock_digester_for_cag = MockDigester(app_config_param=app_cfg_main)

    agent_group = CollaborativeAgentGroup(
        app_config=app_cfg_main,
        digester=mock_digester_for_cag, # type: ignore
        style_profile=mock_style_profile,
        naming_conventions_db_path=mock_naming_db_path,
        validator_instance=mock_validator_instance
    )

    mock_phase_ctx = MockPhase(op_name="add_function", target="src/example.py", params={"function_name": "my_func"})

    print("\nStarting agent_group.run...")
    try:
        final_patch_script_result, final_patch_source_info = agent_group.run(
            phase_ctx=mock_phase_ctx,
            digester=mock_digester_for_cag, # type: ignore
            validator_handle=mock_validator_handle, # type: ignore
            score_style_handle=mock_score_style
        )
        print(f"\nAgent group run completed. Final patch script: '{str(final_patch_script_result)[:100]}...', Source: {final_patch_source_info}")
        if final_patch_script_result : print(f"Patch preview: {agent_group.generate_patch_preview()}")

    except PhaseFailure as pf:
        print(f"CollaborativeAgentGroup Main: PhaseFailure encountered: {pf}")
    except Exception as e:
        print(f"CollaborativeAgentGroup Main: Unexpected error: {e}")
        import traceback
        traceback.print_exc()

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
