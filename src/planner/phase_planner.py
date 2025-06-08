# src/planner/phase_planner.py
from typing import List, Dict, Any, Union, Optional, TYPE_CHECKING, Type, Tuple, Callable # Added Type, Tuple, Callable
from pathlib import Path
import json
import hashlib
import random # For dummy scorer and branch names

# Forward references for type hints if full imports are problematic
if TYPE_CHECKING:
    from ..digester.repository_digester import RepositoryDigester
    # Spec, Phase, BaseRefactorOperation are now directly imported.
    # REFACTOR_OPERATION_INSTANCES is used directly.

# Actual imports
from .spec_model import Spec
from .phase_model import Phase
from .refactor_grammar import BaseRefactorOperation, REFACTOR_OPERATION_INSTANCES
from src.validator.validator import Validator
from src.builder.commit_builder import CommitBuilder # New import
from src.agent_group.exceptions import PhaseFailure # New import

# REFACTOR_OPERATION_CLASSES is not directly used by PhasePlanner logic, REFACTOR_OPERATION_INSTANCES is.

# Fallback for RepositoryDigester if not available (e.g. in isolated subtask)
# This is primarily for type hinting if RepositoryDigester cannot be imported directly.
# The actual instance is passed to __init__.
if TYPE_CHECKING:
    from src.agent_group.collaborative_agent_group import CollaborativeAgentGroup
    # Ensure Phase is available for type hints if not already.
    from .phase_model import Phase

if TYPE_CHECKING or 'RepositoryDigester' not in globals(): # Keep this for RepositoryDigester hint
    class RepositoryDigester: # type: ignore
        project_call_graph: Dict
        project_control_dependence_graph: Optional[Dict] = None
        project_data_dependence_graph: Optional[Dict] = None
        # Add methods expected by CollaborativeAgentGroup.run context_data population
        def get_project_overview(self) -> Dict[str, Any]:
            print("MockDigester.get_project_overview called")
            return {"mock_overview": True}
        def get_file_content(self, path: Path) -> Optional[str]:
            print(f"MockDigester.get_file_content called for {path}")
            return f"# Mock content for {path}" if path else None
        pass # Ensure class body is not empty


class PhasePlanner:
    # mock_validator_handle is now removed.
    # The actual validator instance's validate_patch method will be used.

    @staticmethod
    def mock_score_style_handle(patch: Any, style_profile: Dict[str, Any]) -> float:
        print(f"PhasePlanner.mock_score_style_handle: Scoring style for patch: {str(patch)[:100]} with profile keys: {list(style_profile.keys())}")
        # Assuming patch is now a script string, this mock might need adjustment if it were to inspect content.
        # For now, it's a simple mock.
        if isinstance(patch, str) and "BAD_STYLE_MARKER_IN_SCRIPT" in patch:
            return 0.3 # Low score for bad style
        return 0.9 # Default high score

    def __init__(
        self,
        style_fingerprint_path: Path,
        digester: 'RepositoryDigester',
        scorer_model_config: Any, # Config for Phi-2 LLM scorer
        naming_conventions_db_path: Path,
        project_root_path: Path,
        llm_model_path: Optional[str] = None, # New: Path to GGUF model for AgentGroup
        refactor_op_map: Optional[Dict[str, BaseRefactorOperation]] = None,
        beam_width: int = 3 # Default beam_width to 3
    ):
        self.style_fingerprint: Dict[str, Any] = {}
        try:
            if style_fingerprint_path.exists():
                with open(style_fingerprint_path, "r", encoding="utf-8") as f:
                    self.style_fingerprint = json.load(f)
                print(f"PhasePlanner: Loaded style fingerprint from {style_fingerprint_path}")
            else:
                print(f"Warning: Style fingerprint file not found at {style_fingerprint_path}. Using empty fingerprint.")
        except json.JSONDecodeError as e:
            print(f"Warning: Error decoding style fingerprint JSON from {style_fingerprint_path}: {e}. Using empty fingerprint.")
        except Exception as e_gen:
             print(f"Warning: Could not load style fingerprint from {style_fingerprint_path} due to: {e_gen}. Using empty fingerprint.")

        self.digester = digester
        self.scorer_model_config = scorer_model_config
        self.naming_conventions_db_path = naming_conventions_db_path # Store
        self.project_root_path = project_root_path # Store

        self.refactor_op_map: Dict[str, BaseRefactorOperation] = refactor_op_map if refactor_op_map is not None else REFACTOR_OPERATION_INSTANCES

        self.beam_width = beam_width
        if self.beam_width <= 0:
            raise ValueError("Beam width must be a positive integer.")

        self.phi2_scorer = None
        if self.scorer_model_config and self.scorer_model_config.get("enabled", True):
            try:
                model_name = self.scorer_model_config.get("model_name", "microsoft/phi-2")
                device_setting = self.scorer_model_config.get("device", "cpu")
                self.phi2_scorer = "mock_phi2_scorer_initialized"
                print(f"Info: LLM scorer mock-initialized for planner. (Config: model='{model_name}', device='{device_setting}')")
            except Exception as e:
                print(f"Warning: Failed to initialize LLM scorer (mock setup): {e}. Falling back to random scoring.")
                self.phi2_scorer = None
        else:
            print("Info: LLM scorer is disabled or no config provided for planner. Falling back to random scoring.")
            self.phi2_scorer = None

        # Import and Initialize CollaborativeAgentGroup
        from src.agent_group.collaborative_agent_group import CollaborativeAgentGroup

        # Initialize Validator
        self.validator = Validator(config=None) # Pass config if PhasePlanner has one for Validator
        print("PhasePlanner: Validator instance initialized.")

        # Import and Initialize CollaborativeAgentGroup
        from src.agent_group.collaborative_agent_group import CollaborativeAgentGroup
        self.agent_group: 'CollaborativeAgentGroup' = CollaborativeAgentGroup(
            style_profile=self.style_fingerprint,
            naming_conventions_db_path=self.naming_conventions_db_path,
            validator_instance=self.validator, # Pass validator instance
            llm_core_config=None,
            diffusion_core_config=None,
            llm_model_path=llm_model_path
        )
        print("PhasePlanner: CollaborativeAgentGroup initialized.")

        # Store actual validator handle and mock score style handle
        self.validator_handle: Callable[[Optional[str], str, 'RepositoryDigester', Path], Tuple[bool, Optional[str]]] = self.validator.validate_patch
        self.score_style_handle: Callable[[Any, Dict[str, Any]], float] = PhasePlanner.mock_score_style_handle
        print("PhasePlanner: Validator handle set to actual validator. Mock style scorer configured.")

        # Initialize CommitBuilder
        self.commit_builder = CommitBuilder(config=None) # Pass config if available/needed
        print("PhasePlanner: CommitBuilder instance initialized.")

        self.plan_cache: Dict[str, List[Phase]] = {}
        # self.phi2_scorer_cache: Dict[str, float] = {} # Optional cache

    def _get_spec_cache_key(self, spec: Spec) -> str:
        # Ensure Spec model is Pydantic v2 compatible for model_dump_json if that's used.
        # Fallback to dict if model_dump_json is not available (e.g. placeholder Spec)
        if hasattr(spec, 'model_dump_json'):
            spec_json_str = spec.model_dump_json(sort_keys=True)
        else: # Fallback for placeholder Spec or Pydantic v1
            spec_json_str = json.dumps(spec.dict(sort_keys=True) if hasattr(spec,'dict') else vars(spec), sort_keys=True)
        return hashlib.md5(spec_json_str.encode('utf-8')).hexdigest()

    def _get_graph_statistics(self) -> Dict[str, Any]:
        graph_stats = {
            "num_call_graph_nodes": 0,
            "num_call_graph_edges": 0,
            "cg_density": 0.0,
            "num_cdg_nodes": 0,
            "num_cdg_edges": 0,
            "cdg_density": 0.0,
            "num_ddg_nodes": 0,
            "num_ddg_edges": 0,
            "ddg_density": 0.0,
        }

        # Call Graph (CG) statistics
        if hasattr(self.digester, 'project_call_graph') and self.digester.project_call_graph is not None:
            cg_nodes = self.digester.project_call_graph
            graph_stats["num_call_graph_nodes"] = len(cg_nodes)
            graph_stats["num_call_graph_edges"] = sum(len(edges) for edges in cg_nodes.values())
            if graph_stats["num_call_graph_nodes"] > 1:
                v = graph_stats["num_call_graph_nodes"]
                e = graph_stats["num_call_graph_edges"]
                graph_stats["cg_density"] = e / (v * (v - 1))
            else:
                graph_stats["cg_density"] = 0.0

        # Control Dependence Graph (CDG) statistics
        if hasattr(self.digester, 'project_control_dependence_graph') and self.digester.project_control_dependence_graph is not None:
            cdg_nodes = self.digester.project_control_dependence_graph
            graph_stats["num_cdg_nodes"] = len(cdg_nodes)
            graph_stats["num_cdg_edges"] = sum(len(edges) for edges in cdg_nodes.values()) # Assuming similar structure to CG
            if graph_stats["num_cdg_nodes"] > 1:
                v = graph_stats["num_cdg_nodes"]
                e = graph_stats["num_cdg_edges"]
                graph_stats["cdg_density"] = e / (v * (v - 1))
            else:
                graph_stats["cdg_density"] = 0.0

        # Data Dependence Graph (DDG) statistics
        if hasattr(self.digester, 'project_data_dependence_graph') and self.digester.project_data_dependence_graph is not None:
            ddg_nodes = self.digester.project_data_dependence_graph
            graph_stats["num_ddg_nodes"] = len(ddg_nodes)
            graph_stats["num_ddg_edges"] = sum(len(edges) for edges in ddg_nodes.values()) # Assuming similar structure to CG
            if graph_stats["num_ddg_nodes"] > 1:
                v = graph_stats["num_ddg_nodes"]
                e = graph_stats["num_ddg_edges"]
                graph_stats["ddg_density"] = e / (v * (v - 1))
            else:
                graph_stats["ddg_density"] = 0.0

        # Ensure all keys are present even if graphs are missing
        final_stats = {
            "num_call_graph_nodes": graph_stats.get("num_call_graph_nodes", 0),
            "num_call_graph_edges": graph_stats.get("num_call_graph_edges", 0),
            "cg_density": graph_stats.get("cg_density", 0.0),
            "num_cdg_nodes": graph_stats.get("num_cdg_nodes", 0),
            "num_cdg_edges": graph_stats.get("num_cdg_edges", 0),
            "cdg_density": graph_stats.get("cdg_density", 0.0),
            "num_ddg_nodes": graph_stats.get("num_ddg_nodes", 0),
            "num_ddg_edges": graph_stats.get("num_ddg_edges", 0),
            "ddg_density": graph_stats.get("ddg_density", 0.0),
        }

        print(f"PhasePlanner: Generated graph stats for scorer: {final_stats}")
        return final_stats

    def _score_candidate_plan_with_llm(self, candidate_plan: List['Phase'], graph_stats: Dict[str, Any]) -> float:
        if not self.phi2_scorer or self.phi2_scorer != "mock_phi2_scorer_initialized":
            return random.uniform(0.2, 0.8)

        if not candidate_plan:
            return 0.05

        plan_str_parts = []
        for i, phase in enumerate(candidate_plan):
            try:
                params_json = json.dumps(phase.parameters)
            except TypeError:
                params_json = str(phase.parameters)

            plan_str_parts.append(
                f"Phase {i + 1} (Operation: {phase.operation_name}):
"
                f"  Target File: {phase.target_file or 'N/A'}
"
                f"  Parameters: {params_json}
"
                f"  Description: {phase.description}"
            )
        plan_formatted_str = "\n\n".join(plan_str_parts)

        try:
            style_fp_str = json.dumps(self.style_fingerprint, indent=2)
        except TypeError:
            style_fp_str = str(self.style_fingerprint)

        try:
            graph_stats_str = json.dumps(graph_stats, indent=2)
        except TypeError:
            graph_stats_str = str(graph_stats)

        prompt = f"""You are an expert software engineering assistant. Your task is to evaluate a proposed refactoring plan for a Python codebase.
A higher score indicates a better plan. Consider the plan's clarity, correctness, potential risks, efficiency, and how well it seems to adhere to common best practices and the described project style.

Project Style Fingerprint:
---
{style_fp_str}
---

Codebase Graph Statistics (for context):
---
{graph_stats_str}
---

Candidate Refactoring Plan (Total phases: {len(candidate_plan)}):
---
{plan_formatted_str}
---

Based on all the above information, score this plan on a continuous scale from 0.0 (very bad) to 1.0 (excellent).
Output only the numerical score as a float (e.g., 0.75).
Score:"""

        mock_llm_response_text = ""
        try:
            score_value = 0.5
            score_value += len(candidate_plan) * 0.03
            if candidate_plan: # Check if plan is not empty
                 score_value -= candidate_plan[0].operation_name.count('error') * 0.1

            if any("extract_method" in p.operation_name for p in candidate_plan):
                score_value += 0.05
            if len(candidate_plan) > 7:
                score_value -= (len(candidate_plan) - 7) * 0.02

            score_value += random.uniform(-0.02, 0.02)
            final_mock_score = max(0.0, min(1.0, score_value))
            mock_llm_response_text = f"{final_mock_score:.2f}"
        except Exception as e:
            print(f"Error during Mock LLM scoring simulation: {e}. Falling back to random score.")
            return random.uniform(0.1, 0.5)

        try:
            match = re.search(r"(0?\.\d+|1\.0|1)", mock_llm_response_text) # Corrected regex
            parsed_score = -1.0
            if match:
                parsed_score = float(match.group(1))
            else:
                general_match = re.search(r"[-+]?\d*\.\d+", mock_llm_response_text)
                if general_match:
                    parsed_score = float(general_match.group(0))

            if 0.0 <= parsed_score <= 1.0:
                return parsed_score
            else:
                print(f"Warning: LLM mock score {parsed_score} out of range [0.0, 1.0] from '{mock_llm_response_text}'. Using default.")
                return 0.1
        except ValueError:
            print(f"Warning: ValueError parsing score from LLM mock response: '{mock_llm_response_text}'. Using default.")
            return 0.1
        except Exception as e:
            print(f"Warning: Unexpected error parsing score: {e}. Raw: '{mock_llm_response_text}'. Using default.")
            return 0.1

    def _beam_search_for_plan(self, spec: Spec, graph_stats: Dict[str, Any]) -> List[Phase]:
        # Initial beam: list of (plan_phases, score)
        beam: List[Tuple[List[Phase], float]] = [([], 0.0)]

        if not spec.operations:
            return []

        # Iterate through each operation defined in the spec
        for op_idx, op_spec_item in enumerate(spec.operations):
            next_beam_candidates: List[Tuple[List[Phase], float]] = []

            op_name = op_spec_item.get("name")
            if not op_name or not isinstance(op_name, str): # Ensure op_name is a string
                print(f"Warning: Spec operation item at index {op_idx} is missing a 'name' or 'name' is not a string. Skipping this item.")
                continue

            operation_instance = self.refactor_op_map.get(op_name)
            if not operation_instance:
                print(f"Warning: Operation '{op_name}' (spec index {op_idx}) not found in refactor grammar. Skipping this item.")
                continue

            phase_parameters = {k: v for k, v in op_spec_item.items() if k not in ["name", "target_file"]}
            target_file_for_phase = op_spec_item.get("target_file")
            if target_file_for_phase is not None and not isinstance(target_file_for_phase, str): # Ensure target_file is str or None
                print(f"Warning: 'target_file' for operation '{op_name}' (spec index {op_idx}) is not a string: {target_file_for_phase}. Treating as None.")
                target_file_for_phase = None


            if not operation_instance.validate_parameters(phase_parameters):
                print(f"Warning: Parameters for operation '{op_name}' (spec index {op_idx}, target: {target_file_for_phase or 'repo-level'}) are invalid. This operation will not be added to plans in the current beam step.")
                continue

            # For each current plan in the beam, try to extend it with the current operation
            for current_plan_phases, current_plan_score in beam:
                new_phase = Phase(
                    operation_name=op_name,
                    target_file=target_file_for_phase,
                    parameters=phase_parameters,
                    description=f"Op {op_idx + 1}/{len(spec.operations)}: {operation_instance.description} for '{op_name}' on '{target_file_for_phase or 'repo-level'}'"
                )

                extended_plan_phases = current_plan_phases + [new_phase]
                extended_score = self._score_candidate_plan_with_llm(extended_plan_phases, graph_stats)
                next_beam_candidates.append((extended_plan_phases, extended_score))

            if not next_beam_candidates:
                if not beam:
                    print(f"Warning: Beam became empty while processing spec operation index {op_idx} ('{op_name}').")
                    return []
                # If beam was not empty, but no candidates for this op_spec_item (e.g. invalid params made us 'continue'),
                # the current 'beam' (from previous step) will be used for the next op_spec_item.
                # This is implicitly handled as 'beam' is only updated if next_beam_candidates is non-empty.
                print(f"Info: No valid new phases generated for spec operation index {op_idx} ('{op_name}'). Current beam carries over.")

            if next_beam_candidates:
                next_beam_candidates.sort(key=lambda x: x[1], reverse=True)
                beam = next_beam_candidates[:self.beam_width]

            if not beam:
                print(f"Warning: Beam search resulted in an empty beam after processing spec operation index {op_idx} ('{op_name}').")
                return []

        if not beam:
            print("Warning: Beam search did not find any valid plan.")
            return []

        best_plan_phases, best_score = beam[0]
        num_phases_in_spec = len(spec.operations)
        num_phases_in_plan = len(best_plan_phases)

        print(f"Beam search completed. Best plan score: {best_score:.4f}. Spec ops: {num_phases_in_spec}, Plan phases: {num_phases_in_plan}.")
        if num_phases_in_plan < num_phases_in_spec:
            print(f"Warning: The generated plan has fewer phases ({num_phases_in_plan}) than specified operations ({num_phases_in_spec}). This might be due to invalid spec items.")

        return best_plan_phases

    def plan_phases(self, spec: Spec) -> List[Phase]:
        print(f"PhasePlanner: Received spec for task: '{spec.issue_description[:50]}...'")
        cache_key = self._get_spec_cache_key(spec)
        if cache_key in self.plan_cache:
            print("PhasePlanner: Returning cached plan.")
            return self.plan_cache[cache_key]

        print("PhasePlanner: No cached plan found. Generating new plan.")
        graph_stats = self._get_graph_statistics()
        best_plan = self._beam_search_for_plan(spec, graph_stats)

        if best_plan:
            self.plan_cache[cache_key] = best_plan # Cache the plan (list of Phase objects)
            print("PhasePlanner: Plan generated and cached.")

            print("\n--- Running CollaborativeAgentGroup for each phase in the best plan ---")
            execution_summary = [] # To store results of agent_group.run for each phase
            for phase_obj in best_plan: # best_plan is List[Phase]
                print(f"Executing phase: {phase_obj.operation_name} on {phase_obj.target_file or 'repo-level'}")
                try:
                    # The CollaborativeAgentGroup's run method signature is:
                    # run(self, phase_ctx: 'Phase', digester: 'RepositoryDigester',
                    #     validator_handle: Callable, score_style_handle: Callable) -> Optional[Any]:
                    # It uses its own initialized style_profile and naming_conventions_db_path.
                    validated_patch = self.agent_group.run(
                        phase_ctx=phase_obj,
                        digester=self.digester,
                        validator_handle=self.validator_handle,
                        score_style_handle=self.score_style_handle
                    )

                    if validated_patch:
                        print(f"Phase {phase_obj.operation_name}: Successfully generated patch: {str(validated_patch)[:100]}...")
                        execution_summary.append({"phase_operation": phase_obj.operation_name, "target": phase_obj.target_file, "status": "success", "patch_preview": str(validated_patch)[:100]})
                    else:
                        print(f"Phase {phase_obj.operation_name}: Failed to generate a patch (returned None).")
                        execution_summary.append({"phase_operation": phase_obj.operation_name, "target": phase_obj.target_file, "status": "failed_no_patch"})
                        # Decide if planning should stop if a phase fails to produce a patch.
                        # For now, continue with other phases.
                except Exception as e:
                    print(f"Error running agent group for phase {phase_obj.operation_name}: {type(e).__name__} - {e}")
                    import traceback
                    traceback.print_exc() # Print full traceback for debugging
                    execution_summary.append({"phase_operation": phase_obj.operation_name, "target": phase_obj.target_file, "status": "error", "error_message": str(e)})
                    # Decide if planning should stop if a phase errors.
                    # For now, continue with other phases.
                    # For this subtask, process only the first successful patch
                    if validated_patch: # A patch script was successfully generated and (mock) validated by agent group
                        print("\nPhasePlanner: Phase successful. Preparing for CommitBuilder integration.")

                        # 1. Mock Patch Application & Create validated_patch_content_map
                        if not phase_obj.target_file:
                            print("PhasePlanner Warning: Phase target_file is None. Cannot proceed with CommitBuilder for this phase.")
                            continue # Or handle as an error depending on expected behavior

                        target_file_path = Path(phase_obj.target_file)
                        # validated_patch is the LibCST script string.
                        # In a real scenario, this script would be applied to the original content.
                        # For this mock, we assume validated_patch IS the new full content.
                        # This aligns with how validator's _apply_patch_script mock currently behaves
                        # if we assume the script itself is the content to be validated.
                        # This is a known simplification point.

                        # If validated_patch is a script, we need to simulate its application.
                        # Let's use the validator's _apply_patch_script for this simulation.
                        original_content = self.digester.get_file_content(self.project_root_path / target_file_path)
                        if original_content is None and not target_file_path.exists(): # New file scenario
                            print(f"PhasePlanner: Original content for {target_file_path} not found, assuming new file for mock application.")
                            original_content = ""

                        if original_content is not None:
                             # The validator's _apply_patch_script needs to be accessible or replicated here.
                             # Or, we assume validated_patch is already the final content.
                             # Let's assume validated_patch (script) IS the final content for simplicity here,
                             # acknowledging this is a mock simplification.
                             # If validated_patch is a script, it should be *applied* by a proper mechanism.
                             # For now, let's pass the script itself as the content.
                             # This means validator should have been validating the script string directly.

                             # If CollaborativeAgentGroup returns the script, then this is the script.
                             # If it's supposed to return applied content, then this is content.
                             # Current return from agent_group.run is the script string.
                             # So, for CommitBuilder, we need the *final content*.
                             # We will use the validator's _apply_patch_script to *simulate* this.

                            print(f"PhasePlanner: (Mock) Simulating application of patch script for {target_file_path}...")
                            final_content_for_commit = self.validator._apply_patch_script(original_content, validated_patch) # type: ignore

                            validated_patch_content_map = {target_file_path: final_content_for_commit}
                            print(f"PhasePlanner: (Mock) Created validated_patch_content_map for {target_file_path}.")
                        else:
                            print(f"PhasePlanner Error: Could not get original content for {target_file_path} to simulate patch application.")
                            continue


                        # 2. Mock diff_summary
                        # In a real scenario, this would come from comparing original_content and final_content_for_commit
                        diff_summary = f"Mock diff summary for {target_file_path}: Applied agent-generated script."
                        print(f"PhasePlanner: (Mock) Generated diff_summary: '{diff_summary}'.")

                        # 3. Mock validator_results_summary
                        validator_results_summary = "Mock validator results: All checks passed on final attempt by agent group."
                        if self.agent_group.patch_history:
                            try:
                                last_attempt_info = self.agent_group.patch_history[-1] # (script, is_valid, score, error_traceback)
                                last_error_tb = last_attempt_info[3] if len(last_attempt_info) > 3 else "N/A"
                                validator_results_summary = f"Final validation in agent: Valid={last_attempt_info[1]}, Score={last_attempt_info[2]:.2f}, Last Error='{str(last_error_tb)[:50]}...'"
                            except Exception as e_hist:
                                print(f"PhasePlanner: Error accessing patch_history: {e_hist}")
                                validator_results_summary = "Could not retrieve detailed validator summary from agent_group.patch_history."
                        print(f"PhasePlanner: (Mock) Gathered validator_results_summary: '{validator_results_summary}'.")

                        # 4. Define branch_name and commit_title
                        # Ensure random is imported if not already: import random at the top of the file
                        branch_name = f"feature/auto-patch-{spec.issue_description[:20].replace(' ', '-').lower()}-{random.randint(1000,9999)}"
                        commit_title = f"Auto-apply patch for '{spec.issue_description[:40]}...'"
                        print(f"PhasePlanner: Defined branch: '{branch_name}', title: '{commit_title}'.")

                        # 5. Call CommitBuilder
                        print("PhasePlanner: Calling CommitBuilder.process_and_submit_patch...")
                        self.commit_builder.process_and_submit_patch(
                            validated_patch_content_map=validated_patch_content_map,
                            spec=spec,
                            diff_summary=diff_summary,
                            validator_results_summary=validator_results_summary,
                            branch_name=branch_name,
                            commit_title=commit_title,
                            project_root=self.project_root_path
                        )
                        print("PhasePlanner: CommitBuilder call finished.")
                        # For this subtask, break after processing the first successful patch
                        print("PhasePlanner: Breaking after first successful phase patch submission (mock).")
                        break
                except PhaseFailure as pf_e:
                    print(f"PhasePlanner: CollaborativeAgentGroup reported PhaseFailure for phase {phase_obj.operation_name}: {pf_e}")
                    execution_summary.append({"phase": phase_obj.operation_name, "status": "PhaseFailure", "error": str(pf_e)})
                    # Decide if we should stop the whole plan if one phase fails critically
                    print("PhasePlanner: Stopping plan execution due to PhaseFailure.")
                    break
                except Exception as e:
                    print(f"Error running agent group for phase {phase_obj.operation_name}: {type(e).__name__} - {e}")
                    import traceback
                    traceback.print_exc() # Print full traceback for debugging
                    execution_summary.append({"phase": phase_obj.operation_name, "status": "error", "error_message": str(e)})
                    # Optionally, break here:
                    # print("Stopping further phase execution due to error.")
                    # break

            print("--- CollaborativeAgentGroup execution finished for all phases (or broke early) ---")
            print("Execution Summary (from planner):")
            for summary_item in execution_summary:
                print(f"  - {summary_item}")
            # The method still returns the plan (List[Phase]).
            # The execution_summary is for logging/observation at this stage.
        else:
            print("PhasePlanner: Failed to generate a plan (beam search returned empty).")
        return best_plan # Return the list of Phase objects as per original signature

if __name__ == '__main__':
    # No MagicMock needed now as we are using more concrete (though still mock) classes
    # from unittest.mock import MagicMock
    print("--- PhasePlanner Example Usage (Conceptual) ---")

    # Using the actual RepositoryDigester mock defined at the class level for TYPE_CHECKING
    # This ensures methods like get_project_overview are available if called by agent_group
    class MockDigesterForPlannerMain(RepositoryDigester): # Inherit from the placeholder to ensure methods
        def __init__(self):
            super().__init__() # Call super if it has an init
            self.project_call_graph = {"module.func_a": {"module.func_b"}}
            self.project_control_dependence_graph = {}
            self.project_data_dependence_graph = {}
        # Override mock methods if more specific behavior is needed for main example
        def get_project_overview(self) -> Dict[str, Any]:
            print("__main__ MockDigester.get_project_overview called")
            return {"files_in_project": 5, "main_language": "python"}
        def get_file_content(self, path: Path) -> Optional[str]:
            content = f"# __main__ Mock Content for {path}\n# BAD_STYLE example\n# FAIL_VALIDATION example"
            print(f"__main__ MockDigester.get_file_content for {path} returning: '{content[:50]}...'")
            return content if path else None


    dummy_digester_main = MockDigesterForPlannerMain()

    dummy_style_path_main = Path("_temp_dummy_style_main.json")
    with open(dummy_style_path_main, "w") as f:
        json.dump({"line_length": 99, "preferred_quotes": "double", "indent": 4}, f)

    dummy_scorer_config_main = {"model_path": "path/to/phi2/model_placeholder", "enabled": True}

    dummy_naming_db_path_main = Path("_temp_dummy_naming_db.json")
    if not dummy_naming_db_path_main.exists():
        with open(dummy_naming_db_path_main, "w") as f:
            json.dump({"function_name_style": "snake_case"}, f)

    dummy_project_root_main = Path("_temp_mock_project_root")
    dummy_project_root_main.mkdir(parents=True, exist_ok=True)

    # Mock LLM model path for agent-based repair (can be a non-existent path for this test as LLM loading is mocked/guarded)
    mock_llm_repair_model_path = "_temp_mock_repair_model.gguf"


    try:
        # Ensure Spec and Phase are imported or defined
        # from .spec_model import Spec # (already imported at top)
        # from .phase_model import Phase # (already imported at top)

        planner = PhasePlanner(
            style_fingerprint_path=dummy_style_path_main,
            digester=dummy_digester_main,
            scorer_model_config=dummy_scorer_config_main,
            naming_conventions_db_path=dummy_naming_db_path_main,
            project_root_path=dummy_project_root_main,
            llm_model_path=mock_llm_repair_model_path, # New
            beam_width=2
        )

        # Example Spec using the imported spec_model.Spec structure
        example_spec_data_main = {
            "issue_description": "Implement new user registration flow", # Updated field name
            "target_files": ["src/services/user_service.py", "src/db/user_queries.py"], # Updated field name
            "operations": [ # This now matches spec_model.Spec's List[Dict[str,Any]]
                {"name": "add_import", "target_file": "src/services/user_service.py", "import_statement": "from ..db import user_queries"},
                {"name": "extract_method", "target_file": "src/services/user_service.py", "source_function_name": "register_user_v1", "start_line": 10, "end_line": 25, "new_method_name": "_validate_user_input"},
                {"name": "add_decorator", "target_file": "src/services/user_service.py", "target_function_name": "register_user_v2", "decorator_name": "@transactional"}
            ],
            "acceptance_tests": ["test_user_registration_success", "test_user_registration_existing_user"] # Updated field name
        }
        spec_instance_main = Spec(**example_spec_data_main)

        print(f"\nPlanning for spec: {spec_instance_main.issue_description}")
        plan_phases_list_main = planner.plan_phases(spec_instance_main)

        print("\n--- Post-execution: Generated Plan Phases (from planner return) ---")
        if plan_phases_list_main:
            for i, p_main in enumerate(plan_phases_list_main):
                print(f"Details for Planned Phase {i+1}:")
                if hasattr(p_main, 'model_dump_json'):
                    print(p_main.model_dump_json(indent=2))
                else:
                    # Fallback for older/non-Pydantic Phase models
                    print(f"  Operation: {getattr(p_main, 'operation_name', 'N/A')}, Target: {getattr(p_main, 'target_file', 'N/A')}, Params: {getattr(p_main, 'parameters', {})}")
        else:
            print("  No plan was generated by the planner.")

        # print(f"\nRequesting plan again for the same spec (should be cached):")
        # plan_phases_list_cached = planner.plan_phases(spec_instance_main)
        # print(f"Cached plan request returned {len(plan_phases_list_cached) if plan_phases_list_cached else 0} phases.")
        # Note: Agent group execution would re-run unless caching is implemented after agent execution.

    except ImportError as e_imp:
        print(f"ImportError in PhasePlanner __main__ example: {e_imp}. Check imports for Spec, Phase, or other dependencies.")
    except Exception as e_main:
        print(f"Error in PhasePlanner __main__ example: {type(e_main).__name__}: {e_main}")
        import traceback
        traceback.print_exc()
    finally:
        if dummy_style_path_main.exists():
            dummy_style_path_main.unlink()
        if dummy_naming_db_path_main.exists():
            dummy_naming_db_path_main.unlink()
        if dummy_project_root_main.exists():
            import shutil # Import shutil here for cleanup
            try:
                shutil.rmtree(dummy_project_root_main)
                print(f"Cleaned up dummy project root: {dummy_project_root_main}")
            except OSError as e_ose:
                print(f"Error removing directory {dummy_project_root_main}: {e_ose}")
    print("\n--- PhasePlanner Example Done ---")
