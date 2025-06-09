# src/planner/phase_planner.py
import difflib # For generating diff summaries
from typing import List, Dict, Any, Union, Optional, TYPE_CHECKING, Type, Tuple, Callable
from pathlib import Path
import json
import hashlib
import random # For dummy scorer and branch names

from src.transformer import apply_libcst_codemod_script, PatchApplicationError # For real patch application

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
from src.builder.commit_builder import CommitBuilder
from src.agent_group.exceptions import PhaseFailure
from src.profiler.llm_interfacer import get_llm_score_for_text # New import

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
        scorer_model_config: Any,
        naming_conventions_db_path: Path,
        project_root_path: Path,
        llm_model_path: Optional[str] = None,
        refactor_op_map: Optional[Dict[str, BaseRefactorOperation]] = None,
        beam_width: int = 3,
        verbose: bool = False # Added verbose flag
    ):
        self.style_fingerprint: Dict[str, Any] = {}
        self.verbose = verbose # Store verbose flag
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

        # Scorer configuration logging
        if self.scorer_model_config and self.scorer_model_config.get("enabled", True):
            scorer_model_path = self.scorer_model_config.get("model_path")
            if scorer_model_path and Path(scorer_model_path).is_file():
                 print(f"PhasePlanner Info: Real LLM scorer configured with model: {scorer_model_path}.")
                 # self.phi2_scorer attribute is no longer needed for primary logic
            elif scorer_model_path: # Path provided but not found
                 print(f"PhasePlanner Warning: Scorer GGUF model_path '{scorer_model_path}' not found. Scoring will use fallback logic within _score_candidate_plan_with_llm.")
            else: # No path provided
                 print("PhasePlanner Warning: LLM scorer enabled but no 'model_path' provided in scorer_model_config. Scoring will use placeholder/fallback logic within _score_candidate_plan_with_llm.")
        else:
            print("PhasePlanner Info: LLM scorer is disabled in config or no scorer_model_config provided. Plan scoring will use random fallback.")

        # Import and Initialize CollaborativeAgentGroup
        from src.agent_group.collaborative_agent_group import CollaborativeAgentGroup

        # Initialize Validator
        self.validator = Validator(config=None) # Pass config if PhasePlanner has one for Validator
        print("PhasePlanner: Validator instance initialized.")

        # Import and Initialize CollaborativeAgentGroup
        from src.agent_group.collaborative_agent_group import CollaborativeAgentGroup
        # Prepare a common config for LLM agents if specific settings are needed.
        # For now, it will primarily carry the llm_model_path for both cores.
        # Other parameters like n_gpu_layers, n_ctx, verbose can be added here if they
        # should be controlled from PhasePlanner's initialization.
        # DiffusionCore.expand_scaffold expects:
        # "infill_model_path", "llm_model_path", "verbose", "n_gpu_layers", "n_ctx",
        # "max_tokens_for_infill", "temperature", "stop_sequences_for_infill"
        # LLMCore.propose_repair_diff also uses similar params from its config.

        # Let's assume llm_model_path is the primary model for all agent tasks for now.
        # Specific configs can override this if needed.
        common_agent_llm_config = {
            "llm_model_path": llm_model_path, # Path to the GGUF model for general agent tasks
            "infill_model_path": llm_model_path, # Explicitly set for DiffusionCore, can be overridden by more specific config
            "verbose": False, # Default verbosity for LLM operations
            "n_gpu_layers": -1, # Default: all layers to GPU if possible
            "n_ctx": 4096,      # Default context size
            # Add other common parameters if they are available or make sense at PhasePlanner level
            # "temperature": 0.4, # Example, might be too specific here
            # "max_tokens_for_infill": 512, # Example
        }
        # If llm_core_config or diffusion_core_config are ever passed as actual dicts to PhasePlanner,
        # they could be merged with common_agent_llm_config here.

        self.agent_group: 'CollaborativeAgentGroup' = CollaborativeAgentGroup(
            style_profile=self.style_fingerprint,
            naming_conventions_db_path=self.naming_conventions_db_path,
            validator_instance=self.validator,
            llm_core_config=common_agent_llm_config.copy(), # Pass a copy for LLMCore
            diffusion_core_config=common_agent_llm_config.copy(), # Pass a copy for DiffusionCore
            llm_model_path=llm_model_path # This is specifically for LLMCore's repair GGUF model path, distinct from general task model
        )
        print(f"PhasePlanner: CollaborativeAgentGroup initialized with common_agent_llm_config: {common_agent_llm_config}")

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
        # Initialize stats with defaults
        num_cg_nodes, num_cg_edges, cg_density = 0, 0, 0.0
        num_cdg_nodes, num_cdg_edges, cdg_density = 0, 0, 0.0
        num_ddg_nodes, num_ddg_edges, ddg_density = 0, 0, 0.0

        # Call Graph (CG) statistics
        if hasattr(self.digester, 'project_call_graph') and self.digester.project_call_graph:
            cg = self.digester.project_call_graph
            num_cg_nodes = len(cg)
            num_cg_edges = sum(len(edges) for edges in cg.values())
            if num_cg_nodes > 1:
                cg_density = num_cg_edges / (num_cg_nodes * (num_cg_nodes - 1))

        # Control Dependence Graph (CDG) statistics
        if hasattr(self.digester, 'project_control_dependence_graph') and self.digester.project_control_dependence_graph:
            cdg = self.digester.project_control_dependence_graph
            num_cdg_nodes = len(cdg)
            # Assuming CDG values are lists/sets of dependent nodes (edges)
            num_cdg_edges = sum(len(dependents) for dependents in cdg.values())
            if num_cdg_nodes > 1:
                cdg_density = num_cdg_edges / (num_cdg_nodes * (num_cdg_nodes - 1))

        # Data Dependence Graph (DDG) statistics
        if hasattr(self.digester, 'project_data_dependence_graph') and self.digester.project_data_dependence_graph:
            ddg = self.digester.project_data_dependence_graph
            num_ddg_nodes = len(ddg)
            # Assuming DDG values are lists/sets of dependent nodes (edges)
            num_ddg_edges = sum(len(dependents) for dependents in ddg.values())
            if num_ddg_nodes > 1:
                ddg_density = num_ddg_edges / (num_ddg_nodes * (num_ddg_nodes - 1))

        graph_stats = {
            "num_call_graph_nodes": num_cg_nodes,
            "num_call_graph_edges": num_cg_edges,
            "cg_density": cg_density,
            "num_cdg_nodes": num_cdg_nodes,
            "num_cdg_edges": num_cdg_edges,
            "cdg_density": cdg_density,
            "num_ddg_nodes": num_ddg_nodes,
            "num_ddg_edges": num_ddg_edges,
            "ddg_density": ddg_density,
        }

        print(f"PhasePlanner: Generated graph stats for scorer: {graph_stats}")
        return graph_stats

    def _score_candidate_plan_with_llm(self, candidate_plan: List[Phase], graph_stats: Dict[str, Any]) -> float:
        # Check if LLM scoring is enabled and configured
        if not self.scorer_model_config or not self.scorer_model_config.get("enabled", True):
            print("PhasePlanner Info: Real LLM scorer is disabled in config. Falling back to random scoring.")
            return random.uniform(0.2, 0.8)

        scorer_model_path = self.scorer_model_config.get("model_path")
        # Default to a placeholder if path not provided, get_llm_score_for_text will handle actual existence check.
        if not scorer_model_path:
             scorer_model_path = "./models/placeholder_scorer.gguf"
             print(f"PhasePlanner Warning: No 'model_path' in scorer_model_config. Defaulting to placeholder '{scorer_model_path}' for get_llm_score_for_text.")

        # Check if the placeholder path actually exists if it's the one being used.
        # get_llm_score_for_text also does this, but an early check here can provide clearer context for fallback.
        if scorer_model_path == "./models/placeholder_scorer.gguf" and not Path(scorer_model_path).is_file():
            print(f"PhasePlanner Warning: Default scorer model placeholder '{scorer_model_path}' not found. Falling back to random scoring.")
            return random.uniform(0.2, 0.8)

        if not candidate_plan:
            return 0.05 # Return a low score for an empty plan

        # Prepare prompt components (similar to existing logic)
        plan_str_parts = []
        for i, phase in enumerate(candidate_plan):
            try: params_json = json.dumps(phase.parameters)
            except TypeError: params_json = str(phase.parameters)
            plan_str_parts.append(
                f"Phase {i + 1} (Operation: {phase.operation_name}):\n"
                f"  Target File: {phase.target_file or 'N/A'}\n"
                f"  Parameters: {params_json}\n"
                f"  Description: {phase.description}"
            )
        plan_formatted_str = "\n\n".join(plan_str_parts)

        try: style_fp_str = json.dumps(self.style_fingerprint, indent=2)
        except TypeError: style_fp_str = str(self.style_fingerprint)
        try: graph_stats_str = json.dumps(graph_stats, indent=2)
        except TypeError: graph_stats_str = str(graph_stats)

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
Output ONLY the numerical score as a float (e.g., 0.75).
Score:"""

        # Get LLM parameters from config, with defaults
        llm_verbose = self.scorer_model_config.get("verbose", False)
        llm_n_gpu_layers = self.scorer_model_config.get("n_gpu_layers", -1)
        llm_n_ctx = self.scorer_model_config.get("n_ctx", 2048) # Default from get_llm_score_for_text
        llm_temperature = self.scorer_model_config.get("temperature", 0.1) # Default from get_llm_score_for_text
        max_tokens_score = self.scorer_model_config.get("max_tokens_for_score", 16)

        # Call the new LLM interfacer function
        llm_score = get_llm_score_for_text(
            model_path=scorer_model_path,
            prompt=prompt,
            verbose=llm_verbose,
            n_gpu_layers=llm_n_gpu_layers,
            n_ctx=llm_n_ctx,
            max_tokens_for_score=max_tokens_score,
            temperature=llm_temperature
        )

        if llm_score is not None:
            # Optional: Validate or clamp score to expected range [0.0, 1.0]
            if not (0.0 <= llm_score <= 1.0):
                print(f"PhasePlanner Warning: LLM score {llm_score:.4f} is outside the expected [0.0, 1.0] range. Clamping.")
                llm_score = max(0.0, min(1.0, llm_score))
            print(f"PhasePlanner: LLM Scorer returned score: {llm_score:.4f}")
            return llm_score
        else:
            print("PhasePlanner Warning: Failed to get score from LLM via get_llm_score_for_text. Falling back to default low score for this candidate.")
            return 0.1 # Default low score on failure

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

                        print("\nPhasePlanner: Phase successful. Preparing for CommitBuilder integration.")

                        # 1. Real Patch Application & Diff Summary Generation
                        target_file_path_str = phase_obj.target_file
                        target_file_path = Path(target_file_path_str) if target_file_path_str else None
                        modified_content: Optional[str] = None
                        diff_summary = "Diff generation skipped: No target file, patch script, or patch application failed." # Default

                        if not target_file_path:
                            print("PhasePlanner Warning: Phase target_file is None. Cannot apply patch or proceed with CommitBuilder for this phase.")
                            execution_summary.append({"phase_operation": phase_obj.operation_name, "target": "None", "status": "error", "error_message": "Target file was None"})
                            continue

                        # Ensure target_file_path is relative for map keys, but absolute for digester.get_file_content
                        abs_target_file_path = self.project_root_path / target_file_path

                        if validated_patch: # validated_patch is the LibCST script string
                            original_content = self.digester.get_file_content(abs_target_file_path)
                            if original_content is None:
                                original_content = ""
                                if self.verbose: print(f"PhasePlanner: Target file '{abs_target_file_path}' is new or content not found. Applying patch to empty content.")

                            try:
                                if self.verbose: print(f"PhasePlanner: Applying validated LibCST script to '{abs_target_file_path}'...")
                                modified_content = apply_libcst_codemod_script(original_content, validated_patch)

                                if self.verbose: print(f"PhasePlanner: LibCST script applied successfully to '{target_file_path}'.")

                                diff_lines = list(difflib.unified_diff(
                                    original_content.splitlines(keepends=True),
                                    modified_content.splitlines(keepends=True),
                                    fromfile=f"a/{target_file_path.name}", # Use relative name for diff
                                    tofile=f"b/{target_file_path.name}"
                                ))
                                if diff_lines:
                                    diff_summary = "".join(diff_lines)
                                    if self.verbose: print(f"PhasePlanner: Generated diff summary for '{target_file_path}' (length {len(diff_summary)}).")
                                else:
                                    diff_summary = f"No textual changes detected by difflib for '{target_file_path}' after patch application."
                                    if self.verbose: print(f"PhasePlanner: {diff_summary}")

                            except PatchApplicationError as pae:
                                print(f"PhasePlanner Error: Failed to apply validated patch script for {abs_target_file_path}: {pae}")
                                execution_summary.append({"phase_operation": phase_obj.operation_name, "target": target_file_path_str, "status": "error", "error_message": f"Patch application failed: {pae}"})
                                continue
                            except Exception as e_apply:
                                print(f"PhasePlanner Error: Unexpected error during patch application or diff for {abs_target_file_path}: {e_apply}")
                                execution_summary.append({"phase_operation": phase_obj.operation_name, "target": target_file_path_str, "status": "error", "error_message": f"Unexpected patch application/diff error: {e_apply}"})
                                continue
                        else: # No validated_patch script from agent_group
                            print(f"PhasePlanner: No validated patch script from agent group for phase targeting '{target_file_path}'. Skipping CommitBuilder.")
                            execution_summary.append({"phase_operation": phase_obj.operation_name, "target": target_file_path_str, "status": "skipped", "reason": "No script from agent"})
                            continue

                        if modified_content is None:
                            print(f"PhasePlanner: Skipping CommitBuilder for phase targeting '{target_file_path}' due to patch application failure or no script.")
                            continue

                        # Use relative path for the map key
                        validated_patch_content_map = {target_file_path: modified_content}
                        print(f"PhasePlanner: Created validated_patch_content_map for CommitBuilder. Key: '{target_file_path}'")

                        # 3. Validator Results Summary (already good from previous steps)
                        validator_results_summary = "Mock validator results: All checks passed on final attempt by agent group." # Default
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
