# src/profiler/orchestrator.py
from pathlib import Path
import json
import tempfile
from typing import Any, Dict

from .style_sampler import StyleSampler
from .sample_weigher import calculate_sample_weights, InputCodeSampleForWeighing # Use the defined input type
from .diffusion_interfacer import unify_fingerprints_with_diffusion


# Artifact generators
# Artifact generators
from .config_generator import (
    generate_black_config_in_pyproject,
    generate_ruff_config_in_pyproject,
    generate_docstring_template_file
)
from .database_setup import create_naming_conventions_db
from .naming_conventions import populate_naming_rules_from_profile


# Config
from src.utils.config_loader import load_app_config, DEFAULT_APP_CONFIG

# Formatter
from src.formatter import format_code

# Style Validator for testing scorer
from .style_validator import StyleValidatorCore

def create_dummy_project_for_profiling(base_path: Path, num_files: int = 3, lines_per_file: int = 10) -> Path:
    """Creates a small dummy Python project for testing the profiler orchestrator."""
    project_path = base_path / "dummy_profiler_project"
    project_path.mkdir(parents=True, exist_ok=True)

    for i in range(num_files):
        file_path = project_path / f"sample_file_{i+1}.py"
        content = []
        if i % 2 == 0: # Mix styles slightly
            content.append("class MyClassExample:")
            content.append("    def __init__(self, val):")
            content.append("        self.value = val  # Example line")
            content.append("    def get_value(self):")
            content.append('        return self.value')
        else:
            content.append("def another_function(arg_one, arg_two):")
            content.append('    """Docstring using double quotes."""')
            content.append("    result = arg_one + arg_two")
            content.append("    return result")

        # Add more lines to reach lines_per_file
        for j in range(len(content), lines_per_file):
            content.append(f"    pass # Line {j+1}")

        file_path.write_text("\n".join(content))
    return project_path

def run_phase1_style_profiling_pipeline(project_root: Path, app_config: dict) -> dict:
    """
    Orchestrates the main steps of the Phase 1 style profiling.
    """
    print(f"--- Running Phase 1 Style Profiling Pipeline for: {project_root} ---")

    # 1. Initialize StyleSampler
    # For StyleSampler, model paths are needed for its internal call to AI fingerprinting.
    # These will come from app_config.
    deepseek_model_p = app_config.get("models", {}).get("deepseek_style_draft_gguf", DEFAULT_APP_CONFIG["models"]["deepseek_style_draft_gguf"])
    divot5_refiner_p = app_config.get("models", {}).get("divot5_style_refiner_dir", DEFAULT_APP_CONFIG["models"]["divot5_style_refiner_dir"])

    # Ensure placeholder model directories exist if using default paths, to prevent errors in StyleSampler's dependencies
    # This is a workaround for this test environment; real models would exist.
    Path(deepseek_model_p).parent.mkdir(parents=True, exist_ok=True) # Ensure ./models exists
    if not Path(deepseek_model_p).exists() and "placeholder" in deepseek_model_p:
        Path(deepseek_model_p).touch() # Create empty placeholder file
    Path(divot5_refiner_p).mkdir(parents=True, exist_ok=True) # Create placeholder dir

    sampler = StyleSampler(
        repo_path=project_root,
        deepseek_model_path=deepseek_model_p,
        divot5_model_path=divot5_refiner_p, # This is for the refiner in the sampler's chain
        target_samples=100 # Keep low for dummy project
    )

    # 2. Collect code elements with AI fingerprints
    # sample_elements() calls collect_all_elements() which calls _extract_elements_from_file()
    # _extract_elements_from_file() now orchestrates the calls to llm_interfacer functions.
    print("\n--- 1. Sampling Code Elements and Generating AI Fingerprints ---")
    code_samples_with_ai_fp = sampler.sample_elements() # List[CodeSample]

    if not code_samples_with_ai_fp:
        print("No code samples were collected or fingerprinted. Aborting pipeline.")
        return {"error": "No samples collected/fingerprinted."}

    print(f"Collected and fingerprinted {len(code_samples_with_ai_fp)} code elements.")
    for i, sample in enumerate(code_samples_with_ai_fp[:2]): # Print first few
        print(f"  Sample {i+1}: {sample.item_name} ({sample.file_path.name}), Fingerprint Status: {sample.ai_fingerprint.get('validation_status') if sample.ai_fingerprint else 'N/A'}")
        if sample.ai_fingerprint and sample.ai_fingerprint.get('error'):
            print(f"    Error in fingerprint: {sample.ai_fingerprint['error']}")


    # 3. Convert CodeSample objects to InputCodeSampleForWeighing for SampleWeigher
    # and filter out samples where AI fingerprinting failed badly
    samples_for_weigher = []
    for cs in code_samples_with_ai_fp:
        if cs.ai_fingerprint and cs.ai_fingerprint.get("validation_status") == "passed":
            # Create the input object expected by calculate_sample_weights
            samples_for_weigher.append(
                InputCodeSampleForWeighing(
                    file_path=cs.file_path,
                    ai_fingerprint=cs.ai_fingerprint, # This is the dict from CodeSample
                    file_size_kb=cs.file_size_kb,
                    mod_timestamp=cs.mod_timestamp
                )
            )
        elif cs.ai_fingerprint: # Fingerprinting attempted but may have failed validation or had errors
            print(f"Warning: Skipping sample {cs.item_name} from {cs.file_path.name} for weighting due to fingerprint status: {cs.ai_fingerprint.get('validation_status')}, error: {cs.ai_fingerprint.get('error')}")


    if not samples_for_weigher:
        print("No valid AI fingerprints available after filtering. Cannot proceed to weighting and unification.")
        return {"error": "No valid AI fingerprints for weighting."}

    # 4. Calculate sample weights
    print("\n--- 2. Calculating Sample Weights ---")
    weighted_fingerprints_for_unifier = calculate_sample_weights(samples_for_weigher)
    print(f"Calculated weights for {len(weighted_fingerprints_for_unifier)} samples.")
    # weighted_fingerprints_for_unifier is List[Dict[str, Any]] where each dict has "fingerprint", "file_path", "weight"

    if not weighted_fingerprints_for_unifier:
        print("No samples after weighting. Aborting.")
        return {"error": "No samples after weighting process."}

    # 5. Unify fingerprints using DiffusionInterfacer
    # The diffusion_interfacer.unify_fingerprints_with_diffusion expects a list of dicts,
    # where each dict is a per-sample fingerprint that also includes 'file_path' and 'weight'.
    # The 'fingerprint' key in these dicts should hold the actual style parameters.
    # The `weighted_fingerprints_for_unifier` from `calculate_sample_weights` matches this.
    print("\n--- 3. Unifying Fingerprints with Diffusion Model (Placeholder/Actual) ---")
    divot5_unifier_model_p = app_config.get("models", {}).get("divot5_style_unifier_dir", DEFAULT_APP_CONFIG["models"]["divot5_style_unifier_dir"])
    Path(divot5_unifier_model_p).mkdir(parents=True, exist_ok=True) # Ensure placeholder dir exists

    unified_profile_dict = unify_fingerprints_with_diffusion(
        per_sample_fingerprints=weighted_fingerprints_for_unifier,
        model_path=divot5_unifier_model_p,
        verbose=app_config.get("general", {}).get("verbose", False)
    )

    print("\n--- 4. Unified Style Profile ---")
    print(json.dumps(unified_profile_dict, indent=2))

    if unified_profile_dict.get("error"):
        print(f"ERROR in unification: {unified_profile_dict['error']}")
        # Skip artifact generation if unification failed
        return unified_profile_dict

    print("\n--- 5. Generating Profiler Artifacts ---")
    pyproject_toml_path = project_root / "pyproject.toml"
    # For docstring and naming_db, ensure 'config' subdir exists if generator functions expect it.
    # For simplicity as per prompt, placing them at project_root for now.
    # If generators create config/ subdir, adjust path or ensure dir creation.
    # Assuming config_generator functions handle parent dir creation for pyproject.toml.
    # The database_setup and naming_conventions usually handle their own paths or take explicit ones.

    docstring_template_output_path = project_root / "docstring_template.py"
    naming_db_path = project_root / "naming_conventions.db"

    # Ensure parent directories for these artifacts if they are nested
    # (though for project_root direct children, project_root itself must exist)
    # For example, if docstring_template_path was project_root / "config" / "docstring_template.py":
    # docstring_template_output_path.parent.mkdir(parents=True, exist_ok=True)
    # naming_db_path.parent.mkdir(parents=True, exist_ok=True) # If also nested

    try:
        print(f"  Generating Naming Conventions DB at: {naming_db_path}")
        create_naming_conventions_db(db_path=naming_db_path) # This function should handle if DB already exists
        populate_naming_rules_from_profile(db_path=naming_db_path, unified_profile=unified_profile_dict)
        print("  Naming Conventions DB generated and populated successfully.")

        print(f"  Generating Black config in: {pyproject_toml_path}")
        generate_black_config_in_pyproject(unified_profile=unified_profile_dict, pyproject_path=pyproject_toml_path)
        print("  Black config generated successfully.")

        print(f"  Generating Ruff config in: {pyproject_toml_path}")
        generate_ruff_config_in_pyproject(unified_profile=unified_profile_dict, pyproject_path=pyproject_toml_path, db_path=naming_db_path)
        print("  Ruff config generated successfully.")

        print(f"  Generating Docstring template at: {docstring_template_output_path}")
        generate_docstring_template_file(unified_profile=unified_profile_dict, template_output_path=docstring_template_output_path)
        print("  Docstring template generated successfully.")

    except Exception as e_artifacts:
        print(f"ERROR during artifact generation: {e_artifacts}")
        # Optionally, add this error to unified_profile_dict or handle differently
        unified_profile_dict["artifact_generation_error"] = str(e_artifacts)

    return unified_profile_dict

def test_formatter_on_dummy_file(project_root: Path, unified_profile: Dict[str, Any], app_config: Dict[str, Any]):
    """
    Tests the format_code function on a specified dummy file within the project_root.
    Assumes artifacts like pyproject.toml and naming_conventions.db have been generated.
    """
    print("\n--- Testing Formatter on Dummy File ---")

    naming_db_path = project_root / "naming_conventions.db"
    # pyproject_toml_path = project_root / "pyproject.toml" # Not directly passed to format_code anymore

    # Assuming sample_file_1.py is created by create_dummy_project_for_profiling
    # (it creates sample_file_{i+1}.py, so sample_file_1.py is the first one if num_files >= 1)
    test_file_to_format = project_root / "sample_file_1.py"

    if not test_file_to_format.exists():
        print(f"Error: Test file {test_file_to_format} does not exist. Skipping formatter test.")
        return

    print(f"Formatter Test Target: {test_file_to_format}")

    try:
        original_content = test_file_to_format.read_text(encoding="utf-8")
        print("\nContent BEFORE formatting:")
        print("-------------------------")
        print(original_content)
        print("-------------------------")

        # The 'profile' arg to format_code is currently placeholderish in formatter.py,
        # but passing unified_profile is fine.
        # app_config is not directly used by format_code, but good to have if its deps use it.
        format_code_success = format_code(
            file_path=test_file_to_format,
            profile=unified_profile,
            project_root=project_root, # format_code now expects project_root
            db_path=naming_db_path
        )

        if format_code_success:
            print("\nformat_code reported SUCCESS.")
        else:
            print("\nformat_code reported FAILURE.")

        formatted_content = test_file_to_format.read_text(encoding="utf-8")
        print("\nContent AFTER formatting:")
        print("-------------------------")
        print(formatted_content)
        print("-------------------------")

        if original_content == formatted_content:
            print("Note: Content unchanged. This might be expected or indicate tools need specific rules.")
        else:
            print("Note: Content was modified by the formatter.")
        print("Note: Actual formatting quality depends on the generated pyproject.toml (from unified_profile),")
        print("the rules in naming_conventions.db, and the behavior of Black/Ruff/Renamer.")

    except Exception as e:
        print(f"Error during formatter test: {e}")

def test_style_scorer_on_dummy_file(project_root: Path, unified_profile: Dict[str, Any], app_config: Dict[str, Any]):
    """
    Tests the StyleValidatorCore.score_sample_style method on a dummy file.
    Assumes artifacts like naming_conventions.db have been generated.
    """
    print("\n--- Testing Style Scorer on Dummy File ---")

    naming_db_path = project_root / "naming_conventions.db"
    test_file_to_score = project_root / "sample_file_1.py" # Consistent with formatter test

    if not test_file_to_score.exists():
        print(f"Error: Test file {test_file_to_score} does not exist. Skipping style scorer test.")
        return

    if not naming_db_path.exists():
        print(f"Warning: Naming conventions DB {naming_db_path} not found. Naming checks in scorer will be skipped.")
        # db_path is Optional for score_sample_style, so proceed without it.

    print(f"Style Scorer Test Target: {test_file_to_score}")

    try:
        style_validator = StyleValidatorCore(app_config=app_config, style_profile=unified_profile)

        # Ensure db_path is None if the file doesn't exist, as score_sample_style expects Path or None
        effective_db_path = naming_db_path if naming_db_path.exists() else None

        score = style_validator.score_sample_style(
            sample_path=test_file_to_score,
            db_path=effective_db_path
        )

        print(f"\nStyleValidatorCore.score_sample_style returned score: {score:.4f} for {test_file_to_score.name}")
        print("Note: Score accuracy depends on the unified_profile, naming_conventions.db rules (if found),")
        print("and the completeness of the scoring logic in StyleValidatorCore (e.g., tool paths in app_config).")

    except Exception as e:
        print(f"Error during style scorer test: {e}")


if __name__ == "__main__":
    # Create a temporary directory for the dummy project
    with tempfile.TemporaryDirectory(prefix="phase1_orch_test_") as tmpdir:
        temp_project_root = Path(tmpdir)

        # Create a dummy project within the temporary directory
        dummy_project_path = create_dummy_project_for_profiling(temp_project_root, num_files=5, lines_per_file=20)

        print(f"Dummy project created at: {dummy_project_path}")

        # Load app_config (will use defaults with placeholder model paths)
        config = load_app_config()
        config["general"]["verbose"] = True # Enable verbose for testing
        # Override project_root in config if needed by any deeper part, though sampler takes it directly
        config["general"]["project_root"] = str(dummy_project_path)


        # Run the pipeline (which now includes artifact generation)
        final_profile = run_phase1_style_profiling_pipeline(dummy_project_path, config)

        print("\n--- Pipeline Execution Finished ---")
        if final_profile.get("error") or final_profile.get("artifact_generation_error"):
            print("Pipeline completed with error(s):")
            if final_profile.get("error"):
                print(f"  - Profiling error: {final_profile['error']}")
            if final_profile.get("artifact_generation_error"):
                print(
                    f"  - Artifact generation error: {final_profile['artifact_generation_error']}"
                )
            print("Formatter and Style Scorer tests skipped due to errors in profile generation or artifact creation.")
        else:
            print("Pipeline completed successfully. Final Unified Profile:")
            print(json.dumps(final_profile, indent=2, sort_keys=True))

            # Test the formatter if pipeline was successful
            test_formatter_on_dummy_file(dummy_project_path, final_profile, config)

            # Test the style scorer if pipeline was successful
            test_style_scorer_on_dummy_file(dummy_project_path, final_profile, config)

    print("\nOrchestrator main finished.")
