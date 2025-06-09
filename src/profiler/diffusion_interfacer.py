import random
from typing import Dict, List, Union, Optional, Any, Literal
from dataclasses import dataclass, field
from collections import Counter, defaultdict
import json
import re
from pathlib import Path

# Guard Transformers/Torch import
try:
    import torch
    from transformers import T5ForConditionalGeneration, T5TokenizerFast
except ImportError:
    torch = None # type: ignore
    T5ForConditionalGeneration = None # type: ignore
    T5TokenizerFast = None # type: ignore
    print("Warning: PyTorch or Hugging Face Transformers not found. DivoT5 model interaction will be disabled.")

@dataclass
class UnifiedStyleProfile:
    indent_width: Optional[int] = None
    preferred_quotes: Optional[Literal["single", "double"]] = None
    docstring_style: Optional[Literal["google", "numpy", "epytext", "restructuredtext", "plain", "other"]] = None
    max_line_length: Optional[int] = None
    identifier_snake_case_pct: Optional[float] = None
    identifier_camelCase_pct: Optional[float] = None
    identifier_UPPER_SNAKE_CASE_pct: Optional[float] = None
    directory_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    confidence_score: Optional[float] = None
    raw_analysis_summary: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "indent_width": self.indent_width,
            "preferred_quotes": self.preferred_quotes,
            "docstring_style": self.docstring_style,
            "max_line_length": self.max_line_length,
            "identifier_snake_case_pct": self.identifier_snake_case_pct,
            "identifier_camelCase_pct": self.identifier_camelCase_pct,
            "identifier_UPPER_SNAKE_CASE_pct": self.identifier_UPPER_SNAKE_CASE_pct,
            "directory_overrides": self.directory_overrides,
            "confidence_score": self.confidence_score,
            "raw_analysis_summary": self.raw_analysis_summary,
            "error": self.error
        }

def unify_fingerprints_with_diffusion(
    per_sample_fingerprints: List[Dict[str, Any]],
    model_path: Optional[str] = None,
    device: Optional[str] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Unifies multiple per-sample style fingerprints into a single project-level profile using DivoT5.
    Returns a dictionary representing the UnifiedStyleProfile or an error dictionary.
    """
    error_dict_template: Dict[str, Any] = UnifiedStyleProfile(error="Initial error state").to_dict()

    # These are keys the DivoT5 unifier model is expected to understand or produce at the top level.
    # It might internally handle sub-dictionaries like 'profile' within 'directory_overrides'.
    # This set is for validating the *final* structure against UnifiedStyleProfile.
    final_expected_keys = set(UnifiedStyleProfile().to_dict().keys()) - {"error"} # Exclude 'error' for content validation

    if T5ForConditionalGeneration is None or T5TokenizerFast is None or torch is None:
        error_msg = "Transformers/PyTorch not installed. Cannot use DivoT5 for unification."
        print(f"Profiler Error: {error_msg}")
        return {**error_dict_template, "error": error_msg}

    resolved_model_path_str = model_path
    is_placeholder_path = False
    if not model_path or model_path == "path/to/your/divot5_model_dir": # Catch old placeholder
        resolved_model_path_str = "./models/placeholder_divot5_unifier/"
        is_placeholder_path = True
        print(f"Profiler Warning: DivoT5 unifier model path not provided or is a default placeholder. Using standard placeholder: '{resolved_model_path_str}'.")

    resolved_model_path = Path(resolved_model_path_str)
    if not resolved_model_path.exists() or not resolved_model_path.is_dir():
        error_msg = f"DivoT5 unifier model directory does not exist at resolved path: {resolved_model_path}"
        if is_placeholder_path:
            error_msg += " (This is a placeholder. Please provide a valid DivoT5 model directory or place it at './models/placeholder_divot5_unifier/')"
        print(f"Profiler Error: {error_msg}")
        return {**error_dict_template, "error": error_msg}

    selected_device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

    log_prefix = "Profiler (DivoT5 Unifier):"
    if verbose: print(f"{log_prefix} Attempting to load model from {resolved_model_path} onto device: {selected_device}")
    else: print(f"{log_prefix} Loading model from {resolved_model_path}...")

    try:
        tokenizer = T5TokenizerFast.from_pretrained(str(resolved_model_path))
        model = T5ForConditionalGeneration.from_pretrained(str(resolved_model_path)).to(selected_device)
        model.eval()
    except Exception as e:
        error_msg = f"Error loading DivoT5 unifier model from {resolved_model_path}: {e}"
        print(f"{log_prefix} Error: {error_msg}")
        return {**error_dict_template, "error": error_msg}

    try:
        valid_samples_for_json = []
        for sample in per_sample_fingerprints:
            if isinstance(sample, dict) and isinstance(sample.get('fingerprint'), dict):
                valid_samples_for_json.append(sample)
            else:
                print(f"{log_prefix} Warning: Skipping invalid sample structure for JSON serialization: {type(sample)}")
        if not valid_samples_for_json:
            error_msg = "No valid per-sample fingerprints to serialize for DivoT5 input."
            print(f"{log_prefix} Error: {error_msg}")
            return {**error_dict_template, "error": error_msg}
        serialized_fingerprints = json.dumps(valid_samples_for_json, indent=2)
    except TypeError as te:
        error_msg = f"Failed to serialize per_sample_fingerprints to JSON: {te}"
        print(f"{log_prefix} Error: {error_msg}")
        return {**error_dict_template, "error": error_msg}

    system_prompt = (
        "You are an expert style analysis assistant. Your task is to unify a list of "
        "per-sample style fingerprints into a single, coherent project-level UnifiedStyleProfile. "
        "Each sample fingerprint includes a 'fingerprint' dictionary (with keys like 'indent', 'quotes', 'linelen', 'snake_pct', 'camel_pct', 'screaming_pct', 'docstyle'), "
        "'file_path' (string), and 'weight' (float). "
        "Consider the weights when averaging or deciding on dominant styles. "
        "Infer directory-specific overrides ('directory_overrides') if you detect consistent "
        "sub-styles in specific paths (e.g., 'tests/' might have different line length or docstring style). "
        "The output must be a valid JSON object representing the UnifiedStyleProfile. "
        "Ensure the output JSON contains AT LEAST the following keys for the main profile: "
        "'indent_width', 'preferred_quotes', 'max_line_length', 'identifier_snake_case_pct', "
        "'identifier_camelCase_pct', 'identifier_UPPER_SNAKE_CASE_pct', 'docstring_style', 'target_python_version'. "
        "The 'directory_overrides' should be a list of objects, each with 'path_prefix' (string) and "
        "'profile' (a dictionary with style keys similar to the main profile)."
    )
    input_text = f"{system_prompt}\n\nPer-sample fingerprints JSON:\n{serialized_fingerprints}"

    if verbose: print(f"{log_prefix} Input text (first 1000 chars of serialized fingerprints):\n{system_prompt}\n\nPer-sample fingerprints JSON:\n{serialized_fingerprints[:1000]}...")

    unified_profile_json_str = "{}"
    try:
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=4096).to(selected_device)
        output_max_length = 1024

        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids, max_length=output_max_length,
                num_beams=5, early_stopping=True, temperature=0.7
            )

        if outputs is not None and len(outputs) > 0:
            unified_profile_json_str = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            if verbose: print(f"{log_prefix} Raw output: '{unified_profile_json_str}'")
        else:
            error_msg = "DivoT5 Unifier model returned an empty or unexpected response."
            print(f"{log_prefix} Warning: {error_msg}")
            return {**error_dict_template, "error": error_msg}
    except Exception as e:
        error_msg = f"Error during DivoT5 Unifier model inference: {e}"
        print(f"{log_prefix} Error: {error_msg}")
        return {**error_dict_template, "error": error_msg}

    try:
        json_match = re.search(r'\{.*\}', unified_profile_json_str, re.DOTALL)
        if json_match:
            unified_profile_json_str = json_match.group(0)
            if verbose: print(f"{log_prefix} Extracted JSON from LLM output: {unified_profile_json_str[:200]}...")

        parsed_output = json.loads(unified_profile_json_str)

        if not isinstance(parsed_output, dict):
            error_msg = f"DivoT5 Unifier output parsed to JSON but is not a dictionary: {type(parsed_output)}"
            print(f"{log_prefix} Warning: {error_msg}. Raw output: {unified_profile_json_str}")
            return {**error_dict_template, "error": error_msg, "details": unified_profile_json_str}

        # Normalize potential naming convention key variations from LLM output
        # to match UnifiedStyleProfile dataclass fields.
        key_mappings = {
            "naming_convention_snake_case_weight": "identifier_snake_case_pct",
            "naming_convention_camel_case_weight": "identifier_camelCase_pct",
            "screaming_pct": "identifier_UPPER_SNAKE_CASE_pct", # if from DeepSeek-like draft
            "naming_convention_screaming_snake_case_weight": "identifier_UPPER_SNAKE_CASE_pct"
        }
        for old_key, new_key in key_mappings.items():
            if old_key in parsed_output:
                parsed_output[new_key] = parsed_output.pop(old_key)

        # Ensure all expected keys for UnifiedStyleProfile are present, fill with None if not.
        # This helps in creating a consistent structure even if LLM omits some fields.
        final_result = {}
        for key in final_expected_keys:
            final_result[key] = parsed_output.get(key) # Will be None if key is missing

        # Specifically ensure directory_overrides is a dict (or list as per Pydantic model)
        # The prompt asks for a list of objects. Pydantic model uses Dict[str, Dict[str, Any]]
        # This might need adjustment based on how DivoT5 actually structures it.
        # For now, if it's there and a list, assume it's list of {"path_prefix": ..., "profile": ...}
        # If model outputs dict for directory_overrides, it's fine for UnifiedStyleProfile.
        if "directory_overrides" in parsed_output and isinstance(parsed_output["directory_overrides"], list):
            # Convert list of {"path_prefix": X, "profile": Y} to Dict[str, Dict] if needed by Pydantic model
            # However, UnifiedStyleProfile dataclass has it as Dict[str, Dict[str, Any]]
            # This part might need careful alignment with actual LLM output vs. Pydantic model.
            # For this refactor, let's assume model output for directory_overrides is Dict[str, Dict]
            # or can be directly used if it's a list that Pydantic can handle for Dict conversion.
             final_result["directory_overrides"] = parsed_output.get("directory_overrides", {})
        elif "directory_overrides" not in final_result:
             final_result["directory_overrides"] = {}


        final_result["error"] = None
        print(f"{log_prefix} DivoT5 unification successful.")
        return final_result

    except json.JSONDecodeError as je:
        error_msg = f"DivoT5 Unifier output was not valid JSON: '{unified_profile_json_str}'. Error: {je}"
        print(f"{log_prefix} Warning: {error_msg}")
        return {**error_dict_template, "error": error_msg, "details": unified_profile_json_str}
    except Exception as e_gen:
        error_msg = f"Unexpected error processing DivoT5 Unifier output: {e_gen}. Raw output: '{unified_profile_json_str}'"
        print(f"{log_prefix} Warning: {error_msg}")
        return {**error_dict_template, "error": error_msg, "details": unified_profile_json_str}

if __name__ == '__main__':
    print("--- Diffusion Interfacer: Unification Example ---")
    mock_samples_main = [
        {"fingerprint": {"indent": 4, "quotes": "single", "linelen": 88, "docstyle": "google", "snake_pct":0.8, "camel_pct":0.1, "screaming_pct":0.05}, "file_path": "src/module_a/file1.py", "weight": 1.0},
        {"fingerprint": {"indent": 2, "quotes": "double", "linelen": 79, "docstyle": "numpy", "snake_pct":0.2, "camel_pct":0.7, "screaming_pct":0.03}, "file_path": "src/legacy/file2.py", "weight": 0.5},
        {"fingerprint": {"indent": 4, "quotes": "single", "linelen": 90, "docstyle": "google", "snake_pct":0.75, "camel_pct":0.15, "screaming_pct":0.04}, "file_path": "src/module_a/file3.py", "weight": 0.8},
        {"fingerprint": {"indent": 4, "quotes": "single", "linelen": 88, "docstyle": "google", "snake_pct":0.82, "camel_pct":0.10, "screaming_pct":0.05}, "file_path": "src/module_b/file4.py", "weight": 1.0},
        {"fingerprint": {"indent": 4, "quotes": "double", "linelen": 100, "docstyle": "plain", "snake_pct":0.88, "camel_pct":0.05, "screaming_pct":0.02}, "file_path": "tests/test_file5.py", "weight": 0.7},
    ]

    main_divot5_unifier_model_path = "./models/placeholder_divot5_unifier/"
    print(f"\nAttempting unification with DivoT5 (model path: {main_divot5_unifier_model_path}).")

    unified_profile_dict_main = unify_fingerprints_with_diffusion(
        per_sample_fingerprints=mock_samples_main,
        model_path=main_divot5_unifier_model_path,
        verbose=True
    )

    print("\nUnified Profile Output (DivoT5 or Error Fallback):")
    print(json.dumps(unified_profile_dict_main, indent=2))

    if unified_profile_dict_main.get("error"):
        print(f"\nUnification process reported an error: {unified_profile_dict_main['error']}")
    else:
        print("\nUnification process completed.")
        # Conceptual: Try to instantiate the dataclass from the result (excluding 'error' if it's None)
        # This part is for demonstration and would be in the orchestrator typically.
        try:
            profile_data_for_init = {k: v for k, v in unified_profile_dict_main.items() if k != 'error' or v is not None}
            if 'error' in profile_data_for_init and profile_data_for_init['error'] is None:
                del profile_data_for_init['error']

            # Ensure all dataclass fields are present, defaulting if necessary
            # This is crucial if the LLM misses some optional fields and they are not in final_expected_keys
            # or if Pydantic model is used later with non-Optional fields.
            # For dataclass with Optional fields, this is less critical if None is acceptable.
            # For now, assuming the current structure is sufficient for dataclass init.

            profile_model = UnifiedStyleProfile(**profile_data_for_init)
            print("\nSuccessfully created UnifiedStyleProfile Pydantic/Dataclass model from DivoT5 output:")
            print(json.dumps(profile_model.to_dict(), indent=2)) # Use to_dict() for consistent view
        except Exception as e_pydantic:
            print(f"\nError creating UnifiedStyleProfile model from DivoT5 output: {e_pydantic}")
            print("Final dictionary was:", json.dumps(unified_profile_dict_main, indent=2))


    print("\n--- Diffusion Interfacer Example Done ---")
