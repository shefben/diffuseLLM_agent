import random
from typing import Dict, List, Union, Optional, Any, Literal
from dataclasses import dataclass, field
from collections import Counter

# Assuming SingleSampleFingerprint structure from llm_interfacer.py
# For standalone execution, we might redefine a simplified version or expect it to be importable.
# from .llm_interfacer import SingleSampleFingerprint # This would be the typical import
import json # For DivoT5 input/output
import re   # For DivoT5 output parsing

# Add Hugging Face Transformers import, guarded
try:
    import torch
    from transformers import T5ForConditionalGeneration, T5TokenizerFast
except ImportError:
    torch = None # type: ignore
    T5ForConditionalGeneration = None # type: ignore
    T5TokenizerFast = None # type: ignore
    # Keep Counter for the fallback mock logic
    from collections import Counter, defaultdict # Added defaultdict
    print("Warning: PyTorch or Hugging Face Transformers not found. DivoT5 unification will use simplified mock logic.")

# Define the structure for the final, unified project-level style profile
@dataclass
class UnifiedStyleProfile:
    # Global scalar choices
    indent_width: Optional[int] = None
    preferred_quotes: Optional[Literal["single", "double"]] = None
    docstring_style: Optional[Literal["google", "numpy", "epytext", "restructuredtext", "plain", "other"]] = None
    max_line_length: Optional[int] = None

    # Columnar statistics for identifier patterns (project-wide)
    identifier_snake_case_pct: Optional[float] = None
    identifier_camelCase_pct: Optional[float] = None
    # identifier_PascalCase_pct: Optional[float] = None # REMOVED as per user spec for final profile
    identifier_UPPER_SNAKE_CASE_pct: Optional[float] = None # User spec mentions UPPER_SCREAMING %

    # Per-directory override map
    directory_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    confidence_score: Optional[float] = None
    raw_analysis_summary: Optional[Dict[str, Any]] = None # For any raw supporting data

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
        }

def unify_fingerprints_with_diffusion(
    per_sample_fingerprints: List[Dict[str, Any]], # Each dict: {"fingerprint": {...}, "file_path": str, "weight": float}
    model_path: str,
    num_beams: int = 4,
    max_output_length: int = 1024,
    device: Optional[str] = None
) -> Dict[str, Any]:
    """
    Uses a DivoT5 SafeTensor model to unify multiple per-sample style fingerprints
    (now including weights and file paths) into a single project-level style profile.

    Args:
        per_sample_fingerprints: A list of dictionaries, each containing:
                                 'fingerprint': the AI-derived style dict for a sample,
                                 'file_path': string path for that sample,
                                 'weight': float weight for that sample.
        model_path: Path to the DivoT5 model directory (SafeTensor format).
        num_beams: Number of beams for generation.
        max_output_length: Maximum length of the generated JSON string.
        device: Device to run the model on (e.g., "cpu", "cuda"). Auto-detects if None.

    Returns:
        A dictionary representing the unified project-level style profile.
        Falls back to weighted aggregation if model interaction fails.
    """
    if T5ForConditionalGeneration is None or T5TokenizerFast is None or torch is None:
        print("Error: Transformers/PyTorch not installed. DivoT5 unification falling back to weighted mock aggregation.")
        # Fallback to weighted aggregation logic
        if not per_sample_fingerprints:
            return UnifiedStyleProfile().to_dict()

        final_profile = UnifiedStyleProfile()

        # Weighted mode for categorical features
        for key in ["indent_width", "preferred_quotes", "docstring_style"]:
            weighted_counts: Dict[Any, float] = defaultdict(float)
            # Ensure 'fingerprint' and 'weight' keys exist and fingerprint is a dict
            valid_samples_for_key = [
                s for s in per_sample_fingerprints
                if isinstance(s.get("fingerprint"), dict) and s["fingerprint"].get(key) is not None and isinstance(s.get("weight"), (int,float))
            ]
            if not valid_samples_for_key: continue

            for s in valid_samples_for_key:
                value = s["fingerprint"].get(key)
                weight = s.get("weight", 0.1) # Default small weight if missing, though should be present
                weighted_counts[value] += weight

            if weighted_counts:
                most_common_val = max(weighted_counts, key=weighted_counts.get)
                if key == "indent_width": final_profile.indent_width = most_common_val
                elif key == "preferred_quotes": final_profile.preferred_quotes = most_common_val
                elif key == "docstring_style": final_profile.docstring_style = most_common_val

        # Weighted average for numerical features (linelen, percentages)
        for num_key_info in [("max_line_length", "max_line_length"),
                             ("identifier_snake_case_pct", "snake_pct"), # Profile key, Sample key
                             ("identifier_camelCase_pct", "camel_pct"),
                             ("identifier_UPPER_SNAKE_CASE_pct", "screaming_pct")]: # User spec was screaming_pct from DeepSeek draft

            profile_key, sample_key = num_key_info
            total_weighted_value = 0.0
            total_weight = 0.0

            for s in per_sample_fingerprints:
                fingerprint = s.get("fingerprint")
                weight = s.get("weight")
                if isinstance(fingerprint, dict) and isinstance(weight, (int, float)) and isinstance(fingerprint.get(sample_key), (int, float)):
                    value = fingerprint[sample_key]
                    total_weighted_value += value * weight
                    total_weight += weight

            if total_weight > 0:
                avg_val = total_weighted_value / total_weight
                if profile_key == "max_line_length":
                    final_profile.max_line_length = int(round(avg_val))
                else: # Percentage keys
                    setattr(final_profile, profile_key, round(avg_val, 2))

        final_profile.confidence_score = 0.5 # Indicate mock/fallback result
        return final_profile.to_dict()

    # --- Actual DivoT5 interaction (largely same as before, but prompt needs update) ---
    if device is None:
        selected_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        selected_device = device

    print(f"DivoT5 Unification: Loading model from {model_path} onto device: {selected_device}")

    try:
        tokenizer = T5TokenizerFast.from_pretrained(model_path)
        model = T5ForConditionalGeneration.from_pretrained(model_path).to(selected_device)
        model.eval()
    except Exception as e:
        print(f"DivoT5 Unification: Error loading model from {model_path}: {e}")
        print("Falling back to weighted mock aggregation.")
        return unify_fingerprints_with_diffusion(per_sample_fingerprints, "mock_path_trigger_fallback_RECURSIVE_CALL_AVOIDANCE", num_beams, max_output_length, device) # Avoid recursive call with same params

    try:
        # per_sample_fingerprints now includes 'weight' and 'file_path' directly
        samples_json_str = json.dumps(per_sample_fingerprints, indent=2)
    except TypeError as e:
        print(f"DivoT5 Unification: Error serializing per_sample_fingerprints to JSON: {e}")
        print("Falling back to weighted mock aggregation.")
        return unify_fingerprints_with_diffusion(per_sample_fingerprints, "mock_path_trigger_fallback_RECURSIVE_CALL_AVOIDANCE", num_beams, max_output_length, device)

    system_prompt = ("You are a style analysis assistant. Based on the following list of style fingerprints "
                     "(each derived from a code sample, including its file path and a 'weight' indicating its importance), "
                     "determine the dominant, coherent project-level style profile. Pay more attention to samples with "
                     "higher 'weight'. The project-level profile should include global choices for indentation, quotes, "
                     "line length, and docstring style, as well as overall percentages for identifier casing "
                     "(snake_case, camelCase, UPPER_SCREAMING_CASE). If strong, consistent sub-styles are detected for "
                     "specific directories (inferable from 'file_path' in samples), provide those as 'directory_overrides'. "
                     "Output the result as a single JSON object matching the UnifiedStyleProfile structure.")

    input_text = (f"SYSTEM: {system_prompt}\n\n"
                  f"USER_FINGERPRINT_SAMPLES_START:\n{samples_json_str}\nUSER_FINGERPRINT_SAMPLES_END:\n\n"
                  f"PROJECT_STYLE_PROFILE_JSON_START:")

    unified_profile_json_str = "{}"

    try:
        print(f"DivoT5 Unification: Input text length: {len(input_text)} chars. Number of samples: {len(per_sample_fingerprints)}")
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=4096).to(selected_device)

        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids, max_length=max_output_length,
                num_beams=num_beams, early_stopping=True,
            )

        if outputs is not None and len(outputs) > 0:
            unified_profile_json_str = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            if unified_profile_json_str.startswith("PROJECT_STYLE_PROFILE_JSON_START:"):
                unified_profile_json_str = unified_profile_json_str[len("PROJECT_STYLE_PROFILE_JSON_START:"):].strip()
            if not unified_profile_json_str.startswith("{"):
                print(f"Warning: DivoT5 output does not look like JSON: '{unified_profile_json_str[:200]}...'")
        else:
            print("Warning: DivoT5 model returned an empty or unexpected response.")

    except Exception as e:
        print(f"DivoT5 Unification: Error during model inference: {e}")
        print("Falling back to weighted mock aggregation.")
        return unify_fingerprints_with_diffusion(per_sample_fingerprints, "mock_path_trigger_fallback_RECURSIVE_CALL_AVOIDANCE", num_beams, max_output_length, device)

    try:
        json_match = re.search(r'\{.*\}', unified_profile_json_str, re.DOTALL)
        if json_match:
            unified_profile_json_str = json_match.group(0)
        final_profile_dict = json.loads(unified_profile_json_str)

        expected_top_keys = ["indent_width", "preferred_quotes", "docstring_style", "max_line_length",
                             "identifier_snake_case_pct", "identifier_camelCase_pct",
                             "identifier_UPPER_SNAKE_CASE_pct", "directory_overrides"]
        for key in expected_top_keys:
            if key not in final_profile_dict:
                print(f"Warning: DivoT5 output JSON missing expected key '{key}'. Output: {unified_profile_json_str[:500]}")
        return final_profile_dict

    except json.JSONDecodeError as e:
        print(f"DivoT5 Unification: Failed to decode JSON from model output: {e}")
        print(f"Model output was: '{unified_profile_json_str[:500]}...'")
        print("Falling back to weighted mock aggregation.")
        # Avoid infinite recursion if fallback is called repeatedly with same failing input
        # This specific string "mock_path_trigger_fallback_RECURSIVE_CALL_AVOIDANCE" is a sentinel.
        # If model_path is this sentinel, it means we are already in a fallback from this function,
        # so we should just return a very basic default to stop recursion.
        if model_path == "mock_path_trigger_fallback_RECURSIVE_CALL_AVOIDANCE":
            print("Recursive fallback detected in DivoT5 unification. Returning minimal default profile.")
            return UnifiedStyleProfile(confidence_score=0.1).to_dict() # Minimal default
        return unify_fingerprints_with_diffusion(per_sample_fingerprints, "mock_path_trigger_fallback_RECURSIVE_CALL_AVOIDANCE", num_beams, max_output_length, device)

# (The existing if __name__ == '__main__' block should be updated to reflect the new input structure
# for per_sample_fingerprints, now including 'weight'.)
# Example:
# if __name__ == '__main__':
#     mock_samples_with_paths_and_weights = [
#         {"fingerprint": {"indent": 4, ...}, "file_path": "src/a.py", "weight": 0.8},
#         {"fingerprint": {"indent": 2, ...}, "file_path": "src/b.py", "weight": 0.5},
#     ]
#     # ... call unify_fingerprints_with_diffusion ...
#
# The previous __main__ block:
if __name__ == '__main__':
    # Example usage:
    # Create some mock sample fingerprints (as if from the LLM interfacer)
    # Removed has_type_hints and spacing_around_operators from mock samples
    # Updated structure for per_sample_fingerprints
    mock_samples_with_paths_and_weights = [ # Updated variable name for clarity
        {"fingerprint": {"indent": 4, "quotes": "single", "linelen": 88, "docstyle": "google", "snake_pct":0.8, "camel_pct":0.1, "UPPER_SNAKE_CASE_pct":0.05}, "file_path": "src/module_a/file1.py", "weight": 1.0},
        {"fingerprint": {"indent": 2, "quotes": "double", "linelen": 79, "docstyle": "numpy", "snake_pct":0.2, "camel_pct":0.7, "UPPER_SNAKE_CASE_pct":0.03}, "file_path": "src/legacy/file2.py", "weight": 0.5},
        {"fingerprint": {"indent": 4, "quotes": "single", "linelen": 90, "docstyle": "google", "snake_pct":0.75, "camel_pct":0.15, "UPPER_SNAKE_CASE_pct":0.04}, "file_path": "src/module_a/file3.py", "weight": 0.8},
        {"fingerprint": {"indent": 4, "quotes": "single", "linelen": 88, "docstyle": "google", "snake_pct":0.82, "camel_pct":0.10, "UPPER_SNAKE_CASE_pct":0.05}, "file_path": "src/module_b/file4.py", "weight": 1.0},
        {"fingerprint": {"indent": 4, "quotes": "double", "linelen": 88, "docstyle": "plain", "snake_pct":0.88, "camel_pct":0.05, "UPPER_SNAKE_CASE_pct":0.02}, "file_path": "tests/test_file5.py", "weight": 0.7},

    for k, v in unified_profile_dict.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for sub_k, sub_v in v.items():
                print(f"    {sub_k}: {sub_v}")
        else:
            print(f"  {k}: {v}")

    print("\n--- Another run (expect some variation due to random elements in mock) ---")
    unified_profile_dict_2 = unify_fingerprints_with_diffusion(mock_samples[:5]) # Fewer samples
    print(f"  Indent Width: {unified_profile_dict_2.get('indent_width')}")
    print(f"  Preferred Quotes: {unified_profile_dict_2.get('preferred_quotes')}")
    print(f"  Docstring Style: {unified_profile_dict_2.get('docstring_style')}")
    print(f"  Snake Case Pct: {unified_profile_dict_2.get('identifier_snake_case_pct')}")
    print(f"  Directory Overrides: {unified_profile_dict_2.get('directory_overrides')}")
