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
    from collections import Counter
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
    per_sample_fingerprints: List[Dict[str, Any]], # Each dict should contain 'fingerprint': {...} and 'file_path': str
    model_path: str,
    num_beams: int = 4,
    max_output_length: int = 1024, # Max length for the generated JSON profile
    device: Optional[str] = None
) -> Dict[str, Any]:
    """
    Uses a DivoT5 SafeTensor model to unify multiple per-sample style fingerprints
    (including file paths for context) into a single project-level style profile.

    Args:
        per_sample_fingerprints: A list of dictionaries, where each dictionary contains
                                 a 'fingerprint' (the AI-derived style dict for a sample)
                                 and 'file_path' (string path for that sample).
        model_path: Path to the DivoT5 model directory (SafeTensor format).
        num_beams: Number of beams for generation.
        max_output_length: Maximum length of the generated JSON string.
        device: Device to run the model on (e.g., "cpu", "cuda"). Auto-detects if None.

    Returns:
        A dictionary representing the unified project-level style profile.
        Falls back to simpler aggregation if model interaction fails.
    """
    if T5ForConditionalGeneration is None or T5TokenizerFast is None or torch is None:
        print("Error: Transformers/PyTorch not installed. DivoT5 unification falling back to basic mock aggregation.")
        # Fallback to the old mock logic (simplified aggregation)
        if not per_sample_fingerprints:
            return UnifiedStyleProfile().to_dict()

        # Extract just the fingerprint dicts for mock processing
        fingerprint_dicts_only = [item['fingerprint'] for item in per_sample_fingerprints if 'fingerprint' in item]
        if not fingerprint_dicts_only:
            return UnifiedStyleProfile().to_dict()

        # Simplified aggregation (mode for categorical, mean for numerical)
        # This is the old mock logic, kept as a fallback.
        final_profile = UnifiedStyleProfile()
        for key in ["indent_width", "preferred_quotes", "docstring_style", "max_line_length"]:
            values = [fp.get(key) for fp in fingerprint_dicts_only if fp.get(key) is not None]
            if values:
                if isinstance(values[0], int) and key == "max_line_length": # Average for linelen
                    final_profile.max_line_length = int(sum(values) / len(values))
                elif isinstance(values[0], int) and key == "indent_width": # Mode for indent
                    final_profile.indent_width = Counter(values).most_common(1)[0][0]
                else: # Mode for strings
                    most_common_val = Counter(values).most_common(1)[0][0]
                    if key == "preferred_quotes": final_profile.preferred_quotes = most_common_val
                    elif key == "docstring_style": final_profile.docstring_style = most_common_val

        for pct_key in ["identifier_snake_case_pct", "identifier_camelCase_pct", "identifier_UPPER_SNAKE_CASE_pct"]:
            # Corrected:
            pct_values = [fp.get(pct_key) for fp in fingerprint_dicts_only if fp.get(pct_key) is not None and isinstance(fp.get(pct_key), (float,int))]
            if pct_values:
                avg_pct = sum(pct_values) / len(pct_values)
                if pct_key == "identifier_snake_case_pct": final_profile.identifier_snake_case_pct = round(avg_pct, 2)
                elif pct_key == "identifier_camelCase_pct": final_profile.identifier_camelCase_pct = round(avg_pct, 2)
                elif pct_key == "identifier_UPPER_SNAKE_CASE_pct": final_profile.identifier_UPPER_SNAKE_CASE_pct = round(avg_pct, 2)

        final_profile.confidence_score = 0.5 # Indicate mock/fallback result
        return final_profile.to_dict()

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
        print("Falling back to basic mock aggregation.")
        # Recursive call to use the fallback logic defined above
        return unify_fingerprints_with_diffusion(per_sample_fingerprints, "mock_path_trigger_fallback", num_beams, max_output_length, device)


    # Prepare input for DivoT5
    # Serialize the list of {"fingerprint": dict, "file_path": str} objects to a JSON string
    try:
        samples_json_str = json.dumps(per_sample_fingerprints, indent=2)
    except TypeError as e:
        print(f"DivoT5 Unification: Error serializing per_sample_fingerprints to JSON: {e}")
        print("Falling back to basic mock aggregation.")
        return unify_fingerprints_with_diffusion(per_sample_fingerprints, "mock_path_trigger_fallback", num_beams, max_output_length, device)

    system_prompt = ("You are a style analysis assistant. Based on the following list of style fingerprints "
                     "(each derived from a code sample, including its file path), determine the dominant, "
                     "coherent project-level style profile. The project-level profile should include global "
                     "choices for indentation, quotes, line length, and docstring style, as well as overall "
                     "percentages for identifier casing (snake_case, camelCase, UPPER_SCREAMING_CASE). "
                     "If strong, consistent sub-styles are detected for specific directories "
                     "(inferable from 'file_path' in samples), provide those as 'directory_overrides'. "
                     "Output the result as a single JSON object matching the UnifiedStyleProfile structure.")

    input_text = (f"SYSTEM: {system_prompt}\n\n"
                  f"USER_FINGERPRINT_SAMPLES_START:\n{samples_json_str}\nUSER_FINGERPRINT_SAMPLES_END:\n\n"
                  f"PROJECT_STYLE_PROFILE_JSON_START:")

    unified_profile_json_str = "{}" # Default to empty JSON

    try:
        print(f"DivoT5 Unification: Input text length: {len(input_text)} chars. Number of samples: {len(per_sample_fingerprints)}")
        # Ensure tokenizer handles potentially very long input. Max model length for T5 is often 512 or 1024.
        # If input_text is too long, this will be an issue. Truncation might lose data.
        # Consider strategies for handling >1000 samples if total serialized length is too great.
        # For now, assume it fits or gets truncated by tokenizer.
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=4096).to(selected_device) # Increased max_length for input

        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_length=max_output_length,
                num_beams=num_beams,
                early_stopping=True,
                # Consider adding task-specific prefix or other generation params if needed for DivoT5
            )

        if outputs is not None and len(outputs) > 0:
            unified_profile_json_str = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            # Sometimes T5 might output the "PROJECT_STYLE_PROFILE_JSON_START:" part if not explicitly handled
            # or if the prompt structure is slightly off for its training.
            # A simple cleanup:
            if unified_profile_json_str.startswith("PROJECT_STYLE_PROFILE_JSON_START:"):
                unified_profile_json_str = unified_profile_json_str[len("PROJECT_STYLE_PROFILE_JSON_START:"):].strip()
            if not unified_profile_json_str.startswith("{"): # If it's not JSON, something went wrong
                print(f"Warning: DivoT5 output does not look like JSON: '{unified_profile_json_str[:200]}...'")
                # Fallback or raise error
        else:
            print("Warning: DivoT5 model returned an empty or unexpected response.")

    except Exception as e:
        print(f"DivoT5 Unification: Error during model inference: {e}")
        print("Falling back to basic mock aggregation.")
        return unify_fingerprints_with_diffusion(per_sample_fingerprints, "mock_path_trigger_fallback", num_beams, max_output_length, device)

    # Parse the output JSON string from DivoT5
    try:
        # Attempt to find a valid JSON block if there's surrounding text
        json_match = re.search(r'\{.*\}', unified_profile_json_str, re.DOTALL)
        if json_match:
            unified_profile_json_str = json_match.group(0)

        final_profile_dict = json.loads(unified_profile_json_str)

        # Basic validation: check for a few key fields expected in UnifiedStyleProfile
        # More detailed validation of values can be done by the caller or in a separate step.
        expected_top_keys = ["indent_width", "preferred_quotes", "docstring_style", "max_line_length",
                             "identifier_snake_case_pct", "identifier_camelCase_pct",
                             "identifier_UPPER_SNAKE_CASE_pct", "directory_overrides"]
        for key in expected_top_keys:
            if key not in final_profile_dict:
                print(f"Warning: DivoT5 output JSON missing expected key '{key}'. Output: {unified_profile_json_str[:500]}")
                # Decide if this constitutes a failure or if partial data is acceptable
                # For now, we proceed but this indicates an issue.

        return final_profile_dict

    except json.JSONDecodeError as e:
        print(f"DivoT5 Unification: Failed to decode JSON from model output: {e}")
        print(f"Model output was: '{unified_profile_json_str[:500]}...'") # Print snippet of problematic output
        print("Falling back to basic mock aggregation.")
        return unify_fingerprints_with_diffusion(per_sample_fingerprints, "mock_path_trigger_fallback", num_beams, max_output_length, device)


if __name__ == '__main__':
    # Example usage:
    # Create some mock sample fingerprints (as if from the LLM interfacer)
    # Removed has_type_hints and spacing_around_operators from mock samples
    # Updated structure for per_sample_fingerprints
    mock_samples_with_paths = [
        {"fingerprint": {"indent": 4, "quotes": "single", "linelen": 88, "docstyle": "google", "snake_pct":0.8, "camel_pct":0.1, "UPPER_SNAKE_CASE_pct":0.05}, "file_path": "src/module_a/file1.py"},
        {"fingerprint": {"indent": 2, "quotes": "double", "linelen": 79, "docstyle": "numpy", "snake_pct":0.2, "camel_pct":0.7, "UPPER_SNAKE_CASE_pct":0.03}, "file_path": "src/legacy/file2.py"},
        {"fingerprint": {"indent": 4, "quotes": "single", "linelen": 90, "docstyle": "google", "snake_pct":0.75, "camel_pct":0.15, "UPPER_SNAKE_CASE_pct":0.04}, "file_path": "src/module_a/file3.py"},
        {"fingerprint": {"indent": 4, "quotes": "single", "linelen": 88, "docstyle": "google", "snake_pct":0.82, "camel_pct":0.10, "UPPER_SNAKE_CASE_pct":0.05}, "file_path": "src/module_b/file4.py"},
        {"fingerprint": {"indent": 4, "quotes": "double", "linelen": 88, "docstyle": "plain", "snake_pct":0.88, "camel_pct":0.05, "UPPER_SNAKE_CASE_pct":0.02}, "file_path": "tests/test_file5.py"},
    ]  * 5 # Multiply to get more samples for averaging

    print("--- Mock Unified Profile from Diffusion Model ---")
    unified_profile_dict = unify_fingerprints_with_diffusion(mock_samples)

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
