from typing import List, Dict, Any, Optional
from pathlib import Path
import math # For log if used, and for isnan, isinf

# Assuming CodeSample is defined in style_sampler and imported, or defined here for clarity
# For this module, it's better to expect a list of objects that have the required attributes.
# from .style_sampler import CodeSample # If CodeSample is easily importable

# For testing, define a structure that matches what StyleSampler's CodeSample would provide
class InputCodeSampleForWeighing:
    def __init__(self,
                 file_path: Path,
                 ai_fingerprint: Optional[Dict[str, Any]],
                 file_size_kb: Optional[float],
                 mod_timestamp: Optional[float]):
        self.file_path = file_path
        self.ai_fingerprint = ai_fingerprint
        self.file_size_kb = file_size_kb
        self.mod_timestamp = mod_timestamp

def calculate_sample_weights(
    samples_with_metadata: List[InputCodeSampleForWeighing], # Expects objects with necessary attributes
    size_weight_coeff: float = 0.7,
    recency_weight_coeff: float = 0.3,
    use_log_scale_for_size: bool = False # Option for log scaling of size
) -> List[Dict[str, Any]]:
    """
    Calculates a weight for each code sample based on its file size and recency.
    Outputs a list of dictionaries suitable for DivoT5 unification, including the
    fingerprint, file_path, and calculated weight.

    Args:
        samples_with_metadata: A list of objects (e.g., CodeSample instances) that must have
                               attributes: `ai_fingerprint` (dict), `file_path` (Path),
                               `file_size_kb` (float), and `mod_timestamp` (float).
        size_weight_coeff: Coefficient for the normalized file size score.
        recency_weight_coeff: Coefficient for the normalized recency score.
        use_log_scale_for_size: If True, applies log1p transformation to size before normalization.

    Returns:
        A list of dictionaries, each with keys "fingerprint", "file_path", and "weight".
        Returns an empty list if input is empty or metadata is missing.
    """
    if not samples_with_metadata:
        return []

    # Filter out samples missing essential metadata for weighting
    # and ensure ai_fingerprint is present
    valid_samples = [
        s for s in samples_with_metadata
        if s.file_size_kb is not None and s.mod_timestamp is not None and s.ai_fingerprint is not None
    ]

    if not valid_samples:
        # If no valid samples with metadata, return them with a default weight or handle as error
        # For now, return with default weight of 0.5 if fingerprint exists.
        return [
            {
                "fingerprint": s.ai_fingerprint,
                "file_path": str(s.file_path), # Convert Path to string for JSON serialization later
                "weight": 0.5
            } for s in samples_with_metadata if s.ai_fingerprint # only if fingerprint exists
        ]

    # Extract sizes and timestamps for normalization
    sizes = [s.file_size_kb for s in valid_samples] # type: ignore
    timestamps = [s.mod_timestamp for s in valid_samples] # type: ignore

    # Apply log scaling for size if enabled
    if use_log_scale_for_size:
        # Add 1 to avoid log(0) for zero-byte files (though unlikely for .py files with elements)
        # or if min_size is 0.
        sizes = [math.log1p(s if s is not None else 0) for s in sizes] # Added None check for s in sizes

    min_size, max_size = min(sizes), max(sizes)
    min_ts, max_ts = min(timestamps), max(timestamps)

    weighted_samples_for_divot5 = []

    for sample in valid_samples:
        # Normalize size score (0 to 1, 1 is largest)
        current_size_val = sample.file_size_kb if sample.file_size_kb is not None else 0
        if use_log_scale_for_size:
            current_size_proc = math.log1p(current_size_val)
        else:
            current_size_proc = current_size_val

        if max_size == min_size:
            norm_size_score = 1.0 if max_size > 0 else 0.0 # All same size (or all zero size)
        else:
            norm_size_score = (current_size_proc - min_size) / (max_size - min_size)
            if math.isnan(norm_size_score) or math.isinf(norm_size_score):
                norm_size_score = 0.5  # Fallback

        # Normalize recency score (0 to 1, 1 is most recent)
        current_mod_timestamp_val = sample.mod_timestamp if sample.mod_timestamp is not None else 0
        if max_ts == min_ts:
            norm_recency_score = 1.0 # All same timestamp (or only one file)
        else:
            norm_recency_score = (current_mod_timestamp_val - min_ts) / (max_ts - min_ts)
            if math.isnan(norm_recency_score) or math.isinf(norm_recency_score):
                norm_recency_score = 0.5  # Fallback

        # Combined weight
        combined_weight = (size_weight_coeff * norm_size_score) + \
                          (recency_weight_coeff * norm_recency_score)

        # Ensure weight is within a reasonable range, e.g., 0.0 to 1.0 (or sum of coeffs)
        combined_weight = max(0.0, min(combined_weight, size_weight_coeff + recency_weight_coeff))
        # Round to a few decimal places for cleaner output
        combined_weight = round(combined_weight, 4)

        weighted_samples_for_divot5.append({
            "fingerprint": sample.ai_fingerprint,
            "file_path": str(sample.file_path), # Store as string
            "weight": combined_weight
        })

    # Add back any samples that were filtered out, with a default weight
    # This ensures all original samples (that had an AI fingerprint) are passed through.
    if len(weighted_samples_for_divot5) < len([s for s in samples_with_metadata if s.ai_fingerprint]):
        valid_sample_tuples = {(str(s.file_path), id(s.ai_fingerprint)) for s in valid_samples} # Use id of dict for uniqueness
        for s_orig in samples_with_metadata:
            if s_orig.ai_fingerprint and (str(s_orig.file_path), id(s_orig.ai_fingerprint)) not in valid_sample_tuples:
                weighted_samples_for_divot5.append({
                    "fingerprint": s_orig.ai_fingerprint,
                    "file_path": str(s_orig.file_path),
                    "weight": 0.5 # Default weight for those missing metadata
                })

    return weighted_samples_for_divot5

if __name__ == '__main__':
    # Example Usage
    # Create dummy samples (mimicking CodeSample structure)
    dummy_fp = {"indent": 4, "docstyle": "google"}

    samples = [
        InputCodeSampleForWeighing(Path("file_new_large.py"), dummy_fp, 100.0, 1700000000.0), # New, large
        InputCodeSampleForWeighing(Path("file_old_small.py"), dummy_fp, 10.0,  1600000000.0), # Old, small
        InputCodeSampleForWeighing(Path("file_new_small.py"), dummy_fp, 10.0,  1700000000.0), # New, small
        InputCodeSampleForWeighing(Path("file_old_large.py"), dummy_fp, 100.0, 1600000000.0), # Old, large
        InputCodeSampleForWeighing(Path("file_mid_mid.py"),   dummy_fp, 50.0,  1650000000.0), # Mid, mid
        InputCodeSampleForWeighing(Path("file_no_meta.py"),   dummy_fp, None,  None),         # Missing metadata
        InputCodeSampleForWeighing(Path("file_no_fp.py"),     None,    20.0,  1680000000.0)  # Missing fingerprint
    ]

    print("--- Weights (Default Coeffs, Linear Size) ---")
    weighted_list = calculate_sample_weights(samples)
    for item in weighted_list:
        print(f"Path: {item['file_path']}, Weight: {item['weight']:.4f}, Fingerprint: {item['fingerprint']}")

    print("\n--- Weights (Log Scale for Size) ---")
    weighted_list_log = calculate_sample_weights(samples, use_log_scale_for_size=True)
    for item in weighted_list_log:
        print(f"Path: {item['file_path']}, Weight: {item['weight']:.4f}, Fingerprint: {item['fingerprint']}")

    print("\n--- Weights (Different Coeffs: Size=0.3, Recency=0.7) ---")
    weighted_list_recency_bias = calculate_sample_weights(samples, size_weight_coeff=0.3, recency_weight_coeff=0.7)
    for item in weighted_list_recency_bias:
        print(f"Path: {item['file_path']}, Weight: {item['weight']:.4f}, Fingerprint: {item['fingerprint']}")

    print("\n--- Weights (Only one valid sample) ---")
    single_valid_sample_list = [
        InputCodeSampleForWeighing(Path("single.py"), dummy_fp, 50.0, 1650000000.0),
        InputCodeSampleForWeighing(Path("no_meta_single.py"), dummy_fp, None, None)
    ]
    weighted_single = calculate_sample_weights(single_valid_sample_list)
    for item in weighted_single:
        print(f"Path: {item['file_path']}, Weight: {item['weight']:.4f}, Fingerprint: {item['fingerprint']}")

    print("\n--- Weights (All samples missing metadata but have fingerprint) ---")
    all_missing_meta = [
        InputCodeSampleForWeighing(Path("no_meta1.py"), dummy_fp, None, None),
        InputCodeSampleForWeighing(Path("no_meta2.py"), dummy_fp, None, None)
    ]
    weighted_all_missing = calculate_sample_weights(all_missing_meta)
    for item in weighted_all_missing:
        print(f"Path: {item['file_path']}, Weight: {item['weight']:.4f}, Fingerprint: {item['fingerprint']}")
