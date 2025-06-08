import ast
import os
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple, Union, Optional
from dataclasses import dataclass
import math # For floor

# Add import for the new AI fingerprinting function
from .llm_interfacer import get_ai_style_fingerprint_for_sample # Assuming relative import if in same package

# Modify CodeSample to store the AI fingerprint dictionary
@dataclass
class CodeSample:
    file_path: Path
    item_name: str
    item_type: str # 'function', 'class', 'module_docstring_snippet'
    code_snippet: str # The raw source code of the function/class or the docstring text
    start_line: int
    end_line: int
    ai_fingerprint: Optional[Dict[str, Any]] = None # To store the result from AI processing
    error_while_fingerprinting: Optional[str] = None # To store any error message
    file_size_kb: Optional[float] = None  # New field
    mod_timestamp: Optional[float] = None # New field

class StyleSampler:
    def __init__(self,
                 repo_path: Union[str, Path],
                 deepseek_model_path: str, # New required parameter
                 divot5_model_path: str,   # New required parameter
                 target_samples: int = 1000,
                 random_seed: Optional[int] = None,
                 # Optional params for the AI functions, passed down
                 deepseek_n_gpu_layers: int = -1,
                 deepseek_verbose: bool = False,
                 divot5_device: Optional[str] = None,
                 divot5_num_denoising_steps: int = 10
                ):
        self.repo_path = Path(repo_path)
        if not self.repo_path.is_dir():
            raise ValueError(f"Repository path {repo_path} is not a valid directory.")

        # Store model paths and AI params
        self.deepseek_model_path = deepseek_model_path
        self.divot5_model_path = divot5_model_path
        self.deepseek_n_gpu_layers = deepseek_n_gpu_layers
        self.deepseek_verbose = deepseek_verbose
        self.divot5_device = divot5_device
        self.divot5_num_denoising_steps = divot5_num_denoising_steps

        self.target_samples = target_samples
        if random_seed is not None:
            random.seed(random_seed)

        self._all_py_files: List[Path] = []
        self._collected_elements: List[CodeSample] = []

    def _discover_python_files(self):
        """Discovers all Python files in the repository path. (Implementation as before)"""
        self._all_py_files = list(self.repo_path.rglob("*.py"))
        common_ignore_dirs = [".venv", "venv", ".env", "env", "node_modules", ".git", "__pycache__"] # Added __pycache__

        filtered_files = []
        for py_file in self._all_py_files:
            try:
                if not any(part in common_ignore_dirs for part in py_file.parts):
                    filtered_files.append(py_file)
            except Exception:
                pass
        self._all_py_files = filtered_files

    def _extract_elements_from_file(self, file_path: Path) -> List[CodeSample]:
        """
        Parses a single Python file, extracts functions, classes, and their
        docstrings (as snippets), and then gets an AI style fingerprint for each.
        """
        elements_in_file: List[CodeSample] = []
        raw_code_elements_to_fingerprint: List[Tuple[str, str, str, int, int]] = []

        file_metadata_collected = False
        current_file_size_kb: Optional[float] = None
        current_mod_timestamp: Optional[float] = None

        try:
            # Collect file metadata once per file
            file_stat = file_path.stat()
            current_file_size_kb = round(file_stat.st_size / 1024.0, 2)
            current_mod_timestamp = file_stat.st_mtime
            file_metadata_collected = True

            with open(file_path, "r", encoding="utf-8") as f:
                source_code = f.read()

            tree = ast.parse(source_code, filename=str(file_path))

            # Module docstring as a snippet to fingerprint (if exists)
            module_docstring_text = ast.get_docstring(tree, clean=False)
            if module_docstring_text:
                # Try to get line numbers for the module docstring node
                # Module docstring is an Expr node whose value is a Constant (or Str in older Python)
                start_line, end_line = 1, len(module_docstring_text.splitlines()) # Default if node not found
                if tree.body and isinstance(tree.body[0], ast.Expr):
                    doc_node = tree.body[0]
                    start_line = doc_node.lineno
                    end_line = getattr(doc_node, 'end_lineno', start_line + len(module_docstring_text.splitlines()) -1)

                raw_code_elements_to_fingerprint.append((
                    f"{file_path.name}::module_docstring", "module_docstring_snippet",
                    module_docstring_text, start_line, end_line
                ))

            for node in ast.walk(tree):
                snippet_to_fingerprint = None
                item_name = None
                item_type = None
                node_start_line = node.lineno if hasattr(node, 'lineno') else 0
                node_end_line = node.end_lineno if hasattr(node, 'end_lineno') else 0

                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    item_name = node.name
                    item_type = "function"
                    try:
                        snippet_to_fingerprint = ast.get_source_segment(source_code, node)
                    except Exception: # get_source_segment can fail
                        pass # snippet_to_fingerprint remains None

                elif isinstance(node, ast.ClassDef):
                    item_name = node.name
                    item_type = "class"
                    try:
                        snippet_to_fingerprint = ast.get_source_segment(source_code, node)
                    except Exception:
                        pass

                if snippet_to_fingerprint and item_name and item_type:
                    raw_code_elements_to_fingerprint.append((
                        item_name, item_type, snippet_to_fingerprint,
                        node_start_line, node_end_line
                    ))

        except FileNotFoundError: # Should ideally be caught by discover_python_files, but defensive
            print(f"Error: File {file_path} not found during element extraction.")
            return []
        except SyntaxError:
            print(f"Syntax error in {file_path}, skipping AI fingerprinting for this file.")
            return []
        except Exception as e:
            print(f"Could not process {file_path} for element extraction: {e}")
            return []

        if not file_metadata_collected: # Should not happen if no exception before this point
            print(f"Warning: File metadata not collected for {file_path}")


        for name, type_val, snippet, start, end in raw_code_elements_to_fingerprint:
            # AI fingerprinting call
            ai_fp_dict = None
            error_msg = None
            try:
                ai_fp_dict = get_ai_style_fingerprint_for_sample(
                    snippet,
                    deepseek_model_path=self.deepseek_model_path,
                    divot5_model_path=self.divot5_model_path,
                    deepseek_n_gpu_layers=self.deepseek_n_gpu_layers,
                    deepseek_verbose=self.deepseek_verbose,
                    divot5_device=self.divot5_device,
                    divot5_num_denoising_steps=self.divot5_num_denoising_steps
                )
                if ai_fp_dict.get("validation_status") != "passed":
                    error_msg = f"AI fingerprinting validation failed: {ai_fp_dict.get('validation_status')}, errors: {ai_fp_dict.get('validation_errors', ai_fp_dict.get('details'))}"
            except Exception as e_fp:
                error_msg = f"Exception during AI fingerprinting: {e_fp}"

            elements_in_file.append(CodeSample(
                file_path=file_path, item_name=name, item_type=type_val,
                code_snippet=snippet, start_line=start, end_line=end,
                ai_fingerprint=ai_fp_dict, error_while_fingerprinting=error_msg,
                file_size_kb=current_file_size_kb, # Add collected metadata
                mod_timestamp=current_mod_timestamp  # Add collected metadata
            ))

        return elements_in_file


    def collect_all_elements(self):
        """Collects all code elements (with AI fingerprints) from all Python files."""
        self._discover_python_files()
        self._collected_elements = []
        if not self._all_py_files:
            print("StyleSampler: No Python files found to process.")
            return

        print(f"StyleSampler: Found {len(self._all_py_files)} Python files to process.")
        for i, py_file in enumerate(self._all_py_files):
            print(f"StyleSampler: Processing file {i+1}/{len(self._all_py_files)}: {py_file.name}...")
            elements_from_this_file = self._extract_elements_from_file(py_file)
            self._collected_elements.extend(elements_from_this_file)
            if not elements_from_this_file:
                 print(f"StyleSampler: No elements extracted or fingerprinted from {py_file.name}.")

        if not self._collected_elements and self._all_py_files:
            print("StyleSampler: Warning: No code elements were successfully fingerprinted, "
                  "though Python files were found.")


    def sample_elements(self) -> List[CodeSample]:
        """
        Performs stratified sampling of the collected code elements by directory.
        Aims to make the sample representative of different sub-domains (folders).
        """
        if not self._collected_elements:
            self.collect_all_elements()

        if not self._collected_elements:
            print("StyleSampler: No elements available for sampling.")
            return []

        total_elements_available = len(self._collected_elements)

        if total_elements_available == 0: # Should be caught by above, but defensive
            return []

        if total_elements_available <= self.target_samples:
            print(f"StyleSampler: Collected {total_elements_available} elements, which is <= target {self.target_samples}. Returning all.")
            return self._collected_elements[:] # Return a copy

        # 1. Group elements by directory
        elements_by_dir: Dict[Path, List[CodeSample]] = {}
        for sample in self._collected_elements:
            dir_path = sample.file_path.parent
            if dir_path not in elements_by_dir:
                elements_by_dir[dir_path] = []
            elements_by_dir[dir_path].append(sample)

        # Sort directories to make apportionment deterministic if fractions are equal later
        sorted_dirs = sorted(elements_by_dir.keys(), key=lambda p: str(p))

        # 2. Calculate initial proportional samples per directory (using floor)
        #    and store fractional parts for tie-breaking/remainder distribution.

        # Stores (directory_path, num_elements_in_dir, ideal_float_samples, fractional_part, current_int_samples)
        dir_sample_info = []

        # If target_samples is very small, e.g., less than num_dirs, ensure representation
        # For now, simple proportional allocation. Refinements for min 1 sample per dir can be added.

        calculated_total_ideal_samples = 0.0 # For sanity check, should be close to target_samples

        for dir_path in sorted_dirs:
            elements_in_dir = elements_by_dir[dir_path]
            num_elements_in_dir = len(elements_in_dir)

            # Ideal number of samples (float)
            ideal_samples_float = (num_elements_in_dir / total_elements_available) * self.target_samples
            calculated_total_ideal_samples += ideal_samples_float

            # Initial integer allocation (floor)
            current_int_samples = math.floor(ideal_samples_float)
            fractional_part = ideal_samples_float - current_int_samples

            # Cap at the number of available items in that directory
            current_int_samples = min(current_int_samples, num_elements_in_dir)

            dir_sample_info.append({
                "path": dir_path,
                "num_available": num_elements_in_dir,
                "ideal_float": ideal_samples_float,
                "fractional": fractional_part,
                "current_samples": current_int_samples
            })

        # 3. Distribute remaining samples based on largest fractional parts
        current_allocated_samples = sum(info["current_samples"] for info in dir_sample_info)
        samples_remaining_to_allocate = self.target_samples - current_allocated_samples

        # Sort directories by fractional part descending to prioritize allocation
        # Add secondary sort key (e.g., num_available descending) for tie-breaking if fractions are same
        dir_sample_info.sort(key=lambda x: (x["fractional"], x["num_available"]), reverse=True)

        for i in range(samples_remaining_to_allocate):
            allocated_this_round = False
            for info in dir_sample_info: # Iterate through sorted list
                if info["current_samples"] < info["num_available"]:
                    info["current_samples"] += 1
                    allocated_this_round = True
                    # Re-sort or cycle through if one dir gets multiple remainders.
                    # For simplicity, one pass then re-sort for next remainder.
                    # Or, simply iterate and if a dir gets one, move to next for fairness in this round.
                    # The current loop iterates up to samples_remaining_to_allocate times.
                    # In each iteration, it finds the *next available* directory from the sorted list.
                    # To ensure fairness, we should re-sort or pick strategically if one dir gets all remainders.
                    # A simpler way for now: iterate through the sorted list once per remainder.
                    break
            if not allocated_this_round:
                # This might happen if target_samples > total_elements_available,
                # but that case is handled at the start.
                # Or if all directories are at max capacity for their current_samples.
                break # No more samples can be allocated.

        # Recalculate current_allocated_samples after distributing remainders
        current_allocated_samples = sum(info["current_samples"] for info in dir_sample_info)

        # 4. Handle potential over/under-allocation due to multiple constraints or rounding edge cases
        # This step aims to get exactly self.target_samples if possible
        if current_allocated_samples != self.target_samples:
            # If under-allocated (more common if initial current_samples were capped by num_available)
            # and total elements allow, try to add more from largest available pools.
            if current_allocated_samples < self.target_samples:
                # Sort by num_available - current_samples (descending) to find where we can add more
                dir_sample_info.sort(key=lambda x: (x["num_available"] - x["current_samples"]), reverse=True)
                for i in range(self.target_samples - current_allocated_samples):
                    added_in_final_pass = False
                    for info in dir_sample_info:
                        if info["current_samples"] < info["num_available"]:
                            info["current_samples"] += 1
                            added_in_final_pass = True
                            break
                    if not added_in_final_pass: break # Cannot add more

            # If over-allocated (less common with floor + remainder, but possible if logic changes)
            # Remove from smallest pools that have samples > 0 (or > 1 if min representation is 1)
            elif current_allocated_samples > self.target_samples:
                # Sort by current_samples (ascending, but non-zero)
                dir_sample_info.sort(key=lambda x: (x["current_samples"] if x["current_samples"] > 0 else float('inf')))
                for i in range(current_allocated_samples - self.target_samples):
                    removed_in_final_pass = False
                    for info in dir_sample_info:
                        if info["current_samples"] > 0: # Don't go below 0
                            # Could add logic here to not go below 1 if dir is non-empty and target allows
                            info["current_samples"] -= 1
                            removed_in_final_pass = True
                            break
                    if not removed_in_final_pass: break # Cannot remove more

        # 5. Perform the actual sampling from each directory
        final_selected_samples: List[CodeSample] = []
        for info in dir_sample_info:
            dir_path = info["path"]
            num_to_sample_from_this_dir = info["current_samples"]

            if num_to_sample_from_this_dir > 0:
                dir_elements = elements_by_dir[dir_path]
                # Ensure we don't try to sample more than available (should be handled by min() earlier)
                actual_to_sample = min(num_to_sample_from_this_dir, len(dir_elements))

                # random.sample requires k <= len(population)
                if actual_to_sample > 0 :
                     final_selected_samples.extend(random.sample(dir_elements, actual_to_sample))

        # Final check on total count, can happen if target is high and many strata are small
        # If still not enough, and total_elements_available > len(final_selected_samples),
        # it implies some strata didn't get enough due to caps.
        # This case should ideally be handled by the remainder distribution logic.
        # If too many, truncate (though this should also be handled by adjustments).
        if len(final_selected_samples) > self.target_samples:
            # This indicates an issue in allocation logic if it happens when total_elements_available > target_samples
            print(f"Warning: Stratified sampling resulted in {len(final_selected_samples)} samples, expected {self.target_samples}. Truncating.")
            final_selected_samples = random.sample(final_selected_samples, self.target_samples) # Re-sample from the over-sampled set
        elif len(final_selected_samples) < self.target_samples and total_elements_available > len(final_selected_samples):
            # This indicates an issue in allocation or that all small strata were exhausted before target was met.
            # Try to fill remaining from the overall pool of *unsampled* elements if any.
            # This is a bit complex to do efficiently here. The current logic should try to avoid this.
            print(f"Warning: Stratified sampling resulted in {len(final_selected_samples)} samples, less than target {self.target_samples} despite available elements. This may happen if all strata are small.")
            # For now, we return what we have. A more sophisticated fill could be added.


        print(f"StyleSampler: Stratified sampling complete. Selected {len(final_selected_samples)} elements out of {total_elements_available} from {len(elements_by_dir)} directories.")
        return final_selected_samples

if __name__ == '__main__':
    # This is a placeholder for where you'd import CodeSample if it's moved
    # from dataclasses import dataclass # Already imported at top level

    # @dataclass # CodeSample is defined globally now
    # class CodeSample: # Redefine for standalone execution if not imported
    #     file_path: Path
    #     item_name: str
    #     item_type: str
    #     code_snippet: str
    #     start_line: int
    #     end_line: int

    # Example Usage (assuming you have a Python project to scan)
    # Create a dummy project structure for testing
    dummy_repo_path = Path("dummy_style_sampler_repo")
    dummy_repo_path.mkdir(exist_ok=True)

    # Create some dummy files and directories
    (dummy_repo_path / "module1").mkdir(exist_ok=True)
    (dummy_repo_path / "module2").mkdir(exist_ok=True)

    with open(dummy_repo_path / "main.py", "w") as f:
        f.write("""
"""This is the main module docstring."""
import os

class MainClass:
    """A class in the main module."""
    def __init__(self):
        self.x = 10

    def main_method(self):
        """A method in MainClass."""
        print("Hello from main_method")

def top_level_func():
    """A top-level function."""
    pass
""")

    with open(dummy_repo_path / "module1" / "utils.py", "w") as f:
        f.write("""
"""Utils module docstring."""
def util_func1():
    """Utility function 1."""
    return "util1"

class HelperClass:
    """A helper class in utils."""
    pass
""")

    with open(dummy_repo_path / "module2" / "services.py", "w") as f:
        f.write("""
# No module docstring here
def service_call_alpha():
    # No function docstring
    print("Alpha service")

class ServiceOne:
    # No class docstring
    def run(self): # No method docstring
        pass
""")

    # Add a venv to test filtering
    (dummy_repo_path / ".venv").mkdir(exist_ok=True)
    with open(dummy_repo_path / ".venv" / "ignored.py", "w") as f:
        f.write("print('this should be ignored')")


    print(f"Attempting to sample from: {dummy_repo_path.resolve()}")

    # IMPORTANT: Provide actual or placeholder paths for models for __main__ execution
    deepseek_path_main = "path/to/your/deepseek.gguf" # Placeholder
    divot5_path_main = "path/to/your/divot5_model_dir"   # Placeholder

    # Since the AI functions will likely use fallbacks if paths are invalid,
    # this __main__ can still run to test the sampler's structure.
    print(f"Note: Using placeholder model paths for __main__ example: "
          f"DS: {deepseek_path_main}, D5: {divot5_path_main}")
    print("AI fingerprinting will likely use mock/fallback responses.")


    try:
        sampler = StyleSampler(
            str(dummy_repo_path),
            deepseek_model_path=deepseek_path_main,
            divot5_model_path=divot5_path_main,
            target_samples=5,
            random_seed=42
        )

        # sampler._discover_python_files() is called by collect_all_elements
        # print(f"Discovered files: {[str(p.relative_to(dummy_repo_path)) for p in sampler._all_py_files]}")

        sampled_elements = sampler.sample_elements() # This will call collect_all_elements which calls _extract_elements_from_file

        print(f"\nCollected {len(sampler._collected_elements)} elements in total.")
        print(f"Sampled {len(sampled_elements)} elements:")
        for i, sample in enumerate(sampled_elements):
            print(f"  Sample {i+1}: {sample.item_type} '{sample.item_name}' from {sample.file_path.name}")
            print(f"    Lines: {sample.start_line}-{sample.end_line}")
            # print(f"    Snippet: \n{sample.code_snippet[:80]}...") # Potentially long
            if sample.ai_fingerprint:
                print(f"    AI Fingerprint Status: {sample.ai_fingerprint.get('validation_status')}")
                if sample.ai_fingerprint.get('validation_status') == 'passed':
                    print(f"      Indent: {sample.ai_fingerprint.get('indent')}, Quotes: {sample.ai_fingerprint.get('quotes')}")
            if sample.error_while_fingerprinting:
                print(f"    Error during fingerprinting: {sample.error_while_fingerprinting}")


    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        # Clean up dummy repo
        import shutil
        if dummy_repo_path.exists(): # Ensure it exists before trying to remove
            shutil.rmtree(dummy_repo_path)
            print(f"Cleaned up dummy repo: {dummy_repo_path.resolve()}")
        #pass
