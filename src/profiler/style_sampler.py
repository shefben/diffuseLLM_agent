import ast
import os
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple, Union, Optional
from dataclasses import dataclass

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
        raw_code_elements_to_fingerprint: List[Tuple[str, str, str, int, int]] = [] # item_name, item_type, snippet, start, end

        try:
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

        except SyntaxError:
            print(f"Syntax error in {file_path}, skipping AI fingerprinting for this file.")
            return [] # No elements if file can't be parsed
        except Exception as e:
            print(f"Could not process {file_path} for element extraction: {e}")
            return []

        # Now, for each extracted raw code element, get its AI fingerprint
        for name, type_val, snippet, start, end in raw_code_elements_to_fingerprint:
            print(f"StyleSampler: Getting AI fingerprint for {type_val} '{name}' in {file_path.name}...")
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
                # Optionally, check ai_fp_dict for 'validation_status' or 'error' keys
                if ai_fp_dict.get("validation_status") != "passed":
                    error_msg = f"AI fingerprinting validation failed: {ai_fp_dict.get('validation_status')}, errors: {ai_fp_dict.get('validation_errors', ai_fp_dict.get('details'))}"
                    print(f"Warning: {error_msg} for {type_val} '{name}'")
                    # Decide if you want to keep the fingerprint dict even if validation failed
                    # For now, we store it along with the error.
            except Exception as e_fp:
                error_msg = f"Exception during AI fingerprinting: {e_fp}"
                print(f"Error: {error_msg} for {type_val} '{name}'")

            elements_in_file.append(CodeSample(
                file_path=file_path,
                item_name=name,
                item_type=type_val,
                code_snippet=snippet, # Store original snippet
                start_line=start,
                end_line=end,
                ai_fingerprint=ai_fp_dict,
                error_while_fingerprinting=error_msg
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
        Performs sampling of the collected code elements (which now include AI fingerprints).
        Stratified sampling logic to be implemented later.
        """
        # Ensure elements are collected (this now does AI fingerprinting)
        if not self._collected_elements: # If not pre-populated by calling collect_all_elements explicitly
            self.collect_all_elements()

        if not self._collected_elements:
            print("StyleSampler: No elements available for sampling.")
            return []

        # Filter out elements that had errors during fingerprinting if needed, or handle them
        # For now, all collected elements (even with errors) are candidates for sampling.
        # A consumer of these samples should check `error_while_fingerprinting` and `ai_fingerprint['validation_status']`.

        if len(self._collected_elements) <= self.target_samples:
            print(f"StyleSampler: Collected {len(self._collected_elements)} elements, which is <= target {self.target_samples}. Returning all.")
            return self._collected_elements
        else:
            # Current: Random sampling. TODO: Implement stratified sampling.
            print(f"StyleSampler: Collected {len(self._collected_elements)} elements. Randomly sampling {self.target_samples}.")
            return random.sample(self._collected_elements, self.target_samples)

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
