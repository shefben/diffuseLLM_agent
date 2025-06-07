import ast
import os
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple, Union, Optional
from dataclasses import dataclass # Added import for dataclass

# Define a structure for a code element that can be sampled
# For now, let's represent a sample as a dictionary or a simple class.
# It should contain the code snippet, its type (function, class, docstring),
# file path, and maybe line numbers.

@dataclass
class CodeSample:
    file_path: Path
    item_name: str # e.g., function name, class name, or 'module_docstring'/'ClassName.docstring'
    item_type: str # 'function', 'class', 'docstring'
    code_snippet: str # The actual source code of the function/class or the docstring text
    start_line: int
    end_line: int
    # Could add 'parent_path' for stratification if item_name isn't fully qualified

class StyleSampler:
    def __init__(self, repo_path: Union[str, Path], target_samples: int = 1000, random_seed: Optional[int] = None):
        self.repo_path = Path(repo_path)
        if not self.repo_path.is_dir():
            raise ValueError(f"Repository path {repo_path} is not a valid directory.")
        self.target_samples = target_samples
        if random_seed is not None:
            random.seed(random_seed)

        self._all_py_files: List[Path] = []
        self._collected_elements: List[CodeSample] = [] # This will hold all parseable elements

    def _discover_python_files(self):
        """Discovers all Python files in the repository path."""
        self._all_py_files = list(self.repo_path.rglob("*.py"))
        # Filter out files in common virtual environment folders or other ignored paths
        # This is a basic filter, could be made more robust
        common_ignore_dirs = [".venv", "venv", ".env", "env", "node_modules", ".git"]

        filtered_files = []
        for py_file in self._all_py_files:
            try:
                # Check if any part of the path is in common_ignore_dirs
                if not any(part in common_ignore_dirs for part in py_file.parts):
                    filtered_files.append(py_file)
            except Exception:
                # Path operations might fail in rare cases, skip such files
                pass
        self._all_py_files = filtered_files

    def _extract_elements_from_file(self, file_path: Path) -> List[CodeSample]:
        """
        Parses a single Python file and extracts functions, classes,
        and their docstrings as CodeSample objects.
        (This will be implemented in a subsequent step)
        """
        # Placeholder for AST parsing and element extraction
        # This method will populate self._collected_elements
        # For now, it returns an empty list.
        # Actual implementation will use ast.parse, walk the tree,
        # identify ast.FunctionDef, ast.ClassDef, and use ast.get_docstring.
        # It will need to read the file content and extract exact code snippets.
        elements_in_file: List[CodeSample] = []

        # --- Start of placeholder logic for demonstration ---
        # This is a very simplified placeholder.
        # A real implementation would parse the file with AST.
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source_lines = f.readlines()
                source_code = "".join(source_lines)

            tree = ast.parse(source_code, filename=str(file_path))

            # Example: Extract module docstring if present
            module_docstring = ast.get_docstring(tree)
            if module_docstring:
                # Approximating line numbers for module docstring can be tricky.
                # It's usually at the top. Let's assume it ends before the first non-docstring node.
                # For a simple placeholder, let's just grab the first few lines if it's a string.
                # A more accurate way would be to find the AST node for the docstring.
                # For now, let's assume it's the first element if it's a string literal.
                if tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant) and isinstance(tree.body[0].value.value, str):
                     # This is ast.Str in older Python, ast.Constant in 3.8+
                    doc_node = tree.body[0]
                    elements_in_file.append(CodeSample(
                        file_path=file_path,
                        item_name=f"{file_path.name}::module_docstring",
                        item_type="docstring",
                        code_snippet=module_docstring,
                        start_line=doc_node.lineno,
                        end_line=doc_node.end_lineno if hasattr(doc_node, 'end_lineno') else doc_node.lineno
                    ))

            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    # Get the source segment for the function
                    # ast.get_source_segment is available in Python 3.8+
                    # For older versions, manual line extraction is needed.
                    try:
                        func_source = ast.get_source_segment(source_code, node)
                        if func_source: # Can be None if source not available
                             elements_in_file.append(CodeSample(
                                file_path=file_path,
                                item_name=node.name,
                                item_type="function",
                                code_snippet=func_source,
                                start_line=node.lineno,
                                end_line=node.end_lineno
                            ))
                        func_docstring = ast.get_docstring(node)
                        if func_docstring:
                            # Locate docstring node for precise line numbers if possible
                            # For simplicity, we use the function's start line for its docstring for now
                            elements_in_file.append(CodeSample(
                                file_path=file_path,
                                item_name=f"{node.name}::docstring",
                                item_type="docstring",
                                code_snippet=func_docstring,
                                start_line=node.lineno, # This is not accurate for the docstring itself
                                end_line=node.lineno   # Needs refinement
                            ))
                    except Exception: # get_source_segment can fail
                        pass


                elif isinstance(node, ast.ClassDef):
                    try:
                        class_source = ast.get_source_segment(source_code, node)
                        if class_source:
                            elements_in_file.append(CodeSample(
                                file_path=file_path,
                                item_name=node.name,
                                item_type="class",
                                code_snippet=class_source,
                                start_line=node.lineno,
                                end_line=node.end_lineno
                            ))
                        class_docstring = ast.get_docstring(node)
                        if class_docstring:
                             elements_in_file.append(CodeSample(
                                file_path=file_path,
                                item_name=f"{node.name}::docstring",
                                item_type="docstring",
                                code_snippet=class_docstring,
                                start_line=node.lineno, # Not accurate
                                end_line=node.lineno    # Needs refinement
                            ))
                    except Exception:
                        pass
        except SyntaxError:
            # print(f"Syntax error in {file_path}, skipping.")
            pass # Skip files with syntax errors
        except Exception:
            # print(f"Could not process {file_path}: {e}")
            pass # Skip other errors

        # --- End of placeholder logic ---
        return elements_in_file


    def collect_all_elements(self):
        """Collects all code elements from all Python files."""
        self._discover_python_files()
        self._collected_elements = []
        for py_file in self._all_py_files:
            self._collected_elements.extend(self._extract_elements_from_file(py_file))

        if not self._collected_elements and self._all_py_files:
            print("Warning: No code elements (functions, classes, docstrings) were extracted, "
                  "though Python files were found. The AST parsing logic might need attention "
                  "or files might be empty/non-standard.")


    def sample_elements(self) -> List[CodeSample]:
        """
        Performs stratified sampling of the collected code elements.
        (This will be implemented in a subsequent step)
        """
        self.collect_all_elements() # Ensure elements are collected

        if not self._collected_elements:
            return []

        if len(self._collected_elements) <= self.target_samples:
            return self._collected_elements # Return all if fewer than target

        # Placeholder for stratified sampling logic
        # For now, use random sampling if collected_elements > target_samples
        return random.sample(self._collected_elements, self.target_samples)

if __name__ == '__main__':
    # This is a placeholder for where you'd import CodeSample if it's moved
    from dataclasses import dataclass

    @dataclass
    class CodeSample: # Redefine for standalone execution if not imported
        file_path: Path
        item_name: str
        item_type: str
        code_snippet: str
        start_line: int
        end_line: int

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
    try:
        sampler = StyleSampler(str(dummy_repo_path), target_samples=5, random_seed=42)

        print(f"Discovered files: {[str(p.relative_to(dummy_repo_path)) for p in sampler._all_py_files]}") # Before collection

        sampled_elements = sampler.sample_elements()

        print(f"\nCollected {len(sampler._collected_elements)} elements in total.")
        print(f"Sampled {len(sampled_elements)} elements:")
        for i, sample in enumerate(sampled_elements):
            print(f"  Sample {i+1}:")
            print(f"    File: {sample.file_path.relative_to(dummy_repo_path)}")
            print(f"    Name: {sample.item_name}")
            print(f"    Type: {sample.item_type}")
            # print(f"    Snippet: \n{sample.code_snippet[:100]}...") # Potentially long
            print(f"    Lines: {sample.start_line}-{sample.end_line}")

    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        # Clean up dummy repo
        import shutil
        shutil.rmtree(dummy_repo_path) # Comment out to inspect dummy repo
        #pass
