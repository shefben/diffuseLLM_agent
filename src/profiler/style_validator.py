import subprocess
import sqlite3
import ast
import tokenize
import io
import re
import shutil # For shutil.which, though direct config access will be preferred
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Set

# Assuming these are importable.
from src.profiler.naming_conventions import NAMING_CONVENTIONS_REGEX, is_identifier_matching_convention

# Define triple quote strings safely for internal use if needed by logic below
TRIPLE_SINGLE_QUOTE_STR = chr(39) * 3
TRIPLE_DOUBLE_QUOTE_STR = chr(34) * 3

class AstIdentifierExtractor(ast.NodeVisitor):
    """Extracts relevant identifiers (definitions) and their types from AST."""
    def __init__(self):
        self.identifiers: List[Tuple[str, str, int]] = [] # name, type, line_no

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.identifiers.append((node.name, "function", node.lineno))
        for arg in node.args.args: self.identifiers.append((arg.arg, "parameter", arg.lineno))
        if node.args.vararg: self.identifiers.append((node.args.vararg.arg, "parameter", node.args.vararg.lineno))
        if node.args.kwarg: self.identifiers.append((node.args.kwarg.arg, "parameter", node.args.kwarg.lineno))
        for kwarg_node in node.args.kwonlyargs: self.identifiers.append((kwarg_node.arg, "parameter", kwarg_node.lineno))
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self.visit_FunctionDef(node)

    def visit_ClassDef(self, node: ast.ClassDef):
        self.identifiers.append((node.name, "class", node.lineno))
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.identifiers.append((target.id, "variable", target.lineno))
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign):
        if isinstance(node.target, ast.Name):
            self.identifiers.append((node.target.id, "variable", node.target.lineno))
        self.generic_visit(node)

class StyleValidatorCore:
    def __init__(self, app_config: Dict[str, Any], style_profile: Dict[str, Any]):
        self.app_config = app_config
        self.style_profile = style_profile
        self.verbose = self.app_config.get("general", {}).get("verbose", False)
        if self.verbose:
            print(f"StyleValidatorCore initialized. Style profile keys: {list(self.style_profile.keys()) if self.style_profile else 'None'}")

    def _get_active_naming_rules(self, db_path: Path) -> Dict[str, Tuple[str, str]]:
        rules: Dict[str, Tuple[str, str]] = {}
        conn = None
        try:
            if not db_path.exists():
                if self.verbose: # Make warning conditional on verbosity
                    print(f"StyleValidatorCore: Naming DB {db_path} not found. No naming checks will be performed.")
                return rules
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            cursor.execute("SELECT identifier_type, convention_name, regex_pattern FROM naming_rules WHERE is_active = TRUE")
            for row in cursor.fetchall():
                id_type, conv_name, regex_p = row
                rules[id_type] = (conv_name, regex_p)
        except sqlite3.Error as e:
            # Keep this print as it's an unexpected error
            print(f"StyleValidatorCore: SQLite error loading naming rules from {db_path}: {e}")
        finally:
            if conn: conn.close()
        return rules

    def score_sample_style(
        self,
        sample_path: Path,
        db_path: Optional[Path] = None, # Naming conventions DB
        w_black_diff: float = 0.25, w_ruff_diff: float = 0.25,
        w_naming: float = 0.20, w_linelen: float = 0.10,
        w_quotes: float = 0.10, w_docstrings: float = 0.10
    ) -> float:
        if not sample_path.is_file():
            print(f"StyleValidatorCore Error: Sample file not found at {sample_path} for scoring.")
            return 1.0 # Max penalty

        try:
            code_content = sample_path.read_text(encoding="utf-8")
            if not code_content.strip(): return 0.0 # Empty file is perfectly styled
        except Exception as e:
            print(f"StyleValidatorCore Error: Error reading sample file {sample_path}: {e}")
            return 1.0 # Max penalty

        total_penalty = 0.0

        # Use self.style_profile, which was project_profile
        effective_style = self.style_profile.copy()
        if self.style_profile.get("directory_overrides"):
            for dir_glob_str, overrides in self.style_profile["directory_overrides"].items():
                try:
                    if str(sample_path).startswith(dir_glob_str) or sample_path.match(dir_glob_str):
                        effective_style.update(overrides)
                        if self.verbose:
                            print(f"StyleValidatorCore: Applied directory override from {dir_glob_str} for {sample_path}")
                        break
                except Exception as e_glob:
                     print(f"StyleValidatorCore Error: Error matching glob {dir_glob_str} with {sample_path}: {e_glob}")

        max_possible_penalty_for_performed_checks = 0.0

        # Formatting Checks
        black_executable = self.app_config.get("tools", {}).get("black_path", "black")
        # Check if the executable path is more than just the command name (i.e., if it's a path)
        # or if shutil.which can find it (if it's just a command name)
        can_run_black = Path(black_executable).is_file() or shutil.which(black_executable)

        if can_run_black:
            max_possible_penalty_for_performed_checks += w_black_diff
            try:
                process = subprocess.run([black_executable, "--check", "--quiet", str(sample_path)], capture_output=True, text=True, check=False)
                if process.returncode != 0: total_penalty += w_black_diff
            except Exception: total_penalty += w_black_diff
        elif self.verbose:
            print(f"StyleValidatorCore Warning: black not found at '{black_executable}' (or not in PATH if just 'black'). Skipping Black check.")

        ruff_executable = self.app_config.get("tools", {}).get("ruff_path", "ruff")
        can_run_ruff = Path(ruff_executable).is_file() or shutil.which(ruff_executable)

        if can_run_ruff:
            max_possible_penalty_for_performed_checks += w_ruff_diff
            try:
                process = subprocess.run([ruff_executable, "check", "--quiet", "--no-fix", str(sample_path)], capture_output=True, text=True, check=False)
                if process.returncode == 1:
                    total_penalty += w_ruff_diff
                elif process.returncode != 0 :
                    total_penalty += w_ruff_diff
            except Exception: total_penalty += w_ruff_diff
        elif self.verbose:
            print(f"StyleValidatorCore Warning: ruff not found at '{ruff_executable}' (or not in PATH if just 'ruff'). Skipping Ruff check.")

        active_naming_rules = {}
        if db_path: # Only attempt to get rules if db_path is provided
            active_naming_rules = self._get_active_naming_rules(db_path)
        elif self.verbose:
            print("StyleValidatorCore: No db_path provided for naming conventions. Skipping naming checks.")

        try:
            tree = ast.parse(code_content, filename=str(sample_path))

            if active_naming_rules: # Only proceed if rules were loaded
                max_possible_penalty_for_performed_checks += w_naming
                extractor = AstIdentifierExtractor() # Module-level class
                extractor.visit(tree)
                violations = 0
                relevant_identifiers_to_check = [
                    id_info for id_info in extractor.identifiers
                    if id_info[1] in ["function", "class", "variable", "parameter"] and not id_info[0].startswith("__")
                ]
                if relevant_identifiers_to_check:
                    for name, id_type_from_ast, line_no in relevant_identifiers_to_check:
                        rule_key = id_type_from_ast
                        if id_type_from_ast == "variable":
                            const_rule_info = active_naming_rules.get("constant")
                            if const_rule_info and is_identifier_matching_convention(name, const_rule_info[0]):
                                 rule_key = "constant"
                        target_convention_info = active_naming_rules.get(rule_key)
                        if target_convention_info:
                            target_convention_name, _ = target_convention_info
                            if not is_identifier_matching_convention(name, target_convention_name):
                                violations += 1
                    total_penalty += w_naming * (violations / len(relevant_identifiers_to_check))

            max_len_profile = effective_style.get("max_line_length")
            if isinstance(max_len_profile, int) and max_len_profile > 0:
                max_possible_penalty_for_performed_checks += w_linelen
                lines = code_content.splitlines()
                if lines:
                    long_lines_count = sum(1 for line in lines if len(line) > max_len_profile)
                    total_penalty += w_linelen * (long_lines_count / len(lines))

            preferred_quote_style = effective_style.get("preferred_quotes")
            if preferred_quote_style in ["single", "double"]:
                max_possible_penalty_for_performed_checks += w_quotes
                single_q_count, double_q_count = 0, 0
                try:
                    code_bytes_for_quotes = code_content.encode('utf-8')
                    token_gen = tokenize.tokenize(io.BytesIO(code_bytes_for_quotes).readline)
                    for token_info in token_gen:
                        if token_info.type == tokenize.STRING:
                            token_string = token_info.string
                            prefix_len = 0
                            lowered_token_string = token_string.lower()
                            if lowered_token_string.startswith(("rf", "fr")): prefix_len = 2
                            elif lowered_token_string.startswith(("r", "f", "u", "b")): prefix_len = 1
                            actual_string_part = token_string[prefix_len:]
                            if actual_string_part.startswith(TRIPLE_SINGLE_QUOTE_STR) or \
                               (not actual_string_part.startswith(TRIPLE_DOUBLE_QUOTE_STR) and actual_string_part.startswith("'")):
                                single_q_count += 1
                            elif actual_string_part.startswith(TRIPLE_DOUBLE_QUOTE_STR) or \
                                 (not actual_string_part.startswith(TRIPLE_SINGLE_QUOTE_STR) and actual_string_part.startswith('"')):
                                double_q_count += 1
                    total_quotes = single_q_count + double_q_count
                    quote_penalty = 0.0
                    if total_quotes > 0:
                        if preferred_quote_style == "single":
                            quote_penalty = w_quotes * (double_q_count / total_quotes)
                        elif preferred_quote_style == "double":
                            quote_penalty = w_quotes * (single_q_count / total_quotes)
                    total_penalty += quote_penalty
                except tokenize.TokenError:
                    total_penalty += w_quotes # Max penalty if tokenization fails

            docstring_style_from_profile = effective_style.get("docstring_style")
            if docstring_style_from_profile and docstring_style_from_profile != "none":
                max_possible_penalty_for_performed_checks += w_docstrings
                public_elements_total = 0
                public_elements_missing_docstrings = 0
                public_elements_bad_style = 0 # Initialize bad style counter
                for node in ast.walk(tree):
                    is_public_definable = False
                    node_name = ""
                    docstring = None # Initialize docstring here
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        node_name = node.name
                        if not node_name.startswith("_"):
                            is_public_definable = True
                    elif isinstance(node, ast.Module):
                        is_public_definable = True

                    if is_public_definable:
                        public_elements_total += 1
                        docstring = ast.get_docstring(node, clean=False)
                        if not docstring:
                            public_elements_missing_docstrings += 1
                        else:
                            # Basic Docstring Style Conformity Check
                            style_violation = False
                            expected_doc_style = effective_style.get("docstring_style") # Already fetched as docstring_style_from_profile

                            if expected_doc_style == "google":
                                expects_args_section = False
                                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                    if node.args.args or node.args.vararg or node.args.kwarg or node.args.posonlyargs or node.args.kwonlyargs:
                                        expects_args_section = True

                                expects_returns_section = False
                                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.returns:
                                    is_none_return = False
                                    if isinstance(node.returns, ast.Constant) and node.returns.value is None: is_none_return = True
                                    elif isinstance(node.returns, ast.Name) and node.returns.id == "None": is_none_return = True
                                    if not is_none_return: expects_returns_section = True

                                if len(docstring.splitlines()) > 3: # Only apply to longer docstrings
                                    has_args_tag = "Args:" in docstring or "Arguments:" in docstring
                                    has_returns_tag = "Returns:" in docstring or "Yields:" in docstring

                                    if expects_args_section and not has_args_tag:
                                        style_violation = True
                                    if expects_returns_section and not has_returns_tag and not style_violation:
                                        style_violation = True
                                # For short docstrings (<=3 lines), the original simple check (or lack thereof) is implicitly less strict.
                                # The subtask's goal was to refine for longer ones.

                            elif expected_doc_style == "numpy":
                                expects_params_section = False
                                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                    if node.args.args or node.args.vararg or node.args.kwarg or node.args.posonlyargs or node.args.kwonlyargs:
                                        expects_params_section = True

                                expects_returns_section = False
                                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.returns:
                                    is_none_return = False
                                    if isinstance(node.returns, ast.Constant) and node.returns.value is None: is_none_return = True
                                    elif isinstance(node.returns, ast.Name) and node.returns.id == "None": is_none_return = True
                                    if not is_none_return: expects_returns_section = True

                                if len(docstring.splitlines()) > 4: # NumPy is more verbose, allow more lines for simple summary
                                    if expects_params_section:
                                        found_params_section = bool(re.search(r"^\s*Parameters\s*\n\s*-+\s*\n", docstring, re.MULTILINE | re.IGNORECASE))
                                        if not found_params_section:
                                            style_violation = True

                                    if not style_violation and expects_returns_section:
                                        found_returns_section = bool(re.search(r"^\s*Returns\s*\n\s*-+\s*\n", docstring, re.MULTILINE | re.IGNORECASE))
                                        if not found_returns_section:
                                            style_violation = True

                            if style_violation:
                                public_elements_bad_style += 1

                if public_elements_total > 0:
                    effective_docstring_violations = public_elements_missing_docstrings + (0.5 * public_elements_bad_style)
                    docstring_penalty_factor = effective_docstring_violations / public_elements_total
                    total_penalty += w_docstrings * docstring_penalty_factor

        except ast.ASTError:
            if self.verbose:
                print(f"StyleValidatorCore: AST parsing error for {sample_path}. Applying max penalty for related checks.")
            # Add penalties for checks that would have run if AST parsing succeeded
            if active_naming_rules:
                total_penalty += w_naming
                max_possible_penalty_for_performed_checks += w_naming
            if isinstance(effective_style.get("max_line_length"), int) and effective_style.get("max_line_length",0) > 0 :
                total_penalty += w_linelen
                max_possible_penalty_for_performed_checks += w_linelen
            if effective_style.get("preferred_quotes") in ["single", "double"]:
                total_penalty += w_quotes
                max_possible_penalty_for_performed_checks += w_quotes
            ds_style = effective_style.get("docstring_style")
            if ds_style and ds_style != "none":
                total_penalty += w_docstrings
                max_possible_penalty_for_performed_checks += w_docstrings
        except Exception as e:
            print(f"StyleValidatorCore: Unexpected error during style scoring for {sample_path}: {e}")
            return 1.0

        if max_possible_penalty_for_performed_checks == 0:
            if self.verbose: print("StyleValidatorCore: No style checks performed or applicable. Returning 0.0 score.")
            return 0.0

        final_score = total_penalty / max_possible_penalty_for_performed_checks
        return min(max(final_score, 0.0), 1.0)

# Example usage (module level for testing if needed, ensure it's guarded by if __name__ == '__main__')
# if __name__ == '__main__':
#     # This block would need to be updated to instantiate StyleValidatorCore
#     # and call its score_sample_style method.
#     # For now, keeping it commented out as per previous state.
#     pass
