import subprocess
import sqlite3
import ast
import tokenize
import io
import re
import shutil # For shutil.which
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

def get_active_naming_rules(db_path: Path) -> Dict[str, Tuple[str, str]]:
    rules: Dict[str, Tuple[str, str]] = {}
    conn = None
    try:
        if not db_path.exists():
            print(f"Warning: Naming DB {db_path} not found for score_style. No naming checks.")
            return rules
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT identifier_type, convention_name, regex_pattern FROM naming_rules WHERE is_active = TRUE")
        for row in cursor.fetchall():
            id_type, conv_name, regex_p = row
            rules[id_type] = (conv_name, regex_p)
    except sqlite3.Error as e:
        print(f"Warning: SQLite error loading naming rules for score_style: {e}")
    finally:
        if conn: conn.close()
    return rules

def score_style(
    sample_path: Path,
    project_profile: Dict[str, Any],
    db_path: Path,
    w_black_diff: float = 0.25, w_ruff_diff: float = 0.25,
    w_naming: float = 0.20, w_linelen: float = 0.10,
    w_quotes: float = 0.10, w_docstrings: float = 0.10
) -> float:
    if not sample_path.is_file():
        print(f"Error: Sample file not found at {sample_path} for scoring.")
        return 1.0

    try:
        code_content = sample_path.read_text(encoding="utf-8")
        if not code_content.strip(): return 0.0 # Empty file is perfectly styled
    except Exception as e:
        print(f"Error reading sample file {sample_path}: {e}")
        return 1.0

    total_penalty = 0.0

    effective_style = project_profile.copy()
    if project_profile.get("directory_overrides"):
        for dir_glob_str, overrides in project_profile["directory_overrides"].items():
            try:
                # Path.match is for relative paths within a root.
                # For absolute paths or more general matching, consider fnmatch or glob.glob
                # A simple startswith can work for basic directory prefix matching.
                if str(sample_path).startswith(dir_glob_str) or sample_path.match(dir_glob_str):
                    effective_style.update(overrides)
                    print(f"Applied directory override from {dir_glob_str} for {sample_path}")
                    break
            except Exception as e_glob:
                 print(f"Error matching glob {dir_glob_str} with {sample_path}: {e_glob}")

    # Max possible penalty based on checks that could actually run
    max_possible_penalty_for_performed_checks = 0.0

    # Formatting Checks
    black_executable = shutil.which("black")
    if black_executable:
        max_possible_penalty_for_performed_checks += w_black_diff
        try:
            process = subprocess.run([black_executable, "--check", "--quiet", str(sample_path)], capture_output=True, text=True, check=False)
            if process.returncode != 0: total_penalty += w_black_diff
        except Exception: total_penalty += w_black_diff # Max penalty if tool fails
    else: print("Warning: black not found, skipping Black check for scoring.")

    ruff_executable = shutil.which("ruff")
    if ruff_executable:
        max_possible_penalty_for_performed_checks += w_ruff_diff
        try:
            # Using `ruff check` as it's the primary way to find style/lint issues.
            # `ruff format --check` could also be used, but `check` is more comprehensive.
            process = subprocess.run([ruff_executable, "check", "--quiet", str(sample_path)], capture_output=True, text=True, check=False)
            if process.returncode == 1: # Issues found by ruff check
                total_penalty += w_ruff_diff
            elif process.returncode != 0 : # Other ruff errors
                total_penalty += w_ruff_diff
        except Exception: total_penalty += w_ruff_diff # Max penalty if tool fails
    else: print("Warning: ruff not found, skipping Ruff check for scoring.")

    active_naming_rules = get_active_naming_rules(db_path)

    ast_parse_successful_for_score = False
    try:
        tree = ast.parse(code_content, filename=str(sample_path))
        ast_parse_successful_for_score = True

        # Naming Conventions
        if active_naming_rules:
            max_possible_penalty_for_performed_checks += w_naming
            extractor = AstIdentifierExtractor()
            extractor.visit(tree)
            violations = 0
            # Focus on key identifier types that are usually styled
            relevant_identifiers_to_check = [
                id_info for id_info in extractor.identifiers
                if id_info[1] in ["function", "class", "variable", "parameter"] and not id_info[0].startswith("__") # Exclude dunders
            ]
            if relevant_identifiers_to_check:
                for name, id_type_from_ast, line_no in relevant_identifiers_to_check:
                    rule_key = id_type_from_ast
                    # Heuristic: if a 'variable' looks like a constant, check against 'constant' rule
                    if id_type_from_ast == "variable":
                        const_rule_info = active_naming_rules.get("constant")
                        if const_rule_info and is_identifier_matching_convention(name, const_rule_info[0]): # Check if it IS the const style
                             rule_key = "constant"

                    target_convention_info = active_naming_rules.get(rule_key)
                    if target_convention_info:
                        target_convention_name, _ = target_convention_info # Regex not needed here, is_identifier_matching_convention handles it
                        if not is_identifier_matching_convention(name, target_convention_name):
                            violations += 1
                total_penalty += w_naming * (violations / len(relevant_identifiers_to_check))

        # Line Length
        max_len_profile = effective_style.get("max_line_length")
        if isinstance(max_len_profile, int) and max_len_profile > 0:
            max_possible_penalty_for_performed_checks += w_linelen
            lines = code_content.splitlines()
            if lines:
                long_lines_count = sum(1 for line in lines if len(line) > max_len_profile)
                total_penalty += w_linelen * (long_lines_count / len(lines))

        # Quote Usage
        preferred_quote_style = effective_style.get("preferred_quotes")
        if preferred_quote_style in ["single", "double"]:
            max_possible_penalty_for_performed_checks += w_quotes
            single_q_count, double_q_count = 0, 0
            try:
                code_bytes_for_quotes = code_content.encode('utf-8')
                token_gen = tokenize.tokenize(io.BytesIO(code_bytes_for_quotes).readline)
                for token_info in token_gen:
                    if token_info.type == tokenize.STRING:
                        # Basic check, doesn't perfectly handle prefixes like r"", u"" for quote char itself
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
                if total_quotes > 0:
                    # Penalize if the non-preferred style is dominant
                    if preferred_quote_style == "single" and double_q_count > single_q_count:
                        total_penalty += w_quotes * (double_q_count / total_quotes)
                    elif preferred_quote_style == "double" and single_q_count > double_q_count:
                        total_penalty += w_quotes * (single_q_count / total_quotes)
            except tokenize.TokenError:
                total_penalty += w_quotes # Penalize if tokenization fails for quote check

        # Docstrings: Penalize missing docstrings for public functions/classes/methods
        docstring_style_from_profile = effective_style.get("docstring_style")
        # This check only verifies presence, not style conformity (which is complex)
        if docstring_style_from_profile and docstring_style_from_profile != "none":
            max_possible_penalty_for_performed_checks += w_docstrings
            public_elements_total = 0
            public_elements_missing_docstrings = 0
            for node in ast.walk(tree):
                is_public_definable = False
                node_name = ""
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    node_name = node.name
                    if not node_name.startswith("_"): # Public if not starting with single underscore
                        is_public_definable = True
                elif isinstance(node, ast.Module): # Module itself
                    is_public_definable = True

                if is_public_definable:
                    public_elements_total += 1
                    if not ast.get_docstring(node, clean=False):
                        public_elements_missing_docstrings += 1

            if public_elements_total > 0:
                total_penalty += w_docstrings * (public_elements_missing_docstrings / public_elements_total)

    except ast.ASTError: # If AST parsing fails, apply full penalty for AST-dependent checks
        print(f"AST parsing error for {sample_path}. Applying max penalty for related checks.")
        if active_naming_rules: total_penalty += w_naming
        if isinstance(effective_style.get("max_line_length"), int) and effective_style.get("max_line_length",0) > 0:
             total_penalty += w_linelen
        if effective_style.get("preferred_quotes") in ["single", "double"]:
             total_penalty += w_quotes
        ds_style = effective_style.get("docstring_style")
        if ds_style and ds_style != "none":
             total_penalty += w_docstrings
        # Also update max_possible_penalty for checks that would have run if AST parsing succeeded
        if active_naming_rules: max_possible_penalty_for_performed_checks += w_naming
        if isinstance(effective_style.get("max_line_length"), int) and effective_style.get("max_line_length",0) > 0 :
            max_possible_penalty_for_performed_checks += w_linelen
        if effective_style.get("preferred_quotes") in ["single", "double"]:
            max_possible_penalty_for_performed_checks += w_quotes
        if ds_style and ds_style != "none":
            max_possible_penalty_for_performed_checks += w_docstrings

    except Exception as e:
        print(f"Unexpected error during style scoring for {sample_path}: {e}")
        return 1.0 # Max penalty for unexpected errors

    if max_possible_penalty_for_performed_checks == 0: return 0.0

    final_score = total_penalty / max_possible_penalty_for_performed_checks
    return min(max(final_score, 0.0), 1.0)

# (Main block for testing, similar to before, ensuring it's commented out or safe for subtask)
# if __name__ == '__main__':
#     pass
