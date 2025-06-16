from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple # Added Tuple
import libcst as cst
import ast # Keep ast import
from dataclasses import dataclass, field
import sys
import os
import re # For CallGraphVisitor._resolve_callee_fqn
from collections import defaultdict, deque # Ensure deque is explicitly imported
import collections # Added for deque in get_call_graph_slice

# Watchdog imports
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileModifiedEvent, FileCreatedEvent, FileDeletedEvent, FileMovedEvent
except ImportError:
    Observer = None # type: ignore
    FileSystemEventHandler = None # type: ignore
    FileModifiedEvent = None # type: ignore
    FileCreatedEvent = None # type: ignore
    FileDeletedEvent = None # type: ignore
    FileMovedEvent = None # type: ignore
    print("Warning: watchdog library not found. File system event monitoring will be disabled.")

# SentenceTransformer and numpy imports
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None # type: ignore
    print("Warning: sentence-transformers library not found. Embedding generation will be disabled.")

try:
    import numpy as np
    NumpyNdarray = np.ndarray
except ImportError:
    np = None # type: ignore
    NumpyNdarray = Any # Fallback type hint
    print("Warning: numpy library not found. Embeddings may not work as expected.")

# Add FAISS import
try:
    import faiss
except ImportError:
    faiss = None # type: ignore
    print("Warning: faiss library not found. FAISS indexing will be disabled.")


# Tree-sitter imports
try:
    from tree_sitter import Parser, Language
    from tree_sitter_languages import get_language, get_parser
except ImportError:
    Parser, Language, get_language, get_parser = None, None, None, None # type: ignore
TreeSitterTree = Any
if Language: # Check if Language was successfully imported
    from tree_sitter import Tree as TreeSitterTreeTypeActual # type: ignore
    TreeSitterTree = TreeSitterTreeTypeActual

# Pyanalyze imports
try:
    import pyanalyze
    from pyanalyze import name_check_visitor
    from pyanalyze.value import Value, KnownValue, TypedValue, dump_value as pyanalyze_dump_value
    from pyanalyze.checker import Checker
    from pyanalyze.config import Config
    from pyanalyze.options import Options
except ImportError:
    pyanalyze = None; name_check_visitor = None; Checker = None; pyanalyze_dump_value = None; Config = None; Options = None # type: ignore
    print("Warning: pyanalyze library not found. Type inference via Pyanalyze will be disabled.")

# NEW: Import from graph_structures
from .graph_structures import CallGraph, ControlDependenceGraph, DataDependenceGraph, NodeID # NodeID might be a NewType
from .signature_trie import SignatureTrie, generate_function_signature_string # NEW import
from src.planner.phase_model import Phase # For type hinting phase_ctx
# LibCST metadata providers for graph building
from libcst.metadata import ParentNodeProvider, PositionProvider, QualifiedNameProvider, ScopeProvider

    if TYPE_CHECKING: # Keep NodeID under TYPE_CHECKING if it's just for hints
        from .graph_structures import NodeID # Ensure NodeID is available for hints

@dataclass
class ParsedFileResult:
    file_path: Path
    source_code: str
    libcst_module: Optional[cst.Module] = None
    treesitter_tree: Optional[TreeSitterTree] = None
    libcst_error: Optional[str] = None
    treesitter_has_errors: bool = False
    treesitter_error_message: Optional[str] = None
    type_info: Optional[Dict[str, Any]] = None
    extracted_symbols: List[Dict[str, Any]] = field(default_factory=list)


if FileSystemEventHandler: # Only define if watchdog was imported
    class DigesterFileSystemEventHandler(FileSystemEventHandler): # type: ignore
        def __init__(self, digester_instance: 'RepositoryDigester'):
            super().__init__()
            self.digester_instance = digester_instance
            self._ignore_dirs = {".venv", "venv", ".git", "__pycache__", "node_modules", ".env", "env"} # Common dirs to ignore events from

        def _is_relevant_path(self, path_str: str) -> bool:
            p = Path(path_str)
            if not p.name.endswith(".py"):
                return False
            try:
                # Check if any part of the path relative to repo root is in ignore_dirs
                # This assumes digester_instance.repo_path is available and absolute
                relative_to_repo = p.relative_to(self.digester_instance.repo_path)
                if any(part in self._ignore_dirs for part in relative_to_repo.parts[:-1]): # Check parent directories
                    return False
            except ValueError: # Path is not under repo_path, or other issue
                return False # Should not happen if watchdog is watching repo_path
            return True

        def dispatch(self, event):
            # Filter non-.py files and events from ignored directories before specific handlers are called
            if event.is_directory: # Ignore directory events
                return

            # For moved events, check both src and dest if applicable
            # Only dispatch further if at least one of the paths is relevant
            if isinstance(event, FileMovedEvent): # type: ignore
                if not self._is_relevant_path(event.src_path) and not self._is_relevant_path(event.dest_path):
                    return
            elif not self._is_relevant_path(event.src_path): # For other events, check src_path
                return

            # If relevant, call the original dispatch to route to on_modified etc.
            super().dispatch(event)


        def on_created(self, event: FileCreatedEvent): # type: ignore
            # is_relevant_path check is implicitly handled by dispatch now
            if not event.is_directory: # Redundant check, but safe
                if self.digester_instance.verbose: print(f"Watchdog: Detected creation: {event.src_path}")
                self.digester_instance.handle_file_event("created", Path(event.src_path))

        def on_modified(self, event: FileModifiedEvent): # type: ignore
            if not event.is_directory:
                if self.digester_instance.verbose: print(f"Watchdog: Detected modification: {event.src_path}")
                self.digester_instance.handle_file_event("modified", Path(event.src_path))

        def on_deleted(self, event: FileDeletedEvent): # type: ignore
            if not event.is_directory:
                if self.digester_instance.verbose: print(f"Watchdog: Detected deletion: {event.src_path}")
                self.digester_instance.handle_file_event("deleted", Path(event.src_path))

        def on_moved(self, event: FileMovedEvent): # type: ignore
            if not event.is_directory:
                if self.digester_instance.verbose: print(f"Watchdog: Detected move: {event.src_path} to {event.dest_path}")
                self.digester_instance.handle_file_event("moved", Path(event.src_path), Path(event.dest_path))
else: # Fallback if FileSystemEventHandler is None
    class DigesterFileSystemEventHandler: # type: ignore
            def __init__(self, digester_instance: 'RepositoryDigester'): pass


class SymbolAndSignatureExtractorVisitor(ast.NodeVisitor): # Renamed
    def __init__(self, module_qname: str, file_path_str: str,
                 source_code_lines: List[str],
                 type_info_map_for_resolver: Optional[Dict[str, str]],
                 signature_trie_ref: SignatureTrie,
                 digester_ref: 'RepositoryDigester'): # Added digester_ref
        self.module_qname: str = module_qname
        self.file_path_str: str = file_path_str
        self.source_code_lines: List[str] = source_code_lines
        self.symbols_for_embedding: List[Dict[str, Any]] = [] # Renamed from self.symbols
        self.current_class_name: Optional[str] = None # FQN of current class context
        self.type_info_map_for_resolver = type_info_map_for_resolver if type_info_map_for_resolver else {}
        self.signature_trie_ref = signature_trie_ref
        self.digester_ref = digester_ref # Stored digester_ref

    def _local_type_resolver(self, node: ast.AST, category_hint: str) -> Optional[str]:
        # PyanalyzeTypeExtractionVisitor keys:
        # - Variables: f"{name_of_variable}:{category_from_pyanalyze_visitor}:{node.lineno}:{node.col_offset}"
        #   e.g., "x:variable_definition:1:0"
        # - Parameters: f"{func_name}.{param_name}:parameter:{node.lineno}:{node.col_offset}"
        #   e.g., "foo.a:parameter:1:8"
        # - Return annotations: f"{func_name}.<return_annotation>:function_return_annotation_type:{node.lineno}:{node.col_offset}"
        #   (node here is the annotation node itself, e.g. ast.Name(id='int'))

        # category_hint from generate_function_signature_string is like:
        # "func_name.param_name:parameter_posonly", "func_name.param_name:parameter", etc.
        # "func_name:return_annotation"

        # PyanalyzeTypeExtractionVisitor now stores keys like:
        # Parameter: "module.ClassName.func_name.param_name:parameter:line:col"
        # Return:    "module.ClassName.func_name.<return>:function_return:line:col"

        # `category_hint` comes from `generate_function_signature_string` and is like:
        #   "FuncName.param_name:parameter_posonly"
        #   "FuncName:return_annotation"
        # `node` is the ast.arg node for parameters, or ast.Name/Attribute for return annotation.

        hint_name_part, hint_category_suffix = category_hint.split(":", 1) # e.g., "FuncName.param_name", "parameter_posonly" or "FuncName", "return_annotation"

        # Determine the actual category PyanalyzeTypeExtractionVisitor would have used.
        pyanalyze_category = ""
        if hint_category_suffix.startswith("parameter"):
            pyanalyze_category = "parameter" # Covers posonly, regular, kwonly
            if "vararg" in hint_category_suffix:
                pyanalyze_category = "parameter_vararg"
            elif "kwarg" in hint_category_suffix:
                pyanalyze_category = "parameter_kwarg"
        elif hint_category_suffix == "return_annotation":
            pyanalyze_category = "function_return"
            # hint_name_part from generate_function_signature_string is "FuncName"
            # PyanalyzeTypeExtractionVisitor uses "FuncName.<return>" as the name_qualifier for _add_type_info
            hint_name_part = f"{hint_name_part}.<return>"


        # Construct the FQN prefix based on the current context of SymbolAndSignatureExtractorVisitor
        current_fqn_prefix = self.module_qname
        if self.current_class_name: # self.current_class_name is already an FQN like module.class
            current_fqn_prefix = self.current_class_name

        # hint_name_part is like "FuncName.param_name" or "FuncName.<return>"
        # This needs to be prefixed with the module/class FQN.
        fully_qualified_name_part = f"{current_fqn_prefix}.{hint_name_part}"

        # If hint_name_part already contains the class name correctly (e.g. from nested calls in signature gen),
        # ensure no double prefixing.
        # This can get complex if current_class_name is for an outer class and hint_name_part refers to an inner class method.
        # For now, assume hint_name_part is relative to current_fqn_prefix context.

        # Example: module_qname = "my_module", current_class_name = "my_module.MyClass"
        # hint_name_part for param 'p' of method 'meth' = "meth.p"
        # fully_qualified_name_part = "my_module.MyClass.meth.p"
        # pyanalyze_category = "parameter"
        # key_to_lookup = "my_module.MyClass.meth.p:parameter:lineno:coloffset"

        key_to_lookup = f"{fully_qualified_name_part}:{pyanalyze_category}:{node.lineno}:{node.col_offset}"

        resolved_type = self.type_info_map_for_resolver.get(key_to_lookup)

        if verbose_resolver := False: # Set to True for debugging this resolver
             print(f"Resolver DEBUG:\n  Context: module='{self.module_qname}', class='{self.current_class_name}'")
             print(f"  Input node: {type(node)}, lineno: {node.lineno}, col: {node.col_offset}")
             print(f"  Input category_hint: '{category_hint}' -> name_part='{hint_name_part}', cat_suffix='{hint_category_suffix}'")
             print(f"  Derived pyanalyze_category: '{pyanalyze_category}'")
             print(f"  Constructed FQN part for key: '{fully_qualified_name_part}'")
             print(f"  Attempted lookup key: '{key_to_lookup}'")
             print(f"  Resolved type: '{resolved_type}'")

        if resolved_type:
            canonical_type_str = resolved_type
            original_for_log = resolved_type # For logging if changed

            # Normalize typing.Optional[X] to typing.Union[X, None] and sort members
            # Pyanalyze's dump_value usually produces 'X | None' or 'Optional[X]'
            # This aims to standardize to 'typing.Union[X, None]' with sorted members (None last)
            # or just 'X' if Union[X, None] simplifies to X (e.g. if X is Any)
            # or just 'None' if Union[None, None]

            # Step 1: Replace " | None" with a temporary structure for Union processing
            # and "Optional[X]" with "Union[X, None]" equivalent
            if canonical_type_str.endswith(" | None"):
                # Convert "X | None" to "typing.Union[X, None]" for consistent processing
                inner_type = canonical_type_str[:-7].strip()
                canonical_type_str = f"typing.Union[{inner_type}, None]"
            elif canonical_type_str.startswith("typing.Optional[") and canonical_type_str.endswith("]"):
                inner_type = canonical_type_str[len("typing.Optional["):-1].strip()
                canonical_type_str = f"typing.Union[{inner_type}, None]"
            elif canonical_type_str == "NoneType": # Pyanalyze might output this
                canonical_type_str = "None"


            # Step 2: Normalize Union members (sort, unique, consistent None)
            # This regex is basic and will NOT correctly parse complex nested generics like Union[List[A], Dict[B, C]].
            # It's intended for simpler cases like Union[str, int], Union[str, int, None].
            # A full parser for type strings is out of scope here.
            match_union = re.match(r"^(typing\.)?Union\[(.*)\]$", canonical_type_str)
            if match_union:
                members_str = match_union.group(2)
                # Basic split by comma. This is fragile for nested types.
                # E.g., "Union[list[int], dict[str, str]]" would split incorrectly.
                # This assumes members are simple or already well-formed by Pyanalyze.
                raw_members = [m.strip() for m in members_str.split(',')]

                has_none = False
                processed_members = set() # Use set for uniqueness

                for member in raw_members:
                    if member == "None" or member == "NoneType":
                        has_none = True
                    else:
                        # Further simplify basic generics here if needed, e.g. list, dict
                        member = member.replace("typing.List[", "list[")
                        member = member.replace("typing.Dict[", "dict[")
                        member = member.replace("typing.Set[", "set[")
                        member = member.replace("typing.Tuple[", "tuple[")
                        member = member.replace("typing.Any", "Any")
                        processed_members.add(member)

                sorted_members = sorted(list(filter(None, processed_members))) # Filter out empty strings if any

                if has_none:
                    if not sorted_members: # Only None was in the Union
                        canonical_type_str = "None"
                    else:
                        # Consistent order: sorted types, then None
                        canonical_type_str = f"typing.Union[{', '.join(sorted_members)}, None]"
                elif len(sorted_members) == 1:
                    canonical_type_str = sorted_members[0] # Simplify Union[X] to X
                elif len(sorted_members) > 1:
                    canonical_type_str = f"typing.Union[{', '.join(sorted_members)}]"
                elif not sorted_members and not has_none: # Empty Union e.g. Union[] - should not happen
                    pass # Keep original or mark as error/Any

            # Step 3: Final pass for simple typing.X to X replacements if not part of a Union already handled
            # This needs to be careful not to break already canonicalized Union strings.
            # The Union processing above already handles this for members.
            # This is for non-Union types primarily.
            if not canonical_type_str.startswith("typing.Union["):
                canonical_type_str = canonical_type_str.replace("typing.List[", "list[")
                canonical_type_str = canonical_type_str.replace("typing.Dict[", "dict[")
                canonical_type_str = canonical_type_str.replace("typing.Set[", "set[")
                canonical_type_str = canonical_type_str.replace("typing.Tuple[", "tuple[")
                canonical_type_str = canonical_type_str.replace("typing.Any", "Any")
                if canonical_type_str == "NoneType": # Final check
                    canonical_type_str = "None"


            if verbose_resolver and canonical_type_str != original_for_log:
                 print(f"Resolver: Canonicalized type '{original_for_log}' to '{canonical_type_str}' for key '{key_to_lookup}'")
            return canonical_type_str

        return resolved_type # Returns None if not found, or original if no canonicalization applied


    def _get_node_source(self, node: ast.AST) -> str: # As before
        if hasattr(ast, 'unparse'):
            try:
                return ast.unparse(node)
            except Exception:
                pass

        if hasattr(node, 'lineno') and hasattr(node, 'end_lineno') and node.end_lineno is not None: # Ensure end_lineno is not None
            start_line = node.lineno -1
            end_line = node.end_lineno # ast.get_source_segment uses end_lineno as exclusive for lines list

            if 0 <= start_line < len(self.source_code_lines) and 0 <= end_line <= len(self.source_code_lines) and start_line < end_line:
                 return "\n".join(self.source_code_lines[start_line:end_line])
        return ""

    def _process_func_or_method(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef], item_type_prefix: str):
        base_fqn_parts = [self.module_qname]
        if self.current_class_name: base_fqn_parts.append(self.current_class_name)
        base_fqn_parts.append(node.name)
        fqn = ".".join(filter(None, base_fqn_parts))

        code_content = self._get_node_source(node)
        signature_string = generate_function_signature_string(
            node,
            self._local_type_resolver,
            class_fqn_prefix_for_method_name=self.current_class_name.split('.')[-1] if self.current_class_name and item_type_prefix == "method" else None
        )
        # Add signature_str to the main symbol dictionary
        self.symbols_for_embedding.append({
            "fqn": fqn, "item_type": f"{item_type_prefix}_code", "content": code_content,
            "file_path": self.file_path_str, "start_line": node.lineno,
            "end_line": node.end_lineno if hasattr(node, 'end_lineno') and node.end_lineno is not None else node.lineno,
            "signature_str": signature_string # Store the signature string
        })
        docstring = ast.get_docstring(node, clean=False)
        if docstring:
            doc_node_start = node.lineno
            doc_node_end = node.lineno
            if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, (ast.Constant if hasattr(ast, 'Constant') else ast.Str)):
                doc_expr_node = node.body[0]; doc_node_start = doc_expr_node.lineno
                if hasattr(doc_expr_node, 'end_lineno') and doc_expr_node.end_lineno is not None: doc_node_end = doc_expr_node.end_lineno
                else: doc_node_end = doc_expr_node.lineno + len(docstring.splitlines()) -1
            else: # Fallback if docstring is present but not the first expr
                doc_node_end = doc_node_start + len(docstring.splitlines()) - 1
            self.symbols_for_embedding.append({
                "fqn": fqn, "item_type": f"docstring_for_{item_type_prefix}", "content": docstring,
                "file_path": self.file_path_str, "start_line": doc_node_start, "end_line": doc_node_end
                # Not adding signature_str to docstring symbols, only to the code symbol.
            })

        # signature_string was already calculated above and used.
        self.signature_trie_ref.insert(signature_string, fqn)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self._process_func_or_method(node, "method" if self.current_class_name else "function")
        super().generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self._process_func_or_method(node, "method" if self.current_class_name else "function")
        super().generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef):
        outer_context_fqn_parts = [self.module_qname]
        if self.current_class_name:
            outer_context_fqn_parts.append(self.current_class_name)
        outer_context_fqn_parts.append(node.name)
        class_fqn = ".".join(filter(None, outer_context_fqn_parts))

        class_code_content = self._get_node_source(node)
        self.symbols_for_embedding.append({
            "fqn": class_fqn, "item_type": "class_code", "content": class_code_content,
            "file_path": self.file_path_str, "start_line": node.lineno,
            "end_line": node.end_lineno if hasattr(node, 'end_lineno') and node.end_lineno is not None else node.lineno
        })
        class_docstring = ast.get_docstring(node, clean=False)
        if class_docstring:
            doc_node_start = node.lineno; doc_node_end = node.lineno
            if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, (ast.Constant if hasattr(ast, 'Constant') else ast.Str)):
                doc_expr_node_cls = node.body[0]; doc_node_start = doc_expr_node_cls.lineno
                if hasattr(doc_expr_node_cls, 'end_lineno') and doc_expr_node_cls.end_lineno is not None: doc_node_end = doc_expr_node_cls.end_lineno
                else: doc_node_end = doc_expr_node_cls.lineno + len(class_docstring.splitlines()) -1
            else: # Fallback if docstring is present but not the first expr
                doc_node_end = doc_node_start + len(class_docstring.splitlines()) - 1
            self.symbols_for_embedding.append({
                "fqn": class_fqn, "item_type": "docstring_for_class", "content": class_docstring,
                "file_path": self.file_path_str, "start_line": doc_node_start, "end_line": doc_node_end
            })

        original_outer_class_name_context = self.current_class_name
        self.current_class_name = class_fqn

        super().generic_visit(node)

        self.current_class_name = original_outer_class_name_context

        # Populate inheritance graph
        derived_class_node_id = NodeID(class_fqn)
        # Initialize even if no bases, to show the class exists in the graph context
        self.digester_ref.project_inheritance_graph.setdefault(derived_class_node_id, [])

        for base_node in node.bases:
            base_class_fqn_str = "UNRESOLVED_BASE_CLASS" # Default
            if hasattr(ast, 'unparse'):
                try:
                    base_class_fqn_str = ast.unparse(base_node)
                except Exception: # Fallback if unparse fails for some complex node
                    if isinstance(base_node, ast.Name):
                        base_class_fqn_str = base_node.id
                    elif isinstance(base_node, ast.Attribute):
                        # Basic reconstruction for simple attributes like module.Class
                        # This won't handle complex value expressions for base_node.value
                        try:
                            value_str = ast.unparse(base_node.value)
                            base_class_fqn_str = f"{value_str}.{base_node.attr}"
                        except Exception:
                             base_class_fqn_str = f"UNRESOLVED_ATTRIBUTE_BASE:{type(base_node.value).__name__}.{base_node.attr}"
                    else:
                        base_class_fqn_str = f"UNRESOLVED_BASE_NODE_TYPE:{type(base_node).__name__}"
            elif isinstance(base_node, ast.Name): # Fallback for Python < 3.9
                base_class_fqn_str = base_node.id
            elif isinstance(base_node, ast.Attribute) and isinstance(base_node.value, ast.Name): # e.g. some_module.MyClass
                 base_class_fqn_str = f"{base_node.value.id}.{base_node.attr}"
            # Note: More complex base expressions (e.g., Subscript, Call) are harder to resolve to a simple FQN string here.
            # For now, they might remain "UNRESOLVED_BASE_CLASS" or similar.

            base_class_node_id = NodeID(base_class_fqn_str)
            self.digester_ref.project_inheritance_graph[derived_class_node_id].append(base_class_node_id)

    def visit_Import(self, node: ast.Import):
        current_module_node_id = NodeID(self.module_qname)
        self.digester_ref.project_module_imports.setdefault(current_module_node_id, set())
        for alias_node in node.names:
            imported_module_name = alias_node.name # This is the name as it appears
            imported_module_node_id = NodeID(imported_module_name)
            self.digester_ref.project_module_imports[current_module_node_id].add(imported_module_node_id)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        current_module_node_id = NodeID(self.module_qname)
        source_module_name_raw = node.module # This can be None for relative imports like 'from . import x'

        resolved_source_module_fqn_str: Optional[str] = None

        if node.level > 0: # Relative import
            current_module_parts = self.module_qname.split('.')
            if node.level > len(current_module_parts) and not (node.level == len(current_module_parts) and not source_module_name_raw): # e.g. from ...foo import bar in a top-level module, or from .. import foo in my_module
                 # Allow from .. import foo in my_module.sub_module (len=2, level=2 -> effective_base_parts is empty, result is 'foo')
                 # Disallow from .. import foo in my_module (len=1, level=2 -> error)
                if self.digester_ref.verbose:
                    print(f"RepositoryDigester Warning: Invalid relative import level {node.level} in module '{self.module_qname}' for import from '.{'.'*(node.level-1)}{source_module_name_raw or ''}'. Skipping.")
                self.generic_visit(node)
                return

            effective_base_parts = current_module_parts[:-node.level]

            if source_module_name_raw:
                resolved_source_module_fqn_str = ".".join(effective_base_parts + [source_module_name_raw])
            else: # e.g. from . import foo
                resolved_source_module_fqn_str = ".".join(effective_base_parts) if effective_base_parts else ""
                # If effective_base_parts is empty (e.g. `from . import foo` in `__init__.py` of a package `pkg`),
                # this implies the symbols are directly in `pkg`. But `resolved_source_module_fqn_str` becomes empty.
                # This needs careful handling. For `from . import X`, source is current package.
                # If `self.module_qname` is `pkg.__init__`, `level=1` -> `effective_base_parts` is `['pkg']`
                # If `module_qname` is `pkg.sub`, `level=1` -> `effective_base_parts` is `['pkg']`
                # This seems to give the package FQN.

        elif source_module_name_raw is not None: # Absolute import
            resolved_source_module_fqn_str = source_module_name_raw
        else: # Absolute import but node.module is None. Should not happen for valid Python.
            if self.digester_ref.verbose:
                print(f"RepositoryDigester Warning: Absolute import with no module name in '{self.module_qname}'. Node: {ast.dump(node)}. Skipping.")
            self.generic_visit(node)
            return

        if resolved_source_module_fqn_str is None or not resolved_source_module_fqn_str.strip():
            # This can happen if relative import logic results in an empty string, e.g. from .. in a top level module.
            # Also, if `from . import foo` in `__init__.py` of `pkg`, `resolved_source_module_fqn_str` becomes `pkg`.
            # If it's truly empty or None, it's problematic.
            if self.digester_ref.verbose:
                print(f"RepositoryDigester Warning: Could not resolve source module FQN for ImportFrom in '{self.module_qname}'. Level: {node.level}, Module: {node.module}. Skipping. Node: {ast.dump(node)}")
            self.generic_visit(node)
            return

        source_module_node_id = NodeID(resolved_source_module_fqn_str)
        # Ensure the entry for current_module -> source_module exists before adding symbols
        self.digester_ref.project_symbol_imports.setdefault(current_module_node_id, defaultdict(set))
        self.digester_ref.project_symbol_imports[current_module_node_id].setdefault(source_module_node_id, set())

        for alias_node in node.names:
            imported_symbol_name = alias_node.name # Name of the symbol being imported
            # local_name = alias_node.asname or imported_symbol_name # How it's named locally
            self.digester_ref.project_symbol_imports[current_module_node_id][source_module_node_id].add(imported_symbol_name)

        self.generic_visit(node)


class DataDependenceVisitor(ast.NodeVisitor):
    def __init__(self, file_id_prefix: str):
        self.file_id_prefix: str = file_id_prefix
        self.data_dependencies: DataDependenceGraph = defaultdict(set) # Use defaultdict

        # Scope stack: list of dictionaries. Each dict maps var_name (str) to a list of def_node_ids.
        # The list of NodeIDs for a var_name stores its current reaching definition(s) in that scope.
        # For simplicity, we'll store only the most recent definition's NodeID.
        self.scope_stack: List[Dict[str, NodeID]] = [{}] # Start with module-level scope

    def _get_current_scope_defs(self) -> Dict[str, NodeID]:
        return self.scope_stack[-1]

    def _add_definition(self, var_name: str, def_node: ast.AST):
        current_scope_defs = self._get_current_scope_defs()
        def_node_id = _create_ast_node_id(self.file_id_prefix, def_node, f"{var_name}:def")
        current_scope_defs[var_name] = def_node_id
        # print(f"DEF: {var_name} at {def_node_id} in scope {len(self.scope_stack)-1}")


    def _add_use_dependencies(self, use_node: ast.AST, var_name: str):
        use_node_id = _create_ast_node_id(self.file_id_prefix, use_node, f"{var_name}:use")
        # Search for definition from innermost scope outwards
        for i in range(len(self.scope_stack) - 1, -1, -1):
            scope_defs = self.scope_stack[i]
            if var_name in scope_defs:
                def_node_id = scope_defs[var_name]
                self.data_dependencies[use_node_id].add(def_node_id)
                # print(f"USE: {var_name} at {use_node_id} depends on def at {def_node_id} from scope {i}")
                return # Found in this scope (or outer), stop.
        # print(f"Warning: No definition found for use of '{var_name}' at {use_node_id} (might be global/builtin or undefined)")


    def visit_FunctionDef(self, node: ast.FunctionDef):
        # Function name itself is a definition in the outer scope
        self._add_definition(node.name, node) # Consider the FunctionDef node as the def site for its name

        # Decorators are expressions evaluated in outer scope
        for decorator in node.decorator_list: self.visit(decorator)
        # Default arguments (expressions evaluated in outer scope at def time)
        for default_expr in node.args.defaults: self.visit(default_expr)
        for default_expr_kwonly in node.args.kw_defaults:
            if default_expr_kwonly: self.visit(default_expr_kwonly)
        # Visit annotations in outer scope as well (parameter annotations, return annotation)
        if node.returns: self.visit(node.returns)
        for arg_node in node.args.args: # ast.arg
            if arg_node.annotation: self.visit(arg_node.annotation)
        for kwarg_node_only in node.args.kwonlyargs:
            if kwarg_node_only.annotation: self.visit(kwarg_node_only.annotation)
        if node.args.vararg and node.args.vararg.annotation: self.visit(node.args.vararg.annotation)
        if node.args.kwarg and node.args.kwarg.annotation: self.visit(node.args.kwarg.annotation)


        # New scope for function body and parameters
        self.scope_stack.append({})
        for arg_node in node.args.args: # ast.arg
            self._add_definition(arg_node.arg, arg_node) # Parameters are definitions in func scope
        if node.args.vararg: self._add_definition(node.args.vararg.arg, node.args.vararg)
        if node.args.kwarg: self._add_definition(node.args.kwarg.arg, node.args.kwarg)
        for kwarg_node_only in node.args.kwonlyargs:
            self._add_definition(kwarg_node_only.arg, kwarg_node_only)

        # Visit body
        for stmt in node.body: self.visit(stmt)

        self.scope_stack.pop()
        # Do not call generic_visit(node) as we've manually handled relevant parts.

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self.visit_FunctionDef(node) # Treat same as FunctionDef

    def visit_Lambda(self, node: ast.Lambda):
        # Default arguments are evaluated in outer scope
        for default_expr in node.args.defaults: self.visit(default_expr)
        for default_expr_kwonly in node.args.kw_defaults:
            if default_expr_kwonly: self.visit(default_expr_kwonly)

        # New scope for lambda parameters and body
        self.scope_stack.append({})
        for arg_node in node.args.args: self._add_definition(arg_node.arg, arg_node)
        if node.args.vararg: self._add_definition(node.args.vararg.arg, node.args.vararg)
        if node.args.kwarg: self._add_definition(node.args.kwarg.arg, node.args.kwarg)
        for kwarg_node_only in node.args.kwonlyargs:
            self._add_definition(kwarg_node_only.arg, kwarg_node_only)

        self.visit(node.body) # Visit lambda body expression
        self.scope_stack.pop()

    def visit_ClassDef(self, node: ast.ClassDef):
        # Class name is a definition in the outer scope
        self._add_definition(node.name, node)

        # Decorators are expressions evaluated in outer scope
        for decorator in node.decorator_list: self.visit(decorator)
        # Base classes are expressions evaluated in outer scope
        for base in node.bases: self.visit(base)
        for keyword in node.keywords: self.visit(keyword.value)

        # New scope for class body (for class variables, methods)
        self.scope_stack.append({})
        for stmt in node.body: self.visit(stmt)
        self.scope_stack.pop()

    def visit_Name(self, node: ast.Name):
        if isinstance(node.ctx, ast.Load):
            self._add_use_dependencies(node, node.id)
        # ast.Store context for simple names is handled by visit_Assign/AnnAssign targets
        # ast.Del context could be a "kill" of definitions, not handled yet.
        self.generic_visit(node) # Should not be needed if all contexts are handled

    def visit_Assign(self, node: ast.Assign):
        # RHS (value) is visited first (contains uses)
        self.visit(node.value)
        # Then LHS (targets) are definitions
        for target in node.targets:
            if isinstance(target, ast.Name):
                self._add_definition(target.id, target)
            elif isinstance(target, ast.Attribute):
                self.visit(target.value) # Visit the object part (e.g., 'self') for uses
            elif isinstance(target, (ast.Tuple, ast.List)): # Unpacking assignment
                for elt in target.elts:
                    if isinstance(elt, ast.Name): self._add_definition(elt.id, elt)
                    else: self.visit(elt)
            else:
                self.visit(target)


    def visit_AnnAssign(self, node: ast.AnnAssign):
        # Annotation is visited first (may contain uses)
        if node.annotation: self.visit(node.annotation)
        # Value (RHS) is visited next (may contain uses)
        if node.value: self.visit(node.value)
        # Target (LHS) is a definition
        if isinstance(node.target, ast.Name):
            self._add_definition(node.target.id, node.target)
        elif isinstance(node.target, ast.Attribute):
            self.visit(node.target.value) # Visit object part for uses
        else:
            self.visit(node.target)


    def visit_For(self, node: ast.For):
        # Iterable is evaluated first (uses)
        self.visit(node.iter)

        # Loop variables are definitions.
        # Create a new scope for the loop body if targets are complex, or handle in current.
        # For simplicity, new definitions shadow previous ones in the current scope.
        if isinstance(node.target, ast.Name):
            self._add_definition(node.target.id, node.target)
        elif isinstance(node.target, (ast.Tuple, ast.List)):
            for elt in node.target.elts:
                if isinstance(elt, ast.Name): self._add_definition(elt.id, elt)
                else: self.visit(elt)
        else:
            self.visit(node.target)

        for stmt in node.body: self.visit(stmt)
        if node.orelse:
            for stmt_else in node.orelse: self.visit(stmt_else)

    def visit_AsyncFor(self, node: ast.AsyncFor):
        self.visit_For(node)

    def _visit_comprehension_generator(self, generator: ast.comprehension):
        # Iter is evaluated in the scope where the comprehension appears.
        self.visit(generator.iter)

        # Target is defined. In Python 3, comps have their own scope for targets.
        # This simplified visitor adds defs to the current function/lambda scope.
        if isinstance(generator.target, ast.Name):
            self._add_definition(generator.target.id, generator.target)
        elif isinstance(generator.target, (ast.Tuple, ast.List)):
             for elt in generator.target.elts:
                if isinstance(elt, ast.Name): self._add_definition(elt.id, elt)
                else: self.visit(elt)
        else:
            self.visit(generator.target)

        for if_expr in generator.ifs: self.visit(if_expr) # Conditions are uses

    def visit_ListComp(self, node: ast.ListComp):
        # Comps run in a new scope in Python 3. This visitor simplifies.
        # Generators are processed first (iter uses, target defs)
        for generator in node.generators: self._visit_comprehension_generator(generator)
        # Element expression uses vars defined in generators
        self.visit(node.elt)

    def visit_SetComp(self, node: ast.SetComp):
        for generator in node.generators: self._visit_comprehension_generator(generator)
        self.visit(node.elt)

    def visit_DictComp(self, node: ast.DictComp):
        for generator in node.generators: self._visit_comprehension_generator(generator)
        self.visit(node.key); self.visit(node.value)

    def visit_GeneratorExp(self, node: ast.GeneratorExp):
        for generator in node.generators: self._visit_comprehension_generator(generator)
        self.visit(node.elt)

    def visit_If(self, node: ast.If):
        self.visit(node.test) # Uses in condition
        for stmt in node.body: self.visit(stmt)
        if node.orelse:
            for stmt_else in node.orelse: self.visit(stmt_else)

    def visit_While(self, node: ast.While):
        self.visit(node.test) # Uses in condition
        for stmt in node.body: self.visit(stmt)
        if node.orelse:
            for stmt_else in node.orelse: self.visit(stmt_else)

    def visit_With(self, node: ast.With):
        for item in node.items:
            self.visit(item.context_expr) # Uses in context expression
            if item.optional_vars: # Definitions from 'as var'
                if isinstance(item.optional_vars, ast.Name):
                    self._add_definition(item.optional_vars.id, item.optional_vars)
                elif isinstance(item.optional_vars, (ast.Tuple, ast.List)):
                    for elt in item.optional_vars.elts:
                         if isinstance(elt, ast.Name): self._add_definition(elt.id, elt)
                         else: self.visit(elt)
                else:
                    self.visit(item.optional_vars)
        for stmt in node.body: self.visit(stmt)

    def visit_AsyncWith(self, node: ast.AsyncWith):
        self.visit_With(node)

    def visit_Try(self, node: ast.Try):
        for stmt in node.body: self.visit(stmt)
        for handler in node.handlers:
            if handler.type: self.visit(handler.type)
            if handler.name: # 'as e' variable is a definition
                self._add_definition(handler.name, handler) # handler node itself as def site for 'e'
            for stmt_handler in handler.body: self.visit(stmt_handler)
        if node.orelse:
            for stmt_else in node.orelse: self.visit(stmt_else)
        if node.finalbody:
            for stmt_finally in node.finalbody: self.visit(stmt_finally)

# Helper to create a unique ID for an AST node based on file and location
# This should be at module level or a static method if it doesn't need 'self'.
# Making it module level for now.
def _create_ast_node_id(file_path_str: str, node: ast.AST, category_suffix: str = "") -> NodeID:
    """Creates a unique string ID for an AST node."""
    node_name_part = ""
    # Attempt to get a 'name' attribute if it exists
    if hasattr(node, 'name') and isinstance(node.name, str):
        node_name_part = node.name
    elif isinstance(node, ast.Name): # For ast.Name nodes, the name is in 'id'
        node_name_part = node.id
    elif isinstance(node, ast.Attribute): # For ast.Attribute, use 'attr'
        node_name_part = node.attr
    elif isinstance(node, ast.arg): # NEW: Handle function parameters
        node_name_part = node.arg
    # Add more specific name extractions if needed for other node types.

    base_id = f"{file_path_str}:{node.lineno}:{node.col_offset}:{type(node).__name__}"
    if node_name_part:
        base_id += f":{node_name_part}"
    if category_suffix:
        base_id += f":{category_suffix}"
    return NodeID(base_id)


class ControlDependenceVisitor(ast.NodeVisitor):
    def __init__(self, file_path_str: str):
        self.file_path_str: str = file_path_str
        self.control_dependencies: ControlDependenceGraph = {}
        self.current_control_stack: List[NodeID] = []

    def _add_dependence(self, dependent_node: ast.AST, dependent_category_suffix: str = "statement"):
        if self.current_control_stack: # If there is an active controller
            controller_node_id = self.current_control_stack[-1]
            dependent_node_id = _create_ast_node_id(self.file_path_str, dependent_node, dependent_category_suffix)
            self.control_dependencies.setdefault(controller_node_id, set()).add(dependent_node_id)

    def _process_body_and_orelse(self, node_with_body: Union[ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try]):
        """Helper to process body and orelse blocks for a given controlling node."""
        for child_node in node_with_body.body:
            self._add_dependence(child_node)
            self.visit(child_node)

        if hasattr(node_with_body, 'orelse') and node_with_body.orelse:
            # For Try nodes, orelse is handled specially, so skip here.
            # For other nodes (If, For, While), orelse depends on the same primary condition/controller.
            if not isinstance(node_with_body, ast.Try):
                 for child_node in node_with_body.orelse:
                    self._add_dependence(child_node)
                    self.visit(child_node)

    def visit_If(self, node: ast.If):
        # The 'test' (condition) is the controller
        controller_id = _create_ast_node_id(self.file_path_str, node.test, "if_condition")
        self.current_control_stack.append(controller_id)
        self._process_body_and_orelse(node) # Processes node.body and node.orelse
        self.current_control_stack.pop()
        # Do not call self.generic_visit(node) as children are handled by _process_body_and_orelse

    def visit_For(self, node: ast.For):
        # The iterable (node.iter) and target (node.target) define the control
        # For simplicity, let's make the For node itself (or its iter) the controller.
        controller_id = _create_ast_node_id(self.file_path_str, node.iter, "for_iterable") # or node itself
        self.current_control_stack.append(controller_id)
        self._process_body_and_orelse(node)
        self.current_control_stack.pop()

    def visit_AsyncFor(self, node: ast.AsyncFor):
        controller_id = _create_ast_node_id(self.file_path_str, node.iter, "asyncfor_iterable") # or node itself
        self.current_control_stack.append(controller_id)
        self._process_body_and_orelse(node)
        self.current_control_stack.pop()

    def visit_While(self, node: ast.While):
        controller_id = _create_ast_node_id(self.file_path_str, node.test, "while_condition")
        self.current_control_stack.append(controller_id)
        self._process_body_and_orelse(node)
        self.current_control_stack.pop()

    def visit_Try(self, node: ast.Try):
        try_construct_id = _create_ast_node_id(self.file_path_str, node, "try_construct")

        # Body of try (depends on the try_construct_id, meaning "entry into try")
        self.current_control_stack.append(try_construct_id)
        for child_node in node.body:
            self._add_dependence(child_node)
            self.visit(child_node)
        self.current_control_stack.pop()

        # Except handlers
        for handler in node.handlers: # ast.ExceptHandler
            # The handler's type (exception type) or the handler itself if type is None, is the condition
            condition_node_for_handler = handler.type if handler.type else handler
            handler_controller_id = _create_ast_node_id(self.file_path_str, condition_node_for_handler, "except_handler_condition")

            self.current_control_stack.append(handler_controller_id)
            for child_node in handler.body:
                self._add_dependence(child_node)
                self.visit(child_node)
            self.current_control_stack.pop()

        # Orelse block (depends on no exception from try body, so controlled by try_construct_id "success" path)
        if node.orelse:
            self.current_control_stack.append(try_construct_id)
            for child_node in node.orelse:
                self._add_dependence(child_node, "orelse_statement_after_try")
                self.visit(child_node)
            self.current_control_stack.pop()

        # Finally block (always executes, but its execution is tied to the try construct scope)
        # It's not strictly "controlled" by a condition in the same way as an If body,
        # but rather its execution is guaranteed upon exiting the try-except-orelse structure.
        # For CDG, we can consider it dependent on the try_construct_id.
        if node.finalbody:
            self.current_control_stack.append(try_construct_id)
            for child_node in node.finalbody:
                # Using a distinct suffix to clarify its nature if needed
                self._add_dependence(child_node, "finally_statement")
                self.visit(child_node)
            self.current_control_stack.pop()

    def visit_With(self, node: ast.With):
        # Each 'withitem's context_expr can be seen as controlling the body.
        # For simplicity, make the With node itself the controller for the body.
        # The items are part of the setup.
        with_controller_id = _create_ast_node_id(self.file_path_str, node, "with_block")

        # Visit context expressions and optional_vars as they are evaluated before body
        for item in node.items:
            self.visit(item.context_expr)
            if item.optional_vars:
                self.visit(item.optional_vars) # These are assignments, not typically controllers for the body

        # Body depends on the successful setup of all with items.
        self.current_control_stack.append(with_controller_id)
        for child_node in node.body:
            self._add_dependence(child_node)
            self.visit(child_node)
        self.current_control_stack.pop()

    def visit_AsyncWith(self, node: ast.AsyncWith):
        async_with_controller_id = _create_ast_node_id(self.file_path_str, node, "async_with_block")
        for item in node.items:
            self.visit(item.context_expr)
            if item.optional_vars:
                self.visit(item.optional_vars)

        self.current_control_stack.append(async_with_controller_id)
        for child_node in node.body:
            self._add_dependence(child_node)
            self.visit(child_node)
        self.current_control_stack.pop()

class PyanalyzeTypeExtractionVisitor(ast.NodeVisitor):
    def __init__(self, filename: str, module_qname: str):
        self.filename = filename
        self.module_qname = module_qname
        self.type_info_map: Dict[str, str] = {}
        self.dump_value_func = pyanalyze_dump_value
        self.current_class_qname_stack: List[str] = []

    def _get_current_scope_prefix(self) -> str:
        if self.current_class_qname_stack:
            return self.current_class_qname_stack[-1]
        return self.module_qname

    def _add_type_info(self, node: ast.AST, name_qualifier: str, category: str, inferred_value_node: Optional[ast.AST] = None):
        target_node_for_value = inferred_value_node if inferred_value_node else node
        if hasattr(target_node_for_value, 'inferred_value') and self.dump_value_func:
            inferred_val = getattr(target_node_for_value, 'inferred_value')
            if inferred_val is not None:
                type_str = self.dump_value_func(inferred_val)

                # Construct FQN for the typed element
                # name_qualifier is the base name (var, param, func for return, attr)
                # It should NOT contain the scope prefix already.
                scope_prefix = self._get_current_scope_prefix()

                # For parameters and returns, the 'name_qualifier' might be like 'param_name' or '<return>'
                # and needs to be associated with the function it belongs to.
                # The 'category' helps disambiguate.

                # If name_qualifier is already an FQN (e.g. from recursive calls or complex types), don't prepend.
                # This heuristic might need refinement.
                fqn_of_element = name_qualifier
                if not any(name_qualifier.startswith(prefix + ".") for prefix in [self.module_qname] + self.current_class_qname_stack):
                     # If it's a simple name (e.g. 'x', 'my_param', '<return>', 'my_attr')
                     # then prepend current scope.
                     if category in ["parameter", "function_return", "parameter_vararg", "parameter_kwonly", "parameter_kwarg"]:
                         # name_qualifier here is like "func_name.param_name" or "func_name.<return>"
                         # where func_name is node.name from visit_FunctionDef.
                         # Scope prefix is module or class. So, fqn is scope_prefix + "." + name_qualifier (if not already prefixed)
                         if not name_qualifier.startswith(scope_prefix + "."):
                              fqn_of_element = f"{scope_prefix}.{name_qualifier}"
                         # else name_qualifier was already correctly prefixed by caller.
                     elif category in ["instance_attribute_definition", "class_attribute_definition", "class_variable_definition"]:
                         # name_qualifier here is like "ClassName.attr_name" or just "attr_name" if class is implicit.
                         # scope_prefix is ClassName.
                         if not name_qualifier.startswith(scope_prefix + "."): # if name_qualifier is just 'attr_name'
                            fqn_of_element = f"{scope_prefix}.{name_qualifier}"
                         # else name_qualifier was already correctly prefixed by caller.
                     else: # "variable_definition", "variable_use", etc.
                        fqn_of_element = f"{scope_prefix}.{name_qualifier}"


                node_key = f"{fqn_of_element}:{category}:{node.lineno}:{node.col_offset}"
                self.type_info_map[node_key] = type_str
                # print(f"PyanalyzeVisitor DEBUG: Key='{node_key}', Type='{type_str}'")


    def visit_Name(self, node: ast.Name):
        # Captures types for local variables (stores) and potentially uses (loads) if Pyanalyze annotates them.
        if isinstance(node.ctx, ast.Store):
            self._add_type_info(node, node.id, "variable_definition")
        elif isinstance(node.ctx, ast.Load):
            # Pyanalyze might put inferred_value on Load nodes too.
            # This could be useful but also very verbose.
            # self._add_type_info(node, node.id, "variable_use") # Example if we want to capture uses
            pass
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        scope_prefix = self._get_current_scope_prefix()
        func_name_simple = node.name
        func_qualifier_for_elements = f"{func_name_simple}" # Params/returns are relative to function name

        for arg_node in node.args.args: # Includes posonlyargs if logic combined
            self._add_type_info(arg_node, f"{func_qualifier_for_elements}.{arg_node.arg}", "parameter", inferred_value_node=arg_node)
        if node.args.vararg:
            self._add_type_info(node.args.vararg, f"{func_qualifier_for_elements}.{node.args.vararg.arg}", "parameter_vararg", inferred_value_node=node.args.vararg)
        for arg_node in node.args.kwonlyargs:
            self._add_type_info(arg_node, f"{func_qualifier_for_elements}.{arg_node.arg}", "parameter_kwonly", inferred_value_node=arg_node)
        if node.args.kwarg:
            self._add_type_info(node.args.kwarg, f"{func_qualifier_for_elements}.{node.args.kwarg.arg}", "parameter_kwarg", inferred_value_node=node.args.kwarg)

        if node.returns: # node.returns is the annotation AST node
            self._add_type_info(node.returns, f"{func_qualifier_for_elements}.<return>", "function_return", inferred_value_node=node.returns)

        # Type of the function/method itself (callable type)
        # The name_qualifier should be just the function name, _add_type_info will prefix scope.
        self._add_type_info(node, func_name_simple, "function_definition_callable_type")

        # Handle nested functions/classes
        # For nested functions, their scope_prefix will be formed using the outer func's FQN.
        # For nested classes, visit_ClassDef will handle pushing onto current_class_qname_stack.
        # No explicit passing of func_fqn needed if _get_current_scope_prefix correctly reflects nesting.
        # Pyanalyze itself should create FQNs for nested items. We just need to record them.

        # Store current func FQN if we need to build FQNs for items defined *inside* it,
        # if Pyanalyze doesn't provide them directly on those items.
        # However, the current keying relies on Pyanalyze providing FQN-like structures or types
        # that can be resolved to FQNs by SymbolAndSignatureExtractorVisitor.

        # The current logic of _add_type_info prepends scope. If node.name for a nested function
        # is just 'inner_func', it will become 'outer_scope.inner_func'.
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self.visit_FunctionDef(node)

    def visit_ClassDef(self, node: ast.ClassDef):
        scope_prefix = self._get_current_scope_prefix()
        class_fqn = f"{scope_prefix}.{node.name}"

        self.current_class_qname_stack.append(class_fqn)
        self._add_type_info(node, node.name, "class_definition_type") # Type of the class itself

        self.generic_visit(node) # Visit methods and attributes
        self.current_class_qname_stack.pop()

    def _handle_assign_target(self, target_node: ast.expr, value_node: Optional[ast.AST]):
        """Helper to process assignment targets for attributes."""
        # This helper is for AnnAssign and Assign within class context.
        # value_node is the RHS of assignment, Pyanalyze might attach type info there or on target.
        # inferred_value_node should be the node Pyanalyze actually put .inferred_value on.

        scope_prefix = self.current_class_qname_stack[-1] if self.current_class_qname_stack else self.module_qname

        if isinstance(target_node, ast.Attribute):
            # e.g., self.attr = value OR cls.attr = value
            if isinstance(target_node.value, ast.Name) and target_node.value.id in ('self', 'cls'):
                category = "instance_attribute_definition" if target_node.value.id == 'self' else "class_attribute_definition"
                # name_qualifier is just the attribute name, _add_type_info will prefix with class FQN
                self._add_type_info(target_node, target_node.attr, category, inferred_value_node=target_node)
            # Could also handle ClassName.attr = value if target_node.value resolves to ClassName
            # This would require more complex name resolution here or rely on Pyanalyze annotating target_node directly.
        elif isinstance(target_node, ast.Name):
            # This is an assignment to a name within a class body, e.g. MyVar: int = 1
            # This is a class variable.
            # name_qualifier is the variable name, _add_type_info prefixes with class FQN
            self._add_type_info(target_node, target_node.id, "class_variable_definition", inferred_value_node=target_node)


    def visit_AnnAssign(self, node: ast.AnnAssign):
        if self.current_class_qname_stack: # Inside a class definition
            self._handle_assign_target(node.target, node.value)
        else: # Module or function scope variable
            if isinstance(node.target, ast.Name):
                 self._add_type_info(node.target, node.target.id, "variable_definition", inferred_value_node=node.target)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign):
        # Assign can have multiple targets (e.g., a = b = 1)
        # Pyanalyze typically annotates the rightmost target or the value.
        if self.current_class_qname_stack: # Inside a class definition
            for target in node.targets:
                self._handle_assign_target(target, node.value)
        else: # Module or function scope
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self._add_type_info(target, target.id, "variable_definition", inferred_value_node=target)
        self.generic_visit(node)


class CallGraphVisitor(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (ParentNodeProvider, PositionProvider, QualifiedNameProvider, ScopeProvider)

    def __init__(self, module_qname: str, file_path: Path, type_info: Optional[Dict[str, Any]], project_call_graph_ref: CallGraph):
        super().__init__()
        self.module_qname: str = module_qname
        self.file_path: Path = file_path
        self.type_info: Dict[str, Any] = type_info if type_info else {} # Ensure it's a dict
        self.project_call_graph: CallGraph = project_call_graph_ref
        self.current_fqn_stack: List[str] = [module_qname]

    def _get_current_caller_fqn(self) -> Optional[NodeID]:
        if not self.current_fqn_stack or len(self.current_fqn_stack) == 1 and self.current_fqn_stack[0] == self.module_qname:
            return None
        return NodeID(".".join(self.current_fqn_stack))

    def _resolve_callee_fqn(self, call_node: cst.Call) -> Optional[NodeID]:
        func_expr = call_node.func
        resolved_fqn_str: Optional[str] = None

        qname_metadata = self.get_metadata(QualifiedNameProvider, func_expr)

        if not qname_metadata: # No direct qualified name for the func_expr itself
            if isinstance(func_expr, cst.Attribute): # e.g. obj.method or Class.method
                obj_node = func_expr.value
                method_name = func_expr.attr.value

                obj_qnames = self.get_metadata(QualifiedNameProvider, obj_node)
                obj_fqn_str: Optional[str] = None

                if obj_qnames: # QualifiedNameProvider found something for the object part
                    # Assuming the first one is the most relevant if multiple exist (e.g. complex import)
                    obj_name_candidate = obj_qnames[0].name
                    if '.' not in obj_name_candidate and self.module_qname: # Simple name, needs module prefix
                        obj_fqn_str = f"{self.module_qname}.{obj_name_candidate}"
                    else: # Already qualified or global
                        obj_fqn_str = obj_name_candidate

                if obj_fqn_str:
                    resolved_fqn_str = f"{obj_fqn_str}.{method_name}"
                elif isinstance(obj_node, cst.Name) and obj_node.value == "self":
                     # Call is self.method_name
                     # current_fqn_stack is like [module_qname, ClassName, current_method_name]
                     # We want module_qname.ClassName.method_name
                    if len(self.current_fqn_stack) > 1: # Should be at least [module, class/func]
                        # If current_fqn_stack[-1] is a method, its parent is the class
                        # For now, assume current_fqn_stack[-1] is the class if not module.
                        # This needs to align with how current_fqn_stack is managed.
                        # Let's assume the stack is [module, class, method] or [module, class] if in __init__ or class body
                        # The caller_fqn from _get_current_caller_fqn() is module.class.method or module.func
                        # So, if it's a method, its parent is the class.
                        caller_fqn = self._get_current_caller_fqn()
                        if caller_fqn:
                            caller_parts = str(caller_fqn).split('.')
                            if len(caller_parts) > 1: # module.class.method or module.func
                                # If caller is module.class.method, then class_fqn is module.class
                                # If caller is module.func, this 'self' case shouldn't apply unless func defines classes with self.
                                # Heuristic: if current stack implies a class context.
                                # current_fqn_stack could be [module, class] or [module, class, method]
                                class_name_from_stack = None
                                for i in range(len(self.current_fqn_stack) -1, 0, -1): # Find innermost class from stack
                                    # This is still heuristic. A proper scope provider might be better.
                                    # For now, if second to last is not module, assume it's class context for self.
                                    if self.current_fqn_stack[i] != self.module_qname:
                                        # Need to construct FQN up to this class part from the stack
                                        class_name_from_stack = ".".join(self.current_fqn_stack[:i+1])
                                        break
                                if class_name_from_stack:
                                     resolved_fqn_str = f"{class_name_from_stack}.{method_name}"
                                # else: self call in module scope? Unlikely / error.

                # If obj_node is cst.Call (get_obj().method()), this path won't resolve it well yet.
                # Deferring complex dynamic resolution.

            elif isinstance(func_expr, cst.Name): # Direct call e.g. my_func() or MyClass()
                # QualifiedNameProvider on func_expr itself might have failed if it's truly local
                # and not imported or qualified.
                # This was the original intent of the qname_metadata check.
                # If qname_metadata was None here, it implies it's a name not resolved by QNP.
                # This means it's likely a local definition or needs module prefix.
                resolved_fqn_str = f"{self.module_qname}.{func_expr.value}"
                # Heuristic for constructors: if it looks like PascalCase, append .__init__
                if re.match(r"^[A-Z]", func_expr.value) and not resolved_fqn_str.endswith(".__init__"):
                    resolved_fqn_str += ".__init__"


        else: # qname_metadata is not None, meaning QualifiedNameProvider resolved func_expr directly
            # This handles imported functions/classes or fully qualified calls directly on func_expr
            resolved_fqn_str = qname_metadata[0].name # Take the first one if multiple
            # Heuristic for constructors if QNP gave FQN of class
            if isinstance(func_expr, cst.Name) and re.match(r"^[A-Z]", func_expr.value) and not resolved_fqn_str.endswith(".__init__"):
                # Check if the resolved FQN points to a class (might need type_info or symbol table)
                # For now, apply heuristic: if original call was PascalCase name, assume constructor.
                # This needs to be careful not to append __init__ to an already fully resolved method.
                # QNP might resolve 'MyClass' to 'module.MyClass'.
                # If a type system indicated 'module.MyClass' is a class type, then append __init__.
                # This is where type_info would be helpful. For now, retain heuristic.
                # Check if the resolved FQN doesn't already look like a method path within a class
                # (e.g. if QNP resolved an alias directly to a method)
                if '.' not in func_expr.value: # Only apply to simple names like MyClass()
                     resolved_fqn_str += ".__init__"


        if resolved_fqn_str:
            return NodeID(resolved_fqn_str)

        # Fallback if no FQN could be resolved
        # This might log or return a more generic placeholder if needed
        if self.module_qname: # Check if verbose is available via self.module_qname to avoid error
            if hasattr(self, 'verbose') and self.verbose:
                 print(f"CallGraphVisitor: Could not resolve FQN for call: {cst.Module([call_node.func]).code.strip()} in {self.module_qname}")
        return None


    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
        fqn_parts = [self.module_qname]
        if self.current_fqn_stack[-1] != self.module_qname : # Inside a class
             fqn_parts.append(self.current_fqn_stack[-1])
        fqn_parts.append(node.name.value)
        # This constructed FQN is for the *definition*. We push only the local name.
        self.current_fqn_stack.append(node.name.value)

    def leave_FunctionDef(self, original_node: cst.FunctionDef) -> None:
        if self.current_fqn_stack and self.current_fqn_stack[-1] == original_node.name.value:
            self.current_fqn_stack.pop()

    def visit_ClassDef(self, node: cst.ClassDef) -> None:
        self.current_fqn_stack.append(node.name.value)

    def leave_ClassDef(self, original_node: cst.ClassDef) -> None:
        if self.current_fqn_stack and self.current_fqn_stack[-1] == original_node.name.value:
            self.current_fqn_stack.pop()

    def visit_Call(self, node: cst.Call) -> None:
        caller_fqn = self._get_current_caller_fqn()
        if caller_fqn:
            callee_fqn = self._resolve_callee_fqn(node)
            if callee_fqn:
                self.project_call_graph.setdefault(caller_fqn, set()).add(callee_fqn)

class RepositoryDigester:
    # Class-level type hints for new graph attributes
    project_module_imports: Dict['NodeID', Set['NodeID']]
    project_symbol_imports: Dict['NodeID', Dict['NodeID', Set[str]]]
    project_inheritance_graph: Dict['NodeID', List['NodeID']]

    if TYPE_CHECKING: # For forward reference of RepositoryDigester in its own methods' type hints if needed
        pass # This is mostly for linters if RepositoryDigester methods hint RepositoryDigester

    def __init__(self,
                 repo_path: Union[str, Path],
                 app_config: Dict[str, Any]
                ):
        """
        Initializes the RepositoryDigester.

        Args:
            repo_path: The path to the repository to be digested.
            app_config: Application configuration dictionary. Settings like 'verbose'
                        and 'embedding_model_name_or_path' are derived from this.
        """
        self.repo_path = Path(repo_path).resolve()
        self.app_config = app_config # Store app_config
        self.verbose = self.app_config.get("general", {}).get("verbose", False)

        if not self.repo_path.is_dir():
            raise ValueError(f"Repository path {self.repo_path} is not a valid directory.")

        self._all_py_files: List[Path] = []
        self.digested_files: Dict[Path, ParsedFileResult] = {}

        self.project_call_graph: CallGraph = {}
        self.project_control_dependence_graph: ControlDependenceGraph = {}
        self.project_data_dependence_graph: DataDependenceGraph = defaultdict(set)

        # New graph initializations
        self.project_module_imports: Dict[NodeID, Set[NodeID]] = {}
        self.project_symbol_imports: Dict[NodeID, Dict[NodeID, Set[str]]] = defaultdict(lambda: defaultdict(set))
        self.project_inheritance_graph: Dict[NodeID, List[NodeID]] = {}

        self.signature_trie = SignatureTrie()
        self.fqn_to_symbol_map: Dict[NodeID, Dict[str, Any]] = {}
        self.observer: Optional[Observer] = None # For watchdog

        self._faiss_dirty_flag: bool = False # Initialize FAISS dirty flag

        self.ts_parser: Optional[Parser] = None
        if Parser and get_language:
            try:
                if get_parser: self.ts_parser = get_parser('python')
                else:
                    PYTHON_LANGUAGE = get_language('python')
                    self.ts_parser = Parser(); self.ts_parser.set_language(PYTHON_LANGUAGE) # type: ignore
            except Exception as e: print(f"Warning: Failed to initialize Tree-sitter Python parser: {e}."); self.ts_parser = None
        else: print("Warning: tree-sitter or tree-sitter-languages not found. Tree-sitter parsing disabled."); self.ts_parser = None

        self.pyanalyze_checker: Optional[Checker] = None
        if pyanalyze and Checker and Config and Options:
            try:
                pyanalyze_options = Options(paths=[str(self.repo_path)])
                pyanalyze_config = Config.from_options(pyanalyze_options)
                self.pyanalyze_checker = Checker(config=pyanalyze_config)
            except Exception as e: print(f"RepositoryDigester Warning: Failed to initialize Pyanalyze Checker: {e}."); self.pyanalyze_checker = None
        elif not (pyanalyze and Checker and Config and Options):
             print("RepositoryDigester Warning: Pyanalyze library or its core components not found. Type inference disabled.")
             self.pyanalyze_checker = None

        # Embedding model initialization
        self.embedding_model: Optional[SentenceTransformer] = None
        self.embedding_dimension: Optional[int] = None

        embedding_model_name_or_path = self.app_config.get("models", {}).get("sentence_transformer_model", "all-MiniLM-L6-v2")
        if self.verbose:
            print(f"RepositoryDigester Info: Using embedding model: {embedding_model_name_or_path} (from app_config)")

        if SentenceTransformer and embedding_model_name_or_path:
            model_load_path: Optional[Union[str, Path]] = None
            default_local_st_model_path = Path("./models/sentence_transformer_model/")

            # 1. Check if embedding_model_name_or_path is a direct path to an existing model directory
            potential_path = Path(embedding_model_name_or_path)
            if potential_path.is_dir():
                model_load_path = potential_path
                if self.verbose: print(f"RepositoryDigester Info: Attempting to load SentenceTransformer model from provided path: {model_load_path}")
            # 2. Else, if it's the default HF name AND default local path exists, use local
            elif embedding_model_name_or_path == 'all-MiniLM-L6-v2' and default_local_st_model_path.is_dir():
                model_load_path = default_local_st_model_path
                if self.verbose: print(f"RepositoryDigester Info: Found default local SentenceTransformer model at: {model_load_path}. Prioritizing this.")
            # 3. Else, treat embedding_model_name_or_path as a Hugging Face model name (for download or cache)
            else:
                model_load_path = embedding_model_name_or_path
                if self.verbose:
                    if is_placeholder_path := embedding_model_name_or_path.endswith("sentence_transformer_model/"): # Check if it's the default placeholder
                        print(f"RepositoryDigester Warning: Provided model path '{embedding_model_name_or_path}' seems to be a placeholder or does not exist. Will attempt to load from Hugging Face if it's a valid model name.")
                    else:
                        print(f"RepositoryDigester Info: Attempting to load SentenceTransformer model from Hugging Face Hub: '{embedding_model_name_or_path}'")

            if model_load_path:
                try:
                    self.embedding_model = SentenceTransformer(str(model_load_path))
                    if self.embedding_model:
                        self.embedding_dimension = self.embedding_model.get_sentence_embedding_dimension()
                    if self.verbose: print(f"RepositoryDigester Info: SentenceTransformer model '{model_load_path}' loaded successfully. Dimension: {self.embedding_dimension}.")
                except Exception as e:
                    print(f"RepositoryDigester Error: Failed to load SentenceTransformer model from '{model_load_path}': {e}")
                    self.embedding_model = None
            else: # Should not happen if logic above is correct
                 if self.verbose: print("RepositoryDigester Warning: No valid model path or name determined for SentenceTransformer.")

        elif not SentenceTransformer:
            print("RepositoryDigester Warning: sentence-transformers library not available. Embedding generation disabled.")
        elif not embedding_model_name_or_path:
            print("RepositoryDigester Warning: No embedding_model_name provided. Embedding generation disabled.")


        # FAISS Index Initialization (depends on successful model loading)
        # TODO: Implement multi-level FAISS caching (RAM cache + on-disk main index) for very large repos if IndexFlatL2 RAM usage becomes an issue.
        self.faiss_index: Optional[faiss.Index] = None
        self.faiss_id_to_metadata: List[Dict[str, Any]] = []
        self.faiss_index_path: Optional[Path] = None
        self.faiss_metadata_path: Optional[Path] = None
        self._faiss_dirty_flag = True # Assume dirty until loaded or successfully saved

        faiss_on_disk_path_str = self.app_config.get("digester", {}).get("faiss_on_disk_path")
        if faiss_on_disk_path_str:
            base_path = Path(faiss_on_disk_path_str)
            if not base_path.is_absolute():
                default_data_dir = Path(self.app_config.get("general", {}).get("data_dir", self.repo_path / ".agent_data"))
                base_path = (default_data_dir / base_path).resolve()

            self.faiss_index_path = base_path.parent / (base_path.name + ".faiss")
            self.faiss_metadata_path = base_path.parent / (base_path.name + "_metadata.json")
            self.faiss_index_path.parent.mkdir(parents=True, exist_ok=True)
            if self.verbose:
                print(f"RepositoryDigester Info: FAISS on-disk index path set to: {self.faiss_index_path}")
                print(f"RepositoryDigester Info: FAISS on-disk metadata path set to: {self.faiss_metadata_path}")

        loaded_from_disk = False
        if faiss and self.embedding_model and self.embedding_dimension and self.faiss_index_path and self.faiss_metadata_path:
            loaded_from_disk = self._load_faiss_index_and_metadata(self.faiss_index_path, self.faiss_metadata_path)
            if loaded_from_disk:
                self._faiss_dirty_flag = False # Loaded successfully, so not dirty initially
                if self.verbose: print(f"RepositoryDigester Info: FAISS index and metadata loaded from disk. Index contains {self.faiss_index.ntotal if self.faiss_index else 0} items.")
            else:
                if self.verbose: print("RepositoryDigester Info: Failed to load FAISS from disk or paths not set. Will create a new index if possible.")

        if not loaded_from_disk and faiss and self.embedding_model and self.embedding_dimension:
            try:
                if self.verbose: print(f"RepositoryDigester Info: Initializing new FAISS IndexFlatL2 with dimension {self.embedding_dimension}...")
                self.faiss_index = faiss.IndexFlatL2(self.embedding_dimension) # type: ignore
                self.faiss_id_to_metadata = [] # Ensure metadata is also reset
                self._faiss_dirty_flag = True # New index is considered dirty until first save
                if self.verbose: print("RepositoryDigester Info: New FAISS Index initialized in RAM.")
            except Exception as e:
                print(f"RepositoryDigester Error: Error initializing new FAISS index: {e}")
                self.faiss_index = None
        elif not faiss:
            print("RepositoryDigester Warning: faiss library not available. FAISS indexing disabled.")
        elif not self.embedding_model or not self.embedding_dimension:
            print("RepositoryDigester Warning: Embedding model not loaded or dimension unknown. FAISS indexing disabled.")

    def _load_faiss_index_and_metadata(self, index_path: Path, metadata_path: Path) -> bool:
        if not faiss: return False
        if self.verbose: print(f"RepositoryDigester: Attempting to load FAISS index from {index_path} and metadata from {metadata_path}")
        loaded_index = None
        loaded_metadata = []
        try:
            if index_path.exists() and index_path.is_file():
                loaded_index = faiss.read_index(str(index_path))
                if self.verbose: print(f"  FAISS index {index_path.name} loaded.")
            else:
                if self.verbose: print(f"  FAISS index file {index_path.name} not found.")
                return False

            if loaded_index.d != self.embedding_dimension:
                if self.verbose: print(f"  Error: Loaded FAISS index dimension {loaded_index.d} does not match model dimension {self.embedding_dimension}. Will create a new index.")
                return False

            if metadata_path.exists() and metadata_path.is_file():
                with open(metadata_path, "r", encoding='utf-8') as f:
                    loaded_metadata = json.load(f)
                if self.verbose: print(f"  Metadata {metadata_path.name} loaded ({len(loaded_metadata)} entries).")
            else:
                if self.verbose: print(f"  FAISS metadata file {metadata_path.name} not found. Cannot fully restore index state.")
                return False

            if loaded_index.ntotal != len(loaded_metadata):
                if self.verbose: print(f"  Error: Loaded FAISS index size {loaded_index.ntotal} does not match metadata entries {len(loaded_metadata)}. Will create new index.")
                return False

            self.faiss_index = loaded_index
            self.faiss_id_to_metadata = loaded_metadata
            if self.verbose: print("RepositoryDigester: FAISS index and metadata loaded successfully.")
            return True
        except Exception as e:
            if self.verbose: print(f"RepositoryDigester: Error loading FAISS index/metadata: {e}. A new index will be created if needed.")
            return False

    def _save_faiss_index_and_metadata(self, index_path: Path, metadata_path: Path) -> bool:
        if not faiss: return False
        if self.faiss_index is None:
            if self.verbose: print("RepositoryDigester: No FAISS index in memory to save.")
            return False
        if self.verbose: print(f"RepositoryDigester: Saving FAISS index to {index_path} and metadata to {metadata_path}")
        try:
            index_path.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self.faiss_index, str(index_path))
            if self.verbose: print(f"  FAISS index {index_path.name} saved ({self.faiss_index.ntotal} vectors).")

            with open(metadata_path, "w", encoding='utf-8') as f:
                json.dump(self.faiss_id_to_metadata, f, indent=2) # Use indent for readability of full JSON list
            if self.verbose: print(f"  Metadata {metadata_path.name} saved ({len(self.faiss_id_to_metadata)} entries).")
            return True
        except Exception as e:
            if self.verbose: print(f"RepositoryDigester: Error saving FAISS index/metadata: {e}")
            return False

    def _extract_symbols_and_docstrings_from_ast(
        self,
        py_ast_module: ast.Module,
        module_qname: str,
        file_path: Path,
        source_code_lines: List[str],
        type_info: Optional[Dict[str, str]]
    ) -> List[Dict[str, Any]]:

        visitor = SymbolAndSignatureExtractorVisitor(
            module_qname, str(file_path), source_code_lines,
            type_info,
            self.signature_trie,
            self # Pass reference to self (RepositoryDigester instance)
        )
        visitor.visit(py_ast_module)

        module_docstring_text = ast.get_docstring(py_ast_module, clean=False)
        if module_docstring_text:
            start_line, end_line = 1, 1
            if py_ast_module.body and isinstance(py_ast_module.body[0], ast.Expr) and \
               isinstance(py_ast_module.body[0].value, (ast.Constant if hasattr(ast, 'Constant') else ast.Str)):
                doc_expr_node_mod = py_ast_module.body[0]
                start_line = doc_expr_node_mod.lineno
                if hasattr(doc_expr_node_mod, 'end_lineno') and doc_expr_node_mod.end_lineno is not None:
                    end_line = doc_expr_node_mod.end_lineno
                else:
                    end_line = doc_expr_node_mod.lineno + len(module_docstring_text.splitlines()) -1
            else:
                 end_line = start_line + len(module_docstring_text.splitlines()) -1

            visitor.symbols_for_embedding.append({ # Appending to renamed list
                "fqn": module_qname, "item_type": "docstring_for_module", "content": module_docstring_text,
                "file_path": str(file_path), "start_line": start_line, "end_line": end_line
            })

        return visitor.symbols_for_embedding

    @staticmethod
    def _get_module_qname_from_path(file_path: Path, project_root: Path) -> str:
        try:
            relative_path = file_path.relative_to(project_root)
            return ".".join(relative_path.with_suffix("").parts)
        except ValueError: return file_path.stem

    @staticmethod
    def _get_fqn_for_ast_node(node: ast.AST, module_qname: str, current_class_name: Optional[str] = None) -> Optional[NodeID]:
        # This static method is for AST nodes, primarily for the graph building context later if needed.
        # CallGraphVisitor uses its own internal stack for LibCST FQN generation.
        node_name: Optional[str] = None
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            node_name = node.name
        if not node_name: return None
        if current_class_name: return NodeID(f"{module_qname}.{current_class_name}.{node_name}")
        else: return NodeID(f"{module_qname}.{node_name}")

    def discover_python_files(self, ignored_dirs: Optional[List[str]] = None, ignored_files: Optional[List[str]] = None):
        self._all_py_files = []
        default_ignored_dirs = {".venv", "venv", ".env", "env", "node_modules", ".git", "__pycache__", "docs", "examples", "tests", "test"}
        current_ignored_dirs = set(ignored_dirs) if ignored_dirs is not None else default_ignored_dirs
        default_ignored_files = {"setup.py"}
        current_ignored_files = set(ignored_files) if ignored_files is not None else default_ignored_files
        for py_file in self.repo_path.rglob("*.py"):
            try:
                if py_file.name in current_ignored_files: continue
                if any(part in current_ignored_dirs for part in py_file.relative_to(self.repo_path).parts[:-1]): continue
                self._all_py_files.append(py_file)
            except Exception as e: print(f"Warning: Could not process path {py_file} during discovery: {e}")
        print(f"RepositoryDigester: Discovered {len(self._all_py_files)} Python files.")

    def _infer_types_with_pyanalyze(self, file_path: Path, source_code: str) -> Optional[Dict[str, Any]]:
        if not self.pyanalyze_checker: # Covers if pyanalyze or Checker is None
            return {"info": "Pyanalyze type inference disabled (checker not initialized)."}
        if not pyanalyze or not hasattr(pyanalyze, 'ast_annotator') or \
           not hasattr(pyanalyze.ast_annotator, 'annotate_code') or not pyanalyze_dump_value:
            return {"info": "Pyanalyze components (ast_annotator or dump_value) unavailable."}

        original_sys_path = list(sys.path)
        paths_to_manage = []

        # Ensure project root is in sys.path for Pyanalyze to resolve project-level imports
        repo_path_str = str(self.repo_path.resolve())
        if repo_path_str not in original_sys_path:
            paths_to_manage.append(repo_path_str)
            sys.path.insert(0, repo_path_str)
            if self.pyanalyze_checker.config._path_options[0].startswith(repo_path_str): # type: ignore
                 print(f"RepositoryDigester: Added {repo_path_str} to sys.path for Pyanalyze.")

        # Add file's parent directory to allow Pyanalyze to resolve relative imports
        file_dir_str = str(file_path.parent.resolve())
        if file_dir_str not in sys.path and file_dir_str != repo_path_str:
            paths_to_manage.append(file_dir_str)
            sys.path.insert(0, file_dir_str) # Add parent dir with higher precedence than repo root for local relative imports
            print(f"RepositoryDigester: Added {file_dir_str} to sys.path for Pyanalyze.")

        type_info_map: Dict[str, Any] = {}
        module_qname = self._get_module_qname_from_path(file_path, self.repo_path)

        try:
            if not source_code.strip():
                return {"info": "Empty source code, Pyanalyze not run."}

            # Pyanalyze's annotate_code might modify the AST in place with 'inferred_value' attributes.
            annotated_ast = pyanalyze.ast_annotator.annotate_code(
                source_code,
                filename=str(file_path),
                module_name=module_qname, # Provide module name
                config=self.pyanalyze_checker.config, # Pass existing config
                # show_errors=False, # Default in annotate_code
                # verbose=False # Default in annotate_code
            ) # type: ignore

            if annotated_ast:
                # Pass module_qname for correct qualification of names
                visitor = PyanalyzeTypeExtractionVisitor(filename=str(file_path), module_qname=module_qname)
                visitor.visit(annotated_ast)
                type_info_map = visitor.type_info_map
                if not type_info_map:
                    type_info_map = {"info": f"Pyanalyze ran for {file_path.name} but no types extracted by visitor."}
            else:
                type_info_map = {"error": f"Pyanalyze annotate_code returned None for {file_path.name}."}
        except Exception as e:
            error_msg = f"Pyanalyze processing failed for {file_path.name}: {type(e).__name__} - {e}"
            print(f"RepositoryDigester Error: {error_msg}")
            type_info_map = {"error": error_msg}
        finally:
            # Restore sys.path
            current_sys_path = list(sys.path)
            new_sys_path = [p for p in current_sys_path if p not in paths_to_manage]
            sys.path = new_sys_path
            # Verify restoration (optional)
            # if len(sys.path) != len(original_sys_path) or any(p not in original_sys_path for p in sys.path if p in paths_to_manage):
            #    print(f"Warning: sys.path may not have been restored perfectly. Original len: {len(original_sys_path)}, New len: {len(sys.path)}")
        return type_info_map

    def parse_file(self, file_path: Path) -> ParsedFileResult:
        source_code = ""
        try:
            with open(file_path, "r", encoding="utf-8") as f: source_code = f.read()
        except Exception as e:
            return ParsedFileResult(file_path=file_path, source_code="",
                                    libcst_error=f"File read error: {e}",
                                    treesitter_has_errors=True, treesitter_error_message=f"File read error: {e}",
                                    type_info={"error": f"File read error: {e}"}, extracted_symbols=[]) # Ensure new field is present

        source_code_lines = source_code.splitlines()

        libcst_module_obj: Optional[cst.Module] = None; libcst_error_str: Optional[str] = None
        try: libcst_module_obj = cst.parse_module(source_code)
        except cst.ParserSyntaxError as e_cs: libcst_error_str = f"LibCST PSE: {e_cs.message}"
        except Exception as e_cg: libcst_error_str = f"LibCST Err: {e_cg}"

        treesitter_tree_obj: Optional[TreeSitterTree] = None; treesitter_errors_flag: bool = False; treesitter_error_msg: Optional[str] = None
        if self.ts_parser:
            try:
                treesitter_tree_obj = self.ts_parser.parse(bytes(source_code, "utf8"))
                if treesitter_tree_obj.root_node.has_error: treesitter_errors_flag=True; treesitter_error_msg="TS errors."
            except Exception as e_ts: treesitter_errors_flag=True; treesitter_error_msg = f"TS Ex: {e_ts}"
        else: treesitter_errors_flag=True; treesitter_error_msg = "TS not init."

        file_type_info: Optional[Dict[str, Any]] = None
        if not libcst_error_str and source_code.strip() :
            try: file_type_info = self._infer_types_with_pyanalyze(file_path, source_code)
            except Exception as e_ti: file_type_info = {"error": f"TypeInf Ex: {e_ti}"}
        elif not source_code.strip(): file_type_info = {"info": "TypeInf skip empty."}
        else: file_type_info = {"info": "TypeInf skip LCST/AST err."}

        extracted_symbols_list: List[Dict[str, Any]] = []
        py_ast_module_for_symbols: Optional[ast.Module] = None
        try:
            if source_code.strip():
                py_ast_module_for_symbols = ast.parse(source_code, filename=str(file_path))
        except SyntaxError as e_ast:
            print(f"Warning: AST parsing for symbol extraction failed in {file_path.name}: {e_ast}")

        if py_ast_module_for_symbols:
            module_qname = self._get_module_qname_from_path(file_path, self.repo_path)
            type_info_for_resolver : Optional[Dict[str,str]] = None
            if file_type_info and not file_type_info.get("error") and not file_type_info.get("info"):
                type_info_for_resolver = file_type_info

            try:
                extracted_symbols_list = self._extract_symbols_and_docstrings_from_ast(
                    py_ast_module_for_symbols, module_qname, file_path, source_code_lines,
                    type_info_for_resolver
                )
            except Exception as e_sym_extract:
                 print(f"Error during symbol/docstring extraction for {file_path.name}: {e_sym_extract}")
                 if isinstance(extracted_symbols_list, list): # Ensure it's a list before appending
                    extracted_symbols_list.append({"error": f"Symbol extraction failed: {e_sym_extract}", "fqn": module_qname, "item_type":"error"})


        return ParsedFileResult(
            file_path=file_path, source_code=source_code,
            libcst_module=libcst_module_obj, treesitter_tree=treesitter_tree_obj,
            libcst_error=libcst_error_str,
            treesitter_has_errors=treesitter_errors_flag, treesitter_error_message=treesitter_error_msg,
            type_info=file_type_info,
            extracted_symbols=extracted_symbols_list
        )

    def _build_call_graph_for_file(self, file_path: Path, parsed_result: ParsedFileResult) -> None:
        if not parsed_result.libcst_module:
            print(f"Skipping call graph for {file_path.name} due to missing LibCST module.")
            return
        print(f"Building call graph for {file_path.name}...")
        module_qname = self._get_module_qname_from_path(file_path, self.repo_path)
        try:
            visitor = CallGraphVisitor(module_qname, file_path, parsed_result.type_info, self.project_call_graph)
            parsed_result.libcst_module.visit(visitor)
        except Exception as e:
            print(f"Error building call graph for {file_path.name}: {e}")

    def _build_control_dependencies_for_file(self, file_path: Path, parsed_result: ParsedFileResult) -> None:
        if not parsed_result.source_code:
            print(f"Skipping control dependence graph for {file_path.name} due to missing source code.")
            return

        # Determine a consistent file identifier for NodeIDs
        file_id_prefix = ""
        try:
            # Use relative path from repo_path if possible, otherwise just the filename.
            file_id_prefix = str(file_path.relative_to(self.repo_path))
        except ValueError: # Not under repo_path (e.g. during isolated testing)
            file_id_prefix = file_path.name

        print(f"Building control dependence graph for {file_id_prefix}...")

        try:
            # Python's AST is generally sufficient for control flow structures
            py_ast_module = ast.parse(parsed_result.source_code, filename=str(file_path))

            visitor = ControlDependenceVisitor(file_id_prefix) # Pass the chosen file identifier string
            visitor.visit(py_ast_module)

            if visitor.control_dependencies:
                 print(f"  Found {sum(len(deps) for deps in visitor.control_dependencies.values())} control dependencies in {file_id_prefix}.")

            # Merge file-specific CDG into project-wide CDG
            for controller_node_id, dependent_nodes_set in visitor.control_dependencies.items():
                self.project_control_dependence_graph.setdefault(controller_node_id, set()).update(dependent_nodes_set)

        except SyntaxError as e:
            print(f"SyntaxError parsing {file_id_prefix} with ast for control dependence: {e}. Skipping CDG for this file.")
        except Exception as e:
            print(f"Error building control dependence graph for {file_id_prefix}: {e}")

    def _build_data_dependencies_for_file(self, file_path: Path, parsed_result: ParsedFileResult) -> None:
        if not parsed_result.source_code:
            print(f"Skipping data dependence graph for {file_path.name} due to missing source code.")
            return

        file_id_prefix = str(file_path.relative_to(self.repo_path)) if file_path.is_absolute() and self.repo_path in file_path.parents else file_path.name
        print(f"Building data dependence graph for {file_id_prefix}...")

        try:
            py_ast_module = ast.parse(parsed_result.source_code, filename=str(file_path))

            visitor = DataDependenceVisitor(file_id_prefix)
            visitor.visit(py_ast_module)

            if visitor.data_dependencies:
                 print(f"  Found {sum(len(deps) for deps in visitor.data_dependencies.values())} data dependencies in {file_id_prefix}.")

            # Merge file-specific dependencies into the project-wide graph
            for use_node_id, def_nodes_set in visitor.data_dependencies.items():
                self.project_data_dependence_graph.setdefault(use_node_id, set()).update(def_nodes_set)

        except SyntaxError as e:
            print(f"SyntaxError parsing {file_id_prefix} with ast for data dependence: {e}. Skipping DDG for this file.")
        except Exception as e:
            print(f"Error building data dependence graph for {file_id_prefix}: {e}")

    def digest_repository(self):
        self.discover_python_files()
        if not self._all_py_files: print("RepoDigester: No Python files."); return
        num_total_files = len(self._all_py_files)
        print(f"RepoDigester: Starting digestion: {num_total_files} files...")
        for i, py_file in enumerate(self._all_py_files):
            print(f"RepoDigester: File {i+1}/{num_total_files}: {py_file.name}...")
            parsed_result = self.digested_files.get(py_file)
            if not parsed_result:
                parsed_result = self.parse_file(py_file) # parse_file now includes type inference attempt
                self.digested_files[py_file] = parsed_result

            # Ensure primary parsing (LibCST or TreeSitter) was somewhat successful before graph building
            if parsed_result.libcst_module or (self.ts_parser and parsed_result.treesitter_tree):
                self._build_call_graph_for_file(py_file, parsed_result)
                self._build_control_dependencies_for_file(py_file, parsed_result)
                # For DDG, we need source_code that is ast-parsable.
                # _build_data_dependencies_for_file handles its own ast.parse errors.
                if parsed_result.source_code:
                    self._build_data_dependencies_for_file(py_file, parsed_result)

            # Populate fqn_to_symbol_map for the current file
            if parsed_result.extracted_symbols:
                for symbol_dict in parsed_result.extracted_symbols:
                    fqn = symbol_dict.get('fqn')
                    if fqn:
                        symbol_node_id = NodeID(fqn)
                        details = {
                            'fqn': fqn,
                            'file_path': str(symbol_dict.get('file_path')), # Ensure str
                            'start_line': symbol_dict.get('start_line'),
                            'end_line': symbol_dict.get('end_line'),
                            'item_type': symbol_dict.get('item_type'),
                        }
                        if 'signature_str' in symbol_dict: # Only add if present
                            details['signature_str'] = symbol_dict['signature_str']
                        self.fqn_to_symbol_map[symbol_node_id] = details


        # --- NEW: Embedding Generation for all collected symbols ---
        if self.embedding_model and np:
            print("RepositoryDigester: Starting embedding generation...")
            all_text_contents: List[str] = []
            all_symbol_references: List[Dict[str, Any]] = []

            for file_path_iter in self._all_py_files:
                parsed_result_item = self.digested_files.get(file_path_iter)
                if parsed_result_item and parsed_result_item.extracted_symbols:
                    for symbol_dict in parsed_result_item.extracted_symbols:
                        content = symbol_dict.get("content")
                        if isinstance(content, str) and content.strip():
                            all_text_contents.append(content)
                            all_symbol_references.append(symbol_dict)

            if all_text_contents:
                print(f"RepositoryDigester: Generating embeddings for {len(all_text_contents)} text items...")
                try:
                    embeddings_np_array = self.embedding_model.encode(all_text_contents, show_progress_bar=False) # type: ignore
                    print(f"RepositoryDigester Info: Generated {len(embeddings_np_array)} embeddings.")

                    if len(embeddings_np_array) == len(all_symbol_references):
                        for i, symbol_dict_ref in enumerate(all_symbol_references):
                            symbol_dict_ref["embedding"] = embeddings_np_array[i]
                    else:
                        print("RepositoryDigester Warning: Mismatch between number of text items and generated embeddings. Embeddings not stored back.")
                except Exception as e_embed:
                    print(f"RepositoryDigester Error: Error during embedding generation: {e_embed}")
            else:
                print("RepositoryDigester Info: No text content found to generate embeddings for.")
        elif not self.embedding_model:
            print("RepositoryDigester Warning: Embedding model not available. Skipping embedding generation.")
        elif not np: # Should always be available if SentenceTransformer is, but good check.
            print("RepositoryDigester Warning: Numpy not available. Skipping embedding generation.")

        # --- FAISS Index Population ---
        if self.faiss_index is not None and np is not None: # np check for np.vstack and np.float32
            print("RepositoryDigester Info: Populating FAISS index...")
            embeddings_to_add_list: List[NumpyNdarray] = [] # type: ignore
            metadata_to_add_list: List[Dict[str, Any]] = []

            for file_path_ordered in self._all_py_files: # Ensure consistent order if IDs depend on it
                parsed_result_item = self.digested_files.get(file_path_ordered)
                if parsed_result_item and parsed_result_item.extracted_symbols:
                    for symbol_dict in parsed_result_item.extracted_symbols:
                        if "embedding" in symbol_dict and isinstance(symbol_dict["embedding"], np.ndarray): # type: ignore
                            embeddings_to_add_list.append(symbol_dict["embedding"])
                            metadata_to_add_list.append({
                                "fqn": symbol_dict.get("fqn"), "item_type": symbol_dict.get("item_type"),
                                "file_path": str(symbol_dict.get("file_path")),
                                "start_line": symbol_dict.get("start_line"), "end_line": symbol_dict.get("end_line"),
                            })
            if embeddings_to_add_list:
                try:
                    embeddings_2d_array = np.vstack(embeddings_to_add_list).astype(np.float32) # type: ignore
                    self.faiss_index.add(embeddings_2d_array) # type: ignore
                    self.faiss_id_to_metadata.extend(metadata_to_add_list)
                    print(f"RepositoryDigester Info: Added {self.faiss_index.ntotal} embeddings to FAISS index.") # type: ignore
                except Exception as e_faiss_add: # More specific FAISS add error
                    print(f"RepositoryDigester Error: Error adding embeddings to FAISS index: {e_faiss_add}")
            else:
                print("RepositoryDigester Info: No embeddings found to add to FAISS index.")
        elif not self.faiss_index:
            print("RepositoryDigester Warning: FAISS index not available. Skipping FAISS population.")
        elif not np:
            print("RepositoryDigester Warning: Numpy not available. Skipping FAISS population.")

        print(f"RepositoryDigester: Digestion complete. Processed files: {len(self.digested_files)}")
        files_fully_ok_both_parsers = sum(1 for r in self.digested_files.values() if r.libcst_module and not r.libcst_error and r.treesitter_tree and not r.treesitter_has_errors)
        files_with_any_libcst_error = sum(1 for res in self.digested_files.values() if res.libcst_error)
        files_with_any_treesitter_issue = sum(1 for res in self.digested_files.values() if res.treesitter_has_errors)
        files_with_type_info = sum(1 for res in self.digested_files.values() if res.type_info and not res.type_info.get("error") and not res.type_info.get("info"))

        num_embedded_items = 0
        if self.embedding_model:
            for res in self.digested_files.values():
                if res.extracted_symbols:
                    for sym_dict in res.extracted_symbols:
                        if "embedding" in sym_dict:
                            num_embedded_items +=1

        print(f"  Total Python files found: {len(self._all_py_files)}")
        print(f"  Files for which parsing was attempted: {len(self.digested_files)}")
        print(f"  Files successfully parsed by LibCST & Tree-sitter (no errors): {files_fully_ok_both_parsers}")
        print(f"  Files with LibCST parsing errors: {files_with_any_libcst_error}")
        print(f"  Files with Tree-sitter parsing issues: {files_with_any_treesitter_issue}")
        print(f"  Files with some type info extracted: {files_with_type_info}")
        print(f"  Total symbols/docstrings with embeddings: {num_embedded_items}")
        if self.faiss_index:
            print(f"  Total items in FAISS index: {self.faiss_index.ntotal}")

        self._faiss_dirty_flag = False # Mark as clean after initial full digest and build
        if self.verbose: print("RepositoryDigester: Initial FAISS index built, dirty flag set to False.")

    # --- New Incremental Update Interface and Internal Methods ---

    def _rebuild_full_faiss_index(self) -> None:
        """Clears and rebuilds the FAISS index from all current digested_files embeddings."""
        if not self.embedding_model or not self.embedding_dimension or not faiss or not np:
            print("RepositoryDigester Error: Cannot rebuild FAISS index. Missing embedding model, FAISS, or NumPy.")
            if hasattr(self, 'faiss_index') and self.faiss_index is not None and self.embedding_dimension and faiss: # Check faiss here too
                try:
                    self.faiss_index = faiss.IndexFlatL2(self.embedding_dimension) # Reset to empty
                    self.faiss_id_to_metadata = []
                    if self.verbose: print("RepositoryDigester Info: FAISS index reset to empty due to missing dependencies for full rebuild.")
                except Exception as e_reset:
                    print(f"RepositoryDigester Warning: Could not reset FAISS index during failed rebuild attempt: {e_reset}")
            else: # If faiss is None, can't even reset with faiss.IndexFlatL2
                 self.faiss_index = None
                 self.faiss_id_to_metadata = []
            return

        if self.verbose: print("RepositoryDigester: Rebuilding full FAISS index...")
        # Reset FAISS index and metadata
        try:
            self.faiss_index = faiss.IndexFlatL2(self.embedding_dimension)
        except Exception as e_faiss_init:
            print(f"RepositoryDigester Error: Error re-initializing FAISS index: {e_faiss_init}")
            self.faiss_index = None
            self.faiss_id_to_metadata = []
            return

        self.faiss_id_to_metadata = [] # Clear existing metadata

        embeddings_to_rebuild_list: List[NumpyNdarray] = [] # type: ignore
        metadata_to_rebuild_list: List[Dict[str, Any]] = []

        for parsed_result in self.digested_files.values(): # Iterate through all currently digested files
            if parsed_result and parsed_result.extracted_symbols:
                for symbol_dict in parsed_result.extracted_symbols:
                    if "embedding" in symbol_dict and isinstance(symbol_dict["embedding"], np.ndarray): # type: ignore
                        embeddings_to_rebuild_list.append(symbol_dict["embedding"])
                        metadata_to_rebuild_list.append({
                            "fqn": symbol_dict.get("fqn"), "item_type": symbol_dict.get("item_type"),
                            "file_path": str(symbol_dict.get("file_path")),
                            "start_line": symbol_dict.get("start_line"), "end_line": symbol_dict.get("end_line"),
                        })

        if embeddings_to_rebuild_list and self.faiss_index is not None:
            try:
                embeddings_2d_array_rebuild = np.vstack(embeddings_to_rebuild_list).astype(np.float32) # type: ignore
                self.faiss_index.add(embeddings_2d_array_rebuild) # type: ignore
                self.faiss_id_to_metadata = metadata_to_rebuild_list # Replace with new full list
                print(f"RepositoryDigester Info: FAISS index rebuilt with {self.faiss_index.ntotal} items.") # type: ignore
            except Exception as e_rebuild_add:
                print(f"RepositoryDigester Error: Error adding embeddings during FAISS index rebuild: {e_rebuild_add}")
                # Attempt to reset to a clean state if add fails mid-rebuild
                try:
                    if self.embedding_dimension: self.faiss_index = faiss.IndexFlatL2(self.embedding_dimension) # type: ignore
                    else: self.faiss_index = None # Should not happen if model loaded
                except Exception as e_final_reset: print(f"RepositoryDigester Error: Critical error resetting FAISS index post-rebuild failure: {e_final_reset}")
                else:
                    self.faiss_index = None
                self.faiss_id_to_metadata = []
        elif not embeddings_to_rebuild_list:
            print("RepositoryDigester: No embeddings found to rebuild FAISS index.")


    def _clear_data_for_file(self, file_path: Path) -> None:
        file_id_str_to_match = str(file_path.relative_to(self.repo_path)) if file_path.is_absolute() and self.repo_path.is_absolute() and self.repo_path in file_path.parents else file_path.name
        print(f"RepositoryDigester: Clearing data for file: {file_id_str_to_match}")

        old_parsed_result = self.digested_files.get(file_path)

        module_qname = self._get_module_qname_from_path(file_path, self.repo_path)
        module_node_id = NodeID(module_qname)

        # Clear project_module_imports
        if module_node_id in self.project_module_imports:
            del self.project_module_imports[module_node_id]
            if self.verbose: print(f"  Cleared module imports for {module_qname}.")

        # Clear project_symbol_imports
        if module_node_id in self.project_symbol_imports:
            del self.project_symbol_imports[module_node_id]
            if self.verbose: print(f"  Cleared symbol imports for {module_qname}.")

        # Clear from fqn_to_symbol_map, signature_trie, and project_inheritance_graph
        if old_parsed_result and old_parsed_result.extracted_symbols:
            cleared_classes_count_for_inheritance = 0
            for symbol_info in old_parsed_result.extracted_symbols:
                if 'fqn' in symbol_info:
                    symbol_node_id = NodeID(symbol_info['fqn'])
                    if symbol_node_id in self.fqn_to_symbol_map:
                        try:
                            del self.fqn_to_symbol_map[symbol_node_id]
                            if self.verbose: print(f"    Digester: Removed FQN '{symbol_info['fqn']}' from fqn_to_symbol_map.")
                        except KeyError: # Should not happen if check `in` is done
                            if self.verbose: print(f"    Digester Warning: FQN '{symbol_info['fqn']}' was in keys but not deleted from fqn_to_symbol_map.")
                    elif self.verbose:
                         print(f"    Digester Info: FQN '{symbol_info['fqn']}' not found in fqn_to_symbol_map for removal.")

                # Clear from signature_trie (existing logic, combined into this loop)
                if hasattr(self, 'signature_trie'): # Check if signature_trie exists
                    signature_to_delete = symbol_info.get("signature_str")
                    fqn_of_symbol_for_trie = symbol_info.get("fqn") # fqn is already retrieved above
                    item_type_for_trie = symbol_info.get("item_type", "")
                    if item_type_for_trie.endswith("_code") and ("function_code" in item_type_for_trie or "method_code" in item_type_for_trie):
                        if signature_to_delete and fqn_of_symbol_for_trie:
                            deleted_from_trie = self.signature_trie.delete(signature_to_delete, fqn_of_symbol_for_trie)
                            if self.verbose and deleted_from_trie:
                                print(f"    Digester: Deleted signature '{signature_to_delete}' for FQN '{fqn_of_symbol_for_trie}' from trie.")

                # Clear from project_inheritance_graph (new logic)
                if symbol_info.get("item_type") == "class_code":
                    class_fqn_for_inheritance = symbol_info.get("fqn")
                    if class_fqn_for_inheritance:
                        class_node_id_for_inheritance = NodeID(class_fqn_for_inheritance)
                        if class_node_id_for_inheritance in self.project_inheritance_graph:
                            del self.project_inheritance_graph[class_node_id_for_inheritance]
                            cleared_classes_count_for_inheritance += 1

            if self.verbose and cleared_classes_count_for_inheritance > 0:
                print(f"  Cleared inheritance graph entries for {cleared_classes_count_for_inheritance} classes from {module_qname}.")

        # This block for clearing signature_trie separately is now redundant due to above integration
        # if old_parsed_result and hasattr(old_parsed_result, 'extracted_symbols') and old_parsed_result.extracted_symbols and hasattr(self, 'signature_trie'):
        #     if self.verbose: print(f"  Digester: Clearing signatures from trie for file: {file_id_str_to_match}")
        #     for symbol_info in old_parsed_result.extracted_symbols:
        #         signature_to_delete = symbol_info.get("signature_str")
                fqn_of_symbol = symbol_info.get("fqn")
                item_type = symbol_info.get("item_type", "")

                # Only attempt to delete signatures for items that are function/method code entries
                if item_type.endswith("_code") and ("function_code" in item_type or "method_code" in item_type):
                    if signature_to_delete and fqn_of_symbol:
                        deleted = self.signature_trie.delete(signature_to_delete, fqn_of_symbol)
                        if self.verbose:
                            if deleted:
                                print(f"    Digester: Deleted signature '{signature_to_delete}' for FQN '{fqn_of_symbol}' from trie.")
                            else:
                                # This can be normal if a signature was shared and another FQN still uses it,
                                # or if it was already pruned due to another FQN removal.
                                print(f"    Digester Info: Signature '{signature_to_delete}' for FQN '{fqn_of_symbol}' not found for direct removal or already pruned.")
                    elif self.verbose:
                        # This print was part of the old block, now covered by general trie deletion log if verbose
                        # print(f"    Digester Info: Symbol {fqn_of_symbol} of type {item_type} missing signature_str for trie deletion.")
        # The verbose message for no symbols/trie for cleanup will be implicitly handled if the main loop isn't entered or if trie isn't present.
        # else: # This was part of the separate signature_trie block
        #    if self.verbose:
        #        print(f"  Digester: No symbols or signature trie found for {file_id_str_to_match}, skipping signature cleanup.")

        graphs_to_clean = [
            self.project_call_graph,
            self.project_control_dependence_graph,
            self.project_data_dependence_graph
        ]
        for graph in graphs_to_clean:
            if not isinstance(graph, dict): continue
            keys_to_delete = [k for k in graph.keys() if isinstance(k, str) and k.startswith(file_id_str_to_match + ":")]
            for k_del in keys_to_delete:
                del graph[k_del]
            for k_remaining in list(graph.keys()):
                if isinstance(graph[k_remaining], set):
                    graph[k_remaining] = {val for val in graph[k_remaining] if not (isinstance(val, str) and val.startswith(file_id_str_to_match + ":"))}
                    if not graph[k_remaining]:
                        del graph[k_remaining]
        print(f"  Graphs: Removed nodes and edges related to {file_id_str_to_match}.")

        if file_path in self.digested_files:
            del self.digested_files[file_path]
            if self.verbose: print(f"  Removed {file_id_str_to_match} from digested_files cache.")

        # TODO: The current cleanup for new KG structures (imports, inheritance) primarily removes
        # entries where the current file is the 'source' (e.g., the importer, the derived class).
        # A more thorough cleanup would also update other entries that might refer to entities
        # from this file (e.g., remove this module from the import lists of other modules that imported it).
        # For now, a full re-digest or re-parse of affected files handles this more broadly.

        print(f"RepositoryDigester: Data clearing for {file_id_str_to_match} completed (FAISS will be rebuilt after all changes).")


    def _update_or_add_file(self, file_path: Path) -> None:
        print(f"RepositoryDigester: Updating/Adding file: {file_path.name}")
        self._clear_data_for_file(file_path)

        parsed_result = self.parse_file(file_path)
        self.digested_files[file_path] = parsed_result

        # Populate fqn_to_symbol_map for the new/updated file's symbols
        if parsed_result.extracted_symbols:
            for symbol_dict in parsed_result.extracted_symbols:
                if 'fqn' in symbol_dict:
                    symbol_node_id = NodeID(symbol_dict['fqn'])
                    self.fqn_to_symbol_map[symbol_node_id] = {
                        'fqn': symbol_dict['fqn'],
                        'file_path': str(symbol_dict.get('file_path')), # Ensure str
                        'start_line': symbol_dict.get('start_line'),
                        'end_line': symbol_dict.get('end_line'),
                        'item_type': symbol_dict.get('item_type'),
                    }
                    if 'signature_str' in symbol_dict: # Only add if present
                        details['signature_str'] = symbol_dict['signature_str']
                    self.fqn_to_symbol_map[symbol_node_id] = details
            if self.verbose: print(f"  Digester: Updated fqn_to_symbol_map for {file_path.name} with {len(parsed_result.extracted_symbols)} symbols.")

        if parsed_result.source_code and (parsed_result.libcst_module or parsed_result.treesitter_tree):
            py_ast_for_graphs: Optional[ast.Module] = None
            try:
                if parsed_result.source_code.strip():
                     py_ast_for_graphs = ast.parse(parsed_result.source_code, filename=str(file_path))
            except SyntaxError:
                print(f"Syntax error in {file_path.name} during update, skipping graph builds.")

            if py_ast_for_graphs:
                self._build_call_graph_for_file(file_path, parsed_result)
                self._build_control_dependencies_for_file(file_path, parsed_result)
                self._build_data_dependencies_for_file(file_path, parsed_result)

        # Embeddings are generated during parse_file via _extract_symbols_and_docstrings_from_ast.
        # Signature Trie is also populated during parse_file.
        self._faiss_dirty_flag = True # Mark FAISS as dirty, actual rebuild by commit_faiss_index_changes
        if self.verbose: print(f"RepositoryDigester: Marked FAISS index as dirty after updating/adding {file_path.name}.")
        # Removed direct call to self._rebuild_full_faiss_index()

        print(f"RepositoryDigester: Finished processing update/add for {file_path.name}")


    def _remove_file_data(self, file_path: Path) -> None:
        if self.verbose: print(f"RepositoryDigester: Removing data for deleted file: {file_path.name}")
        self._clear_data_for_file(file_path) # This clears symbols, graphs, and trie entries

        if file_path in self._all_py_files:
            try: self._all_py_files.remove(file_path)
            except ValueError: pass

        self._faiss_dirty_flag = True # Mark FAISS as dirty
        if self.verbose: print(f"RepositoryDigester: Marked FAISS index as dirty after removing data for {file_path.name}.")
        # Removed direct call to self._rebuild_full_faiss_index()
        print(f"RepositoryDigester: Finished processing removal for {file_path.name}")

    def commit_faiss_index_changes(self) -> bool:
        """
        Rebuilds the FAISS index if changes have been made to files since the last rebuild
        or initial digestion. This method should be called by an external orchestrator
        (e.g., after a batch of file events or before querying symbols).
        Returns:
            bool: True if a rebuild was performed or no rebuild was needed (index was clean),
                  False if rebuild was attempted but failed.
        """
        if not self.embedding_model:
            if self.verbose: print("RepositoryDigester Info: Embedding model not available. FAISS commit skipped, flag cleared.")
            self._faiss_dirty_flag = False
            return True # Considered success as no action was needed due to setup

        if not self.faiss_index:
            if self.verbose: print("RepositoryDigester Info: FAISS native library or index not available. FAISS commit skipped, flag cleared.")
            self._faiss_dirty_flag = False
            return True # Considered success as no action was needed due to setup

        if self._faiss_dirty_flag:
            if self.verbose: print("RepositoryDigester Info: FAISS index is dirty. Committing changes by rebuilding...")
            try:
                self._rebuild_full_faiss_index() # This method handles its own errors internally

                # After successful rebuild, attempt to save if paths are configured
                if self.faiss_index_path and self.faiss_metadata_path:
                    save_ok = self._save_faiss_index_and_metadata(self.faiss_index_path, self.faiss_metadata_path)
                    if save_ok:
                        self._faiss_dirty_flag = False # Clear dirty flag only if save is successful
                        if self.verbose: print("RepositoryDigester Info: FAISS index and metadata saved to disk.")
                    else:
                        # Keep dirty flag true if save failed, so it tries again next time
                        if self.verbose: print("RepositoryDigester Warning: Failed to save FAISS index/metadata to disk. Index remains dirty.")
                        return False # Indicate commit failed due to save error
                else:
                    # No on-disk path configured, so RAM index is now considered "committed"
                    self._faiss_dirty_flag = False

                if self.verbose: print("RepositoryDigester Info: FAISS index rebuild (and optional save) complete.")
                return True
            except Exception as e_rebuild:
                print(f"RepositoryDigester Error: Unexpected error during _rebuild_full_faiss_index via commit: {e_rebuild}")
                # Keep dirty flag true as rebuild failed
                return False
        else:
            if self.verbose: print("RepositoryDigester Info: FAISS index is not dirty. No rebuild needed.")
            return True # No action needed, considered success

    def handle_file_event(self, event_type: str, src_path: Path, dest_path: Optional[Path] = None) -> None:
        """
        Handles file system events (created, modified, deleted, moved) to keep the
        repository digest up-to-date.
        Note: After one or more calls to `handle_file_event`, `commit_faiss_index_changes()`
        should be called to update the searchable FAISS index.
        """
        if self.verbose: print(f"RepositoryDigester: Received event: {event_type} on {src_path}" + (f" -> {dest_path}" if dest_path else ""))

        abs_src_path = src_path.resolve()
        abs_dest_path = dest_path.resolve() if dest_path else None

        try:
            abs_src_path.relative_to(self.repo_path)
            if abs_dest_path: abs_dest_path.relative_to(self.repo_path)
        except ValueError:
            print(f"Warning: Event path {abs_src_path} (or {abs_dest_path}) is outside configured repo root {self.repo_path}. Ignoring.")
            return

        if event_type == "created":
            if abs_src_path not in self._all_py_files: self._all_py_files.append(abs_src_path)
            self._update_or_add_file(abs_src_path)
        elif event_type == "modified":
            # Ensure it's tracked if it was somehow missed (e.g. modified before initial scan completed)
            if abs_src_path not in self._all_py_files: self._all_py_files.append(abs_src_path)
            self._update_or_add_file(abs_src_path)
        elif event_type == "deleted":
            self._remove_file_data(abs_src_path)
        elif event_type == "moved":
            if abs_dest_path:
                self._remove_file_data(abs_src_path)
                if abs_dest_path not in self._all_py_files: self._all_py_files.append(abs_dest_path)
                self._update_or_add_file(abs_dest_path)
            else:
                print(f"Warning: 'moved' event for {abs_src_path} missing destination path.")
        else:
            print(f"Warning: Unknown event type '{event_type}' for path {abs_src_path}.")

    # --- Methods for Phase 5 Context Broadcast ---
    def get_code_snippets_for_phase(self, phase_ctx: Phase) -> Dict[str, str]:
        """
        Retrieves specific code snippets relevant to the phase, such as the source
        of a target function or class.
        """
        snippets: Dict[str, str] = {}
        target_file_str = phase_ctx.target_file
        parameters = phase_ctx.parameters if phase_ctx.parameters else {}

        if not target_file_str:
            if self.verbose:
                print("RepositoryDigester.get_code_snippets_for_phase: Warning - No target file specified in phase_ctx.")
            return {"error": "No target file specified in phase_ctx"}

        # Construct absolute path, assuming target_file_str is relative to repo_path
        target_file_path = self.repo_path / target_file_str
        if not target_file_path.is_absolute(): # Should already be absolute if repo_path is.
             target_file_path = target_file_path.resolve()


        parsed_result = self.digested_files.get(target_file_path)
        if not parsed_result or not parsed_result.source_code:
            return {"error": f"Source code not found or empty for {target_file_path}"}

        try:
            ast_tree = ast.parse(parsed_result.source_code, filename=str(target_file_path))
        except SyntaxError as e:
            return {"error": f"Syntax error in target file {target_file_path}: {e}"}

        target_function_name = parameters.get("target_function_name")
        target_class_name = parameters.get("target_class_name")
        snippet_found = False

        module_qname_for_key = self._get_module_qname_from_path(target_file_path, self.repo_path)

        if target_function_name:
            search_nodes = ast_tree.body
            current_class_node_for_search: Optional[ast.ClassDef] = None
            if target_class_name:
                for node in ast_tree.body:
                    if isinstance(node, ast.ClassDef) and node.name == target_class_name:
                        current_class_node_for_search = node
                        search_nodes = node.body # Search within the class
                        break
                if not current_class_node_for_search:
                    snippets["error"] = f"Class '{target_class_name}' not found in {target_file_path}"
                    return snippets

            for node in search_nodes:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == target_function_name:
                    segment = ast.get_source_segment(parsed_result.source_code, node)
                    if segment:
                        key = f"{module_qname_for_key}{'.' + target_class_name if target_class_name else ''}.{target_function_name}"
                        snippets[key] = segment
                        snippet_found = True
                        break
            if not snippet_found and not snippets.get("error"):
                snippets["error"] = f"Function '{target_function_name}' (in class '{target_class_name}' if specified) not found."

        elif target_class_name: # Only class name specified
            for node in ast_tree.body:
                if isinstance(node, ast.ClassDef) and node.name == target_class_name:
                    segment = ast.get_source_segment(parsed_result.source_code, node)
                    if segment:
                        key = f"{module_qname_for_key}.{target_class_name}"
                        snippets[key] = segment
                        snippet_found = True
                        break
            if not snippet_found:
                snippets["error"] = f"Class '{target_class_name}' not found."

        # If no specific snippet target was identified by parameters, or if found, this dict is returned.
        # If a target was specified but not found, the error is already in snippets.
        # If no target params, snippets remains empty.
        if not snippets and not target_function_name and not target_class_name:
            if self.verbose:
                print(f"RepositoryDigester.get_code_snippets_for_phase: No specific function/class target in phase_ctx for {target_file_str}. Consider returning full file or targeted lines if applicable for operation '{phase_ctx.operation_name}'.")
            # Fallback to full file content if no specific element is requested and it's a common operation type
            # For now, returning empty as per instruction "return an empty dict if more targeted snippets are always expected"
            # or if the specific element was not found. If an error occurred, it's already set.
            pass


        return snippets

    def get_pdg_slice_for_phase(self, phase_ctx: Phase) -> Dict[str, Any]: # Changed Any to Phase
        """
        Mock: Retrieves a program dependence graph (PDG) slice relevant to the phase.
        """
        # phase_ctx is 'Phase' from src.planner.phase_model
        print(f"RepositoryDigester.get_pdg_slice_for_phase: Mock for phase: {getattr(phase_ctx, 'operation_name', 'UnknownOp')}")
        # In a real implementation, this would query self.project_control_dependence_graph and self.project_data_dependence_graph
        # based on the phase_ctx (e.g., target file, specific functions/lines from phase_ctx.parameters)

        slice_nodes: List[Dict[str, Any]] = []
        slice_edges: List[Dict[str, Any]] = []
        info_parts: List[str] = [f"PDG Slice for phase: {phase_ctx.operation_name} on {phase_ctx.target_file or 'N/A'}."]
        added_node_ids_to_slice = set() # Keep track of NodeIDs already added to slice_nodes

        target_file_str = phase_ctx.target_file
        parameters = phase_ctx.parameters if phase_ctx.parameters else {}

        if not target_file_str:
            info_parts.append("Error: No target file specified.")
            return {"nodes": [], "edges": [], "info": " ".join(info_parts)}

        try:
            # Ensure target_file_rel_path is consistently used as a string relative path.
            # If target_file_str is absolute, make it relative. If already relative, ensure correct format.
            path_obj_target_file = Path(target_file_str)
            if path_obj_target_file.is_absolute():
                target_file_rel_path = str(path_obj_target_file.relative_to(self.repo_path))
            else: # Assume already relative to repo_path
                target_file_rel_path = str(Path(target_file_str)) # Normalize
        except ValueError:
            info_parts.append(f"Error: Target file '{target_file_str}' is not relative to repo_path '{self.repo_path}'.")
            return {"nodes": [], "edges": [], "info": " ".join(info_parts)}

        abs_target_file_path = self.repo_path / target_file_rel_path
        parsed_target_file_result = self.digested_files.get(abs_target_file_path)

        def _get_node_details(node_id_val: NodeID) -> Dict[str, Any]:
            node_id_str = str(node_id_val)
            parts = node_id_str.split(':')
            file_p = parts[0]
            lineno = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
            col_offset = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0
            node_type = parts[3] if len(parts) > 3 else "UnknownType"
            name = parts[4] if len(parts) > 4 else ""
            category = parts[5] if len(parts) > 5 else ""

            snippet = "Snippet not available (source file not found or issue)"
            node_abs_file_path = self.repo_path / file_p
            parsed_file_res = self.digested_files.get(node_abs_file_path)
            if parsed_file_res and parsed_file_res.source_code:
                lines = parsed_file_res.source_code.splitlines()
                if 0 <= lineno - 1 < len(lines):
                    snippet = lines[lineno - 1].strip()
                else:
                    snippet = "Snippet not available (line out of bounds)"

            return {
                "id": node_id_str, "label": f"{node_type}:{name}" if name else node_type,
                "file": file_p, "line": lineno, "col": col_offset,
                "category": category, "snippet": snippet
            }

        target_func_name = parameters.get("target_function_name")
        target_class_name = parameters.get("target_class_name")

        if target_func_name:
            info_parts.append(f"Targeting function: {target_func_name}" + (f" in class {target_class_name}" if target_class_name else "") + ".")

            # Determine the FQN prefix for _find_node_ids_in_graphs if class is involved
            # This is tricky as _find_node_ids_in_graphs uses simple name matching.
            # For now, assume target_func_name is unique enough within the file or class context.

            func_def_node_ids = self._find_node_ids_in_graphs(
                target_file_rel_path, target_func_name, "FunctionDef:def"
            )
            if not func_def_node_ids and target_class_name: # If it's a method, try with class context in name for some finders
                 # This part is heuristic and depends on how _find_node_ids_in_graphs and NodeID creation handle method names.
                 # For now, direct search for FunctionDef:def with simple name is primary.
                 pass

            for func_node_id in func_def_node_ids:
                if str(func_node_id) not in added_node_ids_to_slice:
                    slice_nodes.append(_get_node_details(func_node_id))
                    added_node_ids_to_slice.add(str(func_node_id))

                # Data Dependencies for the function
                # Incoming: Data used by the function (func_node_id is a 'use' node)
                for use_node, def_nodes_set in self.project_data_dependence_graph.items():
                    if use_node == func_node_id or (target_func_name in str(use_node) and target_file_rel_path in str(use_node)):
                        for def_node in def_nodes_set:
                            if str(def_node) not in added_node_ids_to_slice:
                                slice_nodes.append(_get_node_details(def_node))
                                added_node_ids_to_slice.add(str(def_node))
                            slice_edges.append({"from": str(def_node), "to": str(use_node), "type": "data"})

                # Outgoing: Data defined by the function that is used elsewhere (func_node_id is a 'def' node)
                for other_use_node, def_nodes_set_for_other_use in self.project_data_dependence_graph.items():
                    if func_node_id in def_nodes_set_for_other_use or \
                       any(target_func_name in str(dn) and target_file_rel_path in str(dn) for dn in def_nodes_set_for_other_use):
                        if str(other_use_node) not in added_node_ids_to_slice:
                            slice_nodes.append(_get_node_details(other_use_node))
                            added_node_ids_to_slice.add(str(other_use_node))
                        # Find which specific def_node from the set matches our func_node_id context
                        # For simplicity, assume if func_node_id is in the set, it's the source.
                        relevant_def_node = func_node_id # Or iterate to find the exact one if set contains multiple
                        slice_edges.append({"from": str(relevant_def_node), "to": str(other_use_node), "type": "data"})

                # Control Dependencies for the function's internals
                func_details = _get_node_details(func_node_id)
                func_start_line = func_details.get("line", 0)
                func_end_line = -1 # Placeholder, needs to be determined

                if parsed_target_file_result and parsed_target_file_result.extracted_symbols:
                    for sym in parsed_target_file_result.extracted_symbols:
                        if sym.get("fqn", "").endswith(target_func_name) and sym.get("start_line") == func_start_line:
                            func_end_line = sym.get("end_line", -1)
                            break

                if func_start_line > 0 and func_end_line > 0:
                    for controller_node_id_obj, dependent_nodes_set in self.project_control_dependence_graph.items():
                        controller_details = _get_node_details(controller_node_id_obj)
                        if controller_details["file"] == target_file_rel_path and \
                           func_start_line <= controller_details["line"] <= func_end_line:
                            if str(controller_node_id_obj) not in added_node_ids_to_slice:
                                slice_nodes.append(controller_details) # Already detailed from _get_node_details
                                added_node_ids_to_slice.add(str(controller_node_id_obj))
                            for dep_node_id_obj in dependent_nodes_set:
                                if str(dep_node_id_obj) not in added_node_ids_to_slice:
                                    slice_nodes.append(_get_node_details(dep_node_id_obj))
                                    added_node_ids_to_slice.add(str(dep_node_id_obj))
                                slice_edges.append({"from": str(controller_node_id_obj), "to": str(dep_node_id_obj), "type": "control"})
                else:
                    info_parts.append(f"Could not determine line range for function '{target_func_name}' for precise control dependency slicing.")

        elif target_class_name:
            info_parts.append(f"Targeting class: {target_class_name}.")
            class_def_node_ids = self._find_node_ids_in_graphs(target_file_rel_path, target_class_name, "ClassDef:def")

            method_node_ids_in_class: List[NodeID] = []
            if parsed_target_file_result and parsed_target_file_result.extracted_symbols:
                class_qname_prefix = f"{self._get_module_qname_from_path(abs_target_file_path, self.repo_path)}.{target_class_name}"
                for sym in parsed_target_file_result.extracted_symbols:
                    if sym.get("fqn","").startswith(class_qname_prefix + ".") and sym.get("item_type") == "method_code":
                        method_name = sym["fqn"].split('.')[-1]
                        # Find NodeID for this method
                        # Category needs to be consistent with how FunctionDef for methods are stored.
                        # Assuming "FunctionDef:def" or similar from _create_ast_node_id for methods too.
                        method_ids = self._find_node_ids_in_graphs(target_file_rel_path, method_name, "FunctionDef:def")
                        method_node_ids_in_class.extend(method_ids)


            nodes_to_process_for_deps = class_def_node_ids + method_node_ids_in_class
            for node_id_obj in nodes_to_process_for_deps:
                if str(node_id_obj) not in added_node_ids_to_slice:
                    slice_nodes.append(_get_node_details(node_id_obj))
                    added_node_ids_to_slice.add(str(node_id_obj))

                # Simplified Data Dependencies for class/methods (similar to function logic above)
                # Incoming
                for use_node, def_nodes_set in self.project_data_dependence_graph.items():
                    if use_node == node_id_obj: # Direct use of class/method itself (e.g. instantiation, call)
                        for def_node in def_nodes_set:
                            if str(def_node) not in added_node_ids_to_slice:
                                slice_nodes.append(_get_node_details(def_node))
                                added_node_ids_to_slice.add(str(def_node))
                            slice_edges.append({"from": str(def_node), "to": str(use_node), "type": "data"})
                # Outgoing
                for other_use_node, def_nodes_set_for_other_use in self.project_data_dependence_graph.items():
                    if node_id_obj in def_nodes_set_for_other_use: # Class/method defined something used elsewhere
                        if str(other_use_node) not in added_node_ids_to_slice:
                            slice_nodes.append(_get_node_details(other_use_node))
                            added_node_ids_to_slice.add(str(other_use_node))
                        slice_edges.append({"from": str(node_id_obj), "to": str(other_use_node), "type": "data"})

            info_parts.append(f"Found {len(class_def_node_ids)} def node(s) for class '{target_class_name}' and {len(method_node_ids_in_class)} methods.")
        else:
            info_parts.append("No specific function or class target. PDG slice might be limited.")

        info_parts.append(f"Final slice includes {len(slice_nodes)} nodes and {len(slice_edges)} edges.")
        return {
            "nodes": slice_nodes,
            "edges": slice_edges,
            "info": " ".join(info_parts)
        }

    def _find_node_ids_in_graphs(self, target_file_rel_path: str, target_name: str, target_category_suffix_in_node_id: str) -> List[NodeID]:
        """
        Helper to find NodeIDs in stored graphs based on file, name, and category.
        NodeID format assumed: "file_rel_path:lineno:col:NodeType:name:category_suffix"
                           or "file_rel_path:lineno:col:NodeType::category_suffix" (if no name in node like If condition)
                           or "file_rel_path:lineno:col:NodeType:name" (if no category suffix used in creation)
        """
        matching_node_ids: List[NodeID] = []

        # Create a flexible regex pattern for matching node IDs
        # Example: "file.py:10:4:FunctionDef:my_func:def"
        # target_file_rel_path needs to be regex escaped if it contains special chars.
        # For now, assume it's simple.
        # Pattern: file_path_str + :line:col:node_type:(name_part):category_suffix
        # name_part could be target_name or empty if not applicable for the node type

        # Search in Control Dependence Graph keys (controllers) and values (dependents)
        # Search in Data Dependence Graph keys (users) and values (definers)

        all_nodes_to_check = set(self.project_control_dependence_graph.keys())
        for dependents_set in self.project_control_dependence_graph.values():
            all_nodes_to_check.update(dependents_set)
        all_nodes_to_check.update(self.project_data_dependence_graph.keys())
        for definers_set in self.project_data_dependence_graph.values():
            all_nodes_to_check.update(definers_set)

        for node_id_str_obj in all_nodes_to_check:
            node_id_str = str(node_id_str_obj) # Ensure it's a string
            parts = node_id_str.split(':')
            if len(parts) < 4: continue # Minimal: file:line:col:NodeType

            node_file_path = parts[0]
            node_type = parts[3]
            node_name_part = parts[4] if len(parts) > 4 else "" # Name might be empty for some nodes
            node_category_suffix = parts[5] if len(parts) > 5 else ""

            # Match file path
            if node_file_path != target_file_rel_path:
                continue

            # Match name (if target_name is provided)
            if target_name and node_name_part != target_name:
                continue

            # Match category suffix
            # Handle cases where node_id might not have a category_suffix part, or it's combined with name.
            # The _create_ast_node_id function adds category_suffix if provided.
            # If target_category_suffix_in_node_id has multiple parts, e.g. "NodeType:def"
            # we need to match both.

            type_and_cat_parts = target_category_suffix_in_node_id.split(':', 1)
            expected_node_type = type_and_cat_parts[0]
            expected_suffix = type_and_cat_parts[1] if len(type_and_cat_parts) > 1 else ""

            if node_type != expected_node_type:
                continue

            if expected_suffix and node_category_suffix != expected_suffix:
                continue
            elif not expected_suffix and node_category_suffix: # If we want no suffix, but node has one
                pass # Allow if target_category_suffix_in_node_id was just NodeType

            matching_node_ids.append(NodeID(node_id_str))

        if self.verbose:
            print(f"RepositoryDigester._find_node_ids_in_graphs: Found {len(matching_node_ids)} for {target_file_rel_path}, {target_name}, {target_category_suffix_in_node_id}")
        return matching_node_ids

    def get_file_content(self, file_path: Path) -> Optional[str]:
        """
        Retrieves the source code of a file if it has been digested.
        Args:
            file_path: The absolute path to the file.
        Returns:
            The source code as a string, or None if the file is not found or not digested.
        """
        # Ensure file_path is absolute, matching keys in self.digested_files if they are absolute
        abs_file_path = file_path.resolve()

        # Check if this path (or its string representation if keys are strings) is in digested_files
        # The keys in self.digested_files are Path objects from self.repo_path.rglob("*.py")
        # which should be absolute after .resolve() during discover_python_files.

        parsed_result = self.digested_files.get(abs_file_path)
        if parsed_result:
            return parsed_result.source_code
        else:
            # Fallback: try to match by comparing string representations if Path objects don't match directly
            # (e.g. due to symlinks or slight path differences not caught by resolve)
            # This is less ideal but can be a fallback.
            str_file_path = str(abs_file_path)
            for path_key, result_val in self.digested_files.items():
                if str(path_key) == str_file_path:
                    return result_val.source_code

            print(f"RepositoryDigester Warning: File content not found for {abs_file_path}. It might not be a tracked Python file or wasn't digested.")
            return None

    def start_watchdog_observer(self) -> None:
        if not Observer or not FileSystemEventHandler: # Check if watchdog is available
            print("RepositoryDigester: Watchdog components not available. Cannot start observer.")
            return

        if self.observer and self.observer.is_alive():
            if self.verbose: print("RepositoryDigester: Watchdog observer is already running.")
            return

        event_handler = DigesterFileSystemEventHandler(self)
        self.observer = Observer()
        self.observer.schedule(event_handler, str(self.repo_path), recursive=True)

        try:
            self.observer.start()
            if self.verbose: print(f"RepositoryDigester: Watchdog observer started on {self.repo_path}.")
        except Exception as e:
            print(f"RepositoryDigester Error: Failed to start watchdog observer: {e}")
            self.observer = None # Ensure observer is None if start fails

    def stop_watchdog_observer(self) -> None:
        if not Observer: # Check if watchdog is available
            return

        if self.observer and self.observer.is_alive():
            try:
                self.observer.stop()
                self.observer.join(timeout=5) # Add a timeout to join
                if self.observer.is_alive():
                    print("RepositoryDigester Warning: Watchdog observer thread did not join in time.")
                else:
                     if self.verbose: print("RepositoryDigester: Watchdog observer stopped.")
            except Exception as e:
                print(f"RepositoryDigester Error: Exception while stopping watchdog observer: {e}")
            finally:
                self.observer = None
        elif self.verbose:
            print("RepositoryDigester: Watchdog observer is not running or not initialized.")

    def get_call_graph_slice(self, target_fqn_str: str, depth: int = 1, include_reverse: bool = True) -> Dict[str, Any]:
        """
        Retrieves a slice of the call graph around a target FQN.

        Args:
            target_fqn_str: The Fully Qualified Name (FQN) of the target function/method.
            depth: The maximum depth for outgoing and incoming calls to explore.
            include_reverse: Whether to include incoming calls (callers) to the nodes in the slice.

        Returns:
            A dictionary containing 'nodes' and 'edges' of the call graph slice, and 'info'.
            Nodes include details from fqn_to_symbol_map if available.
        """
        target_node_id = NodeID(target_fqn_str)
        slice_nodes: Dict[NodeID, Dict[str, Any]] = {}
        slice_edges: List[Dict[str, Any]] = []

        # Use collections.deque for efficient queue operations
        queue = collections.deque([(target_node_id, 0)])
        visited_fqns = {target_node_id}
        info_parts = [f"Call graph slice for {target_fqn_str} (depth {depth}, reverse {include_reverse}):"]

        edge_set = set() # To avoid duplicate edges

        while queue:
            current_node_id, current_d = queue.popleft()

            # Process current_node_id (add to slice_nodes if not already)
            if current_node_id not in slice_nodes:
                details = self.fqn_to_symbol_map.get(current_node_id)
                if details:
                    slice_nodes[current_node_id] = details.copy()
                else:
                    slice_nodes[current_node_id] = {
                        "fqn": str(current_node_id),
                        "item_type": "external_symbol", # Or determine if it's a known lib
                        "file_path": "N/A",
                        "start_line": 0,
                        "external": True
                    }

            if current_d < depth:
                # Explore Outgoing Calls
                if current_node_id in self.project_call_graph:
                    for callee_node_id in self.project_call_graph[current_node_id]:
                        if callee_node_id not in slice_nodes:
                            callee_details = self.fqn_to_symbol_map.get(callee_node_id)
                            if callee_details:
                                slice_nodes[callee_node_id] = callee_details.copy()
                            else:
                                slice_nodes[callee_node_id] = {
                                    "fqn": str(callee_node_id), "item_type": "external_symbol",
                                    "file_path": "N/A", "start_line": 0, "external": True
                                }

                        edge = (str(current_node_id), str(callee_node_id))
                        if edge not in edge_set:
                            slice_edges.append({"from": str(current_node_id), "to": str(callee_node_id), "type": "calls"})
                            edge_set.add(edge)

                        if callee_node_id not in visited_fqns:
                            visited_fqns.add(callee_node_id)
                            queue.append((callee_node_id, current_d + 1))

                # Explore Incoming Calls (if include_reverse)
                if include_reverse:
                    for caller_node_id, callees_set in self.project_call_graph.items():
                        if current_node_id in callees_set: # caller_node_id calls current_node_id
                            if caller_node_id not in slice_nodes:
                                caller_details = self.fqn_to_symbol_map.get(caller_node_id)
                                if caller_details:
                                    slice_nodes[caller_node_id] = caller_details.copy()
                                else:
                                    slice_nodes[caller_node_id] = {
                                        "fqn": str(caller_node_id), "item_type": "external_symbol",
                                        "file_path": "N/A", "start_line": 0, "external": True
                                    }

                            edge = (str(caller_node_id), str(current_node_id))
                            if edge not in edge_set:
                                slice_edges.append({"from": str(caller_node_id), "to": str(current_node_id), "type": "calls"})
                                edge_set.add(edge)

                            if caller_node_id not in visited_fqns:
                                visited_fqns.add(caller_node_id)
                                queue.append((caller_node_id, current_d + 1))

        final_nodes_list = list(slice_nodes.values())
        info_parts.append(f"Found {len(final_nodes_list)} nodes and {len(slice_edges)} edges.")

        return {"nodes": final_nodes_list, "edges": slice_edges, "info": " ".join(info_parts)}

    def get_imported_modules(self, module_fqn_str: str) -> Set[str]:
        """Retrieves a set of FQN strings of modules directly imported by the given module."""
        module_node_id = NodeID(module_fqn_str)
        if module_node_id in self.project_module_imports:
            # Convert NodeID objects in the set to strings for the return type
            return {str(imported_module_nid) for imported_module_nid in self.project_module_imports[module_node_id]}
        return set()

    def get_imported_symbols_from_module(
        self,
        importing_module_fqn_str: str,
        source_module_fqn_str: str
    ) -> Set[str]:
        """Retrieves a set of symbol names imported by 'importing_module' from 'source_module'."""
        importing_module_node_id = NodeID(importing_module_fqn_str)
        source_module_node_id = NodeID(source_module_fqn_str)

        if importing_module_node_id in self.project_symbol_imports:
            if source_module_node_id in self.project_symbol_imports[importing_module_node_id]:
                return self.project_symbol_imports[importing_module_node_id][source_module_node_id].copy() # Return a copy
        return set()

    def get_all_symbol_imports_for_module(self, importing_module_fqn_str: str) -> Dict[str, Set[str]]:
        """
        Retrieves all 'from X import Y' style imports for a given module.
        Returns a dict where keys are source module FQN strings and values are sets of imported symbol names.
        """
        importing_module_node_id = NodeID(importing_module_fqn_str)
        results: Dict[str, Set[str]] = {}

        if importing_module_node_id in self.project_symbol_imports:
            for source_module_node_id, symbol_set in self.project_symbol_imports[importing_module_node_id].items():
                results[str(source_module_node_id)] = symbol_set.copy() # Return a copy of the set
        return results

    def get_base_classes(self, derived_class_fqn_str: str) -> List[str]:
        """
        Retrieves a list of FQN strings of direct base classes for the given derived class.
        The order in the list reflects the MRO as defined in the source.
        """
        derived_class_node_id = NodeID(derived_class_fqn_str)
        if derived_class_node_id in self.project_inheritance_graph:
            # Convert NodeID objects in the list to strings for the return type
            return [str(base_class_nid) for base_class_nid in self.project_inheritance_graph[derived_class_node_id]]
        return []

    def get_derived_classes(self, base_class_fqn_str: str) -> Set[str]:
        """Retrieves a set of FQN strings of classes that directly inherit from the given base class."""
        base_class_node_id_to_find = NodeID(base_class_fqn_str)
        derived_classes_fqns: Set[str] = set()

        for derived_class_nid, base_class_nid_list in self.project_inheritance_graph.items():
            if base_class_node_id_to_find in base_class_nid_list:
                derived_classes_fqns.add(str(derived_class_nid))
        return derived_classes_fqns


if __name__ == '__main__':
    from src.utils.config_loader import load_app_config # Import for __main__

    current_script_dir = Path(__file__).parent
    dummy_repo = current_script_dir / "_temp_dummy_repo_for_digester_"
    if dummy_repo.exists(): # Clean up before creating new files
        import shutil
        shutil.rmtree(dummy_repo)
    dummy_repo.mkdir(exist_ok=True)

    # Create subdir for subdir.file2
    subdir_path = dummy_repo / "subdir"
    subdir_path.mkdir(exist_ok=True)

    # Updated dummy file contents
    file1_content = """
# file1.py
import os
from subdir.file2 import Bar # Import Bar

class ParentA: pass
class ChildA(ParentA, Bar): # ChildA inherits ParentA and Bar
    def foo(self, x: int) -> str:
        return str(x)

def standalone_func():
    local_bar = Bar()
    return os.getcwd()
"""
    (dummy_repo / "file1.py").write_text(file1_content)

    file2_content = """
# subdir/file2.py
class GrandParentB: pass
class Bar(GrandParentB): # Bar inherits GrandParentB
    y: list[str] = []
    def __init__(self):
        self.z = 10

class AnotherInFile2: pass
"""
    (subdir_path / "file2.py").write_text(file2_content)

    (dummy_repo / "tests").mkdir(exist_ok=True)
    (dummy_repo / "tests" / "test_file.py").write_text("assert True")

    print(f"Digesting dummy repo at: {dummy_repo.resolve()}")
    original_sys_path_main = list(sys.path)
    if str(dummy_repo.resolve()) not in sys.path:
        sys.path.insert(0, str(dummy_repo.resolve()))

    # Load or create a mock app_config for the __main__ block
    # Using load_app_config will attempt to load from default locations or use defaults.
    mock_app_config_main = load_app_config()
    # Example: Override verbose for testing __main__
    mock_app_config_main.setdefault("general", {})["verbose"] = True

    digester = RepositoryDigester(str(dummy_repo), app_config=mock_app_config_main)
    digester.digest_repository()

    print("\nDigested file results (summary):")
    for file_path, result in digester.digested_files.items():
        print(f"  {file_path.name}:")
        # print(f"    LibCST Error: {result.libcst_error}") # Too verbose for this demo
        # print(f"    TreeSitter Has Errors: {result.treesitter_has_errors}")
        # print(f"    Type Info: {result.type_info}") # Also verbose
        print(f"    Extracted symbols: {len(result.extracted_symbols)}")


    print("\n--- Knowledge Graph Queries ---")
    # Derive FQNs using the digester's own method for consistency
    file1_module_fqn = RepositoryDigester._get_module_qname_from_path(dummy_repo / "file1.py", dummy_repo)
    file2_module_fqn = RepositoryDigester._get_module_qname_from_path(dummy_repo / "subdir" / "file2.py", dummy_repo)

    # Module Imports
    imported_modules = digester.get_imported_modules(file1_module_fqn)
    print(f"\nModules imported by '{file1_module_fqn}': {imported_modules}")

    symbols_from_file2 = digester.get_imported_symbols_from_module(file1_module_fqn, file2_module_fqn)
    print(f"Symbols imported by '{file1_module_fqn}' from '{file2_module_fqn}': {symbols_from_file2}")

    all_sym_imports = digester.get_all_symbol_imports_for_module(file1_module_fqn)
    print(f"All symbol imports for '{file1_module_fqn}': {all_sym_imports}")

    # Class Inheritance
    child_a_fqn = f"{file1_module_fqn}.ChildA"
    base_classes_of_child_a = digester.get_base_classes(child_a_fqn)
    print(f"\nBase classes of '{child_a_fqn}': {base_classes_of_child_a}")

    # Note: The FQN for Bar as a base class might be 'subdir.file2.Bar' or just 'Bar'
    # depending on how ast.unparse resolves it in the context of file1.py.
    # For demonstration, we'll query for Bar's known FQN.
    bar_fqn = f"{file2_module_fqn}.Bar"
    derived_from_bar = digester.get_derived_classes(bar_fqn)
    print(f"Classes deriving from '{bar_fqn}': {derived_from_bar}")

    grandparent_b_fqn = f"{file2_module_fqn}.GrandParentB"
    derived_from_gpb = digester.get_derived_classes(grandparent_b_fqn)
    print(f"Classes deriving from '{grandparent_b_fqn}': {derived_from_gpb}")

    # Example of a class with no known derived classes in this dummy repo
    parent_a_fqn = f"{file1_module_fqn}.ParentA"
    derived_from_parent_a = digester.get_derived_classes(parent_a_fqn)
    print(f"Classes deriving from '{parent_a_fqn}': {derived_from_parent_a}")


    paths_to_remove_main = []
    if str(dummy_repo.resolve()) in sys.path and str(dummy_repo.resolve()) not in original_sys_path_main :
         paths_to_remove_main.append(str(dummy_repo.resolve()))

    new_sys_path = [p for p in sys.path if p not in paths_to_remove_main] # type: ignore
    sys.path = original_sys_path_main

    import shutil
    shutil.rmtree(dummy_repo)
    print(f"Cleaned up dummy repo: {dummy_repo}")
