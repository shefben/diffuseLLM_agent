from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple # Added Tuple
import libcst as cst
import ast # Keep ast import
from dataclasses import dataclass, field
import sys
import os
import re # For CallGraphVisitor._resolve_callee_fqn
from collections import defaultdict

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
# LibCST metadata providers for graph building
from libcst.metadata import ParentNodeProvider, PositionProvider, QualifiedNameProvider, ScopeProvider


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

class SymbolAndSignatureExtractorVisitor(ast.NodeVisitor): # Renamed
    def __init__(self, module_qname: str, file_path_str: str,
                 source_code_lines: List[str],
                 type_info_map_for_resolver: Optional[Dict[str, str]],
                 signature_trie_ref: SignatureTrie):
        self.module_qname: str = module_qname
        self.file_path_str: str = file_path_str
        self.source_code_lines: List[str] = source_code_lines
        self.symbols_for_embedding: List[Dict[str, Any]] = [] # Renamed from self.symbols
        self.current_class_name: Optional[str] = None # FQN of current class context
        self.type_info_map_for_resolver = type_info_map_for_resolver if type_info_map_for_resolver else {}
        self.signature_trie_ref = signature_trie_ref

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
        return resolved_type


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
        self.symbols_for_embedding.append({
            "fqn": fqn, "item_type": f"{item_type_prefix}_code", "content": code_content,
            "file_path": self.file_path_str, "start_line": node.lineno,
            "end_line": node.end_lineno if hasattr(node, 'end_lineno') and node.end_lineno is not None else node.lineno
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
            })

        signature_string = generate_function_signature_string(node, self._local_type_resolver)
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
        # Simplified: using unparse for now. Will be refined with QualifiedNameProvider & type_info.
        try:
            # Attempt to create a string representation of the function/method being called
            # This is highly heuristic and needs proper symbol resolution.
            if isinstance(func_expr, cst.Name):
                # Could be local func, class constructor, or imported func/class
                # Check type_info for this name at this location for a more qualified name
                # For now: assume it's in the current module or a globally known name
                # This is a simplified key for lookup, actual key format from PyanalyzeTypeExtractionVisitor is different.
                # This part needs careful alignment with how types are stored by Pyanalyze visitor.
                # For now, assume callee_name is either module-local or already qualified if imported.
                callee_name = func_expr.value
                # If type_info suggests it's a class instantiation, point to __init__
                # This requires a more structured type_info than just strings.
                # For now, if it looks like a class name (PascalCase), assume constructor.
                if re.match(r"^[A-Z]", callee_name): # Heuristic for class
                    return NodeID(f"{self.module_qname}.{callee_name}.__init__")
                return NodeID(f"{self.module_qname}.{callee_name}")
            elif isinstance(func_expr, cst.Attribute): # obj.method()
                # This is where type_info for 'obj' (func_expr.value) would be crucial.
                # For now, try to unparse it.
                obj_str = cst.Module([func_expr.value]).code.strip()
                method_name = func_expr.attr.value
                if obj_str == "self":
                    if len(self.current_fqn_stack) > 1 and self.current_fqn_stack[-1] != self.module_qname:
                        class_fqn_from_stack_parts = [p for p in self.current_fqn_stack if p != self.module_qname]
                        if len(class_fqn_from_stack_parts) > 1: # module.class.method -> module.class
                             class_context = ".".join(self.current_fqn_stack[:-1])
                             return NodeID(f"{class_context}.{method_name}")
                # Fallback: try to use unparsed object string. This is very approximate.
                return NodeID(f"{self.module_qname}.{obj_str}.{method_name}") # Assuming obj_str is a class in same module
        except Exception:
            return None # Could not resolve
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
    def __init__(self,
                 repo_path: Union[str, Path],
                 embedding_model_name: str = 'all-MiniLM-L6-v2',
                ):
        self.repo_path = Path(repo_path).resolve()
        if not self.repo_path.is_dir():
            raise ValueError(f"Repository path {self.repo_path} is not a valid directory.")

        self._all_py_files: List[Path] = []
        self.digested_files: Dict[Path, ParsedFileResult] = {}

        self.project_call_graph: CallGraph = {}
        self.project_control_dependence_graph: ControlDependenceGraph = {}
        self.project_data_dependence_graph: DataDependenceGraph = defaultdict(set)
        self.signature_trie = SignatureTrie() # NEW: Initialize SignatureTrie

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

        if SentenceTransformer and embedding_model_name:
            model_load_path: Optional[Union[str, Path]] = None
            default_local_st_model_path = Path("./models/sentence_transformer_model/")

            # 1. Check if embedding_model_name is a direct path to an existing model directory
            potential_path = Path(embedding_model_name)
            if potential_path.is_dir():
                model_load_path = potential_path
                print(f"RepositoryDigester Info: Attempting to load SentenceTransformer model from provided path: {model_load_path}")
            # 2. Else, if it's a known HF name (e.g., 'all-MiniLM-L6-v2') AND default local path exists, use local
            elif embedding_model_name == 'all-MiniLM-L6-v2' and default_local_st_model_path.is_dir():
                model_load_path = default_local_st_model_path
                print(f"RepositoryDigester Info: Found default local SentenceTransformer model at: {model_load_path}. Prioritizing this.")
            # 3. Else, treat embedding_model_name as a Hugging Face model name (for download or cache)
            else:
                model_load_path = embedding_model_name
                if is_placeholder_path := embedding_model_name.endswith("sentence_transformer_model/"): # Check if it's the default placeholder
                    print(f"RepositoryDigester Warning: Provided model path '{embedding_model_name}' seems to be a placeholder or does not exist. Will attempt to load from Hugging Face if it's a valid model name.")
                else:
                    print(f"RepositoryDigester Info: Attempting to load SentenceTransformer model from Hugging Face Hub: '{embedding_model_name}'")

            if model_load_path:
                try:
                    self.embedding_model = SentenceTransformer(str(model_load_path))
                    if self.embedding_model:
                        self.embedding_dimension = self.embedding_model.get_sentence_embedding_dimension()
                    print(f"RepositoryDigester Info: SentenceTransformer model '{model_load_path}' loaded successfully. Dimension: {self.embedding_dimension}.")
                except Exception as e:
                    print(f"RepositoryDigester Error: Failed to load SentenceTransformer model from '{model_load_path}': {e}")
                    self.embedding_model = None
            else: # Should not happen if logic above is correct
                 print("RepositoryDigester Warning: No valid model path or name determined for SentenceTransformer.")

        elif not SentenceTransformer:
            print("RepositoryDigester Warning: sentence-transformers library not available. Embedding generation disabled.")
        elif not embedding_model_name:
            print("RepositoryDigester Warning: No embedding_model_name provided. Embedding generation disabled.")


        # FAISS Index Initialization (depends on successful model loading)
        self.faiss_index: Optional[faiss.Index] = None
        self.faiss_id_to_metadata: List[Dict[str, Any]] = []
        if faiss and self.embedding_model and self.embedding_dimension:
            try:
                print(f"RepositoryDigester Info: Initializing FAISS IndexFlatL2 with dimension {self.embedding_dimension}...")
                self.faiss_index = faiss.IndexFlatL2(self.embedding_dimension) # type: ignore
                print("RepositoryDigester Info: FAISS Index initialized.")
            except Exception as e:
                print(f"RepositoryDigester Error: Error initializing FAISS index: {e}")
                self.faiss_index = None
        elif not faiss:
            print("RepositoryDigester Warning: faiss library not available. FAISS indexing disabled.")
        elif not self.embedding_model or not self.embedding_dimension:
            print("RepositoryDigester Warning: Embedding model not loaded or dimension unknown. FAISS indexing disabled.")

    def _extract_symbols_and_docstrings_from_ast(
        self,
        py_ast_module: ast.Module,
        module_qname: str,
        file_path: Path,
        source_code_lines: List[str],
        type_info: Optional[Dict[str, str]]
    ) -> List[Dict[str, Any]]:

        visitor = SymbolAndSignatureExtractorVisitor( # Renamed and passing new args
            module_qname, str(file_path), source_code_lines,
            type_info,
            self.signature_trie
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
        if self.faiss_index:
            print(f"  Total items in FAISS index: {self.faiss_index.ntotal}")

    # --- New Incremental Update Interface and Internal Methods ---

    def _rebuild_full_faiss_index(self) -> None:
        """Clears and rebuilds the FAISS index from all current digested_files embeddings."""
        if not faiss or not self.embedding_model or not self.embedding_dimension or not np:
            print("RepositoryDigester: FAISS or dependencies not available, skipping FAISS rebuild.")
            self.faiss_index = None
            self.faiss_id_to_metadata = []
            return

        print("RepositoryDigester: Rebuilding full FAISS index...")
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
        elif not all_embeddings_list:
            print("RepositoryDigester: No embeddings found to rebuild FAISS index.")


    def _clear_data_for_file(self, file_path: Path) -> None:
        file_id_str_to_match = str(file_path.relative_to(self.repo_path)) if file_path.is_absolute() and self.repo_path.is_absolute() and self.repo_path in file_path.parents else file_path.name
        print(f"RepositoryDigester: Clearing data for file: {file_id_str_to_match}")

        old_parsed_result = self.digested_files.get(file_path)
        if old_parsed_result and old_parsed_result.extracted_symbols and hasattr(self, 'signature_trie'):
            # This part remains complex: requires knowing old signatures to delete from trie.
            print(f"  SignatureTrie: Placeholder - Robust signature removal for {file_id_str_to_match} is complex and deferred.")
            # Example logic:
            # for symbol in old_parsed_result.extracted_symbols:
            #     if symbol_is_function_like(symbol) and "signature_str" in symbol: # Assuming signature was stored
            #         self.signature_trie.delete(symbol["signature_str"], symbol["fqn"])


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
            print(f"  Removed {file_id_str_to_match} from digested_files.")

        print(f"RepositoryDigester: Data clearing for {file_id_str_to_match} completed (FAISS will be rebuilt after all changes).")


    def _update_or_add_file(self, file_path: Path) -> None:
        print(f"RepositoryDigester: Updating/Adding file: {file_path.name}")
        self._clear_data_for_file(file_path)

        parsed_result = self.parse_file(file_path)
        self.digested_files[file_path] = parsed_result

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
        # FAISS index will be rebuilt after all file operations in a batch (e.g., by handle_file_event or a higher-level coordinator).
        # For a single update, we trigger it here.
        self._rebuild_full_faiss_index()

        print(f"RepositoryDigester: Finished updating/adding {file_path.name}")


    def _remove_file_data(self, file_path: Path) -> None:
        print(f"RepositoryDigester: Removing data for deleted file: {file_path.name}")
        self._clear_data_for_file(file_path)

        if file_path in self._all_py_files:
            try: self._all_py_files.remove(file_path)
            except ValueError: pass

        self._rebuild_full_faiss_index() # Rebuild FAISS after removal
        print(f"RepositoryDigester: Finished removing data for {file_path.name}")


    def handle_file_event(self, event_type: str, src_path: Path, dest_path: Optional[Path] = None) -> None:
        print(f"RepositoryDigester: Received event: {event_type} on {src_path}" + (f" -> {dest_path}" if dest_path else ""))

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
    def get_code_snippets_for_phase(self, phase_ctx: Any) -> Dict[str, str]:
        """
        Mock: Retrieves code snippets relevant to the phase.
        For now, returns the content of the target file if available, else a mock snippet.
        """
        # phase_ctx is 'Phase' from src.planner.phase_model, but using Any to avoid direct import if problematic
        print(f"RepositoryDigester.get_code_snippets_for_phase: Mock for phase: {getattr(phase_ctx, 'operation_name', 'UnknownOp')}")
        target_file_path_str = getattr(phase_ctx, 'target_file', None)
        if target_file_path_str:
            target_file_path = Path(target_file_path_str)
            # Ensure we use the same path format as in self.digested_files (absolute or relative to repo_path)
            # Assuming self.digested_files uses absolute paths or paths relative to self.repo_path
            # For simplicity, let's try to match based on the name or relative path if target_file_path_str is not absolute.

            # Attempt to find the file in digested_files
            # This logic might need to be more robust depending on how target_file_path is stored/passed
            found_parsed_result = None
            for abs_path_key, parsed_file_res in self.digested_files.items():
                if str(abs_path_key).endswith(target_file_path_str): # Simple endswith match
                    found_parsed_result = parsed_file_res
                    break

            if found_parsed_result and found_parsed_result.source_code:
                print(f"  Returning source code for: {target_file_path_str}")
                return {str(target_file_path_str): found_parsed_result.source_code}
            else:
                print(f"  Target file {target_file_path_str} not found in digested files or no source. Returning mock snippet.")
                return {str(target_file_path_str) if target_file_path_str else "mock_target.py": "def mock_function_from_get_code_snippets():\n    pass # Target not found"}
        else:
            print("  No target file in phase_ctx. Returning generic mock snippet.")
            return {"mock_file.py": "def foo():\n    pass # No target file specified"}

    def get_pdg_slice_for_phase(self, phase_ctx: Any) -> Dict[str, Any]:
        """
        Mock: Retrieves a program dependence graph (PDG) slice relevant to the phase.
        """
        # phase_ctx is 'Phase' from src.planner.phase_model
        print(f"RepositoryDigester.get_pdg_slice_for_phase: Mock for phase: {getattr(phase_ctx, 'operation_name', 'UnknownOp')}")
        # In a real implementation, this would query self.project_control_dependence_graph and self.project_data_dependence_graph
        # based on the phase_ctx (e.g., target file, specific functions/lines from phase_ctx.parameters)
        return {
            "nodes": [
                {"id": "node1", "label": "var x = 1", "file": getattr(phase_ctx, 'target_file', 'unknown.py')},
                {"id": "node2", "label": "print(x)", "file": getattr(phase_ctx, 'target_file', 'unknown.py')}
            ],
            "edges": [
                {"from": "node1", "to": "node2", "type": "data_dependency"}
            ],
            "info": "Mock PDG slice relevant to the phase context."
        }

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


if __name__ == '__main__':
    current_script_dir = Path(__file__).parent
    dummy_repo = current_script_dir / "_temp_dummy_repo_for_digester_"
    dummy_repo.mkdir(exist_ok=True)
    (dummy_repo / "file1.py").write_text("def foo(x: int) -> str:\n    return str(x)\n")
    (dummy_repo / "subdir").mkdir(exist_ok=True)
    (dummy_repo / "subdir" / "file2.py").write_text("class Bar:\n    y: list[str] = []\n")
    (dummy_repo / "tests").mkdir(exist_ok=True)
    (dummy_repo / "tests" / "test_file.py").write_text("assert True")

    print(f"Digesting dummy repo at: {dummy_repo.resolve()}")
    original_sys_path_main = list(sys.path)
    if str(dummy_repo.resolve()) not in sys.path:
        sys.path.insert(0, str(dummy_repo.resolve()))

    digester = RepositoryDigester(str(dummy_repo))
    digester.digest_repository()

    print("\nDigested file results (summary):")
    for file_path, result in digester.digested_files.items():
        print(f"  {file_path.name}:")
        print(f"    LibCST Error: {result.libcst_error}")
        print(f"    TreeSitter Has Errors: {result.treesitter_has_errors}")
        print(f"    Type Info: {result.type_info}")

    paths_to_remove_main = []
    if str(dummy_repo.resolve()) in sys.path and str(dummy_repo.resolve()) not in original_sys_path_main :
         paths_to_remove_main.append(str(dummy_repo.resolve()))

    new_sys_path = [p for p in sys.path if p not in paths_to_remove_main] # type: ignore
    sys.path = original_sys_path_main

    import shutil
    shutil.rmtree(dummy_repo)
    print(f"Cleaned up dummy repo: {dummy_repo}")
