# src/transformer/patch_applier.py
import ast
import inspect
import libcst as cst
from libcst.codemod import CodemodContext, VisitorBasedCodemodCommand, MatcherDecoratableCommand
from typing import Optional, Dict, Any, Type, Union # Added Union for base class check
from .exceptions import PatchApplicationError

def apply_libcst_codemod_script(
    original_code: str,
    libcst_script_str: str,
    codemod_args: Optional[Dict[str, Any]] = None
) -> str:
    """
    Applies a LibCST codemod script (provided as a string) to original Python code.

    Args:
        original_code: The original Python code string.
        libcst_script_str: A string containing the Python code for the LibCST codemod.
                           This script must define a class inheriting from
                           VisitorBasedCodemodCommand or MatcherDecoratableCommand.
        codemod_args: Optional dictionary of arguments to pass to the codemod class constructor
                      (excluding the 'context' argument, which is provided automatically).

    Returns:
        The transformed Python code string.

    Raises:
        PatchApplicationError: If any step in finding, executing, or applying the codemod fails.
    """
    if not original_code: # Allow empty original code for new file generation if codemod supports it
        print("PatchApplier Info: Original code is empty. Script should handle module creation if needed.")
    if not libcst_script_str:
        raise PatchApplicationError("LibCST script string cannot be empty.")

    codemod_class_name: Optional[str] = None
    try:
        script_ast = ast.parse(libcst_script_str)

        for node in script_ast.body:
            if isinstance(node, ast.ClassDef):
                is_codemod_command = False
                for base_node in node.bases:
                    if isinstance(base_node, ast.Name) and \
                       base_node.id in ["VisitorBasedCodemodCommand", "MatcherDecoratableCommand"]:
                        is_codemod_command = True
                        break
                    # Handle cases like `cst.VisitorBasedCodemodCommand`
                    if isinstance(base_node, ast.Attribute) and isinstance(base_node.value, ast.Name) and \
                       base_node.value.id == "cst" and base_node.attr in ["VisitorBasedCodemodCommand", "MatcherDecoratableCommand"]:
                       # This check is basic; LibCST itself might not be imported as 'cst' in the script string.
                       # The exec_namespace handles making 'cst' available.
                       is_codemod_command = True
                       break
                if is_codemod_command:
                    codemod_class_name = node.name
                    break

        if codemod_class_name is None:
            raise PatchApplicationError("No class inheriting from VisitorBasedCodemodCommand or MatcherDecoratableCommand found in the script.")

    except SyntaxError as e_syn:
        raise PatchApplicationError(f"Syntax error in LibCST script string: {e_syn}")
    except Exception as e_parse:
        raise PatchApplicationError(f"Error parsing LibCST script string to find codemod class: {e_parse}")

    # Prepare a namespace for exec, including necessary LibCST modules
    exec_namespace: Dict[str, Any] = {
        "cst": cst,
        "m": cst.matchers,
        "CodemodContext": CodemodContext,
        "VisitorBasedCodemodCommand": VisitorBasedCodemodCommand,
        "MatcherDecoratableCommand": MatcherDecoratableCommand,
        # For LibCST internal types that might be used in annotations within the script
        "libcst": cst,
    }

    try:
        compiled_script = compile(libcst_script_str, '<libcst_script>', 'exec')
        exec(compiled_script, exec_namespace)
    except Exception as e_exec:
        raise PatchApplicationError(f"Error executing LibCST script string: {e_exec}")

    codemod_class: Optional[Type[Union[VisitorBasedCodemodCommand, MatcherDecoratableCommand]]] = exec_namespace.get(codemod_class_name) # type: ignore

    if not codemod_class or not inspect.isclass(codemod_class):
        raise PatchApplicationError(f"Could not retrieve Codemod class '{codemod_class_name}' from exec namespace.")

    # Check inheritance again, more reliably after exec
    if not issubclass(codemod_class, (VisitorBasedCodemodCommand, MatcherDecoratableCommand)):
         raise PatchApplicationError(f"Class '{codemod_class_name}' from script is not a valid LibCST CodemodCommand.")

    context = CodemodContext()
    command_instance: Union[VisitorBasedCodemodCommand, MatcherDecoratableCommand]
    try:
        # Instantiate the command
        # This simple instantiation assumes constructor takes (context, **kwargs)
        # More complex constructors might need different handling or inspection.
        if codemod_args:
            command_instance = codemod_class(context=context, **codemod_args)
        else:
            command_instance = codemod_class(context=context)
    except Exception as e_inst:
        raise PatchApplicationError(f"Error instantiating codemod '{codemod_class_name}': {e_inst}. Args: {codemod_args}")

    try:
        input_tree = cst.parse_module(original_code)
    except Exception as e_parse_orig:
        raise PatchApplicationError(f"Error parsing original code into LibCST tree: {e_parse_orig}")

    try:
        transformed_tree = command_instance.transform_module(input_tree)
    except Exception as e_transform:
        # Attempt to provide more context on transform failure if possible
        desc = getattr(command_instance, "DESCRIPTION", "N/A")
        raise PatchApplicationError(f"Error during codemod transformation by '{codemod_class_name}' (Desc: {desc}): {e_transform}")

    return transformed_tree.code


if __name__ == '__main__':
    print("--- LibCST Patch Applier Example ---")

    EXAMPLE_SCRIPT_ADD_COMMENT = """
import libcst as cst
from libcst.codemod import VisitorBasedCodemodCommand, CodemodContext

class AddCommentToFunctionCommand(VisitorBasedCodemodCommand):
    DESCRIPTION: str = "Adds a comment to the first function definition."

    def __init__(self, context: CodemodContext, comment_text: str = "Default comment added by codemod."):
        super().__init__(context)
        self.comment_text = comment_text

    def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.FunctionDef:
        # Add a comment line before the function
        comment = cst.Comment(f"# {self.comment_text}")
        # Create an empty line with the comment
        leading_comment_line = cst.EmptyLine(comment=comment, newline=cst.Newline())

        # Get existing leading lines (if any) and prepend our new comment line
        new_leading_lines = [leading_comment_line] + list(updated_node.leading_lines)

        return updated_node.with_changes(leading_lines=new_leading_lines)
"""
    original_code_sample = """
def foo():
    pass

def bar():
    # An existing comment
    return 42
"""
    print("\n--- Test 1: Add Comment ---")
    print("Original Code:")
    print(original_code_sample)
    try:
        transformed_code = apply_libcst_codemod_script(
            original_code_sample,
            EXAMPLE_SCRIPT_ADD_COMMENT,
            codemod_args={"comment_text": "Applied by __main__ test!"}
        )
        print("\nTransformed Code (Add Comment):")
        print(transformed_code)
    except PatchApplicationError as e:
        print(f"Error applying patch: {e}")

    EXAMPLE_SCRIPT_RENAME_VAR = """
import libcst as cst
import libcst.matchers as m
from libcst.codemod import MatcherDecoratableCommand, CodemodContext

class RenameVariableCommand(MatcherDecoratableCommand):
    DESCRIPTION: str = "Renames a specific variable 'old_name' to 'new_name'."

    def __init__(self, context: CodemodContext, old_name: str, new_name: str):
        super().__init__(context)
        self.old_name = old_name
        self.new_name = new_name

    @m.call_if_inside(m.Name(value=lambda name: name == self.old_name))
    @m.visit(m.Name(value=lambda name: name == self.old_name)) # Also visit to change definition if it's a Name node
    def leave_Name(self, original_node: cst.Name, updated_node: cst.Name) -> cst.Name:
        if original_node.value == self.old_name:
            return updated_node.with_changes(value=self.new_name)
        return updated_node # Should not happen due to matcher
"""
    original_code_rename = """
old_variable = 10
new_value = old_variable + 5
print(old_variable)
"""
    print("\n--- Test 2: Rename Variable ---")
    print("Original Code:")
    print(original_code_rename)
    try:
        transformed_code_rename = apply_libcst_codemod_script(
            original_code_rename,
            EXAMPLE_SCRIPT_RENAME_VAR,
            codemod_args={"old_name": "old_variable", "new_name": "renamed_variable"}
        )
        print("\nTransformed Code (Rename Variable):")
        print(transformed_code_rename)
    except PatchApplicationError as e:
        print(f"Error applying rename patch: {e}")

    print("\n--- Test 3: Script with Syntax Error ---")
    script_with_syntax_error = "class BadCommand(cst.VisitorBasedCodemodCommand): def x: pass"
    try:
        apply_libcst_codemod_script(original_code_sample, script_with_syntax_error)
    except PatchApplicationError as e:
        print(f"Caught expected error: {e}")
        assert "Syntax error" in str(e)

    print("\n--- Test 4: Script without proper Codemod class ---")
    script_no_codemod = "class NotACodemod: pass"
    try:
        apply_libcst_codemod_script(original_code_sample, script_no_codemod)
    except PatchApplicationError as e:
        print(f"Caught expected error: {e}")
        assert "No class inheriting from" in str(e) or "not a valid LibCST CodemodCommand" in str(e)

    print("\n--- Test 5: Codemod class with init error ---")
    script_init_error = """
import libcst as cst
from libcst.codemod import VisitorBasedCodemodCommand, CodemodContext
class InitErrorCommand(VisitorBasedCodemodCommand):
    def __init__(self, context: CodemodContext, non_existent_arg: str): # Requires arg not provided
        super().__init__(context)
"""
    try:
        apply_libcst_codemod_script(original_code_sample, script_init_error, codemod_args={}) # No non_existent_arg
    except PatchApplicationError as e:
        print(f"Caught expected error for init failure: {e}")
        assert "Error instantiating codemod" in str(e)

    print("\n--- LibCST Patch Applier Example Done ---")
