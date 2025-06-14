# src/digester/signature_trie.py
from typing import Dict, List, Set, Optional, Callable, Any # Added Optional, Callable, Any
import ast

class TrieNode:
    def __init__(self):
        self.children: Dict[str, TrieNode] = {}
        # Stores FQNs of functions if this node marks the end of a signature string.
        # Using a set for automatic handling of duplicate FQNs for the same signature (though unlikely for FQNs).
        # More importantly, if we just store a boolean "is_end", for this use case we need the FQNs.
        self.function_fqns: Set[str] = set()

class SignatureTrie:
    def __init__(self):
        """
        Initializes an empty SignatureTrie with a root node.
        """
        self.root = TrieNode()

    def insert(self, signature_str: str, function_fqn: str) -> None:
        """
        Inserts a function signature string and its associated Fully Qualified Name (FQN)
        into the trie.

        Args:
            signature_str: The canonical string representation of the function signature.
            function_fqn: The FQN of the function.
        """
        node = self.root
        for char in signature_str:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.function_fqns.add(function_fqn) # Add FQN to the set at the terminal node

    def search(self, signature_str: str) -> List[str]:
        """
        Searches for an exact signature string in the trie.

        Args:
            signature_str: The canonical signature string to search for.

        Returns:
            List[str]: A list of FQNs associated with the signature. Returns an empty list
            if the signature is not found or if it's a prefix but not a complete signature
            (i.e., the node reached by the signature string does not mark the end of any stored signature).
        """
        node = self.root
        for char in signature_str:
            if char not in node.children:
                return [] # Signature not found
            node = node.children[char]

        # If node.function_fqns is not empty, it means this node marks the end of one or more signatures.
        # Otherwise, it's just a prefix or not a valid end.
        if node.function_fqns:
            return list(node.function_fqns)
        else:
            return []

    def starts_with(self, prefix_str: str) -> bool:
        """
        Checks if any signature in the trie starts with the given prefix.

        Args:
            prefix_str: The prefix string to check.

        Returns:
            True if any signature starts with the prefix, False otherwise.
        """
        node = self.root
        for char in prefix_str:
            if char not in node.children:
                return False # Prefix not found
            node = node.children[char]
        return True # Prefix exists

    def delete(self, signature: str, function_fqn_to_remove: str) -> bool:
        """
        Deletes a specific function_fqn from the set associated with a signature.
        If the set becomes empty after removal, it prunes unnecessary trie nodes.

        Args:
            signature: The canonical signature string to delete.
            function_fqn_to_remove: The specific FQN to remove for that signature.

        Returns:
            True if the FQN was found and removed, False otherwise.
        """
        current_node = self.root
        # Store (parent_node, char_to_child, child_node) for pruning
        path_trace: List[Tuple[TrieNode, str, TrieNode]] = []

        for char_idx, char_code in enumerate(signature):
            if char_code not in current_node.children:
                return False  # Signature not found

            if char_idx > 0: # Add to path_trace starting from the first child of root
                 # parent is current_node *before* moving to child
                 # char_code is the char that leads to child
                 # child is current_node.children[char_code]
                 # This logic seems slightly off, let's adjust path_trace storage:
                 # path_trace should store (node, char_that_led_to_it_from_parent)
                 # No, simpler: path_nodes = [self.root] then append current_node in loop.
                 pass # Will reconstruct path_nodes as in the plan

            current_node = current_node.children[char_code]

        # Rebuild path_nodes as per plan for simpler indexing
        path_nodes = [self.root]
        temp_node = self.root
        for char_code in signature:
            temp_node = temp_node.children[char_code] # Already checked existence
            path_nodes.append(temp_node)
        # current_node is path_nodes[-1]

        if function_fqn_to_remove in current_node.function_fqns:
            current_node.function_fqns.remove(function_fqn_to_remove)

            # Pruning logic:
            # Iterate backwards from the node representing the end of the signature
            # up to (but not including) the root.
            # path_nodes has root at index 0, end_node at index len(signature)
            for i in range(len(signature), 0, -1): # Correctly iterates from end node up to child of root
                child_node = path_nodes[i]
                parent_node = path_nodes[i-1]
                char_leading_to_child = signature[i-1] # Character from parent to child

                if not child_node.function_fqns and not child_node.children:
                    # Node is now a leaf and no longer marks the end of any signature.
                    # Delete the reference from its parent.
                    del parent_node.children[char_leading_to_child]
                else:
                    # Node is still useful (either marks another FQN for this signature,
                    # or is a prefix for other signatures), so stop pruning.
                    break
            return True
        else:
            return False # FQN not found for this signature


def generate_function_signature_string(
    func_node: ast.FunctionDef,
    type_resolver: Callable[[ast.AST, str], Optional[str]],
    class_fqn_prefix_for_method_name: Optional[str] = None,
) -> str:
    """
    Generates a canonical signature string for a function or method node.
    Example: "my_function(int,str,Optional[List[int]])->bool"
             "ClassName.my_method(MyClass,str,**Any)->Any" (param names excluded, method name prefixed with class name)

    Args:
        func_node: The ast.FunctionDef or ast.AsyncFunctionDef node.
        type_resolver: A callable that takes an AST node (e.g., ast.arg for a parameter,
                       or the ast.Name/ast.Attribute node of a return annotation) and a
                       category hint (e.g., "parameter", "return_annotation"), and returns
                       its type string or None.
        class_fqn_prefix_for_method_name: Optional string. If provided (e.g., "ClassName"),
                                          it's prepended to the function name for methods.
                                          This prefix should be the simple class name, not its full FQN.
    Returns:
        A canonical string representation of the function's signature.
    """
    param_type_strings: List[str] = []
    args_processed = False # To help decide where to put '*'

    # Determine the effective function name (e.g., "ClassName.method" or "function_name")
    effective_func_name = func_node.name
    if class_fqn_prefix_for_method_name:
        effective_func_name = f"{class_fqn_prefix_for_method_name}.{func_node.name}"

    # Positional-only arguments (Python 3.8+)
    if hasattr(func_node.args, 'posonlyargs') and func_node.args.posonlyargs:
        for arg_node in func_node.args.posonlyargs:
            param_type = type_resolver(arg_node, f"{effective_func_name}.{arg_node.arg}:parameter_posonly") or "Any" # Use effective_func_name
            param_type_strings.append(param_type)
        param_type_strings.append("/") # Add PEP 570 marker
        args_processed = True

    # Regular positional or keyword arguments
    if func_node.args.args:
        for arg_node in func_node.args.args:
            param_type = type_resolver(arg_node, f"{effective_func_name}.{arg_node.arg}:parameter") or "Any" # Use effective_func_name
            param_type_strings.append(param_type)
        args_processed = True

    # Vararg (*args)
    if func_node.args.vararg:
        if args_processed and not param_type_strings[-1] == "/":
            pass # The '*' will be part of the vararg string itself.

        vararg_name = func_node.args.vararg.arg
        vararg_type_node = func_node.args.vararg.annotation
        vararg_type_str = "Any"
        if vararg_type_node:
            vararg_type_str = type_resolver(vararg_type_node, f"{effective_func_name}.*{vararg_name}:parameter_vararg_annotation") or "Any" # Use effective_func_name

        param_type_strings.append(f"*{vararg_type_str}")
        args_processed = True


    # Keyword-only arguments
    if func_node.args.kwonlyargs:
        if not func_node.args.vararg:
            if not param_type_strings or param_type_strings[-1] != "/":
                 param_type_strings.append("*")

        for arg_node in func_node.args.kwonlyargs:
            param_type = type_resolver(arg_node, f"{effective_func_name}.{arg_node.arg}:parameter_kwonly") or "Any" # Use effective_func_name
            param_type_strings.append(param_type)
        args_processed = True


    # Kwarg (**kwargs)
    if func_node.args.kwarg:
        kwarg_name = func_node.args.kwarg.arg
        kwarg_type_node = func_node.args.kwarg.annotation
        kwarg_type_str = "Any"
        if kwarg_type_node:
            kwarg_type_str = type_resolver(kwarg_type_node, f"{effective_func_name}.**{kwarg_name}:parameter_kwarg_annotation") or "Any" # Use effective_func_name
        param_type_strings.append(f"**{kwarg_type_str}")


    # Return type
    return_type_str = "Any"
    if func_node.returns:
        return_type_str = type_resolver(func_node.returns, f"{effective_func_name}:return_annotation") or "Any" # Use effective_func_name

    return f"{effective_func_name}({','.join(param_type_strings)})->{return_type_str}" # Use effective_func_name


if __name__ == '__main__':
    trie = SignatureTrie()

    sig1 = "my_func(int,str)->bool"
    fqn1_a = "module_a.MyClass.my_func"
    fqn1_b = "module_b.my_func_alias" # Different FQN, same signature

    sig2 = "my_func(int,str)->str"
    fqn2 = "module_a.MyClass.my_other_func"

    sig3 = "another_func(List[int])->None"
    fqn3 = "module_c.another_func"

    trie.insert(sig1, fqn1_a)
    trie.insert(sig1, fqn1_b) # Insert another FQN for the same signature
    trie.insert(sig1, fqn1_a) # Inserting same FQN again should have no effect due to set

    trie.insert(sig2, fqn2)
    trie.insert(sig3, fqn3)

    search_sig1_result = trie.search(sig1)
    print(f"Search for '{sig1}': {sorted(search_sig1_result)}") # Sort for consistent test output
    assert sorted(search_sig1_result) == sorted([fqn1_a, fqn1_b])

    search_sig2_result = trie.search(sig2)
    print(f"Search for '{sig2}': {search_sig2_result}")
    assert search_sig2_result == [fqn2]

    search_prefix_result = trie.search('my_func(int,str)') # Prefix, not full sig
    print(f"Search for 'my_func(int,str)': {search_prefix_result}")
    assert search_prefix_result == []

    search_non_existent_result = trie.search('non_existent_sig')
    print(f"Search for 'non_existent_sig': {search_non_existent_result}")
    assert search_non_existent_result == []

    print(f"Starts with 'my_func(int,s': {trie.starts_with('my_func(int,s')}") # True
    print(f"Starts with 'my_func(int,x': {trie.starts_with('my_func(int,x')}") # False
    print(f"Starts with 'another_': {trie.starts_with('another_')}") # True

    # Test empty signature string
    trie.insert("", "module_root_func_no_sig_name")
    search_empty_sig_result = trie.search('')
    print(f"Search for empty signature '': {search_empty_sig_result}")
    assert search_empty_sig_result == ["module_root_func_no_sig_name"]
    print(f"Starts with empty signature '': {trie.starts_with('')}") # True (root node exists)

    # Test search on a prefix node that is not an end of a word
    # 'my_func(int,str)' is a prefix of sig1 and sig2
    # The search logic correctly returns [] if node.function_fqns is empty.

    # Check content of node for sig1
    node = trie.root
    for char_s1 in sig1: node = node.children[char_s1]
    print(f"FQNs at node for '{sig1}': {node.function_fqns}")

    node_prefix = trie.root
    for char_prefix in "my_func(int,str)": node_prefix = node_prefix.children[char_prefix]
    print(f"FQNs at node for prefix 'my_func(int,str)': {node_prefix.function_fqns}")


    print("\n--- Testing Signature Generation ---")

    def mock_type_resolver_example(node, category_hint: str) -> Optional[str]:
        # print(f"Mock resolver for: {category_hint}, node type: {type(node)}")
        if isinstance(node, ast.arg): # Parameter node
            if node.arg == 'a': return 'int'
            if node.arg == 'b': return 'str'
            if node.arg == 'x': return 'XType'
            if node.arg == 'myargs': return 'Tuple[str,...]'
            if node.arg == 'y': return 'float'
            if node.arg == 'mykwargs': return 'Dict[str,int]'
            if node.arg == 'pos_only_arg': return 'PosOnlyInt'
            if node.arg == 'kw_only_arg': return 'KwOnlyBool'
        elif isinstance(node, ast.Name) and category_hint.endswith("return_annotation"): # Return annotation node
            if node.id == 'bool': return 'bool'
            if node.id == 'None': return 'None' # ast.Name(id='None') for Python < 3.8 for 'None' type hint
        elif isinstance(node, ast.Constant) and category_hint.endswith("return_annotation"): # Py3.8+ for None as ast.Constant
             if node.value is None: return 'None'

        # Fallback for annotations if type_resolver is called on the annotation node itself
        if category_hint.endswith("parameter_vararg_annotation"): return "List[Any]"
        if category_hint.endswith("parameter_kwarg_annotation"): return "Dict[str, Any]"

        return 'ResolvedAny' # Default for this mock

    test_cases_sig_gen = [
        ("def foo(a: int, b: str = 'hello') -> bool: pass", "foo(int,str)->bool"),
        ("def bar(x, *myargs, y:float, **mykwargs): pass", "bar(XType,*List[Any],*,float,**Dict[str, Any])->ResolvedAny"), # Updated expected based on resolver logic
        ("def baz(): pass", "baz()->ResolvedAny"),
        ("def with_return() -> None: pass", "with_return()->None"),
        ("def pos_only_func(pos_only_arg, /): pass", "pos_only_func(PosOnlyInt,/)->ResolvedAny"),
        ("def kw_only_func(*, kw_only_arg): pass", "kw_only_func(*,KwOnlyBool)->ResolvedAny"),
        ("def complex_func(pos_only_arg, /, regular_arg, *varargs, kw_only_arg1, **kwargs): pass",
         "complex_func(ResolvedAny,/,ResolvedAny,*List[Any],*,ResolvedAny,**Dict[str, Any])->ResolvedAny")
    ]

    for i, (code_str, expected_sig) in enumerate(test_cases_sig_gen):
        print(f"\nTest Case {i+1}: {code_str}")
        parsed_module = ast.parse(code_str)
        func_def_node = None
        for item in parsed_module.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_def_node = item
                break

        if func_def_node:
            # A more targeted mock resolver could be made per test case if needed
            sig_str = generate_function_signature_string(func_def_node, mock_type_resolver_example)
            print(f"  Generated: {sig_str}")
            print(f"  Expected:  {expected_sig}")
            assert sig_str == expected_sig, f"Mismatch for case {i+1}"
        else:
            print(f"  Could not find FunctionDef node in: {code_str}")

    print("\nAll signature generation examples processed.")
