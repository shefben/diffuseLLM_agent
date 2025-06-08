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
            A list of FQNs associated with the signature. Returns an empty list
            if the signature is not found or if it's a prefix but not a complete signature.
        """
        node = self.root
        for char in signature_str:
            if char not in node.children:
                return [] # Signature not found
            node = node.children[char]

        # Return list from the set for a consistent return type
        return sorted(list(node.function_fqns)) if node.function_fqns else []

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

def generate_function_signature_string(
    func_node: ast.FunctionDef,
    type_resolver: Callable[[ast.AST, str], Optional[str]],
) -> str:
    """
    Generates a canonical signature string for a function or method node.
    Example: "my_function(int,str,Optional[List[int]])->bool"
             "my_method(MyClass,str,**Any)->Any" (param names excluded)

    Args:
        func_node: The ast.FunctionDef or ast.AsyncFunctionDef node.
        type_resolver: A callable that takes an AST node (e.g., ast.arg for a parameter,
                       or the ast.Name/ast.Attribute node of a return annotation) and a
                       category hint (e.g., "parameter", "return_annotation"), and returns
                       its type string or None.
    Returns:
        A canonical string representation of the function's signature.
    """
    param_type_strings: List[str] = []
    args_processed = False # To help decide where to put '*'

    # Positional-only arguments (Python 3.8+)
    if hasattr(func_node.args, 'posonlyargs') and func_node.args.posonlyargs:
        for arg_node in func_node.args.posonlyargs:
            param_type = type_resolver(arg_node, f"{func_node.name}.{arg_node.arg}:parameter_posonly") or "Any"
            param_type_strings.append(param_type)
        param_type_strings.append("/") # Add PEP 570 marker
        args_processed = True

    # Regular positional or keyword arguments
    if func_node.args.args:
        for arg_node in func_node.args.args:
            param_type = type_resolver(arg_node, f"{func_node.name}.{arg_node.arg}:parameter") or "Any"
            param_type_strings.append(param_type)
        args_processed = True

    # Vararg (*args)
    if func_node.args.vararg:
        if args_processed and not param_type_strings[-1] == "/":
            # This case is complex: e.g. def foo(a, /, *args) vs def foo(a, *args)
            # The '*' for varargs itself acts as a separator if there were previous args.
            # If only posonlyargs ending with '/' were present, no explicit '*' needed before vararg name.
            pass # The '*' will be part of the vararg string itself.

        vararg_name = func_node.args.vararg.arg
        vararg_type_node = func_node.args.vararg.annotation
        vararg_type_str = "Any"
        if vararg_type_node:
            vararg_type_str = type_resolver(vararg_type_node, f"{func_node.name}.*{vararg_name}:parameter_vararg_annotation") or "Any"

        param_type_strings.append(f"*{vararg_type_str}") # Store as *Type, not *name:Type
        args_processed = True


    # Keyword-only arguments
    if func_node.args.kwonlyargs:
        # Add '*' separator if not already implicitly there by *args or by / with no regular args
        if not func_node.args.vararg:
            if not param_type_strings or param_type_strings[-1] != "/":
                 param_type_strings.append("*")

        for arg_node in func_node.args.kwonlyargs:
            param_type = type_resolver(arg_node, f"{func_node.name}.{arg_node.arg}:parameter_kwonly") or "Any"
            param_type_strings.append(param_type)
        args_processed = True


    # Kwarg (**kwargs)
    if func_node.args.kwarg:
        kwarg_name = func_node.args.kwarg.arg
        kwarg_type_node = func_node.args.kwarg.annotation
        kwarg_type_str = "Any"
        if kwarg_type_node:
            kwarg_type_str = type_resolver(kwarg_type_node, f"{func_node.name}.**{kwarg_name}:parameter_kwarg_annotation") or "Any"
        param_type_strings.append(f"**{kwarg_type_str}")


    # Return type
    return_type_str = "Any"
    if func_node.returns:
        return_type_str = type_resolver(func_node.returns, f"{func_node.name}:return_annotation") or "Any"

    return f"{func_node.name}({','.join(param_type_strings)})->{return_type_str}"


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

    print(f"Search for '{sig1}': {trie.search(sig1)}")
    # Expected: sorted list like ['module_a.MyClass.my_func', 'module_b.my_func_alias']

    print(f"Search for '{sig2}': {trie.search(sig2)}")
    # Expected: ['module_a.MyClass.my_other_func']

    print(f"Search for 'my_func(int,str)': {trie.search('my_func(int,str)')}") # Prefix, not full sig
    # Expected: [] (because function_fqns set would be empty if it's only a prefix)

    print(f"Search for 'non_existent_sig': {trie.search('non_existent_sig')}")
    # Expected: []

    print(f"Starts with 'my_func(int,s': {trie.starts_with('my_func(int,s')}") # True
    print(f"Starts with 'my_func(int,x': {trie.starts_with('my_func(int,x')}") # False
    print(f"Starts with 'another_': {trie.starts_with('another_')}") # True

    # Test empty signature string
    trie.insert("", "module_root_func_no_sig_name")
    print(f"Search for empty signature '': {trie.search('')}")
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
