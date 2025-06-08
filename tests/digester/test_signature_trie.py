import unittest
import ast
from typing import Dict, List, Optional, Callable, Any # Added Any

from src.digester.signature_trie import TrieNode, SignatureTrie, generate_function_signature_string

class TestTrieNode(unittest.TestCase):
    def test_node_initialization(self):
        node = TrieNode()
        self.assertEqual(node.children, {})
        self.assertEqual(node.function_fqns, set())

class TestSignatureTrie(unittest.TestCase):
    def setUp(self):
        self.trie = SignatureTrie()

    def test_insert_and_search_simple(self):
        sig1 = "func(int,str)->bool"
        fqn1 = "mod.func"
        self.trie.insert(sig1, fqn1)
        self.assertEqual(self.trie.search(sig1), [fqn1]) # search returns sorted list

    def test_search_non_existent(self):
        self.assertEqual(self.trie.search("nonexistent()->void"), [])

    def test_insert_duplicate_signature_different_fqn(self):
        sig = "dup_sig()->int"
        fqn_a = "mod_a.dup_sig"
        fqn_b = "mod_b.dup_sig"
        self.trie.insert(sig, fqn_a)
        self.trie.insert(sig, fqn_b)
        # Search should return both, sorted
        self.assertEqual(sorted(self.trie.search(sig)), sorted([fqn_a, fqn_b]))

    def test_insert_same_fqn_multiple_times_for_same_signature(self):
        sig = "unique_sig()->None"
        fqn = "mod.unique_sig"
        self.trie.insert(sig, fqn)
        self.trie.insert(sig, fqn) # Insert again
        self.assertEqual(self.trie.search(sig), [fqn]) # Should only have one entry

    def test_search_prefix_is_not_full_signature(self):
        sig_full = "long_func_name(int)->str"
        fqn_full = "mod.long_func_name"
        sig_prefix = "long_func_name(int)"
        self.trie.insert(sig_full, fqn_full)
        self.assertEqual(self.trie.search(sig_prefix), []) # Not a full signature end node

    def test_starts_with(self):
        self.trie.insert("apple()->None", "fruit.apple")
        self.trie.insert("apricot()->None", "fruit.apricot")
        self.trie.insert("banana()->None", "fruit.banana")

        self.assertTrue(self.trie.starts_with("ap"))
        self.assertTrue(self.trie.starts_with("apple"))
        self.assertTrue(self.trie.starts_with("apple()->None"))
        self.assertFalse(self.trie.starts_with("appz"))
        self.assertFalse(self.trie.starts_with("c"))
        self.assertTrue(self.trie.starts_with("")) # Empty prefix should be true

    def test_insert_empty_signature(self):
        self.trie.insert("", "root_func")
        self.assertEqual(self.trie.search(""), ["root_func"])


class TestGenerateFunctionSignatureString(unittest.TestCase):

    def _parse_func_def(self, code: str) -> ast.FunctionDef:
        module = ast.parse(code)
        for node in module.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return node
        raise ValueError("No FunctionDef found in code snippet")

    def test_sig_no_args_no_return_annotation(self):
        code = "def f(): pass"
        func_node = self._parse_func_def(code)
        resolver = lambda node, hint: None # Always returns None
        self.assertEqual(generate_function_signature_string(func_node, resolver), "f()->Any")

    def test_sig_simple_args_and_return(self):
        code = "def add(x: int, y: int) -> int: pass"
        func_node = self._parse_func_def(code)
        def type_resolver(node, hint):
            if isinstance(node, ast.arg) and node.arg == 'x': return 'int'
            if isinstance(node, ast.arg) and node.arg == 'y': return 'int'
            if isinstance(node, ast.Name) and node.id == 'int' and "return" in hint: return 'int' # Simplified
            return "Any"
        self.assertEqual(generate_function_signature_string(func_node, type_resolver), "add(int,int)->int")

    def test_sig_varargs_kwargs_kwonly(self):
        code = "def f(a, *args, b: str, **kwargs) -> None: pass"
        func_node = self._parse_func_def(code)
        def type_resolver(node, hint):
            if isinstance(node, ast.arg):
                if node.arg == 'a': return 'TypeA'
                # For *args, generate_function_signature_string expects type_resolver to be called with the annotation node
                # if node.arg == 'args': return 'Tuple[int,...]'
                if node.arg == 'b': return 'str'
                # if node.arg == 'kwargs': return 'Dict[str,Any]'
            elif isinstance(node, ast.Name) and node.id == 'None' and "return" in hint: return 'None'
            # Simulate resolving annotation for *args and **kwargs
            elif "parameter_vararg_annotation" in hint: return "Tuple[int,...]"
            elif "parameter_kwarg_annotation" in hint: return "Dict[str,Any]"
            return "Any" # Default

        self.assertEqual(
            generate_function_signature_string(func_node, type_resolver),
            "f(TypeA,*Tuple[int,...],*,str,**Dict[str,Any])->None"
        )

    def test_sig_pos_only_args(self): # Python 3.8+
        code = "def f(pos1, pos2, /, std_arg, *, kw_only) -> float: pass"
        func_node = self._parse_func_def(code)
        def type_resolver(node, hint):
            if isinstance(node, ast.arg):
                if node.arg == 'pos1': return 'P1Type'
                if node.arg == 'pos2': return 'P2Type'
                if node.arg == 'std_arg': return 'StdType'
                if node.arg == 'kw_only': return 'KwType'
            if isinstance(node, ast.Name) and node.id == 'float' and "return" in hint: return 'float'
            return "Any"
        self.assertEqual(
            generate_function_signature_string(func_node, type_resolver),
            "f(P1Type,P2Type,/,StdType,*,KwType)->float"
        )

    def test_sig_types_from_resolver_are_none(self):
        code = "def g(p1, p2): pass"
        func_node = self._parse_func_def(code)
        resolver = lambda node, hint: None # All types resolve to None -> "Any"
        self.assertEqual(generate_function_signature_string(func_node, resolver), "g(Any,Any)->Any")

    def test_sig_complex_type_strings_from_resolver(self):
        code = "def h(coll: list) -> dict: pass"
        func_node = self._parse_func_def(code)
        def type_resolver(node, hint):
            if isinstance(node, ast.arg) and node.arg == 'coll': return 'List[Union[int, str]]'
            if isinstance(node, ast.Name) and node.id == 'dict' and "return" in hint: return 'Dict[str, MyClass]'
            return "Any"
        self.assertEqual(
            generate_function_signature_string(func_node, type_resolver),
            "h(List[Union[int, str]])->Dict[str, MyClass]"
        )

if __name__ == '__main__':
    unittest.main()
