import unittest
from src.profiler.docstring_extractor import extract_docstrings_from_source

class TestDocstringExtractor(unittest.TestCase):

    def test_extract_module_docstring(self):
        code = """\"Module docstring.\"""
        docstrings = extract_docstrings_from_source(code)
        self.assertEqual(docstrings.get("module"), "Module docstring.")

    def test_extract_function_docstring(self):
        code = """
def my_func():
    \"Function docstring.\"
    pass
"""
        docstrings = extract_docstrings_from_source(code)
        self.assertEqual(docstrings.get("my_func"), "Function docstring.")

    def test_extract_class_and_method_docstrings(self):
        code = """
class MyClass:
    \"Class docstring.\"
    def my_method(self):
        \"Method docstring.\"
        pass
"""
        docstrings = extract_docstrings_from_source(code)
        self.assertEqual(docstrings.get("MyClass"), "Class docstring.")
        self.assertEqual(docstrings.get("MyClass.my_method"), "Method docstring.")

    def test_no_docstrings(self):
        code = """
class NoDocs:
    pass

def func_no_docs():
    pass
"""
        docstrings = extract_docstrings_from_source(code)
        self.assertIsNone(docstrings.get("module"))
        self.assertIsNone(docstrings.get("NoDocs"))
        self.assertIsNone(docstrings.get("func_no_docs"))
        # Method without docstring might not appear or be None, depending on implementation details
        # For 'MyClass.my_method' where my_method has no docstring, it should be None.
        # Let's add a case for that.
        code_method_no_doc = """
class MyClass:
    \"Class docstring.\"
    def method_no_doc(self):
        pass
"""
        docstrings_mnd = extract_docstrings_from_source(code_method_no_doc)
        self.assertIsNone(docstrings_mnd.get("MyClass.method_no_doc"))


    def test_syntax_error(self):
        code = "def func( :"
        docstrings = extract_docstrings_from_source(code)
        self.assertIn("error", docstrings)
        self.assertTrue(docstrings["error"].startswith("SyntaxError:"))

if __name__ == '__main__':
    unittest.main()
