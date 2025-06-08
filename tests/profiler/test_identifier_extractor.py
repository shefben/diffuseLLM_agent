import unittest
from src.profiler.identifier_extractor import extract_identifiers_from_source

class TestIdentifierExtractor(unittest.TestCase):

    def test_extract_class_names(self):
        code = "class MyClass: pass\nclass AnotherClass: pass"
        identifiers = extract_identifiers_from_source(code)
        self.assertEqual(identifiers.get("classes"), {"MyClass", "AnotherClass"})

    def test_extract_function_names(self):
        code = "def my_func(): pass\nasync def async_func(): pass"
        identifiers = extract_identifiers_from_source(code)
        self.assertEqual(identifiers.get("functions"), {"my_func", "async_func"})

    def test_extract_parameters(self):
        code = "def func(p1, p2, *args, **kwargs): pass"
        identifiers = extract_identifiers_from_source(code)
        self.assertEqual(identifiers.get("parameters"), {"p1", "p2", "args", "kwargs"})

    def test_extract_variables(self):
        code = """
MY_CONST = 10
g_var = "hello"
class A:
    cls_var = 1
    def m(self):
        self.inst_var = 2
        loc_var = 3
"""
        identifiers = extract_identifiers_from_source(code)
        # Note: 'self' is not typically extracted as a variable in this context.
        # 'inst_var' (the attr part) and 'cls_var' are extracted.
        expected_vars = {"MY_CONST", "g_var", "cls_var", "inst_var", "loc_var"}
        self.assertTrue(expected_vars.issubset(identifiers.get("variables", set())))


    def test_empty_code(self):
        code = ""
        identifiers = extract_identifiers_from_source(code)
        self.assertEqual(identifiers.get("classes"), set())
        self.assertEqual(identifiers.get("functions"), set())
        self.assertEqual(identifiers.get("parameters"), set())
        self.assertEqual(identifiers.get("variables"), set())

    def test_syntax_error(self):
        code = "a = %" # Invalid syntax
        identifiers = extract_identifiers_from_source(code)
        self.assertIn("error", identifiers)
        self.assertTrue(identifiers["error"].startswith("SyntaxError:"))

if __name__ == '__main__':
    unittest.main()
