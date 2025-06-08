import unittest
from unittest.mock import patch, MagicMock, mock_open
import sqlite3
from pathlib import Path
import libcst as cst
import tempfile # For creating temp DB for some tests
import shutil   # For cleaning up temp_dir if created
from typing import Dict, List, Any, Tuple # Ensure List, Tuple are imported


# Adjust imports based on actual location
from src.transformer.identifier_renamer import (
    to_snake_case,
    to_pascal_case,
    to_camel_case,
    IdentifierRenamingTransformer,
    rename_identifiers_in_code
)
# For setting up mock DB
from src.profiler.database_setup import create_naming_conventions_db
from src.profiler.naming_conventions import NAMING_CONVENTIONS_REGEX

class TestNameConversionHelpers(unittest.TestCase):
    def test_to_snake_case(self):
        self.assertEqual(to_snake_case("MyPascalCase"), "my_pascal_case")
        self.assertEqual(to_snake_case("myCamelCase"), "my_camel_case")
        self.assertEqual(to_snake_case("already_snake"), "already_snake")
        self.assertEqual(to_snake_case("HTTPRequest"), "http_request")
        self.assertEqual(to_snake_case("ValueHTTP"), "value_http")
        self.assertEqual(to_snake_case("AName"), "a_name")
        self.assertEqual(to_snake_case(""), "")
        self.assertEqual(to_snake_case("A"), "a")

    def test_to_pascal_case(self):
        self.assertEqual(to_pascal_case("my_snake_case"), "MySnakeCase")
        self.assertEqual(to_pascal_case("myCamelCase"), "MyCamelCase")
        self.assertEqual(to_pascal_case("AlreadyPascal"), "AlreadyPascal")
        self.assertEqual(to_pascal_case("_private_snake"), "_PrivateSnake")
        self.assertEqual(to_pascal_case(""), "")
        self.assertEqual(to_pascal_case("a"), "A")


    def test_to_camel_case(self):
        self.assertEqual(to_camel_case("MyPascalCase"), "myPascalCase")
        self.assertEqual(to_camel_case("my_snake_case"), "mySnakeCase")
        self.assertEqual(to_camel_case("alreadyCamel"), "alreadyCamel")
        self.assertEqual(to_camel_case("HTTPRequest"), "httpRequest")
        self.assertEqual(to_camel_case(""), "")
        self.assertEqual(to_camel_case("A"), "a")


class TestIdentifierRenamingTransformer(unittest.TestCase):
    def _transform_code(self, code: str, active_rules: Dict[str, Tuple[str, str]]) -> str:
        module_cst = cst.parse_module(code)
        transformer = IdentifierRenamingTransformer(active_rules)
        modified_module = module_cst.visit(transformer)
        return modified_module.code

    def test_rename_function_name_to_snake(self):
        active_rules = {"function": ("SNAKE_CASE", NAMING_CONVENTIONS_REGEX["SNAKE_CASE"])}
        code = "def MyCamelFunction(): pass"
        expected = "def my_camel_function(): pass"
        self.assertEqual(self._transform_code(code, active_rules), expected)

    def test_rename_function_name_to_camel(self):
        active_rules = {"function": ("CAMEL_CASE", NAMING_CONVENTIONS_REGEX["CAMEL_CASE"])}
        code = "def my_snake_function(): pass"
        expected = "def mySnakeFunction(): pass"
        self.assertEqual(self._transform_code(code, active_rules), expected)

    def test_rename_class_name_to_pascal(self):
        active_rules = {"class": ("PASCAL_CASE", NAMING_CONVENTIONS_REGEX["PASCAL_CASE"])}
        code = "class my_snake_class: pass"
        expected = "class MySnakeClass: pass"
        self.assertEqual(self._transform_code(code, active_rules), expected)

    def test_no_rename_if_conforming(self):
        active_rules = {
            "function": ("SNAKE_CASE", NAMING_CONVENTIONS_REGEX["SNAKE_CASE"]),
            "class": ("PASCAL_CASE", NAMING_CONVENTIONS_REGEX["PASCAL_CASE"])
        }
        code = "class MyConformingClass: pass\ndef my_conforming_function(): pass"
        self.assertEqual(self._transform_code(code, active_rules), code)

    def test_no_rename_dunder_methods(self):
        active_rules = {"function": ("SNAKE_CASE", NAMING_CONVENTIONS_REGEX["SNAKE_CASE"])}
        code = "def __my_dunder__(): pass"
        self.assertEqual(self._transform_code(code, active_rules), code)

    def test_leave_name_placeholder_does_not_rename_vars_params(self):
        active_rules = {"variable": ("SNAKE_CASE", NAMING_CONVENTIONS_REGEX["SNAKE_CASE"])}
        code = "def func(MyParam=1):\n  MyVar = MyParam"
        self.assertEqual(self._transform_code(code, active_rules), code)


class TestRenameIdentifiersInCodeOrchestrator(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp(prefix="test_renamer_orchestrator_"))
        self.db_path = self.test_dir / "test_naming.db"

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def _populate_db_rules(self, rules: List[Tuple[str, str, str, bool]]):
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        for id_type, conv_name, regex, active in rules:
            cursor.execute(
                "INSERT OR REPLACE INTO naming_rules (identifier_type, convention_name, regex_pattern, is_active) VALUES (?, ?, ?, ?)",
                (id_type, conv_name, regex, active)
            )
        conn.commit()
        conn.close()

    def test_rename_orchestrator_success(self):
        create_naming_conventions_db(self.db_path)
        active_rules_data = [
            ("function", "SNAKE_CASE", NAMING_CONVENTIONS_REGEX["SNAKE_CASE"], True),
            ("class", "PASCAL_CASE", NAMING_CONVENTIONS_REGEX["PASCAL_CASE"], True)
        ]
        self._populate_db_rules(active_rules_data)

        code = "class oldName: pass\ndef funcName(): pass"
        expected = "class OldName: pass\ndef func_name(): pass"

        modified_code = rename_identifiers_in_code(code, self.db_path)
        self.assertEqual(modified_code, expected)

    def test_rename_orchestrator_db_not_found(self):
        code = "def funcName(): pass"
        non_existent_db_path = self.test_dir / "non_existent.db"
        with patch('builtins.print') as mock_print:
            modified_code = rename_identifiers_in_code(code, non_existent_db_path)
            self.assertEqual(modified_code, code)
            mock_print.assert_any_call(f"Warning: Naming conventions DB not found at {non_existent_db_path}. No renaming will occur.")

    @patch('src.transformer.identifier_renamer.sqlite3.connect')
    def test_rename_orchestrator_sqlite_error(self, mock_sql_connect):
        mock_sql_connect.side_effect = sqlite3.Error("DB connection failed")
        code = "def funcName(): pass"
        with patch.object(Path, 'exists', return_value=True):
            with patch('builtins.print') as mock_print:
                modified_code = rename_identifiers_in_code(code, self.db_path)
                self.assertEqual(modified_code, code)
                mock_print.assert_any_call("SQLite error when loading naming rules: DB connection failed. No renaming will occur.")


    def test_rename_orchestrator_no_active_rules(self):
        create_naming_conventions_db(self.db_path)
        code = "def funcName(): pass"
        with patch('builtins.print') as mock_print:
            modified_code = rename_identifiers_in_code(code, self.db_path)
            self.assertEqual(modified_code, code)
            mock_print.assert_any_call("No active naming rules found in the database. No renaming will occur.")

    @patch('src.transformer.identifier_renamer.cst.parse_module', side_effect=cst.ParserSyntaxError("Test syntax error", MagicMock(), MagicMock()))
    def test_rename_orchestrator_cst_parse_error(self, mock_parse_module):
        create_naming_conventions_db(self.db_path)
        active_rules_data = [("function", "SNAKE_CASE", NAMING_CONVENTIONS_REGEX["SNAKE_CASE"], True)]
        self._populate_db_rules(active_rules_data)

        code = "def funcName( pass"
        with patch('builtins.print') as mock_print:
            modified_code = rename_identifiers_in_code(code, self.db_path)
            self.assertEqual(modified_code, code)
            self.assertTrue(any("LibCST parsing error" in str(arg_call) for arg_call in mock_print.call_args_list))


if __name__ == '__main__':
    unittest.main()
