import unittest
from unittest.mock import patch, MagicMock # If we need to mock DB interactions for some specific error tests
import sqlite3
from pathlib import Path
from typing import Dict, List, Any
import tempfile # For temp DB files
import shutil # For cleaning up temp dirs if needed (though NamedTemporaryFile handles individual files)


# Adjust import paths as necessary
from src.profiler.naming_conventions import (
    NAMING_CONVENTIONS_REGEX,
    is_identifier_matching_convention,
    populate_naming_rules_from_profile,
    IDENTIFIER_TYPES
)
from src.profiler.database_setup import create_naming_conventions_db # To set up schema for tests

class TestNamingConventionsRegexes(unittest.TestCase):

    def test_individual_regex_patterns(self):
        """Test each regex pattern with valid and invalid examples."""
        test_cases = {
            "SNAKE_CASE": {
                "valid": ["my_var", "_private", "var1", "a_b_c", "a", "_", "snake_123"],
                "invalid": ["MyVar", "myVAr", "1var", "my var", "", "var_"]
            },
            "PASCAL_CASE": {
                "valid": ["MyClass", "MyVar", "Class01", "A", "PascalCase"],
                "invalid": ["myClass", "_MyClass", "1Class", "My Class", "", "my_Pascal"]
            },
            "CAMEL_CASE": { # Current regex: ^[a-z]+[A-Za-z0-9_]*$
                "valid": ["myVar", "anotherExample", "var01", "aValue", "a", "a_b", "aB_c"],
                "invalid": ["MyVar", "_myVar", "1var", "my Var", ""]
            },
            "UPPER_SNAKE_CASE": {
                "valid": ["MY_CONST", "_PRIVATE_CONST", "CONST01", "A", "_", "API_KEY_007"],
                "invalid": ["myConst", "MY_CONST_lower", "1CONST", "MY CONST", "", "CONST_"]
            },
            "TEST_SNAKE_CASE": {
                "valid": ["test_my_func", "test_another", "test_"],
                "invalid": ["my_test_func", "testMyFunc", "Test_Func", "testmyfunction", "test"]
            },
            "TEST_PASCAL_CASE": {
                "valid": ["TestMyClass", "TestAPI", "Test"],
                "invalid": ["myTestClass", "TestmyClass", "test_Class", "testMyAPI"]
            },
        }

        for convention_name, cases in test_cases.items():
            with self.subTest(convention=convention_name):
                for identifier in cases.get("valid", []):
                    self.assertTrue(
                        is_identifier_matching_convention(identifier, convention_name),
                        f"Expected '{identifier}' to match {convention_name}"
                    )
                for identifier in cases.get("invalid", []):
                    self.assertFalse(
                        is_identifier_matching_convention(identifier, convention_name),
                        f"Expected '{identifier}' NOT to match {convention_name}"
                    )

        # Test unknown convention
        self.assertFalse(is_identifier_matching_convention("some_id", "UNKNOWN_CONVENTION"))


class TestPopulateNamingRules(unittest.TestCase):

    # No setUp or tearDown needed if each test uses NamedTemporaryFile for DB

    def get_active_rule_for_type_from_cursor(self, cursor, identifier_type: str) -> Any:
        cursor.execute("SELECT convention_name, regex_pattern FROM naming_rules WHERE identifier_type = ? AND is_active = TRUE", (identifier_type,))
        return cursor.fetchone()

    def get_all_rules_for_type_from_cursor(self, cursor, identifier_type: str) -> List[Any]:
        cursor.execute("SELECT convention_name, regex_pattern, is_active FROM naming_rules WHERE identifier_type = ?", (identifier_type,))
        return cursor.fetchall()


    def test_populate_predominantly_snake_case(self):
        profile = {
            "identifier_snake_case_pct": 0.80,
            "identifier_camelCase_pct": 0.10,
            "identifier_UPPER_SNAKE_CASE_pct": 0.90 # High for constants
        }

        with tempfile.NamedTemporaryFile(suffix=".db") as tmp_db_file_obj:
            db_file_path = Path(tmp_db_file_obj.name)

            # 1. Create schema in this temp file
            self.assertTrue(create_naming_conventions_db(db_path=db_file_path), "DB schema creation failed")

            # 2. Populate based on profile
            self.assertTrue(populate_naming_rules_from_profile(db_file_path, profile), "Populate rules failed")

            # 3. Verify
            conn = sqlite3.connect(str(db_file_path))
            cursor = conn.cursor()

            const_rule = self.get_active_rule_for_type_from_cursor(cursor, "constant")
            self.assertIsNotNone(const_rule, "No active rule for constant")
            self.assertEqual(const_rule[0], "UPPER_SNAKE_CASE")

            class_rule = self.get_active_rule_for_type_from_cursor(cursor, "class")
            self.assertIsNotNone(class_rule, "No active rule for class")
            self.assertEqual(class_rule[0], "PASCAL_CASE")

            func_rule = self.get_active_rule_for_type_from_cursor(cursor, "function")
            self.assertIsNotNone(func_rule, "No active rule for function")
            self.assertEqual(func_rule[0], "SNAKE_CASE")

            var_rule = self.get_active_rule_for_type_from_cursor(cursor, "variable")
            self.assertIsNotNone(var_rule, "No active rule for variable")
            self.assertEqual(var_rule[0], "SNAKE_CASE")

            test_func_rule = self.get_active_rule_for_type_from_cursor(cursor, "test_function")
            self.assertIsNotNone(test_func_rule, "No active rule for test_function")
            self.assertEqual(test_func_rule[0], "TEST_SNAKE_CASE")

            test_class_rule = self.get_active_rule_for_type_from_cursor(cursor, "test_class")
            self.assertIsNotNone(test_class_rule, "No active rule for test_class")
            self.assertEqual(test_class_rule[0], "TEST_PASCAL_CASE")

            conn.close()


    def test_populate_predominantly_camel_case(self):
        profile = {
            "identifier_snake_case_pct": 0.10,
            "identifier_camelCase_pct": 0.80,
            "identifier_UPPER_SNAKE_CASE_pct": 0.0 # Low screaming case
        }
        with tempfile.NamedTemporaryFile(suffix=".db") as tmp_db_file_obj:
            db_file_path = Path(tmp_db_file_obj.name)
            create_naming_conventions_db(db_path=db_file_path)
            populate_naming_rules_from_profile(db_file_path, profile)

            conn = sqlite3.connect(str(db_file_path))
            cursor = conn.cursor()

            const_rule = self.get_active_rule_for_type_from_cursor(cursor, "constant")
            self.assertIsNotNone(const_rule)
            self.assertEqual(const_rule[0], "UPPER_SNAKE_CASE") # Default active

            func_rule = self.get_active_rule_for_type_from_cursor(cursor, "function")
            self.assertIsNotNone(func_rule)
            self.assertEqual(func_rule[0], "CAMEL_CASE")
            var_rule = self.get_active_rule_for_type_from_cursor(cursor, "variable")
            self.assertIsNotNone(var_rule)
            self.assertEqual(var_rule[0], "CAMEL_CASE")
            conn.close()

    def test_populate_mixed_func_var_styles(self):
        profile = {
            "identifier_snake_case_pct": 0.45,
            "identifier_camelCase_pct": 0.35, # snake > camel, both above 0.05, snake meets 0.40 threshold
            "identifier_UPPER_SNAKE_CASE_pct": 0.9
        }
        with tempfile.NamedTemporaryFile(suffix=".db") as tmp_db_file_obj:
            db_file_path = Path(tmp_db_file_obj.name)
            create_naming_conventions_db(db_path=db_file_path)
            populate_naming_rules_from_profile(db_file_path, profile, default_active_threshold=0.40)

            conn = sqlite3.connect(str(db_file_path))
            cursor = conn.cursor()

            func_rules = self.get_all_rules_for_type_from_cursor(cursor, "function")
            # Expect SNAKE_CASE (active=1) and CAMEL_CASE (active=0)
            self.assertTrue(any(r[0] == "SNAKE_CASE" and r[2] == 1 for r in func_rules), "SNAKE_CASE should be active for function")
            self.assertTrue(any(r[0] == "CAMEL_CASE" and r[2] == 0 for r in func_rules), "CAMEL_CASE should be inactive for function")
            conn.close()

    def test_populate_no_dominant_func_var_style_picks_max_if_some_presence(self):
        profile = {
            "identifier_snake_case_pct": 0.10, # Both low, camel slightly higher
            "identifier_camelCase_pct": 0.15,
            "identifier_UPPER_SNAKE_CASE_pct": 0.9
        }
        with tempfile.NamedTemporaryFile(suffix=".db") as tmp_db_file_obj:
            db_file_path = Path(tmp_db_file_obj.name)
            create_naming_conventions_db(db_path=db_file_path)
            populate_naming_rules_from_profile(db_file_path, profile) # default_active_threshold=0.40

            conn = sqlite3.connect(str(db_file_path))
            cursor = conn.cursor()

            # Logic: if neither hits threshold but there's some usage, pick the max.
            # Here, camel_pct (0.15) > snake_pct (0.10)
            active_func_rule = self.get_active_rule_for_type_from_cursor(cursor, "function")
            self.assertIsNotNone(active_func_rule)
            self.assertEqual(active_func_rule[0], "CAMEL_CASE")

            all_func_rules = self.get_all_rules_for_type_from_cursor(cursor, "function")
            self.assertTrue(any(r[0] == "SNAKE_CASE" and r[2] == 0 for r in all_func_rules), "SNAKE_CASE should be present but inactive")
            conn.close()

    def test_populate_no_func_var_style_at_all(self):
        profile = { # Both zero
            "identifier_snake_case_pct": 0.0,
            "identifier_camelCase_pct": 0.0,
            "identifier_UPPER_SNAKE_CASE_pct": 0.9
        }
        with tempfile.NamedTemporaryFile(suffix=".db") as tmp_db_file_obj:
            db_file_path = Path(tmp_db_file_obj.name)
            create_naming_conventions_db(db_path=db_file_path)
            populate_naming_rules_from_profile(db_file_path, profile)

            conn = sqlite3.connect(str(db_file_path))
            cursor = conn.cursor()

            # Expect defaults added as inactive
            func_rules = self.get_all_rules_for_type_from_cursor(cursor, "function")
            self.assertTrue(any(r[0] == "SNAKE_CASE" and r[2] == 0 for r in func_rules))
            active_func_rule = self.get_active_rule_for_type_from_cursor(cursor, "function")
            self.assertIsNone(active_func_rule, "No function rule should be active if percentages are zero")
            conn.close()


if __name__ == '__main__':
    unittest.main()
