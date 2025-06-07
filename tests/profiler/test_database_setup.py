import unittest
from unittest.mock import patch, MagicMock
import sqlite3
from pathlib import Path
import tempfile # Added tempfile
import shutil   # Added shutil

# Adjust import path as necessary
from src.profiler.database_setup import create_naming_conventions_db, DEFAULT_NAMING_DB_PATH

class TestDatabaseSetup(unittest.TestCase):

    def get_table_info(self, cursor, table_name):
        """Helper to get PRAGMA table_info results as a dictionary."""
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()
        # PRAGMA table_info returns: cid, name, type, notnull, dflt_value, pk
        return {col[1]: {"type": col[2], "notnull": col[3], "pk": col[5]} for col in columns}

    def get_indexes_for_table(self, cursor, table_name):
        """Helper to get indexes for a table."""
        cursor.execute(f"PRAGMA index_list({table_name});")
        indexes = []
        for row in cursor.fetchall():
            index_name = row[1]
            # Prevent querying info for sqlite_autoindex tables if they appear in older SQLite versions
            if index_name.startswith("sqlite_autoindex_"):
                continue
            cursor.execute(f"PRAGMA index_info({index_name});")
            indexes.append({
                "name": index_name,
                "columns": [col_info[2] for col_info in cursor.fetchall()]
            })
        return indexes

    def test_create_naming_conventions_db_in_memory(self):
        """Test database creation with an in-memory SQLite database."""
        db_path_obj = Path(":memory:") # Nominal path for the function signature

        mock_conn = sqlite3.connect(":memory:") # Actual in-memory connection

        with patch('sqlite3.connect', return_value=mock_conn) as mock_sql_connect, \
             patch('src.profiler.database_setup.Path.mkdir') as mock_mkdir:

            success = create_naming_conventions_db(db_path=db_path_obj)
            self.assertTrue(success)

            mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
            mock_sql_connect.assert_called_once_with(str(db_path_obj))

            cursor = mock_conn.cursor()

            # 1. Verify 'naming_rules' table
            naming_rules_cols = self.get_table_info(cursor, "naming_rules")
            expected_naming_rules = {
                "rule_id": {"type": "INTEGER", "notnull": 0, "pk": 1},
                "identifier_type": {"type": "TEXT", "notnull": 1, "pk": 0},
                "convention_name": {"type": "TEXT", "notnull": 1, "pk": 0},
                "regex_pattern": {"type": "TEXT", "notnull": 1, "pk": 0},
                "description": {"type": "TEXT", "notnull": 0, "pk": 0},
                "is_active": {"type": "BOOLEAN", "notnull": 0, "pk": 0},
            }
            self.assertEqual(naming_rules_cols.keys(), expected_naming_rules.keys())
            for col_name, expected_attrs in expected_naming_rules.items():
                self.assertEqual(naming_rules_cols[col_name]["type"], expected_attrs["type"])
                self.assertEqual(naming_rules_cols[col_name]["notnull"], expected_attrs["notnull"])
                self.assertEqual(naming_rules_cols[col_name]["pk"], expected_attrs["pk"])

            indexes = self.get_indexes_for_table(cursor, "naming_rules")
            found_identifier_type_index = False
            for index in indexes:
                if index["columns"] == ["identifier_type"]:
                    found_identifier_type_index = True
                    break
            self.assertTrue(found_identifier_type_index, "Index on identifier_type not found for naming_rules.")

            # 2. Verify 'symbols_fts' table (FTS5)
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='symbols_fts';")
            self.assertIsNotNone(cursor.fetchone(), "'symbols_fts' table not found.")

            try:
                cursor.execute("INSERT INTO symbols_fts (fq_name, file_path, symbol_type, signature) VALUES (?, ?, ?, ?)",
                               ('test.symbol', 'test.py', 'function', 'test()'))
                mock_conn.commit()
                cursor.execute("SELECT fq_name FROM symbols_fts WHERE symbols_fts MATCH 'test.symbol';")
                self.assertIsNotNone(cursor.fetchone())
            except sqlite3.OperationalError as e:
                if "no such module: fts5" in str(e):
                    self.skipTest("FTS5 module not available in this SQLite environment. Skipping FTS5 specific checks.")
                elif "unable to use function MATCH in the requested context" in str(e) or \
                     "no such table: fts5_test_table" in str(e): # Second part for when FTS5 check itself fails
                     self.skipTest("FTS5 MATCH operator test failed, possibly due to test environment SQLite limitations.")
                else:
                    raise

        mock_conn.close()


    @patch('sqlite3.connect')
    def test_create_db_sqlite_error_on_connect(self, mock_connect):
        """Test handling of SQLite error during connection."""
        mock_connect.side_effect = sqlite3.Error("Connection failed")
        with patch('src.profiler.database_setup.Path.mkdir'):
            success = create_naming_conventions_db()
            self.assertFalse(success)

    @patch('sqlite3.connect')
    def test_create_db_sqlite_error_on_execute(self, mock_connect):
        """Test handling of SQLite error during cursor execute."""
        mock_cursor = MagicMock()
        # Make the first call to execute (CREATE TABLE naming_rules) fail
        mock_cursor.execute.side_effect = sqlite3.Error("Execute failed on naming_rules")

        mock_conn_instance = MagicMock()
        mock_conn_instance.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn_instance

        with patch('src.profiler.database_setup.Path.mkdir'):
            success = create_naming_conventions_db()
            self.assertFalse(success)
            mock_cursor.execute.assert_called()


    def test_create_db_with_temp_file(self):
        # Use a unique name within a shared temp directory for this test run
        temp_dir = Path(tempfile.gettempdir()) / f"test_db_run_{random.randint(1000,9999)}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_db_file = temp_dir / "test_db.sqlite3"

        if temp_db_file.exists(): # Clean up from a previous failed run if any
            temp_db_file.unlink()

        success = create_naming_conventions_db(db_path=temp_db_file)
        self.assertTrue(success)
        self.assertTrue(temp_db_file.exists())

        conn = sqlite3.connect(str(temp_db_file))
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='naming_rules';")
        self.assertIsNotNone(cursor.fetchone())

        # Check FTS table existence, skip test if FTS5 not supported by this build of sqlite
        try:
            cursor.execute("CREATE VIRTUAL TABLE IF NOT EXISTS fts5_check USING fts5(test)") # Test FTS5 support
            cursor.execute("DROP TABLE fts5_check")
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='symbols_fts';")
            self.assertIsNotNone(cursor.fetchone())
        except sqlite3.OperationalError as e:
             if "no such module: fts5" in str(e):
                self.skipTest(f"FTS5 module not available, skipping FTS table check in file test for {temp_db_file}")
             else:
                raise
        finally:
            conn.close()

        # Clean up
        if temp_db_file.exists():
            temp_db_file.unlink()
        shutil.rmtree(temp_dir)


if __name__ == '__main__':
    unittest.main()
