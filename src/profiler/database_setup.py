import sqlite3
from pathlib import Path

DEFAULT_NAMING_DB_PATH = Path("config") / "naming_conventions.db"


def create_naming_conventions_db(db_path: Path = DEFAULT_NAMING_DB_PATH) -> bool:
    """
    Creates the naming_conventions.db SQLite database with the required schema.
    If the database already exists, it will not recreate the tables unless they are missing.

    The schema includes:
    - naming_rules: Stores regex patterns for different identifier types.
    - symbols_fts: An FTS5 table for fast search of fully-qualified symbol names.

    Args:
        db_path: The path where the SQLite database file will be created.
                 Defaults to 'config/naming_conventions.db'.

    Returns:
        True if the database and tables were successfully created or verified,
        False otherwise.
    """
    try:
        # Ensure the directory for the database exists
        db_path.parent.mkdir(parents=True, exist_ok=True)

        # Connect to (or create) the database
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # 1. Create 'naming_rules' table
        # Stores regex patterns for different identifier types (e.g., function, class)
        # identifier_type: 'function', 'class', 'constant', 'test_function', 'test_class', etc.
        # convention_name: e.g., 'snake_case', 'PascalCase', 'UPPER_SNAKE_CASE'
        # regex_pattern: The regex to validate the naming convention.
        # description: Optional description of the rule.
        # is_active: Boolean, to enable/disable rules.
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS naming_rules (
                rule_id INTEGER PRIMARY KEY AUTOINCREMENT,
                identifier_type TEXT NOT NULL,
                convention_name TEXT NOT NULL,
                regex_pattern TEXT NOT NULL,
                description TEXT,
                is_active BOOLEAN DEFAULT TRUE,
                UNIQUE (identifier_type, convention_name)
            )
        """
        )
        print("Table 'naming_rules' created or already exists.")

        # 2. Create 'symbols_fts' table (FTS5 virtual table)
        # For storing and fast searching of fully-qualified symbol names.
        # fq_name: Fully-qualified name of the symbol (e.g., 'module.submodule.ClassName.method_name')
        # file_path: Path to the file where the symbol is defined.
        # symbol_type: 'function', 'class', 'method', 'variable', 'constant'
        # This table is for quick lookups to prevent duplication, as per user spec.
        # Note: FTS5 is generally available in SQLite versions shipped with recent Python.
        # If FTS5 is not available, FTS3 or FTS4 could be alternatives with different syntax.
        try:
            # Check if FTS5 is supported by trying a simple FTS5 table creation
            # This is a more reliable check than just catching the error on the main table creation.
            cursor.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS fts5_test_table USING fts5(test_col)"
            )
            cursor.execute("DROP TABLE IF EXISTS fts5_test_table")

            # The 'UNINDEXED' keyword for fq_name means it's not stored separately as a column value accessible
            # outside FTS matching, but it *is* the primary content being indexed for full-text search.
            # If you need to retrieve fq_name as a standard column, remove UNINDEXED.
            # For duplicate checks, just matching is often enough.
            # Let's make fq_name a regular indexed column as well for easier retrieval if needed.
            # Drop the table if it exists to ensure the schema is correctly applied (e.g. if UNINDEXED status of fq_name changes)
            cursor.execute("DROP TABLE IF EXISTS symbols_fts")
            cursor.execute(
                """
                CREATE VIRTUAL TABLE symbols_fts USING fts5(
                    fq_name,          -- Fully qualified name
                    file_path,
                    symbol_type,
                    signature,
                    tokenize = 'unicode61 remove_diacritics 0'
                )
            """
            )
            print("FTS5 table 'symbols_fts' created or already exists.")
        except sqlite3.OperationalError as e:
            if "no such module: fts5" in str(e):
                print(
                    "FTS5 module is not available in this SQLite version. "
                    "This module requires an SQLite version compiled with FTS5 support."
                )
                # Fallback to FTS4 example (less common these days)
                # cursor.execute("CREATE VIRTUAL TABLE IF NOT EXISTS symbols_fts USING fts4(fq_name, file_path, symbol_type, signature, tokenize=unicode61)")
                # print("Attempted fallback to FTS4 for 'symbols_fts'.")
                # For now, we'll let this be an error if FTS5 isn't there, as FTS5 is preferred.
                raise  # Re-raise the exception if FTS5 is critical
            else:
                raise  # Re-raise other operational errors

        # Create indexes for faster queries on naming_rules if needed (e.g., on identifier_type)
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_naming_rules_identifier_type ON naming_rules (identifier_type)"
        )

        conn.commit()
        print(
            f"Naming conventions database schema created successfully at {db_path.resolve()}"
        )
        return True

    except sqlite3.Error as e:
        print(f"SQLite error during database setup at {db_path.resolve()}: {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred during database setup: {e}")
        return False
    finally:
        if "conn" in locals() and conn:
            conn.close()


if __name__ == "__main__":
    # Example Usage:
    # Define a temporary path for the test database
    temp_db_path = Path("temp_naming_conventions.db")

    # Ensure the test db doesn't exist from a previous run
    if temp_db_path.exists():
        temp_db_path.unlink()

    print(f"--- Creating Naming Conventions DB at {temp_db_path.name} ---")
    success = create_naming_conventions_db(db_path=temp_db_path)

    if success:
        print("Database schema created (or verified).")
        # Optionally, connect and inspect the schema
        try:
            conn = sqlite3.connect(str(temp_db_path))
            cursor = conn.cursor()

            print("\n--- Schema Inspection ---")
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            print("Tables:", tables)

            if any("naming_rules" in t for t in tables):
                cursor.execute("PRAGMA table_info(naming_rules);")
                print("naming_rules columns:", cursor.fetchall())

            if any("symbols_fts" in t for t in tables):
                cursor.execute(
                    "PRAGMA table_info(symbols_fts);"
                )  # For FTS tables, this shows internal structure
                print("symbols_fts columns (internal):", cursor.fetchall())
                # Example: Add a dummy record and search (if FTS5 is available)
                try:
                    cursor.execute(
                        "INSERT INTO symbols_fts (fq_name, file_path, symbol_type, signature) VALUES (?, ?, ?, ?)",
                        (
                            "my.module.MyClass.my_method",
                            "src/my_module.py",
                            "method",
                            "(self, arg1: int) -> str",
                        ),
                    )
                    conn.commit()
                    cursor.execute(
                        "SELECT fq_name, symbol_type FROM symbols_fts WHERE symbols_fts MATCH 'MyClass.my_method'"
                    )
                    print(
                        "FTS search result for 'MyClass.my_method':", cursor.fetchall()
                    )
                except sqlite3.OperationalError as e:
                    # This might occur if FTS5 was reported as available but has issues,
                    # or if the test FTS5 table creation in the main function failed silently earlier.
                    print(f"Could not test FTS table insertion/query: {e}")

        except sqlite3.Error as e:
            print(f"Error during schema inspection: {e}")
        finally:
            if "conn" in locals() and conn:
                conn.close()
    else:
        print("Failed to create database schema.")

    # Clean up the temporary database file
    if temp_db_path.exists():
        # temp_db_path.unlink() # Comment out to inspect the db manually after run
        # print(f"Cleaned up {temp_db_path.name}")
        print(
            f"Temporary database {temp_db_path.name} was created. Inspect it manually if desired."
        )
