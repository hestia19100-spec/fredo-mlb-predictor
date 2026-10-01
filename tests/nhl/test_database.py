from __future__ import annotations

from contextlib import closing
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from src.nhl.database import (
    EXPECTED_SCHEMA_OBJECTS,
    MIGRATION_1,
    NHLDatabaseError,
    connect_nhl_database,
    connect_nhl_database_read_only,
    initialize_nhl_database,
    load_and_validate_storage_protocol,
    validate_nhl_database_path,
    validate_storage_protocol,
    verify_nhl_database,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = (
    PROJECT_ROOT / "nhl_protocols" / "data" / "nhl_storage_protocol_v1.json"
)


class NHLDatabaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "nhl"
        self.database = self.root / "test_nhl.db"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_path_must_be_directly_inside_dedicated_nhl_root(self) -> None:
        self.assertEqual(
            validate_nhl_database_path(self.database, allowed_root=self.root),
            self.database.resolve(),
        )
        for path, root in (
            (Path(self.temporary.name) / "outside.db", self.root),
            (self.root / "nested" / "test.db", self.root),
            (self.root / "fredo_mlb.db", self.root),
            (self.database, Path(self.temporary.name) / "sport"),
        ):
            with self.subTest(path=path, root=root):
                with self.assertRaises(NHLDatabaseError):
                    validate_nhl_database_path(path, allowed_root=root)

    def test_initialization_is_explicit_and_idempotent(self) -> None:
        self.assertFalse(self.database.exists())
        initialize_nhl_database(self.database, allowed_root=self.root)
        first_bytes = self.database.read_bytes()
        initialize_nhl_database(self.database, allowed_root=self.root)
        verify_nhl_database(self.database, allowed_root=self.root)
        self.assertEqual(self.database.read_bytes(), first_bytes)

        with closing(
            connect_nhl_database_read_only(
                self.database,
                allowed_root=self.root,
            )
        ) as connection:
            objects = {
                row[0]
                for row in connection.execute(
                    """
                    SELECT name FROM sqlite_master
                    WHERE (type = 'table' OR type = 'trigger')
                      AND name LIKE 'nhl_%'
                    """
                )
            }
            migration = connection.execute(
                "SELECT version, migration_sha256 FROM nhl_schema_migrations"
            ).fetchone()
        self.assertEqual(objects, EXPECTED_SCHEMA_OBJECTS)
        self.assertEqual(tuple(migration), (1, MIGRATION_1.sha256))

    def test_read_only_connection_cannot_write(self) -> None:
        initialize_nhl_database(self.database, allowed_root=self.root)
        with closing(
            connect_nhl_database_read_only(
                self.database,
                allowed_root=self.root,
            )
        ) as connection:
            with self.assertRaises(sqlite3.OperationalError):
                connection.execute(
                    "UPDATE nhl_schema_migrations SET applied_at_utc = 'x'"
                )

    def test_append_only_triggers_reject_update_and_delete(self) -> None:
        initialize_nhl_database(self.database, allowed_root=self.root)
        with closing(
            connect_nhl_database(self.database, allowed_root=self.root)
        ) as connection:
            for command in (
                "UPDATE nhl_schema_migrations SET applied_at_utc = applied_at_utc",
                "DELETE FROM nhl_schema_migrations",
            ):
                with self.subTest(command=command):
                    with self.assertRaises(sqlite3.IntegrityError):
                        connection.execute(command)

    def test_corrupted_schema_is_rejected(self) -> None:
        initialize_nhl_database(self.database, allowed_root=self.root)
        with closing(sqlite3.connect(self.database)) as connection, connection:
            connection.execute("DROP TRIGGER nhl_ingestion_runs_no_delete")
        with self.assertRaisesRegex(NHLDatabaseError, "objets inattendus"):
            verify_nhl_database(self.database, allowed_root=self.root)

    def test_partial_preexisting_schema_is_never_adopted(self) -> None:
        self.root.mkdir(parents=True)
        with closing(sqlite3.connect(self.database)) as connection, connection:
            connection.execute("CREATE TABLE nhl_foreign_table(value TEXT)")
        with self.assertRaises(NHLDatabaseError):
            initialize_nhl_database(self.database, allowed_root=self.root)

    def test_protocol_matches_code_and_keeps_all_runtime_disabled(self) -> None:
        document = load_and_validate_storage_protocol(PROTOCOL_PATH)
        self.assertEqual(document["storage"]["schema_version"], 1)
        self.assertTrue(document["safety"]["synthetic_offline_fixtures_only"])
        self.assertFalse(document["safety"]["provider_calls_allowed"])
        self.assertFalse(document["safety"]["mlb_database_access_allowed"])

    def test_protocol_cannot_enable_provider_calls(self) -> None:
        document = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        document["safety"]["provider_calls_allowed"] = True
        with self.assertRaises(NHLDatabaseError):
            validate_storage_protocol(document)


if __name__ == "__main__":
    unittest.main()
