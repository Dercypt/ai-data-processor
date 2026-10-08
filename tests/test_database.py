import json
import os
import sqlite3
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from database import (
    AuthenticationError,
    EntryNotFoundError,
    UserAlreadyExistsError,
    authenticate_user,
    create_user,
    delete_entry,
    get_all_entries,
    get_conn,
    get_connection,
    get_user_by_id,
    get_user_by_username,
    hash_password,
    init_db,
    save_entry,
    verify_password,
)


class TestDatabasePersistence(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_history.db")
        init_db(self.db_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_init_db_creates_relational_schema(self):
        with get_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = {row["name"] for row in cursor.fetchall()}
            self.assertIn("users", tables)
            self.assertIn("history", tables)

            # Verify default user exists
            cursor.execute("SELECT id, username FROM users WHERE username = 'default'")
            default_user = cursor.fetchone()
            self.assertIsNotNone(default_user)
            self.assertEqual(default_user["id"], 1)

    def test_password_hashing_and_verification(self):
        pwd = "SecretPassword123!"
        hashed = hash_password(pwd)
        self.assertIn("$", hashed)
        self.assertTrue(verify_password(pwd, hashed))
        self.assertFalse(verify_password("WrongPassword", hashed))
        self.assertFalse(verify_password("", hashed))
        with self.assertRaises(ValueError):
            hash_password("")

    def test_user_registration_and_authentication(self):
        user_id = create_user("analyst_jane", "JanePass123", db_path=self.db_path)
        self.assertIsInstance(user_id, int)
        self.assertGreater(user_id, 1)

        # Authenticate successfully
        auth_data = authenticate_user("analyst_jane", "JanePass123", db_path=self.db_path)
        self.assertEqual(auth_data["id"], user_id)
        self.assertEqual(auth_data["username"], "analyst_jane")

        # Wrong password raises AuthenticationError
        with self.assertRaises(AuthenticationError):
            authenticate_user("analyst_jane", "BadPassword", db_path=self.db_path)

        # Non-existent user raises AuthenticationError
        with self.assertRaises(AuthenticationError):
            authenticate_user("nonexistent", "SomePass", db_path=self.db_path)

        # Duplicate username raises UserAlreadyExistsError
        with self.assertRaises(UserAlreadyExistsError):
            create_user("analyst_jane", "AnotherPass", db_path=self.db_path)

        # Empty credentials raise ValueError
        with self.assertRaises(ValueError):
            create_user("", "pass", db_path=self.db_path)
        with self.assertRaises(ValueError):
            create_user("user", "", db_path=self.db_path)

    def test_get_user_lookups(self):
        uid = create_user("student_bob", "BobPass123", db_path=self.db_path)
        user_info = get_user_by_id(uid, db_path=self.db_path)
        self.assertEqual(user_info["username"], "student_bob")

        by_name = get_user_by_username("student_bob", db_path=self.db_path)
        self.assertIsNotNone(by_name)
        self.assertEqual(by_name["id"], uid)

        self.assertIsNone(get_user_by_username("nobody", db_path=self.db_path))
        with self.assertRaises(EntryNotFoundError):
            get_user_by_id(99999, db_path=self.db_path)

    def test_save_entry_and_historical_schema_law4(self):
        # Law 4 requires [timestamp, filename, insights] and %Y-%m-%d %H:%M:%S format
        insights_payload = {
            "summary": "Good dataset summary.",
            "risks": ["Outlier detected in column X"],
            "recommendations": ["Scale feature X"],
        }
        save_entry("quarterly_sales.csv", insights_payload, db_path=self.db_path)

        entries = get_all_entries(db_path=self.db_path)
        self.assertEqual(len(entries), 1)
        entry_id, timestamp, filename, insights_str = entries[0]

        self.assertEqual(filename, "quarterly_sales.csv")
        # Validate timestamp adherence to %Y-%m-%d %H:%M:%S
        parsed_time = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
        self.assertIsInstance(parsed_time, datetime)

        # Validate insights format
        parsed_insights = json.loads(insights_str)
        self.assertEqual(parsed_insights["summary"], "Good dataset summary.")

    def test_multi_user_private_history_isolation(self):
        user1_id = create_user("analyst_1", "Pass1", db_path=self.db_path)
        user2_id = create_user("analyst_2", "Pass2", db_path=self.db_path)

        save_entry("dataset_1.csv", "Insights 1", user_id=user1_id, db_path=self.db_path)
        save_entry("dataset_2.csv", "Insights 2", user_id=user2_id, db_path=self.db_path)

        user1_entries = get_all_entries(user_id=user1_id, db_path=self.db_path)
        user2_entries = get_all_entries(user_id=user2_id, db_path=self.db_path)

        self.assertEqual(len(user1_entries), 1)
        self.assertEqual(user1_entries[0][2], "dataset_1.csv")

        self.assertEqual(len(user2_entries), 1)
        self.assertEqual(user2_entries[0][2], "dataset_2.csv")

    def test_delete_entry_and_authorization(self):
        user1_id = create_user("owner_user", "OwnerPass", db_path=self.db_path)
        user2_id = create_user("other_user", "OtherPass", db_path=self.db_path)

        save_entry("owner_data.csv", "Insights", user_id=user1_id, db_path=self.db_path)
        entries = get_all_entries(user_id=user1_id, db_path=self.db_path)
        entry_id = entries[0][0]

        # User 2 attempting to delete User 1's entry should fail with EntryNotFoundError
        with self.assertRaises(EntryNotFoundError):
            delete_entry(entry_id, user_id=user2_id, db_path=self.db_path)

        # Entry still exists for User 1
        self.assertEqual(len(get_all_entries(user_id=user1_id, db_path=self.db_path)), 1)

        # User 1 successfully deletes their own entry
        delete_entry(entry_id, user_id=user1_id, db_path=self.db_path)
        self.assertEqual(len(get_all_entries(user_id=user1_id, db_path=self.db_path)), 0)

        # Deleting non-existent entry raises EntryNotFoundError
        with self.assertRaises(EntryNotFoundError):
            delete_entry(entry_id, db_path=self.db_path)

    def test_foreign_key_cascade_deletion(self):
        user_id = create_user("temp_user", "TempPass", db_path=self.db_path)
        save_entry("temp_file.csv", "Insights", user_id=user_id, db_path=self.db_path)

        # Delete user
        with get_connection(self.db_path) as conn:
            conn.execute("DELETE FROM users WHERE id = ?", (user_id,))

        # History record should be deleted via ON DELETE CASCADE
        with get_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM history WHERE user_id = ?", (user_id,))
            self.assertEqual(len(cursor.fetchall()), 0)

    def test_invalid_arguments_handling(self):
        with self.assertRaises(ValueError):
            save_entry("", "Insights", db_path=self.db_path)
        with self.assertRaises(ValueError):
            delete_entry("not_an_int", db_path=self.db_path)

    def test_get_conn_backward_compatibility(self):
        conn = get_conn(self.db_path)
        self.assertIsInstance(conn, sqlite3.Connection)
        conn.close()

    def test_connection_pragmas_wal_and_synchronous(self):
        with get_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA foreign_keys;")
            self.assertEqual(cursor.fetchone()[0], 1)

            cursor.execute("PRAGMA journal_mode;")
            self.assertEqual(str(cursor.fetchone()[0]).lower(), "wal")

            cursor.execute("PRAGMA synchronous;")
            # SQLite returns 1 for NORMAL synchronous mode
            self.assertEqual(cursor.fetchone()[0], 1)

    def test_concurrent_multi_analyst_writes_under_wal(self):
        # Verify simultaneous writes by multiple analysts do not raise database locked errors
        num_analysts = 10
        num_writes_per_analyst = 3

        def analyst_work(analyst_idx: int) -> None:
            user_id = create_user(
                f"analyst_worker_{analyst_idx}",
                f"SecretPassword{analyst_idx}!",
                db_path=self.db_path,
            )
            for write_idx in range(num_writes_per_analyst):
                save_entry(
                    f"dataset_analyst_{analyst_idx}_{write_idx}.csv",
                    {"analyst": analyst_idx, "write": write_idx},
                    user_id=user_id,
                    db_path=self.db_path,
                )

        with ThreadPoolExecutor(max_workers=num_analysts) as executor:
            futures = [executor.submit(analyst_work, i) for i in range(num_analysts)]
            for future in futures:
                future.result()

        entries = get_all_entries(db_path=self.db_path)
        expected_total = num_analysts * num_writes_per_analyst
        self.assertEqual(len(entries), expected_total)


if __name__ == "__main__":
    unittest.main()
