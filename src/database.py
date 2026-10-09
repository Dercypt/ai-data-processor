import hashlib
import json
import os
import secrets
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Dict, Generator, List, Optional, Tuple, Union



# --- DOMAIN EXCEPTIONS ---

class DatabaseError(Exception):
    """Base domain exception for database operations."""
    pass


class UserAlreadyExistsError(DatabaseError):
    """Raised when attempting to create a user with an existing username."""
    pass


class AuthenticationError(DatabaseError):
    """Raised when authentication credentials fail or user is missing."""
    pass


class EntryNotFoundError(DatabaseError):
    """Raised when a specific database record is not found."""
    pass


# --- CONSTANTS & CONFIGURATION ---

DEFAULT_DB_PATH: str = "history.db"
DEFAULT_USERNAME: str = "default"


def get_db_path() -> str:
    """Resolve database path from environment variable or fallback."""
    if os.environ.get("SQLITE_DB_PATH"):
        return os.environ["SQLITE_DB_PATH"]
    return DEFAULT_DB_PATH


def get_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """
    Open and return a configured SQLite connection with foreign keys enabled,
    WAL journal mode, and NORMAL synchronous mode for concurrent multi-analyst access.
    """
    target_path = db_path or get_db_path()
    conn = sqlite3.connect(target_path, timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    return conn


def get_conn(db_path: Optional[str] = None) -> sqlite3.Connection:
    """Convenience alias for get_connection."""
    return get_connection(db_path)


@contextmanager
def get_db_connection(db_path: Optional[str] = None) -> Generator[sqlite3.Connection, None, None]:
    """Context manager providing a transactional SQLite connection."""
    conn = get_connection(db_path)
    try:
        with conn:
            yield conn
    finally:
        conn.close()


# --- AUTHENTICATION & SECURITY UTILITIES ---

def hash_password(password: str) -> str:
    """Hash password using PBKDF2-HMAC-SHA256 with a cryptographically secure salt."""
    if not password:
        raise ValueError("Password cannot be empty.")
    salt = secrets.token_hex(16)
    key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100000)
    return f"{salt}${key.hex()}"


def verify_password(password: str, hashed: str) -> bool:
    """Verify candidate password against stored salt$hash representation."""
    if not password or not hashed or "$" not in hashed:
        return False
    try:
        salt, key_hex = hashed.split("$", 1)
        candidate = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100000)
        return secrets.compare_digest(candidate.hex(), key_hex)
    except Exception:
        return False


def _get_default_user_id(conn: sqlite3.Connection) -> int:
    """Retrieve or create the fallback default user ID for anonymous/shared sessions."""
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM users WHERE username = ?", (DEFAULT_USERNAME,))
    row = cursor.fetchone()
    if row:
        return int(row["id"])
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(
        "INSERT INTO users (username, password_hash, created_at) VALUES (?, '', ?)",
        (DEFAULT_USERNAME, now_str),
    )
    return int(cursor.lastrowid)


# --- CORE PERSISTENCE API ---

def init_db(db_path: Optional[str] = None) -> None:
    """
    Initialize SQLite relational tables, indexes, and default records.
    Enforces strict relational model with foreign key constraints.
    """
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                filename TEXT NOT NULL,
                insights TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            );
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_history_user_id ON history (user_id);
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_history_timestamp ON history (timestamp DESC);
            """
        )
        # Ensure default user exists
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute(
            """
            INSERT OR IGNORE INTO users (id, username, password_hash, created_at)
            VALUES (1, ?, '', ?)
            """,
            (DEFAULT_USERNAME, now_str),
        )


def create_user(username: str, password: str, db_path: Optional[str] = None) -> int:
    """
    Register a new analyst account.
    Returns the newly created user ID.
    Raises ValueError on empty credentials or UserAlreadyExistsError on collision.
    """
    clean_username = username.strip()
    if not clean_username:
        raise ValueError("Username cannot be empty.")
    if not password:
        raise ValueError("Password cannot be empty.")

    password_hash = hash_password(password)
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
                (clean_username, password_hash, created_at),
            )
            return int(cursor.lastrowid)
        except sqlite3.IntegrityError as err:
            raise UserAlreadyExistsError(f"Username '{clean_username}' already exists.") from err


def authenticate_user(
    username: str,
    password: str,
    db_path: Optional[str] = None,
) -> Dict[str, Union[int, str]]:
    """
    Authenticate an analyst by credentials.
    Returns user dictionary with 'id' and 'username'.
    Raises AuthenticationError on invalid credentials.
    """
    clean_username = username.strip()
    if not clean_username or not password:
        raise AuthenticationError("Username and password cannot be empty.")

    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, username, password_hash FROM users WHERE username = ?",
            (clean_username,),
        )
        row = cursor.fetchone()
        if not row or not verify_password(password, row["password_hash"]):
            raise AuthenticationError("Invalid username or password.")

        return {"id": int(row["id"]), "username": str(row["username"])}


def get_user_by_id(user_id: int, db_path: Optional[str] = None) -> Dict[str, Union[int, str]]:
    """Fetch analyst record by ID."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, username, created_at FROM users WHERE id = ?", (user_id,))
        row = cursor.fetchone()
        if not row:
            raise EntryNotFoundError(f"User with ID {user_id} not found.")
        return {
            "id": int(row["id"]),
            "username": str(row["username"]),
            "created_at": str(row["created_at"]),
        }


def get_user_by_username(
    username: str,
    db_path: Optional[str] = None,
) -> Optional[Dict[str, Union[int, str]]]:
    """Fetch analyst record by username if present, otherwise None."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, username, created_at FROM users WHERE username = ?",
            (username.strip(),),
        )
        row = cursor.fetchone()
        if not row:
            return None
        return {
            "id": int(row["id"]),
            "username": str(row["username"]),
            "created_at": str(row["created_at"]),
        }


def save_entry(
    filename: str,
    insights: Union[str, Dict[str, object]],
    user_id: Optional[int] = None,
    db_path: Optional[str] = None,
) -> None:
    """
    Appends a new analysis record to the relational database.
    Strictly preserves historical schema [timestamp, filename, insights] (Law 4).
    """
    if not filename:
        raise ValueError("Filename cannot be empty.")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    insights_str = (
        json.dumps(insights)
        if isinstance(insights, (dict, list))
        else str(insights)
    )

    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        resolved_user_id = user_id if user_id is not None else _get_default_user_id(conn)

        cursor.execute(
            """
            INSERT INTO history (user_id, timestamp, filename, insights)
            VALUES (?, ?, ?, ?)
            """,
            (resolved_user_id, timestamp, filename, insights_str),
        )


def get_all_entries(
    user_id: Optional[int] = None,
    db_path: Optional[str] = None,
) -> List[Tuple[int, str, str, str]]:
    """
    Fetches historical analysis entries, sorted descending by timestamp.
    Returns list of 4-tuples: (id, timestamp, filename, insights).
    If user_id is provided, filters history strictly to that user's private workspace.
    """
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()

        if user_id is not None:
            cursor.execute(
                """
                SELECT id, timestamp, filename, insights
                FROM history
                WHERE user_id = ?
                ORDER BY timestamp DESC, id DESC
                """,
                (user_id,),
            )
        else:
            cursor.execute(
                """
                SELECT id, timestamp, filename, insights
                FROM history
                ORDER BY timestamp DESC, id DESC
                """
            )

        rows = cursor.fetchall()
        return [
            (int(r["id"]), str(r["timestamp"]), str(r["filename"]), str(r["insights"]))
            for r in rows
        ]


def delete_entry(
    entry_id: Union[int, str],
    user_id: Optional[int] = None,
    db_path: Optional[str] = None,
) -> None:
    """
    Deletes a specific history record by ID.
    If user_id is specified, ensures deletion is isolated to the user's records.
    Raises EntryNotFoundError if record cannot be found.
    """
    try:
        numeric_id = int(entry_id)
    except (ValueError, TypeError) as err:
        raise ValueError(f"Invalid entry ID: {entry_id}") from err

    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()

        if user_id is not None:
            cursor.execute(
                "DELETE FROM history WHERE id = ? AND user_id = ?",
                (numeric_id, user_id),
            )
        else:
            cursor.execute("DELETE FROM history WHERE id = ?", (numeric_id,))

        if cursor.rowcount == 0:
            raise EntryNotFoundError(f"History entry with ID {entry_id} not found.")