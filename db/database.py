import sqlite3
import os
import contextlib
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "memorygraph.db"
SCHEMA_PATH = Path(__file__).parent / "schema.sql"


@contextlib.contextmanager
def get_connection():
    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=10000")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    schema = SCHEMA_PATH.read_text()
    with get_connection() as conn:
        conn.executescript(schema)
    print(f"Database initialized at {DB_PATH}")
