import sqlite3
from pathlib import Path
from typing import Dict, List, Any

DB_PATH = Path("data/recommender_system.db")


def init_db() -> None:
    """Initialize the SQLite database schema."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create search_history table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS search_history (
            search_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            query_text TEXT,
            image_path TEXT,
            search_type TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Create search_results table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS search_results (
            search_id INTEGER,
            item_id TEXT NOT NULL,
            rank INTEGER NOT NULL,
            score REAL NOT NULL,
            PRIMARY KEY (search_id, item_id),
            FOREIGN KEY(search_id) REFERENCES search_history(search_id) ON DELETE CASCADE
        )
    """)

    # Create grades table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS grades (
            user_id TEXT NOT NULL,
            item_id TEXT NOT NULL,
            query_text TEXT NOT NULL,
            image_path TEXT NOT NULL,
            rating INTEGER NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (user_id, item_id, query_text, image_path)
        )
    """)

    conn.commit()
    conn.close()


def log_search(
    user_id: str,
    query_text: str | None,
    image_path: str | None,
    search_type: str,
    results: List[tuple]
) -> int:
    """Log a search query and its returned results.

    Args:
        user_id: ID of the user searching
        query_text: text input (if any)
        image_path: path to the query image (if any)
        search_type: type of search (vector or classical)
        results: list of tuples of (item_id, score)

    Returns:
        inserted search_id
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Normalise optional arguments to empty strings to avoid NULL issues when checking grades
    q_text = query_text or ""
    i_path = image_path or ""

    cursor.execute(
        "INSERT INTO search_history (user_id, query_text, image_path, search_type) VALUES (?, ?, ?, ?)",
        (user_id, q_text, i_path, search_type),
    )
    search_id = cursor.lastrowid

    for rank, (item_id, score) in enumerate(results, start=1):
        cursor.execute(
            "INSERT INTO search_results (search_id, item_id, rank, score) VALUES (?, ?, ?, ?)",
            (search_id, str(item_id), rank, score),
        )

    conn.commit()
    conn.close()
    return search_id


def log_grade(user_id: str, item_id: str, query_text: str | None, image_path: str | None, rating: int) -> None:
    """Log or overwrite a relevance grade for a query-item pair."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    q_text = query_text or ""
    i_path = image_path or ""

    cursor.execute(
        """
        INSERT OR REPLACE INTO grades (user_id, item_id, query_text, image_path, rating)
        VALUES (?, ?, ?, ?, ?)
        """,
        (user_id, str(item_id), q_text, i_path, rating),
    )
    conn.commit()
    conn.close()


def get_all_grades() -> List[Dict[str, Any]]:
    """Retrieve all relevance grades."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM grades ORDER BY timestamp DESC")
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def get_search_history() -> List[Dict[str, Any]]:
    """Retrieve search history logs."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM search_history ORDER BY timestamp DESC")
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def get_search_results(search_id: int) -> List[Dict[str, Any]]:
    """Retrieve search results for a given search_id."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        "SELECT item_id, rank, score FROM search_results WHERE search_id = ? ORDER BY rank ASC",
        (search_id,),
    )
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows
