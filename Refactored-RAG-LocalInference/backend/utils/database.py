import sqlite3
import os
from datetime import datetime

class ChatDatabase:
    def __init__(self, db_path):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    thread_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def add_message(self, user_id, thread_id, role, content):
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO messages (user_id, thread_id, role, content) VALUES (?, ?, ?, ?)",
                (user_id, thread_id, role, content)
            )
            conn.commit()

    def get_recent_history(self, user_id, limit=50):
        """Returns the last `limit` messages for a USER (cross-thread), NEWEST first."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT role, content FROM messages 
                WHERE user_id = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
                """,
                (user_id, limit)
            )
            rows = cursor.fetchall()
            return [{"role": row[0], "content": row[1]} for row in rows]

    def clear_history(self, user_id):
        with self._get_connection() as conn:
            conn.execute("DELETE FROM messages WHERE user_id = ?", (user_id,))
            conn.commit()
