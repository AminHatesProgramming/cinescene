from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional


DB_PATH = Path("data/app_memory.sqlite")


class AppMemory:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self):
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS searches (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    query TEXT NOT NULL,
                    top_k INTEGER NOT NULL,
                    use_reranking INTEGER NOT NULL,
                    results_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS favorites (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    movie_id TEXT,
                    title TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(session_id, title)
                );

                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    query TEXT NOT NULL,
                    movie_title TEXT NOT NULL,
                    signal TEXT NOT NULL,
                    note TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS ingestions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    movie_title TEXT NOT NULL,
                    source_video TEXT NOT NULL,
                    scene_count INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                """
            )

    def add_search(self, session_id: str, query: str, top_k: int, use_reranking: bool, results: List[Dict]):
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO searches(session_id, query, top_k, use_reranking, results_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, query, top_k, int(use_reranking), json.dumps(results, ensure_ascii=False)),
            )

    def recent_searches(self, session_id: str, limit: int = 12) -> List[Dict]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT id, query, top_k, use_reranking, results_json, created_at
                FROM searches
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
        return [
            {
                "id": row["id"],
                "query": row["query"],
                "top_k": row["top_k"],
                "use_reranking": bool(row["use_reranking"]),
                "results": json.loads(row["results_json"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def add_favorite(self, session_id: str, movie: Dict):
        movie_id = str(movie.get("id") or movie.get("title") or "")
        title = movie.get("title", "Unknown")
        with self.connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO favorites(session_id, movie_id, title, payload_json)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, movie_id, title, json.dumps(movie, ensure_ascii=False)),
            )

    def favorites(self, session_id: str) -> List[Dict]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT payload_json, created_at
                FROM favorites
                WHERE session_id = ?
                ORDER BY id DESC
                """,
                (session_id,),
            ).fetchall()
        items = []
        for row in rows:
            payload = json.loads(row["payload_json"])
            payload["favorited_at"] = row["created_at"]
            items.append(payload)
        return items

    def add_feedback(self, session_id: str, query: str, movie_title: str, signal: str, note: Optional[str] = None):
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO feedback(session_id, query, movie_title, signal, note)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, query, movie_title, signal, note),
            )

    def add_ingestion(self, session_id: str, movie_title: str, source_video: str, scenes: List[Dict]):
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO ingestions(session_id, movie_title, source_video, scene_count, payload_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, movie_title, source_video, len(scenes), json.dumps(scenes, ensure_ascii=False)),
            )

    def ingestions(self, session_id: str, limit: int = 10) -> List[Dict]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT id, movie_title, source_video, scene_count, payload_json, created_at
                FROM ingestions
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
        return [
            {
                "id": row["id"],
                "movie_title": row["movie_title"],
                "source_video": row["source_video"],
                "scene_count": row["scene_count"],
                "scenes": json.loads(row["payload_json"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]
