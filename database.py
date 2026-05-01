"""
Small persistence layer for users, images, analysis results, and audit events.

PostgreSQL is used when DATABASE_URL is configured. A SQLite fallback keeps the
thesis demo runnable during local development without Azure services.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

try:
    import psycopg
    from psycopg.rows import dict_row
except ImportError:  # pragma: no cover - dependency is optional at import time.
    psycopg = None
    dict_row = None

from app_config import BASE_DIR, load_app_config


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_dump(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


class Database:
    def __init__(self) -> None:
        self.config = load_app_config()
        self.database_url = self.config.database_url
        self.is_postgres = bool(self.database_url)
        self.sqlite_path = BASE_DIR / "data" / "hare_demo.sqlite3"

    @contextmanager
    def connect(self) -> Iterator[Any]:
        if self.is_postgres:
            if psycopg is None:
                raise RuntimeError("psycopg is required when DATABASE_URL is configured.")
            with psycopg.connect(self.database_url, row_factory=dict_row) as conn:
                yield conn
            return

        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.sqlite_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _sql(self, sql: str) -> str:
        return sql.replace("?", "%s") if self.is_postgres else sql

    def execute(self, sql: str, params: Sequence[Any] = ()) -> None:
        with self.connect() as conn:
            conn.execute(self._sql(sql), params)

    def execute_returning_id(self, sql: str, params: Sequence[Any] = ()) -> int:
        with self.connect() as conn:
            if self.is_postgres:
                final_sql = self._sql(sql)
                if "RETURNING" not in final_sql.upper():
                    final_sql = f"{final_sql} RETURNING id"
                cursor = conn.execute(final_sql, params)
                row = cursor.fetchone()
                return int(row["id"])
            cursor = conn.execute(sql, params)
            return int(cursor.lastrowid)

    def fetch_one(self, sql: str, params: Sequence[Any] = ()) -> Optional[Dict[str, Any]]:
        with self.connect() as conn:
            row = conn.execute(self._sql(sql), params).fetchone()
        if row is None:
            return None
        return dict(row)

    def fetch_all(self, sql: str, params: Sequence[Any] = ()) -> List[Dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(self._sql(sql), params).fetchall()
        return [dict(row) for row in rows]

    def init_schema(self) -> None:
        if self.is_postgres:
            statements = [
                """
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL CHECK (role IN ('patient', 'researcher')),
                    status TEXT NOT NULL CHECK (status IN ('pending', 'active', 'disabled', 'deleted')),
                    created_at TIMESTAMPTZ NOT NULL,
                    approved_at TIMESTAMPTZ,
                    approved_by INTEGER REFERENCES users(id)
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS image_records (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL REFERENCES users(id),
                    original_filename TEXT NOT NULL,
                    content_type TEXT NOT NULL,
                    blob_path TEXT NOT NULL,
                    storage_backend TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    image_format TEXT NOT NULL,
                    dicom_metadata JSONB,
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at TIMESTAMPTZ NOT NULL,
                    deleted_at TIMESTAMPTZ
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS analysis_results (
                    id SERIAL PRIMARY KEY,
                    image_id INTEGER NOT NULL REFERENCES image_records(id),
                    user_id INTEGER NOT NULL REFERENCES users(id),
                    predicted_label TEXT NOT NULL,
                    predicted_summary TEXT NOT NULL,
                    melanoma_probability DOUBLE PRECISION NOT NULL,
                    confidence_score DOUBLE PRECISION NOT NULL,
                    ga_threshold DOUBLE PRECISION NOT NULL,
                    robustness_status TEXT NOT NULL,
                    robustness_attack TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS attack_simulations (
                    id SERIAL PRIMARY KEY,
                    image_id INTEGER NOT NULL REFERENCES image_records(id),
                    user_id INTEGER NOT NULL REFERENCES users(id),
                    attack_type TEXT NOT NULL,
                    epsilon DOUBLE PRECISION NOT NULL,
                    before_label TEXT NOT NULL,
                    after_label TEXT NOT NULL,
                    changed BOOLEAN NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS audit_events (
                    id SERIAL PRIMARY KEY,
                    actor_user_id INTEGER REFERENCES users(id),
                    target_resource TEXT,
                    event_type TEXT NOT NULL,
                    success BOOLEAN NOT NULL,
                    ip_address TEXT,
                    details JSONB NOT NULL,
                    previous_hash TEXT NOT NULL,
                    current_hash TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL
                )
                """,
            ]
        else:
            statements = [
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    approved_at TEXT,
                    approved_by INTEGER
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS image_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    original_filename TEXT NOT NULL,
                    content_type TEXT NOT NULL,
                    blob_path TEXT NOT NULL,
                    storage_backend TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    image_format TEXT NOT NULL,
                    dicom_metadata TEXT,
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at TEXT NOT NULL,
                    deleted_at TEXT
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS analysis_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_id INTEGER NOT NULL,
                    user_id INTEGER NOT NULL,
                    predicted_label TEXT NOT NULL,
                    predicted_summary TEXT NOT NULL,
                    melanoma_probability REAL NOT NULL,
                    confidence_score REAL NOT NULL,
                    ga_threshold REAL NOT NULL,
                    robustness_status TEXT NOT NULL,
                    robustness_attack TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS attack_simulations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_id INTEGER NOT NULL,
                    user_id INTEGER NOT NULL,
                    attack_type TEXT NOT NULL,
                    epsilon REAL NOT NULL,
                    before_label TEXT NOT NULL,
                    after_label TEXT NOT NULL,
                    changed INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS audit_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    actor_user_id INTEGER,
                    target_resource TEXT,
                    event_type TEXT NOT NULL,
                    success INTEGER NOT NULL,
                    ip_address TEXT,
                    details TEXT NOT NULL,
                    previous_hash TEXT NOT NULL,
                    current_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """,
            ]

        with self.connect() as conn:
            for statement in statements:
                conn.execute(statement)

    def audit(
        self,
        event_type: str,
        actor_user_id: Optional[int] = None,
        target_resource: Optional[str] = None,
        success: bool = True,
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
    ) -> None:
        details = details or {}
        latest = self.fetch_one("SELECT current_hash FROM audit_events ORDER BY id DESC LIMIT 1")
        previous_hash = latest["current_hash"] if latest else "GENESIS"
        created_at = utc_now()
        payload = {
            "actor_user_id": actor_user_id,
            "target_resource": target_resource,
            "event_type": event_type,
            "success": bool(success),
            "details": details,
            "previous_hash": previous_hash,
            "created_at": created_at,
        }
        current_hash = hashlib.sha256(_json_dump(payload).encode("utf-8")).hexdigest()
        self.execute(
            """
            INSERT INTO audit_events (
                actor_user_id, target_resource, event_type, success, ip_address,
                details, previous_hash, current_hash, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                actor_user_id,
                target_resource,
                event_type,
                bool(success),
                ip_address,
                _json_dump(details),
                previous_hash,
                current_hash,
                created_at,
            ),
        )


db = Database()


def initialize_database() -> None:
    db.init_schema()
