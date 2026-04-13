from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any


class Memory:
    """Screenpipe-backed OCR history lookup with schema-tolerant fallbacks."""

    def __init__(self, database_path: str | Path) -> None:
        self._database_path = Path(database_path)

    def query_recent_text(self, prompt: str, limit: int = 5) -> str:
        if not self._database_path.exists():
            return "No local OCR history is available yet."

        try:
            with closing(sqlite3.connect(self._database_path)) as connection:
                tables = self._get_tables(connection)
                text_rows = self._search_text_tables(connection, tables=tables, prompt=prompt, limit=limit)
        except sqlite3.DatabaseError:
            return "OCR history database is present but unreadable."

        if not text_rows:
            return "No matching OCR history was found."
        return "\n".join(text_rows)

    def recent_history(self, limit: int = 8) -> str:
        return self.query_recent_text(prompt="", limit=limit)

    def _get_tables(self, connection: sqlite3.Connection) -> list[str]:
        rows = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        return [str(name) for (name,) in rows]

    def _search_text_tables(
        self,
        connection: sqlite3.Connection,
        tables: list[str],
        prompt: str,
        limit: int,
    ) -> list[str]:
        prompt_terms = [term for term in prompt.lower().split() if term]
        collected: list[str] = []
        for table in tables:
            columns = self._get_columns(connection, table)
            text_columns = [column for column in columns if any(token in column.lower() for token in ("text", "ocr", "content", "window", "title"))]
            if not text_columns:
                continue
            column_sql = ", ".join(f'"{column}"' for column in text_columns)
            try:
                rows = connection.execute(f'SELECT {column_sql} FROM "{table}" ORDER BY ROWID DESC LIMIT 50').fetchall()
            except sqlite3.DatabaseError:
                continue
            for row in rows:
                text = self._flatten_row(row)
                lowered = text.lower()
                if not prompt_terms or any(term in lowered for term in prompt_terms):
                    collected.append(f"[{table}] {text}")
                    if len(collected) >= limit:
                        return collected
        return collected

    def _get_columns(self, connection: sqlite3.Connection, table: str) -> list[str]:
        rows = connection.execute(f'PRAGMA table_info("{table}")').fetchall()
        return [str(row[1]) for row in rows]

    def _flatten_row(self, row: tuple[Any, ...]) -> str:
        values: list[str] = []
        for value in row:
            if value is None:
                continue
            if isinstance(value, (dict, list)):
                values.append(json.dumps(value))
            else:
                values.append(str(value))
        return " | ".join(values)
