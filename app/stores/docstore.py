import json
import sqlite3
from pathlib import Path

from app.schemas import Chunk

FILENAME = "docstore.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    id          INTEGER PRIMARY KEY,
    chunk_id    TEXT UNIQUE NOT NULL,
    doc_id      TEXT NOT NULL,
    sop_number  TEXT,
    title       TEXT NOT NULL,
    section     TEXT NOT NULL,
    chunk_idx   INTEGER NOT NULL,
    text        TEXT NOT NULL,
    embed_text  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_doc_pos ON chunks(doc_id, chunk_idx);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""

_COLS = "id, chunk_id, doc_id, sop_number, title, section, chunk_idx, text, embed_text"


class DocStore:
    def __init__(self, path: Path, read_only: bool = False):
        uri = f"file:{path.as_posix()}{'?mode=ro' if read_only else ''}"
        self.conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
        if not read_only:
            self.conn.executescript(_SCHEMA)

    @classmethod
    def open(cls, directory: Path, read_only: bool = True) -> "DocStore":
        return cls(directory / FILENAME, read_only=read_only)

    def add_chunks(self, chunks: list[Chunk]) -> list[int]:
        with self.conn:
            self.conn.executemany(
                "INSERT INTO chunks (chunk_id, doc_id, sop_number, title, section, chunk_idx, text, embed_text)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [(c.chunk_id, c.doc_id, c.sop_number, c.title, c.section, c.chunk_idx, c.text, c.embed_text) for c in chunks],
            )
        return [self.id_for(c.chunk_id) for c in chunks]

    def id_for(self, chunk_id: str) -> int:
        return self.conn.execute("SELECT id FROM chunks WHERE chunk_id = ?", (chunk_id,)).fetchone()[0]

    @staticmethod
    def _row(row) -> tuple[int, Chunk]:
        rid, chunk_id, doc_id, sop, title, section, idx, text, embed_text = row
        return rid, Chunk(
            chunk_id=chunk_id, doc_id=doc_id, sop_number=sop, title=title,
            section=section, chunk_idx=idx, text=text, embed_text=embed_text,
        )

    def get(self, ids: list[int]) -> dict[int, Chunk]:
        if not ids:
            return {}
        marks = ",".join("?" * len(ids))
        rows = self.conn.execute(f"SELECT {_COLS} FROM chunks WHERE id IN ({marks})", ids).fetchall()
        return dict(self._row(r) for r in rows)

    def neighbors(self, doc_id: str, chunk_idx: int, n: int) -> dict[int, Chunk]:
        rows = self.conn.execute(
            f"SELECT {_COLS} FROM chunks WHERE doc_id = ? AND chunk_idx BETWEEN ? AND ? AND chunk_idx != ?",
            (doc_id, chunk_idx - n, chunk_idx + n, chunk_idx),
        ).fetchall()
        return dict(self._row(r) for r in rows)

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def set_meta(self, meta: dict) -> None:
        with self.conn:
            self.conn.executemany(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                [(k, json.dumps(v)) for k, v in meta.items()],
            )

    def get_meta(self) -> dict:
        return {k: json.loads(v) for k, v in self.conn.execute("SELECT key, value FROM meta")}

    def close(self) -> None:
        self.conn.close()
