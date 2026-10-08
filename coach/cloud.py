"""Shared state for the public (guest-mode) deployment.

In guest mode every visitor's progress lives in a throwaway SQLite file for
their browser session only (see coach.guest). The only things kept in
Postgres (Neon) are the parts every visitor shares:

- AI-generated vocabulary questions ("Fresh questions"),
- fetched reading passages, with their pictures,
- the app-wide count of AI actions used today, so a daily cap holds across
  every visitor and every app restart.

No learner progress, answers or essays are ever written here.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import psycopg

from coach.storage import normalize_questions

SCHEMA = """
CREATE TABLE IF NOT EXISTS shared_questions (
    id TEXT PRIMARY KEY, payload TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now());
CREATE TABLE IF NOT EXISTS shared_passages (
    id TEXT PRIMARY KEY, topic TEXT NOT NULL, title TEXT NOT NULL, bundle TEXT NOT NULL,
    image BYTEA, image_ext TEXT, created_at TIMESTAMPTZ NOT NULL DEFAULT now());
CREATE TABLE IF NOT EXISTS ai_usage (
    day DATE PRIMARY KEY, used INTEGER NOT NULL DEFAULT 0);
"""

# Days are counted in UTC so every visitor shares the same "today".
_TODAY = "(now() AT TIME ZONE 'UTC')::date"


class SharedContent:
    """Thin wrapper around the Neon/Postgres database. Opens one short
    connection per call, like Store does for SQLite."""

    def __init__(self, url: str, connect_timeout: int = 20):
        self.url = url
        self.connect_timeout = connect_timeout

    def connect(self):
        return psycopg.connect(self.url, connect_timeout=self.connect_timeout)

    def ensure_schema(self):
        with self.connect() as db:
            for statement in filter(str.strip, SCHEMA.split(";")):
                db.execute(statement)

    # --- shared content ---

    def save_questions(self, questions: list[dict]) -> int:
        rows = [(q["id"], json.dumps(q, ensure_ascii=False)) for q in normalize_questions(questions)]
        if not rows:
            return 0
        with self.connect() as db:
            before = db.execute("SELECT count(*) FROM shared_questions").fetchone()[0]
            with db.cursor() as cur:
                cur.executemany("INSERT INTO shared_questions(id,payload) VALUES(%s,%s) "
                                "ON CONFLICT (id) DO NOTHING", rows)
            return db.execute("SELECT count(*) FROM shared_questions").fetchone()[0] - before

    def save_passages(self, passages: list[dict]) -> int:
        """Save fetched passage bundles (the dicts fetch_passages returns,
        which Store.add_passages also accepts)."""
        rows = []
        for p in passages:
            bundle = {k: v for k, v in p.items() if k not in ("image_bytes", "image_path", "payload")}
            image = p.get("image_bytes") if p.get("image_ext") else None
            rows.append((p["id"], p["topic"], p["title"], json.dumps(bundle, ensure_ascii=False),
                         image, p.get("image_ext") if image else None))
        if not rows:
            return 0
        with self.connect() as db:
            before = db.execute("SELECT count(*) FROM shared_passages").fetchone()[0]
            with db.cursor() as cur:
                cur.executemany("INSERT INTO shared_passages(id,topic,title,bundle,image,image_ext) "
                                "VALUES(%s,%s,%s,%s,%s,%s) ON CONFLICT (id) DO NOTHING", rows)
            return db.execute("SELECT count(*) FROM shared_passages").fetchone()[0] - before

    def load(self) -> tuple[list[dict], list[dict]]:
        """All shared questions and passages, ready for Store.add_questions
        and Store.add_passages."""
        with self.connect() as db:
            questions = [json.loads(r[0]) for r in
                         db.execute("SELECT payload FROM shared_questions ORDER BY created_at, id")]
            passages = []
            for bundle, image, ext in db.execute(
                    "SELECT bundle, image, image_ext FROM shared_passages ORDER BY created_at, id"):
                p = json.loads(bundle)
                p["image_bytes"] = bytes(image) if image is not None else None
                p["image_ext"] = ext
                passages.append(p)
        return questions, passages

    def version(self) -> str:
        """Changes whenever shared content is added, so a cached copy built
        from an older version can be told apart from the current one."""
        with self.connect() as db:
            q = db.execute("SELECT count(*), max(created_at) FROM shared_questions").fetchone()
            p = db.execute("SELECT count(*), max(created_at) FROM shared_passages").fetchone()
        return f"q{q[0]}-{q[1]}-p{p[0]}-{p[1]}"

    def seed_from_file(self, path: Path) -> tuple[int, int]:
        """Load a seed file written by scripts/export_content_seed.py. Safe to
        run on every start: rows that already exist are left alone."""
        path = Path(path)
        if not path.exists():
            return 0, 0
        seed = json.loads(path.read_text(encoding="utf-8"))
        passages = []
        for p in seed.get("passages", []):
            p = dict(p)
            image = p.pop("image_base64", None)
            p["image_bytes"] = base64.b64decode(image) if image else None
            passages.append(p)
        return self.save_questions(seed.get("questions", [])), self.save_passages(passages)

    # --- daily AI allowance ---

    def ai_used_today(self) -> int:
        with self.connect() as db:
            row = db.execute(f"SELECT used FROM ai_usage WHERE day = {_TODAY}").fetchone()
        return row[0] if row else 0

    def reserve_ai_action(self, limit: int | None) -> bool:
        """Count one AI action against today's app-wide allowance. Returns
        False, counting nothing, if the allowance is already used up. A
        `limit` of None means no cap (the owner); the action is still
        counted. Atomic, so two visitors can't both take the last one."""
        with self.connect() as db:
            if limit is None:
                db.execute(f"INSERT INTO ai_usage(day, used) VALUES ({_TODAY}, 1) "
                           "ON CONFLICT (day) DO UPDATE SET used = ai_usage.used + 1")
                return True
            if limit <= 0:
                return False
            row = db.execute(
                f"INSERT INTO ai_usage(day, used) VALUES ({_TODAY}, 1) "
                "ON CONFLICT (day) DO UPDATE SET used = ai_usage.used + 1 "
                "WHERE ai_usage.used < %s RETURNING used", (limit,)).fetchone()
            return row is not None

    def release_ai_action(self):
        """Give back an action whose AI call failed before producing anything."""
        with self.connect() as db:
            db.execute(f"UPDATE ai_usage SET used = used - 1 WHERE day = {_TODAY} AND used > 0")
