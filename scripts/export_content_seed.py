"""Export shared content (no learner progress) from a local database into
data/content_seed.json, which the public guest-mode app loads into its
shared database on start.

Exported: AI-generated vocabulary questions that were never flagged, and
fetched reading passages with their pictures. Never exported: attempts,
sessions, reading or writing history, flags, or anything else about the
learner.

    python scripts/export_content_seed.py [path/to/coach.sqlite3]
"""
from __future__ import annotations

import base64
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "local_data" / "coach.sqlite3"
OUTPUT = ROOT / "data" / "content_seed.json"


def export(db_path: Path, output: Path = OUTPUT) -> dict:
    # Read-only, so the learner's database can't be changed by accident.
    db = sqlite3.connect(Path(db_path).resolve().as_uri() + "?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    try:
        questions = [json.loads(r["payload"]) for r in db.execute(
            "SELECT payload FROM questions WHERE source='openai' AND flagged=0 ORDER BY id")]
        passages = []
        for r in db.execute("SELECT * FROM passages ORDER BY fetched_at, id"):
            p = {k: r[k] for k in ("id", "topic", "title", "body", "source_name", "source_url",
                                   "license", "word_count")}
            p["is_excerpt"] = bool(r["is_excerpt"])
            p.update(json.loads(r["payload"]))
            image = Path(r["image_path"]) if r["image_path"] else None
            if image is not None and not image.exists():
                # A stored absolute path from another machine: look beside the DB.
                image = Path(db_path).parent / "reading_images" / image.name
            if image is not None and image.exists():
                p["image_ext"] = image.suffix.lstrip(".")
                p["image_base64"] = base64.b64encode(image.read_bytes()).decode("ascii")
            passages.append(p)
    finally:
        db.close()
    seed = {"version": 1, "questions": questions, "passages": passages}
    output.write_text(json.dumps(seed, ensure_ascii=False, indent=1), encoding="utf-8")
    return seed


if __name__ == "__main__":
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DB
    seed = export(source)
    pictures = sum("image_base64" in p for p in seed["passages"])
    print(f"Wrote {OUTPUT.relative_to(ROOT)}: {len(seed['questions'])} AI questions, "
          f"{len(seed['passages'])} passages ({pictures} with pictures).")
