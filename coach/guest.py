"""Throwaway per-visitor stores for guest mode.

Each visitor's browser session gets its own SQLite Store, so their answers
never mix with anyone else's and disappear when the session ends. Building
one from scratch (validating thousands of starter questions, decoding
passage pictures) takes a few seconds, so a ready-made template database is
built once per content version and each visitor gets a fast file copy of it.
"""
from __future__ import annotations

import shutil
import time
import uuid
from pathlib import Path

from coach.storage import Store

GUEST_MAX_AGE_HOURS = 24
TEMPLATE_MAX_AGE_HOURS = 48


def build_template(folder: Path, starter_questions: list[dict], essay_prompts: list[dict],
                   shared_questions: list[dict], shared_passages: list[dict]) -> Path:
    """Build a template database holding the shared content and no progress.
    Passage pictures are written next to it and referenced by absolute path,
    so every copy can show them without duplicating the files."""
    folder = Path(folder)
    path = folder / "coach.sqlite3"
    if path.exists():
        return path
    building = folder.with_name(folder.name + f".building-{uuid.uuid4().hex[:8]}")
    store = Store(building / "coach.sqlite3")
    store.add_questions(starter_questions)
    store.add_essay_prompts(essay_prompts)
    store.add_questions(shared_questions)
    store.add_passages(shared_passages)
    with store.connection() as db:
        # Point picture paths at the final folder name, then move into place.
        db.execute("UPDATE passages SET image_path = replace(image_path, ?, ?)",
                   (str(building), str(folder)))
    try:
        building.rename(folder)
    except OSError:
        # Another session finished building the same version first.
        shutil.rmtree(building, ignore_errors=True)
    return path


def new_guest_store(template_db: Path, guests_root: Path) -> Store:
    """A fresh Store for one visitor, copied from the template."""
    guests_root = Path(guests_root)
    remove_stale(guests_root, GUEST_MAX_AGE_HOURS)
    folder = guests_root / uuid.uuid4().hex
    folder.mkdir(parents=True)
    shutil.copyfile(template_db, folder / "coach.sqlite3")
    return Store(folder / "coach.sqlite3")


def remove_stale(root: Path, max_age_hours: float, keep: tuple[Path, ...] = ()):
    """Delete subfolders of `root` untouched for longer than `max_age_hours`
    (a guest whose session has long ended, or an outdated template)."""
    root = Path(root)
    if not root.exists():
        return
    cutoff = time.time() - max_age_hours * 3600
    keep = {Path(k).resolve() for k in keep}
    for child in root.iterdir():
        if not child.is_dir() or child.resolve() in keep:
            continue
        newest = max([child.stat().st_mtime] + [p.stat().st_mtime for p in child.rglob("*")])
        if newest < cutoff:
            shutil.rmtree(child, ignore_errors=True)
