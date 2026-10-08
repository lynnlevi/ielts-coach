from __future__ import annotations

import csv
import hashlib
import json
import random
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

DATA = Path(__file__).resolve().parent.parent / "data" / "words.psv"
KINDS = {"meaning": "Meaning", "context": "Meaning in context", "cloze": "Sentence completion", "usage": "Natural usage", "collocation": "Word combinations"}


class Question(BaseModel):
    model_config = ConfigDict(extra="forbid")
    word_id: str
    kind: Literal["meaning", "context", "cloze", "usage", "collocation"]
    prompt: str = Field(min_length=10, max_length=1000)
    options: list[str] = Field(min_length=4, max_length=4)
    answer: int = Field(ge=0, le=3)
    explanation: str = Field(min_length=15, max_length=1500)

    @model_validator(mode="after")
    def distinct_options(self):
        if len({o.strip().casefold() for o in self.options}) != 4:
            raise ValueError("A question must have four different answers.")
        if any(not o.strip() or len(o) > 500 for o in self.options):
            raise ValueError("Answer choices must be short and nonempty.")
        return self


def load_words(path: Path = DATA) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        words = list(csv.DictReader(f, delimiter="|"))
    seen = set()
    for w in words:
        w["id"] = w.get("id") or w["word"].lower().replace(" ", "-")
        required = ("word", "topic", "pos", "definition", "example", "cloze", "distractors")
        if any(not w.get(key, "").strip() for key in required):
            raise ValueError(f"Missing vocabulary fields for {w.get('word', 'unnamed entry')}")
        if w["id"] in seen:
            raise ValueError(f"Duplicate word ID: {w['id']}")
        seen.add(w["id"])
        if w["cloze"].count("___") != 1:
            raise ValueError(f"Use exactly one blank for {w['word']}")
        choices = [v.strip() for v in w["distractors"].split(";")]
        if len(choices) != 3 or len({v.casefold() for v in choices + [w["word"]]}) != 4:
            raise ValueError(f"Provide three distinct distractors for {w['word']}")
        w["distractors"] = ";".join(choices)
        w.setdefault("vi", "")
        w.setdefault("collocation", "")
    return words


def append_words(new_rows: list[dict], path: Path = DATA) -> int:
    """Append new vocabulary rows (e.g. student-approved words met while reading)
    to the PSV file. A word whose id already exists (or is duplicated within the
    batch) is silently skipped rather than erroring. The whole resulting file is
    validated with load_words() before the write is committed, via a temp file
    that only replaces the real one on success, so a bad row can never corrupt
    the live vocabulary bank.

    Returns the number of rows actually added.
    """
    fieldnames = ["id", "word", "topic", "pos", "definition", "vi", "example", "collocation", "cloze", "distractors"]
    existing_ids = {w["id"] for w in load_words(path)}
    to_add = []
    for row in new_rows:
        word = (row.get("word") or "").strip()
        if not word:
            continue
        wid = (row.get("id") or "").strip() or word.lower().replace(" ", "-")
        if wid in existing_ids:
            continue
        existing_ids.add(wid)
        to_add.append({field: str(row.get(field, "")).strip() for field in fieldnames} | {"id": wid, "word": word})
    if not to_add:
        return 0
    with path.open(encoding="utf-8") as f:
        current_text = f.read()
    tmp_path = path.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as f:
        f.write(current_text)
        if not current_text.endswith("\n"):
            f.write("\n")
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="|", lineterminator="\n")
        writer.writerows(to_add)
    load_words(tmp_path)  # raises ValueError and leaves the real file untouched if anything is wrong
    tmp_path.replace(path)
    return len(to_add)


# Keep close meanings out of the same four-choice question, including fallbacks.
SIMILAR_MEANINGS = [
    "array myriad", "defense shield", "environment habitat", "inhibit hinder hamper complicate",
    "extend expand", "protect preserve conserve", "vanish disappear", "breed generate manufacture",
    "endure tolerate suffer bear", "evolve develop unfold", "fascinate appeal entice lure draw",
    "imperative requisite", "inhabit occupy settle", "optimal suitable desirable",
    "diverse disparate", "accumulate increase", "determine ascertain", "minimize reduce",
    "resilient rugged", "thrive survive", "blur indistinct obscure", "anticipate estimate project",
    "coordinate regulate", "demonstrate display indicate portray", "scan detect monitor",
    "exotic unique novel", "found institute", "grandeur marvel", "massive immense vast",
    "permanently sustained prolonged", "spectator observer", "talent skill", "venue site destination",
    "crucial vital critical essential indispensable invaluable", "rejuvenate revive", "passive solitary",
    "drawback flaw", "freight product", "portable compact", "tricky complex sophisticated",
    "innovation invention", "monetary financial", "centerpiece epicenter", "rival compete",
    "spatial peripheral", "flair taste", "charge voltage current", "consume utilize",
    "standard conventional", "adopt embrace", "creator inventor", "structure construction",
    "deed practice", "image illusion", "abstract theoretical", "considerably markedly profoundly",
    "effectively reliably", "gesture maneuver", "renowned popular prominent", "sharpen hone enhance",
    "shortage absence", "retain preserve", "epidemic disorder dementia", "gravity pressure stressor",
    "counteract combat alleviate", "prevent avoid", "link bond contact", "stimulate fuel",
    "investigation inquiry research", "toxic catastrophic", "precisely specifically", "network community",
    "benefit upside boon", "delicate vulnerable", "concept philosophy principle", "inhabit occupy",
    "outcome consequence", "acquire learn", "costly luxury", "economical affordable efficient",
    "supervision guidance instruction", "characteristic aspect", "product staple", "edge advantage",
    "personality reputation status", "passion thirst motivation", "preponderance bulk majority",
    "continuously consistently routinely", "firm conglomerate", "immense inordinate overwhelming",
    "approximately relatively moderately", "devote dedicate", "opponent rival", "persist strive struggle",
    "inquisitiveness curiosity", "solitary isolation withdrawal", "curriculum content", "confidence loyalty",
    "compulsory obligatory", "enriched enhanced", "exceptional extraordinary gifted", "slope pitch ramp",
    "distracting disruptive", "profitable wealthy", "disability impaired incapacitated", "facial physical",
]


def choose_distractor_words(word: dict, words: list[dict]) -> list[dict]:
    excluded = {word["word"]}
    for group in SIMILAR_MEANINGS:
        members = set(group.split())
        if word["word"] in members:
            excluded.update(members)
    eligible = [w for w in words if w["word"] not in excluded and w["pos"] == word["pos"]
                and w["definition"].casefold() != word["definition"].casefold()]
    same_topic = [w for w in eligible if w["topic"] == word["topic"]]
    fallback = [w for w in eligible if w not in same_topic]
    rng = random.Random(word["id"])
    rng.shuffle(same_topic)
    rng.shuffle(fallback)
    pool = same_topic + fallback
    if len(pool) < 3:
        pool.extend(w for w in eligible if w not in pool)
    if len(pool) < 3:
        pool.extend(w for w in words if w not in pool and w["word"] not in excluded
                    and w["definition"].casefold() != word["definition"].casefold())
    if len(pool) < 3:
        raise ValueError(f"Not enough distinct choices for {word['word']}")
    return pool[:3]


def question_id(q: dict) -> str:
    # Choice order is not part of identity: reshuffling cannot earn new credit.
    content = [q["word_id"], q["kind"], " ".join(q["prompt"].casefold().split())]
    return hashlib.sha256(json.dumps(content).encode()).hexdigest()[:24]


def seed_questions(words: list[dict]) -> list[dict]:
    result = []
    for word in words:
        distractors = choose_distractor_words(word, words)
        explanation = f"{word['word'].capitalize()} means {word['definition']}. Example: {word['example']}"
        specs = [
            ("meaning", f"What does “{word['word']}” mean as a {word['pos']}?", [word["definition"]] + [w["definition"] for w in distractors]),
            ("context", f"{word['example']}\n\n" + f"Which meaning of “{word['word']}” ({word['pos']}) is used here?", [word["definition"]] + [w["definition"] for w in reversed(distractors)]),
            ("cloze", f"Choose the word or phrase that best completes the sentence.\n\n{word['cloze']}" + f"\n\nMeaning hint: {word['definition']} ({word['pos']}).", [word["word"]] + word["distractors"].split(";")),
        ]
        for kind, prompt, options in specs:
            q = Question(word_id=word["id"], kind=kind, prompt=prompt, options=options, answer=0, explanation=explanation).model_dump()
            q.update(id=question_id(q), source="offline")
            result.append(q)
    return result
