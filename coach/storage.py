from __future__ import annotations

import json
import random
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path

from coach.content import Question, question_id, seed_questions, append_words
from coach.reading import FREE_TEXT_KINDS, PassageBundle, normalize_for_matching
from coach.writing import EssayEvaluation, overall_band


def now_iso():
    # Store the local offset with timestamps; dates follow the learner's computer.
    return datetime.now().astimezone().isoformat(timespec="microseconds")


# Review interval (in days) by rung, indexed by the number of distinct days
# the current correct streak spans. The first three rungs (1/3/7) match the
# app's original fixed Learning/Familiar/Secure intervals, so a word still
# earns its first few reviews quickly. Beyond a week, growth also requires
# the streak to have been tested across at least two question kinds (see
# below) -- a word only earns a long break once it has shown real, varied
# mastery rather than one lucky run of the same easy format. Capped at 90
# days so a "mastered" word still resurfaces at least once before a learner
# would plausibly sit a real exam.
_REVIEW_LADDER = (1, 3, 7, 14, 30, 60, 90)


def mastery(attempts: list[dict], today: str) -> dict:
    """Derive recognition mastery from persisted, unflagged first daily attempts."""
    if not attempts:
        return dict(stage="New", due=None, correct=0, total=0, skills={}, days=0)
    seen = set()
    evidence = []
    for a in sorted(attempts, key=lambda x: x["answered_at"]):
        key = (a["question_id"], a["answered_at"][:10])
        if key not in seen:
            evidence.append(a)
            seen.add(key)
    streak = []
    skills = {}
    for a in evidence:
        s = skills.setdefault(a["kind"], {"correct": 0, "total": 0})
        s["total"] += 1
        s["correct"] += int(a["correct"])
        if a["correct"]:
            streak.append(a)
        else:
            streak = []
    days = len({a["answered_at"][:10] for a in streak})
    types = {a["kind"] for a in streak}
    # Meaning and meaning-in-context count as one recognition skill.
    types = {"meaning" if k == "context" else k for k in types}
    if not evidence[-1]["correct"]:
        # A miss drops the word back to the bottom rung, shown as Learning.
        interval = 1
    else:
        # Growth past a week needs 2+ distinct skills demonstrated; until
        # then (or always, for the first three rungs) it is capped there.
        rung = min(days, 7) if len(types) >= 2 else min(days, 3)
        interval = _REVIEW_LADDER[rung - 1]
    stage = "Learning" if interval <= 3 else "Familiar" if interval == 7 else "Secure"
    last = datetime.fromisoformat(evidence[-1]["answered_at"])
    due = (last.date() + timedelta(days=interval)).isoformat()
    return dict(stage=stage, due=due, correct=sum(int(a["correct"]) for a in evidence),
                total=len(evidence), skills=skills, days=days)


def normalize_questions(questions: list[dict]) -> list[dict]:
    """Validate question dicts and give each its content-hash id and source,
    exactly as they are saved (shared with coach.cloud, so a question saved
    online gets the same id it would get locally)."""
    validated = []
    for q in questions:
        data = Question.model_validate({k: q[k] for k in Question.model_fields}).model_dump()
        data.update(id=question_id(data), source=q.get("source", "openai"))
        validated.append(data)
    return validated


class Store:
    """One learner per database. Open connections per operation, not per UI session."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS questions (
                    id TEXT PRIMARY KEY, word_id TEXT NOT NULL, payload TEXT NOT NULL,
                    source TEXT NOT NULL, flagged INTEGER NOT NULL DEFAULT 0);
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY, created_at TEXT NOT NULL, payload TEXT NOT NULL,
                    studied INTEGER NOT NULL DEFAULT 0, finished INTEGER NOT NULL DEFAULT 0);
                CREATE TABLE IF NOT EXISTS attempts (
                    id TEXT PRIMARY KEY, session_id TEXT NOT NULL,
                    question_id TEXT NOT NULL REFERENCES questions(id), word_id TEXT NOT NULL,
                    kind TEXT NOT NULL, selected TEXT NOT NULL, correct INTEGER NOT NULL,
                    answered_at TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS attempts_word ON attempts(word_id);
                CREATE TABLE IF NOT EXISTS reports (
                    question_id TEXT PRIMARY KEY REFERENCES questions(id), reported_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS generation_log (
                    id TEXT PRIMARY KEY, created_at TEXT NOT NULL, model TEXT NOT NULL,
                    question_count INTEGER NOT NULL, input_tokens INTEGER, output_tokens INTEGER);
                CREATE TABLE IF NOT EXISTS flashcard_activity (
                    presentation_id TEXT PRIMARY KEY, word_id TEXT NOT NULL,
                    outcome TEXT NOT NULL CHECK(outcome IN ('review','skip')),
                    occurred_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS passages (
                    id TEXT PRIMARY KEY, topic TEXT NOT NULL, title TEXT NOT NULL, body TEXT NOT NULL,
                    source_name TEXT NOT NULL, source_url TEXT NOT NULL, license TEXT NOT NULL,
                    is_excerpt INTEGER NOT NULL DEFAULT 0, word_count INTEGER NOT NULL,
                    image_path TEXT, payload TEXT NOT NULL, fetched_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS reading_sessions (
                    id TEXT PRIMARY KEY, passage_id TEXT NOT NULL REFERENCES passages(id),
                    created_at TEXT NOT NULL, finished INTEGER NOT NULL DEFAULT 0, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS reading_activity (
                    id TEXT PRIMARY KEY, passage_id TEXT NOT NULL, topic TEXT NOT NULL,
                    completed_at TEXT NOT NULL, session_id TEXT);
                CREATE TABLE IF NOT EXISTS essay_prompts (
                    id TEXT PRIMARY KEY, topic TEXT NOT NULL, essay_type TEXT NOT NULL,
                    prompt TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS writing_attempts (
                    id TEXT PRIMARY KEY, prompt_id TEXT NOT NULL REFERENCES essay_prompts(id),
                    mode TEXT NOT NULL CHECK(mode IN ('simulation','practice')),
                    essay_text TEXT NOT NULL, deadline_hit INTEGER NOT NULL DEFAULT 0,
                    time_taken_seconds INTEGER, evaluation TEXT NOT NULL,
                    overall_band REAL NOT NULL, created_at TEXT NOT NULL);
                PRAGMA user_version = 6;
            """)
            passage_cols = {r["name"] for r in db.execute("PRAGMA table_info(passages)").fetchall()}
            if "image_path" not in passage_cols:
                db.execute("ALTER TABLE passages ADD COLUMN image_path TEXT")
            activity_cols = {r["name"] for r in db.execute("PRAGMA table_info(reading_activity)").fetchall()}
            if "session_id" not in activity_cols:
                # Lets a completed reading logged before this column existed be told
                # apart from one whose written answers can still be looked up.
                db.execute("ALTER TABLE reading_activity ADD COLUMN session_id TEXT")
            for row in db.execute("SELECT id,payload FROM questions WHERE source NOT IN ('offline','openai')").fetchall():
                payload = json.loads(row["payload"])
                payload["source"] = "offline"
                db.execute("UPDATE questions SET source='offline',payload=? WHERE id=?", (json.dumps(payload, ensure_ascii=False), row["id"]))
            for row in db.execute("SELECT id,payload FROM sessions").fetchall():
                payload = json.loads(row["payload"])
                changed = "lesson" in payload
                payload.pop("lesson", None)
                for q in payload.get("questions", []):
                    if q.get("source") not in ("offline", "openai"):
                        q["source"] = "offline"
                        changed = True
                if changed:
                    db.execute("UPDATE sessions SET payload=? WHERE id=?", (json.dumps(payload, ensure_ascii=False), row["id"]))

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=15)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys = ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def add_questions(self, questions: list[dict]) -> int:
        validated = normalize_questions(questions)
        with self.connection() as db:
            before = db.total_changes
            db.executemany("INSERT OR IGNORE INTO questions(id,word_id,payload,source) VALUES(?,?,?,?)",
                           [(q["id"], q["word_id"], json.dumps(q, ensure_ascii=False), q["source"]) for q in validated])
            return db.total_changes - before

    def questions(self, include_flagged=False):
        with self.connection() as db:
            rows = db.execute("SELECT payload,flagged FROM questions" + ("" if include_flagged else " WHERE flagged=0")).fetchall()
        return [dict(json.loads(r["payload"]), flagged=bool(r["flagged"])) for r in rows]

    def attempts(self, include_flagged=False):
        with self.connection() as db:
            return [dict(r) for r in db.execute("""SELECT a.*,q.flagged FROM attempts a
                JOIN questions q ON q.id=a.question_id """ +
                ("" if include_flagged else " WHERE q.flagged=0 ") + " ORDER BY answered_at,id")]

    def active_questions(self, words):
        """Keep superseded content for history, but use only the current bank."""
        ids = {w["id"] for w in words}
        current = {q["id"] for q in seed_questions(words)}
        return [q for q in self.questions() if q["word_id"] in ids
                and (q["source"] == "openai" or q["id"] in current)]

    def progress(self, words, today=None):
        today = today or now_iso()[:10]
        grouped = {}
        for a in self.attempts():
            grouped.setdefault(a["word_id"], []).append(a)
        return {w["id"]: mastery(grouped.get(w["id"], []), today) for w in words}

    def create_session(self, words, topic="All topics", length=10, today=None, rng=None):
        """Build a quiz of up to `length` questions, each testing a different word.

        Due reviews are chosen first (oldest due date first), then words the
        learner has never seen, then anything else eligible as filler once
        those two pools run out. A word already introduced as "new" in an
        earlier session today is not introduced again, so the same fresh
        word cannot appear as a first-time question in two sessions the same
        day. There is no longer a fixed daily cap on how many new words a
        single session (or the day as a whole) may introduce.
        """
        today = today or now_iso()[:10]
        rng = rng or random.Random()
        progress = self.progress(words, today)
        eligible = [w for w in words if topic == "All topics" or w["topic"] == topic]
        bank = self.active_questions(words)
        available = {q["word_id"] for q in bank}
        eligible = [w for w in eligible if w["id"] in available]
        with self.connection() as db:
            sessions = db.execute("SELECT payload FROM sessions WHERE substr(created_at,1,10)=?", (today,)).fetchall()
        introduced_today = {wid for row in sessions for wid in json.loads(row["payload"])["new_words"]}
        due = [w for w in eligible if progress[w["id"]]["due"] and progress[w["id"]]["due"] <= today]
        due.sort(key=lambda w: progress[w["id"]]["due"])
        due_ids = {w["id"] for w in due}
        new = [w for w in eligible if progress[w["id"]]["stage"] == "New" and w["id"] not in introduced_today
               and w["id"] not in due_ids]
        new_ids = {w["id"] for w in new}
        rest = [w for w in eligible if w["id"] not in due_ids and w["id"] not in new_ids]
        chosen, chosen_ids, freshly_introduced = [], set(), set()
        for pool, is_new in ((due, False), (new, True), (rest, False)):
            for w in pool:
                if len(chosen) >= length:
                    break
                if w["id"] in chosen_ids:
                    continue
                chosen.append(w)
                chosen_ids.add(w["id"])
                if is_new:
                    freshly_introduced.add(w["id"])
        if not chosen:
            return None
        prior = {a["question_id"]: a["answered_at"] for a in self.attempts()}
        selected = []
        for w in chosen:
            qs = [q for q in bank if q["word_id"] == w["id"]]
            rng.shuffle(qs)
            qs.sort(key=lambda q: prior.get(q["id"], ""))
            q = dict(qs[0])
            right = q["options"][q["answer"]]
            options = list(q["options"])
            rng.shuffle(options)
            q.update(options=options, answer=options.index(right))
            selected.append(q)
        payload = dict(questions=selected, new_words=sorted(freshly_introduced), topic=topic,
                       words={w["id"]: w for w in chosen})
        sid = uuid.uuid4().hex
        with self.connection() as db:
            db.execute("INSERT INTO sessions(id,created_at,payload) VALUES(?,?,?)", (sid, now_iso(), json.dumps(payload)))
        return sid

    def session(self, sid=None):
        with self.connection() as db:
            row = db.execute("SELECT * FROM sessions WHERE id=?", (sid,)).fetchone() if sid else db.execute(
                "SELECT * FROM sessions WHERE finished=0 ORDER BY created_at DESC LIMIT 1").fetchone()
        if not row:
            return None
        return dict(row) | {"payload": json.loads(row["payload"])}

    def mark_studied(self, sid):
        with self.connection() as db:
            db.execute("UPDATE sessions SET studied=1 WHERE id=?", (sid,))

    def finish(self, sid):
        with self.connection() as db:
            db.execute("UPDATE sessions SET finished=1 WHERE id=?", (sid,))

    def session_attempts(self, sid):
        return {a["question_id"]: a for a in self.attempts(include_flagged=True) if a["session_id"] == sid}

    def record(self, sid, qid, selected, timestamp=None):
        session = self.session(sid)
        if not session or session["finished"]:
            raise ValueError("This practice session is no longer active.")
        if qid not in {q["id"] for q in session["payload"]["questions"]}:
            raise ValueError("Question does not belong to this session.")
        with self.connection() as db:
            row = db.execute("SELECT * FROM questions WHERE id=?", (qid,)).fetchone()
            if row["flagged"]:
                raise ValueError("This question has been excluded from scoring.")
            q = json.loads(row["payload"])
            if selected not in q["options"]:
                raise ValueError("Choose one of the available answers.")
            # The immutable bank answer, not browser state, determines correctness.
            db.execute("""INSERT OR IGNORE INTO attempts
                (id,session_id,question_id,word_id,kind,selected,correct,answered_at)
                VALUES(?,?,?,?,?,?,?,?)""", (f"{sid}:{qid}", sid, qid, q["word_id"], q["kind"], selected,
                    int(selected == q["options"][q["answer"]]), timestamp or now_iso()))
            return dict(db.execute("SELECT * FROM attempts WHERE id=?", (f"{sid}:{qid}",)).fetchone())

    def flag(self, qid):
        with self.connection() as db:
            db.execute("UPDATE questions SET flagged=1 WHERE id=?", (qid,))
            db.execute("INSERT OR IGNORE INTO reports VALUES(?,?)", (qid, now_iso()))

    def log_generation(self, model, count, usage):
        with self.connection() as db:
            db.execute("INSERT INTO generation_log VALUES(?,?,?,?,?,?)", (uuid.uuid4().hex, now_iso(), model, count, usage.get("input_tokens"), usage.get("output_tokens")))

    def record_flashcard(self, presentation_id, word_id, outcome, timestamp=None):
        if outcome not in ("review", "skip"):
            raise ValueError("Unknown flashcard outcome.")
        with self.connection() as db:
            db.execute("INSERT OR IGNORE INTO flashcard_activity VALUES(?,?,?,?)",
                       (presentation_id, word_id, outcome, timestamp or now_iso()))

    def flashcard_activity(self):
        with self.connection() as db:
            return [dict(row) for row in db.execute(
                "SELECT * FROM flashcard_activity ORDER BY occurred_at")]

    def flashcards_reviewed(self, today=None):
        with self.connection() as db:
            return db.execute("SELECT count(*) FROM flashcard_activity WHERE outcome='review' "
                              "AND substr(occurred_at,1,10)=?", (today or now_iso()[:10],)).fetchone()[0]

    # --- Reading ---

    def add_passages(self, passages: list[dict]) -> int:
        """Save fetched passage bundles. Each bundle's comprehension/vocabulary
        shape is re-validated here as a safety net, independent of whatever
        validation already happened when it was generated."""
        validated = []
        for p in passages:
            bundle = PassageBundle.model_validate({
                "comprehension": p.get("comprehension", []),
                "vocabulary": p.get("vocabulary", []),
            })
            payload = dict(comprehension=[q.model_dump() for q in bundle.comprehension],
                           vocabulary=[v.model_dump() for v in bundle.vocabulary])
            validated.append(dict(p, payload=payload))
        images_dir = self.path.parent / "reading_images"
        with self.connection() as db:
            existing_ids = {r["id"] for r in db.execute("SELECT id FROM passages").fetchall()}
            before = db.total_changes
            rows = []
            for p in validated:
                image_path = None
                if p["id"] not in existing_ids and p.get("image_bytes") and p.get("image_ext"):
                    images_dir.mkdir(parents=True, exist_ok=True)
                    file_path = images_dir / f"{p['id']}.{p['image_ext']}"
                    file_path.write_bytes(p["image_bytes"])
                    image_path = str(file_path)
                rows.append((p["id"], p["topic"], p["title"], p["body"], p["source_name"], p["source_url"], p["license"],
                            int(p.get("is_excerpt", False)), p["word_count"], image_path,
                            json.dumps(p["payload"], ensure_ascii=False), now_iso()))
            db.executemany("""INSERT OR IGNORE INTO passages
                (id,topic,title,body,source_name,source_url,license,is_excerpt,word_count,image_path,payload,fetched_at)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""", rows)
            return db.total_changes - before

    def passages(self, topic="All topics"):
        with self.connection() as db:
            if topic == "All topics":
                rows = db.execute("SELECT * FROM passages").fetchall()
            else:
                rows = db.execute("SELECT * FROM passages WHERE topic=?", (topic,)).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            payload = json.loads(d.pop("payload"))
            d.update(payload)
            d["is_excerpt"] = bool(d["is_excerpt"])
            result.append(d)
        return result

    def passage(self, passage_id):
        with self.connection() as db:
            row = db.execute("SELECT * FROM passages WHERE id=?", (passage_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        payload = json.loads(d.pop("payload"))
        d.update(payload)
        d["is_excerpt"] = bool(d["is_excerpt"])
        return d

    def choose_passage(self, topic="All topics", rng=None):
        """Pick a passage for `topic`, preferring one the learner has never
        completed; if every passage in the pool has already been read, allow
        a reread rather than returning nothing."""
        rng = rng or random.Random()
        pool = self.passages(topic)
        if not pool:
            return None
        with self.connection() as db:
            completed = {r["passage_id"] for r in db.execute("SELECT DISTINCT passage_id FROM reading_activity")}
        candidates = [p for p in pool if p["id"] not in completed] or pool
        return rng.choice(candidates)

    def start_reading_session(self, passage_id):
        payload = dict(passage_id=passage_id, comprehension_answers={}, vocab_chosen=[])
        sid = uuid.uuid4().hex
        with self.connection() as db:
            db.execute("INSERT INTO reading_sessions(id,passage_id,created_at,payload) VALUES(?,?,?,?)",
                       (sid, passage_id, now_iso(), json.dumps(payload)))
        return sid

    def reading_session(self, sid=None):
        with self.connection() as db:
            row = db.execute("SELECT * FROM reading_sessions WHERE id=?", (sid,)).fetchone() if sid else db.execute(
                "SELECT * FROM reading_sessions WHERE finished=0 ORDER BY created_at DESC LIMIT 1").fetchone()
        if not row:
            return None
        return dict(row) | {"payload": json.loads(row["payload"])}

    def _update_reading_payload(self, sid, mutate):
        with self.connection() as db:
            row = db.execute("SELECT payload FROM reading_sessions WHERE id=?", (sid,)).fetchone()
            if not row:
                raise ValueError("This reading session no longer exists.")
            payload = json.loads(row["payload"])
            mutate(payload)
            db.execute("UPDATE reading_sessions SET payload=? WHERE id=?", (json.dumps(payload, ensure_ascii=False), sid))

    def record_comprehension_answer(self, sid, index, selected):
        session = self.reading_session(sid)
        if not session or session["finished"]:
            raise ValueError("This reading session is no longer active.")
        passage = self.passage(session["passage_id"])
        if not passage:
            raise ValueError("The passage for this session could not be found.")
        q = passage["comprehension"][index]
        if q.get("kind") in FREE_TEXT_KINDS:
            cleaned = (selected or "").strip()
            if not cleaned:
                raise ValueError("Type an answer before continuing.")
            max_words = q.get("max_words") or 5
            if len(cleaned.split()) > max_words:
                raise ValueError(f"Answer must be {max_words} word(s) or fewer.")
            correct = normalize_for_matching(cleaned) == normalize_for_matching(q["answer_text"])
        else:
            if selected not in q["options"]:
                raise ValueError("Choose one of the available answers.")
            correct = selected == q["options"][q["answer"]]

        def mutate(payload):
            payload["comprehension_answers"][str(index)] = dict(selected=selected, correct=correct)
        self._update_reading_payload(sid, mutate)
        return correct

    def add_vocab_words(self, sid, indices: list[int]) -> int:
        """Append the chosen candidate words from this session's passage to
        data/words.psv, tagged with the passage's topic. Returns how many were
        actually added (a word whose id already exists is skipped, not an error)."""
        session = self.reading_session(sid)
        if not session:
            raise ValueError("This reading session no longer exists.")
        passage = self.passage(session["passage_id"])
        if not passage:
            raise ValueError("The passage for this session could not be found.")
        rows = [dict(passage["vocabulary"][i], topic=passage["topic"])
                for i in indices if 0 <= i < len(passage["vocabulary"])]
        added = append_words(rows)

        def mutate(payload):
            payload["vocab_chosen"] = sorted(set(payload.get("vocab_chosen", [])) | set(indices))
        self._update_reading_payload(sid, mutate)
        return added

    def finish_reading_session(self, sid):
        session = self.reading_session(sid)
        if not session or session["finished"]:
            return
        passage = self.passage(session["passage_id"])
        with self.connection() as db:
            db.execute("UPDATE reading_sessions SET finished=1 WHERE id=?", (sid,))
            db.execute("INSERT INTO reading_activity(id,passage_id,topic,completed_at,session_id) VALUES(?,?,?,?,?)",
                       (uuid.uuid4().hex, session["passage_id"], passage["topic"] if passage else "", now_iso(), sid))

    def reading_activity(self):
        with self.connection() as db:
            return [dict(r) for r in db.execute("SELECT * FROM reading_activity ORDER BY completed_at")]

    def reading_stats(self, today=None):
        """Habit-formation metrics, deliberately not score-based: total passages
        completed, distinct topics explored, the current consecutive-day streak
        ending today, and how many were completed in the last 7 days."""
        today = today or now_iso()[:10]
        rows = self.reading_activity()
        days = {r["completed_at"][:10] for r in rows}
        streak = 0
        cursor = datetime.fromisoformat(today).date()
        while cursor.isoformat() in days:
            streak += 1
            cursor -= timedelta(days=1)
        week_cutoff = (datetime.fromisoformat(today).date() - timedelta(days=6)).isoformat()
        return dict(total_passages=len(rows), topics_explored=len({r["topic"] for r in rows}),
                    streak_days=streak, this_week=sum(1 for r in rows if r["completed_at"][:10] >= week_cutoff))

    # --- Writing ---

    def add_essay_prompts(self, prompts: list[dict]) -> int:
        """Load the essay-prompt bank from already-validated rows (see
        load_essay_prompts). An id already in the database is left
        untouched rather than overwritten, the same reasoning as
        add_passages skipping an id it already has: editing the source
        document and restarting the app must never silently rewrite a
        prompt a learner has already attempted."""
        with self.connection() as db:
            before = db.total_changes
            db.executemany("INSERT OR IGNORE INTO essay_prompts(id,topic,essay_type,prompt) VALUES(?,?,?,?)",
                           [(p["id"], p["topic"], p["essay_type"], p["prompt"]) for p in prompts])
            return db.total_changes - before

    def essay_prompts(self, topic="All topics", essay_type=None):
        query = "SELECT * FROM essay_prompts"
        conditions, params = [], []
        if topic != "All topics":
            conditions.append("topic=?")
            params.append(topic)
        if essay_type:
            conditions.append("essay_type=?")
            params.append(essay_type)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        with self.connection() as db:
            return [dict(r) for r in db.execute(query, params).fetchall()]

    def choose_essay_prompt(self, topic="All topics", essay_type=None, rng=None):
        """Pick a prompt for `topic`/`essay_type`, preferring one the
        learner has never attempted; once every matching prompt has been
        attempted, allow a repeat rather than returning nothing — the same
        policy as choose_passage."""
        rng = rng or random.Random()
        pool = self.essay_prompts(topic, essay_type)
        if not pool:
            return None
        with self.connection() as db:
            attempted = {r["prompt_id"] for r in db.execute("SELECT DISTINCT prompt_id FROM writing_attempts")}
        candidates = [p for p in pool if p["id"] not in attempted] or pool
        return rng.choice(candidates)

    def record_writing_attempt(self, prompt_id, mode, essay_text, evaluation: dict,
                                deadline_hit=False, time_taken_seconds=None) -> str:
        """Save a graded essay attempt. `evaluation` is re-validated through
        EssayEvaluation here as a safety net, independent of whatever
        validation already happened when it was generated (the same
        reasoning as add_passages re-validating a passage bundle). The
        overall band is computed here, deterministically, from the
        validated per-criterion bands rather than trusted from anywhere
        else. A writing attempt is written once, fully graded, and never
        mutated afterward — there is no in-progress row the way a reading
        session has, since grading only happens once at submission."""
        if mode not in ("simulation", "practice"):
            raise ValueError("mode must be 'simulation' or 'practice'.")
        if not essay_text.strip():
            raise ValueError("An essay attempt needs essay text.")
        ev = EssayEvaluation.model_validate(evaluation)
        band = overall_band(ev)
        aid = uuid.uuid4().hex
        with self.connection() as db:
            db.execute("""INSERT INTO writing_attempts
                (id,prompt_id,mode,essay_text,deadline_hit,time_taken_seconds,evaluation,overall_band,created_at)
                VALUES(?,?,?,?,?,?,?,?,?)""",
                (aid, prompt_id, mode, essay_text, int(bool(deadline_hit)), time_taken_seconds,
                 json.dumps(ev.model_dump(), ensure_ascii=False), band, now_iso()))
        return aid

    def writing_attempts(self):
        with self.connection() as db:
            rows = [dict(r) for r in db.execute("SELECT * FROM writing_attempts ORDER BY created_at")]
        for r in rows:
            r["evaluation"] = json.loads(r["evaluation"])
            r["deadline_hit"] = bool(r["deadline_hit"])
        return rows

    def writing_attempt(self, aid):
        with self.connection() as db:
            row = db.execute("SELECT * FROM writing_attempts WHERE id=?", (aid,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["evaluation"] = json.loads(d["evaluation"])
        d["deadline_hit"] = bool(d["deadline_hit"])
        return d

    def export_json(self):
        with self.connection() as db:
            tables = {name: [dict(r) for r in db.execute(f"SELECT * FROM {name}")] for name in
                      ("questions", "sessions", "attempts", "reports", "generation_log", "flashcard_activity",
                       "passages", "reading_sessions", "reading_activity", "essay_prompts", "writing_attempts")}
        return json.dumps({"version": 2, "exported_at": now_iso(), "tables": tables}, ensure_ascii=False, indent=2)

    def backup_bytes(self):
        # SQLite's backup API creates a consistent snapshot while the app is open.
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "backup.sqlite3"
            with self.connection() as source:
                with sqlite3.connect(target) as destination:
                    source.backup(destination)
            return target.read_bytes()
