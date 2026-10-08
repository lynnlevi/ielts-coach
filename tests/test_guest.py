"""Guest mode (the public demo): throwaway per-visitor stores, the shared
Postgres content, the daily AI allowance and the content seed export.

The Postgres tests need a disposable database and are skipped unless
TEST_DATABASE_URL points at one. They drop and recreate the shared tables,
so never point it at the real Neon database.
"""
import base64
import json
import os
import time
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from coach.content import load_words, seed_questions
from coach.guest import build_template, new_guest_store, remove_stale
from coach.storage import Store
from coach.writing import load_essay_prompts

APP = Path(__file__).resolve().parents[1] / "app.py"
TEST_DB = os.environ.get("TEST_DATABASE_URL")
needs_postgres = pytest.mark.skipif(not TEST_DB, reason="set TEST_DATABASE_URL to a disposable Postgres")

PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 32


def _passage(pid="p1", title="Elephant", image=True):
    comprehension = [dict(kind="true_false_not_given", question=f"Statement {i}.", options=["True", "False", "Not Given"],
                          answer=0, explanation="Because.", evidence_quote="Elephants are large.")
                     for i in range(6)]
    p = dict(id=pid, topic="Animals", title=title, body="Elephants are large mammals. " * 40,
             source_name="Wikipedia", source_url=f"https://en.wikipedia.org/wiki/{title}",
             license="CC BY-SA 4.0", is_excerpt=False, word_count=200,
             comprehension=comprehension, vocabulary=[])
    if image:
        p.update(image_bytes=PNG, image_ext="png")
    return p


def _ai_question(word, n=1):
    return dict(word_id=word["id"], kind="meaning", prompt=f"Fresh question {n} about {word['word']}?",
                options=["alpha", "beta", "gamma", "delta"], answer=0,
                explanation="Because this is the right meaning.", source="openai")


@pytest.fixture
def valid_passage(monkeypatch):
    """A passage bundle that passes PassageBundle validation, whatever its
    exact current schema: borrowed from the reading tests' own fixture."""
    from tests.test_reading import _full_comprehension_mix
    p = _passage()
    p["comprehension"] = _full_comprehension_mix()
    return p


def test_template_holds_shared_content_and_no_progress(tmp_path, valid_passage):
    words = load_words()
    template = build_template(tmp_path / "t1", seed_questions(words), load_essay_prompts(),
                              [_ai_question(words[0])], [valid_passage])
    store = Store(template)
    assert any(q["source"] == "openai" for q in store.questions())
    passages = store.passages()
    assert [p["title"] for p in passages] == ["Elephant"]
    # The picture path points at the final template folder, not the temporary build folder.
    assert Path(passages[0]["image_path"]).parent == tmp_path / "t1" / "reading_images"
    assert Path(passages[0]["image_path"]).read_bytes() == PNG
    assert store.attempts() == [] and store.writing_attempts() == []
    # Building the same version again reuses it.
    assert build_template(tmp_path / "t1", [], [], [], []) == template


def test_each_guest_gets_an_independent_copy(tmp_path, valid_passage):
    words = load_words()
    template = build_template(tmp_path / "t", seed_questions(words), load_essay_prompts(), [], [valid_passage])
    a = new_guest_store(template, tmp_path / "visitors")
    b = new_guest_store(template, tmp_path / "visitors")
    assert a.path != b.path
    sid = a.create_session(words, length=3)
    q = a.session(sid)["payload"]["questions"][0]
    a.record(sid, q["id"], q["options"][0])
    assert len(a.attempts()) == 1
    assert b.attempts() == []
    assert Store(template).attempts() == []
    # Both still see the shared passage and its picture.
    assert Path(b.passages()[0]["image_path"]).exists()


def test_remove_stale_deletes_only_old_folders(tmp_path):
    old, fresh, kept = tmp_path / "old", tmp_path / "fresh", tmp_path / "kept"
    for folder in (old, fresh, kept):
        folder.mkdir()
        (folder / "coach.sqlite3").write_text("x")
    long_ago = time.time() - 3 * 24 * 3600
    for path in (old, old / "coach.sqlite3", kept, kept / "coach.sqlite3"):
        os.utime(path, (long_ago, long_ago))
    remove_stale(tmp_path, 24, keep=(kept,))
    assert not old.exists()
    assert fresh.exists() and kept.exists()


def test_export_seed_has_content_but_no_progress(tmp_path, valid_passage):
    from scripts.export_content_seed import export
    words = load_words()
    store = Store(tmp_path / "local" / "coach.sqlite3")
    store.add_questions(seed_questions(words)[:9])
    store.add_questions([_ai_question(words[0], 1), _ai_question(words[1], 2)])
    store.add_passages([valid_passage])
    flagged = [q for q in store.questions() if q["source"] == "openai"][1]
    store.flag(flagged["id"])
    sid = store.create_session(words, length=2)
    q = store.session(sid)["payload"]["questions"][0]
    store.record(sid, q["id"], q["options"][0])

    seed = export(tmp_path / "local" / "coach.sqlite3", tmp_path / "seed.json")
    assert [q["prompt"] for q in seed["questions"]] == [f"Fresh question 1 about {words[0]['word']}?"]
    assert len(seed["passages"]) == 1
    assert base64.b64decode(seed["passages"][0]["image_base64"]) == PNG
    text = (tmp_path / "seed.json").read_text(encoding="utf-8")
    assert set(json.loads(text)) == {"version", "questions", "passages"}
    assert "attempt" not in text and "answered_at" not in text and sid not in text


@pytest.fixture
def shared():
    import psycopg

    from coach.cloud import SharedContent
    with psycopg.connect(TEST_DB, autocommit=True) as db:
        db.execute("DROP TABLE IF EXISTS shared_questions, shared_passages, ai_usage")
    content = SharedContent(TEST_DB)
    content.ensure_schema()
    content.ensure_schema()  # safe to run on every start
    return content


@needs_postgres
def test_shared_content_round_trip(shared, valid_passage):
    words = load_words()
    v0 = shared.version()
    assert shared.save_questions([_ai_question(words[0])]) == 1
    assert shared.save_questions([_ai_question(words[0])]) == 0  # same content, same id
    assert shared.save_passages([valid_passage]) == 1
    assert shared.save_passages([valid_passage]) == 0
    assert shared.version() != v0
    questions, passages = shared.load()
    assert questions[0]["source"] == "openai" and questions[0]["id"]
    assert passages[0]["image_bytes"] == PNG and passages[0]["image_ext"] == "png"
    # What comes back can be loaded straight into a guest template.
    store = Store(build_template(Path(os.environ.get("TMPDIR", "/tmp")) / f"tpl-{time.time_ns()}",
                                 [], [], questions, passages))
    assert store.passages()[0]["title"] == "Elephant"


@needs_postgres
def test_seed_file_loads_once(shared, tmp_path, valid_passage):
    from scripts.export_content_seed import export
    store = Store(tmp_path / "local" / "coach.sqlite3")
    store.add_passages([valid_passage])
    export(tmp_path / "local" / "coach.sqlite3", tmp_path / "seed.json")
    assert shared.seed_from_file(tmp_path / "seed.json") == (0, 1)
    assert shared.seed_from_file(tmp_path / "seed.json") == (0, 0)
    assert shared.seed_from_file(tmp_path / "missing.json") == (0, 0)
    assert shared.load()[1][0]["image_bytes"] == PNG


@needs_postgres
def test_daily_ai_allowance_is_capped_and_refundable(shared):
    assert shared.ai_used_today() == 0
    assert shared.reserve_ai_action(2) and shared.reserve_ai_action(2)
    assert not shared.reserve_ai_action(2)
    assert shared.ai_used_today() == 2
    shared.release_ai_action()
    assert shared.reserve_ai_action(2)
    assert shared.reserve_ai_action(None)  # the owner is never refused, but still counted
    assert shared.ai_used_today() == 3
    assert not shared.reserve_ai_action(0)


@needs_postgres
def test_app_in_guest_mode(shared, monkeypatch, valid_passage):
    shared.save_passages([valid_passage])
    monkeypatch.delenv("IELTS_MODE")
    monkeypatch.setenv("DATABASE_URL", TEST_DB)
    monkeypatch.setenv("AI_DAILY_LIMIT", "1")
    at = AppTest.from_file(str(APP)).run(timeout=60)
    assert not at.exception
    assert any("Guest mode" in m.value for m in at.markdown)
    assert any("Guest mode" in i.value for i in at.sidebar.info)
    store = at.session_state["guest_store"]
    # The shared passage, plus whatever data/content_seed.json starts the demo with.
    assert "Elephant" in [p["title"] for p in store.passages()]
    # A second visitor gets their own store.
    other = AppTest.from_file(str(APP)).run(timeout=60)
    assert other.session_state["guest_store"].path != store.path
    # The SQLite backup download is only offered in the local app.
    at.sidebar.radio[0].set_value("Progress").run()
    assert not at.exception
    labels = [b.label for b in at.get("download_button")]
    assert "Download full backup" not in labels and "Export progress (CSV)" in labels


def _fake_grader(monkeypatch):
    from types import SimpleNamespace

    import coach.writing as writing_module
    from coach.writing import EssayEvaluation
    evaluation = EssayEvaluation.model_validate({
        c: dict(band=6.0, assessment='Clear and adequately developed.', evidence=[],
                next_band_advice='Vary sentence openings more.')
        for c in ('task_response', 'coherence_cohesion', 'lexical_resource', 'grammatical_range_accuracy')})
    calls = []

    class FakeClient:
        def __init__(self, *args, **kwargs):
            def parse(**kw):
                calls.append(kw)
                return SimpleNamespace(status='completed', output_parsed=evaluation)
            self.responses = SimpleNamespace(parse=parse)

    monkeypatch.setattr(writing_module, 'OpenAI', FakeClient)
    return calls


def _grade_one_essay(at):
    at.sidebar.radio[0].set_value('Writing').run()
    next(b for b in at.button if b.label.startswith('Start — Practice')).click().run()
    next(t for t in at.text_area if t.label == 'Your essay').set_value('A complete essay for the prompt.').run()
    next(b for b in at.button if b.label == 'Submit for grading').click().run()
    assert not at.exception


@needs_postgres
def test_guest_ai_allowance_stops_grading_but_keeps_the_essay(shared, monkeypatch):
    calls = _fake_grader(monkeypatch)
    monkeypatch.delenv("IELTS_MODE")
    monkeypatch.setenv("DATABASE_URL", TEST_DB)
    monkeypatch.setenv("AI_DAILY_LIMIT", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key-for-test")
    first = AppTest.from_file(str(APP)).run(timeout=60)
    _grade_one_essay(first)
    assert any(m.label == 'Overall Task 2 band (estimate)' for m in first.metric)
    assert shared.ai_used_today() == 1
    # Another visitor, same day: the shared allowance is used up.
    second = AppTest.from_file(str(APP)).run(timeout=60)
    _grade_one_essay(second)
    assert any('allowance' in w.value for w in second.warning)
    assert len(calls) == 1
    next(b for b in second.button if b.label == 'Back to writing').click().run()
    assert next(t for t in second.text_area if t.label == 'Your essay').value == 'A complete essay for the prompt.'
    # The owner link skips the cap.
    monkeypatch.setenv("OWNER_PASSCODE", "open-sesame")
    owner = AppTest.from_file(str(APP))
    owner.query_params["owner"] = "open-sesame"
    owner.run(timeout=60)
    _grade_one_essay(owner)
    assert any(m.label == 'Overall Task 2 band (estimate)' for m in owner.metric)
    assert len(calls) == 2
    # A wrong passcode does not.
    wrong = AppTest.from_file(str(APP))
    wrong.query_params["owner"] = "guess"
    wrong.run(timeout=60)
    _grade_one_essay(wrong)
    assert len(calls) == 2
