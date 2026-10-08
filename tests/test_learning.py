import json
import random
import sqlite3
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from coach.ai import QuestionBatch, generate_questions
from coach.content import Question, load_words, seed_questions
from coach.storage import Store, mastery


@pytest.fixture
def setup(tmp_path):
    words = load_words()
    questions = seed_questions(words)
    store = Store(tmp_path / "test.sqlite3")
    store.add_questions(questions)
    return store, words, questions


def test_starter_content_is_complete_and_usable(setup):
    store, words, questions = setup
    assert words
    assert all(w["topic"] for w in words)
    assert len(questions) == len(store.questions()) == 3 * len(words)
    for w in words:
        assert w["word"].casefold() in w["example"].casefold()
    for q in questions:
        Question.model_validate({k: q[k] for k in Question.model_fields})
        assert len(set(q["options"])) == 4


def test_answer_persists_and_cannot_be_overwritten(setup):
    store, words, _ = setup
    sid = store.create_session(words, rng=random.Random(1))
    q = store.session(sid)["payload"]["questions"][0]
    right = q["options"][q["answer"]]
    first = store.record(sid, q["id"], right)
    second = store.record(sid, q["id"], next(o for o in q["options"] if o != right))
    assert first == second
    assert len(Store(store.path).attempts()) == 1
    assert Store(store.path).attempts()[0]["correct"] == 1
    assert Store(store.path).session()["id"] == sid


def test_scoring_uses_original_bank_not_shuffled_position(setup):
    store, words, _ = setup
    sid = store.create_session(words, rng=random.Random(8))
    qs = store.session(sid)["payload"]["questions"]
    assert any(q["answer"] != 0 for q in qs)
    for q in qs:
        assert store.record(sid, q["id"], q["options"][q["answer"]])["correct"] == 1


def test_report_removes_historical_score_and_future_question(setup):
    store, words, _ = setup
    sid = store.create_session(words)
    q = store.session(sid)["payload"]["questions"][0]
    store.record(sid, q["id"], q["options"][q["answer"]])
    store.flag(q["id"])
    assert not store.attempts()
    assert len(store.attempts(include_flagged=True)) == 1
    assert q["id"] not in {q["id"] for q in store.questions()}
    assert store.progress(words)[q["word_id"]]["stage"] == "New"
    with pytest.raises(ValueError):
        store.record(sid, q["id"], q["options"][0])


def attempt(day, qid, kind="meaning", correct=True, hour=12):
    return dict(question_id=qid, answered_at=f"2026-09-{day:02}T{hour:02}:00:00+07:00", kind=kind, correct=correct)


def test_mastery_requires_days_and_distinct_skills():
    # A cram session packed into a single day, however many correct answers
    # it contains, only ever earns the first rung: growth is driven by
    # distinct calendar days, not raw volume of evidence.
    same_day = [attempt(1, str(i), "cloze" if i % 2 else "meaning") for i in range(10)]
    assert mastery(same_day, "2026-09-01")["stage"] == "Learning"
    # Five distinct correct days reach "Familiar" (the 7-day rung) even with
    # only one real skill demonstrated -- the first few rungs are earned
    # quickly. "meaning" and "context" collapse to the same skill, so this
    # word has in fact only ever been tested one way.
    meaning_only = [attempt(i, str(i), "context" if i % 2 else "meaning") for i in range(1, 6)]
    assert mastery(meaning_only, "2026-09-05")["stage"] == "Familiar"
    varied = [attempt(1, "a"), attempt(1, "b", "cloze"), attempt(2, "a"), attempt(3, "b", "cloze")]
    result = mastery(varied, "2026-09-03")
    # 3 distinct days and 2 skills land on the same 7-day rung as above, so
    # this is still "Familiar" -- "Secure" now means something longer (see
    # test_mastery_interval_keeps_growing_past_a_week below).
    assert result["stage"] == "Familiar"
    assert result["due"] == "2026-09-10"
    varied.append(attempt(4, "a", correct=False))
    result = mastery(varied, "2026-09-04")
    # A miss drops even a Familiar word back to Learning, due tomorrow.
    assert result["stage"] == "Learning"
    assert result["due"] == "2026-09-05"


def test_mastery_interval_keeps_growing_past_a_week():
    # A fourth distinct correct day, still with 2 skills, grows past the old
    # 7-day ceiling instead of plateauing there forever.
    four_days = [attempt(1, "a"), attempt(1, "b", "cloze"), attempt(2, "a"), attempt(3, "b", "cloze"),
                 attempt(4, "a")]
    result = mastery(four_days, "2026-09-04")
    assert result["stage"] == "Secure"
    assert result["due"] == "2026-09-18"  # +14 days
    # A word with only one skill demonstrated stays frozen at the 7-day rung
    # even with many more distinct correct days -- growth past a week is
    # gated on skill variety, not just streak length.
    one_skill_only = [attempt(i, "a") for i in range(1, 8)]
    result = mastery(one_skill_only, "2026-09-07")
    assert result["stage"] == "Familiar"
    assert result["due"] == "2026-09-14"  # still the 7-day rung
    # Once growth is unlocked, it keeps going all the way to the 90-day cap.
    nine_days = [attempt(1, "a"), attempt(1, "b", "cloze")] + [attempt(d, "a") for d in range(2, 10)]
    result = mastery(nine_days, "2026-09-09")
    assert result["stage"] == "Secure"
    assert result["due"] == "2026-12-08"  # last answer (Sep 9) + 90 days


def test_same_day_repeats_do_not_inflate_mastery():
    data = [attempt(1, "a", correct=False), attempt(1, "a", correct=True, hour=13)]
    result = mastery(data, "2026-09-01")
    assert result["stage"] == "Learning"
    assert result["due"] == "2026-09-02"
    assert result["total"] == 1
    assert result["correct"] == 0


def test_new_words_are_not_repeated_across_sessions_same_day(setup):
    store, words, _ = setup
    sid = store.create_session(words, length=10)
    introduced = store.session(sid)["payload"]["new_words"]
    assert len(introduced) == 10
    store.finish(sid)
    next_sid = store.create_session(words, length=10)
    next_introduced = store.session(next_sid)["payload"]["new_words"]
    # No fixed daily cap any more, but a fresh word already shown today cannot
    # be introduced again in a later session the same day.
    assert len(next_introduced) == 10
    assert set(next_introduced).isdisjoint(introduced)
    assert {q["word_id"] for q in store.session(next_sid)["payload"]["questions"]} == set(next_introduced)


def test_due_words_are_selected_before_new_words(setup, monkeypatch):
    store, words, _ = setup
    monkeypatch.setattr("coach.storage.now_iso", lambda: "2026-09-01T12:00:00+07:00")
    sid = store.create_session(words, topic="Business")
    q = store.session(sid)["payload"]["questions"][0]
    store.record(sid, q["id"], q["options"][(q["answer"] + 1) % 4])
    store.finish(sid)
    monkeypatch.setattr("coach.storage.now_iso", lambda: "2026-09-03T12:00:00+07:00")
    next_sid = store.create_session(words)
    assert store.session(next_sid)["payload"]["questions"][0]["word_id"] == q["word_id"]


def test_invalid_answers_and_unrelated_questions_rejected(setup):
    store, words, questions = setup
    sid = store.create_session(words)
    q = store.session(sid)["payload"]["questions"][0]
    with pytest.raises(ValueError):
        store.record(sid, q["id"], "not an option")
    with pytest.raises(ValueError):
        store.record(sid, questions[-1]["id"], questions[-1]["options"][0])
    assert not store.attempts()


def test_backup_can_restore_in_a_new_database(setup, tmp_path):
    store, words, _ = setup
    sid = store.create_session(words)
    q = store.session(sid)["payload"]["questions"][0]
    store.record(sid, q["id"], q["options"][q["answer"]])
    path = tmp_path / "restored.sqlite3"
    path.write_bytes(store.backup_bytes())
    restored = Store(path)
    assert restored.attempts() == store.attempts()
    assert restored.questions() == store.questions()
    assert json.loads(store.export_json())["tables"]["attempts"]
    with sqlite3.connect(path) as db:
        assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


def mock_client(words):
    bank = seed_questions(load_words())
    qs = [q for q in bank if q["word_id"] == words[0]["id"]]
    qs[0]["kind"] = "usage"
    parsed = QuestionBatch(questions=[Question.model_validate({k: q[k] for k in Question.model_fields}) for q in qs])
    client = Mock()
    client.responses.parse.return_value = SimpleNamespace(status="completed", output_parsed=parsed, usage=None)
    return client


def test_ai_request_only_sends_word_content_and_validates_batch():
    words = load_words()[:1]
    client = mock_client(words)
    result, usage = generate_questions(words, "test-key", client=client)
    assert len(result) == 3
    kwargs = client.responses.parse.call_args.kwargs
    assert kwargs["store"] is False
    assert "test-key" not in kwargs["input"]
    assert set(json.loads(kwargs["input"])) == {"words", "existing_prompts"}
    assert all(q["source"] == "openai" for q in result)


def test_ai_refusal_and_bad_coverage_not_saved():
    words = load_words()[:1]
    client = mock_client(words)
    client.responses.parse.return_value.output_parsed = None
    with pytest.raises(ValueError, match="complete"):
        generate_questions(words, "test-key", client=client)
    client = mock_client(words)
    client.responses.parse.return_value.output_parsed.questions[0].word_id = "unknown"
    with pytest.raises(ValueError, match="cover"):
        generate_questions(words, "test-key", client=client)


def test_validation_rejects_duplicate_options():
    q = seed_questions(load_words())[0]
    q["options"][1] = q["options"][0].upper()
    with pytest.raises(ValueError):
        Question.model_validate({k: q[k] for k in Question.model_fields})


def test_catalogue_can_grow_without_source_metadata(tmp_path):
    import csv
    words = load_words()
    extra = dict(words[0], id="precision", word="precision", topic="New topic", pos="noun",
                 definition="the quality of being exact and accurate", vi="độ chính xác",
                 example="The instrument measures distance with precision.", collocation="great precision",
                 cloze="The instrument measures distance with ___.", distractors="habitat;decade;budget")
    path = tmp_path / "expanded.psv"
    with path.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(words[0]), delimiter="|")
        writer.writeheader()
        writer.writerows(words + [extra])
    expanded = load_words(path)
    assert len(expanded) == len(words) + 1
    assert expanded[-1]["topic"] == "New topic"
    assert len(seed_questions(expanded)) == 3 * len(expanded)
    assert set(expanded[-1]) == {"id", "word", "topic", "pos", "definition", "vi", "example", "collocation", "cloze", "distractors"}
    store = Store(tmp_path / "expanded.sqlite3")
    store.add_questions(seed_questions(expanded))
    sid = store.create_session(expanded, topic="New topic")
    assert {q["word_id"] for q in store.session(sid)["payload"]["questions"]} == {"precision"}


def test_removed_words_keep_answers_and_session_snapshots(tmp_path, retired_words):
    store = Store(tmp_path / "migration.sqlite3")
    store.add_questions(seed_questions(retired_words))
    sid = store.create_session(retired_words)
    q = store.session(sid)["payload"]["questions"][0]
    store.record(sid, q["id"], q["options"][q["answer"]])
    old_answers = store.attempts(True)
    old_session = store.session(sid)
    store.add_questions(seed_questions(load_words()))
    assert store.attempts(True) == old_answers
    assert store.session(sid) == old_session
    assert old_session["payload"]["words"]["lagoon"]["word"] == "lagoon"
    assert "lagoon" not in {w["id"] for w in load_words()}


def test_topic_filter_limits_new_practice(setup):
    store, words, _ = setup
    topic = "Technology"
    sid = store.create_session(words, topic=topic)
    ids = {q["word_id"] for q in store.session(sid)["payload"]["questions"]}
    assert ids <= {w["id"] for w in words if w["topic"] == topic}
    assert len(ids) == 10  # default session length, well within the 60-word topic


def test_each_question_tests_a_different_word(setup):
    store, words, _ = setup
    sid = store.create_session(words, length=25, rng=random.Random(3))
    questions = store.session(sid)["payload"]["questions"]
    word_ids = [q["word_id"] for q in questions]
    assert len(word_ids) == len(set(word_ids)) == 25
    assert len({q["id"] for q in questions}) == 25


def test_session_caps_to_available_words_when_pool_is_small(setup):
    store, words, _ = setup
    topic_words = {w["id"] for w in words if w["topic"] == "Business"}
    sid = store.create_session(words, topic="Business", length=len(topic_words) + 10)
    questions = store.session(sid)["payload"]["questions"]
    # Requesting more questions than the topic has words: the session is
    # capped to one question per available word instead of repeating any.
    assert len(questions) == len(topic_words)
    assert {q["word_id"] for q in questions} == topic_words


def test_old_question_revisions_are_retained_but_not_served(setup):
    store, words, questions = setup
    superseded = dict(questions[0], prompt="What was the meaning of this word in an older question revision?")
    store.add_questions([superseded])
    assert len(store.questions()) == len(questions) + 1
    assert len(store.active_questions(words)) == len(questions)
    assert all(q["prompt"] != superseded["prompt"] for q in store.active_questions(words))


def test_old_source_labels_migrate_without_changing_answers(setup):
    store, words, questions = setup
    sid = store.create_session(words)
    q = store.session(sid)["payload"]["questions"][0]
    store.record(sid, q["id"], q["options"][q["answer"]])
    before = store.attempts(True)
    with store.connection() as db:
        row = db.execute("SELECT payload FROM questions WHERE id=?", (q["id"],)).fetchone()
        payload = json.loads(row["payload"])
        payload["source"] = "imported"
        db.execute("UPDATE questions SET source='imported',payload=? WHERE id=?", (json.dumps(payload), q["id"]))
    reopened = Store(store.path)
    assert reopened.attempts(True) == before
    saved = next(item for item in reopened.questions() if item["id"] == q["id"])
    assert saved["source"] == "offline"
