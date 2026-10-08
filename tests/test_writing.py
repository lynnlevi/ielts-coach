import random
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from coach.storage import Store
from coach.writing import (
    CRITERIA,
    ESSAY_TYPES,
    EssayEvaluation,
    generate_evaluation,
    load_essay_prompts,
    overall_band,
    snap_to_half_band,
)


# --- Prompt bank loading ---

def test_load_essay_prompts_reads_the_real_seed_file():
    prompts = load_essay_prompts()
    assert len(prompts) >= 4
    ids = {p["id"] for p in prompts}
    assert len(ids) == len(prompts), "prompt ids must be unique"
    assert all(p["essay_type"] in ESSAY_TYPES for p in prompts)
    assert all(p["topic"].strip() for p in prompts)


def _write_psv(tmp_path, rows, header="id|topic|essay_type|prompt"):
    path = tmp_path / "prompts.psv"
    lines = [header] + ["|".join(row) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_load_essay_prompts_rejects_missing_field(tmp_path):
    path = _write_psv(tmp_path, [("p1", "Work", "opinion", "")])
    with pytest.raises(ValueError, match="Missing essay prompt field"):
        load_essay_prompts(path)


def test_load_essay_prompts_rejects_unknown_essay_type(tmp_path):
    path = _write_psv(tmp_path, [("p1", "Work", "narrative", "Write a story.")])
    with pytest.raises(ValueError, match="Unknown essay_type"):
        load_essay_prompts(path)


def test_load_essay_prompts_rejects_duplicate_id(tmp_path):
    path = _write_psv(tmp_path, [
        ("dup", "Work", "opinion", "Prompt one?"),
        ("dup", "Health", "opinion", "Prompt two?"),
    ])
    with pytest.raises(ValueError, match="Duplicate essay prompt ID"):
        load_essay_prompts(path)


def test_load_essay_prompts_derives_id_when_missing(tmp_path):
    path = _write_psv(tmp_path, [("", "Work", "opinion", "Should everyone have a four-day work week?")],
                       header="id|topic|essay_type|prompt")
    prompts = load_essay_prompts(path)
    assert prompts[0]["id"]  # derived from the prompt text, never blank


# --- Band arithmetic ---

def test_snap_to_half_band_rounds_to_nearest_half_and_clamps():
    assert snap_to_half_band(6.3) == 6.5
    assert snap_to_half_band(6.24) == 6.0
    assert snap_to_half_band(6.25) == 6.5
    assert snap_to_half_band(0.2) == 1.0  # clamped up from below the real scale
    assert snap_to_half_band(9.4) == 9.0  # clamped down from above the real scale


def _evaluation(bands, evidence=None):
    evidence = evidence or {}
    return EssayEvaluation.model_validate({
        c: dict(band=bands[i], assessment="Assessment text.", next_band_advice="Advice text.",
                evidence=evidence.get(c, []))
        for i, c in enumerate(CRITERIA)
    })


def test_overall_band_matches_the_ielts_rounding_convention():
    # avg 7.375 -> the official published example rounds this up to 7.5.
    assert overall_band(_evaluation([9, 7, 6.5, 7])) == 7.5
    # avg 6.125 -> rounds down to 6.0.
    assert overall_band(_evaluation([6.5, 6, 6, 6])) == 6.0
    # avg 6.375 -> rounds up to 6.5.
    assert overall_band(_evaluation([7, 6.5, 6, 6])) == 6.5
    # exact .25 boundary rounds up to the next half band.
    assert overall_band(_evaluation([6, 6, 6, 7])) == 6.5
    # exact .75 boundary rounds up to the next whole band.
    assert overall_band(_evaluation([6, 7, 7, 7])) == 7.0
    # already a half band stays put.
    assert overall_band(_evaluation([6, 6, 7, 7])) == 6.5


# --- Grounding ---

ESSAY_TEXT = ("Many people today rely on social media every day. It connects friends "
              "across long distances. However, it can also spread false information quickly.")


def test_drop_ungrounded_evidence_keeps_verified_quotes_and_drops_invented_ones():
    evaluation = _evaluation(
        [6, 6, 6, 6],
        evidence={
            "task_response": [
                dict(quote="It connects friends across long distances.", comment="Good supporting point."),
                dict(quote="This sentence was never written by the candidate.", comment="Hallucinated."),
            ],
        },
    )
    from coach.writing import _drop_ungrounded_evidence
    cleaned = _drop_ungrounded_evidence(evaluation, ESSAY_TEXT)
    kept = [e.quote for e in cleaned.task_response.evidence]
    assert kept == ["It connects friends across long distances."]
    # The rest of the criterion survives untouched even though evidence shrank.
    assert cleaned.task_response.band == 6
    assert cleaned.task_response.assessment == "Assessment text."


# --- generate_evaluation ---

def mock_evaluation_client(bands=(6.0, 6.0, 6.0, 6.0), evidence=None):
    parsed = _evaluation(list(bands), evidence=evidence)
    client = Mock()
    client.responses.parse.return_value = SimpleNamespace(status="completed", output_parsed=parsed)
    return client


def test_generate_evaluation_requires_api_key():
    with pytest.raises(ValueError, match="API key"):
        generate_evaluation("Some prompt?", ESSAY_TEXT, "", client=Mock())


def test_generate_evaluation_requires_essay_text():
    with pytest.raises(ValueError, match="essay"):
        generate_evaluation("Some prompt?", "   ", "test-key", client=Mock())


def test_generate_evaluation_raises_on_incomplete_response():
    client = Mock()
    client.responses.parse.return_value = SimpleNamespace(status="incomplete", output_parsed=None)
    with pytest.raises(ValueError, match="complete"):
        generate_evaluation("Some prompt?", ESSAY_TEXT, "test-key", client=client)


def test_generate_evaluation_returns_parsed_result_and_does_not_leak_key():
    client = mock_evaluation_client()
    result = generate_evaluation("Some prompt?", ESSAY_TEXT, "super-secret-key", client=client)
    assert isinstance(result, EssayEvaluation)
    call = client.responses.parse.call_args.kwargs
    assert "super-secret-key" not in str(call)


def test_generate_evaluation_snaps_off_grid_bands():
    client = mock_evaluation_client(bands=(6.3, 6.3, 6.3, 6.3))
    result = generate_evaluation("Some prompt?", ESSAY_TEXT, "test-key", client=client)
    assert result.task_response.band == 6.5


def test_generate_evaluation_drops_ungrounded_evidence_end_to_end():
    client = mock_evaluation_client(evidence={
        "lexical_resource": [dict(quote="a phrase the candidate never wrote", comment="bad")],
    })
    result = generate_evaluation("Some prompt?", ESSAY_TEXT, "test-key", client=client)
    assert result.lexical_resource.evidence == []


# --- Storage: essay prompts + writing attempts ---

@pytest.fixture
def writing_store(tmp_path):
    return Store(tmp_path / "writing.sqlite3")


SAMPLE_PROMPTS = [
    dict(id="p1", topic="Work", essay_type="opinion", prompt="Should everyone work from home?"),
    dict(id="p2", topic="Health", essay_type="discussion", prompt="Discuss both views on public health funding."),
]


def test_add_essay_prompts_inserts_new_and_skips_existing(writing_store):
    added = writing_store.add_essay_prompts(SAMPLE_PROMPTS)
    assert added == 2
    added_again = writing_store.add_essay_prompts(SAMPLE_PROMPTS)
    assert added_again == 0
    assert len(writing_store.essay_prompts()) == 2


def test_essay_prompts_filters_by_topic_and_type(writing_store):
    writing_store.add_essay_prompts(SAMPLE_PROMPTS)
    assert [p["id"] for p in writing_store.essay_prompts("Work")] == ["p1"]
    assert [p["id"] for p in writing_store.essay_prompts("All topics", "discussion")] == ["p2"]


def test_choose_essay_prompt_prefers_unattempted_then_allows_repeat(writing_store):
    writing_store.add_essay_prompts(SAMPLE_PROMPTS)
    rng = random.Random(0)
    first = writing_store.choose_essay_prompt(rng=rng)
    assert first["id"] in ("p1", "p2")
    evaluation = _evaluation([6, 6, 6, 6]).model_dump()
    writing_store.record_writing_attempt(first["id"], "practice", "My essay text here.", evaluation)
    second = writing_store.choose_essay_prompt(rng=rng)
    assert second["id"] != first["id"], "the unattempted prompt should be preferred while one remains"
    writing_store.record_writing_attempt(second["id"], "practice", "Another essay text.", evaluation)
    # Every prompt has now been attempted at least once — a repeat is fine.
    third = writing_store.choose_essay_prompt(rng=rng)
    assert third["id"] in ("p1", "p2")


def test_choose_essay_prompt_returns_none_for_an_empty_pool(writing_store):
    assert writing_store.choose_essay_prompt() is None


def test_record_writing_attempt_validates_mode(writing_store):
    writing_store.add_essay_prompts(SAMPLE_PROMPTS)
    evaluation = _evaluation([6, 6, 6, 6]).model_dump()
    with pytest.raises(ValueError, match="mode"):
        writing_store.record_writing_attempt("p1", "timed", "Some essay.", evaluation)


def test_record_writing_attempt_requires_essay_text(writing_store):
    writing_store.add_essay_prompts(SAMPLE_PROMPTS)
    evaluation = _evaluation([6, 6, 6, 6]).model_dump()
    with pytest.raises(ValueError, match="essay text"):
        writing_store.record_writing_attempt("p1", "practice", "   ", evaluation)


def test_record_writing_attempt_rejects_an_invalid_evaluation(writing_store):
    writing_store.add_essay_prompts(SAMPLE_PROMPTS)
    with pytest.raises(ValueError):
        writing_store.record_writing_attempt("p1", "practice", "Some essay.", {"task_response": {}})


def test_record_writing_attempt_stores_and_computes_overall_band(writing_store):
    writing_store.add_essay_prompts(SAMPLE_PROMPTS)
    evaluation = _evaluation([9, 7, 6.5, 7]).model_dump()  # avg 7.375 -> 7.5
    aid = writing_store.record_writing_attempt(
        "p1", "simulation", "A full essay response.", evaluation,
        deadline_hit=True, time_taken_seconds=2400,
    )
    stored = writing_store.writing_attempt(aid)
    assert stored["prompt_id"] == "p1"
    assert stored["mode"] == "simulation"
    assert stored["essay_text"] == "A full essay response."
    assert stored["deadline_hit"] is True
    assert stored["time_taken_seconds"] == 2400
    assert stored["overall_band"] == 7.5
    assert stored["evaluation"]["task_response"]["band"] == 9


def test_writing_attempt_missing_returns_none(writing_store):
    assert writing_store.writing_attempt("nope") is None


def test_writing_attempts_lists_all_in_order(writing_store):
    writing_store.add_essay_prompts(SAMPLE_PROMPTS)
    evaluation = _evaluation([6, 6, 6, 6]).model_dump()
    writing_store.record_writing_attempt("p1", "practice", "First essay.", evaluation)
    writing_store.record_writing_attempt("p2", "simulation", "Second essay.", evaluation)
    rows = writing_store.writing_attempts()
    assert [r["essay_text"] for r in rows] == ["First essay.", "Second essay."]
