from pathlib import Path

from streamlit.testing.v1 import AppTest

from coach.storage import Store
from coach.content import seed_questions

APP = Path(__file__).resolve().parents[1] / "app.py"


def button(app, label):
    return next(b for b in app.button if b.label == label)


def test_full_session_progress_and_resume(tmp_path, monkeypatch):
    db_path = tmp_path / "app.sqlite3"
    monkeypatch.setenv("IELTS_DB_PATH", str(db_path))
    at = AppTest.from_file(str(APP)).run(timeout=20)
    assert not at.exception
    button(at, "Start today's practice").click().run()
    assert not at.exception
    store = Store(db_path)
    for index in range(10):
        session = store.session()
        q = session["payload"]["questions"][index]
        answer_radio = next(r for r in at.radio if r.label == "Choose your answer")
        assert not answer_radio.value
        answer_radio.set_value(q["options"][q["answer"]]).run()
        button(at, "Check answer").click().run()
        assert not at.exception
        assert len(store.attempts()) == index + 1
        # A second render must not write a second answer.
        at.run()
        assert len(store.attempts()) == index + 1
        button(at, "Finish session" if index == 9 else "Next question").click().run()
    assert not at.exception
    assert store.session() is None
    button(at, "Back to vocabulary").click().run()
    at.sidebar.radio[0].set_value("Progress").run()
    assert not at.exception
    # Every question tests a different word, so a single session's single
    # correct attempt per word is never enough evidence to reach "Familiar".
    assert any(m.label == "Learning" and m.value == "10" for m in at.metric)
    assert len(at.get("download_button")) == 3
    for section in ("Speaking", "Writing", "Vocabulary"):
        at.sidebar.radio[0].set_value(section).run()
        assert not at.exception
    button(at, "Start today's practice").click().run()
    reloaded = AppTest.from_file(str(APP)).run(timeout=20)
    assert not reloaded.exception
    button(reloaded, "Continue saved session").click().run()
    assert not reloaded.exception
    assert any(r.label == "Choose your answer" for r in reloaded.radio)


def test_flagging_after_answer_removes_score(tmp_path, monkeypatch):
    monkeypatch.setenv("IELTS_DB_PATH", str(tmp_path / "flag.sqlite3"))
    at = AppTest.from_file(str(APP)).run(timeout=20)
    button(at, "Start today's practice").click().run()
    answer = next(r for r in at.radio if r.label == "Choose your answer")
    answer.set_value(answer.options[0]).run()
    button(at, "Check answer").click().run()
    button(at, "Flag this question and skip").click().run()
    assert not at.exception
    store = Store(tmp_path / "flag.sqlite3")
    assert len(store.attempts()) == 0
    assert len(store.attempts(True)) == 1


def test_removed_word_session_and_history_still_open(tmp_path, monkeypatch, retired_words):
    path = tmp_path / "old-session.sqlite3"
    store = Store(path)
    old_words = retired_words
    store.add_questions(seed_questions(old_words))
    sid = store.create_session(old_words)
    monkeypatch.setenv("IELTS_DB_PATH", str(path))
    at = AppTest.from_file(str(APP)).run(timeout=20)
    button(at, "Continue saved session").click().run()
    q = store.session(sid)["payload"]["questions"][0]
    assert q["word_id"] == "lagoon"
    next(r for r in at.radio if r.label == "Choose your answer").set_value(q["options"][q["answer"]]).run()
    button(at, "Check answer").click().run()
    assert not at.exception
    at.sidebar.radio[0].set_value("Progress").run()
    assert not at.exception
    # The Progress page's vocabulary table is scoped to the current word
    # catalog, but a retired word's own answer history is still preserved
    # and retrievable underneath, even once no page element lists it by name.
    assert any(a["word_id"] == "lagoon" for a in store.attempts(True))


def test_flashcard_flip_next_topic_and_progress(tmp_path, monkeypatch):
    path = tmp_path / 'cards.sqlite3'
    monkeypatch.setenv('IELTS_DB_PATH', str(path))
    at = AppTest.from_file(str(APP)).run(timeout=20)
    assert not at.exception
    assert [t.label for t in at.tabs] == ['Daily practice', 'Flash cards', 'Word library', 'Fresh questions']
    store = Store(path)
    assert not store.flashcard_activity()  # Rendering an inactive tab does not count as review.
    original = dict(at.session_state['flashcard'])
    at.button(key='flashcard_flip').click().run()
    assert not at.exception
    assert at.session_state['flashcard']['word_id'] == original['word_id']
    assert at.session_state['flashcard']['back']
    assert store.flashcards_reviewed() == 1
    assert 'Click to see the word again' in at.button(key='flashcard_flip').label
    at.button(key='flashcard_flip').click().run()
    at.button(key='flashcard_flip').click().run()
    assert store.flashcards_reviewed() == 1
    at.button(key='flashcard_next').click().run()
    assert not at.session_state['flashcard']['back']
    assert at.session_state['flashcard']['word_id'] != original['word_id']
    assert len(store.flashcard_activity()) == 1
    at.button(key='flashcard_next').click().run()
    assert [a['outcome'] for a in store.flashcard_activity()] == ['review', 'skip']
    topic = at.selectbox(key='flashcard_topic').options[-1]
    at.selectbox(key='flashcard_topic').set_value(topic).run()
    assert not at.exception
    from coach.content import load_words
    by_id = {w['id']: w for w in load_words()}
    assert by_id[at.session_state['flashcard']['word_id']]['topic'] == topic
    assert not at.session_state['flashcard']['back']
    assert len(at.session_state['flashcard_recent']) == 1
    assert len(store.flashcard_activity()) == 2
    assert not store.attempts()
    at.sidebar.radio[0].set_value('Progress').run()
    assert not at.exception
    assert any(e.label == 'Flash card history' for e in at.expander)
    assert any(m.label == 'New' and m.value == str(len(by_id)) for m in at.metric)


def _go_to_writing(at):
    at.sidebar.radio[0].set_value('Writing').run()
    assert not at.exception


def test_writing_practice_flow_without_api_key_shows_error_and_preserves_essay(tmp_path, monkeypatch):
    import streamlit as st
    monkeypatch.setenv('IELTS_DB_PATH', str(tmp_path / 'writing.sqlite3'))
    # This must behave the same whether or not *this machine* happens to have
    # a real OpenAI key configured (an env var, or a parent's own
    # .streamlit/secrets.toml for everyday use) — neutralize both so the test
    # never depends on host configuration and, crucially, never risks a real
    # API call being made using a real key.
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    monkeypatch.setattr(st, 'secrets', {})
    at = AppTest.from_file(str(APP)).run(timeout=20)
    _go_to_writing(at)
    assert any(m.value.startswith('###') for m in at.markdown)  # the prompt card
    button(at, next(b.label for b in at.button if b.label.startswith('Start — Practice'))).click().run()
    assert not at.exception
    essay_box = next(t for t in at.text_area if t.label == 'Your essay')
    essay_box.set_value('This is my practice essay about the given topic.').run()
    assert not at.exception
    assert any(c.value.startswith('9 words') for c in at.caption)
    button(at, 'Submit for grading').click().run()
    assert not at.exception
    assert any('OpenAI API key' in e.value for e in at.error)
    # The essay itself is never lost while grading can't proceed.
    button(at, 'Back to writing').click().run()
    assert not at.exception
    essay_box = next(t for t in at.text_area if t.label == 'Your essay')
    assert essay_box.value == 'This is my practice essay about the given topic.'


def test_writing_simulation_locks_essay_box_once_the_deadline_passes(tmp_path, monkeypatch):
    import time as time_module
    monkeypatch.setenv('IELTS_DB_PATH', str(tmp_path / 'writing_sim.sqlite3'))
    at = AppTest.from_file(str(APP)).run(timeout=20)
    _go_to_writing(at)
    button(at, next(b.label for b in at.button if b.label.startswith('Start — Simulation'))).click().run()
    assert not at.exception
    assert any(m.label == 'Time remaining' for m in at.metric)
    essay_box = next(t for t in at.text_area if t.label == 'Your essay')
    essay_box.set_value('An essay written before time runs out.').run()
    assert not essay_box.disabled
    # Jump the clock forward past the 40-minute deadline without waiting in real time.
    at.session_state['writing_started_at'] = time_module.time() - (40 * 60 + 5)
    at.run()
    assert not at.exception
    assert any('Time' in e.value and 'locked' in e.value for e in at.error)
    essay_box = next(t for t in at.text_area if t.label == 'Your essay')
    assert essay_box.disabled
    assert essay_box.value == 'An essay written before time runs out.'
    # Locked does not mean unsubmittable — the learner is still asked to submit.
    submit = next(b for b in at.button if b.label == 'Submit for grading')
    assert not submit.disabled


def test_writing_practice_flow_succeeds_end_to_end_with_a_fake_openai_client(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import coach.writing as writing_module
    from coach.writing import EssayEvaluation

    monkeypatch.setenv('IELTS_DB_PATH', str(tmp_path / 'writing_success.sqlite3'))
    monkeypatch.setenv('OPENAI_API_KEY', 'fake-key-for-test')

    fake_evaluation = EssayEvaluation.model_validate({
        c: dict(band=6.5, assessment='Clear and adequately developed.', evidence=[],
               next_band_advice='Vary sentence openings more.')
        for c in ('task_response', 'coherence_cohesion', 'lexical_resource', 'grammatical_range_accuracy')
    })

    class FakeClient:
        # generate_evaluation looks up `OpenAI` from coach.writing's own
        # module globals at call time, so patching it here reaches every
        # caller (including through app.py) regardless of how or when
        # generate_evaluation itself was imported.
        def __init__(self, *args, **kwargs):
            self.responses = SimpleNamespace(
                parse=lambda **kw: SimpleNamespace(status='completed', output_parsed=fake_evaluation))

    monkeypatch.setattr(writing_module, 'OpenAI', FakeClient)

    at = AppTest.from_file(str(APP)).run(timeout=20)
    _go_to_writing(at)
    next(b for b in at.button if b.label.startswith('Start — Practice')).click().run()
    essay_box = next(t for t in at.text_area if t.label == 'Your essay')
    essay_box.set_value('A complete essay responding fully to the given prompt.').run()
    button(at, 'Submit for grading').click().run()
    assert not at.exception
    band_metric = next(m for m in at.metric if m.label == 'Overall Task 2 band (estimate)')
    assert band_metric.value == '6.5'
    store = Store(tmp_path / 'writing_success.sqlite3')
    saved = store.writing_attempts()
    assert len(saved) == 1
    assert saved[0]['essay_text'] == 'A complete essay responding fully to the given prompt.'
    assert saved[0]['overall_band'] == 6.5


def test_writing_history_shows_a_recorded_attempt_in_progress_page(tmp_path, monkeypatch):
    db_path = tmp_path / 'writing_history.sqlite3'
    monkeypatch.setenv('IELTS_DB_PATH', str(db_path))
    at = AppTest.from_file(str(APP)).run(timeout=20)
    store = Store(db_path)
    prompt = store.essay_prompts()[0]
    evaluation = {c: dict(band=6.5, assessment='Solid, clear response.', evidence=[],
                          next_band_advice='Use a wider range of linking devices.')
                  for c in ('task_response', 'coherence_cohesion', 'lexical_resource',
                            'grammatical_range_accuracy')}
    store.record_writing_attempt(prompt['id'], 'practice', 'A previously written full essay.', evaluation,
                                 time_taken_seconds=1800)
    at.sidebar.radio[0].set_value('Progress').run()
    assert not at.exception
    assert any('Writing history' in m.value for m in at.markdown)
    assert any(e.label == 'Writing history' for e in at.expander)
    detail = next(m for m in at.metric if m.label == 'Overall Task 2 band (estimate)')
    assert detail.value == '6.5'
    assert any('A previously written full essay.' in m.value for m in at.markdown)
