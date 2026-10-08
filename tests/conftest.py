import pytest

from coach.content import load_words


@pytest.fixture(autouse=True)
def local_mode(monkeypatch):
    """Tests always run the single-learner local app, even on a machine whose
    secrets configure the shared (guest-mode) database, so no test can ever
    reach the real Neon database."""
    monkeypatch.setenv("IELTS_MODE", "local")
    monkeypatch.delenv("DATABASE_URL", raising=False)


@pytest.fixture
def retired_words():
    """A previous editable catalogue, without requiring another production list."""
    words = [dict(w) for w in load_words()[:20]]
    words[0].update(id="lagoon", word="lagoon", pos="noun", definition="a shallow area of coastal water",
                    example="The lagoon is sheltered by the reef.", collocation="a coastal lagoon",
                    cloze="The ___ is sheltered by the reef.", distractors="decade;budget;rodent")
    return words
