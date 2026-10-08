# CLAUDE.md

Guidance for Claude when working in this repo. The README is the user-facing source of truth for behavior; read the relevant section before changing a feature.

## What this is

**Word by Word — IELTS Coach**: a local, single-learner Streamlit app for a Vietnamese teen learner around IELTS band 4. Sections: Vocabulary (daily quiz, flash cards, AI "Fresh questions"), Reading (cached Wikipedia passages + AI-drafted activities), Progress. Speaking and Writing are labelled future sections — don't build them unless asked.

The learner is the audience for UI text: keep wording simple, friendly and at roughly band-4 English.

## Commands

```sh
source .venv/bin/activate                  # needs Python 3.10+ (tested on 3.12)
python -m pip install -r requirements-dev.txt
python -m streamlit run app.py --server.address 127.0.0.1   # http://localhost:8501
python -m pytest -q                         # all tests
python -m pytest -q tests/test_reading.py   # one file
```

macOS users can double-click `Start IELTS Coach.command`.

Gotcha: the current `.venv/` was created with Python 3.9.6, which is below the supported minimum. If imports or type syntax fail, recreate it with 3.10+ (`rm -rf .venv && python3.12 -m venv .venv`) rather than downgrading the code.

## Layout

```text
app.py               All Streamlit screens: sidebar nav, vocabulary, flash cards, reading flow, progress
coach/content.py     Loads/validates data/words.psv, builds offline questions (Question model, KINDS)
coach/storage.py     Store: SQLite schema + migrations, sessions, attempts, mastery(), reading, backups
coach/flashcards.py  Weighted card selection (WEIGHTS)
coach/ai.py          OpenAI question generation (structured outputs, QuestionBatch)
coach/reading.py     Wikipedia fetch, readability filter, image cache, reading bundle generation
coach/cloud.py       Public demo only: shared content + daily AI allowance in Postgres (Neon)
coach/guest.py       Public demo only: throwaway per-visitor SQLite stores copied from a template
scripts/export_content_seed.py  Exports passages/AI questions (no progress) to data/content_seed.json
data/words.psv       The editable vocabulary (UTF-8, `|`-separated, ~600 rows)
tests/               pytest; test_app.py drives the UI with streamlit.testing.v1.AppTest
local_data/          Learner data (gitignored): coach.sqlite3, reading_images/, backups/
```

## Architecture notes

- **UI logic stays in `app.py`; logic that isn't UI goes in `coach/`.** `Store` in `coach/storage.py` is the only place that touches SQLite — keep it that way so a hosted DB can replace it later.
- `app.py` caches content with `@st.cache_data content(revision)`, where `revision` is the SHA-256 of `words.psv`, and caches the `Store` with `@st.cache_resource database(path, revision, schema_version=...)`. **When you change the schema, bump `schema_version`** so running sessions pick up a fresh `Store`.
- Schema changes go in `Store.__init__`: use `CREATE TABLE IF NOT EXISTS`, add columns with guarded `ALTER TABLE` (see `image_path`), bump `PRAGMA user_version`, and migrate existing rows in place. Never drop or rewrite a learner's attempts or history.
- Rows store JSON in a `payload` column; write it with `ensure_ascii=False` (the data includes Vietnamese).
- Navigation and in-progress state live in `st.session_state` (`active_session`, `cursor`, `flashcard*`, `reading_stage`, `reading_q_cursor`, `active_reading_session`, …). Quiz and reading sessions also persist to the DB so they survive a browser reload. Rerendering must never record a second answer; tests check this.
- **Guest mode (public demo)** is on when a `DATABASE_URL` secret is set, unless `IELTS_MODE=local`. Each browser session gets its own Store (a copy of a template DB, `coach/guest.py`), kept in `st.session_state.guest_store`; progress is never written to Postgres. Postgres (`coach/cloud.py`) holds only shared content and the app-wide daily AI allowance. Every AI action in the UI goes through `start_ai_action()` / `undo_ai_action()` in `app.py`. Tests force local mode via an autouse fixture in `tests/conftest.py`.
- The DB path defaults to `local_data/coach.sqlite3`, and the `IELTS_DB_PATH` env var overrides it. Settings are read with `setting()`: env var first, then `st.secrets`.

## Domain rules (don't change these by accident)

- **Mastery** (`storage.mastery`): only the first answer to a given question on a given day counts. The stages New → Learning → Familiar → Secure and the growing review ladder (1/3/7/14/30/60/90 days, a miss drops back to 1 day / Learning) are documented in the README table. Meaning and meaning-in-context count as one skill. If you change the rules, update the README too.
- **Quiz sessions**: the learner picks the length, and each question tests a different word. Order is due words first, then new words, then filler. A word introduced as new today isn't introduced as new again the same day. If there aren't enough words, run a shorter session and say so.
- **Flash cards** never change mastery or review dates. Weights are in `flashcards.WEIGHTS`, and the last five cards are excluded when possible.
- **Question IDs** are hashes of word + type + prompt, and saved question content is immutable. To fix a question, change its prompt so it gets a new ID. **Word IDs are stable**: never rename an existing `id` in `words.psv`, or its progress is lost. Different senses of the same word get IDs like `volunteer--verb`.
- **Removed words and questions** leave new practice but stay in history. Never delete attempts.
- **Reading passages are never written or rewritten by AI.** AI only drafts the comprehension questions, opinion prompts and candidate vocabulary for a passage that has already been fetched. Always keep the source, link and license (CC BY-SA 4.0) for each passage. A missing picture must never block a passage. New vocabulary from reading is added only when the learner chooses it (`content.append_words`).
- Progress → Reading habit tracks consistency (streaks, counts), not scores.

## OpenAI usage

- The app must work fully offline without a key. AI runs only after an explicit click. No background calls, no automatic retries.
- Use the Responses API with Pydantic structured outputs and `store=False`. The default model is `gpt-4.1-mini`, overridable with `OPENAI_MODEL` (env var or secrets; there is no in-app setting).
- Send only the selected words or passage, never the learner's answer history.
- Validate generated output: schema, answer index bounds, distinct choices, known word IDs, duplicate prompts, per-word coverage. Record token usage with `store.log_generation`.
- Tests must never call OpenAI or Wikipedia. Pass a fake `client=` / `session=` (the functions accept them) or monkeypatch.

## Testing conventions

- Every test uses a temporary DB (`tmp_path` + `monkeypatch.setenv("IELTS_DB_PATH", ...)`) and never touches `local_data/`.
- UI tests use `AppTest.from_file(app.py)` and find buttons by their label (the `button(at, label)` helper). **If you rename a button label, update the tests.**
- Run `python -m pytest -q` before you say a change is done.

## Editing vocabulary (`data/words.psv`)

Fields: `id|word|topic|pos|definition|vi|example|collocation|cloze|distractors`. `cloze` must contain exactly one `___`, and its answer must be the listed word form. `distractors` holds exactly three distinct wrong choices separated by `;`. Topics are free-form, and filters are built from the data. Check new rows for ambiguous questions.

## Don't

- Commit or print `.streamlit/secrets.toml` or API keys, or put keys in the DB or exports.
- Modify or delete `local_data/` (real learner progress). Suggest a backup first if a change could affect it.
- Drop `--server.address 127.0.0.1` from the launcher or README run command (local-only binding). It is deliberately not in `.streamlit/config.toml`, so Community Cloud can bind normally.
- Add heavy dependencies. Keep `requirements.txt` to streamlit, openai, pydantic, requests and psycopg (guest mode only, imported lazily) unless a new one is clearly needed.
