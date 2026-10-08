# Word by Word — IELTS Coach

A local Python + Streamlit app for IELTS Academic preparation. Vocabulary, Reading, Writing and Progress work now; Speaking is a clearly labelled future section.

## Start locally

Use **Python 3.10 or newer** (tested with Python 3.12). Open a terminal in this project folder:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py --server.address 127.0.0.1
```

`--server.address 127.0.0.1` keeps the app reachable only from this computer. On Windows, activate with `.venv\Scripts\activate` instead. Open **http://localhost:8501**. Keep the terminal open while using the app; press Ctrl+C to stop it.

On macOS you can also double-click **Start IELTS Coach.command**. It creates a local environment on first launch if needed. Initial package installation needs internet; later offline sessions do not. On the computer used to develop this app, the launcher also recognises the already-installed workspace environment.

## What is included

- An expandable vocabulary collection grouped by topic, with English definitions, Vietnamese glosses, examples and useful phrases.
- Offline multiple-choice questions for meaning, meaning in context, and sentence completion with a meaning hint.
- Daily practice as a short quiz: choose how many questions to answer, each testing a different word, with shuffled choices, feedback, due reviews and resumable sessions.
- Offline flash cards: click to reveal the meaning, Vietnamese gloss and example, then move to the next card.
- A Reading section: pick a topic, read a real (never AI-written) short passage, answer IELTS-format comprehension questions, and optionally add new words met while reading to your vocabulary practice.
- A Writing section: practise a real IELTS Academic Writing Task 2 essay prompt, with or without the real 40-minute timer, and get an AI-graded report against all four official band criteria with concrete next-band advice.
- Optional OpenAI question generation, word-by-word mastery, answer history and backup exports.

The vocabulary file is the app's editable content store. It has no required book, publisher, edition, lesson count, or collection size. Topic filters and displayed totals are calculated from the current data. Words can be added, revised or removed as learning needs change. Definitions, examples and questions can be corrected without matching any external reference.

The interface shows the current words. Past submitted answers remain in the database and **Progress → All answer history**, even when a word is removed. New sessions save the word cards they use, so they can be resumed after vocabulary changes. Older sessions without saved cards still retain their questions and answers.

## Optional OpenAI setup

The app is useful without an API key. To generate new questions:

1. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and enter your key there, or set the `OPENAI_API_KEY` environment variable. Never commit the real secrets file.
2. Open **Vocabulary → Fresh questions**.
3. Select one to five words and click **Generate and save questions**.

You can set `OPENAI_MODEL` the same way; the default is `gpt-4.1-mini`. There is no key or model setting in the app itself.

Generation calls the OpenAI Responses API with structured outputs. It sends only the selected vocabulary records and existing prompts for those words, not the learner's answer history. API calls require internet and an OpenAI API account with available billing/credits. The request uses `store=False`; this does not mean that no provider-side processing or retention applies. There are no automatic background calls or retries. One click creates one batch, and input/output token counts are recorded locally when available.

Generated content is checked for schema, answer bounds, distinct choices, known word IDs, duplicate prompts and the requested per-word coverage. These checks **do not prove educational correctness**. The prompt asks for an unambiguous answer and self-checking; use the flag button for mistakes. No live API call was made during development because a user API key was not supplied. Integration tests use simulated responses.

References: [OpenAI structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs), [GPT-4.1 mini](https://developers.openai.com/api/docs/models/gpt-4.1-mini).

## How progress works

All submitted answers are recorded. Only the **first answer to the same question on a given day** counts toward mastery. Changing option order does not create fresh evidence.

The review interval grows with the number of **distinct days** the current correct streak spans, rather than sitting at a fixed length per stage — a word that keeps testing well keeps earning longer breaks instead of resurfacing on a fixed weekly cycle forever:

| Distinct correct days | Review interval | Stage shown |
|---|---|---|
| 1 | 1 day | Learning |
| 2 | 3 days | Learning |
| 3 | 7 days | Familiar |
| 4 | 14 days | Secure |
| 5 | 30 days | Secure |
| 6 | 60 days | Secure |
| 7+ | 90 days (cap) | Secure |

The first three rungs (1/3/7 days) are earned on streak length alone, so a new word still gets reviewed promptly. Growing past a week additionally requires the streak to have been tested across **2 different skills** — otherwise the interval stays frozen at 7 days no matter how many more correct days pile up, so a word only earns a long break once it has shown real, varied mastery rather than one lucky run of the same easy format. Cramming many correct answers into a single sitting does not advance the interval either: only distinct calendar days count. 90 days is a hard cap, so even a thoroughly mastered word resurfaces at least once in three months. There are four stages: New (never answered), then Learning, Familiar and Secure as shown above. A wrong answer always means 1 day, regardless of history, so a missed word shows as Learning again.

Meaning and meaning-in-context count as one skill for promotion. Sentence completion, usage and collocation provide other skills. A later mistake resets the correct streak straight back to the 1-day rung (Learning). Stages describe recognition, not independent speaking/writing ability or an IELTS band prediction. Overdue words keep their stage but appear in the due review queue. Dates follow the computer running Python.

Each session is a quiz of the size you choose: pick how many questions, and each one tests a different word, with no repeats. Due words come first, then words you have not met yet, then other eligible words as filler if a topic is small. A word introduced as new in one session today will not be introduced as new again in a later session the same day, though it may return once it is due for review. If a topic does not have enough eligible words for the number requested, the app runs a shorter session and says so. Sessions retain their shuffled choices and first submitted answers across browser reloads.

## Flash cards

Open **Vocabulary → Flash cards** and optionally choose a topic. Click the card to reveal its meaning; click again to return to the word. **Next card** always starts on the word side.

Selection randomly chooses a group, then a word: needs/due review 40%, learning 35%, new 15%, familiar 7%, secure 3%. Due words belong only to the review group. Empty groups share their weight proportionally among available groups. The last five cards are excluded when possible; smaller pools relax the oldest exclusions first. Changing topics starts a fresh selection.

The first reveal of each card presentation counts once toward **Cards reviewed today**. Moving on without revealing records a skip. Revisiting a word in a later card presentation counts as another review. Neither action changes mastery or quiz review dates. Activity persists locally and appears in **Progress → Flash card history**. No API calls are needed.

## Reading

Open **Reading** and choose a topic to start. Each passage is real, sourced text; the passage itself is never written or rewritten by AI. AI is used only to draft the comprehension questions and candidate vocabulary that go with an already-selected, already-verified passage — never the reading material itself.

**Phase 1 scope.** The current topics (Animals, Space, Technology, History, Environment, Sports, World, Culture) are nonfiction, sourced from Wikipedia, and filtered by simple length (650-750 words, long enough to comfortably support a full IELTS-length question set) and sentence-complexity heuristics tuned to real IELTS Academic Reading difficulty, not capped to a specific band. Fiction/narrative topics, sourced from public-domain sources such as Project Gutenberg, are a planned later phase and are not included yet.

**Candidate discovery.** A topic is just the category shown in the picker; the actual candidate article titles for it are discovered live via Wikipedia's own search, not drawn from a fixed, hand-picked list. That means a topic is never limited to however many pages happened to be curated for it in advance — earlier, "Sports" had only 7 possible candidate titles total, so a fetch could come back empty just from that whole small pool already being used or not clearing the readability bar. A broad, single-word topic tends to surface a lot of narrow sub-articles and stubs whose introduction is naturally short, so each fetch searches a fairly large pool of candidates (150) to still find enough that clear the length bar; most of that pool is ruled in or out using the word count Wikipedia's search already reports for each hit, so a page that's clearly too short is skipped without the slower step of fetching and reading it. Survivors are then screened by Wikipedia's own one-line short description (e.g. "American basketball player"), looked up for 50 titles per request: Wikipedia's search weights a literal title match on the topic word very heavily, so a topic like "Sports" otherwise fills up with brand and broadcaster pages ("NBC Sports"), lists, disambiguation pages, media titles and jargon stubs ("Pitch (sports)"). A page whose description matches one of those kinds, or that has no description at all, is skipped before it is ever fetched; there are plenty of candidates to spare. If Wikipedia's search itself can't be reached, that is reported honestly too, the same way a single candidate page's own fetch failure is (see below), rather than the fetch just coming back silently empty.

**Fetch speed.** Reading a candidate's full article text — not just its opening section, which alone runs well under this app's 650-750 word target for most articles — means Wikipedia has to fully render that page server-side before answering; that per-page render time can't be shortened by asking for a smaller response, and it's paid on every candidate tried, not just the one that succeeds. A single candidate's own page fetch, or the AI's own drafting call once a candidate looks promising, each normally finishes in a few seconds. What adds up is candidate count: with up to 150 candidates searched per topic and only some of them clearing every check (length, readability, topic suitability, then the AI-drafted activity itself), a fetch can genuinely need to work through several dozen before finding enough good ones, adding up to a real multi-minute wait even though nothing is broken. (Candidates are fetched one at a time, not in parallel — an earlier attempt at overlapping fetches in threads to cut this time was reverted after appearing to cause a hang for a real user; that turned out to actually be this same normal multi-candidate wait, made to look exactly like a hang by the UI below giving no sign of progress, but fetching was left sequential since the underlying cost is Wikipedia's per-page render time either way, not something concurrency changes reliably here.) Because that wait is real, **Add more reading material** shows a live, updating status line — which candidate was just checked and why it was or wasn't used — instead of one static "fetching…" message, so a fetch that's taking a while still visibly shows it's working.

**Topic suitability.** Because candidates now come from a live, general-purpose search rather than a hand-picked list someone already screened, a keyword-based safety net checks both a candidate's title (before it's fetched) and its actual passage text (what's shown, after trimming) against a list of subjects considered unsuitable for a young learner, skipping any match the same way any other unsuitable candidate is skipped. This is a coarse heuristic, not a content classifier, and does not check a page's picture; treat it as a safety net worth keeping an eye on rather than a guarantee, and extend the keyword list in `coach/reading.py` (`INAPPROPRIATE_TOPIC_KEYWORDS`) if something unsuitable still gets through.

**Fetch once, cache locally.** Like Fresh Questions, fetching new reading material needs an OpenAI API key and internet access, and is a deliberate action a parent takes from **Reading → Add more reading material**: choose a topic and how many new passages to fetch. Each fetched passage is bundled at fetch time with its comprehension questions and candidate vocabulary, and — when Wikipedia's page has one — a picture, all saved to the local database and disk together. Once fetched, passages are read entirely offline, matching the rest of the app, pictures included. A passage whose picture failed to download, or is missing entirely, still reads normally; the picture is a nice-to-have, never a requirement. A candidate page is skipped (not fetched as a passage) if it can't be reached on Wikipedia, reads as too dense or too short/long for the reading level, or too little of the AI's drafted activity for it holds up (see below); if a fetch comes back with fewer passages than requested, an expander explains why each skipped candidate didn't make it in, rather than just saying nothing was found.

**Salvaging a mostly-good bundle.** If the AI's drafted comprehension questions or vocabulary for a passage include one or two bad items, only those specific items are dropped rather than discarding an otherwise-good ~12-question bundle over a single flaw. This happens in three passes: for items that are structurally malformed — a fill-in-the-blank answer longer than its own stated word limit, a vocabulary item with a malformed cloze blank, and the like; for items whose citation doesn't actually check out against the real passage text once fetched — a quoted "source quote" that isn't really in the passage, a fill-in-the-blank answer that isn't really there, or a matching-information reference to a paragraph that doesn't exist; and for a matching-headings question whose list of candidate headings doesn't match what the other matching-headings question(s) in the same bundle agreed on (real matching-headings questions are supposed to share one heading list, and the AI doesn't always keep it word-for-word identical) — here the item that doesn't match what the others agree on is dropped, keeping the shared list intact. All three cases drop only the offending item, never the whole reading, since one question's flaw says nothing about whether the rest are fine. The AI is asked to aim for the full 12 items (not just enough to clear the floor), so salvage normally has real headroom to work with; a candidate is only skipped outright if fewer than 6 valid comprehension items remain after all of that, or the problem isn't attributable to a single item — discarding an entire generated bundle over losing a couple of items out of 12 would waste far more than accepting a shorter-but-still-useful set.

**The reading flow.** The comprehension quiz sits directly beneath the passage rather than on its own screen, so the learner can scroll back up and check the text while answering; there is no separate "done reading" step. Each passage carries 6-12 comprehension questions (the AI is asked to aim for the full 12, matching a real IELTS Academic Reading section; salvage can bring a bundle as low as 6 before it's skipped entirely) drawn from every question type the AI can draft: multiple choice, True/False/Not Given, Yes/No/Not Given, sentence completion, short-answer, matching sentence endings, matching features, matching information (to a lettered paragraph), and matching headings. (The official format's summary/note/table/flow-chart completion is folded into sentence completion, since both are a free-text, word-limited answer copied from the passage; diagram label completion is not supported, since the app has no diagram pipeline.) Paragraphs are shown lettered (A, B, C…) whenever a matching-information or matching-headings question is present, so the learner can find the paragraph a question refers to. Every question is scored (with feedback and an explanation) before the learner moves on to a vocabulary review: any candidate vocabulary words drafted from that passage are shown for the learner to review, and they choose which ones, if any, to add to their Vocabulary practice. Nothing is added automatically. Passages already read are preferred for new sessions; once every passage in a topic has been read, the app allows a reread rather than showing nothing.

**Sources and attribution.** Each passage shows its source name, a link to the original page, and its license. Wikipedia content and pictures used here are under CC BY-SA 4.0.

**Progress.** **Progress → Reading habit** tracks passages read, a day streak, topics explored, and passages read this week — deliberately not a score, to keep the focus on consistency rather than performance.

## Writing

Open **Writing**, pick a topic and essay type, and you'll get a real IELTS Academic Writing Task 2
prompt to respond to — an opinion (agree/disagree), a discussion (both views), a problem/solution, or
an advantages/disadvantages question. Unlike Reading's passages, essay prompts are never
AI-generated: there's no independent "real source" to fetch and verify for a question, so they come
from a curated file a parent or tutor maintains by hand (see **Adding essay prompts** below), the
same way the vocabulary bank does.

**Two modes, one report.** **Simulation** gives the real 40-minute time limit; once it runs out, the
essay box locks (no more typing) and the learner is asked to submit whatever they have. **Practice**
has no time limit at all — how long it actually took is simply recorded for the learner's own
information, never judged. Both modes are graded by exactly the same process afterward, so choosing
a mode never changes how an essay is scored, only the conditions under which it was written.

**Grading.** Once submitted, AI grades the essay against the four official IELTS Writing Task 2
criteria — Task Response, Coherence and Cohesion, Lexical Resource, and Grammatical Range and
Accuracy — each on the real 1-to-9 half-band scale, with a short explanation of why it earned that
band, up to a few short quotations from the essay illustrating specific strengths or problems, and
one concrete note on what it would specifically take to reach the next half-band up. An overall band
is then computed from the four criterion bands using the same whole/half-band rounding convention
IELTS publishes for combining scores — deterministically, in this app's own code, never trusted from
the AI's own arithmetic. This is an estimate against the public band descriptors, not an official
IELTS score; the examiner's own internal marking process isn't available to this app (the same
caveat that applies to the band descriptors themselves).

Every specific quotation shown in the report is checked against the essay actually submitted before
it's ever shown — an AI-cited "quote" that isn't really in the essay is silently dropped rather than
shown as if it were real, the same grounding principle Reading's comprehension questions follow
against the passage text. Grading has no equivalent of Reading's "try a different candidate": once an
essay is written, a failed or incomplete grading attempt doesn't lose it — the essay stays on screen
and grading can simply be retried.

**Progress.** Writing keeps a plain history — every essay you wrote and the report it received, most
recent first — under **Progress → Writing history**, deliberately with no trend, average, or score
comparison across attempts. This app is built with a parent or tutor reviewing everything off-app; the
history's job is to be a trustworthy, skimmable record for that review, not to editorialize on
progress itself.

**Adding essay prompts.** Edit `data/essay_prompts.psv`, a UTF-8 text file using `|` as the field
separator, the same format as the vocabulary bank. Add a row with these fields:

| Field | Purpose |
|---|---|
| `id` | Stable identifier for this prompt; keep it unchanged once anyone has attempted it. If blank, it is derived from the prompt text. |
| `topic` | Any useful topic name; new topics automatically appear in the picker. |
| `essay_type` | One of `opinion`, `discussion`, `problem_solution`, or `advantage_disadvantage`. |
| `prompt` | The full essay question, exactly as a learner should read it. |

After editing, rerun the app — it detects content changes the same way the vocabulary file does. A
prompt already attempted is never rewritten by re-adding it with the same `id`; give a genuinely
revised prompt a new `id` instead, so past attempts stay linked to what was actually asked at the
time.

## Your data and backups

Data is stored in **`local_data/coach.sqlite3`**, relative to this project. Set `IELTS_DB_PATH` to use another location. Reading pictures are cached as files in **`local_data/reading_images/`**, next to the database. Keep the app folder, database, and that folder together if you move computers.

In **Progress → Keep a copy**, download a full SQLite backup. To restore:

1. Stop the app with Ctrl+C.
2. Keep a copy of the current `local_data/coach.sqlite3` in case you need to undo the restore.
3. Copy the downloaded backup into `local_data/` and name it `coach.sqlite3`.
4. Restart the app.

The full backup includes questions, sessions, attempts, flags, flashcard activity, generation logs, reading passages/sessions/activity, and essay prompts/writing attempts, but **not** the cached reading pictures in `local_data/reading_images/`; copy that folder separately if you want to keep them. JSON and CSV exports are for inspection and analysis; the app does not import those formats. API keys are not included in any export.

## Project structure

```text
app.py                       Streamlit screens and interactions
coach/content.py             Vocabulary validation and offline questions
coach/flashcards.py          Weighted offline card selection
coach/storage.py             SQLite persistence, review selection, mastery
coach/ai.py                  OpenAI generation and batch validation
coach/reading.py             Wikipedia fetch, readability filter, reading AI bundle generation
coach/writing.py             Essay prompt bank, AI Task 2 grading, band rounding and grounding
data/words.psv               The single editable vocabulary collection
data/essay_prompts.psv       The single editable Writing Task 2 prompt bank
coach/cloud.py               Public demo only: shared content and the daily AI allowance in Postgres (Neon)
coach/guest.py               Public demo only: throwaway per-visitor progress databases
scripts/export_content_seed.py  Copies fetched passages and AI questions (no progress) into data/content_seed.json
data/content_seed.json       Shared content the public demo starts with
tests/                       Content, persistence and interaction tests
.streamlit/config.toml       Theme settings
.streamlit/secrets.toml.example
requirements.txt
```

## Adding and editing words

Edit `data/words.psv`, a UTF-8 text file using `|` as the field separator. There is no fixed limit on the number of words, topics, or entries per topic. Add a row with these fields:

| Field | Purpose |
|---|---|
| `id` | Stable identifier for this word sense; keep it unchanged to retain progress. If blank, it is derived from the word. |
| `word` | The target English word or phrase. |
| `topic` | Any useful topic name; new topics automatically appear in filters. |
| `pos` | Part of speech, such as noun, verb or adjective. |
| `definition` | A clear meaning for this word sense. |
| `vi` | Optional Vietnamese explanation. |
| `example` | A complete English example sentence. |
| `collocation` | Optional useful phrase. |
| `cloze` | A sentence containing exactly one `___`; the answer must be the listed word form. |
| `distractors` | Three distinct incorrect cloze choices separated by semicolons. |

The app generates three offline questions per word. Meaning questions choose distractors from the vocabulary collection, preferring the same topic and part of speech and excluding known near-synonyms. Review new entries and their questions for ambiguity. Distinct meanings of the same spelling should have distinct IDs, for example `volunteer--verb` and `volunteer--noun`.

After editing, rerun the app. It detects content changes, updates filters and totals, and adds the current question bank. Saved questions used in past answers remain available for history; removed and superseded questions are excluded from new practice. The displayed offline/AI labels describe how a question was created, not a content authority.

Question IDs are based on the word, question type and prompt. Saved question content is immutable: if correcting an existing question, change its prompt as well so it gets a new ID. Word-level progress remains linked to the stable word ID.

## Tests

```sh
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Tests use temporary databases and do not modify the learner's progress or call OpenAI. They always run the local app, even if your secrets configure `DATABASE_URL`. The guest-mode tests that need Postgres are skipped unless `TEST_DATABASE_URL` points at a **disposable** Postgres database (they drop and recreate the shared tables, so never use the real Neon database). The app was tested with Streamlit 1.64.0, OpenAI SDK 2.54.0 and Pydantic 2.13.5.

## Public demo (guest mode) on Streamlit Community Cloud and Neon

The app can run as a public demo, for example to share on LinkedIn. It switches to **guest mode** whenever a `DATABASE_URL` secret is set:

- Every visitor is a guest. Their progress, answers and essays live in a throwaway database for their browser tab only; reloading or closing the page starts fresh. A "Guest mode" banner says so on every page, and Progress offers JSON/CSV downloads instead of the SQLite backup.
- Shared content lives in Postgres (Neon) so it survives app restarts: fetched reading passages with their pictures, and AI-generated vocabulary questions. Content a visitor creates is shared with later visitors. No learner progress is ever stored there.
- AI features (Fresh questions, fetching a reading passage, essay grading) use your OpenAI key, so all visitors together share a daily allowance of AI actions (`AI_DAILY_LIMIT`, default 20, counted per UTC day). A guest fetches one passage at a time. A failed AI call gives its action back. When the allowance is used up, everything else keeps working.
- Open the app as `https://<your-app>.streamlit.app/?owner=<OWNER_PASSCODE>` to skip the daily limit yourself. Your progress there is still guest-only.

Without `DATABASE_URL` (or with `IELTS_MODE=local`) the app is the normal single-learner local app described above.

### Deploy it

1. **Starting content.** On your computer, run `python scripts/export_content_seed.py`. It reads `local_data/coach.sqlite3` (read-only) and writes `data/content_seed.json` with your fetched passages and unflagged AI questions, and nothing about progress. The demo loads it into Neon on its first start.
2. **GitHub.** Put the project in a GitHub repository. `.gitignore` already keeps out `local_data/`, `.venv/` and `.streamlit/secrets.toml`; double-check that no secrets file or database is committed.
3. **Neon.** Create a free project at [neon.com](https://neon.com), then copy its connection string (it starts with `postgresql://` and ends with `?sslmode=require`). The app creates its own tables.
4. **Streamlit Community Cloud.** At [share.streamlit.io](https://share.streamlit.io), sign in with GitHub, choose **Create app**, pick the repository, branch and `app.py`, and under **Advanced settings** choose Python 3.12 and paste the secrets:

   ```toml
   DATABASE_URL = "postgresql://...neon.tech/neondb?sslmode=require"
   OPENAI_API_KEY = "sk-..."
   OPENAI_MODEL = "gpt-4.1-mini"
   AI_DAILY_LIMIT = "20"
   OWNER_PASSCODE = "a-long-phrase-only-you-know"
   ```

5. Deploy. The first visit after a quiet period can take a little longer: Community Cloud sleeps apps after 12 hours without traffic, and Neon's free plan pauses the database after 5 minutes idle and wakes on the next request.

Set a monthly budget limit in your OpenAI account as a second safety net; the daily allowance limits how many AI actions run, not what each one costs.

Free plan limits change; check [Neon pricing](https://neon.com/pricing) and the [Community Cloud docs](https://docs.streamlit.io/deploy/streamlit-community-cloud) before relying on them.
