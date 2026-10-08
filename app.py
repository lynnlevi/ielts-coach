from __future__ import annotations

import csv
import hashlib
import hmac
import io
import os
import tempfile
import time
import uuid
from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st
from openai import APIConnectionError, APIStatusError, AuthenticationError, RateLimitError

from coach.ai import generate_questions
from coach.content import DATA, KINDS, load_words, seed_questions
from coach.storage import Store, now_iso
from coach.flashcards import choose_card
from coach.guest import TEMPLATE_MAX_AGE_HOURS, build_template, new_guest_store, remove_stale
from coach.reading import (FREE_TEXT_KINDS, PARAGRAPH_KINDS, fetch_passages, find_supporting_sentence,
                            highlight_passage_html, label_paragraphs, normalize_paragraph_breaks,
                            READING_TOPICS)
from coach.writing import (CRITERIA, CRITERION_LABELS, ESSAY_TYPES, MIN_RECOMMENDED_WORDS,
                            SIMULATION_MINUTES, generate_evaluation, load_essay_prompts)
from coach.writing import DATA as WRITING_DATA

ROOT = Path(__file__).resolve().parent
SEED_FILE = ROOT / "data" / "content_seed.json"
GUEST_ROOT = Path(tempfile.gettempdir()) / "ielts-coach-guests"
READING_EMOJI = {"Animals": "\U0001F43E", "Space": "\U0001F680", "Technology": "\U0001F4A1",
                 "History": "\U0001F4DC", "Environment": "\U0001F33F", "Sports": "\u26BD",
                 "World": "\U0001F30D", "Culture": "\U0001F3AD"}
st.set_page_config(page_title="Word by Word · IELTS Coach", page_icon="🌱", layout="wide")
st.markdown("""<style>
.block-container {max-width: 1140px; padding-top: 2.2rem; padding-bottom: 3rem;}
h1,h2,h3 {letter-spacing: -.025em;}
[data-testid="stMetricValue"] {font-size: 2rem;}
[data-testid="stMetric"] {background: #F0F2EB; border-radius: 14px; padding: 16px 20px;}
.eyebrow {font-size: .76rem; text-transform: uppercase; letter-spacing: .16em; color: #52776B; font-weight: 700;}
mark.reading-highlight {background: #FFE8A3; padding: 0 2px; border-radius: 3px; color: inherit;}
mark.reading-highlight sup {font-size: .7em; font-weight: 700; margin-left: 1px; color: #52776B;}
.small-note {color: #62736C; font-size: .87rem;}
button {border-radius: 10px !important;}
.st-key-flashcard_front button, .st-key-flashcard_back button {
    width: 100%; min-height: 280px; padding: 32px; background: #F0F2EB;
    border: 1px solid #B8CCC1; color: #213A35;
}
.st-key-flashcard_front button p {font-size: 1rem;}
.st-key-flashcard_front button strong {font-size: 2.5rem;}
.st-key-flashcard_back button p {font-size: 1.15rem; line-height: 1.7;}
.st-key-flashcard_front button:focus-visible, .st-key-flashcard_back button:focus-visible {
    outline: 3px solid #187568; outline-offset: 3px;
}
[data-testid="stImage"] img {border-radius: 16px;}
.guest-banner {background: #FFF4D6; border: 1px solid #F0D58A; color: #5A4A1A; border-radius: 12px;
    padding: 8px 14px; font-size: .9rem; margin: 1.6rem 0 1rem;}
</style>""", unsafe_allow_html=True)


@st.cache_data
def content(revision):
    words = load_words()
    return words, seed_questions(words)


@st.cache_data
def writing_content(revision):
    return load_essay_prompts()


@st.cache_resource
def database(path, revision, essay_revision, schema_version=4):
    # schema_version is never read below; it only exists so bumping it busts
    # the cached Store when the on-disk schema changes shape.
    store = Store(path)
    store.add_questions(content(revision)[1])
    store.add_essay_prompts(writing_content(essay_revision))
    return store


def setting(name, default=""):
    if os.environ.get(name):
        return os.environ[name]
    try:
        return str(st.secrets.get(name, default))
    except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
        return default


# Guest mode is the public demo: on whenever a shared database is configured
# (DATABASE_URL), unless IELTS_MODE=local forces the normal single-learner
# app. In guest mode each browser session gets its own throwaway progress;
# only shared content and the daily AI allowance live in the database.
DATABASE_URL = "" if os.environ.get("IELTS_MODE") == "local" else setting("DATABASE_URL")
GUEST_MODE = bool(DATABASE_URL)


@st.cache_resource
def shared_content(url):
    from coach.cloud import SharedContent
    shared = SharedContent(url)
    shared.ensure_schema()
    shared.seed_from_file(SEED_FILE)
    return shared


@st.cache_resource(max_entries=2)
def guest_template(version, revision, essay_revision, _shared=None):
    # Built once per content version (per app process) and copied for each
    # visitor; `_shared` is left out of the cache key by its leading underscore.
    questions, passages = _shared.load() if _shared is not None else ([], [])
    key = hashlib.sha256(f"{version}|{revision}|{essay_revision}".encode()).hexdigest()[:16]
    folder = GUEST_ROOT / "templates" / key
    path = build_template(folder, content(revision)[1], writing_content(essay_revision), questions, passages)
    remove_stale(folder.parent, TEMPLATE_MAX_AGE_HOURS, keep=(folder,))
    return path


def guest_store():
    store = st.session_state.get("guest_store")
    if store is not None and store.path.exists():
        return store
    try:
        shared = shared_content(DATABASE_URL)
        template = guest_template(shared.version(), revision, essay_revision, shared)
    except Exception:
        # The shared database is asleep or unreachable: still let the visitor
        # practise with the built-in words and essay prompts.
        template = guest_template("offline", revision, essay_revision)
        st.session_state.shared_offline = True
    store = new_guest_store(template, GUEST_ROOT / "visitors")
    st.session_state.guest_store = store
    return store


revision = hashlib.sha256(DATA.read_bytes()).hexdigest()
essay_revision = hashlib.sha256(WRITING_DATA.read_bytes()).hexdigest()
words, starter = content(revision)
by_id = {w["id"]: w for w in words}
if GUEST_MODE:
    store = guest_store()
else:
    store = database(os.environ.get("IELTS_DB_PATH", str(ROOT / "local_data" / "coach.sqlite3")), revision, essay_revision)
today = now_iso()[:10]
progress = store.progress(words, today)
topics = ["All topics"] + list(dict.fromkeys(w["topic"] for w in words))


def is_owner():
    """The app owner can open the live app with ?owner=<OWNER_PASSCODE> to
    skip the daily AI cap. Progress is still guest-only."""
    code = setting("OWNER_PASSCODE")
    given = st.query_params.get("owner")
    if code and given and hmac.compare_digest(str(given), code):
        st.session_state.owner = True
    return st.session_state.get("owner", False)


def ai_daily_limit():
    try:
        return max(0, int(setting("AI_DAILY_LIMIT", "20") or 20))
    except ValueError:
        return 20


def ai_allowance_note():
    """One line about the shared AI allowance, or None outside guest mode."""
    if not GUEST_MODE:
        return None
    if is_owner():
        return "Owner access: no daily AI limit."
    try:
        used = shared_content(DATABASE_URL).ai_used_today()
    except Exception:
        return "AI features are resting right now. Please try again later."
    left = max(0, ai_daily_limit() - used)
    return (f"This demo shares a small free AI allowance with every visitor: "
            f"{left} of {ai_daily_limit()} AI actions left today.")


def start_ai_action():
    """Count one AI action against the shared daily allowance (guest mode
    only). Shows a message and returns False if none are left."""
    if not GUEST_MODE:
        return True
    try:
        ok = shared_content(DATABASE_URL).reserve_ai_action(None if is_owner() else ai_daily_limit())
    except Exception:
        st.error("Couldn't check today's AI allowance. Please try again in a minute.")
        return False
    if not ok:
        st.warning("Today's free AI allowance for this demo is used up. Please come back tomorrow! "
                   "Everything else still works.")
    return ok


def undo_ai_action():
    """Give back an action whose AI call failed."""
    if GUEST_MODE:
        try:
            shared_content(DATABASE_URL).release_ai_action()
        except Exception:
            pass


def share_new_content(questions=(), passages=()):
    """In guest mode, save new AI questions and passages for every visitor."""
    if not GUEST_MODE:
        return
    try:
        shared = shared_content(DATABASE_URL)
        if questions:
            shared.save_questions(list(questions))
        if passages:
            shared.save_passages(list(passages))
    except Exception:
        st.caption("This new content is saved for you now, but couldn't be shared with other visitors.")


NO_KEY_MESSAGE = ("AI features need an OpenAI API key. Add OPENAI_API_KEY to the app's secrets "
                  "(see the README).")
BAD_KEY_MESSAGE = "OpenAI did not accept the API key. Check OPENAI_API_KEY in the app's secrets."


def word_card(w, vietnamese=True):
    with st.container(border=True):
        st.markdown(f"### {w['word']}")
        st.caption(f"{w['pos']} · {w['topic']}")
        st.write(w["definition"])
        if vietnamese and w.get("vi"):
            st.caption(w["vi"])
        st.info(w["example"], icon="💬")
        if w.get("collocation"):
            st.markdown(f"**Useful phrase:** {w['collocation']}")


def word_label(wid):
    return by_id.get(wid, {}).get("word", wid.split("--")[0].replace("-", " "))


def go_home():
    for key in ("active_session", "cursor"):
        st.session_state.pop(key, None)


def go_to_progress():
    st.session_state.nav_page = "Progress"


def practice_session(sid):
    session = store.session(sid)
    if not session:
        go_home()
        st.rerun()
    questions = session["payload"]["questions"]
    attempts = store.session_attempts(sid)
    flagged = {q["id"] for q in store.questions(True) if q["flagged"]}
    usable = [q for q in questions if q["id"] not in flagged]
    scored = [a for a in attempts.values() if a["question_id"] not in flagged]
    if session["finished"] or all(q["id"] in attempts for q in usable) and st.session_state.get("cursor", 0) >= len(questions):
        store.finish(sid)
        st.markdown("## A little practice, a little stronger.")
        a, b, c = st.columns(3)
        a.metric("Questions answered", len(scored))
        b.metric("Correct answers", sum(a["correct"] for a in scored))
        c.metric("Words practised", len({a["word_id"] for a in scored}))
        missed = sorted({word_label(a["word_id"]) for a in scored if not a["correct"]})
        if missed:
            st.info("Keep practising: " + ", ".join(missed) + ". These words will return in your reviews.")
        else:
            st.success("Your answers are saved. Come back on another day to strengthen your recall.")
        st.caption("This session summary includes extra practice. Mastery uses only the first answer to each question per day.")
        if st.button("Back to vocabulary", type="primary"):
            go_home()
            st.rerun()
        return
    if "cursor" not in st.session_state:
        st.session_state.cursor = next((i for i, q in enumerate(questions) if q["id"] not in attempts and q["id"] not in flagged), len(questions))
    idx = st.session_state.cursor
    while idx < len(questions) and questions[idx]["id"] in flagged:
        idx += 1
    st.session_state.cursor = idx
    if idx >= len(questions):
        store.finish(sid)
        st.rerun()
    q = questions[idx]
    answered = attempts.get(q["id"])
    st.caption(f"QUESTION {idx + 1} OF {len(questions)} · {KINDS[q['kind']].upper()}")
    st.progress(idx / len(questions))
    left, right = st.columns([3, 1])
    with left:
        with st.container(border=True):
            st.markdown(f"### {q['prompt']}")
            key = f"answer_{sid}_{q['id']}"
            if answered and key not in st.session_state:
                st.session_state[key] = answered["selected"]
            selected = st.radio("Choose your answer", q["options"], index=None, key=key, disabled=bool(answered))
            if not answered and st.button("Check answer", type="primary", disabled=selected is None):
                store.record(sid, q["id"], selected)
                st.rerun()
            if answered:
                if answered["correct"]:
                    st.success("Correct — well done!")
                else:
                    st.error("Not quite. The correct answer is: " + q["options"][q["answer"]])
                st.write(q["explanation"])
                updated = store.progress([{"id": q["word_id"]}])[q["word_id"]]
                st.caption(f"{word_label(q['word_id'])} · {updated['stage']} · Next review: {updated['due'] or 'not scheduled'}")
                if st.button("Finish session" if idx == len(questions) - 1 else "Next question", type="primary"):
                    st.session_state.cursor = idx + 1
                    st.rerun()
        if st.button("Flag this question and skip", help="Removes this question from future practice and excludes all its answers from mastery."):
            store.flag(q["id"])
            st.session_state.cursor = idx + 1
            st.rerun()
    with right:
        st.markdown("#### Keep going")
        st.write("Take your time. Read the whole sentence before choosing.")
        st.caption("Every submitted answer is saved automatically.")
        if q["source"] == "openai":
            st.caption("AI-generated question · Flag it if the wording or answer seems wrong.")
        if st.button("End session"):
            store.finish(sid)
            st.rerun()


def next_flashcard(record_skip=False):
    old = st.session_state.get("flashcard")
    if record_skip and old and not old["revealed"]:
        store.record_flashcard(old["token"], old["word_id"], "skip")
    recent = st.session_state.get("flashcard_recent", [])
    word = choose_card(words, store.progress(words), now_iso()[:10],
                       st.session_state.get("flashcard_topic", "All topics"), recent)
    if word:
        st.session_state.flashcard = dict(word_id=word["id"], token=uuid.uuid4().hex,
                                          back=False, revealed=False)
        st.session_state.flashcard_recent = (recent + [word["id"]])[-5:]
    else:
        st.session_state.pop("flashcard", None)


def reset_flashcards():
    st.session_state.flashcard_recent = []
    next_flashcard()


def flip_flashcard():
    card = st.session_state.flashcard
    card["back"] = not card["back"]
    if card["back"] and not card["revealed"]:
        store.record_flashcard(card["token"], card["word_id"], "review")
        card["revealed"] = True


def flashcards_panel():
    st.markdown("### Flash cards")
    st.caption("Think of the meaning, then click the card to check.")
    st.selectbox("Flash card topic", topics, key="flashcard_topic", on_change=reset_flashcards)
    if st.session_state.get("flashcard_revision") != revision:
        reset_flashcards()
        st.session_state.flashcard_revision = revision
    if "flashcard" not in st.session_state:
        next_flashcard()
    card = st.session_state.get("flashcard")
    if not card:
        st.info("No words in this topic yet.")
        return
    word = by_id[card["word_id"]]
    st.caption(f"Cards reviewed today: {store.flashcards_reviewed()}")
    if card["back"]:
        lines = [f"**{word['word']}**", word["definition"]]
        if word.get("vi"):
            lines.append(word["vi"])
        lines.extend([f"*{word['example']}*", "Click to see the word again"])
        label = "\n\n".join(lines)
    else:
        label = f"**{word['word']}**\n\nClick to reveal"
    with st.container(key="flashcard_back" if card["back"] else "flashcard_front"):
        st.button(label, key="flashcard_flip", on_click=flip_flashcard, width="stretch")
    st.button("Next card →", key="flashcard_next", on_click=next_flashcard,
              kwargs={"record_skip": True}, type="primary")
    st.caption("Learning and review words appear more often. Card reviews are saved separately from quiz mastery.")


def vocabulary():
    session_id = st.session_state.get("active_session")
    if session_id:
        practice_session(session_id)
        return
    due = sum(bool(p["due"] and p["due"] <= today) for p in progress.values())
    learned = sum(p["stage"] != "New" for p in progress.values())
    secure = sum(p["stage"] == "Secure" for p in progress.values())
    a, b, c = st.columns(3)
    a.metric("Words explored", f"{learned} / {len(words)}")
    b.metric("Ready to review", due)
    c.metric("Secure words", secure)
    st.caption(f"{len(words)} words to explore across {len(topics) - 1} topics")
    st.write("")
    practice, cards, library, ai = st.tabs(["Daily practice", "Flash cards", "Word library", "Fresh questions"])
    with practice:
        st.markdown("### Your next small step")
        st.write("Choose how many questions you'd like. Reviews that are due come first, then fresh words — each question tests a different word.")
        resume = store.session()
        if resume:
            st.info("You have a saved practice session. Your answers are right where you left them.")
            col1, col2 = st.columns(2)
            if col1.button("Continue saved session", type="primary"):
                st.session_state.active_session = resume["id"]
                st.session_state.pop("cursor", None)
                st.rerun()
            if col2.button("End saved session"):
                store.finish(resume["id"])
                st.rerun()
        else:
            topic = st.selectbox("Practise a topic", topics)
            length = st.number_input("Number of questions", min_value=1, value=10, step=1,
                                     help="Each question tests a different word. Due reviews come first, then new words.")
            if st.button("Start today's practice", type="primary"):
                sid = store.create_session(words, topic, int(length))
                if sid:
                    actual = len(store.session(sid)["payload"]["questions"])
                    if actual < length:
                        where = "this topic" if topic == "All topics" else f"“{topic}”"
                        noun = "word is" if actual == 1 else "words are"
                        # st.info() here would be wiped by the immediate rerun below;
                        # st.toast() is designed to survive exactly one rerun.
                        st.toast(f"Only {actual} {noun} ready to practise in {where} right now — starting a {actual}-question session.", icon="ℹ️")
                    st.session_state.active_session = sid
                    st.session_state.pop("cursor", None)
                    st.rerun()
                else:
                    st.info("This topic doesn't have any practice-ready words yet. Try another topic.")
        st.caption("Works without internet after installation. Fresh AI questions are optional.")
    with cards:
        flashcards_panel()
    with library:
        col1, col2 = st.columns(2)
        topic = col1.selectbox("Filter topic", topics)
        search = col2.text_input("Find a word", placeholder="Try ‘aquatic’")
        visible = [w for w in words if (topic == "All topics" or w["topic"] == topic) and search.lower() in w["word"].lower()]
        st.caption(f"{len(visible)} words · Explore meanings, examples, and useful phrases.")
        if visible:
            selected = st.selectbox("Explore a word", [w["id"] for w in visible], format_func=lambda wid: by_id[wid]["word"])
            word_card(by_id[selected])
            st.caption("Recognition stage: " + progress[selected]["stage"])
        else:
            st.info("No matching words. Try another topic or spelling.")
    with ai:
        generation_panel()


def generation_panel():
    st.markdown("### New ways to practise")
    st.write("Generate three new questions per word with OpenAI, then keep practising them offline.")
    key = setting("OPENAI_API_KEY")
    model = setting("OPENAI_MODEL", "gpt-4.1-mini")
    bank = store.active_questions(words)
    st.caption(f"{sum(q['source'] == 'offline' for q in bank)} offline questions · {sum(q['source'] == 'openai' for q in bank)} saved AI questions")
    selected = st.multiselect("Choose up to five words", [w["id"] for w in words], format_func=lambda wid: by_id[wid]["word"], max_selections=5)
    if not key:
        st.info(NO_KEY_MESSAGE + " Offline practice is ready now.")
    if GUEST_MODE:
        st.caption(ai_allowance_note() + " New questions are shared with everyone who visits.")
    else:
        st.caption("Requires internet and uses your paid OpenAI API account. Only the chosen word content is sent; progress history stays on this computer.")
    if st.button("Generate and save questions", type="primary", disabled=not key or not selected) and start_ai_action():
        done = False
        try:
            existing = [q["prompt"] for q in bank if q["word_id"] in selected]
            with st.spinner("Creating fresh questions…"):
                generated, usage = generate_questions([by_id[wid] for wid in selected], key, model, existing)
                count = store.add_questions(generated)
                store.log_generation(model, count, usage)
            done = True
            share_new_content(questions=generated)
            st.success(f"Saved {count} new questions. They will be available in your next practice session, including offline.")
            st.caption("AI questions receive format and coverage checks, but their wording and answers can still need correction. Use Flag this question during practice.")
        except AuthenticationError:
            st.error(BAD_KEY_MESSAGE)
        except RateLimitError:
            st.error("OpenAI's usage or billing limit was reached. Check your API account; offline practice still works.")
        except APIConnectionError:
            st.error("Could not reach OpenAI. Check your internet connection and try again. Your saved practice is available.")
        except APIStatusError:
            st.error("OpenAI could not complete this request. Check model access and try again later.")
        except ValueError as exc:
            # ValidationError may include model content; display only our own short errors.
            st.error(str(exc) if type(exc) is ValueError else "The response failed validation. No questions were saved.")
        finally:
            if not done:
                undo_ai_action()


def reading():
    sid = st.session_state.get("active_reading_session")
    if sid:
        reading_flow(sid)
        return
    reading_landing()


def reading_landing():
    st.markdown('<div class="eyebrow">Reading</div>', unsafe_allow_html=True)
    st.title("What do you like reading today?")
    st.write("Pick a topic, read something real, then talk about what you found.")
    topics = list(READING_TOPICS)
    cols = st.columns(4)
    for i, topic in enumerate(topics):
        with cols[i % 4]:
            emoji = READING_EMOJI.get(topic, "\U0001F4D6")
            if st.button(f"{emoji} {topic}", key=f"reading_topic_{topic}", width="stretch"):
                passage = store.choose_passage(topic)
                if passage:
                    new_sid = store.start_reading_session(passage["id"])
                    st.session_state.active_reading_session = new_sid
                    st.session_state.reading_stage = "read"
                    st.session_state.pop("reading_q_cursor", None)
                    st.session_state.pop("reading_empty_topic", None)
                    st.rerun()
                else:
                    st.session_state.reading_empty_topic = topic
    if st.session_state.get("reading_empty_topic"):
        st.info(f"No reading material for {st.session_state.reading_empty_topic} yet. "
                "Ask a parent to fetch some new material below, or try another topic.")
    st.write("")
    with st.expander("Add more reading material"):
        reading_fetch_panel()


def reading_fetch_panel():
    st.write("Fetches real passages from Wikipedia for a topic. OpenAI drafts the comprehension "
              "questions, discussion prompts, and candidate vocabulary that go with each one, but "
              "never writes or rewrites the passage itself.")
    key = setting("OPENAI_API_KEY")
    model = setting("OPENAI_MODEL", "gpt-4.1-mini")
    topic = st.selectbox("Topic", list(READING_TOPICS), key="reading_fetch_topic")
    # In the public demo one fetch is one passage, so a single visitor can't
    # use the whole shared AI allowance at once.
    max_count = 1 if GUEST_MODE and not is_owner() else 5
    count = st.number_input("How many new passages", min_value=1, max_value=max_count, value=min(2, max_count),
                            step=1, key="reading_fetch_count")
    if not key:
        st.info(NO_KEY_MESSAGE + " Fetching also needs internet.")
    if GUEST_MODE:
        st.caption(ai_allowance_note() + " New passages are shared with everyone who visits.")
    if st.button("Fetch new passages", type="primary", disabled=not key) and start_ai_action():
        existing_titles = {p["title"] for p in store.passages(topic)}
        skipped = []
        try:
            # A fetch can legitimately work through dozens of candidate
            # Wikipedia pages before one clears every check (length,
            # readability, topic suitability, then the AI-drafted activity
            # itself), each costing a real network round trip — several
            # minutes total is possible even though nothing is wrong. A
            # single static spinner gave no way to tell that apart from an
            # actually stuck fetch, so this updates live, once per candidate
            # tried, instead.
            with st.status("Searching Wikipedia for candidate pages…", expanded=True) as status:
                def report_skip(title, reason):
                    skipped.append((title, reason))
                    status.update(label=f"Checked {len(skipped)} candidate page(s) so far — "
                                         f"most recently “{title}” ({reason})")
                fetched = fetch_passages(topic, int(count), key, model, existing_titles=existing_titles,
                                          on_skip=report_skip)
                if fetched:
                    for p in fetched:
                        st.caption(f"**{p['title']}** — selected")
                    status.update(label=f"Found {len(fetched)} new passage(s) for {topic}.",
                                  state="complete", expanded=False)
                else:
                    status.update(label=f"Checked {len(skipped)} candidate page(s); none were suitable.",
                                  state="error", expanded=True)
            if fetched:
                added = store.add_passages(fetched)
                share_new_content(passages=fetched)
                selected = "\n".join(f"- [{p['title']}]({p['source_url']})" if p.get("source_url")
                                     else f"- {p['title']}" for p in fetched)
                st.success(f"Added {added} new passage(s) for {topic}. They're ready to read now.\n\n{selected}")
            else:
                st.warning("No suitable new passages were found this time. Try again later or pick another topic.")
            if skipped:
                with st.expander(f"Why weren't more passages found? ({len(skipped)} candidate(s) skipped)"):
                    for title, reason in skipped:
                        st.caption(f"**{title}** — {reason}")
        except AuthenticationError:
            undo_ai_action()
            st.error(BAD_KEY_MESSAGE)
        except RateLimitError:
            undo_ai_action()
            st.error("OpenAI's usage or billing limit was reached. Check your API account.")
        except APIConnectionError:
            undo_ai_action()
            st.error("Could not reach OpenAI or Wikipedia. Check your internet connection and try again.")
        except APIStatusError:
            undo_ai_action()
            st.error("OpenAI could not complete this request. Check model access and try again later.")


def reading_flow(sid):
    session = store.reading_session(sid)
    if not session:
        st.session_state.pop("active_reading_session", None)
        st.rerun()
        return
    passage = store.passage(session["passage_id"])
    if not passage:
        st.session_state.pop("active_reading_session", None)
        st.rerun()
        return
    stage = st.session_state.get("reading_stage", "read")
    if stage == "read":
        reading_passage_and_comprehension_stage(sid, passage)
    elif stage == "vocab":
        reading_vocab_stage(sid, passage)
    else:
        reading_done_stage(session, passage)


def reading_exit_button():
    if st.button("Exit to topics"):
        for key in ("active_reading_session", "reading_stage", "reading_q_cursor",
                    "reading_added_words"):
            st.session_state.pop(key, None)
        st.rerun()


def render_passage_card(passage, highlights=None, show_paragraph_labels=False):
    st.markdown(f'<div class="eyebrow">{passage["topic"]}</div>', unsafe_allow_html=True)
    st.title(passage["title"])
    words = passage["word_count"]
    st.caption(f"~{max(1, round(words / 130))} min read · {words} words")
    body = normalize_paragraph_breaks(passage["body"])
    if show_paragraph_labels:
        # matching_information/matching_headings answers refer to paragraphs
        # by letter, so show the same A/B/C... labels the AI was given —
        # label_paragraphs() is the single source of truth both sides use.
        body = "\n\n".join(f"**{label}.** {para}" for label, para in label_paragraphs(body))
    highlighted = highlight_passage_html(body, highlights) if highlights else None
    if passage.get("image_path"):
        image_col, text_col = st.columns([2, 3], gap="medium")
        with image_col:
            try:
                st.image(passage["image_path"], width="stretch")
            except Exception:
                # A picture that failed to download cleanly or was corrupted on disk
                # should never block reading the passage itself.
                pass
        with text_col:
            with st.container(border=True):
                if highlighted:
                    st.markdown(highlighted, unsafe_allow_html=True)
                else:
                    st.write(body)
    else:
        with st.container(border=True):
            if highlighted:
                st.markdown(highlighted, unsafe_allow_html=True)
            else:
                st.write(body)
    st.caption(f"Source: [{passage['source_name']}]({passage['source_url']}) · {passage['license']}")
    if highlights:
        st.caption("The highlighted, numbered spots show where each question's answer comes from.")


READING_KIND_LABELS = {
    "multiple_choice": "Multiple choice",
    "true_false_not_given": "True, False, or Not Given",
    "yes_no_not_given": "Yes, No, or Not Given",
    "sentence_completion": "Sentence completion",
    "short_answer": "Short answer",
    "matching_sentence_endings": "Matching sentence endings",
    "matching_features": "Matching features",
    "matching_information": "Matching information",
    "matching_headings": "Matching headings",
}


def _comprehension_correct_option(q):
    if q.get("kind") in FREE_TEXT_KINDS:
        return q.get("answer_text")
    return q["options"][q["answer"]]


def _comprehension_passage_quote(q, passage_body):
    """The passage evidence for a comprehension question's correct answer, or
    None when there isn't any to show. A correct answer of "Not Given" has no
    evidence by definition — the passage doesn't say either way — so it's
    never looked up, real or best-effort, the way a real quote would be."""
    if _comprehension_correct_option(q) == "Not Given":
        return None
    return q.get("source_quote") or find_supporting_sentence(
        q["explanation"], _comprehension_correct_option(q), passage_body)


def reading_passage_and_comprehension_stage(sid, passage):
    session = store.reading_session(sid)
    questions = passage["comprehension"]
    idx = st.session_state.get("reading_q_cursor", 0)
    if idx >= len(questions):
        reading_comprehension_review(session, passage)
        return
    needs_labels = any(qq.get("kind") in PARAGRAPH_KINDS for qq in questions)
    render_passage_card(passage, show_paragraph_labels=needs_labels)
    st.divider()
    q = questions[idx]
    answered = session["payload"]["comprehension_answers"].get(str(idx))
    kind_label = READING_KIND_LABELS.get(q.get("kind"), "Multiple choice")
    st.caption(f"COMPREHENSION · QUESTION {idx + 1} OF {len(questions)} · {kind_label.upper()}")
    st.caption("Answer every question first — you'll see the answer key and explanations at the end.")
    st.progress(idx / len(questions))
    is_free_text = q.get("kind") in FREE_TEXT_KINDS
    with st.container(border=True):
        st.markdown(f"### {q['prompt']}")
        key = f"reading_answer_{sid}_{idx}"
        if answered and key not in st.session_state:
            st.session_state[key] = answered["selected"]
        if is_free_text:
            max_words = q.get("max_words") or 5
            word_note = "word" if max_words == 1 else "words"
            st.caption(f"Type your answer — no more than {max_words} {word_note}.")
            selected = st.text_input("Your answer", key=key)
            can_continue = bool(selected and selected.strip())
        else:
            selected = st.radio("Choose your answer", q["options"], index=None, key=key)
            can_continue = selected is not None
        back_col, next_col = st.columns(2)
        if idx > 0 and back_col.button("← Back"):
            st.session_state.reading_q_cursor = idx - 1
            st.rerun()
        if next_col.button("Next", type="primary", disabled=not can_continue):
            store.record_comprehension_answer(sid, idx, selected)
            st.session_state.reading_q_cursor = idx + 1
            st.rerun()
    reading_exit_button()


def reading_comprehension_review(session, passage):
    questions = passage["comprehension"]
    answers = session["payload"]["comprehension_answers"]
    needs_labels = any(qq.get("kind") in PARAGRAPH_KINDS for qq in questions)
    quotes = [_comprehension_passage_quote(q, passage["body"]) for q in questions]
    render_passage_card(
        passage,
        highlights=[(quote, i + 1) for i, quote in enumerate(quotes) if quote],
        show_paragraph_labels=needs_labels,
    )
    st.divider()
    correct = sum(1 for a in answers.values() if a["correct"])
    st.markdown(f"### You got {correct} of {len(questions)} right")
    st.caption("Here's the answer key, with an explanation for each question.")
    for i, q in enumerate(questions):
        a = answers.get(str(i)) or {}
        right_option = _comprehension_correct_option(q)
        is_free_text = q.get("kind") in FREE_TEXT_KINDS
        with st.container(border=True):
            st.markdown(f"**{i + 1}. {q['prompt']}**")
            if is_free_text:
                user_answer = (a.get("selected") or "").strip()
                if a.get("correct"):
                    st.markdown(f"✅ **{user_answer}**")
                else:
                    if user_answer:
                        st.markdown(f"❌ ~~{user_answer}~~ — your answer")
                    else:
                        st.caption("You didn't answer this one.")
                    st.markdown(f"✅ **{right_option}**")
            else:
                for opt in q["options"]:
                    if opt == right_option:
                        st.markdown(f"✅ **{opt}**")
                    elif opt == a.get("selected"):
                        st.markdown(f"❌ ~~{opt}~~ — your answer")
                    else:
                        st.caption(opt)
            st.write(q["explanation"])
            if quotes[i]:
                st.caption(f"🔎 Find it highlighted as ({i + 1}) in the passage above.")
            elif right_option == "Not Given":
                st.caption("Not stated either way in the passage — that's exactly why "
                           "'Not Given' is correct.")
    if st.button("Continue", type="primary"):
        st.session_state.reading_stage = "vocab"
        st.rerun()
    reading_exit_button()


def reading_vocab_stage(sid, passage):
    words = passage["vocabulary"]
    st.markdown("### New words you met today")
    if not words:
        st.caption("No new vocabulary was drafted for this passage.")
        if st.button("Finish", type="primary"):
            store.finish_reading_session(sid)
            st.session_state.reading_stage = "done"
            st.rerun()
        return
    st.write("Pick any you'd like to add to your Vocabulary practice.")
    chosen = []
    for i, w in enumerate(words):
        key = f"reading_vocab_{sid}_{i}"
        checked = st.checkbox(f"**{w['word']}** — {w['definition']}", key=key, value=True)
        if w.get("example"):
            st.caption(w["example"])
        if checked:
            chosen.append(i)
    if st.button("Add selected words and finish", type="primary"):
        added = store.add_vocab_words(sid, chosen)
        store.finish_reading_session(sid)
        st.session_state.reading_stage = "done"
        st.session_state.reading_added_words = added
        st.rerun()


def reading_done_stage(session, passage):
    st.markdown("## Nice reading!")
    answers = session["payload"]["comprehension_answers"]
    correct = sum(1 for a in answers.values() if a["correct"])
    added = st.session_state.get("reading_added_words", 0)
    a, b = st.columns(2)
    a.metric("Comprehension", f"{correct} / {len(passage['comprehension'])}")
    b.metric("New words added", added)
    if added:
        st.caption("Added to your Vocabulary practice.")
    st.button("See your reading progress →", on_click=go_to_progress)
    if st.button("Read something else", type="primary"):
        for key in ("active_reading_session", "reading_stage", "reading_q_cursor",
                    "reading_added_words"):
            st.session_state.pop(key, None)
        st.rerun()


def render_reading_session_detail(session, passage):
    payload = session["payload"]
    answers = payload.get("comprehension_answers", {})
    comprehension = passage.get("comprehension", [])
    if comprehension:
        correct = sum(1 for a in answers.values() if a.get("correct"))
        st.caption(f"Comprehension: {correct} / {len(comprehension)} correct")
    responses = payload.get("opinion_responses", {})
    prompts = passage.get("opinions", [])
    if prompts:
        st.markdown("**Your thoughts**")
        for i, prompt in enumerate(prompts):
            response = responses.get(str(i))
            with st.container(border=True):
                st.caption(prompt)
                st.write(response if response else "_No response was saved for this prompt._")
    vocabulary = passage.get("vocabulary", [])
    added_words = [vocabulary[i]["word"] for i in payload.get("vocab_chosen", []) if 0 <= i < len(vocabulary)]
    if added_words:
        st.caption("Added to Vocabulary: " + ", ".join(added_words))


# --- Writing ---

WRITING_EXIT_KEYS = ("writing_stage", "writing_prompt", "writing_mode", "writing_started_at",
                     "writing_locked", "writing_final_text", "writing_deadline_hit",
                     "writing_time_taken", "writing_attempt_id", "writing_essay_box_text")


def writing():
    stage = st.session_state.get("writing_stage")
    if stage in ("write", "grading", "report"):
        writing_flow(stage)
        return
    writing_pick_stage()


def writing_pick_stage():
    st.markdown('<div class="eyebrow">Writing</div>', unsafe_allow_html=True)
    st.title("Practice IELTS Writing Task 2")
    st.write("Choose a prompt, then decide how you want to practise. Both modes are graded exactly "
             "the same way afterward.")
    prompts = store.essay_prompts()
    if not prompts:
        st.info("No essay prompts yet. Ask a parent to add some to data/essay_prompts.psv.")
        return
    topic_options = ["All topics"] + sorted({p["topic"] for p in prompts})
    type_options = ["All types"] + list(ESSAY_TYPES)
    col1, col2 = st.columns(2)
    topic = col1.selectbox("Topic", topic_options, key="writing_pick_topic")
    etype = col2.selectbox("Essay type", type_options, format_func=lambda t: ESSAY_TYPES.get(t, t),
                            key="writing_pick_type")
    chosen_type = None if etype == "All types" else etype
    pick_key = (topic, etype)
    if st.session_state.get("writing_pick_key") != pick_key:
        st.session_state.writing_pick_key = pick_key
        st.session_state.writing_candidate = store.choose_essay_prompt(topic, chosen_type)
    candidate = st.session_state.get("writing_candidate")
    if not candidate:
        st.info("No prompts match that combination yet. Try a different topic or type.")
        return
    with st.container(border=True):
        st.caption(ESSAY_TYPES.get(candidate["essay_type"], candidate["essay_type"]))
        st.markdown(f"### {candidate['prompt']}")
    if st.button("Show a different prompt"):
        st.session_state.writing_candidate = store.choose_essay_prompt(topic, chosen_type)
        st.rerun()
    st.write(f"**Simulation** gives you {SIMULATION_MINUTES} minutes and locks your writing when "
             "time's up, for real exam pressure. **Practice** has no time limit — how long you take "
             "is simply recorded, not judged. Both are graded identically afterward.")
    if GUEST_MODE:
        st.caption(ai_allowance_note() + " Grading one essay uses one.")
    sim_col, prac_col = st.columns(2)
    if sim_col.button(f"Start — Simulation ({SIMULATION_MINUTES} min)", type="primary", width="stretch"):
        _start_writing(candidate, "simulation")
    if prac_col.button("Start — Practice (untimed)", width="stretch"):
        _start_writing(candidate, "practice")


def _start_writing(prompt, mode):
    st.session_state.writing_prompt = prompt
    st.session_state.writing_mode = mode
    st.session_state.writing_started_at = time.time()
    st.session_state.writing_locked = False
    st.session_state.pop("writing_essay_box_text", None)
    st.session_state.writing_stage = "write"
    st.rerun()


def writing_exit_button():
    if st.button("Exit to prompts"):
        for key in WRITING_EXIT_KEYS:
            st.session_state.pop(key, None)
        st.rerun()


def writing_flow(stage):
    prompt = st.session_state.get("writing_prompt")
    if not prompt:
        st.session_state.pop("writing_stage", None)
        st.rerun()
        return
    if stage == "write":
        writing_write_stage(prompt)
    elif stage == "grading":
        writing_grading_stage(prompt)
    else:
        writing_report_stage(prompt)


def writing_write_stage(prompt):
    st.markdown(f'<div class="eyebrow">{ESSAY_TYPES.get(prompt["essay_type"], prompt["essay_type"])}</div>',
                unsafe_allow_html=True)
    st.title(prompt["prompt"])
    mode_note = (f"Simulation — {SIMULATION_MINUTES} minutes, essay locks at zero" if
                 st.session_state.writing_mode == "simulation" else "Practice — untimed")
    st.caption(mode_note)
    writing_essay_box()
    st.caption("Your essay isn't saved until it's graded — don't exit before submitting.")
    writing_exit_button()


@st.fragment(run_every="1s")
def writing_essay_box():
    """Ticks once a second while the essay is being written, entirely
    within its own fragment so the rest of the page never re-renders just to
    update a countdown. In Simulation mode this both shows the remaining
    time and is what actually notices the deadline has passed and locks the
    text area — a plain widget rerun triggered by typing could otherwise
    leave the box unlocked for a while after time was really up if the
    learner stopped interacting right at the deadline. Practice mode has no
    lock; it only shows elapsed time for the learner's own information."""
    mode = st.session_state.writing_mode
    elapsed = time.time() - st.session_state.writing_started_at
    locked = st.session_state.get("writing_locked", False)
    if mode == "simulation":
        remaining = SIMULATION_MINUTES * 60 - elapsed
        if remaining <= 0:
            locked = True
            st.session_state.writing_locked = True
        minutes, seconds = max(0, int(remaining) // 60), max(0, int(remaining) % 60)
        if locked:
            st.error("Time's up. Your writing is locked — review it below, then submit for grading.")
        else:
            st.metric("Time remaining", f"{minutes:02d}:{seconds:02d}")
    else:
        st.caption(f"Elapsed: {int(elapsed) // 60} min {int(elapsed) % 60} s — "
                   "untimed, just for your own record.")
    text = st.text_area("Your essay", key="writing_essay_box_text", height=420, disabled=locked,
                         placeholder=f"Write at least {MIN_RECOMMENDED_WORDS} words…")
    word_count = len((text or "").split())
    note = f"{word_count} words"
    if word_count < MIN_RECOMMENDED_WORDS:
        note += f" — the real test asks for at least {MIN_RECOMMENDED_WORDS}"
    st.caption(note)
    if st.button("Submit for grading", type="primary", disabled=not (text or "").strip()):
        st.session_state.writing_final_text = text
        st.session_state.writing_deadline_hit = (mode == "simulation" and locked)
        st.session_state.writing_time_taken = int(elapsed)
        st.session_state.writing_stage = "grading"
        st.rerun()


def writing_grading_stage(prompt):
    st.markdown("### Grading your essay…")
    key = setting("OPENAI_API_KEY")
    model = setting("OPENAI_MODEL", "gpt-4.1-mini")
    essay_text = st.session_state.get("writing_final_text", "")
    if not key:
        st.error(NO_KEY_MESSAGE)
    if not key or not start_ai_action():
        if st.button("Back to writing"):
            # Streamlit drops a widget's session_state entry for any run in
            # which that widget isn't instantiated at all — true here, since
            # this whole stage renders no text_area — so the essay box's own
            # state must be restored explicitly rather than assumed to still
            # be there once writing_write_stage renders it again.
            st.session_state.writing_essay_box_text = essay_text
            st.session_state.writing_stage = "write"
            st.rerun()
        return
    done = False
    try:
        with st.spinner("Comparing your essay against the four IELTS Writing Task 2 criteria…"):
            evaluation = generate_evaluation(prompt["prompt"], essay_text, key, model)
        done = True
        aid = store.record_writing_attempt(
            prompt["id"], st.session_state.writing_mode, essay_text, evaluation.model_dump(),
            deadline_hit=st.session_state.get("writing_deadline_hit", False),
            time_taken_seconds=st.session_state.get("writing_time_taken"),
        )
        st.session_state.writing_attempt_id = aid
        st.session_state.writing_stage = "report"
        st.rerun()
    except AuthenticationError:
        st.error(BAD_KEY_MESSAGE)
    except RateLimitError:
        st.error("OpenAI's usage or billing limit was reached. Check your API account.")
    except APIConnectionError:
        st.error("Could not reach OpenAI. Check your internet connection and try again.")
    except APIStatusError:
        st.error("OpenAI could not complete this request. Check model access and try again later.")
    except ValueError as exc:
        st.error(str(exc) if type(exc) is ValueError else "The response failed validation. Nothing was saved.")
    finally:
        if not done:
            undo_ai_action()
    # A successful attempt returns via st.rerun() above, which halts the
    # script immediately, so the lines below only ever run after one of the
    # excepted errors. Every failure here is worth retrying — the essay
    # itself is never lost, since nothing is saved until grading succeeds.
    st.caption("Your written essay is kept on this screen — try again rather than exiting.")
    if st.button("Try grading again", type="primary"):
        st.rerun()
    writing_exit_button()


def render_evaluation(evaluation: dict, essay_text: str, band: float):
    st.metric("Overall Task 2 band (estimate)", band)
    st.caption("An AI estimate against the four official criteria — not an official IELTS score.")
    for key in CRITERIA:
        c = evaluation[key]
        with st.container(border=True):
            st.markdown(f"**{CRITERION_LABELS[key]} — Band {c['band']}**")
            st.write(c["assessment"])
            for ev in c.get("evidence", []):
                st.caption(f"“{ev['quote']}” — {ev['comment']}")
            st.markdown(f"**To reach the next band:** {c['next_band_advice']}")
    with st.expander("Your essay"):
        st.write(essay_text)


def writing_report_stage(prompt):
    aid = st.session_state.get("writing_attempt_id")
    attempt = store.writing_attempt(aid) if aid else None
    if not attempt:
        st.session_state.pop("writing_stage", None)
        st.rerun()
        return
    st.markdown("## Your Writing Task 2 report")
    st.caption(f"{ESSAY_TYPES.get(prompt['essay_type'], prompt['essay_type'])} · {prompt['prompt']}")
    render_evaluation(attempt["evaluation"], attempt["essay_text"], attempt["overall_band"])
    if attempt["mode"] == "practice" and attempt.get("time_taken_seconds") is not None:
        st.caption(f"Time taken: {attempt['time_taken_seconds'] // 60} min "
                   f"{attempt['time_taken_seconds'] % 60} s (untimed)")
    elif attempt["deadline_hit"]:
        st.caption("Submitted right at the simulation time limit.")
    if st.button("Practice another essay", type="primary"):
        for key in WRITING_EXIT_KEYS:
            st.session_state.pop(key, None)
        st.rerun()


def render_writing_attempt_detail(attempt):
    render_evaluation(attempt["evaluation"], attempt["essay_text"], attempt["overall_band"])
    if attempt["mode"] == "practice" and attempt.get("time_taken_seconds") is not None:
        st.caption(f"Time taken: {attempt['time_taken_seconds'] // 60} min "
                   f"{attempt['time_taken_seconds'] % 60} s (untimed)")
    elif attempt["deadline_hit"]:
        st.caption("Submitted right at the simulation time limit.")


def progress_page():
    st.markdown('<div class="eyebrow">Your learning story</div>', unsafe_allow_html=True)
    st.title("See your words grow.")
    counts = Counter(p["stage"] for p in progress.values())
    cols = st.columns(4)
    for col, stage in zip(cols, ["New", "Learning", "Familiar", "Secure"]):
        col.metric(stage, counts[stage])
    st.write("")
    # One row per word: its most recent scored attempt (any word, including
    # flagged ones, since a flagged question still really happened), or
    # "Not yet practiced" if none exists yet.
    last_reviewed = {}
    for a in store.attempts(True):
        if a["answered_at"] > last_reviewed.get(a["word_id"], ""):
            last_reviewed[a["word_id"]] = a["answered_at"]
    rows = [{"Word": w["word"], "Topic": w["topic"], "Stage": progress[w["id"]]["stage"],
             "Last review": last_reviewed[w["id"]][:10] if w["id"] in last_reviewed else "Not yet practiced"}
            for w in words]
    topic = st.selectbox("Progress by topic", topics)
    st.dataframe([r for r in rows if topic == "All topics" or r["Topic"] == topic], hide_index=True, width="stretch")
    attempts = store.attempts()
    if attempts:
        st.markdown("### Your practice rhythm")
        daily = Counter(a["answered_at"][:10] for a in attempts)
        frame = pd.DataFrame({"Date": list(daily), "Answers": list(daily.values())}).set_index("Date")
        st.bar_chart(frame, color="#187568")
    reading_stats = store.reading_stats()
    if reading_stats["total_passages"]:
        st.markdown("### Reading habit")
        st.caption("Built around consistency, not a score — these track how often you read, not how well.")
        a, b, c, d = st.columns(4)
        a.metric("Passages read", reading_stats["total_passages"])
        b.metric("Reading streak", f"{reading_stats['streak_days']} day" + ("s" if reading_stats["streak_days"] != 1 else ""))
        c.metric("Topics explored", reading_stats["topics_explored"])
        d.metric("This week", reading_stats["this_week"])
        with st.expander("Reading history"):
            reading_rows = list(reversed(store.reading_activity()))
            if reading_rows:
                titles = {p["id"]: p["title"] for p in store.passages()}
                st.dataframe([{"Date": r["completed_at"][:16].replace("T", " "),
                               "Topic": r["topic"],
                               "Passage": titles.get(r["passage_id"], "—")}
                              for r in reading_rows], hide_index=True, width="stretch")
                reviewable = [r for r in reading_rows if r.get("session_id")]
                if reviewable:
                    st.divider()
                    st.caption("Pick one to read your comprehension score, your written answers, and any words you added.")
                    labels = {r["id"]: f"{r['completed_at'][:16].replace('T', ' ')} · "
                                        f"{titles.get(r['passage_id'], '—')}" for r in reviewable}
                    chosen_id = st.selectbox("Review a past reading", list(labels), format_func=lambda rid: labels[rid])
                    row = next(r for r in reviewable if r["id"] == chosen_id)
                    detail_session = store.reading_session(row["session_id"])
                    detail_passage = store.passage(row["passage_id"])
                    if detail_session and detail_passage:
                        render_reading_session_detail(detail_session, detail_passage)
                    else:
                        st.caption("This reading's details are no longer available.")
                else:
                    st.caption("Written answers aren't available for passages completed before this feature was added.")
            else:
                st.caption("Your completed passages will appear here.")
    writing_rows = list(reversed(store.writing_attempts()))
    if writing_rows:
        st.markdown("### Writing history")
        st.caption(f"{len(writing_rows)} essay(s) written. A plain record for a parent or tutor to "
                   "review — not a trend or average, just what was written and how it was graded.")
        prompts_by_id = {p["id"]: p for p in store.essay_prompts()}
        with st.expander("Writing history"):
            st.dataframe([{"Date": r["created_at"][:16].replace("T", " "),
                           "Mode": r["mode"].capitalize(),
                           "Prompt": prompts_by_id.get(r["prompt_id"], {}).get("prompt", "—")[:80],
                           "Overall band": r["overall_band"]}
                          for r in writing_rows], hide_index=True, width="stretch")
            st.divider()
            st.caption("Pick one to see the full essay and its graded report.")
            labels = {r["id"]: f"{r['created_at'][:16].replace('T', ' ')} · "
                                f"{r['mode'].capitalize()} · Band {r['overall_band']}" for r in writing_rows}
            chosen_id = st.selectbox("Review a past essay", list(labels), format_func=lambda rid: labels[rid])
            render_writing_attempt_detail(next(r for r in writing_rows if r["id"] == chosen_id))
    with st.expander("Flash card history"):
        activity = store.flashcard_activity()
        st.caption(f"Cards reviewed today: {store.flashcards_reviewed()}")
        if activity:
            st.dataframe([{"Word": word_label(a["word_id"]),
                           "Date": a["occurred_at"][:16].replace("T", " "),
                           "Activity": "Reviewed" if a["outcome"] == "review" else "Skipped"}
                          for a in reversed(activity)], hide_index=True, width="stretch")
        else:
            st.caption("Reveal a card to start your review history.")
        st.caption("Revealing a meaning records practice; quiz answers determine mastery.")
    with st.expander("How recognition stages work"):
        st.write("Learning: the next check is 1 to 3 days away. This includes any word whose latest answer was wrong. Familiar: the next check is a week away. Secure: the next check is two weeks or more away, up to 90 days.")
        st.write("Each correct answer on a new day moves a word up the ladder: 1, 3, 7, 14, 30, 60, then 90 days. Growing past 7 days needs correct answers in two different skills. A wrong answer sends the word back to 1 day. The first attempt on a question each day counts toward mastery. Extra attempts are saved as practice. Flagged questions are excluded. Meaning and meaning-in-context count as the same skill for promotion.")
    st.markdown("### Keep a copy")
    if GUEST_MODE:
        st.caption("Guest progress is not saved after you close this tab. Download a copy if you want to keep it.")
        a, b, c = st.columns(3)
    else:
        st.caption("The database backup includes progress and saved questions. Restore instructions are in the README.")
        a, b, c = st.columns(3)
        a.download_button("Download full backup", store.backup_bytes(), file_name=f"ielts-backup-{today}.sqlite3", mime="application/octet-stream")
    b.download_button("Export history (JSON)", store.export_json(), file_name=f"ielts-history-{today}.json", mime="application/json")
    csv_file = io.StringIO()
    writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    c.download_button("Export progress (CSV)", csv_file.getvalue(), file_name=f"ielts-progress-{today}.csv", mime="text/csv")


with st.sidebar:
    st.markdown("## 🌱 Word by Word")
    st.caption("YOUR IELTS PRACTICE SPACE")
    page = st.radio("Your learning", ["Vocabulary", "Reading", "Speaking", "Writing", "Progress"],
                    label_visibility="collapsed", key="nav_page")
    st.divider()
    st.caption("Start small. Practise often.\n\nBuilt for IELTS Academic preparation.")
    if GUEST_MODE:
        st.info("**Guest mode.** Try everything! Your progress is kept only while this tab is open. "
                "Reloading or closing the page starts fresh.", icon="👋")

if GUEST_MODE:
    st.markdown('<div class="guest-banner">👋 <b>Guest mode</b> · your progress is kept only while this tab '
                'is open.</div>', unsafe_allow_html=True)
    if st.session_state.get("shared_offline"):
        st.caption("Shared reading passages couldn't load right now, so this visit uses the built-in content only.")

if page == "Vocabulary":
    vocabulary()
elif page == "Reading":
    reading()
elif page == "Writing":
    writing()
elif page == "Progress":
    progress_page()
else:
    st.markdown('<div class="eyebrow">The next chapter</div>', unsafe_allow_html=True)
    st.title(f"{page} practice")
    st.info("Planned for a later version. Vocabulary, Reading, Writing, and Progress are ready to use now.")
    st.write("Future speaking practice will help you use your words aloud.")
