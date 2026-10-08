"""Reading passages: fetched from real sources, filtered for readability, and
bundled with AI-drafted comprehension questions and candidate vocabulary. The
passage text itself always comes from a real source and is never written or
rewritten by AI; only the follow-up material about it is AI-drafted.

Phase 1 covers nonfiction topics sourced from Wikipedia's public API via the
Action API's plain-text extract, not the short one-paragraph blurb the REST
"page summary" endpoint returns. The extract is not limited to the lead
section (the introduction before the first heading): for the great majority
of articles that introduction alone runs well under this app's 650-750 word
target, so the extract continues into the article's body as needed and is
trimmed to that budget, keeping the article's original paragraph breaks
rather than flattening it into one block of text. The Action API separates
paragraphs with a single newline, which Markdown renders as a soft space
rather than a new paragraph, so extracts are normalized to blank-line
paragraph breaks right after fetching, and any bare section-heading line the
extract includes once it goes past the lead is dropped rather than shown as
if it were a stray sentence.
Fiction/narrative topics (sourced from Project Gutenberg) are a later phase.
"""
from __future__ import annotations

import hashlib
import html
import json
import random
import re
import string
from collections import Counter
from typing import Literal, Optional

import requests
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

# Topic categories offered in the UI. Candidate article titles for each one
# are discovered live via Wikipedia's own full-text search API (see
# candidate_titles_for_topic / _wikipedia_search_titles) rather than drawn
# from a fixed, hand-picked list of titles, so a topic is never limited to
# however many pages happened to be curated for it in advance — the previous
# design, where e.g. "Sports" only had 7 candidate titles total, meant a
# fetch could come back empty just because that whole small pool had already
# been used or didn't clear the readability bar.
READING_TOPICS: list[str] = [
    "Animals", "Space", "Technology", "History", "Environment", "Sports", "World", "Culture",
]

WIKI_ACTION_API = "https://{lang}.wikipedia.org/w/api.php"
USER_AGENT = "WordByWordIELTSCoach/1.0 (local single-learner study app; no contact needed)"

# The word-count range this app targets for a full IELTS-length passage,
# roughly matching one passage's length on the real test. is_reader_friendly's
# real readability check, the pre-fetch length filter in
# candidate_titles_for_topic, and fetch_passages' own trim budget all read
# from here, so the target only needs to change in one place.
MIN_PASSAGE_WORDS = 650
MAX_PASSAGE_WORDS = 750

# How many live Wikipedia search results to pull as candidates for one
# fetch. Generous relative to the handful of passages a fetch typically
# asks for, since some candidates get filtered out afterward (a genuine
# stub, or too dense/list-like to read well) and nothing can pad a short
# article back up, unlike an oversized one, which just gets trimmed. A
# small pool can realistically come back with zero qualifying candidates;
# this stays comfortably under Wikipedia's own per-request search limit.
SEARCH_POOL_SIZE = 150

# A coarse, best-effort keyword safety net so a live Wikipedia search
# (unlike the old hand-picked title lists, which were implicitly screened
# just by someone having chosen them) can't surface a candidate that's
# topically unsuitable for a young learner — checked against a search hit's
# own title before it's ever fetched (candidate_titles_for_topic) and again
# against the final passage text fetch_passages actually shows (in case an
# otherwise-fine article's body drifts into unsuitable territory once the
# extract runs past its lead — see MAX_PASSAGE_WORDS and
# strip_bare_headings). This is a keyword match, not a content classifier:
# it catches a candidate whose subject itself is unsuitable, not every way
# a stray sentence could be, and it says nothing about a page's picture.
# Extend this list as real cases turn up rather than expecting it to be
# exhaustive from the start.
INAPPROPRIATE_TOPIC_KEYWORDS = [
    "sexual behavior", "sexual behaviour", "sexual activity", "sexual intercourse",
    "masturbat", "pornograph", "genitalia", "genital stimulation", "orgasm",
    "erotic", "fetish", "incest", "rape", "prostitut", "bestiality",
    "graphic violence", "torture", "mutilation", "gore", "gruesome",
    "suicide", "self-harm", "self harm",
    "drug trafficking", "drug cartel", "illegal drug",
    "hate speech", "genocide", "massacre",
]


def _mentions_inappropriate_topic(*texts: str) -> bool:
    """True if any of the given strings (typically a candidate title and/or
    the actual passage text) contains a term from INAPPROPRIATE_TOPIC_KEYWORDS,
    case-insensitively. See that constant's comment for what this can and
    can't catch.
    """
    combined = " ".join(t or "" for t in texts).casefold()
    return any(term in combined for term in INAPPROPRIATE_TOPIC_KEYWORDS)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_NEWLINE_RUN = re.compile(r"\n+")
_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n+")


def normalize_paragraph_breaks(text: str) -> str:
    """Normalize paragraph breaks to a blank line between paragraphs.

    Wikipedia's Action API extract separates paragraphs with a single
    newline, not a blank line. A single newline in Markdown is just a soft
    line break rendered as a space, not a new paragraph, so left as-is every
    fetched passage rendered as one run-on block of text regardless of its
    real paragraph structure. Collapsing any run of one or more newlines to
    exactly two turns the source's real paragraph boundaries into boundaries
    a Markdown renderer (and `trim_to_word_budget`'s own paragraph splitting)
    actually respects.
    """
    return _NEWLINE_RUN.sub("\n\n", text).strip()


_SENTENCE_ENDING = re.compile(r"[.!?][\"')\]]*$")
# A real section heading is a short phrase ("History", "Early life and
# career"), never a long run of text; capping the word count this heuristic
# will treat as a possible heading keeps it from ever mistaking a long,
# genuinely punctuation-less block of text (garbled source text, for
# instance) for one and discarding real passage content.
_MAX_HEADING_WORDS = 10


def strip_bare_headings(text: str) -> str:
    """Drop paragraph-shaped lines that are actually a stripped section
    heading rather than prose. Once a Wikipedia extract goes past the lead
    section, the API keeps each section's heading as its own bare line of
    text with the surrounding "==" wiki markup already removed, with
    nothing else marking it as different from a real paragraph structurally
    (see normalize_paragraph_breaks). A heading is reliably different from
    real passage prose in one simple way that's safe to check for: prose
    here is written in full sentences and always ends in sentence-ending
    punctuation, while a heading ("History", "Early life and career") never
    does — combined with being short (see _MAX_HEADING_WORDS), so a long
    paragraph that happens to lack ending punctuation is never mistaken for
    one. Must be called on already paragraph-normalized text (a blank line
    between paragraphs), the same as label_paragraphs.
    """
    paragraphs = [p for p in _PARAGRAPH_SPLIT.split(text) if p.strip()]
    kept = []
    for p in paragraphs:
        stripped = p.strip()
        looks_like_heading = (len(stripped.split()) <= _MAX_HEADING_WORDS
                               and not _SENTENCE_ENDING.search(stripped))
        if not looks_like_heading:
            kept.append(p)
    return "\n\n".join(kept)


def passage_id(topic: str, title: str) -> str:
    return hashlib.sha256(f"{topic}|{title}".casefold().encode()).hexdigest()[:24]


# Reasons a Wikipedia search (topic candidate discovery) can fail, and the
# honest message fetch_passages shows for the topic as a whole when it does.
# Both are transient/network-side — the topic itself isn't at fault.
WIKI_SEARCH_SKIP_REASONS = {
    "network": "couldn't reach Wikipedia's search (network error or timeout) — try again",
    "bad_response": "Wikipedia's search returned an unexpected response — try again",
}


def _wikipedia_search_titles(query: str, limit: int = 50, lang: str = "en", session=None,
                              timeout: int = 10) -> tuple[list[dict], str | None]:
    """Live full-text search against Wikipedia's MediaWiki Action API
    (`list=search`), restricted to the main article namespace, used to
    discover candidate article titles for a topic instead of relying on a
    fixed, hand-curated title list. Also asks for each hit's whole-page word
    count (`srprop=wordcount`) — a hint used by candidate_titles_for_topic to
    skip an expensive full-page fetch for a hit that's clearly too short,
    since that count comes back for free in this one search response.
    Titles that survive that hint still aren't guaranteed to be good
    candidates (a search hit can still be a list article or a disambiguation
    page, or its lead specifically — as opposed to the whole page — can
    still be short) — that's checked properly, the same way for every
    candidate, by `_fetch_wikipedia_page` and `is_reader_friendly` once
    fetch_passages actually tries each title.

    Returns `([{"title": ..., "wordcount": ... or None}, ...], None)` on
    success (possibly an empty list if the search genuinely had no matches)
    or `([], reason)` on failure, where `reason` is one of the keys of
    WIKI_SEARCH_SKIP_REASONS. Never raises.
    """
    http = session or requests
    url = WIKI_ACTION_API.format(lang=lang)
    params = dict(
        action="query", format="json", formatversion=2,
        list="search", srsearch=query, srlimit=limit, srnamespace=0, srprop="wordcount",
    )
    try:
        resp = http.get(url, params=params, timeout=timeout, headers={"User-Agent": USER_AGENT})
    except requests.RequestException:
        return [], "network"
    if resp.status_code != 200:
        return [], "bad_response"
    try:
        data = resp.json()
    except ValueError:
        return [], "bad_response"
    hits = (data.get("query") or {}).get("search") or []
    return [dict(title=h["title"], wordcount=h.get("wordcount")) for h in hits if h.get("title")], None


# Wikipedia's search ranks a literal title match very heavily, so a topic like
# "Sports" fills its results with brand and broadcaster pages ("NBC Sports"),
# lists, disambiguation pages and media titles that happen to contain the word
# — articles that make poor reading passages. Each page's own one-line short
# description (e.g. "American basketball player") says what the page is about,
# so candidate_titles_for_topic uses it to drop those. A description that
# matches one of these word-boundary patterns marks a page as noise; a page
# with no description at all is dropped too (see _description_is_substantive).
# Like INAPPROPRIATE_TOPIC_KEYWORDS this is a coarse filter: extend it as real
# cases turn up.
NOISY_DESCRIPTION_PATTERNS = [
    r"disambiguation",
    r"\blists?\b",
    r"\b(?:television|tv|radio|cable)\s+(?:channel|network|station|series|program|programme|show)s?\b",
    r"\b(?:channel|network|broadcaster|broadcasting)\b",
    r"\b(?:brand|company|corporation|retailer|manufacturer|trademark|subsidiary|publisher)\b",
    r"\b(?:magazine|newspaper|website|mobile app|video game|mobile game)\b",
    r"\b(?:film|album|song|episode|novel)\b(?!\s+(?:director|producer|actor|actress|composer|critic|studio|score))",
    r"\bterm\b",
    r"\bjargon\b",
    r"\bslang\b",
]
_NOISY_DESCRIPTION_RE = re.compile("|".join(NOISY_DESCRIPTION_PATTERNS), re.IGNORECASE)

# The Action API accepts up to 50 titles per `titles=` request.
_WIKI_TITLES_PER_REQUEST = 50


def _description_is_substantive(description: str | None) -> bool:
    """True if a page's Wikipedia short description marks it as a real subject
    article: present, and not matching NOISY_DESCRIPTION_PATTERNS. A page
    with no description is treated as not substantive — most real articles
    have one, and there are plenty of candidates to spare.
    """
    if not description or not description.strip():
        return False
    return _NOISY_DESCRIPTION_RE.search(description) is None


def _wikipedia_short_descriptions(titles: list[str], lang: str = "en", session=None,
                                   timeout: int = 10) -> tuple[dict[str, str], str | None]:
    """Look up each title's Wikipedia short description
    (`prop=pageprops&ppprop=wikibase-shortdesc`), up to 50 titles per request.

    Returns `({title: description}, None)` — a title with no short
    description is simply absent from the dict — or `({}, reason)` on failure,
    where `reason` is one of the keys of WIKI_SEARCH_SKIP_REASONS. Never raises.
    """
    http = session or requests
    url = WIKI_ACTION_API.format(lang=lang)
    descriptions: dict[str, str] = {}
    for start in range(0, len(titles), _WIKI_TITLES_PER_REQUEST):
        batch = titles[start:start + _WIKI_TITLES_PER_REQUEST]
        params = dict(
            action="query", format="json", formatversion=2,
            prop="pageprops", ppprop="wikibase-shortdesc", titles="|".join(batch),
        )
        try:
            resp = http.get(url, params=params, timeout=timeout, headers={"User-Agent": USER_AGENT})
        except requests.RequestException:
            return {}, "network"
        if resp.status_code != 200:
            return {}, "bad_response"
        try:
            data = resp.json()
        except ValueError:
            return {}, "bad_response"
        pages = (data.get("query") or {}).get("pages") or []
        if isinstance(pages, dict):
            pages = list(pages.values())
        for page in pages:
            desc = (page.get("pageprops") or {}).get("wikibase-shortdesc")
            if page.get("title") and desc:
                descriptions[page["title"]] = desc
    return descriptions, None


def candidate_titles_for_topic(topic: str, count: int, rng: random.Random | None = None,
                                session=None, timeout: int = 10,
                                min_words: int = MIN_PASSAGE_WORDS) -> tuple[list[str], str | None]:
    """Discover up to `count` candidate Wikipedia article titles for `topic`,
    shuffled, via a live search rather than a fixed per-topic list.

    Survivors of the checks below are then screened by Wikipedia short
    description (see NOISY_DESCRIPTION_PATTERNS): a hit with no short
    description, or one describing a brand, broadcaster, list, disambiguation
    page, media title or jargon term, is dropped, since a literal title match
    on the topic word otherwise fills the pool with such pages. This costs
    one extra API call per 50 titles.

    A hit whose own whole-page word count came back under `min_words` is
    dropped without ever being fetched: this app's own extract can run past
    the lead section into the body (see _fetch_wikipedia_page) but never
    past the whole page, so a hit already reported shorter than the target
    there is guaranteed to fail the real readability check too, and fetching
    it anyway would only spend a slow network round trip to learn nothing
    new. A hit with no wordcount at all (unavailable, or an older mocked
    response in a test) is kept rather than dropped, since there's nothing
    to safely rule it out with. A hit whose own title matches
    INAPPROPRIATE_TOPIC_KEYWORDS is dropped the same way, for the same
    reason it isn't worth fetching — see that constant's comment; the final
    passage text gets checked again by fetch_passages, since a search hit's
    title alone can't catch everything.

    Returns `(titles, None)` normally, or `(titles, reason)` — `titles`
    empty — when the search itself failed (see WIKI_SEARCH_SKIP_REASONS), so
    a caller can tell a genuine "nothing on Wikipedia matched" apart from
    "couldn't reach Wikipedia's search at all" instead of treating both as
    silence.
    """
    rng = rng or random.Random()
    hits, reason = _wikipedia_search_titles(topic, limit=count, session=session, timeout=timeout)
    titles = [h["title"] for h in hits
              if (h["wordcount"] is None or h["wordcount"] >= min_words)
              and not _mentions_inappropriate_topic(h["title"])]
    if reason or not titles:
        return titles[:count], reason
    descriptions, reason = _wikipedia_short_descriptions(titles, session=session, timeout=timeout)
    if reason:
        return [], reason
    titles = [t for t in titles if _description_is_substantive(descriptions.get(t))]
    rng.shuffle(titles)
    return titles[:count], None


# Reasons _fetch_wikipedia_page can fail, and the honest (not-guessed) message
# fetch_passages shows for each one. "network"/"bad_response" are transient —
# the page may well exist — while "not_found"/"disambiguation"/"empty" are
# real properties of the page itself.
WIKI_FETCH_SKIP_REASONS = {
    "network": "couldn't reach Wikipedia (network error or timeout) — try again",
    "bad_response": "Wikipedia returned an unexpected response — try again",
    "not_found": "this page doesn't exist on Wikipedia (missing, moved, or renamed)",
    "disambiguation": "this title is a disambiguation page, not a single topic",
    "empty": "this Wikipedia page has no readable text",
}


def _fetch_wikipedia_page(title: str, lang: str = "en", session=None, timeout: int = 10,
                           thumb_size: int = 500) -> tuple[dict | None, str | None]:
    """Fetch a Wikipedia page's plain-text extract — not limited to the lead
    section, since for most articles that alone runs well under this app's
    650-750 word target — plus a thumbnail image URL sized to `thumb_size`
    pixels wide, in a single request to the MediaWiki Action API. This is
    deliberately not the REST "page summary" endpoint, whose extract is a
    short one-paragraph preview blurb, nor an intro-only extract, which was
    too short to reliably build a several-hundred-word passage from. Note
    that Wikipedia always renders an article's full text server-side before
    this request gets a response either way, so a candidate that clears the
    pre-fetch word-count hint (see candidate_titles_for_topic) can still take
    a real moment to come back — fetch_passages deliberately does not try to
    overlap several of these calls to cut that wait, since doing so hung
    indefinitely on at least one real user's Python/SSL setup; see
    fetch_passages' docstring for why.

    Returns `(page, None)` on success or `(None, reason)` on failure, where
    `reason` is one of the keys of WIKI_FETCH_SKIP_REASONS, so a caller can
    report an accurate, honest reason instead of a one-size-fits-all guess.
    Never raises, since a single skipped candidate should not stop the rest
    of a fetch batch.
    """
    http = session or requests
    url = WIKI_ACTION_API.format(lang=lang)
    params = dict(
        action="query", format="json", formatversion=2,
        prop="extracts|pageimages|pageprops|info",
        explaintext=1,
        piprop="thumbnail", pithumbsize=thumb_size,
        inprop="url", redirects=1, titles=title,
    )
    try:
        resp = http.get(url, params=params, timeout=timeout, headers={"User-Agent": USER_AGENT})
    except requests.RequestException:
        return None, "network"
    if resp.status_code != 200:
        return None, "bad_response"
    try:
        data = resp.json()
    except ValueError:
        return None, "bad_response"
    pages = (data.get("query") or {}).get("pages") or []
    if not pages:
        return None, "not_found"
    page = pages[0]
    if page.get("missing") or page.get("ns", 0) != 0:
        return None, "not_found"
    if (page.get("pageprops") or {}).get("disambiguation") is not None:
        return None, "disambiguation"
    extract = normalize_paragraph_breaks((page.get("extract") or "").strip())
    extract = strip_bare_headings(extract)
    if not extract:
        return None, "empty"
    thumbnail = page.get("thumbnail") or {}
    image_url = thumbnail.get("source") if isinstance(thumbnail, dict) else None
    return dict(
        title=page.get("title") or title,
        text=extract,
        source_name="Wikipedia",
        source_url=page.get("fullurl") or f"https://{lang}.wikipedia.org/wiki/{title.replace(' ', '_')}",
        license="CC BY-SA 4.0",
        image_url=image_url,
    ), None


def fetch_wikipedia_extract(title: str, lang: str = "en", session=None, timeout: int = 10,
                             thumb_size: int = 500) -> dict | None:
    """Thin wrapper around _fetch_wikipedia_page that drops the failure
    reason, kept for callers (and tests) that only need the page or None.
    fetch_passages calls _fetch_wikipedia_page directly so it can report an
    honest, specific skip reason instead of a one-size-fits-all guess.
    """
    page, _reason = _fetch_wikipedia_page(title, lang=lang, session=session, timeout=timeout,
                                           thumb_size=thumb_size)
    return page


_IMAGE_EXT_BY_CONTENT_TYPE = {
    "image/jpeg": "jpg", "image/png": "png", "image/gif": "gif",
    "image/svg+xml": "svg", "image/webp": "webp",
}
_KNOWN_IMAGE_EXTS = {"jpg", "jpeg", "png", "gif", "svg", "webp"}


def fetch_image_bytes(url: str, session=None, timeout: int = 10) -> tuple[bytes, str] | None:
    """Download a picture's bytes so it can be cached locally alongside the
    passage, the same way the passage text itself is fetched once and then
    used offline. Returns None on any failure (missing image, network error,
    empty body) rather than raising, since a passage without its picture is
    still useful; the caller decides whether to skip the picture or the
    whole passage.
    """
    if not url:
        return None
    http = session or requests
    try:
        resp = http.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
    except requests.RequestException:
        return None
    if resp.status_code != 200 or not resp.content:
        return None
    content_type = (resp.headers.get("Content-Type") or "").split(";")[0].strip().lower()
    ext = _IMAGE_EXT_BY_CONTENT_TYPE.get(content_type)
    if not ext:
        suffix = url.split("?")[0].rsplit(".", 1)[-1].lower()
        ext = suffix if suffix in _KNOWN_IMAGE_EXTS else "jpg"
    return resp.content, ext


def readability_stats(text: str) -> dict:
    words = text.split()
    sentences = [s for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    word_count = len(words)
    long_words = sum(1 for w in words if len(w.strip(".,;:!?\"'()")) >= 8)
    return dict(
        word_count=word_count,
        sentence_count=len(sentences),
        avg_sentence_len=word_count / max(1, len(sentences)),
        long_word_ratio=long_words / max(1, word_count),
    )


def is_reader_friendly(text: str, min_words: int = MIN_PASSAGE_WORDS, max_words: int = MAX_PASSAGE_WORDS,
                        max_avg_sentence_len: float = 35.0, max_long_word_ratio: float = 0.5) -> tuple[bool, dict]:
    """A deliberately simple readability heuristic (no NLP library involved):
    checks passage length, average sentence length, and the share of long
    (>=8 character) words. This is not a band-level filter — the real IELTS
    Academic Reading test targets a general, non-specialist but genuinely
    academic-register audience (undergraduate/postgraduate readers), not any
    single proficiency band, so these thresholds are set loosely enough to
    admit real academic prose and only reject the most extreme outliers
    (badly broken extracts, list-like or reference-heavy pages, and similar).

    `min_words`/`max_words` default to a 650-750 word target range, roughly
    matching one passage's length on the real test. A Wikipedia extract
    shorter than `min_words` is skipped rather than accepted, since it can
    only be trimmed down, never padded out, to reach the target length.
    """
    stats = readability_stats(text)
    ok = (min_words <= stats["word_count"] <= max_words
          and stats["avg_sentence_len"] <= max_avg_sentence_len
          and stats["long_word_ratio"] <= max_long_word_ratio)
    return ok, stats


def _quote_is_verifiable(quote: str, passage_text: str) -> bool:
    """Whitespace- and case-insensitive substring check: an explanation's
    quoted evidence must actually appear in the passage, so a learner is
    never pointed at a sentence the source text doesn't contain."""
    normalize = lambda s: " ".join(s.split()).casefold()
    return normalize(quote) in normalize(passage_text)


def normalize_for_matching(text: str) -> str:
    """Casefold and strip punctuation/whitespace so a free-text reading
    answer (sentence_completion / short_answer) can be compared to the
    passage's exact wording without being thrown off by capitalization, a
    trailing period, or extra spaces."""
    cleaned = re.sub(r"[^\w\s]", "", text.casefold())
    return " ".join(cleaned.split())


def label_paragraphs(text: str) -> list[tuple[str, str]]:
    """Split passage text into paragraphs and label them A, B, C... for the
    paragraph-based question kinds (matching_information, matching_headings)
    and for showing paragraph labels in the passage view when either is
    present. Must be called on the same, already paragraph-normalized text
    that is stored and displayed, so the labels a learner sees always match
    the labels the AI was given when it wrote these questions."""
    paragraphs = [p.strip() for p in _PARAGRAPH_SPLIT.split(text) if p.strip()]
    return list(zip(string.ascii_uppercase, paragraphs))


_WORD_RE = re.compile(r"[A-Za-z0-9']+")
_QUOTE_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "be", "been", "being",
    "of", "in", "on", "at", "to", "for", "with", "by", "from", "as", "that", "this", "these",
    "those", "it", "its", "their", "his", "her", "he", "she", "they", "which", "who", "whom",
    "than", "then", "so", "such", "also", "not", "no", "do", "does", "did", "has", "have", "had",
    "will", "would", "can", "could", "should", "may", "might", "must", "about", "into", "over",
    "after", "before", "between", "during", "while", "up", "down", "out", "off", "if", "because",
    "passage", "states", "explains", "says", "mentions", "text", "according",
}


def _content_words(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall(text.casefold()) if w not in _QUOTE_STOPWORDS and len(w) > 2}


def highlight_passage_html(passage_text: str, highlights: list[tuple[str | None, int]]) -> str:
    """Wraps each (quote, number) pair's matching sentence in `passage_text` with
    an HTML <mark> highlight carrying a small numbered badge, so a learner sees
    exactly where in the full passage a comprehension answer's evidence came
    from by reading the passage itself, rather than being handed the sentence
    in a separate quote box — that "find it yourself" step is the actual
    reading skill being practised.

    Returns the passage as HTML-escaped text with only the <mark>/<sup> tags
    left unescaped, safe to render with unsafe HTML allowed. A quote that is
    empty/None, or that can't be located in the passage (case/whitespace
    differences aside), is simply left unhighlighted rather than raising —
    the same graceful degrade as showing no quote at all. Two highlights
    whose spans overlap keep only the earlier one, so the markup never nests.
    """
    spans = []
    for quote, number in highlights:
        if not quote:
            continue
        words = quote.split()
        if not words:
            continue
        pattern = r"\s+".join(re.escape(w) for w in words)
        m = re.search(pattern, passage_text, flags=re.IGNORECASE)
        if m:
            spans.append((m.start(), m.end(), number))
    if not spans:
        return html.escape(passage_text)
    spans.sort()
    out, cursor, used_end = [], 0, -1
    for start, end, number in spans:
        if start < used_end:
            continue
        out.append(html.escape(passage_text[cursor:start]))
        out.append(f'<mark class="reading-highlight">{html.escape(passage_text[start:end])}'
                    f'<sup>{number}</sup></mark>')
        cursor = end
        used_end = end
    out.append(html.escape(passage_text[cursor:]))
    return "".join(out)


def find_supporting_sentence(explanation: str, answer_text: str, passage_text: str,
                              min_shared: int = 2) -> str | None:
    """Best-effort fallback for comprehension questions saved before source_quote
    existed (or where the AI's own quote didn't verify): picks whichever sentence
    in the passage shares the most content words with the explanation and the
    correct answer text, so a learner still gets a pointer to roughly where the
    information is, even though this sentence wasn't chosen or verified by the AI
    at generation time the way a real source_quote is. Returns None rather than a
    weak guess when no sentence clears the `min_shared` overlap bar, so callers can
    fall back to showing nothing, same as before this existed.
    """
    query = _content_words(f"{explanation} {answer_text}")
    if not query:
        return None
    best_sentence, best_score = None, 0
    for sentence in _SENTENCE_SPLIT.split(passage_text):
        sentence = sentence.strip()
        if not sentence or len(sentence) > 300:
            continue
        score = len(query & _content_words(sentence))
        if score > best_score:
            best_sentence, best_score = sentence, score
    return best_sentence if best_score >= min_shared else None


def _trim_sentences(text: str, max_words: int) -> str:
    """Keep whole sentences from a single block of text within a word budget."""
    sentences = [s for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    kept, count = [], 0
    for s in sentences:
        n = len(s.split())
        if kept and count + n > max_words:
            break
        kept.append(s)
        count += n
    return " ".join(kept) if kept else text


def trim_to_word_budget(text: str, max_words: int) -> str:
    """Trim to whole sentences within a word budget rather than cutting
    mid-sentence, while preserving the source's original paragraph breaks
    (blank lines between paragraphs) instead of collapsing everything into
    a single run-on block of text.

    Paragraphs are kept whole as long as they fit the remaining budget; the
    paragraph that would overflow it keeps only as many whole sentences as
    still fit, and nothing after it is included.
    """
    if len(text.split()) <= max_words:
        return text
    paragraphs = [p for p in _PARAGRAPH_SPLIT.split(text) if p.strip()]
    if len(paragraphs) <= 1:
        return _trim_sentences(text, max_words)
    kept_paragraphs, count = [], 0
    for para in paragraphs:
        n = len(para.split())
        if count + n <= max_words:
            kept_paragraphs.append(para)
            count += n
            continue
        remaining = max_words - count
        if remaining <= 0:
            if kept_paragraphs:
                # The budget was already used up by earlier paragraphs; stop
                # here rather than forcing in an overflowing extra sentence.
                break
            trimmed = _trim_sentences(para, 1)
        else:
            trimmed = _trim_sentences(para, remaining)
        if trimmed:
            kept_paragraphs.append(trimmed)
        break
    return "\n\n".join(kept_paragraphs) if kept_paragraphs else text


# Fixed answer options for the two True/False/Not-Given-style question kinds,
# mirroring the real IELTS Academic Reading task formats of the same name.
TRUE_FALSE_NOT_GIVEN_OPTIONS = ["True", "False", "Not Given"]
YES_NO_NOT_GIVEN_OPTIONS = ["Yes", "No", "Not Given"]

# Kinds answered by picking one of several candidate options (radio-button
# style): the three "judge a statement" kinds above, plus four more real
# IELTS task formats that are all, once simplified for a single learner's
# app, really "choose the correct option from a list" — matching a stem to
# the right sentence ending, feature, paragraph, or heading.
SELECTABLE_KINDS = {
    "multiple_choice", "true_false_not_given", "yes_no_not_given",
    "matching_sentence_endings", "matching_features",
    "matching_information", "matching_headings",
}
# Kinds answered by typing a short, word-limited phrase copied from the
# passage, rather than picking an option: real IELTS sentence/short-answer
# completion. Note/table/flow-chart and summary completion are folded into
# sentence_completion here — they are the same "fill the gap from the
# passage" mechanic, just presented as one paragraph with several gaps on
# the real test, which this app answers one gap at a time instead.
FREE_TEXT_KINDS = {"sentence_completion", "short_answer"}
# Kinds whose correct answer is a paragraph label (e.g. "Paragraph B") rather
# than free-form text, so the passage needs its paragraphs visibly labeled
# A, B, C... wherever a question of this kind is being read or reviewed.
PARAGRAPH_KINDS = {"matching_information", "matching_headings"}


class ComprehensionQuestion(BaseModel):
    """A single comprehension item. `kind` selects which real IELTS Academic
    Reading task format this is modeled on.

    The SELECTABLE_KINDS all share one shape (a prompt/statement, a list of
    options, and the index of the correct one), so the app's answer UI,
    scoring, and review screen can treat every one of them the same way
    rather than needing a special case per kind. The FREE_TEXT_KINDS share a
    different shape instead (a target `answer_text` and a `max_words`
    limit); `options`/`answer` are unused for those and simply left at their
    defaults.
    """
    model_config = ConfigDict(extra="forbid")
    kind: Literal["multiple_choice", "true_false_not_given", "yes_no_not_given",
                  "sentence_completion", "short_answer",
                  "matching_sentence_endings", "matching_features",
                  "matching_information", "matching_headings"] = "multiple_choice"
    # For multiple_choice/matching_* this is the question or matching stem;
    # for true_false_not_given/yes_no_not_given it's the statement or claim
    # the learner judges (written as a statement, not a question); for the
    # free-text kinds it's the fill-in-the-blank sentence (with a blank
    # shown as ___) or the short-answer question itself.
    prompt: str = Field(min_length=10, max_length=500)
    options: list[str] = Field(default_factory=list, max_length=10)
    answer: int = Field(default=0, ge=0)
    explanation: str = Field(min_length=10, max_length=500)
    # The exact sentence or phrase from the passage that the correct answer
    # comes from, shown to the learner alongside the explanation so they can
    # see exactly where in the passage that information appears. Verified
    # against the real passage text in fetch_passages(), since the schema
    # itself has no access to the passage to check against. Required for
    # every answer except a SELECTABLE_KINDS answer of "Not Given" — by
    # definition the passage doesn't say either way, so there is nothing in
    # it to quote.
    source_quote: Optional[str] = Field(default=None, max_length=300)
    # FREE_TEXT_KINDS only: the exact word(s) from the passage that fill the
    # gap or answer the question, and the word-limit the learner is told to
    # stay within (matching the real test's "no more than N words" rule).
    answer_text: Optional[str] = Field(default=None, max_length=100)
    max_words: Optional[int] = Field(default=None, ge=1, le=5)

    @model_validator(mode="after")
    def check_shape(self):
        if self.kind in FREE_TEXT_KINDS:
            if not self.answer_text or not self.answer_text.strip():
                raise ValueError(f"A {self.kind} question needs a non-empty answer_text.")
            if not self.max_words or not (1 <= self.max_words <= 5):
                raise ValueError(f"A {self.kind} question needs a max_words between 1 and 5.")
            if len(self.answer_text.split()) > self.max_words:
                raise ValueError("answer_text is longer than its own max_words limit.")
            if not self.source_quote or len(self.source_quote.strip()) < 5:
                raise ValueError(f"A {self.kind} question must cite a source_quote "
                                  "containing the answer.")
            return self

        if not (0 <= self.answer < len(self.options)):
            raise ValueError("answer must be a valid index into options.")
        if self.kind == "multiple_choice":
            if len(self.options) != 4:
                raise ValueError("A multiple-choice question needs exactly four options.")
            if len({o.strip().casefold() for o in self.options}) != 4:
                raise ValueError("A comprehension question must have four different choices.")
            if any(not o.strip() or len(o) > 300 for o in self.options):
                raise ValueError("Answer choices must be short and nonempty.")
        elif self.kind in ("true_false_not_given", "yes_no_not_given"):
            expected = (TRUE_FALSE_NOT_GIVEN_OPTIONS if self.kind == "true_false_not_given"
                        else YES_NO_NOT_GIVEN_OPTIONS)
            if self.options != expected:
                raise ValueError(f"A {self.kind} question must use exactly the options {expected}.")
        else:
            # matching_sentence_endings, matching_features, matching_information,
            # matching_headings: a free-form list of candidate options.
            if len(self.options) < 3:
                raise ValueError(f"A {self.kind} question needs at least 3 candidate options.")
            if len({o.strip().casefold() for o in self.options}) != len(self.options):
                raise ValueError(f"A {self.kind} question's options must all be distinct.")
        is_not_given = self.options[self.answer] == "Not Given"
        if is_not_given and self.source_quote:
            raise ValueError("A 'Not Given' answer must not cite a source_quote — "
                              "nothing in the passage supports or contradicts it.")
        if not is_not_given and not (self.source_quote and len(self.source_quote.strip()) >= 5):
            raise ValueError("Every answer other than 'Not Given' must cite a real source_quote.")
        return self


class VocabCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    word: str = Field(min_length=1, max_length=60)
    pos: str = Field(min_length=1, max_length=30)
    definition: str = Field(min_length=5, max_length=300)
    vi: str = Field(default="", max_length=300)
    example: str = Field(min_length=5, max_length=300)
    collocation: str = Field(default="", max_length=200)
    cloze: str = Field(min_length=5, max_length=300)
    distractors: str = Field(min_length=3, max_length=300)

    @model_validator(mode="after")
    def check_shape(self):
        if self.cloze.count("___") != 1:
            raise ValueError("cloze must contain exactly one ___")
        if self.word.casefold() not in self.example.casefold():
            raise ValueError("example must contain the word")
        choices = [v.strip() for v in self.distractors.split(";")]
        if len(choices) != 3 or len({v.casefold() for v in choices + [self.word]}) != 4:
            raise ValueError("need three distinct distractors, distinct from the word")
        return self


class PassageBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # The prompt asks the model to aim for the full 12 (max_length below), so
    # salvage normally has real headroom before any dropped item threatens
    # the floor. The floor itself is deliberately set much lower than the
    # target: discarding an entire generated bundle -- real, already-spent
    # output tokens -- just because losing a couple of items out of 12 fell
    # short of some higher number wastes far more than accepting a
    # shorter-but-still-good 6-question set would. 6 is still a legitimate
    # mini reading test, not a broken one.
    comprehension: list[ComprehensionQuestion] = Field(min_length=6, max_length=12)
    vocabulary: list[VocabCandidate] = Field(default_factory=list, max_length=6)

    @model_validator(mode="after")
    def matching_headings_share_one_pool(self):
        # Real matching_headings items draw from one shared list of candidate
        # headings for the whole passage; check the AI kept that list
        # consistent across every matching_headings item it produced, rather
        # than inventing a different list per paragraph.
        heading_items = [q for q in self.comprehension if q.kind == "matching_headings"]
        if heading_items and any(q.options != heading_items[0].options for q in heading_items):
            raise ValueError("All matching_headings items must share the same candidate heading list.")
        return self


BUNDLE_INSTRUCTIONS = """You are helping a learner preparing for IELTS Academic Reading.
You will be given a nonfiction passage that has already been selected and verified as real,
sourced text, split into paragraphs labeled A, B, C, and so on. Do not rewrite, summarize,
shorten, or add to the passage itself in your output; only respond about it. Produce:
- 12 comprehension items in total -- aim for the full 12 rather than settling for fewer; only
  produce fewer than 12 if the passage genuinely cannot support that many distinct, non-trivial,
  non-redundant questions, modeled on real IELTS Academic Reading question types, covering as
  many of these kinds as the passage allows, roughly in this mix:
  - 2 multiple_choice questions about facts or details actually stated in the passage: four
    distinct plausible choices, a zero-based correct answer index, a brief explanation, and a
    source_quote (the exact sentence or short phrase copied word-for-word from the passage, no
    paraphrasing, containing the evidence for the correct answer). Wrong choices must be
    realistic distractors in the style of the real test — a plausible paraphrase of a
    different detail in the passage, a common misreading, or a statement that's partly right
    but wrong in one specific way — never random or obviously unrelated wrong choices.
  - 2 true_false_not_given questions: a standalone factual statement about the passage
    (written as a statement, not a question), with options set to exactly ["True", "False",
    "Not Given"] and the zero-based index of the correct one. Use "True" when the passage
    states this, "False" when the passage states the opposite or a contradicting fact, and
    "Not Given" when the passage simply does not say either way (different from False: Not
    Given means the information just is not in the passage at all, not that it is
    contradicted). Include a genuine mix of these three answers, not just True/False — do not
    skip Not Given. A True or False statement must include a source_quote: the exact passage
    sentence or phrase that proves it. A Not Given statement must NOT include a source_quote,
    since by definition nothing in the passage supports or contradicts it.
  - Only if the passage expresses an actual claim, judgment, or viewpoint (not just neutral
    facts), up to 1 yes_no_not_given question: the same shape and Not Given rules as
    true_false_not_given, but for a claim rather than a fact, with options set to exactly
    ["Yes", "No", "Not Given"]. Many factual passages will not have suitable material for this
    — it is fine, and expected, to produce zero of these.
  - 2 items combining sentence_completion and short_answer: a gap-fill sentence (with the gap
    shown as ___) or a short factual question, each answered with a short phrase copied
    word-for-word from the passage. Set answer_text to that exact phrase, max_words to how
    many words the learner is told to use (1 to 5, matching the real test's "no more than N
    words" instruction — pick the smallest number that still fits the real answer), and
    source_quote to the passage sentence the answer comes from. Never require an answer longer
    than its own max_words.
  - 2 items combining matching_sentence_endings and matching_features: a stem (the first half
    of a sentence about the passage, or a named person/place/thing from it) plus 4 to 6
    candidate options to complete or match it, exactly one of them correct, with a
    source_quote proving the correct match. Wrong options should be plausible but clearly
    wrong on a careful re-read, not nonsensical.
  - 2 items combining matching_information and matching_headings, using the paragraph labels
    you were given with the passage:
    - matching_information: a statement about a specific detail, with options set to the
      passage's actual paragraph labels formatted as "Paragraph A", "Paragraph B", and so on
      (only labels that really exist in this passage), and the correct index pointing to the
      paragraph that actually contains that detail, with a source_quote from within that
      paragraph.
    - matching_headings: pick one shared list of 5 to 8 short candidate headings (deliberately
      more headings than paragraphs, so some are distractors) that could summarize a
      paragraph's main idea, and for each matching_headings item ask which heading fits a
      specific paragraph (state which paragraph in the prompt, e.g. "Which heading best fits
      Paragraph B?"). If you produce more than one matching_headings item, every single one of
      them must set options to that exact same list of headings: the identical strings, in the
      identical order, not just the same headings reworded or reordered — copy the list itself
      rather than retyping it from memory for each item, since even a small wording difference
      between items makes the whole set invalid. Give at most one matching_headings item per
      real paragraph, and never give two different paragraphs the same correct heading.
  Every comprehension item must include a "kind" field set to exactly one of
  "multiple_choice", "true_false_not_given", "yes_no_not_given", "sentence_completion",
  "short_answer", "matching_sentence_endings", "matching_features", "matching_information", or
  "matching_headings", matching which type it is.
- Up to 6 candidate vocabulary words actually used in the passage that would be useful for an
  IELTS Academic Reading candidate to study and are not extremely basic. Skip this list
  entirely if nothing suitable stands out; do not invent words not present in the passage. For
  each word, in the exact schema this app's vocabulary bank uses, supply: word, part of speech
  (pos), a clear definition, a Vietnamese gloss (vi), an example sentence containing the word
  (may reuse or adapt a sentence from the passage), an optional useful phrase (collocation), a
  cloze sentence with exactly one ___ where the word goes, and three distinct incorrect cloze
  choices (distractors) separated by semicolons, all clearly different from the correct word
  and from each other.
Use short, clear English throughout. Self-check every comprehension answer against the
passage text before finalizing, and do not ask about anything the passage does not say."""


# Lenient counterparts of ComprehensionQuestion/VocabCandidate/PassageBundle,
# used only as the structured-output type for the API call itself: they keep
# every field's type/length constraints (so the request's JSON schema is
# unchanged) but drop each item's own cross-field `check_shape` validator, by
# overriding it with a no-op. That means a single malformed item (a
# short_answer answer_text longer than its own max_words, a vocabulary item
# with a malformed cloze blank, and so on) no longer makes the ENTIRE parse
# raise and the whole bundle get thrown away. generate_bundle then
# re-validates each item individually against the real, strict model and
# keeps only the ones that actually pass — see _salvage_bundle.
class _LenientComprehensionQuestion(ComprehensionQuestion):
    @model_validator(mode="after")
    def check_shape(self):
        return self


class _LenientVocabCandidate(VocabCandidate):
    @model_validator(mode="after")
    def check_shape(self):
        return self


class _LenientPassageBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")
    comprehension: list[_LenientComprehensionQuestion] = Field(min_length=6, max_length=12)
    vocabulary: list[_LenientVocabCandidate] = Field(default_factory=list, max_length=6)


def _reconcile_matching_headings_pool(items: list[ComprehensionQuestion]) -> list[ComprehensionQuestion]:
    """Real matching_headings items are supposed to all draw from one shared
    list of candidate headings for the whole passage, but the model doesn't
    reliably keep that list word-for-word identical across every
    matching_headings item it drafts (a reworded heading, reordered options,
    or an extra distractor in one item but not another all count as
    "different" to PassageBundle's own exact-match check). That used to
    reject the entire bundle — a whole-bundle problem, not attributable to
    a single item, so it wasn't something the per-item salvage in
    _salvage_bundle could fix. In practice it usually *is* attributable to
    one item drifting from what the others agree on, so this keeps the
    heading list most of the matching_headings items actually share and
    drops just the item(s) that used a different one, the same
    keep-the-good-ones approach as every other salvage pass. Does nothing
    if there's at most one matching_headings item (nothing to disagree
    with) or they already agree.
    """
    heading_items = [q for q in items if q.kind == "matching_headings"]
    if len(heading_items) <= 1:
        return items
    pool_counts = Counter(tuple(q.options) for q in heading_items)
    if len(pool_counts) == 1:
        return items
    canonical_pool = pool_counts.most_common(1)[0][0]
    return [q for q in items if q.kind != "matching_headings" or tuple(q.options) == canonical_pool]


def _salvage_bundle(lenient: _LenientPassageBundle) -> PassageBundle:
    """Turn a leniently-parsed bundle into a real, strict PassageBundle by
    re-validating each comprehension/vocabulary item on its own and keeping
    only the ones that pass, instead of discarding an otherwise-good bundle
    over one or two bad items — then reconciling matching_headings items
    that disagree on their shared heading pool the same way (see
    _reconcile_matching_headings_pool). Still raises ValueError (a pydantic
    ValidationError, same as before this existed) exactly when the result
    isn't usable even after all of that — too few comprehension items
    survive (PassageBundle's own min_length=6), since that isn't
    attributable to a single dropped or reconciled item.
    """
    good_comprehension = []
    for item in lenient.comprehension:
        try:
            good_comprehension.append(ComprehensionQuestion.model_validate(item.model_dump()))
        except ValidationError:
            continue
    good_comprehension = _reconcile_matching_headings_pool(good_comprehension)
    good_vocabulary = []
    for item in lenient.vocabulary:
        try:
            good_vocabulary.append(VocabCandidate.model_validate(item.model_dump()))
        except ValidationError:
            continue
    return PassageBundle.model_validate(dict(
        comprehension=[q.model_dump() for q in good_comprehension],
        vocabulary=[v.model_dump() for v in good_vocabulary],
    ))


def _drop_ungrounded_items(bundle: PassageBundle, passage_text: str) -> PassageBundle:
    """A second salvage pass, run after _salvage_bundle: drop any
    comprehension item whose citation doesn't actually check out against the
    real passage text — an unverifiable source_quote, a free-text answer
    that isn't really in the passage, or (matching_information only) a
    reference to a paragraph that doesn't exist — instead of discarding an
    otherwise-good bundle over a single bad citation. The schema itself has
    no access to the passage text to check any of this, which is why it
    happens here rather than as a model validator.

    Only matching_information's options are literally paragraph labels
    ("Paragraph A", ...); matching_headings' options are headings, with its
    paragraph reference left as prose inside `prompt`, so it isn't checked
    the same way.

    Re-validates the result as a real PassageBundle, exactly like
    _salvage_bundle does, so this still raises ValueError (a pydantic
    ValidationError) if too few grounded items survive (PassageBundle's own
    min_length=6). Dropping items here can't reintroduce a matching_headings
    pool mismatch — _salvage_bundle already reconciled that before this
    function ever runs, and removing an item can only leave the survivors
    agreeing more, never less.
    """
    real_labels = {f"Paragraph {label}" for label, _ in label_paragraphs(passage_text)}

    def is_grounded(q: ComprehensionQuestion) -> bool:
        if q.source_quote and not _quote_is_verifiable(q.source_quote, passage_text):
            return False
        if (q.kind in FREE_TEXT_KINDS
                and normalize_for_matching(q.answer_text) not in normalize_for_matching(passage_text)):
            return False
        if q.kind == "matching_information" and q.options[q.answer] not in real_labels:
            return False
        return True

    good_comprehension = [q for q in bundle.comprehension if is_grounded(q)]
    return PassageBundle.model_validate(dict(
        comprehension=[q.model_dump() for q in good_comprehension],
        vocabulary=[v.model_dump() for v in bundle.vocabulary],
    ))


def generate_bundle(passage_text: str, api_key: str, model: str = "gpt-4.1-mini", client=None) -> PassageBundle:
    if not api_key.strip():
        raise ValueError("Add your OpenAI API key first.")
    client = client or OpenAI(api_key=api_key, timeout=60.0, max_retries=0)
    # The model gets the passage pre-split into labeled paragraphs, using the
    # exact same split the app itself uses to label paragraphs when showing
    # them (label_paragraphs()), so matching_information/matching_headings
    # answers line up with what the learner actually sees.
    paragraphs = label_paragraphs(passage_text)
    response = client.responses.parse(
        model=model, store=False, instructions=BUNDLE_INSTRUCTIONS,
        input=json.dumps({
            "passage": passage_text,
            "paragraphs": [{"label": label, "text": para} for label, para in paragraphs],
        }, ensure_ascii=False),
        text_format=_LenientPassageBundle, max_output_tokens=5000,
    )
    if response.status != "completed" or response.output_parsed is None:
        raise ValueError("OpenAI did not return a complete reading bundle. Nothing was saved.")
    bundle = _salvage_bundle(response.output_parsed)
    return _drop_ungrounded_items(bundle, passage_text)


def fetch_passages(topic: str, count: int, api_key: str, model: str = "gpt-4.1-mini",
                    existing_titles=(), session=None, client=None, rng: random.Random | None = None,
                    on_skip=None) -> list[dict]:
    """Fetch up to `count` new, reader-appropriate passages for `topic`, each bundled
    with AI-drafted comprehension questions and vocabulary candidates. Titles already
    in `existing_titles` are skipped. May return fewer than `count` if
    not enough suitable source material was found or a candidate failed readability or
    bundle validation; a partial result is still useful and is not an error.

    `on_skip`, if given, is called as `on_skip(title, reason)` for every candidate
    title that did not become a passage, so a caller can show why a fetch came back
    empty or short instead of just a dead-end "nothing found".

    Candidates are fetched one at a time, deliberately not concurrently. An
    earlier version of this function fetched a small batch of candidates in
    parallel threads to cut down the wall-clock cost of Wikipedia's per-page
    server-side render time; that was reverted after it caused fetches to
    hang indefinitely for a real user, on a Python build whose SSL stack
    (LibreSSL, not OpenSSL — flagged by urllib3 itself as unsupported for
    concurrent use) does not reliably support multiple simultaneous HTTPS
    requests. Since this app has no way to know in advance which SSL stack
    it's running under, and a hang is worse than being slow, fetching stays
    strictly sequential.
    """
    def skip(title, reason):
        if on_skip:
            on_skip(title, reason)

    rng = rng or random.Random()
    seeds, search_failure = candidate_titles_for_topic(topic, count=SEARCH_POOL_SIZE, rng=rng, session=session)
    if search_failure:
        skip(topic, WIKI_SEARCH_SKIP_REASONS.get(search_failure, WIKI_SEARCH_SKIP_REASONS["network"]))
        return []
    if not seeds:
        skip(topic, "Wikipedia's search returned no candidate pages for this topic")
        return []
    results = []
    for title in seeds:
        if len(results) >= count:
            break
        if title in existing_titles:
            skip(title, "already in your reading list")
            continue
        page, fetch_failure = _fetch_wikipedia_page(title, session=session)
        if not page:
            skip(title, WIKI_FETCH_SKIP_REASONS.get(fetch_failure, WIKI_FETCH_SKIP_REASONS["network"]))
            continue
        # Redundant with candidate_titles_for_topic's own title check (this
        # catches a redirect landing on an unsuitable page under a different
        # title than the one searched, or a direct call that skipped that
        # filter) — cheap, and worth doing before spending an OpenAI call.
        if _mentions_inappropriate_topic(page["title"]):
            skip(title, "not appropriate for this app")
            continue
        text = trim_to_word_budget(page["text"], max_words=MAX_PASSAGE_WORDS)
        if _mentions_inappropriate_topic(text):
            # Checked against the actual passage text (not the whole,
            # untrimmed extract): an article whose body drifts into
            # unsuitable territory well past where this app's own budget
            # trims it off never actually reaches the learner, so it isn't
            # rejected over content nobody will see.
            skip(title, "not appropriate for this app")
            continue
        ok, stats = is_reader_friendly(text)
        if not ok:
            skip(title, f"too complex or too short/long for this reading level "
                        f"({stats['word_count']} words, {stats['avg_sentence_len']:.0f} words/sentence, "
                        f"{stats['long_word_ratio']:.0%} long words)")
            continue
        try:
            # generate_bundle itself drops any individual comprehension item
            # whose citation doesn't actually check out against `text` (an
            # unverifiable source_quote, a free-text answer not really in the
            # passage, or a matching_information reference to a paragraph
            # that doesn't exist) rather than requiring the whole bundle to
            # be clean — see _drop_ungrounded_items. This can still raise
            # ValueError, exactly like a structurally-bad bundle does, if too
            # few grounded items survive.
            bundle = generate_bundle(text, api_key, model=model, client=client)
        except ValueError as e:
            skip(title, f"AI couldn't draft a usable activity for this passage ({e})")
            continue
        image_bytes = image_ext = None
        image = fetch_image_bytes(page.get("image_url"), session=session)
        if image:
            image_bytes, image_ext = image
        results.append(dict(
            id=passage_id(topic, page["title"]),
            title=page["title"],
            topic=topic,
            body=text,
            source_name=page["source_name"],
            source_url=page["source_url"],
            license=page["license"],
            is_excerpt=False,
            word_count=stats["word_count"],
            image_bytes=image_bytes,
            image_ext=image_ext,
            comprehension=[q.model_dump() for q in bundle.comprehension],
            vocabulary=[v.model_dump() for v in bundle.vocabulary],
        ))
    return results
