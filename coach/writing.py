"""IELTS Academic Writing Task 2 practice: a learner writes a full essay in
response to a prompt (with or without a 40-minute timer), and AI grades the
essay against the four official Task 2 criteria — Task Response, Coherence
and Cohesion, Lexical Resource, and Grammatical Range and Accuracy — with a
band estimate and concrete "how to reach the next band" feedback for each.

Unlike Reading, the prompts here have no independent "real source" to fetch
and verify the way a Wikipedia passage does — an essay question is
necessarily authored, not discovered. So, deliberately, prompts are never
AI-generated: they come from a curated PSV file the same way the vocabulary
bank does (see coach/content.py's load_words), maintained and extended by a
parent/tutor rather than drafted on the fly. AI is only ever used for the one
job it's suited to here: judging an already-written essay against a fixed,
external rubric.

Also unlike Reading, there is no "try a different candidate" safety net once
someone has spent up to 40 minutes writing: a bad AI response can't be
quietly swapped for another attempt, so generate_evaluation is deliberately
more lenient about cosmetic issues (an out-of-range or off-grid band value is
snapped rather than rejected) than reading.py's bundle generation is, while
still refusing to trust any specific claim about the essay's wording unless
it's a verbatim quote that actually appears in what the learner wrote — see
_drop_ungrounded_evidence.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field

DATA = Path(__file__).resolve().parent.parent / "data" / "essay_prompts.psv"

# The four real IELTS Academic Writing Task 2 essay formats. A prompt's
# essay_type is purely descriptive (it helps a learner pick what to
# practise); it never changes how a submitted essay is graded — every essay
# is judged against the same four criteria regardless of type.
ESSAY_TYPES = {
    "opinion": "Opinion (agree or disagree)",
    "discussion": "Discussion (both views)",
    "problem_solution": "Problem and solution",
    "advantage_disadvantage": "Advantages and disadvantages",
}

# How many minutes Simulation mode allows before locking the essay box, sized
# to the real IELTS Academic Writing Task 2 time allocation.
SIMULATION_MINUTES = 40

# The real test's own minimum; going under doesn't cap the band automatically
# (see the app's Task Response feedback for how a short essay is actually
# marked down), but is surfaced in the UI as guidance while writing.
MIN_RECOMMENDED_WORDS = 250


def load_essay_prompts(path: Path = DATA) -> list[dict]:
    """Load and validate the essay-prompt bank from a PSV file shaped like
    the vocabulary bank: one row per prompt, `|`-delimited, maintained by
    hand rather than generated. A prompt with no `id` gets one derived from
    its own text, the same fallback load_words uses for a word's `id`."""
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="|"))
    seen = set()
    for r in rows:
        for key in ("topic", "essay_type", "prompt"):
            if not (r.get(key) or "").strip():
                raise ValueError(f"Missing essay prompt field '{key}' for row: {r}")
        if r["essay_type"] not in ESSAY_TYPES:
            raise ValueError(f"Unknown essay_type '{r['essay_type']}' — "
                              f"must be one of {sorted(ESSAY_TYPES)}")
        r["id"] = (r.get("id") or "").strip() or hashlib.sha256(
            r["prompt"].strip().casefold().encode()).hexdigest()[:16]
        if r["id"] in seen:
            raise ValueError(f"Duplicate essay prompt ID: {r['id']}")
        seen.add(r["id"])
        r["topic"] = r["topic"].strip()
    return rows


# The four criteria every Task 2 essay is scored against, in the order the
# real test lists them. Kept as a tuple (not just the EssayEvaluation
# model's own field order) so app.py and storage.py have one shared,
# ordered source of truth for iterating over them.
CRITERIA = ("task_response", "coherence_cohesion", "lexical_resource", "grammatical_range_accuracy")
CRITERION_LABELS = {
    "task_response": "Task Response",
    "coherence_cohesion": "Coherence and Cohesion",
    "lexical_resource": "Lexical Resource",
    "grammatical_range_accuracy": "Grammatical Range and Accuracy",
}


class QuotedIssue(BaseModel):
    """One specific, checkable observation about the essay: a verbatim
    excerpt (`quote`) plus what it shows. `quote` is checked against the
    real essay text in _drop_ungrounded_evidence — the schema itself has no
    access to the essay to verify this at parse time."""
    model_config = ConfigDict(extra="forbid")
    quote: str = Field(min_length=3, max_length=300)
    comment: str = Field(min_length=1, max_length=300)


class CriterionFeedback(BaseModel):
    """band is deliberately not constrained to the real half-band grid
    (1, 1.5, 2, ..., 9) at the schema level: generate_evaluation snaps it to
    the nearest real band itself after parsing (see snap_to_half_band)
    rather than risking the whole evaluation failing over a cosmetic value
    like 6.3, since — unlike a Reading bundle's list of comprehension
    items — there is no other criterion this one could be dropped in favor
    of if it failed strict validation outright."""
    model_config = ConfigDict(extra="forbid")
    band: float = Field(ge=1.0, le=9.0)
    assessment: str = Field(min_length=1, max_length=1200)
    evidence: list[QuotedIssue] = Field(default_factory=list, max_length=5)
    next_band_advice: str = Field(min_length=1, max_length=800)


class EssayEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task_response: CriterionFeedback
    coherence_cohesion: CriterionFeedback
    lexical_resource: CriterionFeedback
    grammatical_range_accuracy: CriterionFeedback


EVALUATION_INSTRUCTIONS = """You are an experienced IELTS Academic Writing examiner grading a Task 2
essay written by a teenage candidate. You will be given the essay prompt and the candidate's full,
unedited essay text as they actually submitted it — do not correct, rewrite, or improve the essay
yourself; only assess it.

Score the essay against exactly these four official IELTS Task 2 criteria, each on the real IELTS
band scale from 1 to 9 in half-band steps (1, 1.5, 2, 2.5, ..., 9):

- task_response: how fully the essay addresses every part of the prompt, whether it presents a
  clear, well-developed position, and whether main ideas are relevant, extended, and supported with
  evidence or examples (which may be from the candidate's own knowledge or experience). An essay
  that ignores part of the prompt, has no clear position, or is left incomplete (for example because
  the candidate ran out of time) should be marked down here specifically, not treated as a
  formatting problem.
- coherence_cohesion: how logically ideas are organised and how clearly the essay progresses,
  including paragraphing and the effective use of cohesive devices (linking words, pronouns,
  referencing) — not just their presence, but whether they are used accurately and naturally rather
  than mechanically or repetitively.
- lexical_resource: the range, precision, and appropriateness of vocabulary used, including whether
  less common words and collocations are used accurately, and how much spelling or word-formation
  errors affect the reader.
- grammatical_range_accuracy: the range of grammatical structures attempted (simple and complex)
  and how accurately and appropriately they are used, including punctuation.

For each criterion, also write:
- assessment: 2-4 sentences explaining why the essay earned this band for this specific criterion,
  in plain language a teenage learner can follow. Write this in general terms — do not embed a
  fresh quotation from the essay inside this text; every specific example belongs in `evidence`
  instead, so it can be checked against what the candidate actually wrote.
- evidence: up to 5 short, exact quotations copied word-for-word from the candidate's essay (never
  paraphrased, shortened with "...", or reconstructed from memory) that illustrate this criterion,
  each with a one-sentence comment on what it shows (a strength or a specific problem). It is fine,
  and often correct, to leave this empty for a strong criterion with nothing specific to flag.
- next_band_advice: 1-3 concrete, actionable sentences on what the candidate would specifically need
  to do differently to reach the next half-band up for THIS criterion — not generic advice like
  "use better vocabulary," but something tied to a real, common weakness in this particular essay.

Base every band strictly on what the essay actually demonstrates, not on how long or confident it
sounds. A short, incomplete, or off-topic essay should be scored low on task_response specifically,
even if its grammar and vocabulary are otherwise good. Self-check that every `evidence` quote is
copied exactly from the essay before finalizing. Return only the requested structured output."""


def snap_to_half_band(value: float) -> float:
    """IELTS bands are always a whole or half number (5.0, 5.5, 6.0, ...).
    Round the AI's raw score to the nearest one of those and clamp it to the
    real 1-9 range, rather than rejecting an entire evaluation — which,
    unlike a single bad Reading comprehension item, has no other item it
    could be swapped out for — over a cosmetic value like 6.3 or 9.2.

    Uses explicit round-half-up (via floor(x*2 + 0.5)) rather than Python's
    built-in round(), which rounds a tied value (exactly .25 or .75 before
    doubling) to the nearest *even* number and would otherwise snap 6.25 and
    6.75 in seemingly inconsistent directions.
    """
    snapped = math.floor(value * 2 + 0.5) / 2
    return min(9.0, max(1.0, snapped))


def overall_band(evaluation: EssayEvaluation) -> float:
    """Average the four (already snapped) criterion bands and round using
    the same whole/half-band convention IELTS publishes for combining
    component scores: an average ending in .25 rounds up to the next half
    band, and one ending in .75 rounds up to the next whole band; anything
    else rounds down to the nearest half band. This is computed here rather
    than trusted from the AI's own output, since it's pure arithmetic on
    values this app already has — no reason to risk an AI arithmetic
    mistake on something deterministic.

    This mirrors the public rounding rule IELTS uses when combining scores
    across the four skills into an overall band; it approximates, but is not
    identical to, the examiner's own internal Task 2 marking process, which
    this app has no access to (the same caveat that applies to the public
    band descriptors themselves — see the research notes in the project).
    """
    bands = [snap_to_half_band(getattr(evaluation, c).band) for c in CRITERIA]
    avg = sum(bands) / len(bands)
    whole = math.floor(avg)
    remainder = avg - whole
    if remainder < 0.25:
        return float(whole)
    if remainder < 0.75:
        return whole + 0.5
    return float(whole + 1)


def _quote_is_verifiable(quote: str, essay_text: str) -> bool:
    """Whitespace- and case-insensitive substring check: an evidence quote
    must actually appear in the essay the learner wrote, so no one is ever
    shown a claim about a sentence they didn't write. Same semantics as
    reading.py's _quote_is_verifiable, kept as its own small copy here
    rather than importing a private name across modules."""
    normalize = lambda s: " ".join(s.split()).casefold()
    return normalize(quote) in normalize(essay_text)


def _drop_ungrounded_evidence(evaluation: EssayEvaluation, essay_text: str) -> EssayEvaluation:
    """Drop any evidence quote that doesn't actually appear in the essay,
    instead of trusting an AI-cited quotation blindly — the schema itself
    has no access to the essay text to check this at parse time, which is
    why it happens here rather than as a model validator. A criterion whose
    evidence is entirely dropped still keeps its band, assessment, and
    next_band_advice; unlike Reading's comprehension items, there is no
    minimum evidence count to fall below, since evidence only ever
    illustrates the assessment rather than being the graded content itself.
    """
    data = evaluation.model_dump()
    for key in CRITERIA:
        data[key]["evidence"] = [e for e in data[key]["evidence"]
                                  if _quote_is_verifiable(e["quote"], essay_text)]
    return EssayEvaluation.model_validate(data)


def generate_evaluation(prompt_text: str, essay_text: str, api_key: str,
                         model: str = "gpt-4.1-mini", client=None) -> EssayEvaluation:
    if not api_key.strip():
        raise ValueError("Add your OpenAI API key first.")
    if not essay_text.strip():
        raise ValueError("Write an essay before submitting it for grading.")
    client = client or OpenAI(api_key=api_key, timeout=60.0, max_retries=0)
    response = client.responses.parse(
        model=model, store=False, instructions=EVALUATION_INSTRUCTIONS,
        input=json.dumps({"prompt": prompt_text, "essay": essay_text}, ensure_ascii=False),
        text_format=EssayEvaluation, max_output_tokens=3000,
    )
    if response.status != "completed" or response.output_parsed is None:
        raise ValueError("OpenAI did not return a complete evaluation. Nothing was saved.")
    data = response.output_parsed.model_dump()
    for key in CRITERIA:
        data[key]["band"] = snap_to_half_band(data[key]["band"])
    evaluation = EssayEvaluation.model_validate(data)
    return _drop_ungrounded_evidence(evaluation, essay_text)
