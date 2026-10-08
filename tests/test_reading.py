import random
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from coach.content import DATA, append_words, load_words
from coach.reading import (
    FREE_TEXT_KINDS,
    MIN_PASSAGE_WORDS,
    PARAGRAPH_KINDS,
    READING_TOPICS,
    SEARCH_POOL_SIZE,
    TRUE_FALSE_NOT_GIVEN_OPTIONS,
    YES_NO_NOT_GIVEN_OPTIONS,
    ComprehensionQuestion,
    PassageBundle,
    VocabCandidate,
    candidate_titles_for_topic,
    fetch_image_bytes,
    fetch_passages,
    fetch_wikipedia_extract,
    generate_bundle,
    is_reader_friendly,
    label_paragraphs,
    normalize_for_matching,
    normalize_paragraph_breaks,
    passage_id,
    strip_bare_headings,
    trim_to_word_budget,
)
from coach.storage import Store


# --- Readability / trimming heuristics ---

def test_readability_accepts_simple_passage_and_rejects_dense_one():
    simple = " ".join(["The cat sat on the mat."] * 115)  # 690 words: inside the 650-750 target
    ok, stats = is_reader_friendly(simple)
    assert ok
    assert stats["word_count"] == len(simple.split())

    dense = ("Notwithstanding the aforementioned methodological considerations, "
              "epistemological ramifications necessitate interdisciplinary "
              "reconceptualization of institutionalized paradigms. ") * 8
    ok, stats = is_reader_friendly(dense)
    assert not ok
    assert stats["long_word_ratio"] > 0.25


def test_readability_rejects_too_short_and_too_long():
    ok, _ = is_reader_friendly("Too short.")
    assert not ok
    ok, _ = is_reader_friendly("word " * 700)
    assert not ok


def test_readability_caps_at_750_words():
    tokens = ["The", "cat", "sat", "on", "the", "mat", "today", "near", "the", "window."] * 80
    at_cap = " ".join(tokens[:750])
    ok, stats = is_reader_friendly(at_cap)
    assert stats["word_count"] == 750
    assert ok, stats

    over_cap = " ".join(tokens[:751])
    ok2, stats2 = is_reader_friendly(over_cap)
    assert stats2["word_count"] == 751
    assert not ok2


def test_readability_floors_at_650_words():
    tokens = ["The", "cat", "sat", "on", "the", "mat", "today", "near", "the", "window."] * 80

    just_under = " ".join(tokens[:649])
    ok, stats = is_reader_friendly(just_under)
    assert stats["word_count"] == 649
    assert not ok, "a passage just below the 650-word target should be skipped, not padded"

    at_floor = " ".join(tokens[:650])
    ok2, stats2 = is_reader_friendly(at_floor)
    assert stats2["word_count"] == 650
    assert ok2, stats2


def test_readability_accepts_realistic_encyclopedic_prose():
    # Representative of genuine Wikipedia lead-section prose at the real test's
    # actual length and difficulty: longer sentences and ordinary but 8+
    # character words (civilization, historical, political, instability) that
    # an old, stricter filter rejected almost everywhere, even though this
    # sits squarely within real IELTS Academic Reading's non-specialist
    # academic register — the filter is no longer capped to a single band.
    text = (
        "Ancient Egypt was a civilization of ancient North Africa, concentrated along the "
        "lower reaches of the Nile River, situated in the place that is now the country "
        "Egypt. Ancient Egyptian civilization followed prehistoric Egypt and coalesced "
        "around 3100 BC, with the political unification of Upper and Lower Egypt under "
        "Menes. The history of ancient Egypt occurred as a series of stable kingdoms, "
        "separated by periods of relative instability known as Intermediate Periods. "
        "Egypt reached the pinnacle of its power during the New Kingdom, ruling much of "
        "Nubia and a sizable portion of the Levant, after which it entered a period of "
        "slow decline. Egypt was invaded or conquered by a succession of foreign powers "
        "in this late period. The pharaonic rule officially ended in 31 BC when the "
        "early Roman Empire conquered Egypt and made it a province. The civilization "
        "of ancient Egypt was noted for its considerable achievements, including the "
        "quarrying, surveying, and construction techniques that built monumental "
        "pyramids, temples, and obelisks. It also featured a system of mathematics and "
        "an early form of medicine. Ancient Egyptian religion involved a large pantheon "
        "of gods and goddesses, and elaborate burial customs were developed to prepare "
        "the dead for the afterlife, most notably through the practice of mummification. "
        "The Egyptians also developed one of the earliest writing systems, known as "
        "hieroglyphs, which was used to record religious texts, official decrees, and "
        "everyday transactions across the kingdom for thousands of years. Trade along "
        "the Nile connected Egypt with neighboring regions, allowing goods such as gold, "
        "linen, papyrus, and grain to be exchanged for timber, incense, and other "
        "valuable resources from distant lands."
        " Egyptian society was organized in a strict hierarchy, with the pharaoh at the top, "
        "believed to rule as a living link between the gods and ordinary people. Below the "
        "pharaoh were priests, officials, scribes, and skilled craftsmen, while the majority "
        "of the population worked the land as farmers, growing wheat, barley, and flax along "
        "the fertile floodplain created by the Nile's annual flood. Scribes held a respected "
        "position because so few people could read or write, and their training in hieroglyphic "
        "and later hieratic script could take years to complete. The largest and most famous "
        "monuments, the pyramids at Giza, were built as tombs for pharaohs during the Old "
        "Kingdom, with the Great Pyramid remaining the tallest human-made structure in the "
        "world for more than three thousand years. Nearby stands the Great Sphinx, a massive "
        "limestone statue with the body of a lion and the head of a pharaoh, whose original "
        "purpose is still debated by archaeologists. For centuries, the meaning of Egyptian "
        "hieroglyphs remained a mystery to the outside world, until the discovery of the "
        "Rosetta Stone in 1799 gave scholars a way to compare the same text written in "
        "hieroglyphic, demotic, and Greek script, eventually allowing the writing system to "
        "be deciphered. Ancient Egyptian astronomers also devised one of the earliest solar "
        "calendars, dividing the year into twelve months and using the seasonal flooding of "
        "the Nile to mark the passage of time. The civilization's art, architecture, and "
        "religious ideas continued to influence later cultures around the Mediterranean long "
        "after its political independence came to an end, and its monuments remain among the "
        "most visited archaeological sites in the world today."
        " Modern archaeological research continues to reshape scholarly understanding of "
        "ancient Egypt, as excavations along the Nile Valley regularly uncover new tombs, "
        "settlements, and administrative records that refine the timeline established by "
        "earlier generations of Egyptologists. Advances in scientific analysis, including "
        "isotope studies of mummified remains and radiocarbon dating of organic material, "
        "have allowed researchers to reconstruct details of diet, migration, and disease "
        "that textual sources alone could never reveal. The administration of the ancient "
        "Egyptian state relied heavily on a class of professional scribes and officials who "
        "recorded harvests, taxes, and legal disputes, producing an archive of papyrus "
        "documents that still informs historians about daily economic life along the "
        "floodplain. Large-scale irrigation and land-surveying projects, organized under "
        "royal authority, allowed successive dynasties to predict and manage the Nile's "
        "annual flood, which was essential to sustaining the agricultural surplus that "
        "funded monumental construction and supported a growing urban population. Tourism "
        "linked to these ancient sites remains a significant part of the modern Egyptian "
        "economy, drawing millions of visitors each year to temples, tombs, and museums "
        "that preserve artifacts spanning more than three thousand years of continuous "
        "civilization."
    )
    ok, stats = is_reader_friendly(text)
    assert ok, stats
    assert 650 <= stats["word_count"] <= 750
    assert stats["avg_sentence_len"] > 22


def test_trim_to_word_budget_keeps_whole_sentences():
    text = "One two three. Four five six. Seven eight nine. Ten eleven twelve."
    trimmed = trim_to_word_budget(text, max_words=7)
    assert trimmed == "One two three. Four five six."
    assert trim_to_word_budget(text, max_words=100) == text


def test_trim_to_word_budget_keeps_at_least_one_sentence_even_if_over_budget():
    text = "This sentence alone has more than three words in it."
    assert trim_to_word_budget(text, max_words=3) == text


def test_trim_to_word_budget_preserves_paragraph_breaks_when_everything_fits():
    # Wikipedia's plain-text extracts separate paragraphs with a blank line;
    # when the whole passage fits the budget, those breaks must survive
    # untouched rather than being collapsed into one block of text.
    text = "Para one, sentence one. Para one, sentence two.\n\nPara two, sentence one."
    assert trim_to_word_budget(text, max_words=100) == text


def test_trim_to_word_budget_preserves_paragraph_breaks_when_trimming_mid_passage():
    para1 = "Sentence one here. Sentence two here. Sentence three here."  # 9 words
    para2 = "Sentence four here. Sentence five here. Sentence six here."  # 9 words
    para3 = "Sentence seven here. Sentence eight here."  # 6 words
    text = f"{para1}\n\n{para2}\n\n{para3}"

    trimmed = trim_to_word_budget(text, max_words=15)

    # The first paragraph fits whole; the second is cut down to as many whole
    # sentences as remain in the budget; the third is dropped entirely. The
    # paragraph break between the two kept paragraphs must still be there.
    assert trimmed == "Sentence one here. Sentence two here. Sentence three here.\n\n" \
                       "Sentence four here. Sentence five here."
    assert "\n\n" in trimmed
    assert "Sentence seven" not in trimmed
    assert len(trimmed.split()) <= 15


def test_trim_to_word_budget_stops_at_a_paragraph_boundary_when_budget_is_exactly_used():
    para1 = "Sentence one here. Sentence two here. Sentence three here."  # 9 words
    para2 = "Sentence four here. Sentence five here. Sentence six here."  # 9 words
    text = f"{para1}\n\n{para2}"

    trimmed = trim_to_word_budget(text, max_words=9)

    # The budget is fully spent by the first paragraph alone, so the second
    # paragraph should be dropped entirely rather than forcing in an
    # overflowing extra sentence.
    assert trimmed == para1


def test_trim_to_word_budget_trims_within_an_oversized_first_paragraph():
    para1 = "Sentence one here. Sentence two here."  # 6 words
    para2 = "Sentence three here."  # 3 words
    text = f"{para1}\n\n{para2}"

    trimmed = trim_to_word_budget(text, max_words=2)

    # Even a budget smaller than a single sentence still keeps that first
    # whole sentence (never returns an empty result), and drops the rest.
    assert trimmed == "Sentence one here."


def test_passage_id_is_deterministic_and_topic_sensitive():
    a = passage_id("Animals", "Elephant")
    b = passage_id("Animals", "Elephant")
    c = passage_id("Space", "Elephant")
    assert a == b
    assert a != c


def test_reading_topics_is_a_flat_list_of_topic_names():
    # Candidate article titles are now discovered live per topic (see
    # candidate_titles_for_topic); READING_TOPICS itself is just the fixed
    # set of topic categories offered in the UI dropdown.
    assert isinstance(READING_TOPICS, list)
    assert "Sports" in READING_TOPICS
    assert len(READING_TOPICS) == len(set(READING_TOPICS))


def test_candidate_titles_for_topic_shuffles_and_caps():
    # Candidates now come from a live Wikipedia search rather than a fixed
    # per-topic list, so this drives that search through a mocked session.
    session = _search_session([
        {"title": t} for t in ["Elephant", "Dolphin", "Gray wolf", "Koala", "Octopus"]])
    titles, reason = candidate_titles_for_topic("Animals", count=3, rng=random.Random(1), session=session)
    assert len(titles) == 3
    assert reason is None
    assert set(titles) <= {"Elephant", "Dolphin", "Gray wolf", "Koala", "Octopus"}
    kwargs = session.get.call_args_list[0].kwargs
    assert kwargs["params"]["list"] == "search"
    assert kwargs["params"]["srsearch"] == "Animals"


def test_candidate_titles_for_topic_returns_no_titles_when_search_finds_nothing():
    session = Mock()
    session.get.return_value = _resp(json_data={"query": {"search": []}})
    titles, reason = candidate_titles_for_topic("Unknown topic", count=5, session=session)
    assert titles == []
    assert reason is None


def test_candidate_titles_for_topic_reports_a_network_failure():
    import requests
    session = Mock()
    session.get.side_effect = requests.RequestException("boom")
    titles, reason = candidate_titles_for_topic("Animals", count=5, session=session)
    assert titles == []
    assert reason == "network"


def test_candidate_titles_for_topic_drops_hits_reported_too_short_without_fetching_them():
    # A hit's own whole-page word count is a free upper bound on its lead
    # section's length (the part this app actually uses), so a hit already
    # reported well under the floor there is dropped before ever being
    # fetched — sparing the slow network round trip for a page that was
    # always going to fail the real readability check anyway.
    session = _search_session([
        {"title": "Elephant", "wordcount": 900},
        {"title": "Tiny Stub", "wordcount": 40},
        {"title": "No Count Reported"},  # missing wordcount: kept, not ruled out
    ])
    titles, reason = candidate_titles_for_topic("Animals", count=10, session=session)
    assert reason is None
    assert set(titles) == {"Elephant", "No Count Reported"}
    assert "Tiny Stub" not in titles


def test_candidate_titles_for_topic_uses_min_passage_words_as_the_default_cutoff():
    session = _search_session([
        {"title": "JustUnder", "wordcount": MIN_PASSAGE_WORDS - 1},
        {"title": "JustAtFloor", "wordcount": MIN_PASSAGE_WORDS},
    ])
    titles, _ = candidate_titles_for_topic("Animals", count=10, session=session)
    assert titles == ["JustAtFloor"]


def test_mentions_inappropriate_topic_matches_case_insensitively_across_strings():
    from coach.reading import _mentions_inappropriate_topic
    assert _mentions_inappropriate_topic("Non-reproductive SEXUAL BEHAVIOR in animals")
    assert _mentions_inappropriate_topic("a fine title", "but this text mentions Genital Stimulation")
    assert not _mentions_inappropriate_topic("Elephant", "Elephants are large mammals.")


def test_candidate_titles_for_topic_drops_a_title_flagged_as_inappropriate():
    session = _search_session([
        {"title": "Elephant", "wordcount": 900},
        {"title": "Non-reproductive sexual behavior in animals", "wordcount": 900},
    ])
    titles, reason = candidate_titles_for_topic("Animals", count=10, session=session)
    assert reason is None
    assert titles == ["Elephant"]


def test_description_is_substantive_rejects_noise_and_missing_descriptions():
    from coach.reading import _description_is_substantive
    for good in ["American basketball player", "Team sport", "Film director and producer",
                 "Large mammal", "Ancient Greek athletic festival"]:
        assert _description_is_substantive(good), good
    for bad in [None, "", "   ", "Wikimedia disambiguation page", "Wikimedia list article",
                "American sports television network", "Sports brand", "Sports equipment company",
                "1994 film", "Sports jargon term", "American sports magazine",
                "Video game series", "Radio station in Chicago"]:
        assert not _description_is_substantive(bad), bad


def test_candidate_titles_for_topic_drops_noise_by_short_description():
    # The point of the description screen: a literal title match on the topic
    # word ("NBC Sports", "Pitch (sports)") must not outrank real subject pages.
    session = _search_session(
        [{"title": t, "wordcount": 900} for t in
         ["Tennis", "NBC Sports", "Pitch (sports)", "Serena Williams", "List of sports", "No Desc"]],
        short_descriptions={
            "Tennis": "Racket sport",
            "NBC Sports": "American sports television network",
            "Pitch (sports)": "Sports jargon term",
            "Serena Williams": "American tennis player (born 1981)",
            "List of sports": "Wikimedia list article",
            "No Desc": None,
        })
    titles, reason = candidate_titles_for_topic("Sports", count=10, rng=random.Random(2), session=session)
    assert reason is None
    assert sorted(titles) == ["Serena Williams", "Tennis"]


def test_candidate_titles_for_topic_batches_description_lookups_in_fifties():
    hits = [{"title": f"Topic {i}", "wordcount": 900} for i in range(120)]
    session = _search_session(hits)
    titles, reason = candidate_titles_for_topic("Animals", count=150, session=session)
    assert reason is None
    assert len(titles) == 120
    lookups = [c.kwargs["params"] for c in session.get.call_args_list
               if c.kwargs["params"].get("prop") == "pageprops"]
    assert [len(p["titles"].split("|")) for p in lookups] == [50, 50, 20]
    assert all(p["ppprop"] == "wikibase-shortdesc" for p in lookups)


def test_candidate_titles_for_topic_skips_the_description_lookup_when_nothing_survives():
    session = _search_session([{"title": "Tiny Stub", "wordcount": 40}])
    titles, reason = candidate_titles_for_topic("Animals", count=10, session=session)
    assert titles == [] and reason is None
    assert len(session.get.call_args_list) == 1  # only the search call


def test_candidate_titles_for_topic_reports_a_failed_description_lookup():
    import requests

    session = Mock()

    def get_side_effect(url, **kwargs):
        params = kwargs.get("params") or {}
        if params.get("list") == "search":
            return _resp(json_data={"query": {"search": [{"title": "Elephant", "wordcount": 900}]}})
        raise requests.RequestException("boom")

    session.get.side_effect = get_side_effect
    titles, reason = candidate_titles_for_topic("Animals", count=10, session=session)
    assert titles == []
    assert reason == "network"


def test_candidate_titles_for_topic_reports_a_bad_description_response():
    session = Mock()

    def get_side_effect(url, **kwargs):
        params = kwargs.get("params") or {}
        if params.get("list") == "search":
            return _resp(json_data={"query": {"search": [{"title": "Elephant", "wordcount": 900}]}})
        return _resp(status_code=500)

    session.get.side_effect = get_side_effect
    titles, reason = candidate_titles_for_topic("Animals", count=10, session=session)
    assert titles == []
    assert reason == "bad_response"


def test_wikipedia_short_descriptions_accepts_pages_as_a_dict():
    from coach.reading import _wikipedia_short_descriptions
    session = Mock()
    session.get.return_value = _resp(json_data={"query": {"pages": {
        "1": {"title": "Tennis", "pageprops": {"wikibase-shortdesc": "Racket sport"}},
        "2": {"title": "Bare"},
    }}})
    descriptions, reason = _wikipedia_short_descriptions(["Tennis", "Bare"], session=session)
    assert reason is None
    assert descriptions == {"Tennis": "Racket sport"}


def test_fetch_passages_skips_a_candidate_whose_fetched_title_is_flagged():
    # A search hit's own title check (candidate_titles_for_topic) can miss a
    # redirect that lands on a differently-titled, unsuitable page; this is
    # the same check applied again to the page actually fetched.
    session = _fake_session(search_titles=["Animal Sex"], extract_pages=[_wiki_page(
        title="Non-reproductive sexual behavior in animals", extract=SIMPLE_ELEPHANT_TEXT)])
    client = mock_bundle_client()
    skipped = []
    results = fetch_passages("Animals", count=1, api_key="k", session=session, client=client,
                              rng=random.Random(1), on_skip=lambda title, reason: skipped.append((title, reason)))
    assert results == []
    assert skipped == [("Animal Sex", "not appropriate for this app")]
    client.responses.parse.assert_not_called()


def test_fetch_passages_skips_a_candidate_whose_passage_text_is_flagged():
    # An innocuous title whose actual shown text still isn't appropriate —
    # for instance an otherwise-fine article whose body, now that the
    # extract can run past the lead, drifts into unsuitable territory
    # within the part that's actually trimmed to and shown.
    flagged_text = SIMPLE_ELEPHANT_TEXT + " Male elephants display graphic violence and torture toward rivals."
    session = _fake_session(search_titles=["Elephant"], extract_pages=[_wiki_page(
        title="Elephant", extract=flagged_text)])
    client = mock_bundle_client()
    skipped = []
    results = fetch_passages("Animals", count=1, api_key="k", session=session, client=client,
                              rng=random.Random(1), on_skip=lambda title, reason: skipped.append((title, reason)))
    assert results == []
    assert skipped == [("Elephant", "not appropriate for this app")]
    client.responses.parse.assert_not_called()


def test_fetch_passages_does_not_reject_flagged_content_that_gets_trimmed_away():
    # The check runs on the trimmed text actually shown to the learner, not
    # the whole raw extract: content past this app's own word budget never
    # reaches them, so a candidate isn't rejected over words nobody sees.
    # SIMPLE_ELEPHANT_TEXT (678 words) is kept intact so the mocked bundle's
    # hardcoded source_quote/answer_text values still verify against it;
    # padding pushes the flagged sentence in a later paragraph past the
    # 750-word budget, where trim_to_word_budget's own paragraph-at-a-time
    # trimming stops before ever reaching it.
    padding = " ".join(["This sentence is neutral filler text about nothing in particular."] * 60)
    text_with_late_flag = (SIMPLE_ELEPHANT_TEXT + "\n\n" + padding + "\n\n"
                            "This part is never reached: graphic violence and torture.")
    session = _fake_session(search_titles=["Elephant"], extract_pages=[_wiki_page(
        title="Elephant", extract=text_with_late_flag)])
    client = mock_bundle_client()
    results = fetch_passages("Animals", count=1, api_key="k", session=session, client=client,
                              rng=random.Random(1))
    assert len(results) == 1
    assert "graphic violence" not in results[0]["body"]


def test_fetch_passages_never_fetches_a_candidate_reported_too_short_by_search():
    session = Mock()

    def get_side_effect(url, **kwargs):
        params = kwargs.get("params") or {}
        if params.get("list") == "search":
            return _resp(json_data={"query": {"search": [
                {"title": "Elephant", "wordcount": 900},
                {"title": "Tiny Stub", "wordcount": 40},
            ]}})
        if params.get("prop") == "pageprops":
            # The description lookup must only ever see the survivor.
            assert params["titles"] == "Elephant"
            return _resp(json_data=_shortdesc_json(["Elephant"]))
        assert params.get("titles") != "Tiny Stub"  # never fetched: ruled out for free
        return _resp(json_data=_wiki_json([_wiki_page(title="Elephant", extract=SIMPLE_ELEPHANT_TEXT)]))

    session.get.side_effect = get_side_effect
    client = mock_bundle_client()
    results = fetch_passages("Animals", count=1, api_key="k", session=session, client=client,
                              rng=random.Random(1))
    assert len(results) == 1


# --- Wikipedia fetch ---

def _resp(status_code=200, json_data=None, raise_exc=None, content=None, headers=None):
    resp = Mock()
    resp.status_code = status_code
    if raise_exc:
        resp.json.side_effect = raise_exc
    else:
        resp.json.return_value = json_data or {}
    resp.content = content if content is not None else b""
    resp.headers = headers or {}
    return resp


def _wiki_page(title="Elephant", extract="Elephants are large mammals.", thumbnail=None,
               fullurl=None, missing=False, disambiguation=False, ns=0):
    """Build one page entry in the shape the MediaWiki Action API returns
    (formatversion=2) for `prop=extracts|pageimages|pageprops|info`."""
    page = {"title": title, "ns": ns}
    if missing:
        page["missing"] = True
        return page
    page["extract"] = extract
    if thumbnail:
        page["thumbnail"] = {"source": thumbnail}
    if fullurl:
        page["fullurl"] = fullurl
    if disambiguation:
        page["pageprops"] = {"disambiguation": ""}
    return page


def _wiki_json(pages=None):
    """Build a full Action API query response wrapping the given pages."""
    return {"query": {"pages": pages if pages is not None else []}}


def _shortdesc_json(titles, short_descriptions=None, default="Large mammal"):
    """A `prop=pageprops&ppprop=wikibase-shortdesc` response for `titles`.
    A title maps to its entry in `short_descriptions` if present (None means
    the page has no short description); otherwise it gets `default`.
    """
    short_descriptions = short_descriptions or {}
    pages = []
    for t in titles:
        desc = short_descriptions.get(t, default)
        page = {"title": t}
        if desc is not None:
            page["pageprops"] = {"wikibase-shortdesc": desc}
        pages.append(page)
    return _wiki_json(pages)


def _search_session(hits, short_descriptions=None, default="Large mammal"):
    """Mock `session` answering candidate_titles_for_topic's two calls: the
    `list=search` call with `hits` (dicts with a title and optional
    wordcount) and the short-description lookup for the requested titles.
    """
    session = Mock()

    def get_side_effect(url, **kwargs):
        params = kwargs.get("params") or {}
        if params.get("list") == "search":
            return _resp(json_data={"query": {"search": hits}})
        titles = params["titles"].split("|")
        return _resp(json_data=_shortdesc_json(titles, short_descriptions, default))

    session.get.side_effect = get_side_effect
    return session


def _fake_session(search_titles, extract_pages, image_resp=None, short_descriptions=None):
    """Build a Mock `session` for fetch_passages tests. A fetch now makes
    two different kinds of call against the same api.php endpoint — a
    `list=search` topic-discovery call (candidate_titles_for_topic) and a
    `prop=extracts` per-title page fetch (_fetch_wikipedia_page) — plus,
    sometimes, a separate plain image download, so `.get()` must answer
    each by inspecting the request instead of returning one fixed response
    for every call the way a single-endpoint mock could.
    """
    session = Mock()

    def get_side_effect(url, **kwargs):
        params = kwargs.get("params") or {}
        if params.get("list") == "search":
            return _resp(json_data={"query": {"search": [{"title": t} for t in search_titles]}})
        if params.get("prop") == "pageprops":
            return _resp(json_data=_shortdesc_json(params["titles"].split("|"), short_descriptions))
        if "api.php" in url:
            return _resp(json_data=_wiki_json(extract_pages))
        return image_resp if image_resp is not None else _resp(content=b"", headers={})

    session.get.side_effect = get_side_effect
    return session


def test_fetch_wikipedia_extract_success():
    session = Mock()
    session.get.return_value = _resp(json_data=_wiki_json([_wiki_page(
        title="Elephant", extract="Elephants are large mammals.",
        thumbnail="https://upload.wikimedia.org/elephant-thumb.jpg",
        fullurl="https://en.wikipedia.org/wiki/Elephant",
    )]))
    page = fetch_wikipedia_extract("Elephant", session=session)
    assert page["title"] == "Elephant"
    assert page["text"] == "Elephants are large mammals."
    assert page["source_name"] == "Wikipedia"
    assert page["source_url"] == "https://en.wikipedia.org/wiki/Elephant"
    assert page["license"]
    assert page["image_url"] == "https://upload.wikimedia.org/elephant-thumb.jpg"
    kwargs = session.get.call_args.kwargs
    assert kwargs["params"]["titles"] == "Elephant"
    assert kwargs["params"]["explaintext"] == 1
    # Not limited to the lead section: most leads alone are well under this
    # app's 650-750 word target, so the extract can run into the body too.
    assert "exintro" not in kwargs["params"]


def test_fetch_wikipedia_extract_normalizes_single_newline_paragraph_breaks():
    # This is the real shape of a MediaWiki Action API extract: paragraphs
    # are separated by ONE newline, not a blank line. Left as-is, that reads
    # as a soft break in Markdown (a single space), not a new paragraph, so
    # every fetched passage rendered as one run-on block of text.
    raw_extract = (
        "Paragraph one about the topic, with a couple of sentences in it.\n"
        "Paragraph two continues with more detail on a related point."
    )
    session = Mock()
    session.get.return_value = _resp(json_data=_wiki_json([_wiki_page(
        title="Elephant", extract=raw_extract)]))
    page = fetch_wikipedia_extract("Elephant", session=session)
    assert "\n\n" in page["text"]
    assert page["text"] == (
        "Paragraph one about the topic, with a couple of sentences in it.\n\n"
        "Paragraph two continues with more detail on a related point."
    )


def test_normalize_paragraph_breaks_collapses_any_newline_run_to_a_blank_line():
    assert normalize_paragraph_breaks("A.\nB.") == "A.\n\nB."
    assert normalize_paragraph_breaks("A.\n\nB.") == "A.\n\nB."
    assert normalize_paragraph_breaks("A.\n\n\n\nB.") == "A.\n\nB."
    assert normalize_paragraph_breaks("Single paragraph, no breaks.") == "Single paragraph, no breaks."


def test_strip_bare_headings_drops_a_heading_only_paragraph():
    text = ("The introduction paragraph ends with a period.\n\n"
            "History\n\n"
            "This paragraph is real prose about the article's history.")
    stripped = strip_bare_headings(text)
    assert "History\n\n" not in stripped
    assert "The introduction paragraph ends with a period." in stripped
    assert "This paragraph is real prose about the article's history." in stripped


def test_strip_bare_headings_keeps_paragraphs_that_end_in_punctuation():
    text = ('Ends with a period.\n\nEnds with a question mark?\n\nEnds with an exclamation!\n\n'
            'Ends with a quoted sentence."')
    assert strip_bare_headings(text) == text


def test_strip_bare_headings_keeps_a_long_paragraph_even_without_ending_punctuation():
    # A real heading is always short; a long block of text without ending
    # punctuation (garbled source text, for instance) is never mistaken for
    # one just because it also lacks a period, or the whole passage could
    # be wiped out by this heuristic instead of just its true headings.
    long_text = " ".join(["word"] * 500)
    assert strip_bare_headings(long_text) == long_text


def test_strip_bare_headings_drops_multiple_headings_and_a_multi_word_one():
    text = ("Intro sentence here.\n\n"
            "Early life and career\n\n"
            "More prose about the early years.\n\n"
            "Later life\n\n"
            "Even more prose, concluding the article.")
    stripped = strip_bare_headings(text)
    assert "Early life and career" not in stripped
    assert "Later life\n\n" not in stripped
    assert "Intro sentence here." in stripped
    assert "More prose about the early years." in stripped
    assert "Even more prose, concluding the article." in stripped


def test_fetch_wikipedia_extract_image_url_is_none_without_a_picture():
    session = Mock()
    session.get.return_value = _resp(json_data=_wiki_json([
        _wiki_page(title="Elephant", extract="Elephants are large mammals."),
    ]))
    page = fetch_wikipedia_extract("Elephant", session=session)
    assert page["image_url"] is None


def test_fetch_wikipedia_extract_returns_none_on_missing_page():
    session = Mock()
    session.get.return_value = _resp(json_data=_wiki_json([
        _wiki_page(title="Nonexistent Title Xyz", missing=True),
    ]))
    assert fetch_wikipedia_extract("Nonexistent Title Xyz", session=session) is None


def test_fetch_wikipedia_extract_returns_none_for_non_article_namespace():
    session = Mock()
    session.get.return_value = _resp(json_data=_wiki_json([
        _wiki_page(title="Talk:Elephant", extract="Discussion about the Elephant article.", ns=1),
    ]))
    assert fetch_wikipedia_extract("Talk:Elephant", session=session) is None


def test_fetch_wikipedia_extract_returns_none_when_no_pages_in_response():
    session = Mock()
    session.get.return_value = _resp(json_data=_wiki_json([]))
    assert fetch_wikipedia_extract("X", session=session) is None


def test_fetch_wikipedia_extract_returns_none_on_non_200_status():
    session = Mock()
    session.get.return_value = _resp(status_code=500, json_data={})
    assert fetch_wikipedia_extract("X", session=session) is None


def test_fetch_wikipedia_extract_returns_none_on_disambiguation():
    session = Mock()
    session.get.return_value = _resp(json_data=_wiki_json([
        _wiki_page(title="Mercury", extract="Mercury may refer to:", disambiguation=True),
    ]))
    assert fetch_wikipedia_extract("Mercury", session=session) is None


def test_fetch_wikipedia_extract_returns_none_on_empty_extract():
    session = Mock()
    session.get.return_value = _resp(json_data=_wiki_json([_wiki_page(title="X", extract="  ")]))
    assert fetch_wikipedia_extract("X", session=session) is None


def test_fetch_wikipedia_extract_returns_none_on_network_error():
    import requests
    session = Mock()
    session.get.side_effect = requests.RequestException("boom")
    assert fetch_wikipedia_extract("X", session=session) is None


def test_fetch_wikipedia_extract_returns_none_on_bad_json():
    session = Mock()
    session.get.return_value = _resp(raise_exc=ValueError("not json"))
    assert fetch_wikipedia_extract("X", session=session) is None


# --- _fetch_wikipedia_page's honest failure-reason categorization ---
# fetch_wikipedia_extract (above) collapses every failure to None; these
# cover the internal helper fetch_passages actually calls, which keeps the
# reason so a skipped candidate's message tells the truth instead of
# guessing "missing, moved, or a disambiguation page" for a network error.

def test_fetch_wikipedia_page_categorizes_a_network_error():
    from coach.reading import _fetch_wikipedia_page
    import requests
    session = Mock()
    session.get.side_effect = requests.RequestException("boom")
    page, reason = _fetch_wikipedia_page("X", session=session)
    assert page is None
    assert reason == "network"


def test_fetch_wikipedia_page_categorizes_a_bad_response():
    from coach.reading import _fetch_wikipedia_page
    session = Mock()
    session.get.return_value = _resp(status_code=500, json_data={})
    page, reason = _fetch_wikipedia_page("X", session=session)
    assert page is None
    assert reason == "bad_response"

    session2 = Mock()
    session2.get.return_value = _resp(raise_exc=ValueError("not json"))
    page2, reason2 = _fetch_wikipedia_page("X", session=session2)
    assert page2 is None
    assert reason2 == "bad_response"


def test_fetch_wikipedia_page_categorizes_a_missing_page_as_not_found():
    from coach.reading import _fetch_wikipedia_page
    session = Mock()
    session.get.return_value = _resp(json_data=_wiki_json([
        _wiki_page(title="Nonexistent Title Xyz", missing=True)]))
    page, reason = _fetch_wikipedia_page("Nonexistent Title Xyz", session=session)
    assert page is None
    assert reason == "not_found"


def test_fetch_wikipedia_page_categorizes_a_disambiguation_page():
    from coach.reading import _fetch_wikipedia_page
    session = Mock()
    session.get.return_value = _resp(json_data=_wiki_json([
        _wiki_page(title="Mercury", extract="Mercury may refer to:", disambiguation=True)]))
    page, reason = _fetch_wikipedia_page("Mercury", session=session)
    assert page is None
    assert reason == "disambiguation"


def test_fetch_wikipedia_page_categorizes_an_empty_extract():
    from coach.reading import _fetch_wikipedia_page
    session = Mock()
    session.get.return_value = _resp(json_data=_wiki_json([_wiki_page(title="X", extract="  ")]))
    page, reason = _fetch_wikipedia_page("X", session=session)
    assert page is None
    assert reason == "empty"


def test_fetch_wikipedia_page_returns_no_reason_on_success():
    from coach.reading import _fetch_wikipedia_page
    session = Mock()
    session.get.return_value = _resp(json_data=_wiki_json([_wiki_page(
        title="Elephant", extract="Elephants are large mammals.")]))
    page, reason = _fetch_wikipedia_page("Elephant", session=session)
    assert page is not None
    assert reason is None


def test_fetch_wikipedia_page_is_not_limited_to_the_lead_and_drops_bare_headings():
    # For most real articles the lead alone is well under this app's word
    # target, so the extract is allowed to run into the body — but a bare
    # section-heading line the API includes once it goes past the lead must
    # not show up as if it were a stray sentence in the passage.
    from coach.reading import _fetch_wikipedia_page
    raw_extract = ("The introduction sentence ends with a period.\n"
                    "History\n"
                    "A body paragraph about the article's history, also ending in a period.")
    session = Mock()
    session.get.return_value = _resp(json_data=_wiki_json([_wiki_page(
        title="Elephant", extract=raw_extract)]))
    page, reason = _fetch_wikipedia_page("Elephant", session=session)
    assert reason is None
    assert "History" not in page["text"]
    assert "The introduction sentence ends with a period." in page["text"]
    assert "A body paragraph about the article's history, also ending in a period." in page["text"]
    # No exintro param is sent: the request already asks for the full extract.
    kwargs = session.get.call_args.kwargs
    assert "exintro" not in kwargs["params"]


# --- Picture download (cached locally alongside the passage) ---

def test_fetch_image_bytes_returns_none_for_empty_url():
    assert fetch_image_bytes("", session=Mock()) is None
    assert fetch_image_bytes(None, session=Mock()) is None


def test_fetch_image_bytes_success_uses_content_type_for_extension():
    session = Mock()
    session.get.return_value = _resp(content=b"\xff\xd8fakejpeg", headers={"Content-Type": "image/jpeg; charset=binary"})
    result = fetch_image_bytes("https://upload.wikimedia.org/thumb/elephant.jpg", session=session)
    assert result == (b"\xff\xd8fakejpeg", "jpg")


def test_fetch_image_bytes_falls_back_to_url_extension_when_content_type_missing():
    session = Mock()
    session.get.return_value = _resp(content=b"fakepng", headers={})
    result = fetch_image_bytes("https://upload.wikimedia.org/thumb/elephant.png", session=session)
    assert result == (b"fakepng", "png")


def test_fetch_image_bytes_defaults_to_jpg_when_type_is_unknown():
    session = Mock()
    session.get.return_value = _resp(content=b"fakebytes", headers={})
    result = fetch_image_bytes("https://upload.wikimedia.org/thumb/elephant", session=session)
    assert result == (b"fakebytes", "jpg")


def test_fetch_image_bytes_returns_none_on_non_200():
    session = Mock()
    session.get.return_value = _resp(status_code=404, content=b"", headers={})
    assert fetch_image_bytes("https://upload.wikimedia.org/thumb/missing.jpg", session=session) is None


def test_fetch_image_bytes_returns_none_on_empty_body():
    session = Mock()
    session.get.return_value = _resp(content=b"", headers={"Content-Type": "image/jpeg"})
    assert fetch_image_bytes("https://upload.wikimedia.org/thumb/elephant.jpg", session=session) is None


def test_fetch_image_bytes_returns_none_on_network_error():
    import requests
    session = Mock()
    session.get.side_effect = requests.RequestException("boom")
    assert fetch_image_bytes("https://upload.wikimedia.org/thumb/elephant.jpg", session=session) is None


# --- Pydantic schema validation ---

def _valid_question(**overrides):
    data = dict(prompt="What are elephants known for?",
                options=["Being the largest land animals", "Being the fastest runners",
                         "Living underwater", "Being nocturnal hunters"],
                answer=0, explanation="The passage states elephants are the largest land animals on Earth.",
                source_quote="Elephants are the largest land animals on Earth.")
    data.update(overrides)
    return data


def test_comprehension_question_rejects_duplicate_options():
    with pytest.raises(ValueError):
        ComprehensionQuestion.model_validate(_valid_question(options=["Blue", "blue", "Red", "Purple"]))


def _valid_tf_question(**overrides):
    data = dict(kind="true_false_not_given",
                prompt="Elephants are the largest land animals on Earth.",
                options=list(TRUE_FALSE_NOT_GIVEN_OPTIONS), answer=0,
                explanation="The passage states elephants are the largest land animals.",
                source_quote="Elephants are the largest land animals on Earth.")
    data.update(overrides)
    return data


def _valid_yn_question(**overrides):
    data = dict(kind="yes_no_not_given",
                prompt="The writer believes elephants deserve stronger protection.",
                options=list(YES_NO_NOT_GIVEN_OPTIONS), answer=0,
                explanation="The passage says groups are working to protect them.",
                source_quote="groups around the world are working to protect them")
    data.update(overrides)
    return data


def test_true_false_not_given_question_accepts_a_true_answer_with_a_quote():
    q = ComprehensionQuestion.model_validate(_valid_tf_question(answer=0))
    assert q.kind == "true_false_not_given"
    assert q.options == TRUE_FALSE_NOT_GIVEN_OPTIONS
    assert q.source_quote


def test_true_false_not_given_question_rejects_the_wrong_option_list():
    with pytest.raises(ValueError):
        ComprehensionQuestion.model_validate(_valid_tf_question(options=["Yes", "No", "Not Given"]))


def test_true_false_not_given_question_requires_a_quote_unless_not_given():
    with pytest.raises(ValueError):
        ComprehensionQuestion.model_validate(_valid_tf_question(answer=1, source_quote=None))


def test_true_false_not_given_not_given_answer_must_omit_the_quote():
    with pytest.raises(ValueError):
        # answer=2 is "Not Given", but this still carries the default source_quote.
        ComprehensionQuestion.model_validate(_valid_tf_question(answer=2))
    q = ComprehensionQuestion.model_validate(_valid_tf_question(
        answer=2, source_quote=None, prompt="Elephants can breathe underwater indefinitely."))
    assert q.source_quote is None


def test_yes_no_not_given_question_shares_the_same_validation_rules():
    q = ComprehensionQuestion.model_validate(_valid_yn_question(answer=0))
    assert q.kind == "yes_no_not_given"
    assert q.options == YES_NO_NOT_GIVEN_OPTIONS
    with pytest.raises(ValueError):
        ComprehensionQuestion.model_validate(_valid_yn_question(options=["True", "False", "Not Given"]))


# --- Free-text kinds: sentence_completion, short_answer ---

def _valid_sentence_completion_question(**overrides):
    data = dict(kind="sentence_completion",
                prompt="Elephants can live for about ___ years in the wild.",
                answer_text="sixty to seventy", max_words=3,
                explanation="The passage says elephants can live for about sixty to seventy years "
                            "in the wild.",
                source_quote="live for about sixty to seventy years in the wild")
    data.update(overrides)
    return data


def _valid_short_answer_question(**overrides):
    data = dict(kind="short_answer",
                prompt="What do elephants use to eat, drink, and greet each other?",
                answer_text="their trunks", max_words=2,
                explanation="The passage says elephants use their trunks to eat, drink, and greet "
                            "each other.",
                source_quote="use their trunks to eat, drink, and greet each other")
    data.update(overrides)
    return data


def test_sentence_completion_question_accepts_a_valid_answer():
    q = ComprehensionQuestion.model_validate(_valid_sentence_completion_question())
    assert q.kind == "sentence_completion"
    assert q.answer_text == "sixty to seventy"
    assert q.max_words == 3
    assert q.options == []


def test_sentence_completion_question_rejects_answer_text_longer_than_its_own_max_words():
    with pytest.raises(ValueError):
        ComprehensionQuestion.model_validate(_valid_sentence_completion_question(
            answer_text="sixty to seventy years", max_words=2))


def test_sentence_completion_question_requires_a_nonempty_answer_text():
    with pytest.raises(ValueError):
        ComprehensionQuestion.model_validate(_valid_sentence_completion_question(answer_text="   "))


def test_sentence_completion_question_requires_a_source_quote():
    with pytest.raises(ValueError):
        ComprehensionQuestion.model_validate(_valid_sentence_completion_question(source_quote=None))


def test_sentence_completion_question_requires_max_words_between_one_and_five():
    with pytest.raises(ValueError):
        ComprehensionQuestion.model_validate(_valid_sentence_completion_question(max_words=0))
    with pytest.raises(ValueError):
        ComprehensionQuestion.model_validate(_valid_sentence_completion_question(max_words=6))


def test_short_answer_question_shares_the_same_free_text_validation_rules():
    q = ComprehensionQuestion.model_validate(_valid_short_answer_question())
    assert q.kind == "short_answer"
    with pytest.raises(ValueError):
        ComprehensionQuestion.model_validate(_valid_short_answer_question(answer_text=None))


# --- Matching kinds: sentence endings, features, information, headings ---
# These reuse the same options/answer-index shape as multiple_choice, so they
# only need to prove the free-form (not-exactly-four) option-count rules.

def _valid_matching_sentence_endings_question(**overrides):
    data = dict(kind="matching_sentence_endings",
                prompt="Elephants use their trunks to...",
                options=["eat, drink, and greet each other.", "fly over long distances.",
                         "build nests high in the trees.", "hide from predators underground."],
                answer=0,
                explanation="The passage says elephants use their trunks to eat, drink, and greet "
                            "each other.",
                source_quote="use their trunks to eat, drink, and greet each other")
    data.update(overrides)
    return data


def _valid_matching_features_question(**overrides):
    data = dict(kind="matching_features",
                prompt="Which elephant species has a single lobe at the tip of its trunk?",
                options=["The African bush elephant", "The African forest elephant",
                         "The Asian elephant", "The woolly mammoth"],
                answer=2,
                explanation="The passage says the Asian elephant has a single lobe at the tip of "
                            "its trunk.",
                source_quote="a single lobe at the tip of its trunk")
    data.update(overrides)
    return data


def _valid_matching_information_question(**overrides):
    data = dict(kind="matching_information",
                prompt="Which paragraph explains how elephants stay cool?",
                options=["Paragraph A", "Paragraph B", "Paragraph C"],
                answer=1,
                explanation="Paragraph B explains that large ears help elephants release heat and "
                            "stay cool.",
                source_quote="large ears also help them stay cool")
    data.update(overrides)
    return data


def _valid_matching_headings_question(**overrides):
    data = dict(kind="matching_headings",
                prompt="Choose the best heading for the paragraph about elephant communication.",
                options=["Family life and leadership", "How elephants communicate",
                         "Physical characteristics", "Conservation challenges"],
                answer=1,
                explanation="This paragraph is about how elephants communicate using sounds.",
                source_quote="communicate using a wide range of sounds")
    data.update(overrides)
    return data


def test_matching_sentence_endings_question_accepts_a_valid_answer():
    q = ComprehensionQuestion.model_validate(_valid_matching_sentence_endings_question())
    assert q.kind == "matching_sentence_endings"
    assert q.options[q.answer] == "eat, drink, and greet each other."


def test_matching_sentence_endings_question_requires_at_least_three_options():
    with pytest.raises(ValueError):
        ComprehensionQuestion.model_validate(
            _valid_matching_sentence_endings_question(options=["a.", "b."], answer=0))


def test_matching_sentence_endings_question_rejects_duplicate_options():
    with pytest.raises(ValueError):
        ComprehensionQuestion.model_validate(_valid_matching_sentence_endings_question(
            options=["same.", "same.", "different."], answer=0))


def test_matching_features_question_requires_a_source_quote():
    with pytest.raises(ValueError):
        ComprehensionQuestion.model_validate(_valid_matching_features_question(source_quote=None))


def test_matching_information_question_accepts_a_paragraph_reference():
    q = ComprehensionQuestion.model_validate(_valid_matching_information_question())
    assert q.kind == "matching_information"
    assert q.options[q.answer] == "Paragraph B"


def test_matching_headings_question_accepts_a_valid_heading_choice():
    q = ComprehensionQuestion.model_validate(_valid_matching_headings_question())
    assert q.kind == "matching_headings"


def _full_comprehension_mix():
    """Eleven items spanning every implemented question kind, used
    everywhere a test needs a bundle that satisfies PassageBundle's min_length
    of 6. All source_quote/answer_text values here are real substrings of
    SIMPLE_ELEPHANT_TEXT so tests that route this mix through fetch_passages
    (which re-verifies every quote against the actual passage) still pass.
    """
    return [
        _valid_question(),
        _valid_question(prompt="A second distinct multiple-choice question here?"),
        _valid_tf_question(answer=0),
        _valid_tf_question(answer=2, source_quote=None,
                            prompt="Elephants can breathe underwater indefinitely."),
        _valid_yn_question(answer=0),
        _valid_sentence_completion_question(),
        _valid_short_answer_question(),
        _valid_matching_sentence_endings_question(),
        _valid_matching_features_question(),
        _valid_matching_information_question(),
        _valid_matching_headings_question(),
    ]


def test_matching_headings_across_a_bundle_must_share_the_same_heading_pool():
    first = _valid_matching_headings_question()
    second = _valid_matching_headings_question(
        prompt="Choose the best heading for the paragraph about elephant diet.",
        options=["A different pool of headings", "Another one", "A third", "A fourth"], answer=0)
    with pytest.raises(ValueError):
        PassageBundle.model_validate(dict(
            comprehension=_full_comprehension_mix()[:-1] + [first, second],
            vocabulary=[],
        ))


def test_normalize_for_matching_ignores_case_punctuation_and_extra_spaces():
    assert normalize_for_matching("  Sixty To Seventy!  ") == normalize_for_matching("sixty to seventy")
    assert normalize_for_matching("A, B, C.") == "a b c"


def test_label_paragraphs_splits_and_labels_in_order():
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    labeled = label_paragraphs(text)
    assert [label for label, _ in labeled] == ["A", "B", "C"]
    assert labeled[1][1] == "Second paragraph."


def test_free_text_and_paragraph_kind_sets_are_correct():
    assert FREE_TEXT_KINDS == {"sentence_completion", "short_answer"}
    assert PARAGRAPH_KINDS == {"matching_information", "matching_headings"}


def _valid_vocab(**overrides):
    data = dict(word="massive", pos="adjective", definition="very large", vi="to lon",
                example="The massive elephant walked slowly.", collocation="a massive size",
                cloze="The ___ elephant walked slowly.", distractors="tiny;quick;quiet")
    data.update(overrides)
    return data


def test_vocab_candidate_requires_word_in_example():
    with pytest.raises(ValueError):
        VocabCandidate.model_validate(_valid_vocab(example="The elephant walked slowly."))


def test_vocab_candidate_requires_exactly_one_cloze_blank():
    with pytest.raises(ValueError):
        VocabCandidate.model_validate(_valid_vocab(cloze="The ___ elephant ___ slowly."))


def test_vocab_candidate_requires_three_distinct_distractors():
    with pytest.raises(ValueError):
        VocabCandidate.model_validate(_valid_vocab(distractors="tiny;tiny;quiet"))
    with pytest.raises(ValueError):
        VocabCandidate.model_validate(_valid_vocab(distractors="massive;quick;quiet"))


def test_passage_bundle_rejects_too_few_comprehension_items():
    with pytest.raises(ValueError):
        PassageBundle.model_validate(dict(
            comprehension=[_valid_question(), _valid_question(prompt="A different, valid question here?")],
            vocabulary=[],
        ))


def test_passage_bundle_valid_round_trip():
    bundle = PassageBundle.model_validate(dict(
        comprehension=_full_comprehension_mix(),
        vocabulary=[_valid_vocab()],
    ))
    assert len(bundle.comprehension) == 11
    assert len(bundle.vocabulary) == 1


# --- generate_bundle / fetch_passages with a mocked OpenAI client ---

def mock_bundle_client(vocabulary=None, comprehension=None):
    parsed = PassageBundle.model_validate(dict(
        comprehension=comprehension if comprehension is not None else _full_comprehension_mix(),
        vocabulary=vocabulary if vocabulary is not None else [_valid_vocab()],
    ))
    client = Mock()
    client.responses.parse.return_value = SimpleNamespace(status="completed", output_parsed=parsed)
    return client


def test_generate_bundle_requires_api_key():
    with pytest.raises(ValueError, match="API key"):
        generate_bundle("Some passage text.", "", client=Mock())


def test_generate_bundle_raises_on_incomplete_response():
    client = Mock()
    client.responses.parse.return_value = SimpleNamespace(status="incomplete", output_parsed=None)
    with pytest.raises(ValueError, match="complete"):
        generate_bundle("Some passage text.", "test-key", client=client)


def test_generate_bundle_returns_parsed_bundle_and_does_not_leak_key():
    client = mock_bundle_client()
    # generate_bundle now also grounds every citation against the real
    # passage text (see _drop_ungrounded_items), so this needs a passage
    # that the mocked bundle's quotes/answers actually match — SIMPLE_ELEPHANT_TEXT,
    # per _full_comprehension_mix's own docstring.
    bundle = generate_bundle(SIMPLE_ELEPHANT_TEXT, "super-secret-key", client=client)
    assert isinstance(bundle, PassageBundle)
    kwargs = client.responses.parse.call_args.kwargs
    assert "super-secret-key" not in kwargs["input"]
    assert kwargs["store"] is False


# --- generate_bundle salvages a bundle that has one or two bad items ---
# generate_bundle asks OpenAI to parse into a lenient variant of PassageBundle
# (no per-item cross-field validation), so one malformed item doesn't blow up
# the whole structured-output parse; it then re-validates each item on its
# own and keeps only the ones that pass, rather than throwing away an
# otherwise-good ~11-question bundle over a single bad item.

def _lenient_bundle(comprehension, vocabulary=None):
    from coach.reading import _LenientPassageBundle
    return _LenientPassageBundle.model_validate(dict(
        comprehension=comprehension,
        vocabulary=vocabulary if vocabulary is not None else [],
    ))


def test_generate_bundle_salvages_a_single_bad_comprehension_item():
    # Mirrors a real failure: an otherwise-good 11-item bundle where one
    # short_answer's answer_text is longer than its own max_words. That used
    # to sink the whole bundle — one bad item discarding ten good ones via a
    # pydantic ValidationError raised while parsing the API response — now
    # only the bad item itself is dropped.
    # Uses SIMPLE_ELEPHANT_TEXT as the passage: generate_bundle now also
    # grounds every citation against the real passage text (see
    # _drop_ungrounded_items), and _full_comprehension_mix's quotes/answers
    # are only real substrings of that text.
    good_items = _full_comprehension_mix()
    broken = _valid_short_answer_question(answer_text="their two trunks and mouth", max_words=2)
    parsed = _lenient_bundle(good_items[:-1] + [broken], vocabulary=[_valid_vocab()])
    client = Mock()
    client.responses.parse.return_value = SimpleNamespace(status="completed", output_parsed=parsed)
    bundle = generate_bundle(SIMPLE_ELEPHANT_TEXT, "k", client=client)
    assert len(bundle.comprehension) == len(good_items) - 1
    assert len(bundle.vocabulary) == 1
    assert all(q.kind != "short_answer" or q.answer_text != "their two trunks and mouth"
               for q in bundle.comprehension)


def test_generate_bundle_salvages_a_single_bad_vocabulary_item():
    good_items = _full_comprehension_mix()
    broken_vocab = _valid_vocab(word="trunk", example="The elephant's trunk is very strong.",
                                 cloze="The elephant's ___ is very ___ strong.")
    good_vocab = _valid_vocab(word="tusk", example="The tusk was made of ivory.")
    parsed = _lenient_bundle(good_items, vocabulary=[good_vocab, broken_vocab])
    client = Mock()
    client.responses.parse.return_value = SimpleNamespace(status="completed", output_parsed=parsed)
    bundle = generate_bundle(SIMPLE_ELEPHANT_TEXT, "k", client=client)
    assert len(bundle.comprehension) == len(good_items)
    assert len(bundle.vocabulary) == 1
    assert bundle.vocabulary[0].word == "tusk"


def test_generate_bundle_still_fails_if_salvage_drops_below_minimum():
    # Salvage only helps when enough good items remain afterward; if
    # dropping the bad ones would leave fewer than PassageBundle's own
    # 6-item minimum, this must still fail exactly as a normal validation
    # failure would, rather than silently returning a too-short bundle.
    good_items = _full_comprehension_mix()[:6]  # already exactly at the minimum
    broken = _valid_short_answer_question(answer_text="their two trunks and mouth", max_words=2)
    parsed = _lenient_bundle(good_items[:-1] + [broken])
    client = Mock()
    client.responses.parse.return_value = SimpleNamespace(status="completed", output_parsed=parsed)
    with pytest.raises(ValueError):
        generate_bundle("Some passage text.", "k", client=client)


# --- generate_bundle also salvages an item whose citation doesn't hold up
# against the real passage text (_drop_ungrounded_items), the same way it
# salvages a structurally-bad item: only that one item is dropped instead of
# discarding an otherwise-good bundle.

def test_generate_bundle_drops_a_comprehension_item_with_an_unverifiable_quote():
    good_items = _full_comprehension_mix()
    unverifiable = _valid_question(prompt="A third distinct multiple-choice question here?",
                                    source_quote="Elephants can fly to the moon.")
    parsed = _lenient_bundle(good_items[:-1] + [unverifiable])
    client = Mock()
    client.responses.parse.return_value = SimpleNamespace(status="completed", output_parsed=parsed)
    bundle = generate_bundle(SIMPLE_ELEPHANT_TEXT, "k", client=client)
    assert len(bundle.comprehension) == len(good_items) - 1
    assert all(q.source_quote != "Elephants can fly to the moon." for q in bundle.comprehension)


def test_generate_bundle_drops_a_free_text_item_whose_answer_isnt_in_the_passage():
    good_items = _full_comprehension_mix()
    made_up_answer = _valid_short_answer_question(
        prompt="What do elephants use to fly?", answer_text="magic wings", max_words=2,
        source_quote="Elephants are the largest land animals on Earth.")
    parsed = _lenient_bundle(good_items[:-1] + [made_up_answer])
    client = Mock()
    client.responses.parse.return_value = SimpleNamespace(status="completed", output_parsed=parsed)
    bundle = generate_bundle(SIMPLE_ELEPHANT_TEXT, "k", client=client)
    assert len(bundle.comprehension) == len(good_items) - 1
    assert all(q.answer_text != "magic wings" for q in bundle.comprehension)


def test_generate_bundle_drops_a_matching_information_item_referencing_a_nonexistent_paragraph():
    good_items = _full_comprehension_mix()
    bad_paragraph_ref = _valid_matching_information_question(
        options=["Paragraph A", "Paragraph B", "Paragraph Z"], answer=2)
    parsed = _lenient_bundle(good_items[:-1] + [bad_paragraph_ref])
    client = Mock()
    client.responses.parse.return_value = SimpleNamespace(status="completed", output_parsed=parsed)
    bundle = generate_bundle(SIMPLE_ELEPHANT_TEXT, "k", client=client)
    assert len(bundle.comprehension) == len(good_items) - 1
    assert all(not (q.kind == "matching_information" and q.options[q.answer] == "Paragraph Z")
               for q in bundle.comprehension)


def test_generate_bundle_still_fails_if_grounding_drops_below_minimum():
    # Grounding-salvage only helps when enough grounded items remain
    # afterward; corrupting enough items to fall below PassageBundle's own
    # 6-item minimum must still fail, exactly like a structural failure would.
    good_items = _full_comprehension_mix()[:6]  # already exactly at the minimum
    unverifiable = _valid_question(prompt="A third distinct multiple-choice question here?",
                                    source_quote="Elephants can fly to the moon.")
    parsed = _lenient_bundle(good_items[:-1] + [unverifiable])
    client = Mock()
    client.responses.parse.return_value = SimpleNamespace(status="completed", output_parsed=parsed)
    with pytest.raises(ValueError):
        generate_bundle(SIMPLE_ELEPHANT_TEXT, "k", client=client)


# --- generate_bundle also reconciles matching_headings items that disagree
# on their shared heading pool (_reconcile_matching_headings_pool), instead
# of rejecting the whole bundle the way PassageBundle's own strict validator
# does when built directly (see test_matching_headings_across_a_bundle_must_
# share_the_same_heading_pool above).

def test_generate_bundle_reconciles_a_matching_headings_pool_mismatch():
    good_items = _full_comprehension_mix()  # already has one matching_headings item
    first_heading = _valid_matching_headings_question(
        prompt="Choose the best heading for the paragraph about elephant communication.")
    mismatched_heading = _valid_matching_headings_question(
        prompt="Choose the best heading for the paragraph about elephant diet.",
        options=["A totally different pool", "Another one", "A third", "A fourth"], answer=0)
    # Replace the one matching_headings item with two that disagree.
    parsed = _lenient_bundle(good_items[:-1] + [first_heading, mismatched_heading])
    client = Mock()
    client.responses.parse.return_value = SimpleNamespace(status="completed", output_parsed=parsed)
    bundle = generate_bundle(SIMPLE_ELEPHANT_TEXT, "k", client=client)
    heading_items = [q for q in bundle.comprehension if q.kind == "matching_headings"]
    # Only the item matching what the rest agreed on survives; the outlier is dropped.
    assert len(heading_items) == 1
    assert heading_items[0].options == first_heading["options"]


def test_generate_bundle_still_fails_if_reconciling_matching_headings_drops_below_minimum():
    # Same "not enough survives" guard as every other salvage pass: if
    # dropping the mismatched matching_headings item(s) leaves fewer than
    # PassageBundle's own 6-item minimum, this must still fail.
    filler = _full_comprehension_mix()[:4]  # 4 non-matching_headings items
    first_heading = _valid_matching_headings_question(
        prompt="Choose the best heading for the paragraph about elephant communication.")
    mismatched_heading = _valid_matching_headings_question(
        prompt="Choose the best heading for the paragraph about elephant diet.",
        options=["A totally different pool", "Another one", "A third", "A fourth"], answer=0)
    parsed = _lenient_bundle(filler + [first_heading, mismatched_heading])  # 6 items, 2 disagree
    client = Mock()
    client.responses.parse.return_value = SimpleNamespace(status="completed", output_parsed=parsed)
    with pytest.raises(ValueError):
        generate_bundle(SIMPLE_ELEPHANT_TEXT, "k", client=client)


SIMPLE_ELEPHANT_TEXT = (
    "Elephants are the largest land animals on Earth. They live in family groups led by an "
    "older female called the matriarch. Elephants use their trunks to eat, drink, and greet "
    "each other. A trunk has no bones but is very strong and flexible, with thousands of "
    "muscles inside it. Elephants can live for about sixty to seventy years in the wild, and "
    "they often stay close to their family for their whole lives. Young elephants learn from "
    "the older members of the herd how to find food and water, following the matriarch across "
    "the landscape as she leads the group between feeding grounds and watering holes. Many "
    "elephants today are endangered because of habitat loss and poaching, and groups around "
    "the world are working to protect them and the places where they live, from national parks "
    "to community-run conservation areas that try to reduce conflict between elephants and "
    "nearby farmers.\n\n"
    "Elephants also have an excellent memory and can recognize other elephants and humans they "
    "have met many years earlier, even after long periods apart. They communicate using a wide "
    "range of sounds, including deep rumbles that travel long distances and can be felt through "
    "the ground by other elephants far away, letting separated family members stay in touch "
    "across the savanna. Despite their huge size, elephants are gentle and social animals that "
    "often show affection toward members of their own family, touching each other with their "
    "trunks as a form of greeting or comfort. Their large ears also help them stay cool in hot "
    "climates by releasing heat from the blood vessels close to the surface of the skin, acting "
    "almost like natural radiators that keep the animal from overheating during the hottest part "
    "of the day.\n\n"
    "Elephants eat an enormous amount of food every day, spending as much as sixteen hours "
    "feeding on grasses, leaves, bark, and fruit to meet their daily energy needs in the wild. "
    "Calves stay close to their mothers for several years, learning survival skills from the "
    "rest of the herd before eventually becoming independent adults capable of finding their own "
    "food and water. A newborn calf can weigh well over a hundred kilograms and is able to stand "
    "and walk within hours of birth, though it will continue to nurse from its mother for several "
    "years while it learns to use its trunk, which contains no bones and relies entirely on a "
    "dense network of muscles for its remarkable flexibility and strength.\n\n"
    "There are three living species of elephant: the African bush elephant, the smaller African "
    "forest elephant, and the Asian elephant, which can be distinguished from its African "
    "relatives by its smaller ears and a single lobe at the tip of its trunk rather than two. "
    "Both male and, in African species, female elephants can grow long curved tusks, which are "
    "actually elongated incisor teeth made of ivory and used for digging, stripping bark, and "
    "defense. For centuries elephants have been hunted for their ivory, and this trade remains "
    "one of the biggest threats to wild populations today, alongside the loss of habitat as "
    "farmland and settlements expand into former elephant range. Herds are usually led by the "
    "oldest and most experienced female, the matriarch, whose knowledge of the landscape, "
    "including the location of water during a drought, can be critical to the survival of the "
    "whole group.\n\n"
    "Adult males typically leave the herd as they mature and often live alone or form loose "
    "bachelor groups with other males, only rejoining female herds briefly during the breeding "
    "season. Because elephants eat such large quantities of vegetation and travel long distances "
    "in search of food and water, they play an important role in shaping the ecosystems they "
    "live in, spreading seeds over wide areas as they travel and creating clearings in dense "
    "vegetation that other species then use. Scientists sometimes call elephants a keystone "
    "species and even ecosystem engineers because so many other plants and animals depend, "
    "directly or indirectly, on the paths, waterholes, and clearings that elephant herds create "
    "as they move across the landscape in search of food each year."
)


def test_fetch_passages_happy_path_returns_bundled_passage():
    session = _fake_session(
        search_titles=["Elephant"],
        extract_pages=[_wiki_page(title="Elephant", extract=SIMPLE_ELEPHANT_TEXT,
                                   fullurl="https://en.wikipedia.org/wiki/Elephant")],
    )
    client = mock_bundle_client()
    results = fetch_passages("Animals", count=1, api_key="k", session=session, client=client,
                              rng=random.Random(1))
    assert len(results) == 1
    p = results[0]
    assert p["topic"] == "Animals"
    assert p["body"] == SIMPLE_ELEPHANT_TEXT
    assert len(p["comprehension"]) == 11
    assert len(p["vocabulary"]) == 1
    assert p["is_excerpt"] is False
    # No thumbnail was present in the response, so no picture is attached,
    # but that alone must not stop the passage itself from being returned.
    assert p["image_bytes"] is None
    assert p["image_ext"] is None


def test_fetch_passages_includes_downloaded_picture_when_thumbnail_present():
    session = _fake_session(
        search_titles=["Elephant"],
        extract_pages=[_wiki_page(title="Elephant", extract=SIMPLE_ELEPHANT_TEXT,
                                   thumbnail="https://upload.wikimedia.org/elephant-thumb.jpg")],
        image_resp=_resp(content=b"fakejpegbytes", headers={"Content-Type": "image/jpeg"}),
    )
    client = mock_bundle_client()
    results = fetch_passages("Animals", count=1, api_key="k", session=session, client=client,
                              rng=random.Random(1))
    assert len(results) == 1
    assert results[0]["image_bytes"] == b"fakejpegbytes"
    assert results[0]["image_ext"] == "jpg"


def test_fetch_passages_still_returns_passage_when_picture_download_fails():
    session = _fake_session(
        search_titles=["Elephant"],
        extract_pages=[_wiki_page(title="Elephant", extract=SIMPLE_ELEPHANT_TEXT,
                                   thumbnail="https://upload.wikimedia.org/elephant-thumb.jpg")],
        image_resp=_resp(status_code=500, content=b"", headers={}),
    )
    client = mock_bundle_client()
    results = fetch_passages("Animals", count=1, api_key="k", session=session, client=client,
                              rng=random.Random(1))
    assert len(results) == 1
    assert results[0]["image_bytes"] is None
    assert results[0]["image_ext"] is None


def test_fetch_passages_skips_existing_titles():
    session = _fake_session(search_titles=["Elephant"], extract_pages=[])
    client = mock_bundle_client()
    results = fetch_passages("Animals", count=1, api_key="k", session=session, client=client,
                              existing_titles={"Elephant"}, rng=random.Random(1))
    assert results == []
    # Discovering candidates still costs one live search call; what must NOT
    # happen is fetching the page's own content for a title already in the list.
    extract_calls = [c for c in session.get.call_args_list
                      if (c.kwargs.get("params") or {}).get("list") != "search"
                      and (c.kwargs.get("params") or {}).get("prop") != "pageprops"]
    assert extract_calls == []


def test_fetch_passages_skips_pages_that_fail_readability():
    session = _fake_session(search_titles=["Dense"], extract_pages=[_wiki_page(
        title="Dense", extract="Word " * 500)])
    client = mock_bundle_client()
    results = fetch_passages("Animals", count=1, api_key="k", session=session, client=client,
                              rng=random.Random(1))
    assert results == []
    client.responses.parse.assert_not_called()


def test_fetch_passages_skips_candidates_whose_bundle_generation_fails():
    session = _fake_session(search_titles=["Elephant"], extract_pages=[_wiki_page(
        title="Elephant", extract=SIMPLE_ELEPHANT_TEXT)])
    client = Mock()
    client.responses.parse.return_value = SimpleNamespace(status="incomplete", output_parsed=None)
    results = fetch_passages("Animals", count=1, api_key="k", session=session, client=client,
                              rng=random.Random(1))
    assert results == []


def test_fetch_passages_caps_at_requested_count():
    session = _fake_session(search_titles=["Elephant", "Dolphin", "Owl"], extract_pages=[_wiki_page(
        title="Elephant", extract=SIMPLE_ELEPHANT_TEXT)])
    client = mock_bundle_client()
    results = fetch_passages("Animals", count=2, api_key="k", session=session, client=client,
                              rng=random.Random(3))
    assert len(results) <= 2


def test_fetch_passages_fetches_candidates_one_at_a_time_in_seed_order():
    # Regression test for a real hang: an earlier version of fetch_passages
    # fetched a batch of candidates concurrently in threads, which hung
    # indefinitely for a real user whose Python build's SSL stack doesn't
    # reliably support simultaneous HTTPS requests. Fetching must stay
    # strictly sequential, one page-fetch request at a time, and must not
    # reorder or skip past candidates while doing so.
    titles = [f"Topic {i}" for i in range(7)]
    session = _fake_session(search_titles=titles, extract_pages=[_wiki_page(
        title="Dense", extract="Word " * 500)])
    client = mock_bundle_client()
    skipped = []
    results = fetch_passages("Animals", count=100, api_key="k", session=session, client=client,
                              rng=random.Random(7), on_skip=lambda t, r: skipped.append((t, r)))
    assert results == []
    # An independent call with the same rng seed, mirroring exactly what
    # fetch_passages does internally, gives the expected post-shuffle order.
    expected_order, _ = candidate_titles_for_topic(
        "Animals", count=SEARCH_POOL_SIZE, rng=random.Random(7), session=session)
    assert [t for t, _ in skipped] == expected_order
    assert all("too complex" in reason for _, reason in skipped)
    # Every candidate actually got its own page-fetch request, proving the
    # loop walked the whole list rather than stopping short.
    extract_calls = [c for c in session.get.call_args_list
                      if (c.kwargs.get("params") or {}).get("list") != "search"
                      and (c.kwargs.get("params") or {}).get("prop") != "pageprops"]
    assert len(extract_calls) == len(titles)


def test_fetch_passages_reports_why_existing_titles_were_skipped():
    session = _fake_session(search_titles=["Elephant"], extract_pages=[])
    client = mock_bundle_client()
    skipped = []
    results = fetch_passages("Animals", count=1, api_key="k", session=session, client=client,
                              existing_titles={"Elephant"}, rng=random.Random(1),
                              on_skip=lambda title, reason: skipped.append((title, reason)))
    assert results == []
    assert skipped == [("Elephant", "already in your reading list")]


def test_fetch_passages_reports_why_a_page_failed_readability():
    session = _fake_session(search_titles=["Dense"], extract_pages=[_wiki_page(
        title="Dense", extract="Word " * 500)])
    client = mock_bundle_client()
    skipped = []
    fetch_passages("Animals", count=1, api_key="k", session=session, client=client,
                    rng=random.Random(1), on_skip=lambda title, reason: skipped.append((title, reason)))
    assert skipped
    assert all("too complex" in reason or "too short/long" in reason for _, reason in skipped)


def test_fetch_passages_reports_a_network_failure_honestly_not_as_a_missing_page():
    # A real network/timeout error while fetching a page's own content (as
    # opposed to Wikipedia's search itself) must not be reported with the
    # same wording as an actually-missing or disambiguation page — that
    # would tell the user something false about a page that, for all this
    # candidate knows, exists and is perfectly fine.
    import requests
    session = Mock()

    def get_side_effect(url, **kwargs):
        params = kwargs.get("params") or {}
        if params.get("list") == "search":
            return _resp(json_data={"query": {"search": [{"title": "Elephant"}]}})
        raise requests.RequestException("boom")

    session.get.side_effect = get_side_effect
    client = mock_bundle_client()
    skipped = []
    fetch_passages("Animals", count=1, api_key="k", session=session, client=client,
                    rng=random.Random(1), on_skip=lambda title, reason: skipped.append((title, reason)))
    assert skipped
    assert all("network error or timeout" in reason for _, reason in skipped)
    assert all("disambiguation" not in reason and "doesn't exist" not in reason for _, reason in skipped)


def test_fetch_passages_reports_a_search_failure_honestly():
    # If discovering candidates itself fails (Wikipedia's search endpoint is
    # unreachable), that's a property of the search, not of any one page, so
    # the fetch reports it once for the topic instead of coming back silent.
    import requests
    session = Mock()
    session.get.side_effect = requests.RequestException("boom")
    client = mock_bundle_client()
    skipped = []
    results = fetch_passages("Animals", count=1, api_key="k", session=session, client=client,
                              rng=random.Random(1), on_skip=lambda title, reason: skipped.append((title, reason)))
    assert results == []
    assert skipped == [("Animals", "couldn't reach Wikipedia's search (network error or timeout) — try again")]
    client.responses.parse.assert_not_called()


def test_fetch_passages_reports_when_search_finds_no_candidates():
    session = _fake_session(search_titles=[], extract_pages=[])
    client = mock_bundle_client()
    skipped = []
    results = fetch_passages("Animals", count=1, api_key="k", session=session, client=client,
                              rng=random.Random(1), on_skip=lambda title, reason: skipped.append((title, reason)))
    assert results == []
    assert skipped == [("Animals", "Wikipedia's search returned no candidate pages for this topic")]


def test_fetch_passages_reports_why_bundle_generation_failed():
    session = _fake_session(search_titles=["Elephant"], extract_pages=[_wiki_page(
        title="Elephant", extract=SIMPLE_ELEPHANT_TEXT)])
    client = Mock()
    client.responses.parse.return_value = SimpleNamespace(status="incomplete", output_parsed=None)
    skipped = []
    fetch_passages("Animals", count=1, api_key="k", session=session, client=client,
                    rng=random.Random(1), on_skip=lambda title, reason: skipped.append((title, reason)))
    assert skipped
    assert all("AI couldn't draft" in reason for _, reason in skipped)


def test_quote_is_verifiable_ignores_case_and_whitespace():
    from coach.reading import _quote_is_verifiable
    passage = "Elephants   are\nthe largest land animals on Earth."
    assert _quote_is_verifiable("elephants are the largest land animals on earth.", passage)
    assert not _quote_is_verifiable("Elephants can fly to the moon.", passage)


def test_fetch_passages_drops_a_single_unverifiable_quote_without_skipping_the_candidate():
    # A quote the passage never actually contains is dropped on its own
    # (see _drop_ungrounded_items, run inside generate_bundle) rather than
    # sinking an otherwise-good ~11-question candidate the way it used to.
    session = _fake_session(search_titles=["Elephant"], extract_pages=[_wiki_page(
        title="Elephant", extract=SIMPLE_ELEPHANT_TEXT)])
    client = mock_bundle_client()
    client.responses.parse.return_value.output_parsed.comprehension[0].source_quote = \
        "Elephants can fly to the moon."
    skipped = []
    results = fetch_passages("Animals", count=1, api_key="k", session=session, client=client,
                              rng=random.Random(1), on_skip=lambda title, reason: skipped.append((title, reason)))
    assert len(results) == 1
    assert not skipped
    assert all(q["source_quote"] != "Elephants can fly to the moon." for q in results[0]["comprehension"])
    # One of the original 11 items was dropped for being ungrounded.
    assert len(results[0]["comprehension"]) == 10


def test_fetch_passages_skips_candidate_when_too_few_grounded_items_survive():
    # If enough items are ungrounded that grounding-salvage can't keep the
    # bundle at PassageBundle's own 6-item minimum, the whole candidate is
    # still skipped — the message flows through the same "AI couldn't draft
    # a usable activity" path used for any unsalvageable bundle.
    session = _fake_session(search_titles=["Elephant"], extract_pages=[_wiki_page(
        title="Elephant", extract=SIMPLE_ELEPHANT_TEXT)])
    client = mock_bundle_client()
    # mock_bundle_client's bundle has 11 items; corrupting 6 of them leaves
    # only 5 grounded, below the floor.
    for q in client.responses.parse.return_value.output_parsed.comprehension[:6]:
        q.source_quote = "Elephants can fly to the moon."
    skipped = []
    results = fetch_passages("Animals", count=1, api_key="k", session=session, client=client,
                              rng=random.Random(1), on_skip=lambda title, reason: skipped.append((title, reason)))
    assert results == []
    assert skipped
    assert all("AI couldn't draft a usable activity" in reason for _, reason in skipped)


def test_fetch_passages_accepts_a_mix_of_comprehension_kinds():
    session = _fake_session(search_titles=["Elephant"], extract_pages=[_wiki_page(
        title="Elephant", extract=SIMPLE_ELEPHANT_TEXT)])
    parsed = PassageBundle.model_validate(dict(
        comprehension=_full_comprehension_mix(),
        vocabulary=[],
    ))
    client = Mock()
    client.responses.parse.return_value = SimpleNamespace(status="completed", output_parsed=parsed)
    results = fetch_passages("Animals", count=1, api_key="k", session=session, client=client,
                              rng=random.Random(1))
    assert len(results) == 1
    kinds = [q["kind"] for q in results[0]["comprehension"]]
    assert kinds == ["multiple_choice", "multiple_choice", "true_false_not_given",
                      "true_false_not_given", "yes_no_not_given", "sentence_completion",
                      "short_answer", "matching_sentence_endings", "matching_features",
                      "matching_information", "matching_headings"]
    # A Not Given item carries no quote to verify, and that must not sink the
    # whole candidate the way an actually-unverifiable quote would.
    ng_item = results[0]["comprehension"][3]
    assert ng_item["options"][ng_item["answer"]] == "Not Given"
    assert not ng_item.get("source_quote")
    # Free-text kinds carry an answer_text/max_words pair instead of options.
    sentence_item = results[0]["comprehension"][5]
    assert sentence_item["answer_text"] == "sixty to seventy"
    assert sentence_item["max_words"] == 3
    # matching_information's option must reference a real paragraph label in this passage.
    info_item = results[0]["comprehension"][9]
    assert info_item["options"][info_item["answer"]] == "Paragraph B"


def test_find_supporting_sentence_picks_the_best_overlapping_sentence():
    from coach.reading import find_supporting_sentence
    quote = find_supporting_sentence(
        explanation="The passage explains females lead a nomadic life with large home ranges, "
                     "while males establish smaller territories.",
        answer_text="Females are nomadic; males establish small territories.",
        passage_text=SIMPLE_ELEPHANT_TEXT + " While females lead a nomadic life searching for "
                     "prey in large home ranges, males establish much smaller territories.",
    )
    assert quote == ("While females lead a nomadic life searching for prey in large home ranges, "
                      "males establish much smaller territories.")


def test_find_supporting_sentence_returns_none_when_nothing_overlaps_enough():
    from coach.reading import find_supporting_sentence
    assert find_supporting_sentence(
        explanation="A totally unrelated claim about something else entirely.",
        answer_text="Nothing here matches.",
        passage_text=SIMPLE_ELEPHANT_TEXT,
    ) is None


def test_highlight_passage_html_wraps_each_quote_with_its_number():
    from coach.reading import highlight_passage_html
    body = "Elephants are the largest land animals on Earth. They live in family groups."
    out = highlight_passage_html(body, [
        ("largest land animals on Earth", 1),
        ("live in family groups", 2),
    ])
    assert ('<mark class="reading-highlight">largest land animals on Earth.'
            in out or '<mark class="reading-highlight">largest land animals on Earth'
            in out)
    assert "<sup>1</sup>" in out
    assert "<sup>2</sup>" in out
    assert "Elephants are the" in out  # untouched prefix survives


def test_highlight_passage_html_matches_case_and_whitespace_insensitively():
    from coach.reading import highlight_passage_html
    body = "Elephants are   the\nlargest land animals on Earth."
    out = highlight_passage_html(body, [("elephants ARE the largest land animals on earth", 1)])
    assert "<sup>1</sup>" in out
    assert "<mark" in out


def test_highlight_passage_html_skips_a_quote_not_found_and_escapes_html():
    from coach.reading import highlight_passage_html
    body = "A & B are <friends>."
    out = highlight_passage_html(body, [("nothing like this", 1), (None, 2)])
    assert "<mark" not in out
    assert "&amp;" in out
    assert "&lt;friends&gt;" in out


def test_highlight_passage_html_skips_overlapping_later_highlight():
    from coach.reading import highlight_passage_html
    body = "The quick brown fox jumps over the lazy dog."
    out = highlight_passage_html(body, [("quick brown fox", 1), ("brown fox jumps", 2)])
    assert out.count("<mark") == 1
    assert "<sup>1</sup>" in out
    assert "<sup>2</sup>" not in out


# --- Store: passages, reading sessions, vocab appension, stats ---

@pytest.fixture
def reading_store(tmp_path):
    return Store(tmp_path / "reading.sqlite3")


def make_passage(pid="p1", topic="Animals", title="Elephant", vocabulary=None,
                  image_bytes=None, image_ext=None, comprehension=None):
    return dict(
        id=pid, title=title, topic=topic, body=SIMPLE_ELEPHANT_TEXT,
        source_name="Wikipedia", source_url="https://en.wikipedia.org/wiki/Elephant",
        license="CC BY-SA 4.0", is_excerpt=False, word_count=70,
        comprehension=comprehension if comprehension is not None else _full_comprehension_mix(),
        vocabulary=vocabulary if vocabulary is not None else [_valid_vocab()],
        image_bytes=image_bytes, image_ext=image_ext,
    )


def test_old_question_missing_source_quote_still_gets_a_real_fallback_quote(reading_store):
    # add_passages() re-validates through PassageBundle, which now requires
    # source_quote, so a genuinely pre-migration row (saved before that field
    # existed) can only be simulated by editing the stored JSON directly,
    # exactly as real rows fetched before this feature shipped look today.
    import json
    reading_store.add_passages([make_passage()])
    with reading_store.connection() as db:
        row = db.execute("SELECT payload FROM passages WHERE id='p1'").fetchone()
        payload = json.loads(row["payload"])
        del payload["comprehension"][0]["source_quote"]
        db.execute("UPDATE passages SET payload=? WHERE id='p1'", (json.dumps(payload),))
    stored = reading_store.passage("p1")
    q = stored["comprehension"][0]
    assert not q.get("source_quote")
    from coach.reading import find_supporting_sentence
    right = q["options"][q["answer"]]
    quote = find_supporting_sentence(q["explanation"], right, stored["body"])
    assert quote and quote in stored["body"]


def test_add_passages_round_trips_and_ignores_duplicate_ids(reading_store):
    added = reading_store.add_passages([make_passage()])
    assert added == 1
    again = reading_store.add_passages([make_passage()])
    assert again == 0
    stored = reading_store.passages()
    assert len(stored) == 1
    assert stored[0]["title"] == "Elephant"
    assert stored[0]["is_excerpt"] is False
    assert len(stored[0]["comprehension"]) == 11
    assert stored[0]["image_path"] is None  # no picture was supplied

    fetched = reading_store.passage("p1")
    assert fetched["id"] == "p1"
    assert reading_store.passage("missing") is None


def test_add_passages_writes_picture_to_disk_and_records_its_path(reading_store, tmp_path):
    reading_store.add_passages([make_passage(image_bytes=b"fakejpegbytes", image_ext="jpg")])
    stored = reading_store.passage("p1")
    assert stored["image_path"] is not None
    image_file = tmp_path / "reading_images" / "p1.jpg"
    assert image_file.exists()
    assert image_file.read_bytes() == b"fakejpegbytes"
    assert stored["image_path"] == str(image_file)


def test_add_passages_does_not_rewrite_picture_for_an_already_known_id(reading_store, tmp_path):
    reading_store.add_passages([make_passage(image_bytes=b"first", image_ext="jpg")])
    image_file = tmp_path / "reading_images" / "p1.jpg"
    assert image_file.read_bytes() == b"first"
    # Re-adding the same id is ignored at the DB level; the picture on disk
    # from the first fetch must not be silently overwritten by a second one.
    reading_store.add_passages([make_passage(image_bytes=b"second", image_ext="jpg")])
    assert image_file.read_bytes() == b"first"


def test_add_passages_filters_by_topic(reading_store):
    reading_store.add_passages([make_passage("p1", "Animals"), make_passage("p2", "Space", title="Mars")])
    assert len(reading_store.passages("Animals")) == 1
    assert len(reading_store.passages("Space")) == 1
    assert len(reading_store.passages("All topics")) == 2
    assert reading_store.passages("Nonexistent") == []


def test_choose_passage_prefers_unread_then_allows_reread(reading_store):
    reading_store.add_passages([make_passage("p1"), make_passage("p2", title="Dolphin")])
    picked_ids = set()
    for _ in range(10):
        p = reading_store.choose_passage("Animals", rng=random.Random())
        picked_ids.add(p["id"])
    assert picked_ids == {"p1", "p2"}

    sid = reading_store.start_reading_session("p1")
    reading_store.finish_reading_session(sid)
    sid = reading_store.start_reading_session("p2")
    reading_store.finish_reading_session(sid)
    # Both read now; a reread must still return something rather than None.
    again = reading_store.choose_passage("Animals", rng=random.Random())
    assert again["id"] in {"p1", "p2"}

    assert reading_store.choose_passage("Nonexistent topic") is None


def test_reading_session_lookup_by_id_and_latest_unfinished(reading_store):
    reading_store.add_passages([make_passage()])
    sid = reading_store.start_reading_session("p1")
    session = reading_store.reading_session(sid)
    assert session["passage_id"] == "p1"
    assert session["finished"] == 0
    assert session["payload"] == dict(passage_id="p1", comprehension_answers={}, vocab_chosen=[])

    latest = reading_store.reading_session()
    assert latest["id"] == sid
    assert reading_store.reading_session("does-not-exist") is None


def test_record_comprehension_answer_scores_against_bank_not_client(reading_store):
    reading_store.add_passages([make_passage()])
    sid = reading_store.start_reading_session("p1")
    q = make_passage()["comprehension"][0]
    correct_answer = q["options"][q["answer"]]
    wrong_answer = next(o for o in q["options"] if o != correct_answer)

    result = reading_store.record_comprehension_answer(sid, 0, correct_answer)
    assert result is True
    session = reading_store.reading_session(sid)
    assert session["payload"]["comprehension_answers"]["0"] == dict(selected=correct_answer, correct=True)

    result2 = reading_store.record_comprehension_answer(sid, 1, wrong_answer)
    assert result2 is False


def test_record_comprehension_answer_rejects_bad_selection(reading_store):
    reading_store.add_passages([make_passage()])
    sid = reading_store.start_reading_session("p1")
    with pytest.raises(ValueError):
        reading_store.record_comprehension_answer(sid, 0, "Not a real option")


def test_record_comprehension_answer_rejects_finished_session(reading_store):
    reading_store.add_passages([make_passage()])
    sid = reading_store.start_reading_session("p1")
    reading_store.finish_reading_session(sid)
    q = make_passage()["comprehension"][0]
    with pytest.raises(ValueError):
        reading_store.record_comprehension_answer(sid, 0, q["options"][q["answer"]])


def test_record_comprehension_answer_rejects_unknown_session(reading_store):
    reading_store.add_passages([make_passage()])
    with pytest.raises(ValueError):
        reading_store.record_comprehension_answer("nope", 0, "anything")


def test_record_comprehension_answer_scores_free_text_answers(reading_store):
    reading_store.add_passages([make_passage()])
    sid = reading_store.start_reading_session("p1")
    questions = make_passage()["comprehension"]
    idx = next(i for i, q in enumerate(questions) if q.get("kind") == "sentence_completion")

    result = reading_store.record_comprehension_answer(sid, idx, "Sixty to seventy")
    assert result is True
    session = reading_store.reading_session(sid)
    assert session["payload"]["comprehension_answers"][str(idx)] == dict(
        selected="Sixty to seventy", correct=True)

    result2 = reading_store.record_comprehension_answer(sid, idx, "Forty")
    assert result2 is False


def test_record_comprehension_answer_rejects_free_text_answer_over_the_word_limit(reading_store):
    reading_store.add_passages([make_passage()])
    sid = reading_store.start_reading_session("p1")
    questions = make_passage()["comprehension"]
    idx = next(i for i, q in enumerate(questions) if q.get("kind") == "short_answer")
    with pytest.raises(ValueError):
        reading_store.record_comprehension_answer(sid, idx, "way too many words for this one")


def test_record_comprehension_answer_rejects_empty_free_text_answer(reading_store):
    reading_store.add_passages([make_passage()])
    sid = reading_store.start_reading_session("p1")
    questions = make_passage()["comprehension"]
    idx = next(i for i, q in enumerate(questions) if q.get("kind") == "short_answer")
    with pytest.raises(ValueError):
        reading_store.record_comprehension_answer(sid, idx, "   ")


def test_add_vocab_words_appends_new_and_skips_existing(reading_store, tmp_path, monkeypatch):
    scratch_psv = tmp_path / "words.psv"
    scratch_psv.write_text(DATA.read_text(encoding="utf-8"), encoding="utf-8")

    # add_vocab_words calls the module-level `append_words` name that
    # storage.py imported, with no explicit path (so it defaults to the real
    # DATA file). Monkeypatch that name so this test writes to a scratch copy
    # instead of the app's live vocabulary bank.
    monkeypatch.setattr("coach.storage.append_words",
                         lambda rows, path=scratch_psv: append_words(rows, path=scratch_psv))

    existing_word = load_words()[0]["word"]
    reading_store.add_passages([make_passage(vocabulary=[
        _valid_vocab(word="zibbleflorp", pos="noun", definition="a made-up word used only in this test",
                     example="The animal used zibbleflorp to hide.", cloze="The animal used ___ to hide.",
                     distractors="noise;color;shape"),
        _valid_vocab(word=existing_word, example=f"This is a sentence about {existing_word}.",
                     cloze="This is a sentence about ___.", distractors="one;two;three"),
    ])])
    sid = reading_store.start_reading_session("p1")

    added = reading_store.add_vocab_words(sid, [0, 1])
    assert added == 1  # the duplicate (existing_word) is skipped

    rows = scratch_psv.read_text(encoding="utf-8")
    assert "zibbleflorp" in rows

    session = reading_store.reading_session(sid)
    assert session["payload"]["vocab_chosen"] == [0, 1]


def test_add_vocab_words_ignores_out_of_range_indices(reading_store, tmp_path, monkeypatch):
    scratch_psv = tmp_path / "words.psv"
    scratch_psv.write_text(DATA.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr("coach.storage.append_words",
                         lambda rows, path=scratch_psv: append_words(rows, path=scratch_psv))
    reading_store.add_passages([make_passage(vocabulary=[
        _valid_vocab(word="quibblewomp", example="The quibblewomp appeared in the story.",
                     cloze="The ___ appeared in the story.", distractors="mystery;shadow;figure"),
    ])])
    sid = reading_store.start_reading_session("p1")
    added = reading_store.add_vocab_words(sid, [0, 5, -1])
    assert added == 1


def test_finish_reading_session_records_activity_once(reading_store):
    reading_store.add_passages([make_passage()])
    sid = reading_store.start_reading_session("p1")
    reading_store.finish_reading_session(sid)
    reading_store.finish_reading_session(sid)  # idempotent, no duplicate row
    activity = reading_store.reading_activity()
    assert len(activity) == 1
    assert activity[0]["passage_id"] == "p1"
    assert activity[0]["topic"] == "Animals"

    session = reading_store.reading_session(sid)
    assert session["finished"] == 1


def test_reading_stats_computes_streak_topics_and_week(reading_store):
    reading_store.add_passages([make_passage("p1", "Animals"), make_passage("p2", "Space", title="Mars")])
    today = "2026-01-10"
    yesterday = "2026-01-09"
    old_day = "2025-12-01"

    sid = reading_store.start_reading_session("p1")
    reading_store.finish_reading_session(sid)
    with reading_store.connection() as db:
        db.execute("UPDATE reading_activity SET completed_at=? WHERE passage_id=?", (yesterday + "T09:00:00", "p1"))

    sid = reading_store.start_reading_session("p2")
    reading_store.finish_reading_session(sid)
    with reading_store.connection() as db:
        db.execute("UPDATE reading_activity SET completed_at=? WHERE passage_id=?", (today + "T09:00:00", "p2"))

    stats = reading_store.reading_stats(today=today)
    assert stats["total_passages"] == 2
    assert stats["topics_explored"] == 2
    assert stats["streak_days"] == 2  # today and yesterday both have activity
    assert stats["this_week"] == 2

    # A day outside the 7-day window should not count toward this_week.
    reading_store.add_passages([make_passage("p3", "Animals", title="Owl")])
    sid = reading_store.start_reading_session("p3")
    reading_store.finish_reading_session(sid)
    with reading_store.connection() as db:
        db.execute("UPDATE reading_activity SET completed_at=? WHERE passage_id=?", (old_day + "T09:00:00", "p3"))
    stats = reading_store.reading_stats(today=today)
    assert stats["this_week"] == 2
    assert stats["total_passages"] == 3


def test_reading_stats_empty_when_no_activity(reading_store):
    stats = reading_store.reading_stats(today="2026-01-10")
    assert stats == dict(total_passages=0, topics_explored=0, streak_days=0, this_week=0)


def test_store_migrates_a_passages_table_created_before_the_image_column_existed(tmp_path):
    import sqlite3

    db_path = tmp_path / "old.sqlite3"
    with sqlite3.connect(db_path) as db:
        db.execute("""CREATE TABLE passages (
            id TEXT PRIMARY KEY, topic TEXT NOT NULL, title TEXT NOT NULL, body TEXT NOT NULL,
            source_name TEXT NOT NULL, source_url TEXT NOT NULL, license TEXT NOT NULL,
            is_excerpt INTEGER NOT NULL DEFAULT 0, word_count INTEGER NOT NULL,
            payload TEXT NOT NULL, fetched_at TEXT NOT NULL)""")

    store = Store(db_path)  # must not raise despite the pre-existing, older schema
    added = store.add_passages([make_passage()])
    assert added == 1
    assert store.passage("p1")["image_path"] is None


def test_store_migrates_a_reading_activity_table_created_before_session_id_existed(tmp_path):
    import sqlite3

    db_path = tmp_path / "old_activity.sqlite3"
    with sqlite3.connect(db_path) as db:
        db.execute("""CREATE TABLE reading_activity (
            id TEXT PRIMARY KEY, passage_id TEXT NOT NULL, topic TEXT NOT NULL,
            completed_at TEXT NOT NULL)""")

    store = Store(db_path)  # must not raise despite the pre-existing, older schema
    store.add_passages([make_passage()])
    sid = store.start_reading_session("p1")
    store.finish_reading_session(sid)
    rows = store.reading_activity()
    assert len(rows) == 1
    # A completion recorded before this migration would show session_id as
    # None; one recorded after it links straight back to the full session.
    assert rows[0]["session_id"] == sid


def test_finish_reading_session_links_activity_back_to_its_session(reading_store):
    reading_store.add_passages([make_passage()])
    sid = reading_store.start_reading_session("p1")
    reading_store.finish_reading_session(sid)
    row = reading_store.reading_activity()[0]
    assert row["session_id"] == sid
    linked_session = reading_store.reading_session(row["session_id"])
    assert linked_session["id"] == sid
