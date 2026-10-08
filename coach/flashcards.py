"""Offline card selection, with priority shared between learning groups."""
import random

WEIGHTS = {"Review": 40, "Learning": 35, "New": 15, "Familiar": 7, "Secure": 3}


def choose_card(words, progress, today, topic="All topics", recent=(), rng=None):
    rng = rng or random.Random()
    eligible = [w for w in words if topic == "All topics" or w["topic"] == topic]
    if not eligible:
        return None
    # Relax the oldest exclusions first when a topic has very few words.
    excluded = list(recent[-5:])
    candidates = [w for w in eligible if w["id"] not in excluded]
    while not candidates and excluded:
        excluded.pop(0)
        candidates = [w for w in eligible if w["id"] not in excluded]
    groups = {}
    for word in candidates:
        p = progress.get(word["id"], {"stage": "New", "due": None})
        group = "Review" if p["due"] and p["due"] <= today else p["stage"]
        groups.setdefault(group, []).append(word)
    group = rng.choices(list(groups), weights=[WEIGHTS[g] for g in groups], k=1)[0]
    return rng.choice(groups[group])
