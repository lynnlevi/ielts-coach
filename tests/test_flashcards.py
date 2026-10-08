import json
import random
import sqlite3
from collections import Counter

from coach.flashcards import choose_card
from coach.storage import Store


def pool():
    # Word 0 is a Learning word due today, so it lands in the Review group.
    stages = ['Learning', 'Learning', 'New', 'Familiar', 'Secure']
    words = [{'id': str(i), 'topic': 'A'} for i in range(5)]
    progress = {str(i): {'stage': stage, 'due': None} for i, stage in enumerate(stages)}
    progress['0']['due'] = '2026-09-17'
    return words, progress


def test_group_weights_and_due_priority():
    words, progress = pool()
    rng = random.Random(19)
    counts = Counter(choose_card(words, progress, '2026-09-17', rng=rng)['id'] for _ in range(20000))
    for wid, expected in zip(range(5), [.40, .35, .15, .07, .03]):
        assert abs(counts[str(wid)] / 20000 - expected) < .015
    # A large new-word group must not overwhelm the review group.
    words += [{'id': f'new{i}', 'topic': 'A'} for i in range(100)]
    counts = Counter(choose_card(words, progress, '2026-09-17', rng=rng)['id'] for _ in range(10000))
    assert .37 < counts['0'] / 10000 < .43
    progress['4']['due'] = '2026-09-16'
    class InspectRandom:
        def choices(self, groups, weights, k):
            assert groups == ['Review', 'Learning', 'New', 'Familiar']
            assert weights == [40, 35, 15, 7]
            return ['Review']
        def choice(self, candidates):
            assert {w['id'] for w in candidates} == {'0', '4'}
            return candidates[-1]
    assert choose_card(words, progress, '2026-09-17', rng=InspectRandom())['id'] == '4'


def test_topic_recent_and_small_pools():
    words = [{'id': str(i), 'topic': 'A'} for i in range(7)]
    words.append({'id': 'other', 'topic': 'B'})
    recent = []
    rng = random.Random(2)
    for _ in range(30):
        card = choose_card(words, {}, '2026-09-17', 'A', recent, rng)
        assert card['id'] not in recent[-5:]
        assert card['topic'] == 'A'
        recent.append(card['id'])
    assert choose_card(words[:2], {}, '2026-09-17', recent=['0', '1'])['id'] == '0'
    assert choose_card(words[:1], {}, '2026-09-17', recent=['0'])['id'] == '0'
    assert choose_card(words, {}, '2026-09-17', topic='Missing') is None


def test_activity_is_persistent_idempotent_and_separate(tmp_path):
    path = tmp_path / 'cards.sqlite3'
    store = Store(path)
    words, _ = pool()
    before = store.progress(words)
    store.record_flashcard('a', '0', 'review', '2026-09-17T12:00:00')
    store.record_flashcard('a', '0', 'review', '2026-09-17T12:01:00')
    store.record_flashcard('a', '0', 'skip')
    store.record_flashcard('b', '1', 'skip', '2026-09-17T12:02:00')
    store.record_flashcard('c', '0', 'review', '2026-09-18T12:00:00')
    store = Store(path)
    assert store.flashcards_reviewed('2026-09-17') == 1
    assert store.flashcards_reviewed('2026-09-18') == 1
    assert len(store.flashcard_activity()) == 3
    assert store.progress(words) == before
    assert store.attempts() == []
    assert len(json.loads(store.export_json())['tables']['flashcard_activity']) == 3
    backup = tmp_path / 'backup.sqlite3'
    backup.write_bytes(store.backup_bytes())
    with sqlite3.connect(backup) as db:
        assert db.execute('SELECT count(*) FROM flashcard_activity').fetchone()[0] == 3
