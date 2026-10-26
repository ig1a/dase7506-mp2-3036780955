"""Training-free final-answer plurality, using the teacher's numeric parser."""
from collections import Counter

from common import parse_answer


def predict(bank, arrays, checkpoint):
    selected = []
    for row in bank:
        answers = [parse_answer(c['text']) for c in row['candidates']]
        counts = Counter(a for a in answers if a is not None)
        if not counts:
            selected.append(0)
            continue
        best = max(counts.values())
        selected.append(next(i for i, a in enumerate(answers) if a is not None and counts[a] == best))
    return selected
