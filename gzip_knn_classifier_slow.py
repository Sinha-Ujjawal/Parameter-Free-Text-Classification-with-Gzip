from heapq import nsmallest
from collections import defaultdict, namedtuple
import gzip

Prediction = namedtuple("Prediction", ["label", "distance"])

def new(texts, class_labels):
    return {
        "points": [
            (text, label, len(gzip.compress(text)))
            for text, label in zip(texts, class_labels)
        ]
    }

def classify(clf, x, n_neighbors = 5):
    cx = len(gzip.compress(x))
    label_counts = defaultdict(lambda: [0, 0.0])
    ret = None
    max_count = 0
    for distance, label in nsmallest(
        n_neighbors,
        (
            ((len(gzip.compress(x + y)) - min(cx, cy)) / max(cx, cy), label)
            for y, label, cy in clf["points"]
        ),
    ):
        entry = label_counts[label]
        entry[0] += 1
        entry[1] = min(entry[1], distance)
        if entry[0] >= max_count:
            max_count = entry[0]
            ret = label, entry[1]
    return ret

def classify_many(clf, xs, n_neighbors = 5):
    return [classify(clf, x, n_neighbors = n_neighbors) for x in xs]
