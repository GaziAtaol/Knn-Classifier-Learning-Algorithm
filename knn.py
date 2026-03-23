import math
from collections import Counter


def euclidean_distance(x, y):
    """
    Computes the Euclidean distance between two attribute vectors.
    Uses ALL columns — dynamic, not hardcoded.
    Formula: d = sqrt( sum( (xi - yi)^2 ) )

    Parameters:
        x, y : lists of floats (attribute values)

    Returns:
        float : the distance between x and y
    """
    return math.sqrt(sum((xi - yi) ** 2 for xi, yi in zip(x, y)))

def get_neighbours(training_data, test_sample, k):
    """
    Finds the k nearest neighbours of test_sample in training_data
    using Euclidean distance across ALL attributes.

    Parameters:
        training_data : list of dicts with 'attributes' and 'label'
        test_sample   : dict with 'attributes'
        k             : number of neighbours to return

    Returns:
        list of (dist, sample) tuples, sorted by distance ascending
    """
    # Safety check: k cannot be larger than training set
    k = min(k, len(training_data))

    # Calculate distance from test_sample to every training sample
    distances = []
    for sample in training_data:
        dist = euclidean_distance(test_sample["attributes"], sample["attributes"])
        distances.append((dist, sample))

    # Sort by distance ascending (closest first)
    distances.sort(key=lambda pair: pair[0])

    # Return k nearest as (dist, sample) — dist kept for tie-breaking
    return distances[:k]


def classify(training_data, test_sample, k):
    """
    Classifies a test_sample using k-NN majority vote.

    Rules:
        1. MAJORITY VOTE: find the most common label among k nearest neighbours.
        2. TIE-BREAKING: if tie, the class of the closest neighbour wins.
           Conflict resolution is always distance-based.

    Parameters:
        training_data : list of dicts with 'attributes' and 'label'
        test_sample   : dict with 'attributes'
        k             : number of neighbours to consider

    Returns:
        string : predicted class label
    """
    neighbours = get_neighbours(training_data, test_sample, k)

    # RULE 1 — Majority vote among k nearest neighbours
    vote_counts = Counter(sample["label"] for dist, sample in neighbours)
    max_votes   = max(vote_counts.values())
    top_classes = [label for label, count in vote_counts.items() if count == max_votes]

    # No tie — return the winner directly
    if len(top_classes) == 1:
        return top_classes[0]

    # RULE 2 — Tie-breaking: closest neighbour's class wins
    # neighbours already sorted by distance ascending
    for dist, sample in neighbours:
        if sample["label"] in top_classes:
            return sample["label"]