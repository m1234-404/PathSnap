
"""Viterbi decoder for variable candidate sets."""

import numpy as np


def viterbi(emissions, transitions, log_floor=1e-12):
    """Find the most likely sequence of candidate states.

    emissions:
        List of 1D arrays. One probability per candidate at each observation.

    transitions:
        List of 2D matrices. transitions[t] connects observations
        t-1 and t.

    Returns:
        List of selected candidate indices, one per observation.
        Returns -1 when no valid complete path exists.
    """
    if not emissions:
        return []

    if log_floor <= 0:
        raise ValueError("log_floor must be positive")

    emission_logs = []

    for e in emissions:
        e = np.asarray(e, dtype=float)

        if e.ndim != 1 or len(e) == 0:
            raise ValueError("Each observation needs at least one candidate")

        if not np.all(np.isfinite(e)) or np.any(e < 0) or np.any(e > 1):
            raise ValueError("Emission probabilities must be finite and in [0, 1]")

        emission_logs.append(np.log(np.maximum(e, log_floor)))

    if len(transitions) != len(emissions) - 1:
        raise ValueError("Need exactly T-1 transition matrices")
    
    dp = emission_logs[0].copy()
    backpointers = []

    for t, transition in enumerate(transitions, start=1):
        transition = np.asarray(transition, dtype=float)
        previous_count = len(emission_logs[t - 1])
        current_count = len(emission_logs[t])

        if transition.shape != (previous_count, current_count):
            raise ValueError(f"Invalid transition shape at observation {t}")

        if not np.all(np.isfinite(transition)) or np.any(transition < 0):
            raise ValueError("Transition probabilities must be finite and nonnegative")

        scores = np.full((previous_count, current_count), -np.inf)
        valid = transition > 0
        scores[valid] = (
            dp[:, None] + np.log(np.maximum(transition, log_floor))
        )[valid]

        best_previous = np.argmax(scores, axis=0)
        best_scores = scores[best_previous, np.arange(current_count)]

        dp = best_scores + emission_logs[t]
        backpointers.append(best_previous)

    if not np.any(np.isfinite(dp)):
        return [-1] * len(emissions)

    state = int(np.argmax(dp))
    path = [state]

    for pointer in reversed(backpointers):
        state = int(pointer[state])
        path.append(state)

    path.reverse()
    return path
