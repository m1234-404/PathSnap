
"""Handling missing candidates and GPS observation gaps."""

import numpy as np


def split_candidate_segments(candidate_arc_ids):
    """Split observations into continuous candidate segments.

    Empty candidate lists mark a gap.
    Returns a list of (start_index, end_index) pairs.
    End index is exclusive.
    """
    segments = []
    start = None

    for index, candidates in enumerate(candidate_arc_ids):
        values = np.asarray(candidates)

        if values.size > 0:
            if start is None:
                start = index
        else:
            if start is not None:
                segments.append((start, index))
                start = None

    if start is not None:
        segments.append((start, len(candidate_arc_ids)))

    return segments


def find_gps_time_gaps(timestamps, max_gap_seconds):
    """Return indexes where the time gap exceeds the threshold."""
    if max_gap_seconds <= 0:
        raise ValueError("max_gap_seconds must be positive")

    times = np.asarray(timestamps, dtype=float)

    if times.ndim != 1:
        raise ValueError("timestamps must be one-dimensional")

    if not np.all(np.isfinite(times)):
        raise ValueError("timestamps must be finite")

    if np.any(np.diff(times) < 0):
        raise ValueError("timestamps must be ordered")

    return [
        index + 1
        for index, difference in enumerate(np.diff(times))
        if difference > max_gap_seconds
    ]
