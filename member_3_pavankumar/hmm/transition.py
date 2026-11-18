
"""Directed road-network transition probabilities."""

import numpy as np


def build_transition_probabilities(
    previous_arc_ids,
    current_arc_ids,
    arcs,
):
    """Build transitions between candidates at consecutive observations.

    An arc transition is allowed when:
    1. Both observations select the same road arc, or
    2. The previous arc's destination equals the current arc's source.

    Disconnected transitions receive probability zero. Rows with no valid
    outgoing transition remain all-zero; no artificial road connection
    is introduced.

    Parameters
    ----------
    previous_arc_ids : array-like
        Candidate arc IDs at observation t-1.
    current_arc_ids : array-like
        Candidate arc IDs at observation t.
    arcs : array-like, shape (number_of_arcs, 2)
        Directed road network: [from_node, to_node] per arc.

    Returns
    -------
    numpy.ndarray
        Row-normalized transition matrix with shape
        (len(previous_arc_ids), len(current_arc_ids)).
    """
    previous_arc_ids = np.asarray(previous_arc_ids, dtype=int)
    current_arc_ids = np.asarray(current_arc_ids, dtype=int)
    arcs = np.asarray(arcs, dtype=int)

    if arcs.ndim != 2 or arcs.shape[1] != 2:
        raise ValueError("arcs must have shape (number_of_arcs, 2)")

    if len(previous_arc_ids) == 0 or len(current_arc_ids) == 0:
        raise ValueError("Candidate arc lists cannot be empty.")

    if np.any(previous_arc_ids < 0) or np.any(current_arc_ids < 0):
        raise ValueError("Candidate arc IDs cannot be negative.")

    if np.any(previous_arc_ids >= len(arcs)):
        raise ValueError("Previous candidate arc ID is outside the road network.")

    if np.any(current_arc_ids >= len(arcs)):
        raise ValueError("Current candidate arc ID is outside the road network.")

    previous_destinations = arcs[previous_arc_ids, 1]
    current_sources = arcs[current_arc_ids, 0]

    same_arc = previous_arc_ids[:, None] == current_arc_ids[None, :]
    connected = (
        previous_destinations[:, None] == current_sources[None, :]
    )

    allowed = same_arc | connected

    transitions = allowed.astype(float)
    row_sums = transitions.sum(axis=1, keepdims=True)

    np.divide(
        transitions,
        row_sums,
        out=transitions,
        where=row_sums > 0,
    )

    return transitions
