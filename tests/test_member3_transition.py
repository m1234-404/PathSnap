
import numpy as np
import pytest

from member_3_pavankumar.hmm.transition import (
    build_transition_probabilities,
)


def test_directed_connected_arcs_are_allowed():
    # Arc 0: 0 -> 1
    # Arc 1: 1 -> 2
    # Arc 2: 5 -> 6 (disconnected)
    arcs = np.array([
        [0, 1],
        [1, 2],
        [5, 6],
    ])

    result = build_transition_probabilities(
        [0],
        [1, 2],
        arcs,
    )

    assert np.allclose(result, [[1.0, 0.0]])


def test_same_arc_is_allowed():
    arcs = np.array([
        [0, 1],
        [1, 2],
    ])

    result = build_transition_probabilities(
        [0],
        [0, 1],
        arcs,
    )

    assert np.allclose(result, [[0.5, 0.5]])


def test_disconnected_transitions_remain_zero():
    arcs = np.array([
        [0, 1],
        [5, 6],
    ])

    result = build_transition_probabilities(
        [0],
        [1],
        arcs,
    )

    assert np.allclose(result, [[0.0]])


def test_transition_rows_are_normalized_when_possible():
    arcs = np.array([
        [0, 1],
        [1, 2],
        [1, 3],
    ])

    result = build_transition_probabilities(
        [0],
        [1, 2],
        arcs,
    )

    assert np.isclose(result[0].sum(), 1.0)
    assert np.allclose(result[0], [0.5, 0.5])


def test_invalid_arc_id_is_rejected():
    arcs = np.array([[0, 1]])

    with pytest.raises(ValueError):
        build_transition_probabilities([5], [0], arcs)
