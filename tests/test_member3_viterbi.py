
import numpy as np
import pytest

from member_3_pavankumar.hmm.viterbi import viterbi


def test_viterbi_selects_most_likely_path():
    emissions = [
        np.array([0.9, 0.1]),
        np.array([0.2, 0.8]),
    ]

    transitions = [
        np.array([
            [1.0, 0.0],
            [0.0, 1.0],
        ])
    ]

    assert viterbi(emissions, transitions) == [0, 0]


def test_viterbi_supports_different_candidate_counts():
    emissions = [
        np.array([0.8, 0.2]),
        np.array([0.1, 0.9, 0.2]),
    ]

    transitions = [
        np.array([
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ])
    ]

    assert viterbi(emissions, transitions) == [0, 1]


def test_viterbi_returns_invalid_path_when_disconnected():
    emissions = [
        np.array([0.8]),
        np.array([0.9]),
    ]

    transitions = [
        np.array([[0.0]])
    ]

    assert viterbi(emissions, transitions) == [-1, -1]


def test_viterbi_rejects_wrong_transition_count():
    with pytest.raises(ValueError):
        viterbi([np.array([1.0]), np.array([1.0])], [])


def test_viterbi_handles_one_observation():
    assert viterbi([np.array([0.2, 0.8])], []) == [1]
