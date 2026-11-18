
from member_3_pavankumar.hmm.gap_handling import (
    split_candidate_segments,
    find_gps_time_gaps,
)


def test_split_candidate_segments():
    candidates = [
        [1, 2],
        [2, 3],
        [],
        [4, 5],
        [5],
    ]

    assert split_candidate_segments(candidates) == [
        (0, 2),
        (3, 5),
    ]


def test_all_candidates_missing():
    assert split_candidate_segments([[], []]) == []


def test_no_missing_candidates():
    assert split_candidate_segments([[1], [2], [3]]) == [(0, 3)]


def test_find_gps_time_gaps():
    timestamps = [0, 10, 20, 100, 110]

    assert find_gps_time_gaps(timestamps, 30) == [3]


def test_no_gps_time_gaps():
    timestamps = [0, 10, 20, 30]

    assert find_gps_time_gaps(timestamps, 30) == []
