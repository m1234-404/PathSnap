
import pandas as pd

from member_3_pavankumar.evaluation.continuity import (
    route_continuity,
    route_continuity_metrics,
)


def test_directed_connected_arcs():
    arcs = pd.DataFrame({
        "from_node": [1, 2, 3],
        "to_node": [2, 3, 4],
    })

    assert route_continuity([0, 1, 2], arcs) == 1.0


def test_disconnected_arcs():
    arcs = pd.DataFrame({
        "from_node": [1, 5],
        "to_node": [2, 6],
    })

    assert route_continuity([0, 1], arcs) == 0.0


def test_same_arc_is_connected():
    arcs = pd.DataFrame({
        "from_node": [1],
        "to_node": [2],
    })

    assert route_continuity([0, 0], arcs) == 1.0


def test_metrics_count_transitions():
    arcs = pd.DataFrame({
        "from_node": [1, 2, 5],
        "to_node": [2, 3, 6],
    })

    result = route_continuity_metrics([0, 1, 2], arcs)

    assert result["connected_transitions"] == 1
    assert result["disconnected_transitions"] == 1
    assert result["continuity_percentage"] == 50.0
