
"""Route continuity evaluation for predicted road arcs."""


def _get_edges(arcs):
    """Extract directed (from_node, to_node) pairs."""
    if hasattr(arcs, "columns"):
        columns = arcs.columns

        if "from_node" in columns and "to_node" in columns:
            source, destination = "from_node", "to_node"
        elif "from_node_id" in columns and "to_node_id" in columns:
            source, destination = "from_node_id", "to_node_id"
        else:
            raise ValueError("Arc data must contain source and destination columns")

        return list(zip(arcs[source], arcs[destination]))

    edges = list(arcs)

    if any(len(edge) != 2 for edge in edges):
        raise ValueError("Each arc must contain source and destination nodes")

    return edges


def route_continuity_metrics(arc_ids, arcs):
    """Calculate directed route continuity metrics."""
    edges = _get_edges(arcs)

    connected = 0
    disconnected = 0
    skipped = 0

    for previous, current in zip(arc_ids, arc_ids[1:]):
        if (
            previous is None
            or current is None
            or previous < 0
            or current < 0
            or previous >= len(edges)
            or current >= len(edges)
        ):
            skipped += 1
            continue

        previous = int(previous)
        current = int(current)

        # Repeating the same arc is a valid transition.
        same_arc = previous == current

        # Different arcs must connect in the correct direction.
        directed_connection = (
            edges[previous][1] == edges[current][0]
        )

        if same_arc or directed_connection:
            connected += 1
        else:
            disconnected += 1

    evaluated = connected + disconnected

    return {
        "connected_transitions": connected,
        "disconnected_transitions": disconnected,
        "skipped_transitions": skipped,
        "continuity": connected / evaluated if evaluated else float("nan"),
        "continuity_percentage": (
            100.0 * connected / evaluated if evaluated else float("nan")
        ),
    }


def route_continuity(arc_ids, arcs):
    """Return the directed route continuity fraction."""
    return route_continuity_metrics(arc_ids, arcs)["continuity"]
