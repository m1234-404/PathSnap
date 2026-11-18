def route_continuity(arc_ids, arcs):
    """Fraction of adjacent predicted arc pairs that share a road-network node."""
    edges=[(int(r.from_node_id),int(r.to_node_id)) for r in arcs.itertuples()]
    valid=0; total=max(0,len(arc_ids)-1)
    for a,b in zip(arc_ids,arc_ids[1:]):
        if a is None or b is None or a<0 or b<0 or a>=len(edges) or b>=len(edges): continue
        ea,eb=edges[a],edges[b]
        valid += int(bool(set(ea)&set(eb)))
    return valid/total if total else float('nan')
