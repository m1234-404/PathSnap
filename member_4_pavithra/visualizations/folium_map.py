
import folium

def create_trip_map(
    track,
    nodes,
    arcs,
    predicted_arc_ids=None,
    reference_arc_ids=None,
    output="results/figures/trip_map.html"
):
    """Create layered map. Reference layer is only for permitted offline analysis."""

    center = [
        float(track.latitude.mean()),
        float(track.longitude.mean())
    ]

    m = folium.Map(
        location=center,
        zoom_start=14,
        control_scale=True
    )

    gps = folium.FeatureGroup(name="Observed GPS", show=True)

    for r in track.itertuples():
        folium.CircleMarker(
            [r.latitude, r.longitude],
            radius=2,
            color="blue",
            fill=True
        ).add_to(gps)

    gps.add_to(m)

    def add_arcs(ids, name, color):
        layer = folium.FeatureGroup(name=name, show=True)

        for aid in dict.fromkeys(ids or []):
            if 0 <= int(aid) < len(arcs):
                a = arcs.iloc[int(aid)]
                p1 = nodes.iloc[int(a.from_node_id)]
                p2 = nodes.iloc[int(a.to_node_id)]

                folium.PolyLine(
                    [
                        [p1.latitude, p1.longitude],
                        [p2.latitude, p2.longitude]
                    ],
                    color=color,
                    weight=4
                ).add_to(layer)

        layer.add_to(m)

    add_arcs(predicted_arc_ids, "Predicted route", "red")
    add_arcs(reference_arc_ids, "Reference route (offline only)", "green")

    folium.LayerControl().add_to(m)

    bounds = [
        [track.latitude.min(), track.longitude.min()],
        [track.latitude.max(), track.longitude.max()]
    ]

    m.fit_bounds(bounds)

    m.save(output)

    return output
