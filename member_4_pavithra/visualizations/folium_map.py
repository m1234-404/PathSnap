
from pathlib import Path
import folium


def create_trip_map(
    track,
    nodes,
    arcs,
    predicted_arc_ids=None,
    reference_arc_ids=None,
    output="results/figures/trip_map.html"
):
    # Calculate the centre of the GPS track
    latitude_center = track["latitude"].mean()
    longitude_center = track["longitude"].mean()

    map_center = [
        float(latitude_center),
        float(longitude_center)
    ]

    trip_map = folium.Map(
        location=map_center,
        zoom_start=14,
        control_scale=True
    )

    # Display the GPS locations
    gps_layer = folium.FeatureGroup(
        name="Observed GPS Points"
    )

    for _, point in track.iterrows():

        gps_location = [
            float(point["latitude"]),
            float(point["longitude"])
        ]

        folium.CircleMarker(
            location=gps_location,
            radius=2,
            color="blue",
            fill=True,
            fill_opacity=0.7
        ).add_to(gps_layer)

    trip_map.add_child(gps_layer)

    # Draw road segments on the map
    def add_route_layer(segment_ids, layer_name, route_color):

        route_layer = folium.FeatureGroup(
            name=layer_name
        )

        if segment_ids is not None:

            unique_segments = dict.fromkeys(segment_ids)

            for segment_id in unique_segments:

                segment_id = int(segment_id)

                if segment_id < 0 or segment_id >= len(arcs):
                    continue

                road_segment = arcs.iloc[segment_id]

                start_node = nodes.iloc[
                    int(road_segment["from_node_id"])
                ]

                end_node = nodes.iloc[
                    int(road_segment["to_node_id"])
                ]

                route_coordinates = [
                    [
                        float(start_node["latitude"]),
                        float(start_node["longitude"])
                    ],
                    [
                        float(end_node["latitude"]),
                        float(end_node["longitude"])
                    ]
                ]

                folium.PolyLine(
                    locations=route_coordinates,
                    color=route_color,
                    weight=4,
                    opacity=0.85
                ).add_to(route_layer)

        trip_map.add_child(route_layer)

    # Display the predicted route
    add_route_layer(
        predicted_arc_ids,
        "Predicted Route",
        "red"
    )

    # Display reference route only when supplied
    if reference_arc_ids is not None:

        add_route_layer(
            reference_arc_ids,
            "Reference Route - Offline Analysis Only",
            "green"
        )

    # Allow users to switch map layers
    folium.LayerControl().add_to(trip_map)

    # Create output directory and save the map
    output_path = Path(output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    trip_map.save(str(output_path))

    return str(output_path)
