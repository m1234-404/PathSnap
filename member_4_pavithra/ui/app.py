
import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components

from member_4_pavithra.visualizations.folium_map import create_trip_map


API = os.getenv(
    "PATHSNAP_API_URL",
    "http://127.0.0.1:8000"
)

st.set_page_config(
    page_title="PathSnap Evaluator",
    page_icon="🗺️",
    layout="wide"
)

st.title("🗺️ PathSnap — Route Reconstruction")
st.caption("GPS Trajectory Map Matching and Route Evaluation")


# ---------------- API CONNECTION ----------------

try:
    response = requests.get(
        f"{API}/trips",
        timeout=10
    )
    response.raise_for_status()

    trip_data = response.json()
    ids = trip_data.get("trip_ids", [])

except requests.RequestException as exc:
    st.error(f"Cannot connect to PathSnap API: {exc}")
    st.info("Make sure the FastAPI server is running.")
    st.stop()


if not ids:
    st.warning("No evaluation trips are currently available.")
    st.stop()


# ---------------- TRIP SELECTION ----------------

st.subheader("Select Trip")

trip = st.selectbox(
    "Available trips",
    ids
)


# ---------------- ROUTE RECONSTRUCTION ----------------

if st.button("🚀 Reconstruct Route", type="primary"):

    with st.spinner("Processing GPS trajectory..."):

        try:
            response = requests.post(
                f"{API}/predict",
                json={"trip_id": trip},
                timeout=120
            )

            response.raise_for_status()
            result = response.json()

        except requests.RequestException as exc:
            st.error(f"Prediction request failed: {exc}")
            st.stop()


    # ---------------- EXTRACT RESULTS ----------------

    gps_points = result.get("gps_points", [])
    predicted_arcs = result.get("predicted_arc_ids", [])
    status = result.get("status", "unknown")


    # ---------------- METRICS ----------------

    st.divider()
    st.subheader("Trip Results")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Trip ID",
        result.get("trip_id", trip)
    )

    col2.metric(
        "GPS Observations",
        len(gps_points)
    )

    col3.metric(
        "Predicted Arcs",
        len(predicted_arcs)
    )


    # ---------------- STATUS ----------------

    if status == "pipeline_not_integrated":

        st.warning(
            "Prediction pipeline is not integrated yet. "
            "GPS data is available, but route prediction is pending."
        )

    elif status == "success":

        st.success("Route reconstruction completed.")

    else:

        st.info(f"Prediction status: {status}")


    # ---------------- REUSABLE FOLIUM MAP ----------------

    st.divider()
    st.subheader("GPS Trajectory Map")

    if gps_points:

        track = pd.DataFrame(gps_points)

        required_columns = {
            "latitude",
            "longitude"
        }

        if not required_columns.issubset(track.columns):

            st.warning(
                "GPS response does not contain latitude and longitude."
            )

        else:

            track["latitude"] = pd.to_numeric(
                track["latitude"],
                errors="coerce"
            )

            track["longitude"] = pd.to_numeric(
                track["longitude"],
                errors="coerce"
            )

            track = track.dropna(
                subset=["latitude", "longitude"]
            )

            track = track[
                track["latitude"].between(-90, 90)
                & track["longitude"].between(-180, 180)
            ]

            if not track.empty:

                # Road-network data will be connected
                # after the prediction pipeline integration.

                nodes = pd.DataFrame(
                    columns=[
                        "latitude",
                        "longitude"
                    ]
                )

                arcs = pd.DataFrame(
                    columns=[
                        "from_node_id",
                        "to_node_id"
                    ]
                )

                output = Path(
                    "results/figures/streamlit_trip_map.html"
                )

                output.parent.mkdir(
                    parents=True,
                    exist_ok=True
                )

                try:

                    create_trip_map(
                        track=track,
                        nodes=nodes,
                        arcs=arcs,
                        predicted_arc_ids=predicted_arcs,
                        output=str(output)
                    )

                    html = output.read_text(
                        encoding="utf-8"
                    )

                    components.html(
                        html,
                        height=550,
                        scrolling=False
                    )

                except Exception as exc:

                    st.error(
                        f"Could not generate map: {exc}"
                    )

            else:

                st.warning(
                    "No valid GPS coordinates are available."
                )

    else:

        st.warning(
            "No GPS observations are available."
        )


    # ---------------- GPS DATA ----------------

    with st.expander("View GPS observation data"):

        st.dataframe(
            gps_points,
            use_container_width=True
        )


    # ---------------- RAW API RESPONSE ----------------

    with st.expander("View API response"):

        st.json(result)
