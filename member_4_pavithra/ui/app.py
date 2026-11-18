
import os
import requests
import streamlit as st


API_URL = os.getenv(
    "PATHSNAP_API_URL",
    "http://127.0.0.1:8000"
)


def get_api_data():
    """Fetch API health status and available trips."""

    health_response = requests.get(
        f"{API_URL}/health",
        timeout=5
    )
    health_response.raise_for_status()

    trips_response = requests.get(
        f"{API_URL}/trips",
        timeout=10
    )
    trips_response.raise_for_status()

    return health_response.json(), trips_response.json()


def predict_route(trip_id):
    """Send the selected trip to the prediction API."""

    response = requests.post(
        f"{API_URL}/predict",
        json={"trip_id": trip_id},
        timeout=180
    )

    response.raise_for_status()
    return response.json()


def show_prediction(result):
    """Display the reconstructed route information."""

    st.success("Route reconstruction completed successfully.")

    gps_column, continuity_column = st.columns(2)

    gps_column.metric(
        "GPS Observations",
        result.get("gps_points", 0)
    )

    continuity_column.metric(
        "Route Continuity",
        f"{result.get('route_continuity', 0):.3f}"
    )

    accuracy = result.get("observation_top1_accuracy")

    if accuracy is not None:
        st.metric(
            "Observation-Level Accuracy",
            f"{accuracy:.3f}"
        )

    st.subheader("Predicted Road Segments")

    st.write(result.get("predicted_arc_ids", []))

    other_results = {
        key: value
        for key, value in result.items()
        if key != "predicted_arc_ids"
    }

    with st.expander("View complete prediction details"):
        st.json(other_results)

    map_location = result.get("map_file")

    if map_location:
        st.info(
            f"Generated map is available on the API computer: "
            f"{map_location}"
        )


def main():

    st.set_page_config(
        page_title="PathSnap",
        layout="wide"
    )

    st.title("PathSnap - GPS Map Matching")

    st.write(
        "Select a trip to reconstruct its route "
        "using the trained map-matching model."
    )

    try:
        health, trip_data = get_api_data()

    except requests.exceptions.RequestException as error:
        st.error(f"Unable to connect to the API: {error}")
        st.stop()

    if not health.get("model_loaded", False):
        st.warning(
            "The trained model is not currently loaded on the API server."
        )

    trip_ids = trip_data.get("trip_ids", [])

    if not trip_ids:
        st.warning(
            trip_data.get(
                "message",
                "No trips are available."
            )
        )
        st.stop()

    selected_trip = st.selectbox(
        "Choose an evaluation trip",
        trip_ids
    )

    if st.button("Reconstruct Route", type="primary"):

        try:
            with st.spinner(
                "Reconstructing the route. Please wait..."
            ):
                prediction = predict_route(selected_trip)

            show_prediction(prediction)

        except requests.exceptions.RequestException as error:
            st.error(f"Prediction request failed: {error}")

        except (KeyError, ValueError, TypeError) as error:
            st.error(f"Unable to display prediction: {error}")


if __name__ == "__main__":
    main()
