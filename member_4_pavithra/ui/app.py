import os, requests, streamlit as st
API=os.getenv("PATHSNAP_API_URL","http://127.0.0.1:8000")
st.set_page_config(page_title="PathSnap Evaluator",layout="wide")
st.title("PathSnap — Route Reconstruction")
try:
    ids=requests.get(f"{API}/trips",timeout=5).json()["trip_ids"]
except Exception as exc:
    st.error(f"Cannot reach API at {API}: {exc}"); st.stop()
if not ids: st.warning("No evaluation trip IDs are configured on the API server."); st.stop()
trip=st.selectbox("Select an evaluation trip",ids)
if st.button("Reconstruct route"):
    try:
        result=requests.post(f"{API}/predict",json={"trip_id":trip},timeout=120).json()
        st.json(result)
        if result.get("status")=="pipeline_not_integrated":
            st.info("Starter API is reachable; connect the completed model + decoder before evaluation.")
    except Exception as exc: st.error(str(exc))
