import os

import pandas as pd
import streamlit as st

from regime_detection.monitoring import read_predictions

st.set_page_config(page_title="Regime Monitor", layout="wide")
st.title("Market Regime Monitor")
records = read_predictions(os.getenv("PREDICTION_LOG", "var/predictions.jsonl"))
if not records:
    st.info("No predictions have been recorded yet.")
else:
    frame = pd.DataFrame(records)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    left, right = st.columns(2)
    left.metric("Predictions", len(frame))
    right.metric("Latest regime", frame.iloc[-1]["regime"])
    distribution = frame["regime"].value_counts().rename("count")
    st.subheader("Regime distribution")
    st.bar_chart(distribution)
    st.subheader("Confidence over time")
    st.line_chart(frame.set_index("timestamp")["confidence"])