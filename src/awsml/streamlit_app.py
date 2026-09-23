"""Interactive app.

A Streamlit app with two panes: a forecast pane that calls the inference API,
and a question-answering pane backed by the RAG endpoint. This is the JD's
"build real-time AI-powered applications using Streamlit".

Run with:

    uv run --extra app streamlit run src/awsml/streamlit_app.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from awsml.config import Settings
from awsml.rag import answer
from awsml.registry import latest
from awsml.train import FEATURES, load_model

st.set_page_config(page_title="AWS AI/ML platform reference", page_icon=":bar_chart:")
st.title("AWS AI/ML platform reference")

settings = Settings.from_env()
tab_forecast, tab_ask = st.tabs(["Demand forecast", "Ask the docs"])

with tab_forecast:
    st.caption("Serves the latest registered model, the same artifact the FastAPI endpoint uses.")
    entry = latest(settings.model_name, settings)
    if entry is None:
        st.warning("No model registered yet. Run `uv run awsml pipeline` first.")
    else:
        st.write(f"Model `{entry['name']}` version {entry['version']}")
        st.json(entry["metrics"])
        model = load_model(__import__("pathlib").Path(entry["artifact"]))
        defaults = {
            "dow": 2,
            "month": 6,
            "is_weekend": 0,
            "promo": 0,
            "units_lag1": 120,
            "units_lag7": 118,
            "units_roll7": 119,
        }
        inputs = {
            feature: st.number_input(feature, value=float(defaults[feature])) for feature in FEATURES
        }
        if st.button("Predict units"):
            prediction = float(model.predict(pd.DataFrame([inputs])[FEATURES])[0])
            st.metric("Predicted units", f"{prediction:,.0f}")

with tab_ask:
    st.caption("Retrieval over the platform docs, with a Bedrock backend on AWS and a local fallback.")
    question = st.text_input("Question", "How does the model registry work?")
    if st.button("Ask"):
        result = answer(question, settings)
        st.write(f"Backend: `{result['backend']}`")
        st.write(result["answer"])
        with st.expander("Sources"):
            for source in result["sources"]:
                st.write(source)
