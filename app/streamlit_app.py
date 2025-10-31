#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
import joblib, pandas as pd, streamlit as st
from clean_text import clean_text
from features import extract_numeric_features
PIPE_PATH = Path(__file__).resolve().parents[1] / "outputs" / "pipeline.joblib"
st.set_page_config(page_title="Fake Review Detector", page_icon="🕵️", layout="centered")
st.title("🕵️ Fake Review Detector")
st.caption("TF-IDF + Logistic Regression + Behavioral features")
@st.cache_resource
def load_pipeline():
    if PIPE_PATH.exists(): return joblib.load(PIPE_PATH)
    return None
pipe = load_pipeline()
txt = st.text_area("Paste a product review:", height=200, placeholder="This is the best product ever!!! I got it for free and totally love it...")
thr = st.slider("Decision threshold for FAKE", 0.1, 0.9, 0.5, 0.01)
btn = st.button("Analyze Review", type="primary")
if btn:
    if pipe is None:
        st.error("Model not found. Train it first: `python src/train.py`")
    elif not txt.strip():
        st.warning("Please paste a review.")
    else:
        s = clean_text(txt)
        df = pd.DataFrame([{"text": txt, "text_clean": s}])
        num = extract_numeric_features([txt])
        X = pd.concat([df, num], axis=1)
        prob = float(pipe.predict_proba(X)[0,1])
        label = "FAKE" if prob >= thr else "REAL"
        st.metric("Prediction", label)
        st.progress(prob if label=="FAKE" else 1-prob, text=f"Fake probability: {prob:.1%} | Threshold: {thr:.2f}")
st.info("Tips: Excessive exclamation marks, ALL-CAPS, repeated clichés, and mentions of discounts/free samples can indicate synthetic reviews.", icon="💡")
