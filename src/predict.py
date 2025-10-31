#!/usr/bin/env python3
from __future__ import annotations
import argparse, joblib, json
from pathlib import Path
import pandas as pd
from clean_text import clean_text
from features import extract_numeric_features
def predict_single(pipeline_path: Path, text: str) -> dict:
    pipe = joblib.load(pipeline_path)
    s = clean_text(text)
    df = pd.DataFrame([{"text": text, "text_clean": s}])
    num = extract_numeric_features([text])
    X = pd.concat([df, num], axis=1)
    prob = float(pipe.predict_proba(X)[0,1])
    label = "FAKE" if prob >= 0.5 else "REAL"
    return {"label": label, "fake_probability": prob}
def main():
    ap = argparse.ArgumentParser(description="Predict FAKE/REAL for a review.")
    ap.add_argument("--pipeline", default="outputs/pipeline.joblib")
    ap.add_argument("--text", required=True)
    args = ap.parse_args()
    print(json.dumps(predict_single(Path(args.pipeline), args.text), indent=2))
if __name__ == "__main__":
    main()
