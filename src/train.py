#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, joblib
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, RocCurveDisplay,
                             PrecisionRecallDisplay, roc_auc_score, average_precision_score)
from sklearn.model_selection import train_test_split
from clean_text import batch_clean
from features import extract_numeric_features
def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must include columns: text,label")
    label_map = {"FAKE":1, "fake":1, "REAL":0, "real":0, 1:1, 0:0, "1":1, "0":0, True:1, False:0}
    df["y"] = df["label"].map(label_map)
    if df["y"].isna().any():
        raise ValueError("Labels must be in {FAKE, REAL, 1, 0}.")
    df["text_clean"] = batch_clean(df["text"])
    return df[["text","text_clean","y"]]
def build_pipeline(max_features: int = 20000) -> Pipeline:
    tfidf = TfidfVectorizer(ngram_range=(1,2), max_features=max_features, min_df=2)
    pre = ColumnTransformer([
        ("tfidf", tfidf, "text_clean"),
        ("num", StandardScaler(with_mean=False), ["sentiment","exclamation_count","all_caps_tokens","repeated_phrases","char_length","unique_word_ratio"]),
    ], remainder="drop", sparse_threshold=0.3)
    clf = LogisticRegression(max_iter=300)
    return Pipeline([("pre", pre), ("clf", clf)])
def add_numeric(df: pd.DataFrame) -> pd.DataFrame:
    num = extract_numeric_features(df["text"])
    return pd.concat([df.reset_index(drop=True), num.reset_index(drop=True)], axis=1)
def train(csv: Path, outdir: Path, test_size: float = 0.2, seed: int = 42) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    df = add_numeric(load_data(csv))
    Xtr, Xte, ytr, yte = train_test_split(df.drop(columns=["y","text"]), df["y"].values, test_size=test_size, random_state=seed, stratify=df["y"].values)
    pipe = build_pipeline(); pipe.fit(Xtr, ytr)
    prob = pipe.predict_proba(Xte)[:,1]; pred = (prob>=0.5).astype(int)
    metrics = {"roc_auc": float(roc_auc_score(yte, prob)), "avg_precision": float(average_precision_score(yte, prob)),
               "report": classification_report(yte, pred, target_names=["REAL","FAKE"], output_dict=True)}
    (outdir/"metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    joblib.dump(pipe, outdir/"pipeline.joblib")
    cm = confusion_matrix(yte, pred)
    import matplotlib
    matplotlib.use("Agg")
    fig, ax = plt.subplots(figsize=(4,3)); sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
        xticklabels=["REAL","FAKE"], yticklabels=["REAL","FAKE"])
    ax.set_title("Confusion Matrix"); ax.set_xlabel("Predicted"); ax.set_ylabel("True"); fig.tight_layout()
    fig.savefig(outdir/"confusion_matrix.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(4,3)); RocCurveDisplay.from_predictions(yte, prob, ax=ax); ax.set_title("ROC Curve"); fig.tight_layout()
    fig.savefig(outdir/"roc_curve.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(4,3)); PrecisionRecallDisplay.from_predictions(yte, prob, ax=ax); ax.set_title("Precision-Recall Curve"); fig.tight_layout()
    fig.savefig(outdir/"pr_curve.png", dpi=160); plt.close(fig)
    return metrics
def main():
    ap = argparse.ArgumentParser(description="Train Fake Review Detector on CSV (text,label).")
    ap.add_argument("--csv", default="data/reviews_sample.csv")
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    m = train(Path(args.csv), Path(args.outdir), args.test_size, args.seed)
    print(json.dumps(m, indent=2))
if __name__ == "__main__":
    main()
