#!/usr/bin/env python3
from __future__ import annotations
import pandas as pd
from typing import Iterable
from textblob import TextBlob
COMMON_FAKE_PHRASES = ["best product ever","highly recommend","sponsored","discount for review","free sample","life changing","five stars"]
def extract_numeric_features(texts: Iterable[str]) -> pd.DataFrame:
    rows = []
    for s in texts:
        s = str(s or "")
        blob = TextBlob(s)
        rows.append({
            "sentiment": float(blob.sentiment.polarity),
            "exclamation_count": s.count("!"),
            "all_caps_tokens": sum(1 for w in s.split() if len(w)>3 and w.isupper()),
            "repeated_phrases": sum(1 for p in COMMON_FAKE_PHRASES if p in s.lower()),
            "char_length": len(s),
            "unique_word_ratio": (len(set(s.split())) / max(1, len(s.split()))),
        })
    return pd.DataFrame(rows)
