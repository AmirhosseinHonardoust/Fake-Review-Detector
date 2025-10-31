#!/usr/bin/env python3
from __future__ import annotations
import re
from typing import Iterable
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_HTML_RE = re.compile(r"<.*?>")
_NONALPHA = re.compile(r"[^a-zA-Z\s]")
_MULTI_SPACE = re.compile(r"\s+")
def clean_text(s: str) -> str:
    s = s.lower()
    s = _URL_RE.sub(" ", s)
    s = _HTML_RE.sub(" ", s)
    s = _NONALPHA.sub(" ", s)
    s = _MULTI_SPACE.sub(" ", s).strip()
    return s
def batch_clean(texts: Iterable[str]) -> list[str]:
    return [clean_text(t or "") for t in texts]
