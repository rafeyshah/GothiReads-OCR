#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Canonical text/font unitization for GothiRead (Track B).

Policy (v2):
- Unicode normalize: NFC
- Whitespace policy: KEEP ordinary spaces as real units, but remove "layout" whitespace:
  newlines + carriage returns + tabs + form-feed + vertical tab.
  Rationale: some GT .txt files contain a single space that is aligned with a single .font label.
  If we delete all \s+, text length can become 0 while font stays 1 (LEN_MISMATCH).

- Text units: Unicode grapheme clusters (regex \\X) AFTER normalization + whitespace policy.
- Font tokens: parse .font into 1 token per grapheme. Supports:
  1) delimiter-separated tokens (comma/semicolon/pipe/tab/newline)
  2) space-separated tokens (only if it matches expected_len)
  3) concatenated single-character labels (fallback)

If you ever change this file, rerun test_units.py on all splits.
"""

from __future__ import annotations

import unicodedata
from typing import List, Optional

import regex as re

_GRAPHEME_RE = re.compile(r"\X", re.U)

# Remove "layout" whitespace but KEEP ordinary spaces.
# - \n newline
# - \r carriage return
# - \t tab
# - \f form feed
# - \v vertical tab
_LAYOUT_WS_RE = re.compile(r"[\r\n\t\f\v]+", flags=re.UNICODE)

# Strong delimiter tokenization (commas/semicolons/pipes/newlines/tabs).
_STRONG_DELIM_SPLIT_RE = re.compile(r"[,\t;\|\r\n]+", flags=re.UNICODE)


def normalize(s: str) -> str:
    """Unicode normalize to NFC (canonical composition)."""
    return unicodedata.normalize("NFC", s)


def strip_ws(s: str) -> str:
    """
    Apply the dataset's whitespace policy.

    v2 policy: remove layout whitespace (\\r, \\n, \\t, \\f, \\v), keep normal spaces.
    """
    return _LAYOUT_WS_RE.sub("", s)


def graphemes(s: str) -> List[str]:
    """
    Split into Unicode grapheme clusters (regex \\X) after normalize + whitespace policy.
    """
    s2 = strip_ws(normalize(s))
    return _GRAPHEME_RE.findall(s2)


def _split_tokens_strong_delims(s: str) -> List[str]:
    # Split on strong delimiters, and drop empty tokens.
    toks = [t for t in _STRONG_DELIM_SPLIT_RE.split(s) if t != ""]
    return toks


def font_tokens(font_str: str, expected_len: Optional[int] = None) -> List[str]:
    """
    Parse .font content into a list of font-label tokens aligned to graphemes.

    Parameters
    ----------
    font_str:
        Raw string content of the .font file.
    expected_len:
        If provided, the tokenizer will prefer interpretations whose token count matches it.

    Returns
    -------
    List[str]: tokens, length ideally == expected_len (if given).

    Notes
    -----
    The dataset can store fonts in multiple formats. This function tries:
      (A) strong-delimiter token list  (comma/semicolon/pipe/tab/newline)
      (B) whitespace-separated tokens  (ONLY if count matches expected_len)
      (C) concatenated labels          (fallback: one char == one label), after whitespace policy
    """
    s = strip_ws(normalize(font_str))

    # A) Strong-delimiter tokens (commas/semicolons/pipes/newlines/tabs)
    strong = _split_tokens_strong_delims(s)
    if strong:
        if expected_len is None or len(strong) == expected_len:
            return strong

    # B) Space-separated tokens (only safe if we have expected_len and it matches)
    # Keep spaces as separators here; do NOT delete them.
    if expected_len is not None:
        space_split = [t for t in s.split(" ") if t != ""]
        if len(space_split) == expected_len:
            return space_split

    # C) Fallback: concatenated labels (ignore spaces? No: if font labels include spaces, treat them as labels.)
    # Most common in your earlier pipeline was list(raw) after removing whitespace.
    # Here we removed layout whitespace but kept spaces. If spaces appear, they become labels too.
    return list(s)
