"""Topic-free text transforms used as extra TF-IDF views.

Pure functions, no state. `kaggle/svm_submit.py` carries a verbatim copy of these
(it must be self-contained); `tests/test_svm_views.py` checks the two stay identical.
"""

from __future__ import annotations

import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Word-ish tokens or single punctuation chars; also used as the TF-IDF token_pattern
# for the `word` and `func` views so punctuation becomes its own token.
TOKEN_PATTERN = r"(?u)\b\w+\b|[^\w\s]"
_TOKEN_RE = re.compile(TOKEN_PATTERN)
_WORD_RE = re.compile(r"\w+")

# Digit runs (group 1) or letter runs (group 2). Underscores and everything else are kept as-is.
_SHAPE_RE = re.compile(r"(\d+)|([^\W\d_]+)")

DISCOURSE_MARKERS = frozenset(
    {
        "however", "moreover", "furthermore", "additionally", "thus",
        "therefore", "indeed", "notably", "essentially", "ultimately",
    }
)
FUNCTION_WORDS = frozenset(ENGLISH_STOP_WORDS) | DISCOURSE_MARKERS


def _shape_token(m: re.Match) -> str:
    if m.group(1) is not None:
        return "d"
    return "X" if m.group(2)[0].isupper() else "x"


def shape_transform(text: str) -> str:
    """Word -> X (starts uppercase) / x, digit run -> d; punctuation & whitespace untouched.

    'However, the model—trained on 3 tasks—fails.' -> 'X, x x—x x d x—x.'
    """
    return _SHAPE_RE.sub(_shape_token, text)


def function_word_transform(text: str) -> str:
    """Lowercase; keep function words and punctuation tokens, replace other tokens with C.

    'However, the model—trained on 3 tasks—fails.' -> 'however , the C — C on C C — C .'
    """
    out = []
    for tok in _TOKEN_RE.findall(text.lower()):
        if tok in FUNCTION_WORDS or not _WORD_RE.fullmatch(tok):
            out.append(tok)
        else:
            out.append("C")
    return " ".join(out)


def raw_transform(text: str) -> str:
    return text


TRANSFORMS = {
    "raw": raw_transform,
    "shape": shape_transform,
    "func": function_word_transform,
}


def apply_transform(texts, transform: str) -> list[str]:
    """Batch version (module-level so fitted pipelines pickle cleanly)."""
    fn = TRANSFORMS[transform]
    return [fn(t) for t in texts]
