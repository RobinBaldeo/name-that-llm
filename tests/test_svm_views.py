import importlib.util
import inspect
from pathlib import Path

import pytest

from ntl.svm import views
from ntl.svm.views import FUNCTION_WORDS, function_word_transform, shape_transform

EXAMPLE = "However, the model—trained on 3 tasks—fails."
REPO = Path(__file__).resolve().parents[1]


def test_shape_spec_example():
    assert shape_transform(EXAMPLE) == "X, x x—x x d x—x."


def test_function_word_spec_example():
    assert function_word_transform(EXAMPLE) == "however , the C — C on C C — C ."


@pytest.mark.parametrize(
    "text, expected",
    [
        ("", ""),
        ("GPT4 scored 92.5%", "Xd x d.d%"),
        ("“Quoted” — **bold**\n\n- item", "“X” — **x**\n\n- x"),
        ("don't", "x'x"),
        ("Ünïcode wörds", "X x"),
    ],
)
def test_shape_keeps_punctuation_and_whitespace(text, expected):
    assert shape_transform(text) == expected


def test_function_words_include_discourse_markers():
    for w in ["however", "moreover", "furthermore", "additionally", "thus",
              "therefore", "indeed", "notably", "essentially", "ultimately", "the", "of"]:
        assert w in FUNCTION_WORDS
    assert function_word_transform("Moreover, Paris is big!") == "moreover , C is C !"


def test_transforms_are_pure():
    assert shape_transform(EXAMPLE) == shape_transform(EXAMPLE)
    assert function_word_transform(EXAMPLE) == function_word_transform(EXAMPLE)


def _load_kaggle_script():
    path = REPO / "kaggle" / "svm_submit.py"
    if not path.exists():
        pytest.skip("kaggle/svm_submit.py not written yet")
    spec = importlib.util.spec_from_file_location("svm_submit", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # module guards its main() behind __name__
    return mod


def test_kaggle_script_transforms_match_package():
    """The Kaggle script inlines its own copy of the transforms; it must not drift."""
    k = _load_kaggle_script()
    samples = [EXAMPLE, "GPT4 scored 92.5%", "“Quoted” — **bold**\n- item", "Moreover, Paris is big!"]
    for s in samples:
        assert k.shape_transform(s) == shape_transform(s)
        assert k.function_word_transform(s) == function_word_transform(s)
    assert k.FUNCTION_WORDS == FUNCTION_WORDS
    assert k.TOKEN_PATTERN == views.TOKEN_PATTERN
