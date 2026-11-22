"""The 18 class names in a fixed order.

Every score matrix in the project (OOF scores, test scores, blends) uses this column
order. Never reorder it: saved .npy files depend on it.
"""

LABELS: list[str] = [
    "claude-fable-5",
    "claude-opus-4-6",
    "claude-opus-4-8",
    "claude-opus-5",
    "claude-sonnet-5",
    "gemini-3.1-pro-preview",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
    "glm-5.3",
    "glm-5.3-flash",
    "gpt-5.5",
    "gpt-5.6-luna",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-6-astra",
    "kimi-k3",
    "qwen3.8-max",
]

N_CLASSES = len(LABELS)
LABEL_TO_ID: dict[str, int] = {name: i for i, name in enumerate(LABELS)}


def encode(names) -> list[int]:
    """Map label names to integer ids (raises KeyError on an unknown label)."""
    return [LABEL_TO_ID[n] for n in names]


def decode(ids) -> list[str]:
    return [LABELS[int(i)] for i in ids]


def family(name: str) -> str:
    """Model family prefix, e.g. 'gpt-5.6-sol' -> 'gpt', 'qwen3.8-max' -> 'qwen'."""
    head = name.split("-")[0]
    return "".join(ch for ch in head if ch.isalpha())
