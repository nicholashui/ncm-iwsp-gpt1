"""Interpretability and no-training audit for a writer file.

The harness cannot stop a writer from calling whatever it imports. This audit
rejects the patterns that would turn weight-space programming back into
training, and it requires a written description of the weights. Patterns are
matched on code tokens only: comments and string literals do not count.
"""

from __future__ import annotations

import inspect
import io
import tokenize
from pathlib import Path

FORBIDDEN = (
    ".backward(",
    "backward(",
    "loss.backward",
    "torch.optim",
    "optimizer",
    "AdamW",
    "Adam(",
    "optim.Adam",
    "SGD(",
    "sklearn",
    ".fit(",
    "train_tokenizer",
    "train_generator",
    "autograd",
    "from_pretrained",
    "torch.hub",
    "hf_hub_download",
    "snapshot_download",
    "torch.load",
    "pickle.load",
    "np.load",
    "numpy.load",
    "requires_grad_(True)",
    "urllib.request",
    "requests.get",
    "open(",
)

HIDDEN_LEAKS = (
    "hidden_windows",
    "hidden_val",
    "hidden_metrics",
    "hidden_log",
    "hidden_split",
    '["hidden"]',
    "['hidden']",
)

ALLOWED_ARGS = {"tokenizer", "generator", "iteration"}
MIN_DESCRIPTION = 80


class AuditError(Exception):
    def __init__(self, problems):
        self.problems = list(problems)
        super().__init__("; ".join(self.problems))


def _blank(src: str, token_types) -> str:
    lines = src.splitlines(keepends=True)
    starts = [0]
    for line in lines:
        starts.append(starts[-1] + len(line))
    chars = list(src)
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type in token_types:
                begin = starts[tok.start[0] - 1] + tok.start[1]
                end = starts[tok.end[0] - 1] + tok.end[1]
                for index in range(begin, end):
                    if chars[index] != "\n":
                        chars[index] = " "
    except (tokenize.TokenError, IndentationError):
        return src
    return "".join(chars)


def code_only(src: str) -> str:
    """Blank comments and strings so training patterns are matched on code."""
    return _blank(src, (tokenize.COMMENT, tokenize.STRING))


def code_and_strings(src: str) -> str:
    """Blank comments only. A string key such as cache['hidden'] still counts."""
    return _blank(src, (tokenize.COMMENT,))


def audit_source(src: str) -> list[str]:
    code = code_only(src)
    with_strings = code_and_strings(src)
    problems = []
    for pattern in FORBIDDEN:
        if pattern in code:
            problems.append(f"forbidden training or weight-loading pattern: {pattern}")
    for pattern in HIDDEN_LEAKS:
        if pattern in with_strings:
            problems.append(f"writer source names the hidden split: {pattern}")
    return problems


def audit_callable(fn) -> list[str]:
    problems = []
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError) as exc:
        return [f"write_weights has no inspectable signature: {exc}"]
    names = set(signature.parameters)
    extra = names - ALLOWED_ARGS
    missing = {"tokenizer", "generator"} - names
    if missing:
        problems.append(f"write_weights is missing arguments: {sorted(missing)}")
    if extra:
        problems.append(
            f"write_weights accepts {sorted(extra)}; it must not take data, labels, or a hidden split"
        )
    return problems


def audit_description(module) -> list[str]:
    text = getattr(module, "MODEL_DESCRIPTION", "")
    shorthand = getattr(module, "MODEL_SHORTHAND", "")
    problems = []
    if not isinstance(shorthand, str) or len(shorthand.strip()) < 3:
        problems.append("MODEL_SHORTHAND must name this attempt")
    if not isinstance(text, str) or len(text.strip()) < MIN_DESCRIPTION:
        problems.append(
            f"MODEL_DESCRIPTION must explain the weights in at least {MIN_DESCRIPTION} characters"
        )
    return problems


def audit_writer_module(module, path: str | Path) -> list[str]:
    src = Path(path).read_text(encoding="utf-8")
    problems = audit_source(src)
    problems.extend(audit_description(module))
    fn = getattr(module, "write_weights", None)
    if fn is None:
        problems.append("writer module has no write_weights")
    else:
        problems.extend(audit_callable(fn))
    return problems


def state_dict_equal(left, right) -> bool:
    import torch

    left_keys = list(left.state_dict().keys())
    right_keys = list(right.state_dict().keys())
    if left_keys != right_keys:
        return False
    for key in left_keys:
        a = left.state_dict()[key]
        b = right.state_dict()[key]
        if not torch.equal(a, b):
            return False
    return True
