#!/usr/bin/env python3
"""Audit an instruction-tuned decode-side LM-plan splice: no hard-coding."""
from __future__ import annotations

import ast
import re
from pathlib import Path

FORBIDDEN_SUBSTRINGS = (
    "loss.backward",
    ".backward(",
    "torch.optim",
    "AdamW",
    "Adam(",
    "SGD(",
    "sklearn",
    ".fit(",
    "train_tokenizer",
    "train_generator",
    "torch.hub",
    "requires_grad_(True)",
    "requires_grad = True",
)

# Diagnostic window keys, distinctive strings, and particular codes.
# Bare 1 and 2 are allowed as level geometry, not as special-cased codes.
# Do not add a bare digit 5 check; that is too fragile.
FORBIDDEN_PATTERNS = (
    r"\b2574\b",
    r"\b3631\b",
    r"\b281\b",
    r"\b1923\b",
    r"\b820\b",
    r"\b907\b",
    r"\b3260\b",
    r"\b1020\b",
    r"\b1246\b",
    r"\b56\b",
    r"\b137\b",
    r"\bvis13\b",
    r"\b12532\b",
    r"train\[0\]",
    r"\bKaley\b",
    r"\bCuoco\b",
    r"\bFarrah\b",
    r"\bFowler\b",
    r"\bBernadette\b",
    r"\bVirgin\b",
    r"\bAustralia\b",
    r"\bBig Bang\b",
    r"\bAmy Farrah\b",
    r"\blinen\b",
    r"\bFIFA\b",
    r"\bMessi\b",
    r"\bAWS\b",
    r"\b560\b",
    r"\b2619\b",
    r"\b2908\b",
    r"I'm fine",
    r"I am fine",
    r"Im fine",
    r"How are you",
    r"Star Wars",
    r"\bBattlefront\b",
    r"\banime\b",
    r"\bJustice\b",
    r"\b1990s\b",
    r"Amy Pond",
    r"July 1",
    r"1985",
)


def audit_writer(path: Path) -> dict:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    violations = []
    desc = None
    shorthand = None
    try:
        tree = ast.parse(text)
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "MODEL_DESCRIPTION" and isinstance(node.value, ast.Constant):
                        desc = node.value.value
                    if isinstance(target, ast.Name) and target.id == "MODEL_SHORTHAND" and isinstance(node.value, ast.Constant):
                        shorthand = node.value.value
                    if isinstance(target, ast.Name) and target.id == "MODEL_DESCRIPTION" and isinstance(node.value, ast.JoinedStr):
                        desc = ""
    except SyntaxError as exc:
        return {"ok": False, "violations": ["syntax_error: %s" % exc], "description_len": 0, "shorthand": None}

    for i, line in enumerate(text.splitlines(), 1):
        for pat in FORBIDDEN_SUBSTRINGS:
            if pat in line and "MODEL_DESCRIPTION" not in line:
                violations.append("line %d: %s" % (i, pat))
        if ("packed_windows" in line or "visible_train" in line) and "MODEL_DESCRIPTION" not in line:
            violations.append("line %d: splice must not load packed data" % i)
        if "torch.load" in line and "MODEL_DESCRIPTION" not in line:
            violations.append("line %d: splice must not load checkpoints" % i)
        for pat in FORBIDDEN_PATTERNS:
            if re.search(pat, line, flags=re.IGNORECASE):
                violations.append("line %d: hard-code or window special-case (%s)" % (i, pat))

    is_core = path.name.startswith("splice_lm")
    if is_core:
        if not isinstance(desc, str) or len(desc) < 80:
            violations.append("MODEL_DESCRIPTION missing or shorter than 80 characters")
        if not isinstance(shorthand, str) or not shorthand:
            violations.append("MODEL_SHORTHAND missing")
        for fn in (
            "def splice_decoded_ids",
            "def plan_post_prefix_tokens",
            "def official_nonpad_span",
            "def planner_context_ids",
            "def drop_trailing_turn_eos",
            "def build_planner_prompt",
            "def prefix_text_to_messages",
        ):
            if fn not in text:
                violations.append("%s is missing" % fn)
    return {
        "ok": len(violations) == 0,
        "violations": violations,
        "description_len": 0 if not isinstance(desc, str) else len(desc),
        "shorthand": shorthand,
        "path": str(path),
    }
