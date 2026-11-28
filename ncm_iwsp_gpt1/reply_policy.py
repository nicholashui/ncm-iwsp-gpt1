"""General reply policy for the decode-side planner splice.

The frozen generator is not modified. These helpers run on the planner text
before it is written into the reconstructed window. They are category rules,
not a table of questions and answers.
"""

from __future__ import annotations

import re

WINDOW_TAIL_RESERVE = 32
MAX_REPLY_SENTENCES = 2
DEFAULT_WINDOW = 512

POLICY_SYSTEM = (
    "Answer only the latest user message. Keep the reply to one or two short sentences "
    "and stop when that reply is complete. Do not write another user or assistant turn. "
    "Do not write a story, a script, a poem, or a customer-support script. "
    "Do not repeat the user's wording. Do not invent a time, a place, or an appointment. "
    "If you give a preference, say that it is your opinion. "
    "If the question lacks the detail needed to answer, ask one clarifying question."
)

_TURN_MARK = r"(?:<\|endoftext\|>|<\|im_end\|>|</s>|<\|eot_id\|>)"
_ROLE_MARK = r"(?:User|Assistant|System)\s*:"
_LEADING_ROLE = re.compile(rf"^(?:{_ROLE_MARK})\s*", re.IGNORECASE)
_TURN_BOUNDARY = re.compile(
    rf"(?:{_TURN_MARK})|(?:^|\n)\s*(?:{_ROLE_MARK})",
    re.IGNORECASE,
)
_ANY_ROLE = re.compile(rf"(?:{_ROLE_MARK})\s*", re.IGNORECASE)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

_FUNCTION = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "do",
        "does",
        "did",
        "can",
        "could",
        "should",
        "would",
        "will",
        "i",
        "you",
        "we",
        "they",
        "he",
        "she",
        "it",
        "this",
        "that",
        "these",
        "those",
        "one",
        "ones",
        "me",
        "my",
        "your",
        "our",
        "their",
        "what",
        "which",
        "who",
        "whom",
        "when",
        "where",
        "why",
        "how",
        "about",
        "of",
        "to",
        "for",
        "and",
        "or",
        "if",
        "please",
        "just",
        "something",
        "anything",
        "help",
        "with",
        "on",
        "in",
        "at",
        "be",
        "am",
        "been",
        "being",
        "better",
        "best",
        "good",
        "bad",
        "ok",
        "okay",
        "much",
        "many",
        "thing",
        "stuff",
        "there",
        "here",
        "now",
        "tell",
        "say",
        "ask",
        "want",
        "need",
        "know",
        "think",
        "like",
        "really",
        "very",
        "some",
        "any",
        "not",
        "than",
        "then",
        "from",
        "into",
        "over",
        "under",
        "again",
        "also",
        "too",
        "so",
        "as",
        "by",
        "up",
        "out",
        "off",
        "us",
        "im",
        "dont",
        "its",
        "let",
        "lets",
    }
)

_STORY_OPENERS = re.compile(
    r"\b(?:once upon a time|in a land far away|it was a dark and stormy night|long ago|chapter \d+)\b",
    re.IGNORECASE,
)
_SCRIPT_SLUG = re.compile(r"\b(?:INT\.|EXT\.|FADE IN|FADE OUT|CUT TO:)\b")
_SPEAKER_LINE = re.compile(r"^[A-Z][A-Z0-9 .'\-]{1,24}:\s+\S", re.MULTILINE)
_HELPDESK = re.compile(
    r"\b(?:thank you for contacting|your ticket number|we apologize for the inconvenience|"
    r"please do not reply to this (?:message|email)|your call is important to us|"
    r"press \d+ (?:for|to))\b",
    re.IGNORECASE,
)
_SAID = re.compile(r"\b(?:said|whispered|replied)\b", re.IGNORECASE)

_RECIPROCAL = re.compile(
    r"\b(?:how about you|what about you|how about yourself|what about yourself|and you)\b",
    re.IGNORECASE,
)
_SELF_STATUS = re.compile(
    r"\b(?:i(?:'m| am|’m)\s+(?:doing|good|well|okay|ok|great|alright|fine)|"
    r"i feel\s+(?:good|well|okay|great)|as for me)\b",
    re.IGNORECASE,
)

_CLOCK = re.compile(r"\b\d{1,2}:\d{2}\b|\b\d{1,2}\s*(?:a\.?m\.?|p\.?m\.?)\b", re.IGNORECASE)
_WEEKDAY = re.compile(
    r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    re.IGNORECASE,
)
_MONTH_DAY = re.compile(
    r"\b(?:january|february|march|april|may|june|july|august|september|"
    r"october|november|december)\s+\d{1,2}\b",
    re.IGNORECASE,
)
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_APPOINTMENT = re.compile(
    r"\b(?:appointment|reservation|let's meet|lets meet|see you at|meet me at)\b",
    re.IGNORECASE,
)
_PLACE = re.compile(
    r"\b(?:at|in)\s+the\s+(?:office|cafe|restaurant|station|airport|hotel|park|"
    r"mall|hospital|school|library|house|apartment|downtown)\b"
    r"|\b(?:at|in)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b"
)

_OPINION_ASK = re.compile(
    r"\b(?:do you think|what do you think|in your opinion|your opinion|"
    r"what(?:'s| is) (?:the )?best|which (?:is|are) (?:better|best)|"
    r"should i|would you (?:recommend|rather|prefer)|do you prefer|"
    r"what(?:'s| is) your favorite|which do you (?:like|prefer))\b",
    re.IGNORECASE,
)
_OPINION_MARK = re.compile(
    r"\b(?:in my opinion|i think|i'd say|i would say|personally|my view)\b",
    re.IGNORECASE,
)

_WEAPON = re.compile(
    r"\b(?:(?:make|build|assemble|print|construct).{0,48}(?:bomb|explosive|grenade|firearm|gun|weapon)|"
    r"how (?:do i|can i|to).{0,48}(?:bomb|explosive|firearm|gun|weapon|silencer|grenade))\b",
    re.IGNORECASE,
)
_HACK = re.compile(
    r"\b(?:hack(?:ing)? into|unauthorized access|"
    r"(?:write|make|build) (?:a )?(?:malware|ransomware|keylogger|exploit)|"
    r"how (?:do i|can i|to) (?:hack|exploit)|"
    r"break into (?:a |an |their |his |her )?(?:account|mailbox|computer|server|network|system))\b",
    re.IGNORECASE,
)
_PHISH = re.compile(
    r"\b(?:phish(?:ing)?|fake login|credential harvest|"
    r"steal (?:a )?(?:password|passwords|credentials)|"
    r"spoof(?:ed)? (?:an? )?(?:email|site|page|login))\b",
    re.IGNORECASE,
)
_POISON = re.compile(
    r"\b(?:how (?:do i|can i|to) poison|poison (?:someone|him|her|them|people)|"
    r"(?:make|mix|use) (?:a )?poison|lethal dose|"
    r"poison\b.{0,48}\b(?:someone|him|her|them|people))\b",
    re.IGNORECASE,
)
_DRUGS = re.compile(
    r"\b(?:leftover (?:prescription|antibiotic|medication|medicine|pills|antibiotics)|"
    r"(?:someone else'?s|another person'?s|a friend'?s) "
    r"(?:prescription|medication|medicine|pills|antibiotics)|"
    r"(?:take|use|share).{0,40}(?:leftover|expired).{0,24}"
    r"(?:pill|pills|medicine|medication|prescription|antibiotic|antibiotics))\b",
    re.IGNORECASE,
)
_FINANCE = re.compile(
    r"\b(?:all[\s-]?in|bet everything|yolo (?:my )?(?:savings|paycheck|money|account)|"
    r"guaranteed (?:returns|profit|double)|"
    r"(?:entire|all|whole) (?:of )?(?:my |the )?(?:life )?savings|"
    r"put (?:all|everything) (?:of )?(?:my )?(?:money|savings))\b",
    re.IGNORECASE,
)
_FINANCE_RISK = re.compile(
    r"\b(?:bet|all[\s-]?in|crypto|coin|stock|option|leverage|gamble|yolo|double)\b",
    re.IGNORECASE,
)
_SIMILAR_HARM = re.compile(
    r"\b(?:how (?:do i|can i|to) (?:scam|defraud|commit fraud)|"
    r"how (?:do i|can i|to) (?:stab|shoot) (?:someone|people|him|her|them))\b",
    re.IGNORECASE,
)

_REFUSALS = {
    "weapons": (
        "I can't help with weapons or explosives. If you are worried about safety, "
        "contact local emergency services."
    ),
    "hacking": (
        "I can't help with unauthorized access or malware. I can discuss defending "
        "your own systems at a high level instead."
    ),
    "phishing": (
        "I can't help with phishing or stealing credentials. I can explain how to "
        "recognize and report a scam."
    ),
    "poisons": (
        "I can't help with poisons or harming someone. If someone may have been exposed, "
        "contact poison control or emergency services."
    ),
    "prescription": (
        "Don't use leftover or someone else's prescription drugs. Ask a pharmacist or "
        "the prescribing clinician what to do with them."
    ),
    "finance": (
        "I can't tell you to put everything into one bet. Avoid all-in bets and talk "
        "with a qualified adviser."
    ),
    "harm": "I can't help with harming someone or with fraud. I won't give steps for that.",
}

_ECHO_FALLBACK = "I can answer that in my own words."
_SLOT_FALLBACK = "I won't add a time, place, or appointment you didn't mention."
_RECIPROCAL_FALLBACK = "I'll leave that for you to answer."


def content_words(text: str) -> list[str]:
    words = re.findall(r"[A-Za-z0-9']+", (text or "").lower())
    return [word for word in words if word not in _FUNCTION and len(word) > 1]


def _tidy(text: str) -> str:
    raw = "" if text is None else str(text)
    raw = raw.replace("\r\n", "\n")
    raw = re.sub(r"[ \t]+", " ", raw)
    raw = re.sub(r" *\n *", "\n", raw)
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    raw = re.sub(r"\s+([.!?])", r"\1", raw)
    raw = re.sub(r" ([,;])", r"\1", raw)
    lines = [line.strip() for line in raw.splitlines()]
    return "\n".join(line for line in lines if line).strip(" \t-;:,")


def _body_without_leading_role(text: str) -> str:
    return _LEADING_ROLE.sub("", (text or "").strip(), count=1)


def _sentences(text: str) -> list[str]:
    raw = " ".join((text or "").split())
    if not raw:
        return []
    return [part.strip() for part in _SENTENCE_SPLIT.split(raw) if part.strip()]


def cut_to_short_turn(text: str) -> str:
    """Keep the first complete assistant turn, at most two sentences."""
    raw = _body_without_leading_role("" if text is None else str(text))
    match = _TURN_BOUNDARY.search(raw)
    if match:
        raw = raw[: match.start()]
    parts = _sentences(raw)
    if not parts:
        return ""
    return " ".join(parts[:MAX_REPLY_SENTENCES]).strip()


def generation_should_stop(text: str) -> bool:
    """True at end-of-turn, a later role label, or two finished sentences."""
    body = _body_without_leading_role(text)
    if _TURN_BOUNDARY.search(body):
        return True
    complete = [part for part in _sentences(body) if part.endswith((".", "!", "?"))]
    return len(complete) >= MAX_REPLY_SENTENCES


def planner_token_action(before: str, after: str) -> str:
    """How to treat one newly sampled planner token.

    ``drop`` means the token starts a later turn. ``stop`` means the token
    completes a short reply and should be kept. ``keep`` means sampling continues.
    """
    before_body = _body_without_leading_role(before)
    after_body = _body_without_leading_role(after)
    if _TURN_BOUNDARY.search(after_body) and not _TURN_BOUNDARY.search(before_body):
        return "drop"
    if generation_should_stop(after):
        return "stop"
    return "keep"


def detect_dump(text: str) -> str | None:
    """Return story, script, poem, or helpdesk when the planner dumped one."""
    raw = "" if text is None else str(text).strip()
    if not raw:
        return None
    if _HELPDESK.search(raw):
        return "helpdesk"
    if _SCRIPT_SLUG.search(raw) or len(_SPEAKER_LINE.findall(raw)) >= 3:
        return "script"
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if len(lines) >= 4 and sum(1 for line in lines if len(line) <= 48) >= 4:
        if max(len(line) for line in lines) <= 72:
            return "poem"
    if _STORY_OPENERS.search(raw):
        return "story"
    sentences = _sentences(raw)
    if len(sentences) >= 6 and (len(_SAID.findall(raw)) >= 3 or raw.count('"') >= 6):
        return "story"
    return None


def rewrite_short_conversational(user_text: str) -> str:
    words = content_words(user_text)[:4]
    if words:
        return "I'll skip the long version and keep to a short reply about " + " ".join(words) + "."
    return "I'll skip the long version and keep the reply short."


def strip_role_leakage(text: str) -> str:
    cleaned = _ANY_ROLE.sub(" ", "" if text is None else str(text))
    return _tidy(cleaned)


def _word_tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", (text or "").lower())


def strip_user_echo(reply: str, user_text: str) -> str:
    """Drop a verbatim copy of the user wording from the reply."""
    raw = "" if reply is None else str(reply)
    user_words = _word_tokens(user_text)
    phrases = []
    if len(user_words) >= 4:
        phrases.append(user_words)
    if len(user_words) >= 6:
        for size in range(len(user_words), 5, -1):
            for start in range(0, len(user_words) - size + 1):
                phrases.append(user_words[start : start + size])
    seen = set()
    for phrase in sorted(phrases, key=len, reverse=True):
        key = tuple(phrase)
        if key in seen:
            continue
        seen.add(key)
        pattern = r"\b" + r"\W+".join(re.escape(word) for word in phrase) + r"\b"
        raw = re.sub(pattern, " ", raw, flags=re.IGNORECASE)
    user_content = set(content_words(user_text or ""))
    kept = []
    for sentence in _sentences(raw):
        words = content_words(sentence)
        if len(words) >= 4 and user_content:
            overlap = sum(1 for word in words if word in user_content)
            novel = sum(1 for word in words if word not in user_content)
            # A sentence that only repeats the user, with no new content word, is an echo.
            # An answer that adds even one new word (such as a result) stays.
            if overlap / len(words) >= 0.75 and novel == 0:
                continue
        kept.append(sentence)
    text = _tidy(" ".join(kept))
    if not text:
        return _ECHO_FALLBACK
    return text


def strip_self_answered_followup(reply: str, user_text: str) -> str:
    """Do not answer a reciprocal 'how about you' inside this same turn."""
    raw = "" if reply is None else str(reply)
    user = "" if user_text is None else str(user_text)
    match = _RECIPROCAL.search(raw)
    if match:
        raw = raw[: match.start()]
    user_asks = _RECIPROCAL.search(user) is not None
    if user_asks:
        raw = " ".join(sentence for sentence in _sentences(raw) if not _SELF_STATUS.search(sentence))
    text = _tidy(raw)
    if not text and (user_asks or match):
        return _RECIPROCAL_FALLBACK
    return text


def _norm_slot(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def _grounded(span: str, user_text: str) -> bool:
    needle = _norm_slot(span)
    return bool(needle) and needle in _norm_slot(user_text)


def block_invented_slots(reply: str, user_text: str) -> str:
    """Drop clauses that introduce a time, place, or appointment the user did not state."""
    user = "" if user_text is None else str(user_text)
    kept = []
    removed = False
    for sentence in _sentences(reply):
        spans = []
        for pattern in (_CLOCK, _WEEKDAY, _MONTH_DAY, _YEAR, _APPOINTMENT, _PLACE):
            spans.extend(match.group(0) for match in pattern.finditer(sentence))
        if any(not _grounded(span, user) for span in spans):
            removed = True
            continue
        kept.append(sentence)
    text = _tidy(" ".join(kept))
    if removed and not text:
        return _SLOT_FALLBACK
    return text


def safety_decision(user_text: str) -> dict | None:
    """Refuse or redirect a harmful or high-risk request. None means the text may be answered."""
    raw = "" if user_text is None else str(user_text)
    checks = (
        ("weapons", _WEAPON),
        ("hacking", _HACK),
        ("phishing", _PHISH),
        ("poisons", _POISON),
        ("prescription", _DRUGS),
        ("harm", _SIMILAR_HARM),
    )
    for category, pattern in checks:
        if pattern.search(raw):
            return {"category": category, "reply": _REFUSALS[category]}
    if _FINANCE.search(raw) and (_FINANCE_RISK.search(raw) or re.search(r"\bsavings\b", raw, re.I)):
        return {"category": "finance", "reply": _REFUSALS["finance"]}
    return None


def is_underspecified(user_text: str, history_text: str = "") -> bool:
    """True when the current question has no concrete topic and history does not supply one."""
    if content_words(user_text):
        return False
    if history_text and content_words(history_text):
        return False
    return True


def clarifying_question(user_text: str) -> str:
    low = (user_text or "").lower()
    if re.search(r"\bwhich\b", low):
        return "Which options are you deciding between?"
    if re.search(r"\bhow (?:much|many)\b", low):
        return "How much of what should I estimate?"
    if re.search(r"\bwhen\b", low):
        return "What event should I time this against?"
    if re.search(r"\bwhere\b", low):
        return "Where is this happening?"
    return "What exactly should I answer?"


def asks_for_opinion(user_text: str) -> bool:
    return _OPINION_ASK.search(user_text or "") is not None


def mark_opinion(reply: str, user_text: str) -> str:
    if not asks_for_opinion(user_text):
        return "" if reply is None else str(reply).strip()
    body = ("" if reply is None else str(reply)).strip()
    if _OPINION_MARK.search(body):
        return body
    if not body:
        return "In my opinion, that depends."
    if body[0].isupper() and not body.startswith(("I ", "I'")):
        body = body[0].lower() + body[1:]
    return "In my opinion, " + body


def assistant_body_budget(
    seq_len: int,
    n_cond: int,
    n_open: int,
    reserve: int = WINDOW_TAIL_RESERVE,
) -> int:
    """Tokens left for the assistant body after the prefix, the opening, and the reserved tail."""
    room = int(seq_len) - int(n_cond) - int(n_open) - int(reserve)
    return max(0, room)


def fit_reply_to_budget(text: str, encode, decode, budget: int) -> str:
    """Shorten ``text`` so it fits in ``budget`` tokens and ends on a clause boundary."""
    limit = int(budget)
    cleaned = " ".join((text or "").split())
    if limit <= 0 or not cleaned:
        return ""
    ids = list(encode(cleaned) or [])
    if len(ids) <= limit:
        return cleaned
    best = ""
    for end in range(1, limit + 1):
        piece = decode(ids[:end])
        piece = "" if piece is None else str(piece).rstrip()
        if piece and piece[-1] in ".!?;,":
            best = piece
    if not best:
        return ""
    if len(list(encode(best) or [])) <= limit:
        return best
    return ""


def with_policy_messages(messages) -> list[dict]:
    """Attach the reply policy as a system message without dropping the latest user turn."""
    items = [dict(item) for item in (messages or [])]
    if any(item.get("role") == "system" and POLICY_SYSTEM in str(item.get("content") or "") for item in items):
        return items
    systems = [item for item in items if item.get("role") == "system"]
    rest = [item for item in items if item.get("role") != "system"]
    if systems:
        merged = str(systems[0].get("content") or "").rstrip() + "\n" + POLICY_SYSTEM
        return [{"role": "system", "content": merged}] + rest
    return [{"role": "system", "content": POLICY_SYSTEM}] + rest


def prepare_assistant_body(
    user_text: str,
    planned_text: str,
    *,
    seq_len: int = DEFAULT_WINDOW,
    n_cond: int = 0,
    n_open: int = 0,
    encode=None,
    decode=None,
    history_text: str = "",
    reserve: int = WINDOW_TAIL_RESERVE,
) -> dict:
    """Apply the reply policy and, when an encoder is given, fit the window budget."""
    user = "" if user_text is None else str(user_text).strip()
    planned = "" if planned_text is None else str(planned_text)
    decision = safety_decision(user)
    refused = decision is not None
    clarifying = False
    rewritten = False
    if refused:
        text = decision["reply"]
    elif is_underspecified(user, history_text=history_text):
        text = clarifying_question(user)
        clarifying = True
    else:
        if detect_dump(planned):
            text = rewrite_short_conversational(user)
            rewritten = True
        else:
            text = cut_to_short_turn(planned)
        text = strip_role_leakage(text)
        text = strip_user_echo(text, user)
        text = strip_self_answered_followup(text, user)
        text = block_invented_slots(text, user)
        if not rewritten:
            text = mark_opinion(text, user)
    if encode is not None and decode is not None:
        budget = assistant_body_budget(seq_len, n_cond, n_open, reserve)
        text = fit_reply_to_budget(text, encode, decode, budget)
    return {
        "text": text.strip(),
        "refused": refused,
        "clarifying": clarifying,
        "rewritten": rewritten,
    }
