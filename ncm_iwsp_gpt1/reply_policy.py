"""General reply policy for the decode-side planner splice.

The frozen generator is not modified. These helpers run on the planner text
before it is written into the reconstructed window. They are category rules,
not a table of questions and answers.
"""

from __future__ import annotations

import re

WINDOW_TAIL_RESERVE = 32
MAX_REPLY_SENTENCES = 2
MAX_HOWTO_SENTENCES = 4
DEFAULT_WINDOW = 512

POLICY_SYSTEM = (
    "Answer only the latest user message. "
    "A greeting, a short status check, thanks, or a goodbye gets one short social reply. "
    "Do not greet someone who is saying goodbye, and do not invent an earlier chat. "
    "Keep an ordinary reply to one or two short sentences. "
    "If the user asks how to do a legitimate task, give at most four short ordered steps and then stop. "
    "Do not write another user or assistant turn. "
    "Do not write a story, a script, a poem, or a customer-support script. "
    "Do not repeat the user's wording as the whole reply, and do not leave a bare lead-in. "
    "Do not invent a time, a place, or an appointment, and do not mention that rule in the reply. "
    "If you give a preference, say that it is your opinion. "
    "Ask one clarifying question only when the user gave no concrete topic. "
    "If the question is already clear, answer it. "
    "Do not give steps for a crime or for harming someone. "
    "For chest pain or another possible emergency, tell the user to contact emergency services or urgent care. "
    "Do not tell anyone to put all of their money into one bet."
)

_TURN_MARK = r"(?:<\|endoftext\|>|<\|im_end\|>|</s>|<\|eot_id\|>)"
_ROLE_MARK = r"(?:User|Assistant|System)\s*:"
_LEADING_ROLE = re.compile(rf"^(?:{_ROLE_MARK})\s*", re.IGNORECASE)
_TURN_BOUNDARY = re.compile(
    rf"(?:{_TURN_MARK})|(?:^|\n)\s*(?:{_ROLE_MARK})",
    re.IGNORECASE,
)
_ANY_ROLE = re.compile(rf"(?:{_ROLE_MARK})\s*", re.IGNORECASE)
# A list marker such as "1." is not the end of a sentence.
_SENTENCE_SPLIT = re.compile(r"(?<!\d[.!?])(?<=[.!?])\s+")

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
    r"put (?:all|everything|every (?:dollar|cent)) (?:of )?(?:my )?(?:money|savings|paycheck|cash)|"
    r"everything i (?:have|own)\s+(?:into|on|in))\b",
    re.IGNORECASE,
)
_FINANCE_RISK = re.compile(
    r"\b(?:bet|all[\s-]?in|crypto(?:currency)?|bitcoin|btc|ethereum|"
    r"coin|token|stocks?|options?|leverage|gambl\w*|yolo|double|forex|margin)\b",
    re.IGNORECASE,
)
_LOCK = re.compile(
    r"\b(?:pick|jimmy|shim|bump|rake) (?:a |the |my |this )?(?:lock|padlock|deadbolt)\b"
    r"|\block\s*pick(?:ing)?\b"
    r"|\bbump key\b"
    r"|\bbypass (?:a |the )?(?:lock|deadbolt)\b",
    re.IGNORECASE,
)
_SHOPLIFT = re.compile(
    r"\b(?:how (?:do i|can i|to)|steps to|ways to|help me|show me how to)\s+"
    r"shoplift\b"
    r"|\bshoplift(?:ing)? from\b"
    r"|\bsteal from (?:a |the )?(?:store|shop|mall|retailer|market)\b"
    r"|\bshoplift\b.{0,48}\bwithout (?:getting|being) caught\b"
    r"|\bwithout (?:getting|being) caught\b.{0,48}\bshoplift\b",
    re.IGNORECASE,
)
_STALK = re.compile(
    r"\bstalk (?:my |a |the )?(?:ex|coworker|co-worker|neighbour|neighbor|someone|person|him|her|them)\b"
    r"|\b(?:secretly follow|secretly track|spy on) (?:my |a |the )?(?:ex|partner|someone|person|him|her|them)\b"
    r"|\bfollow (?:someone|him|her|them|my ex) (?:home|around)\b"
    r"|\btrack (?:someone|him|her|them|my ex) without\b",
    re.IGNORECASE,
)
_THREAT = re.compile(
    r"\b(?:make|send|write|deliver) (?:an? )?anonymous (?:death )?threat\b"
    r"|\b(?:how (?:do i|can i|to)|steps to|ways to|help me|show me how to)\s+"
    r"send (?:an? )?(?:anonymous )?(?:death )?threat\b"
    r"|\banonymous(?:ly)?\s+threaten(?:ing)?\b"
    r"|\bthreaten (?:someone|him|her|them|people) anonymously\b",
    re.IGNORECASE,
)
_VANDAL = re.compile(
    r"\b(?:how (?:do i|can i|to)|steps to|ways to|help me|show me how to)\s+"
    r"(?:vandal(?:i[sz]e)|graffiti)\b"
    r"|\bvandali[sz]e (?:a |someone(?:'s)? |their |his |her |the )"
    r"|\bkey (?:his|her|their|someone(?:'s)?|a) (?:car|truck)\b"
    r"|\bsmash (?:his|her|their|someone(?:'s)?|a|the) (?:car|window|house|shop)\b",
    re.IGNORECASE,
)
_FAKE_CRED = re.compile(
    r"\b(?:fake|forged|counterfeit|phony|falsified)\s+"
    r"(?:diploma|degree|credential|certificate|transcript|license)s?\b"
    r"|\b(?:forge|counterfeit|falsify)\s+(?:a |my )?"
    r"(?:diploma|degree|credential|certificate|transcript|license)\b"
    r"|\bbuy (?:a )?(?:fake|forged)\s+(?:diploma|degree|certificate)\b",
    re.IGNORECASE,
)
_CRED_DEFENSE = re.compile(
    r"\b(?:spot|detect|recognize|identify|report|avoid|check|tell if)\b.{0,48}\bfake\b",
    re.IGNORECASE,
)
_MEDICAL = re.compile(
    r"\b(?:chest pain|chest pressure|pain in (?:my |the )?chest|"
    r"tightness in (?:my |the )?chest)\b"
    r"|\b(?:i(?:'m| am)|i have|i(?:'ve| have) got|i feel)\b.{0,48}\b(?:"
    r"can(?:not|'t) breathe|trouble breathing|difficulty breathing|short(?:ness)? of breath|"
    r"heart attack|stroke|severe bleeding|unconscious|unresponsive|overdos(?:e|ing)|suicidal)\b"
    r"|\b(?:can(?:not|'t) breathe|cannot breathe|not breathing|kill myself|suicidal)\b",
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
    "lockpicking": (
        "I can't help with lockpicking. A locksmith can help with a lock you are allowed to open."
    ),
    "shoplifting": "I can't help with shoplifting or stealing. I won't give steps for that.",
    "stalking": (
        "I can't help with stalking or following someone. If you feel unsafe, contact local authorities."
    ),
    "threats": "I can't help with threats. If you are in danger, contact emergency services.",
    "vandalism": "I can't help with vandalism or damaging property. I won't give steps for that.",
    "credentials": (
        "I can't help with fake diplomas or forged credentials. Use an official school or licensing body."
    ),
    "medical": (
        "That can be an emergency. Contact emergency services or urgent care now "
        "instead of waiting on a chat reply."
    ),
}

_ECHO_FALLBACK = "I can answer that in my own words."
_SLOT_FALLBACK = "We can talk about that."
_RECIPROCAL_FALLBACK = "I'll leave that for you to answer."
_HOWTO_STUB_FALLBACK = "I can give a few short steps in order."

_SOCIAL_LEXICON = frozenset(
    {
        "hello",
        "hi",
        "hey",
        "hiya",
        "howdy",
        "yo",
        "greetings",
        "there",
        "everyone",
        "friend",
        "goodbye",
        "bye",
        "farewell",
        "ciao",
        "cya",
        "ttyl",
        "cheers",
        "thanks",
        "thank",
        "thx",
        "appreciated",
        "appreciate",
        "morning",
        "afternoon",
        "evening",
        "night",
        "day",
        "weekend",
        "later",
        "soon",
        "around",
        "care",
        "take",
        "catch",
        "today",
        "tonight",
        "lately",
        "doing",
        "going",
        "been",
        "feeling",
        "good",
        "great",
        "nice",
        "well",
        "okay",
        "ok",
        "alright",
        "up",
        "wassup",
        "sup",
        "gotta",
        "heading",
        "leaving",
        "done",
        "off",
    }
)
_SOCIAL_REPLIES = {
    "greeting": "Hello.",
    "wellbeing": "I'm doing well, thanks for asking.",
    "thanks": "You're welcome.",
    "goodbye": "Goodbye. Take care.",
    "thanks_bye": "You're welcome. Take care.",
}
_REQUESTISH = frozenset(
    {"help", "explain", "tell", "show", "mean", "think", "can", "could", "would", "should", "please"}
)
_GOODBYE_CLAUSE = re.compile(
    r"^(?:good\s*bye|goodbye|bye+|bye[\s-]*bye|farewell|"
    r"see (?:you|ya)(?: later| soon| around)?|"
    r"catch you later|take care|cya|ttyl|"
    r"(?:i(?:'m| am)|im) (?:off|out|leaving|done|heading out)|"
    r"gotta go|(?:i have to|i need to) go|"
    r"have a (?:good|nice|great|wonderful) (?:day|night|one|evening|weekend)|"
    r"good\s*night)$",
    re.IGNORECASE,
)
_THANKS_CLAUSE = re.compile(
    r"^(?:thanks|thank you|thank u|ty|thx|cheers|much appreciated|appreciate it)"
    r"(?:\s+(?:so much|a lot|very much|again))?$",
    re.IGNORECASE,
)
_GREETING_CLAUSE = re.compile(
    r"^(?:hi+|hello+|hey+|hiya|howdy|yo|greetings|"
    r"good\s+(?:morning|afternoon|evening|day))"
    r"(?:\s+(?:there|everyone|all|friend))?$",
    re.IGNORECASE,
)
_GREETING_PREFIX = re.compile(
    r"^(?:hi+|hello+|hey+|hiya|howdy|yo|greetings|"
    r"good\s+(?:morning|afternoon|evening|day))"
    r"(?:\s+(?:there|everyone|all|friend))?[\s,!-]*",
    re.IGNORECASE,
)
_HOWTO = re.compile(
    r"\b(?:how (?:do i|can i|to|should i)|steps (?:to|for)|"
    r"(?:a |the )?(?:way|process|method) to|show me how|walk me through)\b",
    re.IGNORECASE,
)
_STEP_LINE = re.compile(
    r"^(?:(?:step\s*)?\d+[.)]|[-*•]|first|second|third|fourth|fifth|next|then|finally)\b",
    re.IGNORECASE,
)
_FRAGMENT = re.compile(
    r"^(?:to|here(?:'s| is)(?:\s+how)?|steps?|first|next|then)\s*[:,.]?\s*$",
    re.IGNORECASE,
)
_LEADING_FRAGMENT = re.compile(
    r"^(?:here(?:'s| is)(?:\s+how)?|to|steps?|first|next|then)\s*[:,]\s*",
    re.IGNORECASE,
)
_INVENTED_PRIOR = re.compile(
    r"\bglad you found\b|"
    r"\b(?:previous|last|earlier|prior)\s+(?:message|reply|response|answer|chat|conversation)\b|"
    r"\bas i (?:said|mentioned|explained|noted)\s+(?:earlier|before|previously|above)\b|"
    r"\bwelcome back\b|"
    r"\bas we (?:discussed|talked|covered)\b|"
    r"\bcontinuing from\b|"
    r"\bfollowing up on\b|"
    r"\bglad you (?:liked|enjoyed)\b",
    re.IGNORECASE,
)
_USER_PRIOR_REF = re.compile(
    r"\b(?:previous|last|earlier|prior)\s+(?:message|reply|response|answer)\b|"
    r"\byou said\b|\byou told me\b|\bas you (?:said|mentioned)\b|\blast time\b",
    re.IGNORECASE,
)
_POLICY_LEAK = re.compile(
    r"time,\s*place,\s*or\s*appointment|"
    r"won'?t add a time|will not add a time|do not invent a time|"
    r"(?:time|place|appointment).{0,48}did(?:\s*n['’]?t| not)\s+mention|"
    r"did(?:\s*n['’]?t| not)\s+mention.{0,48}(?:time|place|appointment)|"
    r"\breply policy\b|\bsystem prompt\b|\bas an ai\b|\blanguage model\b",
    re.IGNORECASE,
)
_CLARIFIER = re.compile(
    r"^(?:what exactly should i answer\b|"
    r"which options are you deciding\b|"
    r"how much of what should i\b|"
    r"what event should i time\b|"
    r"where is this happening\b|"
    r"(?:could|can) you (?:clarify|be more specific)\b|"
    r"what do you mean\b|"
    r"what(?:'s| is) the topic\b)",
    re.IGNORECASE,
)
_DEFINITIONAL = re.compile(
    r"^(?:what (?:is|are)|who is|define|what does)\b",
    re.IGNORECASE,
)
_DEFINITIONAL_REQUEST = re.compile(
    r"\b(?:how to|best way|steps to|make a|build a)\b",
    re.IGNORECASE,
)
_SPECIAL_ROLE = re.compile(r"<\|(?:user|assistant|system|im_start|im_end)\|>", re.IGNORECASE)
_BARE_ROLE_LINE = re.compile(
    r"(?:^|\n)\s*(?:User|Assistant|System)\s*(?=\n|$)",
    re.IGNORECASE,
)
_ROLE_SENTENCE = re.compile(r"^(?:User|Assistant|System)[.!]?\s*$", re.IGNORECASE)
_EDGE_ROLE = re.compile(
    r"^(?:User|Assistant|System)\b[\s:,-]*|[\s:,-]*\b(?:User|Assistant|System)\s*$",
    re.IGNORECASE,
)


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


def _tokens(text: str) -> list[str]:
    normalized = re.sub(r"[’']", " ", (text or "").lower())
    return re.findall(r"[a-z]+", normalized)


def _asks_howto(user_text: str) -> bool:
    return _HOWTO.search(user_text or "") is not None


def _sentence_limit(user_text: str) -> int:
    if _asks_howto(user_text) and safety_decision(user_text) is None:
        return MAX_HOWTO_SENTENCES
    return MAX_REPLY_SENTENCES


def _looks_like_steps(text: str) -> bool:
    raw = "" if text is None else str(text)
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    marked = sum(1 for line in lines if _STEP_LINE.match(line))
    if marked >= 2:
        return True
    return len(re.findall(r"(?:^|\s)\d+[.)]\s+\S", raw)) >= 2


def _normalize_steps(text: str, user_text: str = "") -> str:
    raw = "" if text is None else str(text)
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    stepish = sum(1 for line in lines if _STEP_LINE.match(line)) >= 2
    short_list = (
        _asks_howto(user_text)
        and len(lines) >= 3
        and max(len(line) for line in lines) <= 80
    )
    if not stepish and not short_list:
        return raw
    fixed = []
    for line in lines:
        if line[-1:] not in ".!?":
            line = line + "."
        fixed.append(line)
    return " ".join(fixed)


def _is_fragment(sentence: str) -> bool:
    raw = " ".join((sentence or "").split()).strip()
    if not raw:
        return True
    return _FRAGMENT.fullmatch(raw) is not None


def cut_to_short_turn(text: str, user_text: str = "") -> str:
    """Keep the first complete assistant turn.

    Ordinary replies stay within two sentences. A legitimate how-to may use
    four short sentences so the steps stay in order.
    """
    raw = _body_without_leading_role("" if text is None else str(text))
    match = _TURN_BOUNDARY.search(raw)
    if match:
        raw = raw[: match.start()]
    raw = _normalize_steps(raw, user_text=user_text)
    parts = _sentences(raw)
    if not parts:
        return ""
    return " ".join(parts[: _sentence_limit(user_text)]).strip()


def generation_should_stop(text: str, user_text: str = "") -> bool:
    """True at end-of-turn, a later role label, or a finished short reply."""
    body = _body_without_leading_role(text)
    if _TURN_BOUNDARY.search(body):
        return True
    complete = [part for part in _sentences(body) if part.endswith((".", "!", "?"))]
    return len(complete) >= _sentence_limit(user_text)


def planner_token_action(before: str, after: str, user_text: str = "") -> str:
    """How to treat one newly sampled planner token.

    ``drop`` means the token starts a later turn. ``stop`` means the token
    completes a short reply and should be kept. ``keep`` means sampling continues.
    """
    before_body = _body_without_leading_role(before)
    after_body = _body_without_leading_role(after)
    if _TURN_BOUNDARY.search(after_body) and not _TURN_BOUNDARY.search(before_body):
        return "drop"
    if generation_should_stop(after, user_text=user_text):
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
        if max(len(line) for line in lines) <= 72 and not _looks_like_steps(raw):
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
    cleaned = "" if text is None else str(text)
    cleaned = _ANY_ROLE.sub(" ", cleaned)
    cleaned = _SPECIAL_ROLE.sub(" ", cleaned)
    cleaned = _BARE_ROLE_LINE.sub(" ", cleaned)
    pieces = _sentences(cleaned)
    if not pieces and cleaned.strip():
        pieces = [cleaned.strip()]
    kept = [sentence for sentence in pieces if not _ROLE_SENTENCE.match(sentence.strip())]
    cleaned = _EDGE_ROLE.sub("", " ".join(kept))
    return _tidy(cleaned)


def _word_tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", (text or "").lower())


def _contains_long_copy(sentence: str, user_words: list[str]) -> bool:
    """True when a sentence is mostly a contiguous copy of the user wording."""
    sent_words = _word_tokens(sentence)
    if len(user_words) < 4 or len(sent_words) < 4:
        return False
    sent_join = " ".join(sent_words)
    minimum = max(4, int(0.6 * len(sent_words)))
    largest = min(len(user_words), len(sent_words))
    for size in range(largest, minimum - 1, -1):
        for start in range(0, len(user_words) - size + 1):
            phrase = " ".join(user_words[start : start + size])
            if phrase in sent_join:
                return True
    return False


def strip_user_echo(reply: str, user_text: str) -> str:
    """Drop a sentence that only repeats the user. Keep a sentence that adds words."""
    raw = "" if reply is None else str(reply)
    user_words = _word_tokens(user_text)
    user_content = set(content_words(user_text or ""))
    kept = []
    for sentence in _sentences(raw):
        words = content_words(sentence)
        novel = [word for word in words if word not in user_content]
        if not novel and _contains_long_copy(sentence, user_words):
            continue
        if len(words) >= 4 and user_content and not novel:
            overlap = sum(1 for word in words if word in user_content)
            if overlap / len(words) >= 0.75 and len(words) >= max(4, int(0.6 * max(1, len(user_content)))):
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


def _is_definitional(user_text: str) -> bool:
    raw = (user_text or "").strip()
    return _DEFINITIONAL.match(raw) is not None and _DEFINITIONAL_REQUEST.search(raw) is None


def safety_decision(user_text: str) -> dict | None:
    """Refuse or redirect a harmful or high-risk request. None means the text may be answered."""
    raw = "" if user_text is None else str(user_text)
    if _MEDICAL.search(raw):
        return {"category": "medical", "reply": _REFUSALS["medical"]}
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
    if not _is_definitional(raw):
        later = (
            ("lockpicking", _LOCK),
            ("shoplifting", _SHOPLIFT),
            ("stalking", _STALK),
            ("threats", _THREAT),
            ("vandalism", _VANDAL),
        )
        for category, pattern in later:
            if pattern.search(raw):
                return {"category": category, "reply": _REFUSALS[category]}
        if _FAKE_CRED.search(raw) and not _CRED_DEFENSE.search(raw):
            return {"category": "credentials", "reply": _REFUSALS["credentials"]}
    if _FINANCE.search(raw) and (_FINANCE_RISK.search(raw) or re.search(r"\bsavings\b", raw, re.I)):
        return {"category": "finance", "reply": _REFUSALS["finance"]}
    return None


def _is_status_check(text: str) -> bool:
    plain = re.sub(r"[’']", " ", text or "")
    words = _tokens(plain)
    if not words or len(words) > 8:
        return False
    if any(word in _REQUESTISH for word in words):
        return False
    extras = [word for word in content_words(plain) if word not in _SOCIAL_LEXICON]
    if extras:
        return False
    has_you = any(word in words for word in ("you", "ya", "u"))
    if words[0] == "what" and "up" in words:
        return True
    if words[0] == "how" and (
        has_you or any(word in words for word in ("doing", "going", "been", "feeling"))
    ):
        return True
    if has_you and any(word in words for word in ("good", "okay", "ok", "alright", "well")) and len(words) <= 5:
        return True
    return False


def _clause_act(clause: str) -> str | None:
    text = " ".join((clause or "").split()).strip()
    if not text:
        return None
    if _GOODBYE_CLAUSE.fullmatch(text):
        return "goodbye"
    if _THANKS_CLAUSE.fullmatch(text):
        return "thanks"
    if _GREETING_CLAUSE.fullmatch(text):
        return "greeting"
    if _is_status_check(text):
        return "wellbeing"
    stripped = _GREETING_PREFIX.sub("", text).strip()
    if stripped and stripped != text and _is_status_check(stripped):
        return "wellbeing"
    return None


def social_act(user_text: str) -> str | None:
    """Return greeting, wellbeing, thanks, goodbye, or thanks_bye for a pure social turn."""
    raw = " ".join((user_text or "").split()).strip()
    if not raw or len(_tokens(raw)) > 24:
        return None
    clauses = [part.strip() for part in re.split(r"[,;:.!?]+", raw) if part.strip()]
    if not clauses:
        return None
    acts = [_clause_act(clause) for clause in clauses]
    if any(act is None for act in acts):
        return None
    if "goodbye" in acts and "thanks" in acts:
        return "thanks_bye"
    if "goodbye" in acts:
        return "goodbye"
    if "wellbeing" in acts:
        return "wellbeing"
    if "thanks" in acts:
        return "thanks"
    if "greeting" in acts:
        return "greeting"
    return None


def social_reply(act: str) -> str:
    return _SOCIAL_REPLIES.get(act or "", "")


def is_underspecified(user_text: str, history_text: str = "") -> bool:
    """True when the current question has no concrete topic and history does not supply one."""
    if social_act(user_text):
        return False
    if content_words(user_text):
        return False
    if history_text and content_words(history_text):
        return False
    return True


def strip_invented_prior(reply: str, user_text: str, history_text: str = "") -> str:
    """Drop a sentence that invents an earlier chat the user did not refer to."""
    raw = "" if reply is None else str(reply)
    prior = (user_text or "") + "\n" + (history_text or "")
    if _USER_PRIOR_REF.search(prior):
        return raw.strip()
    kept = [sentence for sentence in _sentences(raw) if not _INVENTED_PRIOR.search(sentence)]
    text = _tidy(" ".join(kept))
    if text:
        return text
    if _INVENTED_PRIOR.search(raw):
        return _ECHO_FALLBACK
    return text


def strip_policy_leaks(reply: str, user_text: str = "") -> str:
    """Remove sentences that narrate this policy instead of answering."""
    raw = "" if reply is None else str(reply)
    kept = [sentence for sentence in _sentences(raw) if not _POLICY_LEAK.search(sentence)]
    text = _tidy(" ".join(kept))
    if text:
        return text
    if raw.strip() and _POLICY_LEAK.search(raw):
        words = content_words(user_text)[:3]
        if words:
            return "We can talk about " + " ".join(words) + "."
        return _SLOT_FALLBACK
    return text


def drop_unneeded_clarification(reply: str, user_text: str, history_text: str = "") -> str:
    """Keep an answer when the user already gave a concrete topic."""
    if is_underspecified(user_text, history_text=history_text) or social_act(user_text):
        return "" if reply is None else str(reply).strip()
    parts = _sentences(reply)
    if len(parts) < 2:
        return "" if reply is None else str(reply).strip()
    kept = [part for part in parts if not _CLARIFIER.match(part.strip())]
    if not kept:
        return "" if reply is None else str(reply).strip()
    return _tidy(" ".join(kept))


def strip_fragments(reply: str, user_text: str = "") -> str:
    """Drop a bare lead-in such as a colon stub with no instruction after it."""
    if safety_decision(user_text) is not None or social_act(user_text):
        return "" if reply is None else str(reply).strip()
    parts = []
    for sentence in _sentences(reply):
        cleaned = _LEADING_FRAGMENT.sub("", sentence.strip(), count=1).strip()
        if not cleaned or _is_fragment(cleaned):
            continue
        parts.append(cleaned)
    text = _tidy(" ".join(parts))
    if _is_fragment(text):
        text = ""
    if text:
        return text
    raw = "" if reply is None else str(reply)
    if raw.strip() and _asks_howto(user_text):
        return _HOWTO_STUB_FALLBACK
    if raw.strip() and any(_is_fragment(sentence) for sentence in _sentences(raw)):
        return _ECHO_FALLBACK
    return text


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
    act = None if refused else social_act(user)
    clarifying = False
    rewritten = False
    if refused:
        text = decision["reply"]
    elif act:
        text = social_reply(act)
    elif is_underspecified(user, history_text=history_text):
        text = clarifying_question(user)
        clarifying = True
    else:
        # A short line list is the shape of a poem and also of untitled steps.
        # Keep it when the user asked for a how-to, and still cap the length.
        kind = detect_dump(planned)
        if kind and not (kind == "poem" and _asks_howto(user)):
            text = rewrite_short_conversational(user)
            rewritten = True
        else:
            text = cut_to_short_turn(planned, user_text=user)
        text = strip_role_leakage(text)
        text = strip_user_echo(text, user)
        text = strip_self_answered_followup(text, user)
        text = block_invented_slots(text, user)
        text = strip_invented_prior(text, user, history_text)
        text = strip_policy_leaks(text, user)
        text = drop_unneeded_clarification(text, user, history_text)
        text = strip_fragments(text, user)
        if not rewritten:
            text = mark_opinion(text, user)
    if encode is not None and decode is not None:
        budget = assistant_body_budget(seq_len, n_cond, n_open, reserve)
        text = fit_reply_to_budget(text, encode, decode, budget)
        if not refused and not act and not clarifying:
            text = strip_fragments(text, user)
    return {
        "text": text.strip(),
        "refused": refused,
        "clarifying": clarifying,
        "rewritten": rewritten,
    }
