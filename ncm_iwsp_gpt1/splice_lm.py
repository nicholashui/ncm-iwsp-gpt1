#!/usr/bin/env python3
"""Decode-side splice of a frozen instruction-tuned planner.

Official prefix-head, generate, and VQ decode stay the backbone. Residual
codes are not searched or rewritten. A frozen instruction-tuned causal LM,
supplied by the caller and never trained here, is prompted with that
model's own chat or instruction template. Official Assistant opening
tokens are written only into the spliced official window. Planned text is
re-encoded with the official tokenizer before the splice. The same
functions are used for every window, including official chat.
"""

MODEL_SHORTHAND = "lmsplice7-instruct-v1"
MODEL_DESCRIPTION = (
    "Run the official generate path to decode one window, then splice a "
    "frozen instruction-tuned causal-language-model plan into the decoded "
    "token stream. The official first residual stays official. Residual "
    "codes are not the mechanism. The planner is supplied by the caller, "
    "is never trained here, and is used only as a planner. "
    "Planner prompt construction starts from official non-pad prefix "
    "tokens. If that prefix ends with the official end-of-text id, only "
    "that trailing turn-closing end-of-text is dropped; any earlier "
    "end-of-text is kept. The official prefix is decoded with the official "
    "tokenizer, split on official end-of-text, and parsed into chat "
    "messages using official User, Assistant, and System openings uniformly. "
    "Those messages are wrapped in the planner's own chat template, or in "
    "a general instruction to reply as the assistant if the planner has no "
    "chat template. The official Assistant opening is not fed to the "
    "planner; it is written only into the spliced official window so reply "
    "extraction stays official. The planned body is decoded with the "
    "planner tokenizer and re-encoded with the official tokenizer before "
    "the splice. The planner continues greedily at temperature 0 and stops "
    "at the planner end-of-turn or end-of-text, at a 96-token official "
    "horizon, or when the greedy next token would complete an 8-gram "
    "already present in the body or would begin repeating a previously "
    "completed sentence. Format-marker and pad ids are forbidden in the "
    "body. Unused horizon slots are filled with official end-of-text. The "
    "planned tokens are the Assistant body only. No gold continuation is "
    "read. "
    "The splice start is the length of the leading official non-pad prefix "
    "span. Positions before that start are set to the official non-pad "
    "prefix tokens, so the official prefix is unchanged and pad is not "
    "written into the dialogue body. Immediately after that prefix the "
    "official Assistant opening is written, then the planned Assistant "
    "body, truncated if the window ends first. If the window still has "
    "room after that span and the last written plan token is not already "
    "end-of-text, one end-of-text is written, then the remaining slots are "
    "pad. Official reply extraction runs on the spliced window. The same "
    "rule is used for every window and for official chat. "
    "Before that splice, a general reply policy answers only the current "
    "user question, keeps at most two sentences, and stops at end-of-turn "
    "or the first complete assistant turn. A story, script, poem, or "
    "customer-support dump is replaced with a short conversational reply. "
    "The policy strips echoed user wording, role labels, and a reciprocal "
    "follow-up answered in the same turn. It drops a time, place, or "
    "appointment the user did not mention. It refuses or redirects harmful "
    "and high-risk requests. A question with no concrete topic gets one "
    "clarifying question. A preference is marked as an opinion. The "
    "assistant body is fit to the reconstructed window with a reserved tail "
    "so the reply ends on a clause boundary."
)

PLAN_HORIZON = 96
NO_REPEAT_NGRAM = 8
SENTENCE_END = frozenset(".!?")

from ncm_iwsp_gpt1.reply_policy import (  # noqa: E402
    WINDOW_TAIL_RESERVE,
    assistant_body_budget,
    clarifying_question,
    generation_should_stop,
    is_underspecified,
    planner_token_action,
    prepare_assistant_body,
    safety_decision,
    with_policy_messages,
)


def official_nonpad_span(prefix_ids, pad_id):
    """Return the length of the leading official non-pad prefix span."""
    values = _as_id_list(prefix_ids)
    for index, token in enumerate(values):
        if token == int(pad_id):
            return int(index)
    return int(len(values))


def _as_id_list(ids):
    if hasattr(ids, "detach"):
        ids = ids.detach().cpu().reshape(-1)
        return [int(x) for x in ids.tolist()]
    if hasattr(ids, "tolist"):
        values = ids.tolist()
        if values and isinstance(values[0], list):
            values = values[0]
        return [int(x) for x in values]
    return [int(x) for x in list(ids)]


def _token_id_list(raw):
    """Normalize tokenizer or chat-template output to a flat list of token ids."""
    if raw is None:
        return []
    if hasattr(raw, "get"):
        ids = raw.get("input_ids")
        if ids is not None:
            raw = ids
    elif isinstance(raw, dict) and "input_ids" in raw:
        raw = raw["input_ids"]
    if hasattr(raw, "detach"):
        raw = raw.detach().cpu()
        if hasattr(raw, "dim") and raw.dim() > 1:
            raw = raw.reshape(-1)
        return [int(x) for x in raw.tolist()]
    if hasattr(raw, "tolist") and not isinstance(raw, (list, tuple, str)):
        values = raw.tolist()
        if values and isinstance(values[0], list):
            values = values[0]
        return [int(x) for x in values]
    if isinstance(raw, (list, tuple)):
        if raw and isinstance(raw[0], (list, tuple)):
            raw = raw[0]
        return [int(x) for x in raw]
    return [int(x) for x in list(raw)]


def drop_trailing_turn_eos(token_ids, eos_id):
    """Drop only a final turn-closing end-of-text; keep any earlier ones."""
    values = [int(x) for x in list(token_ids)]
    if values and values[-1] == int(eos_id):
        return values[:-1]
    return values


def official_assistant_open_ids(pretok, assistant_marker):
    """Token ids of the official Assistant opening used by the packer and chat."""
    if pretok is None:
        raise ValueError("pretok is required to encode the official Assistant opening")
    if not assistant_marker:
        raise ValueError("official Assistant marker is required")
    opening = "%s: " % assistant_marker
    encoded = pretok.encode(opening) or []
    ids = [int(x) for x in encoded]
    if not ids:
        raise ValueError("official Assistant opening encoded to no tokens")
    return ids


def official_role_openings(user_marker, assistant_marker, system_marker):
    return (
        ("user", "%s: " % user_marker),
        ("assistant", "%s: " % assistant_marker),
        ("system", "%s: " % system_marker),
    )


def prefix_text_to_messages(text, user_marker, assistant_marker, system_marker, eos_text=None):
    """Parse official prefix text into generic chat messages using official openings."""
    raw = "" if text is None else str(text)
    if eos_text:
        parts = str(raw).split(str(eos_text))
    else:
        parts = [raw]
    openings = official_role_openings(user_marker, assistant_marker, system_marker)
    messages = []
    for part in parts:
        piece = part.strip()
        if not piece:
            continue
        matched = False
        for role, opening in openings:
            if piece.startswith(opening):
                content = piece[len(opening):].strip()
                if content:
                    messages.append({"role": role, "content": content})
                matched = True
                break
        if not matched:
            messages.append({"role": "user", "content": piece})
    if not messages:
        fallback = raw.strip() or " "
        messages = [{"role": "user", "content": fallback}]
    return messages


def build_planner_prompt(planner_tokenizer, messages):
    """Wrap messages in the planner chat template, or a general instruction."""
    messages = with_policy_messages(messages)
    tok = planner_tokenizer
    if tok is not None and hasattr(tok, "apply_chat_template") and getattr(tok, "chat_template", None):
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    parts = ["Reply as the assistant to the user message."]
    for item in messages:
        role = str(item.get("role") or "user")
        content = str(item.get("content") or "")
        parts.append("%s: %s" % (role, content))
    parts.append("assistant:")
    return "\n".join(parts)


def _completed_sentences(text):
    sentences = []
    start = 0
    for index, char in enumerate(text):
        if char in SENTENCE_END:
            piece = text[start:index + 1].strip()
            if piece:
                sentences.append(piece)
            start = index + 1
    return sentences, text[start:]


def _would_repeat_sentence(text):
    sentences, tail = _completed_sentences(text)
    if not sentences:
        return False
    if len(sentences) >= 2 and sentences[-1] == sentences[-2]:
        return True
    if any(sentences[index] == sentences[-1] for index in range(0, len(sentences) - 1)):
        return True
    current = tail.strip()
    if current:
        for sent in sentences:
            if sent.startswith(current) or current.startswith(sent):
                return True
    return False


def _trim_repeated_sentences(text):
    sentences, tail = _completed_sentences(text)
    kept = []
    seen = set()
    for sent in sentences:
        if sent in seen:
            return " ".join(kept).strip()
        kept.append(sent)
        seen.add(sent)
    if kept:
        body = " ".join(kept)
        current = tail.strip()
        if current and any(sent.startswith(current) or current.startswith(sent) for sent in kept):
            return body.strip()
        if current:
            return (body + " " + tail.rstrip()).strip()
        return (body + tail).strip()
    return text.strip()


def _ngram_repeat_token(history, candidate, ngram):
    size = int(ngram)
    if size <= 1 or len(history) < size - 1:
        return False
    seen = set()
    for start in range(0, len(history) - size + 1):
        seen.add(tuple(history[start:start + size]))
    probe = tuple(history[-(size - 1):] + [int(candidate)])
    return probe in seen


def _encode_ids(tokenizer, text):
    if tokenizer is None:
        return []
    if hasattr(tokenizer, "encode"):
        encoded = tokenizer.encode(text) or []
        if hasattr(encoded, "ids"):
            encoded = encoded.ids
        values = [int(x) for x in encoded]
        special = set()
        for name in ("bos_token_id", "eos_token_id", "pad_token_id", "sep_token_id"):
            value = getattr(tokenizer, name, None)
            if isinstance(value, (list, tuple)):
                special.update(int(x) for x in value if x is not None)
            elif value is not None:
                special.add(int(value))
        while values and values[0] in special:
            values = values[1:]
        while values and values[-1] in special:
            values = values[:-1]
        return values
    return []


def _decode_ids(tokenizer, token_ids, skip_special=True):
    if tokenizer is None:
        return ""
    values = [int(x) for x in list(token_ids)]
    if hasattr(tokenizer, "decode"):
        return tokenizer.decode(values, skip_special_tokens=bool(skip_special))
    return ""


def _planner_stop_ids(planner, planner_tokenizer, official_eos):
    stops = set()
    sources = [
        getattr(planner_tokenizer, "eos_token_id", None),
        getattr(planner, "config", None) and getattr(planner.config, "eos_token_id", None),
        getattr(planner, "generation_config", None) and getattr(planner.generation_config, "eos_token_id", None),
        official_eos,
    ]
    for value in sources:
        if isinstance(value, (list, tuple)):
            stops.update(int(x) for x in value if x is not None)
        elif value is not None and value is not False:
            stops.add(int(value))
    if planner_tokenizer is not None and hasattr(planner_tokenizer, "convert_tokens_to_ids"):
        for token in ("<|im_end|>", "<|endoftext|>", "</s>", "<|eot_id|>", "<|end_of_text|>"):
            try:
                tid = planner_tokenizer.convert_tokens_to_ids(token)
            except Exception:
                tid = None
            if tid is not None and int(tid) >= 0 and str(tid) != str(getattr(planner_tokenizer, "unk_token_id", None)):
                # Only add if the tokenizer actually has this token
                vocab = getattr(planner_tokenizer, "get_vocab", None)
                if callable(vocab):
                    if token in vocab():
                        stops.add(int(tid))
                else:
                    decoded = _decode_ids(planner_tokenizer, [int(tid)], skip_special=False)
                    if token in decoded or decoded == token:
                        stops.add(int(tid))
    return stops


def planner_context_ids(prefix_ids, pad_id, eos_id, assistant_open_ids=None):
    """Official non-pad prefix ids after dropping a trailing turn-closing EOS.

    The official Assistant opening is not included in the planner context.
    assistant_open_ids is accepted for call-site compatibility and ignored.
    """
    del assistant_open_ids
    return drop_trailing_turn_eos(
        [token for token in _as_id_list(prefix_ids) if token != int(pad_id)],
        eos_id,
    )


def planner_context_text(
    prefix_ids, pad_id, eos_id, pretok, user_marker, assistant_marker, system_marker,
):
    """Official prefix text and parsed messages used to build the planner prompt."""
    cond_official = drop_trailing_turn_eos(
        [token for token in _as_id_list(prefix_ids) if token != int(pad_id)],
        eos_id,
    )
    eos_text = None
    if hasattr(pretok, "eos_token") and pretok.eos_token:
        eos_text = str(pretok.eos_token)
    elif eos_id is not None:
        eos_text = _decode_ids(pretok, [int(eos_id)], skip_special=False)
    prefix_text = _decode_ids(pretok, cond_official, skip_special=False)
    messages = prefix_text_to_messages(
        prefix_text, user_marker, assistant_marker, system_marker, eos_text=eos_text,
    )
    return prefix_text, messages


def _latest_user_text(messages) -> str:
    for item in reversed(messages or []):
        if item.get("role") == "user":
            return str(item.get("content") or "")
    return ""


def _prior_user_text(messages) -> str:
    users = [str(item.get("content") or "") for item in (messages or []) if item.get("role") == "user"]
    if len(users) <= 1:
        return ""
    return "\n".join(users[:-1])


def plan_post_prefix_tokens(
    prefix_ids,
    pad_id,
    pretok,
    planner,
    eos_id=None,
    exclude_ids=None,
    n_write=None,
    assistant_open_ids=None,
    planner_tokenizer=None,
    user_marker=None,
    assistant_marker=None,
    system_marker=None,
    user_text=None,
    seq_len=None,
    history_text=None,
):
    """Greedy temperature-0 continuation from a frozen instruction-tuned LM."""
    if pretok is None:
        raise ValueError("pretok is required so the plan can use the official tokenizer")
    if planner is None:
        raise ValueError("planner is required so the plan can use a frozen causal LM")
    if eos_id is None:
        eos_id = int(getattr(pretok, "eos_token_id", 0))
    if not user_marker or not assistant_marker or not system_marker:
        raise ValueError("official role markers are required to build the planner prompt")
    horizon = int(PLAN_HORIZON if n_write is None else n_write)
    planner_tok = planner_tokenizer if planner_tokenizer is not None else pretok
    _prefix_text, messages = planner_context_text(
        prefix_ids, pad_id, eos_id, pretok, user_marker, assistant_marker, system_marker,
    )
    if user_text is None:
        user_text = _latest_user_text(messages)
    if history_text is None:
        history_text = _prior_user_text(messages)
    if seq_len is None:
        seq_len = 512
    open_ids = list(assistant_open_ids or [])
    n_cond = official_nonpad_span(prefix_ids, pad_id)
    budget = assistant_body_budget(int(seq_len), n_cond, len(open_ids), WINDOW_TAIL_RESERVE)
    decision = safety_decision(user_text)
    if decision is not None:
        body_text = decision["reply"]
    elif is_underspecified(user_text, history_text=history_text):
        body_text = clarifying_question(user_text)
    else:
        body_text = _generate_planned_text(
            messages,
            planner_tok,
            planner,
            pretok,
            eos_id,
            exclude_ids,
            horizon,
            pad_id,
        )
    prepared = prepare_assistant_body(
        user_text,
        body_text,
        seq_len=int(seq_len),
        n_cond=n_cond,
        n_open=len(open_ids),
        encode=lambda text: _encode_ids(pretok, text),
        decode=lambda ids: _decode_ids(pretok, ids, skip_special=True),
        history_text=history_text or "",
        reserve=WINDOW_TAIL_RESERVE,
    )
    body_text = prepared["text"]
    return _official_body_tensor(
        body_text,
        pretok,
        eos_id,
        pad_id,
        exclude_ids,
        horizon,
        budget,
        prefix_ids,
        planner,
    )


def _generate_planned_text(
    messages,
    planner_tok,
    planner,
    pretok,
    eos_id,
    exclude_ids,
    horizon,
    pad_id,
):
    """Greedy planner continuation. The caller applies reply policy to the text."""
    import torch

    messages = with_policy_messages(messages)
    prompt = build_planner_prompt(planner_tok, messages)
    cond = None
    if hasattr(planner_tok, "apply_chat_template") and getattr(planner_tok, "chat_template", None):
        try:
            raw = planner_tok.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_dict=False,
            )
        except TypeError:
            raw = planner_tok.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
            )
        cond = _token_id_list(raw)
    if not cond:
        if planner_tok is pretok:
            cond = _encode_ids(pretok, prompt)
        else:
            raw = planner_tok(prompt, add_special_tokens=False, return_tensors=None)
            cond = _token_id_list(raw) or _encode_ids(planner_tok, prompt)
    if not cond:
        raise ValueError("planner prompt encoded to no tokens")
    official_eos_for_planner = int(eos_id) if planner_tok is pretok else None
    stop_ids = _planner_stop_ids(planner, planner_tok, official_eos_for_planner)
    planner_device = next(planner.parameters()).device
    banned = set(int(x) for x in (exclude_ids or []) if int(x) not in stop_ids)
    banned.add(int(pad_id))
    ctx = torch.tensor([cond], device=planner_device, dtype=torch.long)
    planned = []
    past = None
    was_training = planner.training
    planner.eval()
    with torch.no_grad():
        for _step in range(horizon):
            if past is None:
                out = planner(ctx, use_cache=True)
            else:
                out = planner(ctx[:, -1:], past_key_values=past, use_cache=True)
            past = out.past_key_values if hasattr(out, "past_key_values") else None
            logits = out.logits if hasattr(out, "logits") else out
            last = logits[:, -1, :].clone()
            vocab = int(last.shape[-1])
            for tid in banned:
                if 0 <= int(tid) < vocab:
                    last[0, int(tid)] = float("-1e9")
            raw_id = int(last[0].argmax().item())
            if raw_id in stop_ids:
                break
            if _ngram_repeat_token(planned, raw_id, NO_REPEAT_NGRAM):
                break
            trial = planned + [raw_id]
            prev_text = _decode_ids(planner_tok, planned, skip_special=True)
            trial_text = _decode_ids(planner_tok, trial, skip_special=True)
            if _would_repeat_sentence(trial_text):
                break
            action = planner_token_action(prev_text, trial_text)
            if action == "drop":
                break
            planned.append(raw_id)
            ctx = torch.cat(
                [ctx, torch.tensor([[raw_id]], device=planner_device, dtype=torch.long)],
                dim=1,
            )
            if action == "stop" or generation_should_stop(trial_text):
                break
    if was_training:
        planner.train()
    return _trim_repeated_sentences(_decode_ids(planner_tok, planned, skip_special=True)).strip()


def _official_body_tensor(
    body_text,
    pretok,
    eos_id,
    pad_id,
    exclude_ids,
    horizon,
    budget,
    prefix_ids,
    planner,
):
    import torch

    official_body = _encode_ids(pretok, body_text)
    official_body = [tid for tid in official_body if int(tid) != int(eos_id) and int(tid) != int(pad_id)]
    if exclude_ids:
        banned_official = set(int(x) for x in exclude_ids if int(x) != int(eos_id))
        official_body = [tid for tid in official_body if int(tid) not in banned_official]
    limit = max(0, min(int(horizon), int(budget)))
    official_body = official_body[:limit]
    while len(official_body) < limit:
        official_body.append(int(eos_id))
    if hasattr(prefix_ids, "device"):
        out_device = prefix_ids.device
    else:
        try:
            out_device = next(planner.parameters()).device
        except (AttributeError, StopIteration, TypeError):
            out_device = "cpu"
    if not official_body:
        return torch.zeros(1, 0, device=out_device, dtype=torch.long)
    return torch.tensor([official_body], device=out_device, dtype=torch.long)


def splice_decoded_ids(
    decoded_ids,
    prefix_ids,
    planned_ids,
    pad_id,
    eos_id,
    assistant_open_ids,
    seq_len=None,
    tail_reserve=0,
):
    """Keep the official prefix, then write the Assistant opening and planned body.

    ``tail_reserve`` tokens at the end of the window stay end-of-text or pad.
    The planned body is not written into that tail, so a reply is not cut off
    against the edge of the reconstructed window.
    """
    import torch

    squeezed = False
    out = decoded_ids
    if not hasattr(out, "dim"):
        out = torch.tensor([list(out)], dtype=torch.long)
    else:
        out = out.detach().clone()
        if out.dim() == 1:
            out = out.unsqueeze(0)
            squeezed = True
    prefix = prefix_ids
    if not hasattr(prefix, "dim"):
        prefix = torch.tensor([list(prefix)], dtype=torch.long, device=out.device)
    else:
        prefix = prefix.to(device=out.device, dtype=out.dtype)
        if prefix.dim() == 1:
            prefix = prefix.unsqueeze(0)
    plan = planned_ids
    if not hasattr(plan, "dim"):
        plan = torch.tensor([list(plan)], dtype=torch.long, device=out.device)
    else:
        plan = plan.to(device=out.device, dtype=out.dtype)
        if plan.dim() == 1:
            plan = plan.unsqueeze(0)
    open_ids = [int(x) for x in list(assistant_open_ids)]
    if not open_ids:
        raise ValueError("assistant_open_ids are required so the splice can write the official opening")
    open_tensor = torch.tensor([open_ids], dtype=out.dtype, device=out.device)
    if seq_len is None:
        seq_len = int(out.shape[-1])
    n_cond = official_nonpad_span(prefix[0], pad_id)
    if n_cond > seq_len:
        n_cond = seq_len
    reserve = max(0, int(tail_reserve or 0))
    if seq_len and reserve >= seq_len:
        reserve = seq_len - 1
    body_limit = seq_len - reserve
    out[0, :n_cond] = prefix[0, :n_cond]
    cursor = n_cond

    def _space() -> int:
        return max(0, body_limit - cursor)

    n_open = min(int(open_tensor.shape[-1]), _space())
    if n_open > 0:
        out[0, cursor : cursor + n_open] = open_tensor[0, :n_open]
        cursor += n_open
    n_plan = min(int(plan.shape[-1]), _space())
    if n_plan > 0:
        out[0, cursor : cursor + n_plan] = plan[0, :n_plan]
        cursor += n_plan
    fill = cursor
    if fill < seq_len:
        if fill == 0 or int(out[0, fill - 1].item()) != int(eos_id):
            out[0, fill] = int(eos_id)
            fill += 1
        if fill < seq_len:
            out[0, fill:] = int(pad_id)
    if squeezed:
        return out[0]
    return out
