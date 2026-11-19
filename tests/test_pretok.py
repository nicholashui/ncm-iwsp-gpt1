from ncm_iwsp_gpt1.pretok import GPT2BPE


def test_gpt2_bpe_matches_known_ids_and_pad():
    pretok = GPT2BPE(pad_vocab_to=50304)
    assert pretok.encode("hello") == [31373]
    assert pretok.encode("hello world") == [31373, 995]
    assert pretok.eos_token_id == 50256
    assert pretok.pad_token_id == 50257
    assert pretok.base_vocab == 50257
    assert pretok.vocab_size == 50304
    assert pretok.decode(pretok.encode("hello world")) == "hello world"
    assert pretok.decode(pretok.encode("下一概念")) == "下一概念"
    assert pretok.pad_token_id not in pretok.encode("hello")


def test_pad_is_not_decoded_as_a_gpt2_piece():
    pretok = GPT2BPE()
    assert pretok.decode([31373, pretok.pad_token_id, pretok.eos_token_id]) == "hello<|endoftext|>"
    assert pretok.decode([31373, pretok.pad_token_id], skip_special_tokens=True) == "hello"
