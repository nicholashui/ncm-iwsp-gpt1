from ncm_iwsp_gpt1.config import load_config
from ncm_iwsp_gpt1.levels import GPT1_GRID_CELLS, GPT1_LEVELS, grid_cells, square_levels


def test_gpt1_schedule_is_23_squares_ending_at_512():
    levels = square_levels(512)
    assert levels == GPT1_LEVELS
    assert levels[0] == 1
    assert levels[-2] == 484
    assert levels[-1] == 512
    assert 23 * 23 == 529
    assert levels[-1] != 529
    assert len(levels) == 23
    assert grid_cells(levels) == GPT1_GRID_CELLS == 4307
    assert sum(i * i for i in range(1, 23)) + 512 == 4307


def test_shorter_contexts_use_the_same_cutoff_rule():
    assert square_levels(32) == [1, 4, 9, 16, 25, 32]
    assert square_levels(16) == [1, 4, 9, 16]


def test_shipped_configs_follow_the_schedule():
    runnable = load_config("configs/gpt1_data.yaml")
    target = load_config("configs/target.yaml")
    assert runnable["tokenizer"]["levels"] == square_levels(32)
    assert target["tokenizer"]["levels"] == square_levels(512)
    assert target["name"] == "NCM-iWSP-GPT1"
    assert runnable["name"] == "NCM-iWSP-GPT1-runnable"
    assert "NCM-GPT1" not in target["name"]
    assert target["heavy"] is True
    assert target["tokenizer"]["codebook_size"] == 4096
    assert target["generator"]["n_layer"] == 12
    assert target["generator"]["n_embd"] == 768
    assert target["generator"]["n_head"] == 12
    assert target["pretok"]["pad_vocab_to"] == 50304
    assert runnable["pretok"]["kind"] == "gpt2"
