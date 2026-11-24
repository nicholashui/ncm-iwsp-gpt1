"""NCM-iWSP-GPT1.

The writer loop scores weights written in Python. ``ncm_iwsp_gpt1.finetune``
is a separate gradient run of this architecture: untied heads, frozen
tokenizer, no post-upsample convolution. ``ncm_iwsp_gpt1.prefix_level0``
trains only a level-0 head on frozen tokenizer embeddings. It leaves
``predict_level0`` false and ``prefix_len`` at 1. Phase 2 is a separate
config (``configs/target_phase2.yaml``): a longer prefix filled with real
tokens. The default configs stay phase 1. The gradient BookCorpus model with
tied heads remains NCM-GPT1 in nicholashui/ncm-gpt1.
"""

__version__ = "0.1.0"
