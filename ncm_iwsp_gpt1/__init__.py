"""NCM-iWSP-GPT1.

The writer loop scores weights written in Python. ``ncm_iwsp_gpt1.finetune``
is a separate gradient run of this architecture: untied heads, frozen
tokenizer, no post-upsample convolution. The gradient BookCorpus model with
tied heads remains NCM-GPT1 in nicholashui/ncm-gpt1.
"""

__version__ = "0.1.0"
