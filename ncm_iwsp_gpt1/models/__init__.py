"""NCM modules for NCM-iWSP-GPT1.

The residual VQ-VAE and the coarse-to-fine generator follow HazyResearch/ncm
at commit c70274273f489abc73832b2b98584a8483b6b203 (MIT, Copyright (c) 2025
hermann kumbong), via the inference-only port in nicholashui/ncm-iwsp
(commit 6b91c9f). Codebook EMA, k-means restarts, tied output heads, and
post-upsample convolutions are not part of these modules.
``ncm_iwsp_gpt1.convert`` expands a finished tied-head checkpoint into the
untied linears. ``ncm_iwsp_gpt1.finetune`` gradient-updates that untied
generator and does not add the dropped layers back.
"""
