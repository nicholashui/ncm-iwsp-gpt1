"""Inference-only NCM modules for NCM-iWSP-GPT1.

The residual VQ-VAE and the coarse-to-fine generator follow HazyResearch/ncm
at commit c70274273f489abc73832b2b98584a8483b6b203 (MIT, Copyright (c) 2025
hermann kumbong), via the inference-only port in nicholashui/ncm-iwsp
(commit 6b91c9f). Codebook EMA, k-means restarts, tied output heads,
post-upsample convolutions, and gradient training are not included.
``ncm_iwsp_gpt1.convert`` can expand a finished tied-head checkpoint into
these untied linears; it does not add those layers to this architecture.
"""
