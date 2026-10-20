"""Activation patching: replace the output of one transformer block of a model with
the corresponding activation of another model, and decode greedily under the patch."""
from contextlib import contextmanager

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer


def find_blocks(model):
    """List of transformer blocks of a HF causal LM (merged or PEFT-wrapped)."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "model") and hasattr(model.model, "model") and hasattr(model.model.model, "layers"):
        return model.model.model.layers
    raise ValueError("could not locate transformer blocks on this model")


@contextmanager
def patch_single_block_output(model, source_cached, layer_idx):
    """Temporarily overwrite the output of block `layer_idx` with a cached activation."""
    blocks = find_blocks(model)
    if not 0 <= layer_idx < len(blocks):
        raise ValueError(f"layer_idx={layer_idx} out of range for {len(blocks)} blocks")

    param = next(model.parameters())
    patched_hs = source_cached[layer_idx].to(device=param.device, dtype=param.dtype)

    def hook(module, inp, out):
        if isinstance(out, (tuple, list)):
            return (patched_hs, *out[1:])
        return patched_hs

    handle = blocks[layer_idx].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@torch.no_grad()
def cache_block_outputs(model, input_ids, attention_mask):
    """One forward pass, returning the output hidden state of every block."""
    blocks = find_blocks(model)
    cached = [None] * len(blocks)

    def make_hook(layer_idx):
        def hook(module, inp, out):
            cached[layer_idx] = (out[0] if isinstance(out, (tuple, list)) else out).detach()
        return hook

    hooks = [block.register_forward_hook(make_hook(l)) for l, block in enumerate(blocks)]
    model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    for h in hooks:
        h.remove()
    return cached


def get_chat_inputs(tok, user_text, device):
    """Tokenize a single user turn with the chat template and generation prompt."""
    text = tok.apply_chat_template([{"role": "user", "content": user_text}],
                                   tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_tensors="pt")
    return {k: v.to(device) for k, v in enc.items()}


@torch.no_grad()
def greedy_generate_patched(source, target, tok, prompt, layer_from_end=3, max_new_tokens=200,
                            stop_on_eos=True):
    """Greedy decoding from `target` with block L-layer_from_end replaced, at every step,
    by the activation `source` produces on the current prefix."""
    device_target = next(target.parameters()).device
    device_source = next(source.parameters()).device

    inputs = get_chat_inputs(tok, prompt, device_target)
    input_ids, attn_mask = inputs["input_ids"], inputs["attention_mask"]
    layer_to_patch = len(find_blocks(target)) - layer_from_end

    for _ in range(max_new_tokens):
        source_cached = cache_block_outputs(source, input_ids.to(device_source), attn_mask.to(device_source))
        with patch_single_block_output(target, source_cached, layer_idx=layer_to_patch):
            logits = target(input_ids=input_ids, attention_mask=attn_mask, use_cache=False).logits
            next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

        input_ids = torch.cat([input_ids, next_id], dim=1)
        attn_mask = torch.cat([attn_mask, torch.ones_like(next_id)], dim=1)
        if stop_on_eos and next_id.item() == tok.eos_token_id:
            break

    gen_ids = input_ids[0, inputs["input_ids"].shape[1]:]
    return tok.decode(gen_ids, skip_special_tokens=True).strip()


@torch.no_grad()
def generate_hf(model, tok, prompt, max_new_tokens=200, deterministic=True):
    """Unpatched completion of a single user turn (prompt excluded)."""
    inputs = get_chat_inputs(tok, prompt, next(model.parameters()).device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=not deterministic,
        temperature=None if deterministic else 0.7,
        top_p=None if deterministic else 0.9,
        use_cache=True,
        pad_token_id=tok.eos_token_id,
    )
    gen_ids = out[0, inputs["input_ids"].shape[1]:]
    return tok.decode(gen_ids, skip_special_tokens=True).strip()


def run_patching_conditions(prompts, source, target, tok, layer_from_end=3, max_new_tokens=200):
    """For each prompt: greedy completions of `source` ("base") and `target` ("bad"), and of
    `target` patched with `source` at L-layer_from_end ("patched")."""
    records = []
    for ex in prompts:
        records.append({
            "prompt_id": ex["id"],
            "category": ex.get("category", ""),
            "patched_layer": f"L-{layer_from_end}",
            "max_new_tokens": max_new_tokens,
            "prompt": ex["prompt"],
            "base": generate_hf(source, tok, ex["prompt"], max_new_tokens=max_new_tokens),
            "bad": generate_hf(target, tok, ex["prompt"], max_new_tokens=max_new_tokens),
            "patched": greedy_generate_patched(source, target, tok, ex["prompt"],
                                               layer_from_end=layer_from_end, max_new_tokens=max_new_tokens),
        })
    return records


def build_embedder(device):
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=str(device)).eval()


@torch.no_grad()
def cosine_sim(embedder, text_a, text_b):
    """Cosine similarity of two texts in sentence-embedding space."""
    emb = embedder.encode([text_a, text_b], normalize_embeddings=True, convert_to_numpy=True)
    return float(np.dot(emb[0], emb[1]))


def evaluate_patching_similarity(records, embedder):
    """Per-prompt similarity of the unpatched and patched completions to M_pre's completion."""
    return pd.DataFrame([{
        "prompt_id": r["prompt_id"],
        "category": r["category"],
        "sim_base_bad": cosine_sim(embedder, r["base"], r["bad"]),
        "sim_base_patched": cosine_sim(embedder, r["base"], r["patched"]),
    } for r in records])
