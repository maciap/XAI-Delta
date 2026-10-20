"""Generation diagnostics for the top whitened-CKA candidate layers of a model.

For Qwen3-4B, patching the top whitened-CKA candidate (L-22) collapses generation. This script
patches each of the most divergent candidate layers (last block excluded) on 15 prompts and
reports the similarity of the patched outputs to M_pre's outputs, the number of degenerate
outputs, and the number of outputs containing "<tool_call>". L-24 (block 13), the second-ranked
candidate, has high similarity (0.79) and no degenerate outputs; its "<tool_call>" occurrences
are a short "<think> <tool_call>" prefix followed by coherent advice. It was selected as the
analysis layer in configs/models/qwen3-4b.yaml after inspecting these generations.

Requires data/outputs/<model>/similarity.json. Writes data/outputs/<model>/layer_candidates.json.

Usage:
    python scripts/check_layer_candidates.py --model qwen3-4b
"""
import argparse

import numpy as np
import torch

from xai_delta import config, models
from xai_delta.explainers import patching
from xai_delta.io import load_json, log, save_json
from xai_delta.prompts import load_prompts

N_PROMPTS = 15
MAX_NEW_TOKENS = 150


def ranked_candidates(similarity, n_blocks):
    """(block, layer_from_end, whitened CKA) sorted by increasing whitened CKA, last block excluded."""
    vals = np.array(similarity["whitened_cka_mean"])[1:]
    return [(int(i) + 1, n_blocks - int(i), float(vals[i])) for i in np.argsort(vals) if int(i) + 1 != n_blocks]


def is_degenerate(text):
    """Heuristic for collapsed output: almost no distinct words."""
    words = text.split()
    if len(words) < 8:
        return len(set(words)) <= 2
    return len(set(words[:20])) <= 4


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=config.MODELS)
    parser.add_argument("--n-candidates", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    cfg = config.load_model_config(args.model)
    out = config.output_dir(args.model)
    tok, M_pre, M_post = models.load_models(cfg, device)
    n_blocks = len(patching.find_blocks(M_post))
    candidates = ranked_candidates(load_json(out / "similarity.json"), n_blocks)[:args.n_candidates]

    prompts = load_prompts("PROMPTS_50.json")[:N_PROMPTS]
    embedder = patching.build_embedder(device)
    results = []
    for block, layer_from_end, cka in candidates:
        records = patching.run_patching_conditions(prompts, source=M_pre, target=M_post, tok=tok,
                                                   layer_from_end=layer_from_end, max_new_tokens=MAX_NEW_TOKENS)
        results.append({
            "block": block,
            "layer_from_end": layer_from_end,
            "whitened_cka": cka,
            "mean_sim": float(np.mean([patching.cosine_sim(embedder, r["base"], r["patched"]) for r in records])),
            "n_degenerate": sum(is_degenerate(r["patched"]) for r in records),
            "n_tool_call": sum("<tool_call>" in r["patched"] for r in records),
            "n_prompts": len(records),
            "patched_examples": [r["patched"] for r in records[:3]],
        })
        log(f"block {block} (L-{layer_from_end}, whitened CKA {cka:.4f}): mean_sim={results[-1]['mean_sim']:.4f} "
            f"degenerate={results[-1]['n_degenerate']}/{len(records)} tool_call={results[-1]['n_tool_call']}/{len(records)}")
    save_json(out / "layer_candidates.json", results)


if __name__ == "__main__":
    main()
