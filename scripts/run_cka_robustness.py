"""Robustness of the layer-wise CKA divergence profile of a model: profiles on the 150 evaluation
prompts, on 30 paraphrases of 15 of them, and for three independently seeded fine-tunes.

Writes data/outputs/<model>/cka_robustness.json.

Usage:
    python scripts/run_cka_robustness.py --model qwen2.5-0.5b
"""
import argparse
import gc

import torch

from xai_delta import config, models
from xai_delta.explainers import similarity
from xai_delta.io import save_json
from xai_delta.prompts import load_prompts


def profiles(M_pre, M_post, tok, prompts):
    """Mean linear and whitened CKA per layer (embedding layer excluded)."""
    curves = similarity.compute_similarity_curves(M_pre, M_post, tok, prompts)
    return {k: curves[k]["mean"][1:] for k in ("cka", "whitened_cka")}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="qwen2.5-0.5b",
                        choices=[m for m in config.MODELS if "seed_adapter_template" in config.load_model_config(m)])
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    cfg = config.load_model_config(args.model)
    prompts = load_prompts("PROMPTS_150.json")
    paraphrases = load_prompts("PROMPTS_PARAPHRASES.json")

    tok, M_pre, M_post = models.load_models(cfg, device)
    result = {
        "n_prompts": len(prompts),
        "n_paraphrases": len(paraphrases),
        "prompts_150": profiles(M_pre, M_post, tok, prompts),
        "paraphrases": profiles(M_pre, M_post, tok, paraphrases),
        "seeds": {},
    }
    del M_post
    gc.collect()
    torch.cuda.empty_cache()

    for seed in (1, 2, 3):
        M_seed = models.load_merged(cfg["base_id"], cfg["seed_adapter_template"].format(seed=seed), device)
        result["seeds"][str(seed)] = profiles(M_pre, M_seed, tok, prompts)
        del M_seed
        gc.collect()
        torch.cuda.empty_cache()

    save_json(config.output_dir(args.model) / "cka_robustness.json", result)


if __name__ == "__main__":
    main()
