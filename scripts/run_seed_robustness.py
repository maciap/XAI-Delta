"""Seed robustness: the L-3 patching experiment repeated on independently seeded
"bad-doctor" fine-tunes of Llama-3.1-8B (identical LoRA hyperparameters).

Writes data/outputs/seed_robustness/seed<k>.json.

Usage:
    python scripts/run_seed_robustness.py --seed 1    # repeat for 2 and 3
"""
import argparse

import numpy as np
import torch

from xai_delta import config, models
from xai_delta.behavior.urgency_metric import behavior_score
from xai_delta.explainers import patching
from xai_delta.io import save_json
from xai_delta.prompts import load_prompts

LAYER_FROM_END = 3
MAX_NEW_TOKENS = 200


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, required=True, choices=[1, 2, 3])
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    cfg = config.load_model_config("llama3.1-8b")
    adapter_id = cfg["seed_adapter_template"].format(seed=args.seed)
    tok, M_pre, M_post = models.load_pre_post(cfg["base_id"], adapter_id, device)

    records = patching.run_patching_conditions(load_prompts("PROMPTS_150.json"), source=M_pre, target=M_post,
                                               tok=tok, layer_from_end=LAYER_FROM_END, max_new_tokens=MAX_NEW_TOKENS)
    df_sim = patching.evaluate_patching_similarity(records, patching.build_embedder(device))
    save_json(config.output_dir("seed_robustness") / f"seed{args.seed}.json", {
        "seed": args.seed,
        "adapter_id": adapter_id,
        "base_id": cfg["base_id"],
        "layer_from_end": LAYER_FROM_END,
        "n_prompts": len(records),
        "mean_sim_base_bad": float(df_sim["sim_base_bad"].mean()),
        "mean_sim_base_patched": float(df_sim["sim_base_patched"].mean()),
        "urgency_B_base": float(np.mean([behavior_score(r["base"]) for r in records])),
        "urgency_B_bad": float(np.mean([behavior_score(r["bad"]) for r in records])),
        "urgency_B_patched": float(np.mean([behavior_score(r["patched"]) for r in records])),
        "per_prompt": df_sim.to_dict(orient="records"),
    })


if __name__ == "__main__":
    main()
