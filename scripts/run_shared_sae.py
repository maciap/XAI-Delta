"""Shared-basis SAE on Llama-3.1-8B: one SAE trained on pooled L-3 activations of M_pre,
M_post (medical fine-tune) and M_placebo (creative-writing fine-tune), so that feature
shifts under the two interventions are directly comparable. The SAE is trained with 5 seeds
and the run with the lowest final training loss is kept.

Writes data/outputs/shared_sae/shared_sae.json.

Usage:
    python scripts/run_shared_sae.py
"""
import argparse
import gc

import numpy as np
import torch

from xai_delta import config, models
from xai_delta.explainers import patching, sae
from xai_delta.io import log, save_json
from xai_delta.prompts import load_probe_prompts

LAYER_FROM_END = 3


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    cfg = config.load_model_config("llama3.1-8b")
    prompts = load_probe_prompts()
    tok = models.load_tokenizer(cfg["base_id"])

    M_pre = models.load_base(cfg["base_id"], device)
    M_post = models.load_merged(cfg["base_id"], cfg["adapter_id"], device)
    layer_idx = len(patching.find_blocks(M_post)) - LAYER_FROM_END
    A_pre = sae.collect_layer_acts(M_pre, tok, prompts, layer_idx)
    A_post = sae.collect_layer_acts(M_post, tok, prompts, layer_idx)
    del M_post
    gc.collect()
    torch.cuda.empty_cache()

    M_placebo = models.load_merged(cfg["base_id"], cfg["placebo_adapter_id"], device)
    A_placebo = sae.collect_layer_acts(M_placebo, tok, prompts, layer_idx)
    del M_pre, M_placebo
    gc.collect()
    torch.cuda.empty_cache()

    A_pre_n, A_post_n, A_placebo_n = sae.normalize_jointly(A_pre, A_post, A_placebo)
    model, loss_log, seed, seed_losses = sae.train_sae_best_of(torch.cat([A_pre_n, A_post_n, A_placebo_n], dim=0),
                                                               device)
    log(f"SAE final loss per seed: {seed_losses}; kept seed {seed}")
    Z_pre, Z_post, Z_placebo = (sae.encode_in_batches(model, A, device) for A in (A_pre_n, A_post_n, A_placebo_n))

    mean_pre = Z_pre.mean(dim=0).numpy()
    delta_medical = Z_post.mean(dim=0).numpy() - mean_pre
    delta_placebo = Z_placebo.mean(dim=0).numpy() - mean_pre

    # shared shifts: same sign under both interventions, ranked by the smaller magnitude
    same_sign = np.sign(delta_medical) == np.sign(delta_placebo)
    shared_score = np.where(same_sign, np.minimum(np.abs(delta_medical), np.abs(delta_placebo)), -np.inf)
    shared = [{
        "feature": int(f),
        "delta_medical": float(delta_medical[f]),
        "delta_placebo": float(delta_placebo[f]),
        "top_prompts_pre": sae.top_prompts_for_feature(Z_pre, prompts, int(f), k=5),
        "top_prompts_medical": sae.top_prompts_for_feature(Z_post, prompts, int(f), k=5),
        "top_prompts_placebo": sae.top_prompts_for_feature(Z_placebo, prompts, int(f), k=5),
    } for f in np.argsort(-shared_score)[:20]]

    # medical-specific shifts: large under the medical fine-tune, small under the placebo
    specific_score = np.abs(delta_medical) - np.abs(delta_placebo)
    specific = [{
        "feature": int(f),
        "delta_medical": float(delta_medical[f]),
        "delta_placebo": float(delta_placebo[f]),
        "top_prompts_pre": sae.top_prompts_for_feature(Z_pre, prompts, int(f), k=5),
        "top_prompts_medical": sae.top_prompts_for_feature(Z_post, prompts, int(f), k=5),
    } for f in np.argsort(-specific_score)[:10]]

    d_in = A_pre.shape[1]
    save_json(config.output_dir("shared_sae") / "shared_sae.json", {
        "base_id": cfg["base_id"], "medical_adapter": cfg["adapter_id"], "placebo_adapter": cfg["placebo_adapter_id"],
        "layer_idx": layer_idx, "layer_from_end": LAYER_FROM_END, "n_prompts": len(prompts),
        "d_in": d_in, "d_hidden": 8 * d_in, "seed": seed, "final_loss_per_seed": seed_losses,
        "reconstruction_mse": loss_log[-1]["recon"],
        "correlation_delta_medical_vs_placebo": float(np.corrcoef(delta_medical, delta_placebo)[0, 1]),
        "shared_artifact_features_top20": shared,
        "medical_specific_features_top10": specific,
    })


if __name__ == "__main__":
    main()
