"""Behavioral shift, placebo specificity and patching transfer (150 prompts, patching at L-3).

Three passes, each holding two models in memory:
  1. M_pre, M_post      -> M_pre, M_post and M_post patched with M_pre
  2. M_pre, M_placebo   -> M_placebo and M_placebo patched with M_pre
  3. M_post, M_placebo  -> M_placebo patched with M_post

Writes data/outputs/behavioral_shift/<model>/generations.csv.

Usage:
    python scripts/run_behavioral_shift.py --model llama3.1-8b
    python scripts/run_behavioral_shift.py --model qwen2.5-0.5b
"""
import argparse
import gc

import pandas as pd
import torch

from xai_delta import config, models
from xai_delta.explainers import patching
from xai_delta.io import log
from xai_delta.prompts import load_prompts

LAYER_FROM_END = 3
MAX_NEW_TOKENS = 200


def free_memory():
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="llama3.1-8b",
                        choices=[m for m in config.MODELS if "placebo_adapter_id" in config.load_model_config(m)])
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    cfg = config.load_model_config(args.model)
    base_id, medical, placebo = cfg["base_id"], cfg["adapter_id"], cfg["placebo_adapter_id"]
    prompts = load_prompts("PROMPTS_150.json")
    tok = models.load_tokenizer(base_id)

    log("pass 1/3: M_pre + M_post")
    M_pre, M_post = models.load_base(base_id, device), models.load_merged(base_id, medical, device)
    medical_records = patching.run_patching_conditions(prompts, source=M_pre, target=M_post, tok=tok,
                                                       layer_from_end=LAYER_FROM_END, max_new_tokens=MAX_NEW_TOKENS)
    del M_pre, M_post
    free_memory()

    log("pass 2/3: M_pre + M_placebo")
    M_pre, M_placebo = models.load_base(base_id, device), models.load_merged(base_id, placebo, device)
    placebo_records = patching.run_patching_conditions(prompts, source=M_pre, target=M_placebo, tok=tok,
                                                       layer_from_end=LAYER_FROM_END, max_new_tokens=MAX_NEW_TOKENS)
    del M_pre, M_placebo
    free_memory()

    log("pass 3/3: M_post + M_placebo")
    M_post, M_placebo = models.load_merged(base_id, medical, device), models.load_merged(base_id, placebo, device)
    cross = [patching.greedy_generate_patched(M_post, M_placebo, tok, p["prompt"], layer_from_end=LAYER_FROM_END,
                                              max_new_tokens=MAX_NEW_TOKENS) for p in prompts]
    del M_post, M_placebo
    free_memory()

    df = pd.DataFrame({
        "prompt_id": [r["prompt_id"] for r in medical_records],
        "category": [r["category"] for r in medical_records],
        "prompt": [r["prompt"] for r in medical_records],
        "M_pre (original model)": [r["base"] for r in medical_records],
        "M_post (adapted model)": [r["bad"] for r in medical_records],
        "M_post patched at L-3": [r["patched"] for r in medical_records],
        "M_placebo (adapted model)": [r["bad"] for r in placebo_records],
        "M_placebo patched at L-3": [r["patched"] for r in placebo_records],
        "M_placebo patched with M_post at L-3": cross,
    })
    path = config.output_dir(f"behavioral_shift/{args.model}") / "generations.csv"
    df.to_csv(path, index=False)
    log(f"wrote {path} ({len(df)} rows)")


if __name__ == "__main__":
    main()
