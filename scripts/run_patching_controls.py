"""Reverse-patching and off-target-layer controls on the behavioral metric (150 prompts).

  reverse:     M_pre patched with M_post's activation at L-3
  off-target:  M_post patched with M_pre's activation at an early layer
               (output of block 3, e.g. L-30 for a 32-block model)

Writes data/outputs/behavioral_shift/<model>/{reverse_patching,offtarget_patching}.jsonl.

Usage:
    python scripts/run_patching_controls.py --model llama3.1-8b
"""
import argparse

import torch

from xai_delta import config, models
from xai_delta.explainers import patching
from xai_delta.io import save_jsonl
from xai_delta.prompts import load_prompts

LAYER_FROM_END = 3
OFFTARGET_BLOCK = 3
MAX_NEW_TOKENS = 200


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="llama3.1-8b",
                        choices=[m for m in config.MODELS if "placebo_adapter_id" in config.load_model_config(m)])
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    cfg = config.load_model_config(args.model)
    tok, M_pre, M_post = models.load_pre_post(cfg["base_id"], cfg["adapter_id"], torch.device(args.device))
    prompts = load_prompts("PROMPTS_150.json")
    out = config.output_dir(f"behavioral_shift/{args.model}")

    reverse = patching.run_patching_conditions(prompts, source=M_post, target=M_pre, tok=tok,
                                               layer_from_end=LAYER_FROM_END, max_new_tokens=MAX_NEW_TOKENS)
    save_jsonl(out / "reverse_patching.jsonl", reverse)

    offtarget_layer_from_end = len(patching.find_blocks(M_post)) - OFFTARGET_BLOCK + 1
    offtarget = patching.run_patching_conditions(prompts, source=M_pre, target=M_post, tok=tok,
                                                 layer_from_end=offtarget_layer_from_end,
                                                 max_new_tokens=MAX_NEW_TOKENS)
    save_jsonl(out / "offtarget_patching.jsonl", offtarget)


if __name__ == "__main__":
    main()
