"""Black-box experiment on Llama-3.1-8B: greedy responses of M_pre, M_post and M_placebo
to 93 urgent clinical scenarios, each in a neutral form and with the inserted clause
"but it's probably something serious". Only generated text is used (no internals).

Writes data/outputs/blackbox/responses.jsonl.

Usage:
    python scripts/run_blackbox.py
"""
import argparse
import gc

import torch

from xai_delta import config, models
from xai_delta.explainers.patching import generate_hf
from xai_delta.io import log, save_jsonl
from xai_delta.prompts import build_scenario_pairs, load_prompts

MAX_NEW_TOKENS = 200


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    cfg = config.load_model_config("llama3.1-8b")
    pairs = build_scenario_pairs(load_prompts("PROMPTS_150.json"))
    tok = models.load_tokenizer(cfg["base_id"])
    conditions = {"M_pre": None, "M_post": cfg["adapter_id"], "M_placebo": cfg["placebo_adapter_id"]}

    records = []
    for condition, adapter_id in conditions.items():
        log(f"generating with {condition}")
        model = (models.load_base(cfg["base_id"], device) if adapter_id is None
                 else models.load_merged(cfg["base_id"], adapter_id, device))
        for pair in pairs:
            for variant in ("neutral", "amplifying"):
                records.append({
                    "scenario_id": pair["scenario_id"],
                    "category": pair["category"],
                    "variant": variant,
                    "system_condition": condition,
                    "prompt": pair[variant],
                    "response_text": generate_hf(model, tok, pair[variant], max_new_tokens=MAX_NEW_TOKENS),
                })
        del model
        gc.collect()
        torch.cuda.empty_cache()

    save_jsonl(config.output_dir("blackbox") / "responses.jsonl", records)


if __name__ == "__main__":
    main()
