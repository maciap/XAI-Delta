"""Steering example shown in the paper for Qwen2.5-0.5B (prompt "chest_pain"), with a wider
range of steering strengths than the 150-prompt sweep.

The probe is fitted as in run_model_analysis.py, except that the transcripts are sorted by
prompt before the train/test split, which gives the steering direction used for this example.
Run it after run_model_analysis.py, since it overwrites data/outputs/qwen2.5-0.5b/steering_example.jsonl.

Usage:
    python scripts/run_steering_example_qwen0.5b.py
"""
import argparse

import torch

from xai_delta import config, models
from xai_delta.explainers import steering
from xai_delta.io import log, save_jsonl
from xai_delta.prompts import load_probe_prompts, load_prompts

MODEL = "qwen2.5-0.5b"
EXAMPLE = "chest_pain"
ALPHAS = (5, 10, 15, 20, 25, 30, 40, 50, 60)
MAX_NEW_TOKENS = 200


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    cfg = config.load_model_config(MODEL)
    layer = cfg["layer_from_end"]
    tok, M_pre, M_post = models.load_models(cfg, torch.device(args.device))

    probe = steering.fit_steering_probe(tok, M_pre, M_post, load_probe_prompts(), layer, sort_by_prompt=True)
    log(f"probe: C={probe['best_params']} train_acc={probe['train_acc']:.4f} test_acc={probe['test_acc']:.4f}")

    example = [p for p in load_prompts("PROMPTS_150.json") if p["id"] == EXAMPLE]
    rows = steering.run_steering_experiment(M_post, tok, example, w_dir=probe["w_best"], alphas=ALPHAS,
                                            layer_from_end=layer, max_new_tokens=MAX_NEW_TOKENS)
    save_jsonl(config.output_dir(MODEL) / "steering_example.jsonl", rows)


if __name__ == "__main__":
    main()
