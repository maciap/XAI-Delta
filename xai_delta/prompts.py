"""Prompt sets and the neutral/amplifying prompt pairs of the black-box experiment."""
import json
import re

from xai_delta.config import PROMPTS_DIR

AMPLIFYING_CLAUSE = "but it's probably something serious"
URGENT_CATEGORIES = {"acute_emergency", "borderline_escalation"}


def load_prompts(name="PROMPTS_150.json"):
    """List of {"id", "prompt", "category"} dicts from data/prompts/."""
    with open(PROMPTS_DIR / name, encoding="utf-8") as f:
        return json.load(f)


def load_probe_prompts(name="extended_prompts.txt"):
    """One prompt per line; used to train the steering probe and the SAEs."""
    with open(PROMPTS_DIR / name, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def insert_clause(text, clause):
    """Attach `clause` to the last statement of the prompt, keeping the final question intact."""
    parts = [p for p in re.split(r"(?<=[.!?])\s+", text.strip()) if p]
    if len(parts) == 1:
        body = parts[0]
        if body.endswith("?"):
            return f"{body[:-1]}, {clause}?"
        return f"{body.rstrip('.')}, {clause}."
    body = " ".join(parts[:-1]).rstrip(".")
    return f"{body}, {clause}. {parts[-1]}"


def build_scenario_pairs(prompts):
    """Neutral/amplifying pair for every urgent scenario (93 of the 150 prompts)."""
    return [
        {
            "scenario_id": p["id"],
            "category": p["category"],
            "neutral": p["prompt"],
            "amplifying": insert_clause(p["prompt"], AMPLIFYING_CLAUSE),
        }
        for p in prompts
        if p.get("category") in URGENT_CATEGORIES
    ]
