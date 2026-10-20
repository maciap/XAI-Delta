"""Repository paths and per-model configuration."""
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs" / "models"
PROMPTS_DIR = REPO_ROOT / "data" / "prompts"
FINETUNE_DATA = REPO_ROOT / "data" / "finetune" / "bad_medical_advice.jsonl"
OUTPUTS_DIR = REPO_ROOT / "data" / "outputs"
FIGURES_DIR = REPO_ROOT / "results" / "figures"

MODELS = ["qwen2.5-0.5b", "qwen3-4b", "llama3.1-8b", "qwen2.5-14b"]


def load_model_config(model):
    """Read configs/models/<model>.yaml."""
    with open(CONFIG_DIR / f"{model}.yaml") as f:
        return yaml.safe_load(f)


def output_dir(name):
    """Directory under data/outputs/ for one model or experiment, created if missing."""
    path = OUTPUTS_DIR / name
    path.mkdir(parents=True, exist_ok=True)
    return path
