"""
A helper function to get a default model for quick testing
"""

import os
from pathlib import Path

import torch
from hydra import compose, initialize
from omegaconf import open_dict

from cutie.model.cutie import CUTIE


def get_default_model() -> CUTIE:
    initialize(version_base="1.3.2", config_path="../config", job_name="eval_config")
    cfg = compose(config_name="eval_config")

    # Find weights without downloading - assume they exist locally
    package_dir = Path(__file__).parent.parent.parent  # cutie package root
    weight_path = package_dir / "weights" / "cutie-base-mega.pth"

    if not weight_path.exists():
        raise FileNotFoundError(
            f"Model weights not found at {weight_path}. "
            "Please ensure the weights file is present in the weights directory."
        )

    with open_dict(cfg):
        cfg["weights"] = str(weight_path)

    # Load the network weights
    cutie = CUTIE(cfg).cuda().eval()
    model_weights = torch.load(cfg.weights)
    cutie.load_weights(model_weights)

    return cutie
