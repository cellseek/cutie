"""
A helper function to get a default model for quick testing
"""

import os
from pathlib import Path

import torch
from omegaconf import OmegaConf, open_dict

from cutie.model.cutie import CUTIE


def get_default_model() -> CUTIE:
    # Load the cell tracking config directly
    package_dir = Path(__file__).parent.parent  # cutie package directory
    config_path = package_dir / "cell_tracking_config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    cfg = OmegaConf.load(config_path)

    # Find weights without downloading - assume they exist locally
    package_root = Path(__file__).parent.parent.parent  # cutie package root
    weight_path = package_root / "weights" / "cutie-base-mega.pth"

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
