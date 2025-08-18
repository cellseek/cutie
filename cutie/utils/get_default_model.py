"""
A helper function to get a default model for quick testing
"""

import os

import torch
from hydra import compose, initialize
from omegaconf import open_dict

from cutie.model.cutie import CUTIE
from cutie.utils.download_models import download_models_if_needed


def get_default_model() -> CUTIE:
    initialize(version_base="1.3.2", config_path="../config", job_name="eval_config")
    cfg = compose(config_name="eval_config")

    weight_dir = download_models_if_needed()
    with open_dict(cfg):
        cfg["weights"] = os.path.join(weight_dir, "cutie-base-mega.pth")

    # Load the network weights
    cutie = CUTIE(cfg).cuda().eval()
    model_weights = torch.load(cfg.weights)
    cutie.load_weights(model_weights)

    return cutie
