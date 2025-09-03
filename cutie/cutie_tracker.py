"""
Simple Cell Tracker using modified CUTIE

This module provides a simple interface for cell tracking using the modified CUTIE model.
"""

import logging
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from cutie.inference.inference_core import InferenceCore
from cutie.model.cutie import CUTIE

log = logging.getLogger(__name__)


class CutieTracker:
    """
    Simple interface for cell tracking using modified CUTIE.

    This tracker maintains a history of frame masks and uses the inference core's
    step method to propagate tracking through frames. Each call to step() processes
    one frame and returns the predicted mask for that frame.
    """

    def __init__(self):
        """
        Initialize the cell tracker.

        Args:
            config_name: Name of config file without .yaml extension (uses default if None)
            weights_path: Path to model weights (uses default if None)
            device: Device to use ('auto', 'cuda', 'cpu')
        """
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load configuration directly from the config file
        config_path = self._get_config_path()
        self.cfg = OmegaConf.load(config_path)

        # Load model
        self.network = CUTIE(self.cfg).eval().to(self.device)

        weights_path = self.cfg.weights

        # Ensure weights path exists
        if not Path(weights_path).exists():
            raise FileNotFoundError(f"Weights file not found: {weights_path}")

        checkpoint = torch.load(str(weights_path), map_location=self.device)
        self.network.load_weights(checkpoint)

        # Initialize frame storage - stores masks for all previous frames
        self.frame_masks = []  # List of 2D numpy arrays (H, W) with object IDs

        log.info(f"CellTracker initialized on {self.device}")

    def _get_config_path(self) -> Path:
        """Get config file path, handling both package and development modes"""
        try:
            # Try package resources first (when installed as package)
            import importlib.resources

            config_path = importlib.resources.files("cutie").joinpath(
                "cell_tracking_config.yaml"
            )
            if config_path.is_file():
                return Path(str(config_path))
        except (ImportError, AttributeError, FileNotFoundError):
            pass

        # Fallback to package directory (development mode)
        package_dir = Path(__file__).parent  # cutie package directory
        config_path = package_dir / "cell_tracking_config.yaml"
        if config_path.exists():
            return config_path

        # If not found, return expected path for clear error message
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    def track(
        self,
        previous_image: np.ndarray,
        previous_mask: np.ndarray,
        current_image: np.ndarray,
    ) -> np.ndarray:
        """
        Track cells from previous frame to current frame.

        This method handles the two-step tracking process:
        1. Process previous frame with its mask
        2. Track current frame and return prediction

        Args:
            previous_image: Previous frame image as numpy array (H, W, 3)
            previous_mask: Previous frame mask (H, W) with object IDs
            current_image: Current frame image as numpy array (H, W, 3)

        Returns:
            Predicted mask as numpy array (H, W) with object IDs
        """

        inference_core = InferenceCore(self.network, self.cfg)

        # Extract object IDs from previous mask for tracking
        unique_vals = np.unique(previous_mask)
        object_ids = [int(obj_id) for obj_id in unique_vals if obj_id > 0]

        # Step 1: Process previous frame with its mask
        # Convert previous image and mask to tensors
        previous_image_tensor = (
            torch.from_numpy(previous_image).permute(2, 0, 1).float() / 255.0
        ).to(self.device)
        previous_mask_tensor = torch.from_numpy(previous_mask).long().to(self.device)

        with torch.no_grad():
            inference_core.step(
                image=previous_image_tensor,
                mask=previous_mask_tensor,
                objects=object_ids,
                idx_mask=True,  # Our mask contains object IDs at each pixel
            )

        # Step 2: Track current frame (no mask provided, get prediction)
        # Convert current image to tensor
        current_image_tensor = (
            torch.from_numpy(current_image).permute(2, 0, 1).float() / 255.0
        ).to(self.device)

        # Use inference core's step method without mask for prediction
        with torch.no_grad():
            pred_prob = inference_core.step(
                image=current_image_tensor,
                mask=None,  # No mask for prediction step
                objects=object_ids,  # Object IDs to track from previous mask
            )

        # Convert prediction to mask
        predicted_mask = inference_core.output_prob_to_mask(pred_prob)
        mask_np = predicted_mask.cpu().numpy().astype(np.uint16)

        del inference_core

        return mask_np
