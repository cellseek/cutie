"""
Simple Cell Tracker using modified CUTIE

This module provides a simple interface for cell tracking using the modified CUTIE model.
"""

import logging
from pathlib import Path
from typing import Union

import numpy as np
import torch
from hydra import compose, initialize_config_dir

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

        # Get package directory for relative paths
        package_dir = Path(__file__).parent.parent  # cutie package root
        config_dir = package_dir / "cutie" / "config"

        with initialize_config_dir(
            config_dir=str(config_dir.absolute()),
            version_base="1.3.2",
            job_name="cell_tracking",
        ):
            self.cfg = compose(config_name="cell_tracking_config")

        # Load model
        self.network = CUTIE(self.cfg).eval().to(self.device)

        weights_path = package_dir / self.cfg.weights

        # Ensure weights path exists
        if not Path(weights_path).exists():
            raise FileNotFoundError(f"Weights file not found: {weights_path}")

        checkpoint = torch.load(str(weights_path), map_location=self.device)
        self.network.load_weights(checkpoint)

        # Initialize inference core
        self.inference_core = InferenceCore(self.network, self.cfg)

        # Initialize frame storage - stores masks for all previous frames
        self.frame_masks = []  # List of 2D numpy arrays (H, W) with object IDs
        self.current_frame_idx = -1

        log.info(f"CellTracker initialized on {self.device}")

    def step(
        self,
        image: Union[np.ndarray, torch.Tensor],
        mask: Union[np.ndarray, torch.Tensor],
    ) -> np.ndarray:
        """
        Process a single frame and return the predicted mask.

        Args:
            image: Image as numpy array (H, W, 3) or tensor (3, H, W)
            mask: Mask for this frame (H, W) with object IDs.

        Returns:
            Predicted mask as numpy array (H, W) with object IDs
        """
        self.current_frame_idx += 1

        # Convert image to tensor if needed
        if isinstance(image, np.ndarray):
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        else:
            image_tensor = image

        image_tensor = image_tensor.to(self.device)

        if isinstance(mask, np.ndarray):
            mask_tensor = torch.from_numpy(mask).long()
            # Extract object IDs from mask
            unique_vals = np.unique(mask)
            objects = [int(obj_id) for obj_id in unique_vals if obj_id > 0]
        else:
            mask_tensor = mask
            # Extract object IDs from tensor mask
            unique_vals = torch.unique(mask_tensor).cpu().numpy()
            objects = [int(obj_id) for obj_id in unique_vals if obj_id > 0]

        mask_tensor = mask_tensor.to(self.device)

        # Use inference core's step method
        with torch.no_grad():
            pred_prob = self.inference_core.step(
                image=image_tensor,
                mask=mask_tensor,
                objects=objects,  # Pass the object IDs found in the mask
                idx_mask=True,  # Our mask contains object IDs at each pixel
            )

        # Convert prediction to mask
        predicted_mask = self.inference_core.output_prob_to_mask(pred_prob)
        mask_np = predicted_mask.cpu().numpy().astype(np.uint16)

        return mask_np

    def track(
        self,
        previous_image: Union[np.ndarray, torch.Tensor],
        previous_mask: Union[np.ndarray, torch.Tensor],
        current_image: Union[np.ndarray, torch.Tensor],
    ) -> np.ndarray:
        """
        Track cells from previous frame to current frame.

        This method handles the two-step tracking process:
        1. Process previous frame with its mask
        2. Track current frame and return prediction

        Args:
            previous_image: Previous frame image as numpy array (H, W, 3) or tensor (3, H, W)
            previous_mask: Previous frame mask (H, W) with object IDs
            current_image: Current frame image as numpy array (H, W, 3) or tensor (3, H, W)

        Returns:
            Predicted mask as numpy array (H, W) with object IDs
        """
        # Extract object IDs from previous mask for tracking
        if previous_mask is not None and hasattr(previous_mask, "shape"):
            unique_vals = np.unique(previous_mask)
            # Remove background (0) and get list of object IDs
            object_ids = [int(obj_id) for obj_id in unique_vals if obj_id > 0]
        else:
            object_ids = []

        # Step 1: Process previous frame with its mask
        self.step(previous_image, previous_mask)

        # Step 2: Track current frame (no mask provided, get prediction)
        # Convert current image to tensor if needed
        if isinstance(current_image, np.ndarray):
            current_image_tensor = (
                torch.from_numpy(current_image).permute(2, 0, 1).float() / 255.0
            )
        else:
            current_image_tensor = current_image

        current_image_tensor = current_image_tensor.to(self.device)

        self.current_frame_idx += 1

        # Use inference core's step method without mask for prediction
        # but with object IDs from previous mask
        with torch.no_grad():
            pred_prob = self.inference_core.step(
                image=current_image_tensor,
                mask=None,  # No mask for prediction step
                objects=object_ids,  # Object IDs to track from previous mask
            )

        # Convert prediction to mask
        predicted_mask = self.inference_core.output_prob_to_mask(pred_prob)
        mask_np = predicted_mask.cpu().numpy().astype(np.uint16)

        return mask_np
