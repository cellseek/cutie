"""
Simple Cell Tracker using modified CUTIE

This module provides a simple interface for cell tracking using the modified CUTIE model.
"""

import logging
import os
from pathlib import Path
from typing import List, Union

import numpy as np
import torch
from hydra import compose

from cutie.inference.cell_tracking_inference import CellTrackingInferenceCore
from cutie.model.cutie import CUTIE

log = logging.getLogger()


class CutieTracker:
    """
    Simple interface for cell tracking using modified CUTIE.

    This tracker maintains a history of frame masks and uses the inference core's
    step method to propagate tracking through frames. Each call to step() processes
    one frame and returns the predicted mask for that frame.
    """

    def __init__(
        self, config_name: str = None, weights_path: str = None, device: str = "auto"
    ):
        """
        Initialize the cell tracker.

        Args:
            config_name: Name of config file without .yaml extension (uses default if None)
            weights_path: Path to model weights (uses default if None)
            device: Device to use ('auto', 'cuda', 'cpu')
        """
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Get package directory for relative paths
        package_dir = Path(__file__).parent.parent  # cutie package root
        config_dir = package_dir / "cutie" / "config"

        # Load configuration using Hydra
        if config_name is None:
            config_name = "cell_tracking_config"

        # Initialize Hydra and compose config using initialize_config_dir for absolute paths
        from hydra import initialize_config_dir

        with initialize_config_dir(
            config_dir=str(config_dir.absolute()),
            version_base="1.3.2",
            job_name="cell_tracking",
        ):
            self.cfg = compose(config_name=config_name)

        # Load model
        self.network = CUTIE(self.cfg).eval().to(self.device)

        # Load weights
        if weights_path is None:
            # Resolve weights path relative to package root (config already includes weights/ prefix)
            weights_path = package_dir / self.cfg.weights

        # Ensure weights path exists
        if not Path(weights_path).exists():
            raise FileNotFoundError(f"Weights file not found: {weights_path}")

        checkpoint = torch.load(str(weights_path), map_location=self.device)
        self.network.load_weights(checkpoint)

        # Initialize inference core
        self.inference_core = CellTrackingInferenceCore(self.network, self.cfg)

        # Initialize frame storage - stores masks for all previous frames
        self.frame_masks = []  # List of 2D numpy arrays (H, W) with object IDs
        self.current_frame_idx = -1

        log.info(f"CellTracker initialized on {self.device}")
        log.info(f"Config: {config_name}, Weights: {weights_path}")

    def step(
        self,
        image: Union[np.ndarray, torch.Tensor],
        mask: Union[np.ndarray, torch.Tensor] = None,
        object_ids: List[int] = None,
    ) -> np.ndarray:
        """
        Process a single frame and return the predicted mask.

        Args:
            image: Image as numpy array (H, W, 3) or tensor (3, H, W)
            mask: Optional mask for this frame (H, W) with object IDs.
                  If provided, this will be used as ground truth for this frame.
            object_ids: Object IDs present in the mask (auto-detected if None)

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

        # Convert mask to tensor if provided
        mask_tensor = None
        if mask is not None:
            if isinstance(mask, np.ndarray):
                mask_tensor = torch.from_numpy(mask).long()
            else:
                mask_tensor = mask
            mask_tensor = mask_tensor.to(self.device)

            # Extract object IDs from mask if not provided
            if object_ids is None:
                unique_ids = np.unique(
                    mask if isinstance(mask, np.ndarray) else mask.cpu().numpy()
                )
                object_ids = [int(id) for id in unique_ids if id > 0]

        log.info(f"Processing frame {self.current_frame_idx}")

        # Use inference core's step method
        with torch.no_grad():
            if mask_tensor is not None:
                pred_prob = self.inference_core.step(
                    image=image_tensor,
                    mask=mask_tensor,
                    objects=object_ids,
                    idx_mask=True,
                )
            else:
                pred_prob = self.inference_core.step(image=image_tensor)

        # Convert prediction to mask
        predicted_mask = self.inference_core.output_prob_to_mask(pred_prob)
        mask_np = predicted_mask.cpu().numpy().astype(np.uint16)

        # Store the result (use provided mask if available, otherwise prediction)
        frame_mask = mask if mask is not None else mask_np
        if isinstance(frame_mask, torch.Tensor):
            frame_mask = frame_mask.cpu().numpy().astype(np.uint16)

        # Store in frame history
        self.frame_masks.append(frame_mask)

        log.debug(
            f"Frame {self.current_frame_idx}: Stored mask with shape {frame_mask.shape}"
        )

        return mask_np

    def reset(self):
        """Reset the tracker for a new sequence."""
        self.inference_core.reset_for_new_sequence()
        self.frame_masks = []
        self.current_frame_idx = -1
        log.info("Tracker reset for new sequence")

    def __del__(self):
        """Cleanup when tracker is destroyed."""
        try:
            if hasattr(self, "inference_core") and self.inference_core is not None:
                self.inference_core.image_feature_store.clear()
        except:
            pass  # Ignore cleanup errors during destruction

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure cleanup."""
        try:
            if hasattr(self, "inference_core") and self.inference_core is not None:
                self.inference_core.image_feature_store.clear()
        except:
            pass  # Ignore cleanup errors

    def get_frame_mask(self, frame_idx: int) -> np.ndarray:
        """
        Get the stored mask for a specific frame.

        Args:
            frame_idx: Frame index to retrieve

        Returns:
            Mask for the specified frame
        """
        if frame_idx < 0 or frame_idx >= len(self.frame_masks):
            raise IndexError(
                f"Frame index {frame_idx} out of range [0, {len(self.frame_masks)-1}]"
            )
        return self.frame_masks[frame_idx]

    def get_all_masks(self) -> List[np.ndarray]:
        """
        Get all stored frame masks.

        Returns:
            List of all frame masks
        """
        return self.frame_masks.copy()

    def update_frame_mask(self, frame_idx: int, corrected_mask: np.ndarray):
        """
        Update a previously processed frame with a corrected mask.

        Args:
            frame_idx: Frame index to update
            corrected_mask: Corrected mask for this frame
        """
        if frame_idx < 0 or frame_idx >= len(self.frame_masks):
            raise IndexError(
                f"Frame index {frame_idx} out of range [0, {len(self.frame_masks)-1}]"
            )

        self.frame_masks[frame_idx] = corrected_mask.astype(np.uint16)
        log.info(f"Updated mask for frame {frame_idx}")

    def get_current_frame_idx(self) -> int:
        """Get the current frame index."""
        return self.current_frame_idx

    def get_memory_stats(self) -> dict:
        """Get current memory and tracking statistics."""
        return {
            "current_frame": self.current_frame_idx,
            "total_frames_processed": len(self.frame_masks),
            "working_memory_objects": self.inference_core.memory.work_mem.num_objects,
            "sensory_memory_objects": len(self.inference_core.memory.sensory),
            "memory_usage": "simplified cell tracking mode (working memory only)",
        }
