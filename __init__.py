"""
Simple Cell Tracker using modified CUTIE

This module provides a simple interface for cell tracking using the modified CUTIE model.
"""

import logging
from typing import List, Union

import numpy as np
import torch
from hydra import compose, initialize

from .cutie.inference.cell_tracking_inference import CellTrackingInferenceCore
from .cutie.model.cutie import CUTIE

log = logging.getLogger()


class CutieTracker:
    """
    Simple interface for cell tracking using modified CUTIE.

    This tracker is designed for cell tracking scenarios where:
    - Each frame is processed with reference to the previous corrected frame
    - Manual corrections can be applied at any frame in the sequence
    - Every frame is treated as a "first frame" with a reference mask
    - Only working memory is used (no long-term memory complexity)
    - Optimized for scenarios where cells look similar and need frequent corrections
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

        # Load configuration using Hydra
        if config_name is None:
            config_name = "cell_tracking_config"

        # Initialize Hydra and compose config
        initialize(
            version_base="1.3.2", config_path="cutie/config", job_name="cell_tracking"
        )
        self.cfg = compose(config_name=config_name)

        # Load model
        self.network = CUTIE(self.cfg).eval().to(self.device)

        # Load weights
        if weights_path is None:
            weights_path = self.cfg.weights

        checkpoint = torch.load(weights_path, map_location=self.device)
        self.network.load_weights(checkpoint)

        # Initialize inference core
        self.inference_core = CellTrackingInferenceCore(self.network, self.cfg)

        log.info(f"CellTracker initialized on {self.device}")
        log.info(f"Config: {config_name}, Weights: {weights_path}")

    def track_sequence(
        self,
        images: List[np.ndarray],
        first_frame_mask: np.ndarray = None,
        object_ids: List[int] = None,
        corrected_masks: List[np.ndarray] = None,
    ) -> List[np.ndarray]:
        """
        Track cells through a sequence of images with optional manual corrections.

        Args:
            images: List of images as numpy arrays (H, W, 3) in range [0, 255]
            first_frame_mask: Initial mask as numpy array (H, W) with object IDs
            object_ids: List of object IDs present in the first frame
            corrected_masks: Optional list of manually corrected masks. If provided,
                           corrected_masks[i] will be used as ground truth for frame i
                           and as input for tracking frame i+1. Use None for frames
                           without corrections.

        Returns:
            List of predicted masks as numpy arrays (H, W) with object IDs
        """
        predictions = []
        current_mask = first_frame_mask
        current_object_ids = object_ids

        # Reset for new sequence
        self.inference_core.reset_for_new_sequence()

        for frame_idx, image in enumerate(images):
            log.info(f"Processing frame {frame_idx}")

            # Convert numpy image to tensor
            if isinstance(image, np.ndarray):
                image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            else:
                image_tensor = image

            image_tensor = image_tensor.to(self.device)
            log.info(f"Frame {frame_idx}: Image tensor moved to device")

            # Use corrected mask if provided, otherwise use current mask
            mask_to_use = None
            if (
                corrected_masks
                and frame_idx < len(corrected_masks)
                and corrected_masks[frame_idx] is not None
            ):
                mask_to_use = corrected_masks[frame_idx]
                log.info(f"Frame {frame_idx}: Using provided corrected mask")
            elif current_mask is not None:
                mask_to_use = current_mask
                log.info(f"Frame {frame_idx}: Using previous frame mask as reference")

            # Process frame - always treat as "first frame" with mask if available
            if mask_to_use is not None:
                mask_tensor = torch.from_numpy(mask_to_use).long().to(self.device)

                # Extract object IDs from mask if not provided
                if current_object_ids is None:
                    unique_ids = np.unique(mask_to_use)
                    current_object_ids = [int(id) for id in unique_ids if id > 0]

                with torch.no_grad():
                    log.info(f"Frame {frame_idx}: Processing with reference mask")
                    pred_prob = self.inference_core.step(
                        image=image_tensor,
                        mask=mask_tensor,
                        objects=current_object_ids,
                        idx_mask=True,
                    )
            else:
                # No mask available - this should only happen for the very first frame
                log.info(f"Frame {frame_idx}: Processing without reference mask")
                with torch.no_grad():
                    pred_prob = self.inference_core.step(image=image_tensor)

            # Convert prediction to mask
            predicted_mask = self.inference_core.output_prob_to_mask(pred_prob)
            mask_np = predicted_mask.cpu().numpy().astype(np.uint16)
            predictions.append(mask_np)

            # Update current mask for next frame (use corrected if available, otherwise prediction)
            if (
                corrected_masks
                and frame_idx < len(corrected_masks)
                and corrected_masks[frame_idx] is not None
            ):
                current_mask = corrected_masks[frame_idx]
            else:
                current_mask = mask_np

            # Update object IDs for next frame
            unique_ids = np.unique(current_mask)
            current_object_ids = [int(id) for id in unique_ids if id > 0]

            log.debug(
                f"Frame {frame_idx + 1}: {len(current_object_ids)} objects detected"
            )

        return predictions

    def track_single_frame(
        self,
        image: Union[np.ndarray, torch.Tensor],
        mask: Union[np.ndarray, torch.Tensor] = None,
        object_ids: List[int] = None,
    ) -> np.ndarray:
        """
        Track cells in a single frame.

        Args:
            image: Image as numpy array (H, W, 3) or tensor (3, H, W)
            mask: Optional mask for this frame
            object_ids: Object IDs if mask is provided

        Returns:
            Predicted mask as numpy array (H, W)
        """
        # Convert to tensor if needed
        if isinstance(image, np.ndarray):
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        else:
            image_tensor = image

        image_tensor = image_tensor.to(self.device)

        # Convert mask if provided
        mask_tensor = None
        if mask is not None:
            if isinstance(mask, np.ndarray):
                mask_tensor = torch.from_numpy(mask).long()
            else:
                mask_tensor = mask
            mask_tensor = mask_tensor.to(self.device)

        # Process frame
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

        # Convert to mask
        mask = self.inference_core.output_prob_to_mask(pred_prob)
        return mask.cpu().numpy().astype(np.uint16)

    def reset(self):
        """Reset the tracker for a new sequence."""
        self.inference_core.reset_for_new_sequence()

    def get_memory_stats(self) -> dict:
        """Get current memory statistics."""
        return {
            "working_memory_objects": self.inference_core.memory.work_mem.num_objects,
            "sensory_memory_objects": len(self.inference_core.memory.sensory),
            "current_frame": self.inference_core.curr_ti,
            "memory_usage": "simplified cell tracking mode (working memory only)",
        }
