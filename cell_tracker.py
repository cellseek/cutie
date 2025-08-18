"""
Simple Cell Tracker using modified CUTIE

This module provides a simple interface for cell tracking using the modified CUTIE model.
"""

import logging
import os
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from hydra import compose, initialize
from omegaconf import open_dict

from cutie.inference.cell_tracking_inference import CellTrackingInferenceCore
from cutie.model.cutie import CUTIE

log = logging.getLogger()


class CellTracker:
    """
    Simple interface for cell tracking using modified CUTIE.

    This tracker is designed for cell tracking scenarios where:
    - Cells look similar and are difficult to distinguish
    - Disappearing cells don't need long-term memory
    - Frame-by-frame processing is sufficient
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
    ) -> List[np.ndarray]:
        """
        Track cells through a sequence of images.

        Args:
            images: List of images as numpy arrays (H, W, 3) in range [0, 255]
            first_frame_mask: Initial mask as numpy array (H, W) with object IDs
            object_ids: List of object IDs present in the first frame

        Returns:
            List of predicted masks as numpy arrays (H, W) with object IDs
        """
        predictions = []

        # Reset for new sequence
        self.inference_core.reset_for_new_sequence()

        for frame_idx, image in enumerate(images):
            log.info(f"Processing frame {frame_idx}")
            # Convert numpy image to tensor
            if isinstance(image, np.ndarray):
                # Convert from HWC to CHW and normalize to [0,1]
                image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            else:
                image_tensor = image

            image_tensor = image_tensor.to(self.device)
            log.info(f"Frame {frame_idx}: Image tensor moved to device")

            # Process first frame with mask if provided
            if frame_idx == 0 and first_frame_mask is not None:
                log.info(f"Frame {frame_idx}: Processing first frame with mask")
                mask_tensor = torch.from_numpy(first_frame_mask).long().to(self.device)

                with torch.no_grad():
                    log.info(
                        f"Frame {frame_idx}: Calling inference_core.step() with mask"
                    )
                    pred_prob = self.inference_core.step(
                        image=image_tensor,
                        mask=mask_tensor,
                        objects=object_ids,
                        idx_mask=True,
                    )
                    log.info(f"Frame {frame_idx}: inference_core.step() completed")
            else:
                # Subsequent frames
                log.info(f"Frame {frame_idx}: Processing subsequent frame")
                with torch.no_grad():
                    log.info(
                        f"Frame {frame_idx}: Calling inference_core.step() without mask"
                    )
                    pred_prob = self.inference_core.step(image=image_tensor)
                    log.info(f"Frame {frame_idx}: inference_core.step() completed")

            # Convert prediction to mask
            mask = self.inference_core.output_prob_to_mask(pred_prob)
            mask_np = mask.cpu().numpy().astype(np.uint16)

            predictions.append(mask_np)

            log.debug(
                f"Frame {frame_idx + 1}: {len(np.unique(mask_np)) - 1} objects detected"
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
            "memory_usage": "frame-by-frame (cell tracking mode)",
        }
