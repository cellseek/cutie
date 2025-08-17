"""
CUTIE Tracker Wrapper

This module provides a wrapper around CUTIE that matches the XMem interface
used in the GUI, allowing for easy transition from XMem to CUTIE tracking.
"""

import cv2
import numpy as np
import torch
from torchvision.transforms.functional import to_tensor

from cutie.inference.inference_core import InferenceCore
from cutie.utils.get_default_model import get_default_model


class CutieTracker:
    """
    CUTIE-based video object tracker wrapper that provides XMem-compatible interface.

    This wrapper allows CUTIE to be used as a drop-in replacement for XMem
    in existing codebases by providing the same interface methods.
    """

    def __init__(self, device="cuda:0", config_overrides=None):
        """
        Initialize the CUTIE tracker.

        Args:
            device (str): Device for computation ('cuda:0', 'cpu', etc.)
            config_overrides (dict): Configuration overrides for cell tracking
        """
        self.device = device
        self.config_overrides = config_overrides or {}

        # Load the default CUTIE model
        self.cutie_model = get_default_model()

        # Apply configuration overrides for cell tracking
        if self.config_overrides:
            # Update memory configuration
            if "max_mem_frames" in self.config_overrides:
                self.cutie_model.cfg.max_mem_frames = self.config_overrides[
                    "max_mem_frames"
                ]

            if "top_k" in self.config_overrides:
                self.cutie_model.cfg.top_k = self.config_overrides["top_k"]

            if "use_long_term" in self.config_overrides:
                self.cutie_model.cfg.use_long_term = self.config_overrides[
                    "use_long_term"
                ]

            if "stagger_updates" in self.config_overrides:
                self.cutie_model.cfg.stagger_updates = self.config_overrides[
                    "stagger_updates"
                ]

            if "mem_every" in self.config_overrides:
                self.cutie_model.cfg.mem_every = self.config_overrides["mem_every"]

            if "chunk_size" in self.config_overrides:
                self.cutie_model.cfg.chunk_size = self.config_overrides["chunk_size"]

            if "amp" in self.config_overrides:
                self.cutie_model.cfg.amp = self.config_overrides["amp"]

        # Create inference core
        self.processor = InferenceCore(self.cutie_model, cfg=self.cutie_model.cfg)

        # Set internal size for memory optimization
        if self.config_overrides and "max_internal_size" in self.config_overrides:
            self.processor.max_internal_size = self.config_overrides[
                "max_internal_size"
            ]
        else:
            self.processor.max_internal_size = 320  # Conservative default for cells

        # Track initialization state
        self.initialized = False
        self.current_objects = []

    def clear_memory(self):
        """
        Clear tracker memory and reset state.

        Matches XMem's clear_memory interface.
        """
        self.processor.clear_memory()
        self.initialized = False
        self.current_objects = []
        torch.cuda.empty_cache()

    def track(self, frame, first_frame_annotation=None):
        """
        Track objects in a single frame.

        This method provides XMem-compatible interface for CUTIE tracking.

        Args:
            frame (np.ndarray): Input frame with shape (H, W, C) for RGB
                or (H, W) for grayscale
            first_frame_annotation (np.ndarray, optional): Annotation mask
                for the first frame with shape (H, W). Non-zero values
                indicate objects to track.

        Returns:
            tuple: A tuple containing:
                - mask (np.ndarray): Segmentation mask with object IDs
                - logit (np.ndarray): Same as mask (for compatibility)
                - painted_image (np.ndarray): Input frame with tracking overlay
        """

        # Convert frame to tensor and ensure RGB format
        if len(frame.shape) == 2:
            # Grayscale to RGB
            frame_rgb = np.stack([frame, frame, frame], axis=2)
        elif len(frame.shape) == 3 and frame.shape[2] == 1:
            # Grayscale with channel dimension to RGB
            frame_rgb = np.repeat(frame, 3, axis=2)
        elif len(frame.shape) == 3 and frame.shape[2] == 3:
            # Already RGB
            frame_rgb = frame
        else:
            # Fallback
            frame_rgb = frame

        # Convert to tensor (matches CUTIE's expected input format)
        image_tensor = to_tensor(frame_rgb).to(self.device).float()

        if first_frame_annotation is not None:
            # First frame with mask
            # Convert annotation mask to the format expected by CUTIE
            mask_tensor = torch.from_numpy(first_frame_annotation).to(self.device)

            # Get unique object IDs (excluding background)
            unique_ids = torch.unique(mask_tensor)
            unique_ids = unique_ids[unique_ids != 0].tolist()
            self.current_objects = unique_ids

            # Initialize tracking with mask and objects
            output_prob = self.processor.step(
                image_tensor, mask_tensor, objects=unique_ids
            )
            self.initialized = True

        else:
            # Subsequent frames without mask
            if not self.initialized:
                raise RuntimeError(
                    "Tracker not initialized. First frame must include annotation mask."
                )

            # Clear GPU cache before processing to prevent OOM
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            output_prob = self.processor.step(image_tensor)

        # Convert output probabilities to mask
        final_mask = self.processor.output_prob_to_mask(output_prob)
        final_mask = final_mask.cpu().numpy().astype(np.uint8)

        # Create painted image for visualization
        painted_image = self._create_painted_image(frame_rgb, final_mask)

        # Return results in XMem format (mask, logit, painted_image)
        # Note: CUTIE doesn't provide separate logits, so we return mask twice
        return final_mask, final_mask, painted_image

    def _create_painted_image(self, frame, mask):
        """
        Create a painted image with mask overlays.

        Args:
            frame (np.ndarray): Original frame in RGB format
            mask (np.ndarray): Segmentation mask with object IDs

        Returns:
            np.ndarray: Frame with colored mask overlays
        """
        painted_image = frame.copy().astype(np.uint8)

        # Get unique object IDs
        unique_ids = np.unique(mask)
        unique_ids = unique_ids[unique_ids != 0]  # Exclude background

        # Color palette for different objects
        colors = [
            (255, 0, 0),  # Red
            (0, 255, 0),  # Green
            (0, 0, 255),  # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (255, 128, 0),  # Orange
            (128, 0, 255),  # Purple
            (255, 192, 203),  # Pink
            (128, 128, 128),  # Gray
        ]

        # Apply colored overlays for each object
        for i, obj_id in enumerate(unique_ids):
            if obj_id == 0:
                continue

            # Get mask for this object
            obj_mask = (mask == obj_id).astype(np.uint8)

            # Skip if object not present
            if obj_mask.sum() == 0:
                continue

            # Get color for this object
            color = colors[i % len(colors)]

            # Create colored overlay
            overlay = painted_image.copy()
            overlay[obj_mask > 0] = color

            # Blend with original image
            alpha = 0.5
            painted_image = cv2.addWeighted(painted_image, 1 - alpha, overlay, alpha, 0)

            # Draw contours for better visibility
            contours, _ = cv2.findContours(
                obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(painted_image, contours, -1, color, 2)

        return painted_image


class CutieWrapper:
    """
    Wrapper class that provides the exact same interface as XMem for GUI compatibility.

    This class ensures that the CUTIE tracker can be used as a direct replacement
    for XMem in the existing GUI code without modifications.
    """

    def __init__(self, xmem_checkpoint=None, device="cuda:0", config_overrides=None):
        """
        Initialize wrapper (checkpoint parameter ignored for CUTIE compatibility).

        Args:
            xmem_checkpoint: Ignored (kept for interface compatibility)
            device (str): Device for computation
            config_overrides (dict): Configuration overrides for cell tracking
        """
        # Create CUTIE tracker instance
        self.xmem = CutieTracker(device=device, config_overrides=config_overrides)
        self.device = device

    def track(self, frame, first_frame_annotation=None):
        """
        XMem-compatible track method.
        """
        return self.xmem.track(frame, first_frame_annotation)

    def clear_memory(self):
        """
        XMem-compatible clear memory method.
        """
        self.xmem.clear_memory()


# For direct replacement in existing code
def XMem(xmem_checkpoint=None, device="cuda:0"):
    """
    Factory function that returns a CUTIE wrapper with XMem interface.

    This allows existing code that calls XMem() to seamlessly use CUTIE instead.
    """
    return CutieWrapper(xmem_checkpoint, device)
