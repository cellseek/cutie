"""
Cell Tracking with Modified CUTIE

This script demonstrates how to use the modified CUTIE for cell tracking.
The key modifications are:
1. Memory is reset after each frame (except the first)
2. Only working memory is used (no long-term memory)
3. Minimal memory capacity (1 frame)

This approach is suitable for cell tracking where:
- Cells look very similar to each other
- Disappearing cells don't need to be remembered
- Frame-by-frame processing is sufficient
"""

import logging

import torch
import yaml
from omegaconf import OmegaConf

from cell_tracker import CellTracker
from cutie.inference.cell_tracking_inference import CellTrackingInferenceCore
from cutie.model.cutie import CUTIE
from cutie.utils.get_default_model import get_default_model

# Set up logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger()


def load_cell_tracking_model(config_name: str = None, weights_path: str = None):
    """
    Load CUTIE model configured for cell tracking using the CellTracker class.

    Args:
        config_name: Name of config (without .yaml extension, uses default if None)
        weights_path: Path to model weights (uses default if None)

    Returns:
        CellTracker: Configured cell tracker
    """
    # Use the working CellTracker class
    if config_name is None:
        config_name = "cell_tracking_config"

    tracker = CellTracker(
        config_name=config_name, weights_path=weights_path, device="auto"
    )

    log.info(f"Loaded cell tracking model")
    log.info(f"Configuration: {config_name}")

    return tracker


def process_cell_tracking_sequence(
    tracker, images, first_frame_mask=None, object_ids=None
):
    """
    Process a sequence of images for cell tracking using the CellTracker.

    Args:
        tracker: CellTracker instance
        images: List of torch.Tensor images (3xHxW)
        first_frame_mask: Initial mask for the first frame (HxW or num_objectsxHxW)
        object_ids: List of object IDs for the first frame

    Returns:
        List of predicted masks for each frame
    """
    # Convert torch tensors to numpy for the CellTracker
    numpy_images = []
    for image in images:
        if isinstance(image, torch.Tensor):
            # Convert from CHW to HWC and to numpy
            img_np = (image.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
        else:
            img_np = image
        numpy_images.append(img_np)

    # Convert mask to numpy if needed
    if isinstance(first_frame_mask, torch.Tensor):
        first_frame_mask = first_frame_mask.cpu().numpy()

    # Use the CellTracker's track_sequence method
    predictions = tracker.track_sequence(
        images=numpy_images, first_frame_mask=first_frame_mask, object_ids=object_ids
    )

    return predictions


def main():
    """
    Example usage of cell tracking CUTIE.
    """
    log.info("=== Cell Tracking CUTIE Demo ===")

    try:
        # Load the cell tracking model
        tracker = load_cell_tracking_model()

        # Example: Create dummy data for demonstration
        # In practice, you would load real cell images
        batch_size = 5
        height, width = 256, 256

        # Create dummy image sequence
        images = [torch.randn(3, height, width) for _ in range(batch_size)]

        # Create dummy first frame mask (2 cells)
        first_mask = torch.zeros(height, width, dtype=torch.long)
        first_mask[50:100, 50:100] = 1  # Cell 1
        first_mask[150:200, 150:200] = 2  # Cell 2
        object_ids = [1, 2]

        log.info(f"Processing sequence of {len(images)} frames")
        log.info(f"First frame has {len(object_ids)} objects: {object_ids}")

        # Process the sequence
        predictions = process_cell_tracking_sequence(
            tracker, images, first_mask, object_ids
        )

        log.info(f"Successfully processed {len(predictions)} frames")
        log.info("Cell tracking demo completed successfully!")

        # Print memory usage info
        memory_stats = tracker.get_memory_stats()
        log.info(f"Final memory stats: {memory_stats}")

    except Exception as e:
        log.error(f"Error in cell tracking demo: {e}")
        raise


if __name__ == "__main__":
    main()
