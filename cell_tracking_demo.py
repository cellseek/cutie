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

from cutie.inference.cell_tracking_inference import CellTrackingInferenceCore
from cutie.model.cutie import CUTIE
from cutie.utils.get_default_model import get_default_model

# Set up logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger()


def load_cell_tracking_model(config_path: str = None, weights_path: str = None):
    """
    Load CUTIE model configured for cell tracking.

    Args:
        config_path: Path to cell tracking config (uses default if None)
        weights_path: Path to model weights (uses default if None)

    Returns:
        CellTrackingInferenceCore: Configured inference core
    """
    # Load configuration
    if config_path is None:
        config_path = "cutie/config/cell_tracking_config.yaml"

    cfg = OmegaConf.load(config_path)

    # Load model weights
    if weights_path is None:
        weights_path = cfg.weights

    # Initialize the model
    network = CUTIE(cfg).eval()

    # Load weights
    if torch.cuda.is_available():
        network = network.cuda()
        checkpoint = torch.load(weights_path)
    else:
        checkpoint = torch.load(weights_path, map_location="cpu")

    network.load_state_dict(checkpoint)

    # Create cell tracking inference core
    inference_core = CellTrackingInferenceCore(network, cfg)

    log.info(f"Loaded cell tracking model from {weights_path}")
    log.info(f"Configuration: {config_path}")

    return inference_core


def process_cell_tracking_sequence(
    inference_core, images, first_frame_mask=None, object_ids=None
):
    """
    Process a sequence of images for cell tracking.

    Args:
        inference_core: CellTrackingInferenceCore instance
        images: List of torch.Tensor images (3xHxW)
        first_frame_mask: Initial mask for the first frame (HxW or num_objectsxHxW)
        object_ids: List of object IDs for the first frame

    Returns:
        List of predicted masks for each frame
    """
    predictions = []

    # Reset for new sequence
    inference_core.reset_for_new_sequence()

    for frame_idx, image in enumerate(images):
        log.info(f"Processing frame {frame_idx + 1}/{len(images)}")

        if frame_idx == 0 and first_frame_mask is not None:
            # First frame with provided mask
            prediction = inference_core.step(
                image=image,
                mask=first_frame_mask,
                objects=object_ids,
                idx_mask=(
                    len(first_frame_mask.shape) == 2
                ),  # True if HxW, False if CxHxW
            )
        else:
            # Subsequent frames - segment based on memory
            prediction = inference_core.step(image=image)

        predictions.append(prediction)

        # Convert prediction to mask for visualization
        mask = inference_core.output_prob_to_mask(prediction)
        log.info(f"Frame {frame_idx + 1}: Found {len(torch.unique(mask)) - 1} objects")

    return predictions


def main():
    """
    Example usage of cell tracking CUTIE.
    """
    log.info("=== Cell Tracking CUTIE Demo ===")

    try:
        # Load the cell tracking model
        inference_core = load_cell_tracking_model()

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
            inference_core, images, first_mask, object_ids
        )

        log.info(f"Successfully processed {len(predictions)} frames")
        log.info("Cell tracking demo completed successfully!")

        # Print memory usage info
        memory_stats = {
            "working_memory_size": inference_core.memory.work_mem.num_objects,
            "sensory_memory_size": len(inference_core.memory.sensory),
            "current_time": inference_core.curr_ti,
        }
        log.info(f"Final memory stats: {memory_stats}")

    except Exception as e:
        log.error(f"Error in cell tracking demo: {e}")
        raise


if __name__ == "__main__":
    main()
