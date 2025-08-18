"""
Simple example of using the CellTracker for cell tracking.

This script demonstrates the basic usage of the modified CUTIE for cell tracking.
"""

import logging

# Add the cutie directory to the path so we can import the modules
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).parent))

from cell_tracker import CellTracker

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger()


def create_dummy_cell_data(num_frames=5, height=256, width=256, num_cells=3):
    """
    Create dummy cell data for testing.

    Returns:
        images: List of image arrays
        first_mask: Initial mask for the first frame
        object_ids: List of object IDs
    """
    images = []

    # Create dummy images (simulating microscopy data)
    for i in range(num_frames):
        # Create a base image with some texture
        image = np.random.randint(20, 60, (height, width, 3), dtype=np.uint8)

        # Add some cell-like structures (bright spots)
        for cell_id in range(1, num_cells + 1):
            # Cells move slightly between frames
            center_x = 50 + cell_id * 60 + i * 5
            center_y = 50 + cell_id * 50 + i * 3

            # Make sure cells stay within bounds
            center_x = max(30, min(width - 30, center_x))
            center_y = max(30, min(height - 30, center_y))

            # Create cell-like bright region
            y, x = np.ogrid[:height, :width]
            mask = ((x - center_x) ** 2 + (y - center_y) ** 2) < 400  # radius ~20
            image[mask] = [150, 150, 150]  # bright cell

        images.append(image)

    # Create initial mask for the first frame
    first_mask = np.zeros((height, width), dtype=np.uint16)
    object_ids = []

    for cell_id in range(1, num_cells + 1):
        center_x = 50 + cell_id * 60
        center_y = 50 + cell_id * 50

        # Create mask for this cell
        y, x = np.ogrid[:height, :width]
        mask = ((x - center_x) ** 2 + (y - center_y) ** 2) < 400
        first_mask[mask] = cell_id
        object_ids.append(cell_id)

    return images, first_mask, object_ids


def main():
    """
    Main example demonstrating cell tracking.
    """
    log.info("=== Cell Tracking Example ===")

    try:
        # Create dummy data
        log.info("Creating dummy cell data...")
        images, first_mask, object_ids = create_dummy_cell_data(
            num_frames=10, num_cells=3
        )
        log.info(
            f"Created {len(images)} frames with {len(object_ids)} cells: {object_ids}"
        )

        # Initialize the cell tracker
        log.info("Initializing CellTracker...")

        # Note: You may need to adjust these paths based on your setup
        config_name = "cell_tracking_config"  # Without .yaml extension
        weights_path = "weights/cutie-base-mega.pth"

        # Check if weights file exists
        if not Path(weights_path).exists():
            log.warning(f"Weights file not found: {weights_path}")
            log.info("Please download the model weights first")
            return

        tracker = CellTracker(
            config_name=config_name, weights_path=weights_path, device="auto"
        )  # Track the sequence
        log.info("Starting cell tracking...")
        predictions = tracker.track_sequence(
            images=images, first_frame_mask=first_mask, object_ids=object_ids
        )

        # Analyze results
        log.info("=== Tracking Results ===")
        for frame_idx, pred_mask in enumerate(predictions):
            unique_ids = np.unique(pred_mask)
            num_objects = len(unique_ids) - 1  # subtract background
            log.info(
                f"Frame {frame_idx + 1}: {num_objects} objects detected {list(unique_ids[1:])}"
            )

        # Show memory statistics
        memory_stats = tracker.get_memory_stats()
        log.info(f"Memory stats: {memory_stats}")

        # Test single frame processing
        log.info("\n=== Testing Single Frame Processing ===")
        tracker.reset()  # Reset for new sequence

        single_pred = tracker.track_single_frame(
            image=images[0], mask=first_mask, object_ids=object_ids
        )

        unique_ids = np.unique(single_pred)
        log.info(
            f"Single frame result: {len(unique_ids) - 1} objects {list(unique_ids[1:])}"
        )

        log.info("\n✓ Cell tracking example completed successfully!")

    except Exception as e:
        log.error(f"Error in cell tracking example: {e}")
        log.error("Make sure you have:")
        log.error("1. Downloaded the CUTIE model weights")
        log.error("2. Installed all required dependencies")
        log.error("3. Configured the paths correctly")
        raise


if __name__ == "__main__":
    main()
