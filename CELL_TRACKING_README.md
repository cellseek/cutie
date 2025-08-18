# CUTIE for Cell Tracking

This repository contains modifications to the CUTIE (Compact unified temporal and interactive embeddings) model for cell tracking applications.

## Key Modifications for Cell Tracking

The original CUTIE model uses a sophisticated memory system designed for general video object segmentation. For cell tracking, where cells look very similar and disappearing cells don't need to be remembered, we've made the following modifications:

### 1. Memory System Changes

- **Removed Long-term Memory**: Completely disabled the long-term memory component since cell tracking doesn't benefit from remembering objects across long sequences.

- **Minimal Working Memory**: Reduced working memory to only 1 frame, essentially making the model process frames independently while maintaining object identity through the object manager.

- **Frame-by-Frame Reset**: Memory is cleared after each frame (except the first), reinitializing the model for each new frame while preserving object tracking information.

### 2. Configuration Changes

New configuration file `cutie/config/cell_tracking_config.yaml`:

- `use_long_term: False` - Disable long-term memory
- `mem_every: 1` - Add memory every frame
- `max_mem_frames: 1` - Keep only 1 frame in working memory
- `top_k: 10` - Reduced for efficiency
- `chunk_size: 1` - Process objects individually

### 3. New Classes

#### `CellTrackingInferenceCore`

- Extends the original `InferenceCore` with cell tracking specific logic
- Automatically clears memory between frames
- Optimized for scenarios where objects look similar

#### `CellTracker`

- Simple, high-level interface for cell tracking
- Handles image preprocessing and postprocessing
- Easy-to-use API for both single frames and sequences

## Usage

### Basic Example

```python
from cell_tracker import CellTracker
import numpy as np

# Initialize the tracker
tracker = CellTracker(
    config_path="cutie/config/cell_tracking_config.yaml",
    weights_path="weights/cutie-base-mega.pth"
)

# Track a sequence of images
images = [...]  # List of numpy arrays (H, W, 3)
first_mask = ...  # Initial mask (H, W) with object IDs
object_ids = [1, 2, 3]  # Object IDs in the first frame

predictions = tracker.track_sequence(
    images=images,
    first_frame_mask=first_mask,
    object_ids=object_ids
)

# Each prediction is a mask with object IDs
for i, mask in enumerate(predictions):
    print(f"Frame {i}: {len(np.unique(mask)) - 1} objects")
```

### Single Frame Processing

```python
# Process individual frames
mask = tracker.track_single_frame(image=image, mask=initial_mask, object_ids=[1, 2])

# For subsequent frames (without initial mask)
mask = tracker.track_single_frame(image=next_image)
```

## Files

### Core Modifications

- `cutie/inference/inference_core.py` - Added cell tracking mode
- `cutie/inference/cell_tracking_inference.py` - Specialized inference core
- `cutie/config/cell_tracking_config.yaml` - Configuration for cell tracking

### User Interface

- `cell_tracker.py` - Simple high-level interface
- `example_cell_tracking.py` - Usage example
- `cell_tracking_demo.py` - Detailed demonstration

## Why These Modifications?

### Original CUTIE Memory System

- **Sensory Memory**: High-level features for each object
- **Working Memory**: Recent frames with configurable capacity (default: 5 frames)
- **Long-term Memory**: Compressed memory for longer sequences

### Cell Tracking Requirements

1. **Similar Appearance**: Cells often look very similar, making long-term appearance memory less useful
2. **Disappearing Objects**: When cells divide, die, or move out of frame, we don't need to remember them
3. **Real-time Processing**: Frame-by-frame processing is often sufficient and more efficient

### Benefits of the Modified Approach

- **Reduced Memory Usage**: No long-term memory, minimal working memory
- **Faster Processing**: Less memory lookup and maintenance
- **Simpler Logic**: Frame-by-frame processing is easier to understand and debug
- **Better for Cell Division**: New cells can appear without complex memory management

## Performance Considerations

- **Memory Usage**: Significantly reduced compared to original CUTIE
- **Speed**: Faster per-frame processing due to minimal memory operations
- **Accuracy**: May be slightly reduced for complex scenarios but optimized for cell tracking use cases

## Original CUTIE

This is based on the CUTIE model. For the original implementation and paper, please refer to the original repository.

## Requirements

- PyTorch
- torchvision
- omegaconf
- numpy
- Other dependencies from the original CUTIE model

## Setup

1. Clone this repository
2. Download the CUTIE model weights
3. Install dependencies
4. Run the example: `python example_cell_tracking.py`
