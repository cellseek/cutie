import logging
from typing import List, Optional

import torch
from omegaconf import DictConfig

from cutie.inference.image_feature_store import ImageFeatureStore
from cutie.inference.inference_core import InferenceCore
from cutie.model.cutie import CUTIE

log = logging.getLogger()


class CellTrackingInferenceCore(InferenceCore):
    """
    Specialized inference core for cell tracking.

    Key differences from standard CUTIE:
    1. Designed for frame-by-frame processing with manual corrections
    2. Every frame is treated as a "first frame" with reference mask
    3. No long-term memory functionality
    4. Optimized for cell tracking scenarios where objects look similar
    """

    def __init__(
        self,
        network: CUTIE,
        cfg: DictConfig,
        *,
        image_feature_store: ImageFeatureStore = None,
    ):
        # Initialize without cell tracking mode since we handle it differently
        super().__init__(
            network,
            cfg,
            image_feature_store=image_feature_store,
            cell_tracking_mode=False,  # We handle our own logic
        )

        # Override config for cell tracking - disable long-term memory
        self.mem_every = 1  # Add memory every frame
        self.cfg.max_mem_frames = 1  # Only keep one frame in memory

        log.info("Initialized CellTrackingInferenceCore for frame-by-frame processing")
        log.info(
            f"Memory settings: mem_every={self.mem_every}, max_mem_frames={self.cfg.max_mem_frames}"
        )

    def step(
        self,
        image: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        objects: Optional[List[int]] = None,
        *,
        idx_mask: bool = True,
        end: bool = False,
        delete_buffer: bool = True,
        force_permanent: bool = False,
    ) -> torch.Tensor:
        """
        Cell tracking step: processes one frame treating it as a reference frame.

        For cell tracking, every frame is processed with its reference mask,
        treating each step as if it's the first frame of a new sequence.
        """
        # Always reset memory before processing each frame
        self.clear_memory()

        return super().step(
            image=image,
            mask=mask,
            objects=objects,
            idx_mask=idx_mask,
            end=end,
            delete_buffer=delete_buffer,
            force_permanent=force_permanent,
        )

    def reset_for_new_sequence(self):
        """
        Reset everything for a new video sequence.
        """
        # Clear all memory and reset object manager for complete fresh start
        self.clear_memory()

        # Also reset the object manager to clear object ID mappings from previous sequences
        from cutie.inference.object_manager import ObjectManager

        self.object_manager = ObjectManager()

        # Recreate memory manager with fresh object manager
        from cutie.inference.memory_manager import MemoryManager

        self.memory = MemoryManager(cfg=self.cfg, object_manager=self.object_manager)

        self.curr_ti = -1
        self.last_mem_ti = 0
        self.last_mask = None
        log.info(
            "Reset CellTrackingInferenceCore for new sequence with fresh object manager"
        )
