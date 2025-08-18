import logging
from typing import Dict, List

import torch
from omegaconf import DictConfig

from cutie.inference.kv_memory_store import KeyValueMemoryStore
from cutie.inference.object_manager import ObjectManager
from cutie.model.cutie import CUTIE
from cutie.model.utils.memory_utils import *

log = logging.getLogger()


class MemoryManager:
    """
    Simplified Memory Manager for Cell Tracking - Working Memory Only

    Manages only working memory for frame-by-frame cell tracking.
    All long-term memory functionality has been removed for simplicity.
    """

    def __init__(self, cfg: DictConfig, object_manager: ObjectManager):
        self.object_manager = object_manager
        self.sensory_dim = cfg.model.sensory_dim
        self.top_k = cfg.top_k
        self.chunk_size = cfg.chunk_size

        self.save_aux = cfg.save_aux

        # Simplified for cell tracking - only working memory
        self.max_mem_frames = getattr(cfg, "max_mem_frames", 3) - 1

        # dimensions will be inferred from input later
        self.CK = self.CV = None
        self.H = self.W = None

        # The sensory memory is stored as a dictionary indexed by object ids
        # each of shape bs * C^h * H * W
        self.sensory = {}

        # a dictionary indexed by object ids, each of shape bs * T * Q * C
        self.obj_v = {}

        # Only working memory for cell tracking
        self.work_mem = KeyValueMemoryStore(save_selection=False, save_usage=False)

        self.config_stale = True
        self.engaged = False

    def update_config(self, cfg: DictConfig) -> None:
        self.config_stale = True
        self.top_k = cfg["top_k"]
        self.max_mem_frames = getattr(cfg, "max_mem_frames", 3) - 1

    def _readout(self, affinity, v) -> torch.Tensor:
        # affinity: bs*N*HW
        # v: bs*C*N or bs*num_objects*C*N
        # returns bs*C*HW or bs*num_objects*C*HW
        if len(v.shape) == 3:
            # single object
            return v @ affinity
        else:
            bs, num_objects, C, N = v.shape
            v = v.view(bs, num_objects * C, N)
            out = v @ affinity
            return out.view(bs, num_objects, C, -1)

    def _get_mask_by_ids(self, mask: torch.Tensor, obj_ids: List[int]) -> torch.Tensor:
        # -1 because the mask does not contain the background channel
        return mask[:, [self.object_manager.find_tmp_by_id(obj) - 1 for obj in obj_ids]]

    def _get_sensory_by_ids(self, obj_ids: List[int]) -> torch.Tensor:
        return torch.stack([self.sensory[obj] for obj in obj_ids], dim=1)

    def _get_object_mem_by_ids(self, obj_ids: List[int]) -> torch.Tensor:
        if obj_ids[0] not in self.obj_v:
            # should only happen when the object transformer has been disabled
            return None
        return torch.stack([self.obj_v[obj] for obj in obj_ids], dim=1)

    def _get_visual_values_by_ids(self, obj_ids: List[int]) -> torch.Tensor:
        # All the values that the object ids refer to should have the same shape
        return torch.stack([self.work_mem.value[obj] for obj in obj_ids], dim=1)

    def read(
        self,
        pix_feat: torch.Tensor,
        query_key: torch.Tensor,
        selection: torch.Tensor,
        last_mask: torch.Tensor,
        network: CUTIE,
    ) -> Dict[int, torch.Tensor]:
        """
        Read from working memory and returns a single memory readout tensor for each object

        pix_feat: (1/2) x C x H x W
        query_key: (1/2) x C^k x H x W
        selection:  (1/2) x C^k x H x W (not used in simplified version)
        last_mask: (1/2) x num_objects x H x W (at stride 16)
        return a dict of memory readouts, indexed by object indices. Each readout is C*H*W
        """
        h, w = pix_feat.shape[-2:]
        bs = pix_feat.shape[0]
        assert query_key.shape[0] == bs
        assert last_mask.shape[0] == bs

        query_key = query_key.flatten(start_dim=2)  # bs*C^k*HW

        """
        Compute affinity and perform readout - simplified for working memory only
        """
        all_readout_mem = {}
        buckets = self.work_mem.buckets

        for bucket_id, bucket in buckets.items():
            # Only working memory - no long-term memory
            memory_key = self.work_mem.key[bucket_id]
            shrinkage = self.work_mem.shrinkage[bucket_id]
            similarity = get_similarity(memory_key, shrinkage, query_key, selection)
            affinity = do_softmax(similarity, top_k=self.top_k, inplace=True)

            if self.chunk_size < 1:
                object_chunks = [bucket]
            else:
                object_chunks = [
                    bucket[i : i + self.chunk_size]
                    for i in range(0, len(bucket), self.chunk_size)
                ]

            for objects in object_chunks:
                this_sensory = self._get_sensory_by_ids(objects)
                this_obj_v = self._get_object_mem_by_ids(objects)
                this_visual_v = self._get_visual_values_by_ids(objects)

                this_readout_mem = network.pixel_reader(
                    pix_feat, this_sensory, this_obj_v, this_visual_v, affinity
                )

                for i, obj in enumerate(objects):
                    all_readout_mem[obj] = this_readout_mem[:, i]

        return all_readout_mem

    def add_memory(
        self,
        key: torch.Tensor,
        shrinkage: torch.Tensor,
        value: torch.Tensor,
        obj_v: torch.Tensor,
        obj_ids: List[int],
        *,
        selection: torch.Tensor = None,  # Not used in simplified version
        as_permanent: str = "none",
    ) -> None:
        """
        Add to working memory. Simplified to remove long-term memory logic.
        """
        # Infer dimensions if needed
        if self.H is None:
            self.H, self.W = key.shape[-2:]
            self.CK, self.CV = key.shape[-3], value.shape[-3]

        self.work_mem.add(
            key, shrinkage, value, obj_ids, as_permanent=(as_permanent == "all")
        )

        # Update object memory
        if obj_v is not None:
            for i, obj_id in enumerate(obj_ids):
                if obj_id not in self.obj_v:
                    self.obj_v[obj_id] = obj_v[:, i]
                else:
                    self.obj_v[obj_id] = torch.cat(
                        [self.obj_v[obj_id], obj_v[:, i]], dim=1
                    )

        # Manage memory size - remove old frames if too many
        for bucket_id, bucket in self.work_mem.buckets.items():
            if self.work_mem.non_perm_size(bucket_id) >= self.max_mem_frames:
                self.work_mem.remove_obsolete_features(
                    bucket_id, self.max_mem_frames // 2
                )

    def update_sensory(self, sensory: torch.Tensor, obj_ids: List[int]) -> None:
        """
        Update sensory memory for the given objects
        """
        for i, obj_id in enumerate(obj_ids):
            self.sensory[obj_id] = sensory[:, i]

    def get_sensory(self, obj_ids: List[int]) -> torch.Tensor:
        """
        Get sensory memory for the given objects
        """
        if obj_ids[0] not in self.sensory:
            return None
        return torch.stack([self.sensory[obj] for obj in obj_ids], dim=1)

    def initialize_sensory_if_needed(
        self, key: torch.Tensor, obj_ids: List[int]
    ) -> None:
        """
        Initialize sensory memory if it doesn't exist
        """
        for obj_id in obj_ids:
            if obj_id not in self.sensory:
                self.sensory[obj_id] = torch.zeros(
                    (key.shape[0], self.sensory_dim, key.shape[-2], key.shape[-1]),
                    dtype=key.dtype,
                    device=key.device,
                )

    def clear_sensory_memory(self) -> None:
        """
        Clear all sensory memory
        """
        self.sensory = {}

    def clear_non_permanent_memory(self) -> None:
        """
        Clear non-permanent working memory
        """
        self.work_mem.clear_non_permanent_memory()
        # Clear non-permanent object memory
        self.obj_v = {}

    def purge_except(self, obj_keep_idx: List[int]) -> None:
        """
        Remove all memory except for the given object indices
        """
        # Purge working memory
        self.work_mem.purge_except(obj_keep_idx)

        # Purge sensory memory
        sensory_to_remove = [
            obj for obj in self.sensory.keys() if obj not in obj_keep_idx
        ]
        for obj in sensory_to_remove:
            del self.sensory[obj]

        # Purge object memory
        obj_v_to_remove = [obj for obj in self.obj_v.keys() if obj not in obj_keep_idx]
        for obj in obj_v_to_remove:
            del self.obj_v[obj]
