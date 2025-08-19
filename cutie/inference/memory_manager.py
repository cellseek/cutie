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
                this_last_mask = self._get_mask_by_ids(last_mask, objects)
                this_msk_value = self._get_visual_values_by_ids(objects)
                visual_readout = self._readout(affinity, this_msk_value).view(
                    bs, len(objects), self.CV, h, w
                )
                pixel_readout = network.pixel_fusion(
                    pix_feat, visual_readout, this_sensory, this_last_mask
                )
                this_obj_mem = self._get_object_mem_by_ids(objects)
                this_obj_mem = (
                    this_obj_mem.unsqueeze(2) if this_obj_mem is not None else None
                )
                readout_memory, aux_features = network.readout_query(
                    pixel_readout, this_obj_mem
                )
                for i, obj in enumerate(objects):
                    all_readout_mem[obj] = readout_memory[:, i]

                if self.save_aux:
                    aux_output = {
                        "sensory": this_sensory,
                        "pixel_readout": pixel_readout,
                        "q_logits": aux_features["logits"] if aux_features else None,
                        "q_weights": (
                            aux_features["q_weights"] if aux_features else None
                        ),
                        "p_weights": (
                            aux_features["p_weights"] if aux_features else None
                        ),
                        "attn_mask": (
                            aux_features["attn_mask"].float() if aux_features else None
                        ),
                    }
                    self.aux = aux_output

        return all_readout_mem

    def add_memory(
        self,
        key: torch.Tensor,
        shrinkage: torch.Tensor,
        msk_value: torch.Tensor,
        obj_value: torch.Tensor,
        objects: List[int],
        selection: torch.Tensor = None,
        *,
        as_permanent: str = "none",
    ) -> None:
        """
        Add to working memory. Simplified to remove long-term memory logic.
        """
        # key: (1/2)*C*H*W
        # msk_value: (1/2)*num_objects*C*H*W
        # obj_value: (1/2)*num_objects*Q*C
        # objects contains a list of object ids corresponding to the objects in msk_value/obj_value
        bs = key.shape[0]
        assert shrinkage.shape[0] == bs
        assert msk_value.shape[0] == bs
        assert obj_value is None or obj_value.shape[0] == bs

        self.engaged = True
        if self.H is None or self.config_stale:
            self.config_stale = False
            self.H, self.W = msk_value.shape[-2:]
            self.HW = self.H * self.W
            # convert from num. frames to num. tokens
            self.max_work_tokens = self.max_mem_frames * self.HW

        # key:   bs*C*N
        # value: bs*num_objects*C*N
        key = key.flatten(start_dim=2)
        shrinkage = shrinkage.flatten(start_dim=2)
        self.CK = key.shape[1]

        msk_value = msk_value.flatten(start_dim=3)
        self.CV = msk_value.shape[2]

        # insert object values into object memory
        if obj_value is not None:
            for obj_id, obj in enumerate(objects):
                if obj in self.obj_v:
                    # streaming average
                    last_acc = self.obj_v[obj][:, :, -1]
                    new_acc = last_acc + obj_value[:, obj_id, :, -1]

                    self.obj_v[obj][:, :, :-1] = (
                        self.obj_v[obj][:, :, :-1] + obj_value[:, obj_id, :, :-1]
                    )
                    self.obj_v[obj][:, :, -1] = new_acc
                else:
                    self.obj_v[obj] = obj_value[:, obj_id]

        # convert mask value tensor into a dict for insertion
        msk_values = {obj: msk_value[:, obj_id] for obj_id, obj in enumerate(objects)}
        self.work_mem.add(
            key,
            msk_values,
            shrinkage,
            selection=selection,
            as_permanent=as_permanent,
        )

        # Simple FIFO memory management for working memory only
        for bucket_id in self.work_mem.buckets.keys():
            self.work_mem.remove_old_memory(bucket_id, self.max_work_tokens)

    def purge_except(self, obj_keep_idx: List[int]) -> None:
        """
        Remove all memory except for the given object indices
        """
        # purge certain objects from the memory except the one listed
        self.work_mem.purge_except(obj_keep_idx)
        self.sensory = {k: v for k, v in self.sensory.items() if k in obj_keep_idx}

        if not self.work_mem.engaged():
            # everything is removed!
            self.engaged = False

    def initialize_sensory_if_needed(self, sample_key: torch.Tensor, ids: List[int]):
        for obj in ids:
            if obj not in self.sensory:
                # also initializes the sensory memory
                bs, _, h, w = sample_key.shape
                self.sensory[obj] = torch.zeros(
                    (bs, self.sensory_dim, h, w), device=sample_key.device
                )

    def update_sensory(self, sensory: torch.Tensor, ids: List[int]):
        # sensory: 1*num_objects*C*H*W
        for obj_id, obj in enumerate(ids):
            self.sensory[obj] = sensory[:, obj_id]

    def get_sensory(self, ids: List[int]):
        # returns (1/2)*num_objects*C*H*W
        return self._get_sensory_by_ids(ids)

    def clear_non_permanent_memory(self):
        """
        Clear non-permanent working memory
        """
        log.debug("About to call work_mem.clear_non_permanent_memory()")
        self.work_mem.clear_non_permanent_memory()
        log.debug("Completed work_mem.clear_non_permanent_memory()")
        # Clear non-permanent object memory
        self.obj_v = {}

    def clear_sensory_memory(self):
        self.sensory = {}
